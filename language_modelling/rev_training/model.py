import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import prod
from rev_training.weight_drop import WeightDrop
from rev_training.revlocked_dropout import RevLockedDropout

sys.path.insert(0, os.path.abspath(".."))
from embed_regularize import EmbedDropout

sys.path.insert(0, os.path.abspath("../.."))
from revlstm import RevLSTM
from revgru import RevGRU
from fixed_util import ConvertToFloat, ConvertToFixed
from buffer import InformationBuffer

forget_radix = 10
hidden_radix = 23

class RNNModel(nn.Module):
    
    def __init__(self, rnn_type, ntoken, in_size, h_size, nlayers, dropout=0.5,
        dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0, max_forget=0.875,
        use_buffers=False):
        
        super(RNNModel, self).__init__()
        self.rnn_type = rnn_type
        self.in_size = in_size
        self.h_size = h_size
        self.nlayers = nlayers
        self.use_buffers = use_buffers

        self.lockdropi = RevLockedDropout(dropouti, in_size)
        self.lockdrophs = [RevLockedDropout(
            dropouth, h_size if l !=nlayers-1 else in_size) for l in range(nlayers)] # TODO: think we just need nlayers - 1
        self.lockdrop = RevLockedDropout(dropout, in_size)
        
        self.embed_drop = EmbedDropout(nn.Embedding(ntoken, in_size), dropoute)

        # Construct RNN cells and apply weight dropping if specified.
        if rnn_type == 'revgru':
            rnn = RevGRU
            module_names = ['ih2_to_zr1', 'irh2_to_g1', 'ih1_to_zr2', 'irh1_to_g2']
        elif rnn_type == 'revlstm':
            rnn = RevLSTM
            module_names = ['ih2_to_zgfop1', 'ih1_to_zgfop2']
        
        self.rnns = [rnn(in_size if l == 0 else h_size,
            h_size if l != nlayers-1 else in_size, max_forget) 
            for l in range(nlayers)]
        self.rnns = [WeightDrop(rnn, module_names, wdrop) for rnn in self.rnns]
        self.rnns = nn.ModuleList(self.rnns)

        # Initialize linear transform from hidden states to log probs.
        self.out = nn.Linear(in_size, ntoken)
        self.out.weight = self.embed_drop.embed.raw_weight 
        self.init_weights()

    def forward(self, input_seq, hiddens):
        """
        Arguments:
            input_seq (LongTensor): of shape (seq_length, batch_size)
            hiddens (list): list of Tensors of length nlayers

        """
        self.set_masks(input_seq.size(1), input_seq.device)

        # Intialize information buffers. To limit unused space at the end of each buffer, 
        # use the same information buffer for all hiddens of the same size.
        seq_length, batch_size = input_seq.size()
        buffers = [None for _ in range(self.nlayers)]
        if self.use_buffers:
            buffers = []       
            for l in range(self.nlayers):
                if l in [0, self.nlayers-1]:
                    buf_dim = self.h_size//2 if l != self.nlayers-1 else self.in_size//2
                    buf = InformationBuffer(batch_size, buf_dim, input_seq.device)
                    buf_tup = (buf, buf) if self.rnn_type == 'revgru' else (buf, buf, buf, buf)
                else:
                    buf_tup = buffers[l-1]
                buffers.append(buf_tup)

        # Embed input sequence.
        input_seq = self.lockdropi(self.embed_drop(input_seq))

        # Process input sequence through model. Start with finding all hidden states
        # for current layer. Then use these hidden states as inputs to the next layer.
        output_dict = {"optimal_bits": 0}
        last_hiddens = []
        curr_seq = input_seq
        for l, (rnn, buf) in enumerate(zip(self.rnns, buffers)):
            curr_hiddens = []
            prev_hidden = hiddens[l] 
            
            for t in range(len(curr_seq)):
                curr_hidden, stats = rnn(curr_seq[t], prev_hidden, buf)
                prev_hidden = curr_hidden['recurrent_hidden']
                curr_hiddens.append(curr_hidden['output_hidden'])
                output_dict['optimal_bits'] += stats['optimal_bits']

            last_hiddens.append(prev_hidden)
            curr_seq = torch.stack(curr_hiddens, dim=0) #[length, batch, hidden]

            if l != self.nlayers-1: 
                curr_seq = self.lockdrophs[l](curr_seq)
                        
        curr_seq = self.lockdrop(curr_seq)
        decoded = self.out(curr_seq.view(curr_seq.size(0) * curr_seq.size(1), -1))
        output_dict['decoded'] = decoded.view(curr_seq.size(0), curr_seq.size(1), -1)
        output_dict['last_h'] = last_hiddens
        
        return output_dict

    def forward_and_backward(self, input_seq, target_seq, hiddens):
        """
        Arguments:
            input_seq (LongTensor): of shape (seq_length, batch_size)
            hiddens (tuple): tuple of Tensors of length nlayers
        """
        hiddens = list(hiddens)
        self.set_masks(input_seq.size(1), input_seq.device) # COMMENT OUT IF TESTING USING FORWARD

        # Intialize information buffers. To limit unused space at the end of each buffer, 
        # use the same information buffer for all hiddens of the same size.
        seq_length, batch_size = input_seq.size()
        buffers = []       
        for l in range(self.nlayers):
            if l in [0, self.nlayers-1]:
                buf_dim = self.h_size//2 if l != self.nlayers-1 else self.in_size//2
                buf = InformationBuffer(batch_size, buf_dim, input_seq.device)
                buf_tup = (buf, buf) if self.rnn_type == 'revgru' else (buf, buf, buf, buf)
            else:
                buf_tup = buffers[l-1]
            buffers.append(buf_tup)

        # Initialize output dictionary.
        output_dict = {'optimal_bits': 0}
        output_dict['normal_bits'] = sum([32*seq_length*batch_size*(self.h_size if l != self.nlayers-1 else self.in_size) for l in range(self.nlayers)])
        if self.rnn_type == 'revlstm':
            output_dict['normal_bits'] *= 2

        # Find last hidden states of model.
        # TODO: figure out way to have wdrop not mask at each step if this takes significant time
        with torch.no_grad():
            for t in range(len(input_seq)):
                curr_input = self.lockdropi(self.embed_drop(input_seq[t]))
                for l, (rnn, buf, lockdroph, hidden) in enumerate(zip(self.rnns, buffers, self.lockdrophs, hiddens)):
                    next_hidden, stats = rnn(curr_input, hidden, buf)
                    if l != self.nlayers-1:
                        curr_input = lockdroph(next_hidden['output_hidden'])
                    hiddens[l] = next_hidden['recurrent_hidden']
                    output_dict['optimal_bits'] += stats['optimal_bits']

        output_dict['last_h'] = hiddens
        output_dict['used_bits'] = 0
        consumed_bufs = []
        for buf in buffers:
            for group_buf in buf:
                if group_buf not in consumed_bufs:
                    output_dict['used_bits'] += group_buf.bit_usage()
                    consumed_bufs.append(group_buf)

        scaled_ce = lambda output, target: (1./seq_length)*F.cross_entropy(output, target) 

        # Loop back through time, reversing computations with help of buffers and using
        # autodiff to compute gradients.
        total_loss = 0
        grad_hiddens = [next(self.parameters()).new_zeros(h.size()) for h in hiddens]
        for t in reversed(range(seq_length)):
            top_hidden = hiddens[-1].requires_grad_()
            top_hidden_ = ConvertToFloat.apply(top_hidden[:,:self.in_size], hidden_radix)
            top_hidden_ = self.lockdrop(top_hidden_)

            output = self.out(top_hidden_)
            last_loss = scaled_ce(output, target_seq[t])
            last_loss.backward()
            grad_hiddens[-1] += top_hidden.grad

            total_loss += last_loss
            
            for l in reversed(range(self.nlayers)):
                rnn, buf, hidden = self.rnns[l], buffers[l], hiddens[l]

                # Reconstruct previous hidden state.
                with torch.no_grad():
                    if l != 0:
                        curr_input = hiddens[l-1]
                        drop_input = self.lockdrophs[l-1](ConvertToFloat.apply(curr_input[:, :self.h_size], hidden_radix))
                    else:
                        curr_input = input_seq[t]
                        drop_input = self.lockdropi(self.embed_drop(curr_input))
                    prev_hidden = rnn.reverse(drop_input, hidden, buf)

                # Rerun forwards pass from previous hidden to hidden at time t to construct
                # computation graph and compute gradients.
                prev_hidden.requires_grad_()
                if l != 0:
                    curr_input.requires_grad_()
                    drop_input = self.lockdrophs[l-1](ConvertToFloat.apply(curr_input[:, :self.h_size], hidden_radix))
                else:
                    drop_input = self.lockdropi(self.embed_drop(curr_input))
                curr_hidden, _ = rnn(drop_input, prev_hidden)
                torch.autograd.backward(curr_hidden['recurrent_hidden'], grad_tensors=grad_hiddens[l])
                hiddens[l] = prev_hidden.detach()
                grad_hiddens[l] = prev_hidden.grad.data

                if l != 0:
                    grad_hiddens[l-1] += curr_input.grad.data

        output_dict['loss'] = total_loss
        return output_dict

    def init_weights(self):
        initrange = 0.1
        self.out.bias.data.fill_(0)
        self.out.weight.data.uniform_(-initrange, initrange)

    def init_hiddens(self, batch_size):
        weight = next(self.parameters()).data
        if self.rnn_type == 'revlstm':
            return [weight.new(batch_size, 2*(self.h_size if l != self.nlayers-1
                else self.in_size)).zero_().int() for l in range(self.nlayers)]
        else:
            return [weight.new(batch_size, self.h_size if l != self.nlayers -1
                else self.in_size).zero_().int() for l in range(self.nlayers)]

    def set_masks(self, batch_size, device):
        self.lockdropi.set_mask(batch_size, device)
        for lockdroph in self.lockdrophs:
            lockdroph.set_mask(batch_size, device)
        self.lockdrop.set_mask(batch_size, device)
        self.embed_drop.set_mask()
        for rnn in self.rnns:
            rnn.set_mask()

if __name__ == "__main__":
    # Test reversibility.
    # Confirmed:
    # -- Model reverses exactly with {1, 2}-layer Rev{GRU, LSTM}
    # -- Model computes correct gradients for sequence of length 1 with individual drops=0.5 and all drops=0.5
    # -- Model computes correct gradients (up to 1 degree) for sequences of length 70

    import numpy as np
    torch.manual_seed(3)
    np.random.seed(0)
    batch_size = 20
    h_size = 400
    emb_size = 100
    ntokens = 30
    seq_length = 70
    nlayers = 1


    def unit_vector(vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)

    def angle_between(v1, v2):
        """ Returns the angle in degrees between vectors 'v1' and 'v2'::

                >>> angle_between((1, 0, 0), (0, 1, 0))
                1.5707963267948966
                >>> angle_between((1, 0, 0), (1, 0, 0))
                0.0
                >>> angle_between((1, 0, 0), (-1, 0, 0))
                3.141592653589793
        """
        v1_u = unit_vector(v1)
        v2_u = unit_vector(v2)
        return 57.2958 * np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    def create_grad_dict(model):
        grads = {}
        for param in model.parameters():
            grads[param] = param.grad.data.clone().numpy()
        return grads

    def compare_grads(grads1, grads2):
        print("COMPARING GRADS")
        grads_okay = True
        for k in grads1:
            all_close = np.allclose(grads1[k], grads2[k])
            grads_okay = grads_okay and all_close
            if not all_close:
                flatten1 = grads1[k].flatten()
                flatten2 = grads2[k].flatten()
                print("Angle: " + str(angle_between(flatten1, flatten2)))

        print("grads okay: " + str(grads_okay))

    input_seq = torch.from_numpy(np.random.choice(ntokens, size=(seq_length, batch_size)))
    target_seq = torch.from_numpy(np.random.choice(ntokens, size=(seq_length, batch_size)))
    model = RNNModel(rnn_type="revlstm", ntoken=ntokens, nlayers=nlayers, in_size=emb_size, h_size=h_size,
        dropout=0, dropouth=0, dropouti=0, dropoute=0, wdrop=0, max_forget=0.875, use_buffers=True)
    assert False, "You need to comment out set masks in forward_and_backward"
    hiddens = model.init_hiddens(batch_size)

    output_dict = model.forward(input_seq, hiddens)
    output = output_dict['decoded']
    normal_loss = F.cross_entropy(output.view(-1, ntokens), target_seq.view(-1))
    normal_loss.backward()
    normal_grads = create_grad_dict(model)
    model.zero_grad()

    reverse_output_dict = model.forward_and_backward(input_seq, target_seq, hiddens)
    reverse_grads = create_grad_dict(model)

    print("normal loss: " + str(normal_loss.detach().numpy()))
    print("reverse loss: " + str(reverse_output_dict['loss'].detach().numpy()))
    compare_grads(normal_grads, reverse_grads)
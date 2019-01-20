import sys
from numpy import prod

import torch
import torch.nn as nn
import torch.nn.functional as F

import onmt.modules
# from onmt.Utils import aeq
# from onmt.Models import RNNDecoderState

sys.path.insert(0, '..')
from revlocked_dropout import RevLockedDropout
from revweight_drop import RevWeightDrop
from fixed_util import ConvertToFloat, ConvertToFixed
from buffer import InformationBuffer
from revgru import RevGRU
from revlstm import RevLSTM

hidden_radix = 23
forget_radix = 10

class RevEncoder(nn.Module):
    
    def __init__(self, rnn_type, h_size, nlayers, embedding, slice_dim=100,
        max_forget=0.875, use_buffers=True, dropouti=0, dropouth=0, wdrop=0):

        super(RevEncoder, self).__init__()
        if rnn_type == 'revgru':
            rnn = RevGRU
            module_names = ['ih2_to_zr1', 'irh2_to_g1', 'ih1_to_zr2', 'irh1_to_g2']
        elif rnn_type == 'revlstm':
            rnn = RevLSTM
            module_names = ['ih2_to_zgfop1', 'ih1_to_zgfop2']

        self.rnns = [rnn(h_size, h_size, max_forget) for l in range(nlayers)]
        self.rnns = [RevWeightDrop(rnn, module_names, wdrop) for rnn in self.rnns]
        self.rnns = nn.ModuleList(self.rnns)

        self.lockdropi = RevLockedDropout(dropouti, h_size)
        self.lockdroph = RevLockedDropout(dropouth, h_size)

        self.rnn_type = rnn_type
        self.h_size = h_size
        self.nlayers = nlayers
        self.encoder = embedding

        self.slice_dim = slice_dim
        self.use_buffers = use_buffers
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.wdrop = wdrop

    def forward(self, input_seq, lengths=None, hiddens=None):
        """
        Arguments:
            input_seq (LongTensor): size is (seq_length, batch_size)
            lengths (IntTensor): size is (batch_size,)
            hiddens (list): list of Tensors of length nlayers
        """
        # Set-up.
        seq_length, batch_size = input_seq.size()
        hiddens = self.init_hiddens(batch_size) if hiddens is None else hiddens
        max_length = len(input_seq) if lengths is None else lengths.max().item()
        self.set_masks(batch_size, input_seq.device)

        # Intialize information buffers. To limit unused space at the end of each buffer, 
        # use the same information buffer for all hiddens of the same size.        
        main_buf = InformationBuffer(batch_size, self.h_size//2, input_seq.device)
        slice_buf = InformationBuffer(batch_size, self.h_size//2 - self.slice_dim,
            input_seq.device)
        buffers = []
        for l in range(len(hiddens)):
            buf_h1 = slice_buf if l == self.nlayers - 1 else main_buf
            buf_h2 = buf_c1 = buf_c2 = main_buf
            if self.rnn_type == 'revlstm':
                buffers.append((buf_h1, buf_h2, buf_c1, buf_c2))
            else:
                buffers.append((buf_h1, buf_h2))

        # Initialize output dictionary.
        output_dict = {'optimal_bits': 0}
        output_dict['normal_bits'] = sum([32*seq_length*batch_size*self.h_size for l in range(self.nlayers)])
        if self.rnn_type == 'revlstm':
            output_dict['normal_bits'] *= 2

        # Find last hidden states of model.
        input_seq = self.lockdropi(self.encoder(input_seq))
        saved_hiddens = []
        with torch.no_grad():
            for t in range(len(input_seq)):
                mask = None if lengths is None else (t < lengths).int() 
                curr_input = input_seq[t]
                
                for l, (rnn, buf, hidden) in enumerate(zip(self.rnns, buffers, hiddens)):
                    if l == self.nlayers - 1:
                        saved_hiddens.append(hidden[:, :self.slice_dim])
                    
                    next_hidden, stats = rnn(curr_input, hidden, buf,
                        self.slice_dim if l==self.nlayers-1 else 0, mask)
                    
                    if l != self.nlayers-1:
                        curr_input = self.lockdroph(next_hidden['output_hidden'])

                    hiddens[l] = next_hidden['recurrent_hidden']
                    output_dict['optimal_bits'] += stats['optimal_bits']

            saved_hiddens.append(hiddens[-1][:, :self.slice_dim])

        output_dict['last_h'] = hiddens
        output_dict['used_bits'] = main_buf.bit_usage() + slice_buf.bit_usage() +\
            32*self.slice_dim*batch_size*max_length

        return hiddens, saved_hiddens, buffers, main_buf, output_dict

    def init_hiddens(self, batch_size):
        weight = next(self.parameters())
        if self.rnn_type == 'revlstm':
            return [weight.new(torch.zeros(batch_size, 2 * self.h_size)).zero_().int()
                for l in range(self.nlayers)]
        else:
            return [weight.new(torch.zeros(batch_size, self.h_size)).zero_().int() 
                for l in range(self.nlayers)]

    def set_masks(self, batch_size, device='cuda'):
        self.lockdropi.set_mask(batch_size, device=device)
        self.lockdroph.set_mask(batch_size, device=device)
        for rnn in self.rnns:
            rnn.set_mask()

    def reverse(self, input_seq, lengths, last_hiddens, last_hidden_grads, saved_hiddens,
        saved_hidden_grads, buffers):
        """
        Arguments:
            input_seq (LongTensor): size is (seq_length, batch_size, 1)
            lengths (IntTensor): size is (batch_size,)
            last_hiddens (list): list of IntTensors (each with size (batch_size, h_size)) 
                of length nlayers
            last_hidden_grads (list): list of FloatTensors (each with size (batch_size, h_size))
                of length nlayers
            saved_hiddens (list): list of IntTensors (each with size (batch_size, slice_dim))
                of length seq_length + 1
            saved_hidden_grads (list): list of FloatTensors (each with size (batch_size, slice_dim))
                of length seq_length + 1
            buffers (list): list of InformationBuffers of length nlayers        
        """
        hiddens = last_hiddens
        hidden_grads = last_hidden_grads

        # TODO(mmackay): replace saved_hidden_grads with just use .grad attribute of the 
        # hiddens in saved_hiddens
        for t in reversed(range(len(input_seq))):
            mask = None if lengths is None else (t < lengths).int() 
            for l in reversed(range(self.nlayers)):
                rnn, buf, hidden = self.rnns[l], buffers[l], hiddens[l]
                # Reconstruct previous hidden state.
                with torch.no_grad():
                    if l != 0:
                        curr_input = hiddens[l-1]
                        drop_input = self.lockdroph(ConvertToFloat.apply(
                            curr_input[:, :self.h_size], hidden_radix))
                    else:
                        curr_input = input_seq[t]
                        drop_input = self.lockdropi(self.encoder(curr_input))
                    
                    prev_hidden = rnn.reverse(drop_input, hidden, buf,
                        self.slice_dim if l == self.nlayers-1 else 0,
                        saved_hiddens[t] if l == self.nlayers-1 else None, mask)

                # Rerun forwards pass from previous hidden to hidden at time t to construct
                # computation graph and compute gradients.
                prev_hidden.requires_grad_()
                if l != 0:
                    curr_input.requires_grad_()
                    drop_input = self.lockdroph(ConvertToFloat.apply(
                        curr_input[:, :self.h_size], hidden_radix))
                else:
                    drop_input = self.lockdropi(self.encoder(curr_input))

                curr_hidden, _ = rnn(drop_input, prev_hidden, mask=mask)
                curr_hidden_grad = hidden_grads[l]
                if l == self.nlayers - 1:
                    curr_hidden_grad[:, :self.slice_dim] += saved_hidden_grads[t+1]
                torch.autograd.backward(curr_hidden['recurrent_hidden'], grad_tensors=curr_hidden_grad)
                hiddens[l] = prev_hidden.detach()
                hidden_grads[l] = prev_hidden.grad.data
                if l != 0:
                    hidden_grads[l-1] += curr_input.grad.data

    def test_forward(self, input_seq, lengths=None, hiddens=None):
        """
        Used for testing correctness of gradients in reverse computation.
        Arguments:
            input_seq (LongTensor): size is (seq_length, batch_size, 1)
            lengths (IntTensor): size is (batch_size,)
            hiddens (list): list of Tensors of length nlayers
        """
        # Set-up. We don't set masks. It is assumed we will call forward before
        # this method, which will set the masks which should remain the same in this method
        # to ensure gradients are equal.
        seq_length, batch_size = input_seq.size()
        hiddens = self.init_hiddens(batch_size) if hiddens is None else hiddens
        max_length = len(input_seq) if lengths is None else lengths.max().item()
        # self.set_masks(batch_size, input_seq.device)

        # Intialize information buffers. To limit unused space at the end of each buffer, 
        # use the same information buffer for all hiddens of the same size.        
        main_buf = InformationBuffer(batch_size, self.h_size//2, input_seq.device)
        slice_buf = InformationBuffer(batch_size, self.h_size//2 - self.slice_dim,
            input_seq.device)
        buffers = []
        for l in range(len(hiddens)):
            buf_h1 = slice_buf if l == self.nlayers - 1 else main_buf
            buf_h2 = buf_c1 = buf_c2 = main_buf
            if self.rnn_type == 'revlstm':
                buffers.append((buf_h1, buf_h2, buf_c1, buf_c2))
            else:
                buffers.append((buf_h1, buf_h2))

        # Find last hidden states of model.
        input_seq = self.lockdropi(self.encoder(input_seq))
        saved_hiddens = []
        for t in range(len(input_seq)):
            mask = None if lengths is None else (t < lengths).int() 
            curr_input = input_seq[t]
                
            for l, (rnn, buf, hidden) in enumerate(zip(self.rnns, buffers, hiddens)):
                if l == self.nlayers - 1:
                    saved_hiddens.append(hidden[:, :self.slice_dim])
                
                next_hidden, stats = rnn(curr_input, hidden, buf,
                    self.slice_dim if l==self.nlayers-1 else 0, mask)
                
                if l != self.nlayers-1:
                    curr_input = self.lockdroph(next_hidden['output_hidden'])

                hiddens[l] = next_hidden['recurrent_hidden']

        saved_hiddens.append(hiddens[-1][:, :self.slice_dim])

        return hiddens, saved_hiddens, buffers, {}


if __name__ == '__main__':
    # Testing reversibility in encoder
    import numpy as np
    from testing_util import create_grad_dict, compare_grads

    torch.manual_seed(3)
    np.random.seed(0)
    batch_size = 20
    h_size = 200
    emb_size = h_size
    slice_dim = 20
    nlayers = 2

    enc_ntokens = 30
    enc_seq_length = 32
    
    dec_ntokens = 36
    dec_seq_length = 34
    context_type = 'slice'

    enc_input_seq = torch.from_numpy(np.random.choice(enc_ntokens, size=(enc_seq_length, batch_size)))
    enc_lengths = torch.from_numpy(np.random.choice(np.arange(1, enc_seq_length+1), size=(batch_size)))
    enc_lengths[-1] = enc_seq_length
    enc_embedding = nn.Embedding(enc_ntokens, emb_size)
    rev_enc = RevEncoder(rnn_type='revgru', h_size=h_size, nlayers=nlayers,
        embedding=enc_embedding, slice_dim=slice_dim, max_forget=0.875, use_buffers=True,
        dropouti=0.5, dropouth=0.5, wdrop=0.5)

    def test_encoder():
        hiddens = rev_enc.init_hiddens(batch_size)

        last_hidden_grads = [torch.ones(batch_size, h_size) for _ in range(nlayers)]
        saved_hidden_grads = [torch.ones(batch_size, slice_dim) for _ in range(enc_seq_length)] 
        saved_hidden_grads += [torch.ones(batch_size, slice_dim)]

        hiddens, saved_hiddens, buffers, _, output_dict = rev_enc(enc_input_seq, enc_lengths, hiddens)
        rev_enc.reverse(enc_input_seq, enc_lengths, last_hiddens=hiddens, last_hidden_grads=last_hidden_grads,
            saved_hiddens=saved_hiddens, saved_hidden_grads=saved_hidden_grads, buffers=buffers)


        reverse_grads = create_grad_dict(rev_enc)
        rev_enc.zero_grad()

        hiddens = rev_enc.init_hiddens(batch_size)
        hiddens, saved_hiddens, _, _ = rev_enc.test_forward(enc_input_seq, enc_lengths, hiddens)
        loss = torch.sum(ConvertToFloat.apply(torch.stack(saved_hiddens), hidden_radix))
        loss += torch.sum(ConvertToFloat.apply(torch.stack(hiddens), hidden_radix))
        loss.backward()
        normal_grads = create_grad_dict(rev_enc)
        
        compare_grads(normal_grads, reverse_grads)
        
    test_encoder()
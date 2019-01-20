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

class RevDecoder(nn.Module):

    def __init__(self, rnn_type, h_size, nlayers, embedding, attn_type='general',
        context_size=None, dropouti=0, dropouth=0, dropouto=0, wdrop=0, dropouts=0,
        max_forget=0.875, context_type='slice', slice_dim=0, use_buffers=True):

        super(RevDecoder, self).__init__()
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
        self.lockdropo = RevLockedDropout(dropouto, h_size)
        self.lockdrops = RevLockedDropout(dropouts, slice_dim)

        # Basic attributes.
        self.rnn_type = rnn_type
        self.decoder_type = 'rnn' 
        self.context_type = context_type
        self.context_size = context_size
        self.nlayers = nlayers
        self.h_size = h_size
        self.embedding = embedding
        self.wdrop = wdrop
        self.use_buffers = use_buffers
        self.generator = nn.Linear(h_size, embedding.num_embeddings)

        # Set up the standard attention.
        self.attn_type = attn_type
        if attn_type != 'none':
            self.attn = onmt.modules.MultiSizeAttention(
                h_size, context_size=context_size, attn_type=attn_type)


    def forward(self, input_seq, hiddens, main_buf, dec_lengths=None):
        """
        Arguments:
            input_seq (LongTensor): size is (seq_length, batch_size)
            hiddens (list): list of IntTensors (each with size (batch_size, h_size))
                of length nlayers
            main_buf (InformationBuffer): storage for hidden states with size
                (batch_size, h_size)
            dec_lengths (IntTensor): size is (batch_size,)
        """
        # Set up.
        seq_length, batch_size = input_seq.size()
        self.set_masks(batch_size, input_seq.device)

        # Intialize information buffers. To limit unused space at the end of each buffer, 
        # use the same information buffer for all hiddens of the same size.
        # This means using the buffer 'main_buf' from the encoder.      
        buffers = []
        for l in range(len(hiddens)):
            if self.rnn_type == 'revlstm':
                buffers.append((main_buf, main_buf, main_buf, main_buf))
            else:
                buffers.append((main_buf, main_buf))

        # Initialize output dictionary.
        output_dict = {'optimal_bits': 0}
        output_dict['normal_bits'] = sum(
            [32*seq_length*batch_size*self.h_size for l in range(self.nlayers)])
        if self.rnn_type == 'revlstm':
            output_dict['normal_bits'] *= 2


        output_dict['hid_seq'] = []
        for l in range(self.nlayers):
            output_dict['hid_seq'].append([hiddens[l].data.clone().numpy()])

        # Find last hidden states of model.
        input_seq = self.lockdropi(self.embedding(input_seq))
        with torch.no_grad():
            for t in range(len(input_seq)):
                mask = None if dec_lengths is None else (t < dec_lengths).int() 
                curr_input = input_seq[t]
                
                for l, (rnn, buf, hidden) in enumerate(zip(self.rnns, buffers, hiddens)):                   
                    next_hidden, stats = rnn(curr_input, hidden, buf, slice_dim=0,
                        mask=mask)
                    if l != self.nlayers-1:
                        curr_input = self.lockdroph(next_hidden['output_hidden'])

                    hiddens[l] = next_hidden['recurrent_hidden']
                    output_dict['hid_seq'][l].append(hiddens[l].data.clone().numpy())
                    output_dict['optimal_bits'] += stats['optimal_bits']

        output_dict['last_h'] = hiddens
        output_dict['used_bits'] = float('inf') # TODO(mmackay): figure out right way to compute bits
        return hiddens, buffers, output_dict

    def reverse(self, input_seq, target_seq, last_hiddens, saved_hiddens, buffers,
        token_weights, dec_lengths, enc_lengths, enc_input_seq, rev_enc):
        """
        Arguments:
            input_seq (LongTensor): size is (seq_length, batch_size)
            hiddens (list): list of IntTensors (each with size (batch_size, h_size))
                of length nlayers
            main_buf (InformationBuffer): storage for hidden states with size
                (batch_size, h_size)
            dec_lengths (IntTensor): size is (batch_size,)
        """

        hiddens = last_hiddens
        hidden_grads = [next(self.parameters()).new_zeros(h.size()) for h in hiddens]
        loss_fun = lambda output, target: F.cross_entropy(
            output, target, weight=token_weights, size_average=False)

        saved_hiddens = [hidden.requires_grad_() for hidden in saved_hiddens]
        total_loss = 0.

        output_dict = {'hid_seq': []}
        for l in range(self.nlayers):
            output_dict['hid_seq'].append([last_hiddens[l].data.clone().numpy()])
        output_dict['drop_hs'] = []

        for t in reversed(range(len(input_seq))):
            top_hidden = hiddens[-1].requires_grad_()
            top_hidden_ = ConvertToFloat.apply(top_hidden[:,:self.h_size], hidden_radix)
            top_hidden_ = self.lockdropo(top_hidden_).unsqueeze(0)

            output_dict['drop_hs'].append(top_hidden_.data.clone().numpy())

            context = self.construct_context(saved_hiddens, enc_input_seq, rev_enc)            
            attn_hidden, _ = self.attn(top_hidden_.transpose(0, 1).contiguous(),
                context.transpose(0, 1), context_lengths=enc_lengths)
            output = self.generator(attn_hidden[0])
            last_loss = loss_fun(output, target_seq[t])
            last_loss.backward()
            hidden_grads[-1] += top_hidden.grad
            total_loss += last_loss

            mask = None if dec_lengths is None else (t < dec_lengths).int() 
            for l in reversed(range(self.nlayers)):
                rnn, buf, hidden = self.rnns[l], buffers[l], hiddens[l]
                # Reconstruct previous hidden state.
                with torch.no_grad():
                    if l != 0:
                        curr_input = hiddens[l-1]
                        drop_input = self.lockdroph(ConvertToFloat.apply(
                            curr_input[:, :self.h_size], hidden_radix))
                    else:
                        curr_input = input_seq[t].squeeze()
                        drop_input = self.lockdropi(self.embedding(curr_input))
                    
                    prev_hidden = rnn.reverse(drop_input, hidden, buf, slice_dim=0,
                        saved_hidden=None, mask=mask)

                # Rerun forwards pass from previous hidden to hidden at time t to construct
                # computation graph and compute gradients.
                prev_hidden.requires_grad_()
                if l != 0:
                    curr_input.requires_grad_()
                    drop_input = self.lockdroph(ConvertToFloat.apply(
                        curr_input[:, :self.h_size], hidden_radix))
                else:
                    drop_input = self.lockdropi(self.embedding(curr_input))

                curr_hidden, _ = rnn(drop_input, prev_hidden, mask=mask)
                torch.autograd.backward(
                    curr_hidden['recurrent_hidden'], grad_tensors=hidden_grads[l])
                hiddens[l] = prev_hidden.detach()
                hidden_grads[l] = prev_hidden.grad.data
                if l != 0:
                    hidden_grads[l-1] += curr_input.grad.data

                output_dict['hid_seq'][l].append(prev_hidden.data.clone().numpy())

        return hiddens, hidden_grads, saved_hiddens, buffers, total_loss, output_dict

    def construct_context(self, saved_hiddens, enc_input_seq, rev_enc):
        if self.context_type == 'emb':
            context = rev_enc.lockdropi(rev_enc.encoder(enc_input_seq))
        elif self.context_type == 'slice':
            context = self.lockdrops(ConvertToFloat.apply(
                torch.stack(saved_hiddens[1:]), hidden_radix))
        elif self.context_type == 'slice_emb':
            embs = rev_enc.lockdropi(rev_enc.encoder(enc_input_seq))
            slices = self.lockdrops(ConvertToFloat.apply(
                torch.stack(saved_hiddens[1:]), hidden_radix))
            context = torch.cat([embs, slices], dim=2)

        return context

    def test_forward(self, input_seq, hiddens, main_buf, dec_lengths=None):
        """
        Arguments:
            input_seq (LongTensor): size is (seq_length, batch_size)
            hiddens (list): list of IntTensors (each with size (batch_size, h_size))
                of length nlayers
            main_buf (InformationBuffer): storage for hidden states with size
                (batch_size, h_size)
            dec_lengths (IntTensor): size is (batch_size,)
        """
        # Set up.
        seq_length, batch_size = input_seq.size()
        # self.set_masks(batch_size, input_seq.device)

        # Intialize information buffers. To limit unused space at the end of each buffer, 
        # use the same information buffer for all hiddens of the same size.
        # This means using the buffer 'main_buf' from the encoder.      
        buffers = []
        for l in range(len(hiddens)):
            if self.rnn_type == 'revlstm':
                buffers.append((main_buf, main_buf, main_buf, main_buf))
            else:
                buffers.append((main_buf, main_buf))

        output_dict = {'hid_seq': []}
        for l in range(self.nlayers):
            output_dict['hid_seq'].append([hiddens[l].data.clone().numpy()])

        # Find last hidden states of model.
        input_seq = self.lockdropi(self.embedding(input_seq))
        top_hiddens = []
        for t in range(len(input_seq)):
            mask = None if dec_lengths is None else (t < dec_lengths).int() 
            curr_input = input_seq[t]
            
            for l, (rnn, buf, hidden) in enumerate(zip(self.rnns, buffers, hiddens)):                   
                next_hidden, stats = rnn(curr_input, hidden, buf, slice_dim=0,
                    mask=mask)
                if l != self.nlayers-1:
                    curr_input = self.lockdroph(next_hidden['output_hidden'])

                hiddens[l] = next_hidden['recurrent_hidden']
                output_dict['hid_seq'][l].append(hiddens[l].data.clone().numpy())

                if l == self.nlayers-1:
                    top_hiddens.append(next_hidden['output_hidden'])

        return top_hiddens, hiddens, output_dict

    def test_compute_loss(self, top_hiddens, target_seq, saved_hiddens, token_weights, enc_lengths,
        enc_input_seq, rev_enc):
        """
        Used to test correctness of gradients in reverse computation.
        """
        top_hiddens = self.lockdropo(torch.stack(top_hiddens))

        output_dict = {'drop_hs': []}
        for t in range(len(top_hiddens)):
            output_dict['drop_hs'].append(top_hiddens[t].data.clone().numpy())

        if self.context_type == 'emb':
            context = rev_enc.lockdropi(rev_enc.encoder(enc_input_seq))
        elif self.context_type == 'slice':
            context = self.lockdrops(ConvertToFloat.apply(
                torch.stack(saved_hiddens[1:]), hidden_radix))
        elif self.context_type == 'slice_emb':
            embs = rev_enc.lockdropi(rev_enc.encoder(enc_input_seq))
            slices = self.lockdrops(ConvertToFloat.apply(
                torch.stack(saved_hiddens[1:]), hidden_radix))
            context = torch.cat([embs, slices], dim=2)
        
        attn_hiddens, _ = self.attn(top_hiddens.transpose(0, 1).contiguous(), 
            context.transpose(0, 1), context_lengths=enc_lengths)

        output = self.generator(attn_hiddens.view(-1, attn_hiddens.size(2)))
        loss = F.cross_entropy(output, target_seq.view(-1), weight=token_weights,
            size_average=False)

        return loss, output_dict



    @property
    def _input_size(self):
        """
        Private helper returning the number of expected features.
        """
        return self.embeddings.embedding_dim  # IMPORTANT: This is where the issues are

    def set_masks(self, batch_size, device='cuda'):
        self.lockdropi.set_mask(batch_size, device=device)
        self.lockdroph.set_mask(batch_size, device=device)
        self.lockdropo.set_mask(batch_size, device=device)
        self.lockdrops.set_mask(batch_size, device=device)
        for rnn in self.rnns:
            rnn.set_mask()

    def init_hiddens(self, batch_size):
        ''' For testing purposes, should never be called in practice'''
        weight = next(self.parameters())
        if self.rnn_type == 'revlstm':
            return [weight.new(torch.zeros(batch_size, 2 * self.h_size)).zero_().int()
                for l in range(self.nlayers)]
        else:
            return [weight.new(torch.zeros(batch_size, self.h_size)).zero_().int() 
                for l in range(self.nlayers)]

if __name__ == '__main__':
    import numpy as np
    from testing_util import angle_between, create_grad_dict, compare_grads, detail_grads

    torch.manual_seed(3)
    np.random.seed(0)
    batch_size = 30
    h_size = 200
    emb_size = h_size
    slice_dim = 20
    nlayers = 2

    enc_ntokens = 30
    enc_seq_length = 32
    
    dec_ntokens = 36
    dec_seq_length = 34
    context_type = 'slice_emb'

    enc_input_seq = torch.from_numpy(np.random.choice(enc_ntokens, size=(enc_seq_length, batch_size)))
    enc_lengths = torch.from_numpy(np.random.choice(np.arange(1, enc_seq_length+1), size=(batch_size)))
    enc_lengths[-1] = enc_seq_length
    enc_embedding = nn.Embedding(enc_ntokens, emb_size)
    from revencoder import RevEncoder
    rev_enc = RevEncoder(rnn_type='revgru', h_size=h_size, nlayers=nlayers,
        embedding=enc_embedding, slice_dim=slice_dim, max_forget=0.875, use_buffers=True,
        dropouti=0, dropouth=0, wdrop=0)

    dec_input_seq = torch.from_numpy(np.random.choice(dec_ntokens, size=(dec_seq_length, batch_size)))
    dec_lengths = torch.from_numpy(np.random.choice(np.arange(1, dec_seq_length+1), size=(batch_size)))
    dec_lengths[-1] = dec_seq_length
    target_seq = torch.from_numpy(np.random.choice(dec_ntokens, size=(dec_seq_length, batch_size)))
    token_weights = torch.ones(dec_ntokens)
    embedding = nn.Embedding(dec_ntokens, emb_size)
    if context_type == 'emb':
        context_size = emb_size
    elif context_type == 'slice':
        context_size = slice_dim
    elif context_type == 'slice_emb':
        context_size = emb_size + slice_dim
    rev_dec = RevDecoder(rnn_type='revgru', h_size=h_size, nlayers=nlayers,
        embedding=embedding, attn_type='general', context_size=context_size,
        dropouti=0.5, dropouth=0.5, dropouto=0.5, wdrop=0.5, dropouts=0.5, max_forget=0.875,
        context_type=context_type, slice_dim=slice_dim, use_buffers=True)

    hiddens = rev_enc.init_hiddens(batch_size)
    enc_hiddens, saved_hiddens, buffers, main_buf, _ = rev_enc(enc_input_seq, enc_lengths, hiddens)

    def test_reconstruction():
        '''
        Roadmap:
            Confirmed: hiddens constructed in forward and hiddens reconstructed in reverse are the same
            Confirmed: hiddens constructed in forward and test_forward are the same
            
            Confirm hiddens being passed to attention module are the same b/t reverse and forward
            Confirm attn_hiddens are the same b/t reverse and test_compute_loss
            Confirm loss value is the same b/t reverse and test_compute_loss
            Confirm gradients are the same

            Try changing token weights and ensure gradients remain the same
        '''
        main_buf = InformationBuffer(batch_size, h_size//2, 'cpu')
        hiddens, buffers, output_dict = rev_dec(dec_input_seq, enc_hiddens, main_buf, dec_lengths)

        hiddens, hidden_grads, saved_hiddens, buffers, loss, rev_dict = rev_dec.reverse(dec_input_seq,
            target_seq, hiddens, saved_hiddens, buffers, token_weights, dec_lengths,
            enc_lengths, enc_input_seq, rev_enc)

        for l in range(nlayers):
            for for_h, rev_h in zip(output_dict['hid_seq'][l], reversed(rev_dict['hid_seq'][l])):
                print((for_h == rev_h).all())
                print(for_h)
                print(rev_h)

    def test_construction():
        hiddens = rev_enc.init_hiddens(batch_size)
        enc_hiddens, saved_hiddens, buffers, main_buf, _ = rev_enc(enc_input_seq, enc_lengths, hiddens)

        main_buf = InformationBuffer(batch_size, h_size//2, 'cpu')
        hiddens, buffers, output_dict = rev_dec(dec_input_seq, enc_hiddens, main_buf, dec_lengths)

        hiddens = rev_enc.init_hiddens(batch_size)
        enc_hiddens, saved_hiddens, buffers, main_buf, _ = rev_enc(enc_input_seq, enc_lengths, hiddens)
        main_buf = InformationBuffer(batch_size, h_size//2, 'cpu')
        top_hiddens, last_hiddens, normal_dict = rev_dec.test_forward(dec_input_seq, enc_hiddens, main_buf, dec_lengths)

        for l in range(nlayers):
            print("LAYER: " + str(l))
            for for_h, nor_h in zip(output_dict['hid_seq'][l], normal_dict['hid_seq'][l]):
                print((for_h == nor_h).all())

    def test_loss():
        hiddens = rev_enc.init_hiddens(batch_size)
        enc_hiddens, saved_hiddens, buffers, main_buf, _ = rev_enc(enc_input_seq, enc_lengths, hiddens)
        main_buf = InformationBuffer(batch_size, h_size//2, 'cpu')
        
        hiddens, buffers, output_dict = rev_dec(dec_input_seq, enc_hiddens, main_buf, dec_lengths)
        hiddens, hidden_grads, saved_hiddens, buffers, rev_loss, rev_dict = rev_dec.reverse(dec_input_seq,
            target_seq, hiddens, saved_hiddens, buffers, token_weights, dec_lengths,
            enc_lengths, enc_input_seq, rev_enc)

        hiddens = rev_enc.init_hiddens(batch_size)
        enc_hiddens, saved_hiddens, buffers, main_buf, _ = rev_enc(enc_input_seq, enc_lengths, hiddens)
        main_buf = InformationBuffer(batch_size, h_size//2, 'cpu')
        
        top_hiddens, last_hiddens, normal_dict = rev_dec.test_forward(dec_input_seq, enc_hiddens, main_buf, dec_lengths)
        nor_loss, normal_dict = rev_dec.test_compute_loss(top_hiddens, target_seq, saved_hiddens,
            token_weights, enc_lengths, enc_input_seq, rev_enc)

        print(rev_loss.data.numpy())
        print(nor_loss.data.numpy())

    def test_grads():
        hiddens = rev_enc.init_hiddens(batch_size)
        enc_hiddens, saved_hiddens, buffers, main_buf, _ = rev_enc(enc_input_seq, enc_lengths, hiddens)
        main_buf = InformationBuffer(batch_size, h_size//2, 'cpu')
        
        hiddens, buffers, output_dict = rev_dec(dec_input_seq, enc_hiddens, main_buf, dec_lengths)
        hiddens, hidden_grads, saved_hiddens, buffers, rev_loss, rev_dict = rev_dec.reverse(dec_input_seq,
            target_seq, hiddens, saved_hiddens, buffers, token_weights, dec_lengths,
            enc_lengths, enc_input_seq, rev_enc)
        rev_grads = create_grad_dict(rev_dec)
        if context_type in ['slice', 'slice_emb']:
            saved_hidden_rev_grads = [hid.grad.data.clone().numpy() for hid in saved_hiddens[1:]]

        rev_dec.zero_grad()
        hiddens = rev_enc.init_hiddens(batch_size)
        enc_hiddens, saved_hiddens, buffers, main_buf, _ = rev_enc(enc_input_seq, enc_lengths, hiddens)
        main_buf = InformationBuffer(batch_size, h_size//2, 'cpu')
        
        saved_hiddens = [hid.requires_grad_() for hid in saved_hiddens]
        top_hiddens, last_hiddens, normal_dict = rev_dec.test_forward(dec_input_seq, enc_hiddens, main_buf, dec_lengths)
        nor_loss, normal_dict = rev_dec.test_compute_loss(top_hiddens, target_seq, saved_hiddens,
            token_weights, enc_lengths, enc_input_seq, rev_enc)
        nor_loss.backward()
        nor_grads = create_grad_dict(rev_dec)
        if context_type in ['slice', 'slice_emb']:
            saved_hidden_nor_grads = [hid.grad.data.clone().numpy() for hid in saved_hiddens[1:]]

        compare_grads(rev_grads, nor_grads)

        if context_type in ['slice', 'slice_emb']:
            for rev_grad, nor_grad in zip(saved_hidden_rev_grads, saved_hidden_nor_grads):
                print(angle_between(rev_grad.flatten(), nor_grad.flatten()))

        
    test_grads()
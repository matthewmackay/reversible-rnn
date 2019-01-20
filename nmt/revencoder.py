import sys
import ipdb
from numpy import prod

import torch
import torch.nn as nn
import torch.nn.functional as F

import onmt.modules

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
        max_forget=0.875, use_buffers=True, dropouti=0, dropouth=0, wdrop=0, 
        context_type='slice'):

        super(RevEncoder, self).__init__()
        if rnn_type == 'revgru':
            rnn = RevGRU
            module_names = ['ih2_to_zr1', 'irh2_to_g1', 'ih1_to_zr2', 'irh1_to_g2']
        elif rnn_type == 'revlstm':
            rnn = RevLSTM
            module_names = ['ih2_to_zgfop1', 'ih1_to_zgfop2']

        self.rnns = [rnn(h_size, h_size, max_forget) for l in range(nlayers)]
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
        self.context_type = context_type

        if slice_dim == h_size:
            self.rev_forward = self.rev_forward_allh
            self.forward = self.rev_forward_allh
            self.reverse = self.reverse_allh

    def forward_test(self, input_seq, lengths=None, hiddens=None):
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

        # Find last hidden states of model.
        input_seq = self.lockdropi(self.encoder(input_seq))
        all_hiddens = []
        with torch.set_grad_enabled(self.training):
            for t in range(len(input_seq)):
                mask = None if lengths is None else (t < lengths).int()
                curr_input = input_seq[t]

                for l, (rnn, hidden) in enumerate(zip(self.rnns, hiddens)):
                    if l == self.nlayers - 1:
                        all_hiddens.append(hidden)

                    next_hidden, stats = rnn(curr_input, hidden, buf=None, slice_dim=self.slice_dim if l==self.nlayers-1 else 0, mask=mask)

                    if l != self.nlayers-1:
                        curr_input = self.lockdroph(next_hidden['output_hidden'])

                    hiddens[l] = next_hidden['recurrent_hidden']

            all_hiddens.append(hiddens[-1])

            hidden_context = ConvertToFloat.apply(torch.stack(all_hiddens[1:]), hidden_radix)

            if self.context_type == 'hidden':
                context = hidden_context
            elif self.context_type == 'emb':
                context = input_seq
            elif self.context_type == 'slice':
                context = hidden_context[:,:,:self.slice_dim]
            elif self.context_type == 'slice_emb':
                context = torch.cat([input_seq, hidden_context[:,:,:self.slice_dim]], dim=2)

        return hiddens, context

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
        slice_buf = InformationBuffer(batch_size, self.h_size//2 - self.slice_dim, input_seq.device)
        buffers = []
        for l in range(len(hiddens)):
            if self.training and self.use_buffers:
                buf_h1 = slice_buf if l == self.nlayers - 1 else main_buf
                buf_h2 = buf_c1 = buf_c2 = main_buf
            else:
                buf_h1 = buf_h2 = buf_c1 = buf_c2 = None
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
        with torch.set_grad_enabled(self.training):
            for t in range(len(input_seq)):
                mask = None if lengths is None else (t < lengths).int()
                curr_input = input_seq[t]

                for l, (rnn, buf, hidden) in enumerate(zip(self.rnns, buffers, hiddens)):
                    if l == self.nlayers - 1:
                        saved_hiddens.append(hidden[:, :self.slice_dim])

                    next_hidden, stats = rnn(curr_input, hidden, buf, self.slice_dim if l==self.nlayers-1 else 0, mask)

                    if l != self.nlayers-1:
                        curr_input = self.lockdroph(next_hidden['output_hidden'])

                    hiddens[l] = next_hidden['recurrent_hidden']
                    output_dict['optimal_bits'] += stats['optimal_bits']

            saved_hiddens.append(hiddens[-1][:, :self.slice_dim])

        return hiddens, saved_hiddens, buffers, main_buf, slice_buf, output_dict

    def rev_forward(self, input_seq, lengths=None, hiddens=None):
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
                    if l == self.nlayers - 1 and self.slice_dim > 0:
                        saved_hiddens.append(hidden[:, :self.slice_dim])

                    next_hidden, stats = rnn(curr_input, hidden, buf, self.slice_dim if l==self.nlayers-1 else 0, mask)

                    if l != self.nlayers-1:
                        curr_input = self.lockdroph(next_hidden['output_hidden'])

                    hiddens[l] = next_hidden['recurrent_hidden']
                    output_dict['optimal_bits'] += stats['optimal_bits']
            if self.slice_dim > 0:
                saved_hiddens.append(hiddens[-1][:, :self.slice_dim])

        output_dict['last_h'] = hiddens
        output_dict['used_bits'] = main_buf.bit_usage() + slice_buf.bit_usage() +\
            32*self.slice_dim*batch_size*max_length

        return hiddens, saved_hiddens, buffers, main_buf, slice_buf, output_dict

    def rev_forward_allh(self, input_seq, lengths=None, hiddens=None):
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
        buffers = []
        for l in range(len(hiddens)):
            if l == self.nlayers - 1:
                buffers.append(None)
            else:
                buf_h1 = buf_h2 = buf_c1 = buf_c2 = main_buf
                if self.rnn_type == 'revlstm':
                    buffers.append((buf_h1, buf_h2, buf_c1, buf_c2))
                else:
                    buffers.append((buf_h1, buf_h2))

        # Find last hidden states of model.
        input_seq = self.lockdropi(self.encoder(input_seq))
        saved_hiddens = []
        with torch.no_grad():
            for t in range(len(input_seq)):
                mask = None if lengths is None else (t < lengths).int()
                curr_input = input_seq[t]

                for l, (rnn, buf, hidden) in enumerate(zip(self.rnns, buffers, hiddens)):
                    if l == self.nlayers - 1:
                        saved_hiddens.append(hidden.detach().clone())

                    next_hidden, stats = rnn(curr_input, hidden, buf, 0, mask)
                    if l != self.nlayers-1:
                        curr_input = self.lockdroph(next_hidden['output_hidden'])
                    hiddens[l] = next_hidden['recurrent_hidden']

            saved_hiddens.append(hiddens[-1])

        total_bits = sum([32*seq_length*batch_size*self.h_size for l in range(self.nlayers)])
        output_dict = {'normal_bits': total_bits, 'optimal_bits': total_bits, 
            'used_bits': total_bits}
        if self.rnn_type == 'revlstm':
            for k in ['normal_bits', 'optimal_bits', 'used_bits']:
                output_dict[k] *= 2
        output_dict['last_h'] = hiddens

        return hiddens, saved_hiddens, buffers, main_buf, None, output_dict

    def init_hiddens(self, batch_size):
        weight = next(self.parameters())
        if self.rnn_type == 'revlstm':
            return [weight.new(batch_size, 2 * self.h_size).zero_().int() for l in range(self.nlayers)]
        else:
            return [weight.new(batch_size, self.h_size).zero_().int() for l in range(self.nlayers)]

    def set_masks(self, batch_size, device='cuda'):
        self.lockdropi.set_mask(batch_size, device=device)
        self.lockdroph.set_mask(batch_size, device=device)

    def reverse(self, input_seq, lengths, last_hiddens, last_hidden_grads, saved_hiddens,
        buffers):
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
            buffers (list): list of InformationBuffers of length nlayers
        """
        hiddens = last_hiddens
        hidden_grads = last_hidden_grads

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
                        saved_hiddens[t] if l == self.nlayers-1 and self.slice_dim > 0 else None,
                        mask)

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
                if l == self.nlayers - 1 and self.slice_dim > 0:
                    curr_hidden_grad[:, :self.slice_dim] += saved_hiddens[t+1].grad
                torch.autograd.backward(curr_hidden['recurrent_hidden'], grad_tensors=curr_hidden_grad)
                hiddens[l] = prev_hidden.detach()
                hidden_grads[l] = prev_hidden.grad.data
                if l != 0:
                    hidden_grads[l-1] += curr_input.grad.data

    def reverse_allh(self, input_seq, lengths, last_hiddens, last_hidden_grads,
        saved_hiddens, buffers):
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
            buffers (list): list of InformationBuffers of length nlayers
        """
        hiddens = last_hiddens
        hidden_grads = last_hidden_grads

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

                    prev_hidden = rnn.reverse(drop_input, hidden, buf, 0, 
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
                    curr_hidden_grad += saved_hiddens[t+1].grad
                torch.autograd.backward(curr_hidden['recurrent_hidden'],
                    grad_tensors=curr_hidden_grad)
                hiddens[l] = prev_hidden.detach()
                hidden_grads[l] = prev_hidden.grad.data
                if l != 0:
                    hidden_grads[l-1] += curr_input.grad.data

if __name__ == '__main__':
    import numpy as np

    torch.manual_seed(3)
    np.random.seed(0)
    batch_size = 30
    h_size = 200
    emb_size = h_size
    slice_dim = 200
    nlayers = 2
    max_forget = 0.875

    enc_ntokens = 30
    enc_seq_length = 32

    dec_ntokens = 36
    dec_seq_length = 34
    context_type = 'slice'

    enc_input_seq = torch.from_numpy(np.random.choice(enc_ntokens, size=(enc_seq_length, batch_size)))
    enc_lengths = torch.from_numpy(np.random.choice(np.arange(1, enc_seq_length+1), size=(batch_size)))
    enc_lengths[-1] = enc_seq_length
    enc_embedding = nn.Embedding(enc_ntokens, emb_size)

    use_separate_buffers = True

    # Construct RevEncoder.
    from revencoder import RevEncoder
    rev_enc = RevEncoder(rnn_type='revlstm', h_size=h_size, nlayers=nlayers, 
        embedding=enc_embedding, slice_dim=slice_dim, max_forget=max_forget,
        use_buffers=True, dropouti=0, dropouth=0, wdrop=0)

    # Construct decoder inputs, target sequences, embedding
    dec_input_seq = torch.from_numpy(np.random.choice(dec_ntokens, size=(dec_seq_length, batch_size)))
    dec_lengths = torch.from_numpy(np.random.choice(np.arange(1, dec_seq_length+1), size=(batch_size)))
    dec_lengths[-1] = dec_seq_length
    target_seq = torch.from_numpy(np.random.choice(dec_ntokens, size=(dec_seq_length, batch_size)))
    token_weights = torch.ones(dec_ntokens)
    dec_embedding = nn.Embedding(dec_ntokens, emb_size)
    context_size = emb_size * (context_type in ['emb', 'slice_emb']) + slice_dim * (context_type in ['slice', 'slice_emb'])

    # Construct RevDecoder.
    from revdecoder import RevDecoder
    rev_dec = RevDecoder(rnn_type='revlstm', h_size=h_size, nlayers=nlayers,
        embedding=dec_embedding, attn_type='general', context_size=context_size,
        dropouti=0, dropouth=0, dropouto=0, wdrop=0, dropouts=0, max_forget=max_forget,
        context_type=context_type, slice_dim=slice_dim, use_buffers=True)

    hiddens = rev_enc.init_hiddens(batch_size)

    # Forward through encoder.
    enc_hiddens, saved_hiddens, enc_buffers, main_buf, slice_buf, enc_dict = \
        rev_enc.rev_forward_allh(enc_input_seq, enc_lengths, hiddens)

    # Forward through decoder.
    if use_separate_buffers:
        # IF using the different buffers in encoder/decoder
        hiddens, dec_buffers, dec_main_buf, dec_dict =\
            rev_dec.rev_forward(dec_input_seq, enc_hiddens, None, dec_lengths)
    else:
        # IF using the same buffer in encoder/decoder
        hiddens, dec_buffers, main_buf, dec_dict =\
            rev_dec.rev_forward(dec_input_seq, enc_hiddens, main_buf, dec_lengths)
    
    if use_separate_buffers:
        dec_used_bits = dec_main_buf.bit_usage()
        dec_optimal_bits = float(dec_dict['optimal_bits'])
        dec_normal_bits = float(dec_dict['normal_bits'])
        print("Enc actual memory ratio {}".format(1.))
        print("Enc optimal memory ratio {}".format(1.))
        print("Dec actual memory ratio {}".format(dec_normal_bits / dec_used_bits))
        print("Dec optimal memory ratio {}".format(dec_normal_bits / dec_optimal_bits))
    else:
        used_bits = enc_dict['used_bits'] + main_buf.bit_usage()
        optimal_bits = float(enc_dict['optimal_bits'] + dec_dict['optimal_bits'])
        normal_bits = float(enc_dict['normal_bits'] + dec_dict['normal_bits'])
        print("Actual memory ratio {}".format(normal_bits / used_bits))
        print("Optimal memory ratio {}".format(normal_bits / optimal_bits))

    # Reverse through decoder.
    total_loss, outputs, hiddens, hidden_grads, saved_hiddens =\
        rev_dec.reverse(dec_input_seq, target_seq, hiddens, saved_hiddens, dec_buffers,
            token_weights, dec_lengths, enc_lengths, enc_input_seq, rev_enc)

    rev_enc.reverse_allh(enc_input_seq, enc_lengths, hiddens, hidden_grads, saved_hiddens,
        enc_buffers)


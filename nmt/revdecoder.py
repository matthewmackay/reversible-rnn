import sys
import ipdb
from numpy import prod

import torch
import torch.nn as nn
import torch.nn.functional as F

import onmt.modules
from onmt.Models import RNNDecoderState

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
        self.slice_dim = slice_dim

        # Set up the standard attention.
        self.attn_type = attn_type
        if attn_type != 'none':
            self.attn = onmt.modules.MultiSizeAttention(h_size, context_size=context_size,
                attn_type=attn_type)

    def forward_test(self, input_seq, hiddens, context, enc_lengths, dec_lengths=None):
        # Set up.
        seq_length, batch_size = input_seq.size()
        self.set_masks(batch_size, input_seq.device)

        # Find last hidden states of model.
        top_hiddens = []
        input_seq = self.lockdropi(self.embedding(input_seq))
        with torch.set_grad_enabled(self.training):
            for t in range(len(input_seq)):
                mask = None if dec_lengths is None else (t < dec_lengths).int()
                curr_input = input_seq[t]

                for l, (rnn, hidden) in enumerate(zip(self.rnns, hiddens)):
                    next_hidden, stats = rnn(curr_input, hidden, buf=None, slice_dim=0, mask=mask)
                    if l != self.nlayers-1:
                        curr_input = self.lockdroph(next_hidden['output_hidden'])
                    else:
                        top_hiddens.append(next_hidden['output_hidden'])

                    hiddens[l] = next_hidden['recurrent_hidden']

        top_hiddens = self.lockdropo(torch.stack(top_hiddens))
        attn_hiddens, attn_scores = self.attn(top_hiddens.transpose(0, 1).contiguous(),
            context.transpose(0, 1), context_lengths=enc_lengths)

        attns = {"std": attn_scores}
        output = self.generator(attn_hiddens.view(-1, attn_hiddens.size(2)))
        return output, hiddens, attns


    def forward(self, input_seq, target_seq, hiddens, saved_hiddens, main_buf,
                token_weights, padding_idx, dec_lengths, enc_lengths, enc_input_seq, rev_enc):
        # Set up.
        seq_length, batch_size = input_seq.size()
        self.set_masks(batch_size, input_seq.device)

        total_loss = 0.
        num_words = 0.0
        num_correct = 0.0

        # Intialize information buffers. To limit unused space at the end of each buffer,
        # use the same information buffer for all hiddens of the same size.
        # This means using the buffer 'main_buf' from the encoder.
        buffers = []
        if main_buf is None and self.use_buffers:
            main_buf = InformationBuffer(batch_size, self.h_size//2, input_seq.device)
        for l in range(len(hiddens)):
            if self.rnn_type == 'revlstm':
                buffers.append((main_buf, main_buf, main_buf, main_buf)) # Note: main_buf can be None
            else:
                buffers.append((main_buf, main_buf))

        # Initialize output dictionary.
        output_dict = {'optimal_bits': 0}
        output_dict['normal_bits'] = sum([32*seq_length*batch_size*self.h_size for l in range(self.nlayers)])
        if self.rnn_type == 'revlstm':
            output_dict['normal_bits'] *= 2

        # Find last hidden states of model.
        top_hiddens = []
        input_seq = self.lockdropi(self.embedding(input_seq))
        with torch.set_grad_enabled(self.training):
            for t in range(len(input_seq)):
                mask = None if dec_lengths is None else (t < dec_lengths).int()
                curr_input = input_seq[t]

                for l, (rnn, buf, hidden) in enumerate(zip(self.rnns, buffers, hiddens)):
                    next_hidden, stats = rnn(curr_input, hidden, buf, slice_dim=0, mask=mask)
                    if l != self.nlayers-1:
                        curr_input = self.lockdroph(next_hidden['output_hidden'])
                    else:
                        top_hidden_ = next_hidden['output_hidden']

                    hiddens[l] = next_hidden['recurrent_hidden']
                    output_dict['optimal_bits'] += stats['optimal_bits']

                top_hidden_ = self.lockdropo(top_hidden_).unsqueeze(0)

                context = self.construct_context(saved_hiddens, enc_input_seq, rev_enc)
                attn_hidden, _ = self.attn(top_hidden_.transpose(0, 1).contiguous(),
                    context.transpose(0, 1), context_lengths=enc_lengths)
                output = self.generator(attn_hidden[0])

                loss = F.cross_entropy(output, target_seq[t], weight=token_weights, size_average=False)
                total_loss += loss

                non_padding = target_seq[t].ne(padding_idx)
                pred = F.log_softmax(output).max(1)[1]
                num_words += non_padding.sum().item()
                num_correct += pred.eq(target_seq[t]).masked_select(non_padding).sum().item()

        attns = {}
        output_dict['last_h'] = hiddens
        output_dict['used_bits'] = float('inf')
        return total_loss, num_words, num_correct, attns, main_buf, output_dict


    def rev_forward(self, input_seq, hiddens, main_buf, dec_lengths=None):
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
        if main_buf is None:
            main_buf = InformationBuffer(batch_size, self.h_size//2, input_seq.device)
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
                    output_dict['optimal_bits'] += stats['optimal_bits']

        output_dict['last_h'] = hiddens
        output_dict['used_bits'] = float('inf') # TODO(mmackay): figure out right way to compute bits
        return hiddens, buffers, main_buf, output_dict


    def reverse(self, input_seq, target_seq, last_hiddens, saved_hiddens, buffers,
                token_weights, padding_idx, dec_lengths, enc_lengths, enc_input_seq, rev_enc):
        """
        Arguments:
            input_seq (LongTensor): size is (dec_seq_length, batch_size)
            target_seq (IntTensor): size is (dec_seq_length, batch_size)
            last_hiddens (list): list of IntTensors (each with size (batch_size, h_size))
                of length nlayers
            saved_hiddens (list): list of IntTensors (each with size (batch_size, slice_dim))
                of length enc_seq_length
            buffers (list): list of InformationBuffers of length nlayers
            token_weights (FloatTensor): of size (dec_ntokens,)
            dec_lengths (IntTensor): size is (batch_size,)
            enc_lengths (IntTensor): size is (batch_size,)
            enc_input_seq (LongTensor): size is (enc_seq_length, batch_size)
            rev_enc (RevEncoder): RevEncoder module used before the RevDecoder
        """

        batch_size = float(enc_lengths.shape[0])

        hiddens = last_hiddens
        hidden_grads = [next(self.parameters()).new_zeros(h.size()) for h in hiddens]
        loss_fun = lambda output, target: F.cross_entropy(output, target, weight=token_weights, size_average=False)

        saved_hiddens = [hidden.requires_grad_() for hidden in saved_hiddens]
        total_loss = 0.
        num_words = 0.0
        num_correct = 0.0

        for t in reversed(range(len(input_seq))):
            top_hidden = hiddens[-1].requires_grad_()
            top_hidden_ = ConvertToFloat.apply(top_hidden[:,:self.h_size], hidden_radix)
            top_hidden_ = self.lockdropo(top_hidden_).unsqueeze(0)

            context = self.construct_context(saved_hiddens, enc_input_seq, rev_enc)
            attn_hidden, _ = self.attn(top_hidden_.transpose(0, 1).contiguous(),
                context.transpose(0, 1), context_lengths=enc_lengths)
            output = self.generator(attn_hidden[0])

            last_loss = loss_fun(output, target_seq[t])
            last_loss.div(batch_size).backward()

            hidden_grads[-1] += top_hidden.grad
            total_loss += last_loss

            non_padding = target_seq[t].ne(padding_idx)
            pred = F.log_softmax(output).max(1)[1]
            num_words += non_padding.sum().item()
            num_correct += pred.eq(target_seq[t]).masked_select(non_padding).sum().item()

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

        return total_loss, num_words, num_correct, hiddens, hidden_grads, saved_hiddens

    def construct_context(self, saved_hiddens, enc_input_seq, rev_enc):
        if self.context_type == 'emb':
            context = rev_enc.lockdropi(rev_enc.encoder(enc_input_seq))
        elif self.context_type == 'slice':
            context = ConvertToFloat.apply(torch.stack(saved_hiddens[1:]), hidden_radix)
            context = context[:,:,:self.slice_dim]
            context = self.lockdrops(context)
        elif self.context_type == 'slice_emb':
            embs = rev_enc.lockdropi(rev_enc.encoder(enc_input_seq))
            slices = ConvertToFloat.apply(torch.stack(saved_hiddens[1:]), hidden_radix)
            slices = slices[:,:,:self.slice_dim]
            slices = self.lockdrops(slices)
            context = torch.cat([embs, slices], dim=2)
        return context

    def set_masks(self, batch_size, device='cuda'):
        self.lockdropi.set_mask(batch_size, device=device)
        self.lockdroph.set_mask(batch_size, device=device)
        self.lockdropo.set_mask(batch_size, device=device)
        self.lockdrops.set_mask(batch_size, device=device)
        for rnn in self.rnns:
            rnn.set_mask()

    @property
    def _input_size(self):
        """
        Private helper returning the number of expected features.
        """
        return self.embeddings.embedding_dim

if __name__ == '__main__':
    import numpy as np

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
    context_type = 'slice'

    use_separate_buffers = True

    for max_forget in [0.5, 0.75, 0.875, 0.96875, 1]:

        print('\nMAX FORGETTING {} BITS'.format(max_forget))
        print('=' * 80)

        # Construct encoder inputs, embedding.
        enc_input_seq = torch.from_numpy(np.random.choice(enc_ntokens, size=(enc_seq_length, batch_size)))
        enc_lengths = torch.from_numpy(np.random.choice(np.arange(1, enc_seq_length+1), size=(batch_size)))
        enc_lengths[-1] = enc_seq_length
        enc_embedding = nn.Embedding(enc_ntokens, emb_size)

        # Construct RevEncoder.
        from revencoder import RevEncoder
        rev_enc = RevEncoder(rnn_type='revgru', h_size=h_size, nlayers=nlayers, embedding=enc_embedding,
            slice_dim=slice_dim, max_forget=max_forget, use_buffers=True, dropouti=0, dropouth=0, wdrop=0)

        # Construct decoder inputs, target sequences, embedding
        dec_input_seq = torch.from_numpy(np.random.choice(dec_ntokens, size=(dec_seq_length, batch_size)))
        dec_lengths = torch.from_numpy(np.random.choice(np.arange(1, dec_seq_length+1), size=(batch_size)))
        dec_lengths[-1] = dec_seq_length
        target_seq = torch.from_numpy(np.random.choice(dec_ntokens, size=(dec_seq_length, batch_size)))
        token_weights = torch.ones(dec_ntokens)
        dec_embedding = nn.Embedding(dec_ntokens, emb_size)
        context_size = emb_size * (context_type in ['emb', 'slice_emb']) + slice_dim * (context_type in ['slice', 'slice_emb'])

        # Construct RevDecoder.
        rev_dec = RevDecoder(rnn_type='revgru', h_size=h_size, nlayers=nlayers,
            embedding=dec_embedding, attn_type='general', context_size=context_size,
            dropouti=0, dropouth=0, dropouto=0, wdrop=0, dropouts=0, max_forget=max_forget,
            context_type=context_type, slice_dim=slice_dim, use_buffers=True)

        #### EXAMPLE USAGE (REVERSIBLE TRAINING)
        print('REVERSIBLE TRAINING')
        print('-' * 80)

        hiddens = rev_enc.init_hiddens(batch_size)

        # Forward through encoder.
        enc_hiddens, saved_hiddens, enc_buffers, main_buf, slice_buf, enc_dict = \
            rev_enc.rev_forward(enc_input_seq, enc_lengths, hiddens)

        # Forward through decoder.
        if use_separate_buffers:
            # IF using the different buffers in encoder/decoder
            hiddens, dec_buffers, dec_main_buf, dec_dict = rev_dec.rev_forward(dec_input_seq, enc_hiddens, None, dec_lengths)
        else:
            # IF using the same buffer in encoder/decoder
            hiddens, dec_buffers, main_buf, dec_dict = rev_dec.rev_forward(dec_input_seq, enc_hiddens, main_buf, dec_lengths)

        # Compute memory usage.
        if use_separate_buffers:
            # IF using separate buffers in encoder/decoder
            enc_used_bits = main_buf.bit_usage() + slice_buf.bit_usage() + 32*slice_dim*batch_size*enc_seq_length
            dec_used_bits = dec_main_buf.bit_usage()
            enc_optimal_bits = float(enc_dict['optimal_bits'])
            dec_optimal_bits = float(dec_dict['optimal_bits'])
            enc_normal_bits = float(enc_dict['normal_bits'])
            dec_normal_bits = float(dec_dict['normal_bits'])
            print("Enc actual memory ratio {}".format(enc_normal_bits / enc_used_bits))
            print("Enc optimal memory ratio {}".format(enc_normal_bits / enc_optimal_bits))
            print("Dec actual memory ratio {}".format(dec_normal_bits / dec_used_bits))
            print("Dec optimal memory ratio {}".format(dec_normal_bits / dec_optimal_bits))
        else:
            # IF using the same buffer in encoder/decoder
            used_bits = main_buf.bit_usage() + slice_buf.bit_usage() + 32*slice_dim*batch_size*enc_seq_length
            optimal_bits = float(enc_dict['optimal_bits'] + dec_dict['optimal_bits'])
            normal_bits = float(enc_dict['normal_bits'] + dec_dict['normal_bits'])
            print("Actual memory ratio {}".format(normal_bits / used_bits))
            print("Optimal memory ratio {}".format(normal_bits / optimal_bits))

        # Reverse through decoder.
        total_loss, outputs, hiddens, hidden_grads, saved_hiddens = rev_dec.reverse(dec_input_seq, target_seq, hiddens,
            saved_hiddens, dec_buffers, token_weights, dec_lengths, enc_lengths, enc_input_seq, rev_enc)

        # Reverse through encoder.
        rev_enc.reverse(enc_input_seq, enc_lengths, hiddens, hidden_grads, saved_hiddens, enc_buffers)




        ### EXAMPLE USAGE (NORMAL TRAINING)
        print('NORMAL TRAINING')
        print('-' * 80)

        hiddens = rev_enc.init_hiddens(batch_size)

        enc_hiddens, saved_hiddens, enc_buffers, main_buf, slice_buf, enc_dict = rev_enc(enc_input_seq, enc_lengths, hiddens)

        if use_separate_buffers:
            # IF using separate buffer in encoder/decoder
            loss, output, attns, dec_main_buf, dec_dict = rev_dec(dec_input_seq, target_seq, hiddens, saved_hiddens, None,
                token_weights, dec_lengths, enc_lengths, enc_input_seq, rev_enc)
        else:
            # IF using the same buffer in encoder/decoder
            loss, output, attns, main_buf, dec_dict = rev_dec(dec_input_seq, target_seq, hiddens, saved_hiddens, main_buf,
                token_weights, dec_lengths, enc_lengths, enc_input_seq, rev_enc)

        # Compute memory usage.
        if use_separate_buffers:
            # IF using separate buffers in encoder/decoder
            enc_used_bits = main_buf.bit_usage() + slice_buf.bit_usage() + 32*slice_dim*batch_size*enc_seq_length
            dec_used_bits = dec_main_buf.bit_usage()
            enc_optimal_bits = float(enc_dict['optimal_bits'])
            dec_optimal_bits = float(dec_dict['optimal_bits'])
            enc_normal_bits = float(enc_dict['normal_bits'])
            dec_normal_bits = float(dec_dict['normal_bits'])
            print("Enc actual memory ratio {}".format(enc_normal_bits / enc_used_bits))
            print("Enc optimal memory ratio {}".format(enc_normal_bits / enc_optimal_bits))
            print("Dec actual memory ratio {}".format(dec_normal_bits / dec_used_bits))
            print("Dec optimal memory ratio {}".format(dec_normal_bits / dec_optimal_bits))
        else:
            # IF using the same buffer in encoder/decoder
            used_bits = main_buf.bit_usage() + slice_buf.bit_usage() + 32*slice_dim*batch_size*enc_seq_length
            optimal_bits = float(enc_dict['optimal_bits'] + dec_dict['optimal_bits'])
            normal_bits = float(enc_dict['normal_bits'] + dec_dict['normal_bits'])
            print("Actual memory ratio {}".format(normal_bits / used_bits))
            print("Optimal memory ratio {}".format(normal_bits / optimal_bits))

import ipdb

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

import onmt
from onmt.Utils import aeq


class EncoderBase(nn.Module):
    """
    Base encoder class. Specifies the interface used by different encoder types
    and required by :obj:`onmt.Models.NMTModel`.

    .. mermaid::

       graph BT
          A[Input]
          subgraph RNN
            C[Pos 1]
            D[Pos 2]
            E[Pos N]
          end
          F[Context]
          G[Final]
          A-->C
          A-->D
          A-->E
          C-->F
          D-->F
          E-->F
          E-->G
    """
    def _check_args(self, input, lengths=None, hidden=None):
        s_len, n_batch, n_feats = input.size()
        if lengths is not None:
            n_batch_, = lengths.size()
            aeq(n_batch, n_batch_)

    def forward(self, input, lengths=None, hidden=None):
        """
        Args:
            input (:obj:`LongTensor`):
               padded sequences of sparse indices `[src_len x batch x nfeat]`
            lengths (:obj:`LongTensor`): length of each sequence `[batch]`
            hidden (class specific):
               initial hidden state.

        Returns:k
            (tuple of :obj:`FloatTensor`, :obj:`FloatTensor`):
                * final encoder state, used to initialize decoder
                   `[layers x batch x hidden]`
                * contexts for attention, `[src_len x batch x hidden]`
        """
        raise NotImplementedError


class RNNEncoder(EncoderBase):
    """ A generic recurrent neural network encoder.

    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional (bool) : use a bidirectional RNN
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """
    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout=0.0, embeddings=None):
        super(RNNEncoder, self).__init__()
        assert embeddings is not None

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.embeddings = embeddings
        self.no_pack_padded_seq = False

        # Use pytorch version when available.
        if rnn_type == "SRU":
            # SRU doesn't support PackedSequence.
            self.no_pack_padded_seq = True
            self.rnn = onmt.modules.SRU(
                    input_size=embeddings.embedding_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    bidirectional=bidirectional)
        else:
            self.rnn = getattr(nn, rnn_type)(
                    input_size=embeddings.embedding_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    bidirectional=bidirectional)

    def forward(self, input, lengths=None, hidden=None):
        "See :obj:`EncoderBase.forward()`"
        self._check_args(input, lengths, hidden)

        emb = self.embeddings(input)
        s_len, batch, emb_dim = emb.size()

        packed_emb = emb
        if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Variable.
            lengths = lengths.view(-1).tolist()
            packed_emb = pack(emb, lengths)

        outputs, hidden_t = self.rnn(packed_emb, hidden)

        if lengths is not None and not self.no_pack_padded_seq:
            outputs = unpack(outputs)[0]

        return hidden_t, outputs


class RNNDecoderBase(nn.Module):
    """
    Base recurrent attention-based decoder class.
    Specifies the interface used by different decoder types
    and required by :obj:`onmt.Models.NMTModel`.


    .. mermaid::

       graph BT
          A[Input]
          subgraph RNN
             C[Pos 1]
             D[Pos 2]
             E[Pos N]
          end
          G[Decoder State]
          H[Decoder State]
          I[Outputs]
          F[Context]
          A--emb-->C
          A--emb-->D
          A--emb-->E
          H-->C
          C-- attn --- F
          D-- attn --- F
          E-- attn --- F
          C-->I
          D-->I
          E-->I
          E-->G
          F---I

    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional_encoder (bool) : use with a bidirectional encoder
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       attn_type (str) : see :obj:`onmt.modules.GlobalAttention`
       coverage_attn (str): see :obj:`onmt.modules.GlobalAttention`
       context_gate (str): see :obj:`onmt.modules.ContextGate`
       copy_attn (bool): setup a separate copy attention mechanism
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """
    def __init__(self, rnn_type, bidirectional_encoder, num_layers,
                 hidden_size, attn_type="general",
                 coverage_attn=False, context_gate=None,
                 copy_attn=False, dropout=0.0, embeddings=None):
        super(RNNDecoderBase, self).__init__()

        # Basic attributes.
        self.decoder_type = 'rnn'
        self.bidirectional_encoder = bidirectional_encoder
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.dropout = nn.Dropout(dropout)

        # Build the RNN.
        self.rnn = self._build_rnn(rnn_type, self._input_size, hidden_size,
                                   num_layers, dropout)

        # Set up the context gate.
        self.context_gate = None
        if context_gate is not None:
            self.context_gate = onmt.modules.context_gate_factory(
                context_gate, self._input_size,
                hidden_size, hidden_size, hidden_size
            )

        # Set up the standard attention.
        self._coverage = coverage_attn
        self.attn = onmt.modules.GlobalAttention(
            hidden_size, coverage=coverage_attn,
            attn_type=attn_type
        )

        # Set up a separated copy attention layer, if needed.
        self._copy = False
        if copy_attn:
            self.copy_attn = onmt.modules.GlobalAttention(
                hidden_size, attn_type=attn_type
            )
            self._copy = True

    def forward(self, input, context, state, context_lengths=None):
        """
        Args:
            input (`LongTensor`): sequences of padded tokens
                                `[tgt_len x batch x nfeats]`.
            context (`FloatTensor`): vectors from the encoder
                 `[src_len x batch x hidden]`.
            state (:obj:`onmt.Models.DecoderState`):
                 decoder state object to initialize the decoder
            context_lengths (`LongTensor`): the padded source lengths
                `[batch]`.
        Returns:
            (`FloatTensor`,:obj:`onmt.Models.DecoderState`,`FloatTensor`):
                * outputs: output from the decoder
                         `[tgt_len x batch x hidden]`.
                * state: final hidden state from the decoder
                * attns: distribution over src at each tgt
                        `[tgt_len x batch x src_len]`.
        """
        # Args Check
        assert isinstance(state, RNNDecoderState)
        input_len, input_batch, _ = input.size()
        contxt_len, contxt_batch, _ = context.size()
        aeq(input_batch, contxt_batch)
        # END Args Check

        # Run the forward pass of the RNN.
        hidden, outputs, attns, coverage = self._run_forward_pass(input, context, state, context_lengths=context_lengths)

        # Update the state with the result.
        final_output = outputs[-1]
        state.update_state(hidden, final_output.unsqueeze(0), coverage.unsqueeze(0) if coverage is not None else None)

        # Concatenates sequence of tensors along a new dimension.
        outputs = torch.stack(outputs)
        for k in attns:
            attns[k] = torch.stack(attns[k])

        return outputs, state, attns

    def _fix_enc_hidden(self, h):
        """
        The encoder hidden is  (layers*directions) x batch x dim.
        We need to convert it to layers x batch x (directions*dim).
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def init_decoder_state(self, src, context, enc_hidden):
        if isinstance(enc_hidden, tuple):  # LSTM
            return RNNDecoderState(context, self.hidden_size,
                                   tuple([self._fix_enc_hidden(enc_hidden[i])
                                         for i in range(len(enc_hidden))]))
        else:  # GRU
            return RNNDecoderState(context, self.hidden_size,
                                   self._fix_enc_hidden(enc_hidden))


class StdRNNDecoder(RNNDecoderBase):
    """
    Standard fully batched RNN decoder with attention.
    Faster implementation, uses CuDNN for implementation.
    See :obj:`RNNDecoderBase` for options.


    Based around the approach from
    "Neural Machine Translation By Jointly Learning To Align and Translate"
    :cite:`Bahdanau2015`


    Implemented without input_feeding and currently with no `coverage_attn`
    or `copy_attn` support.
    """
    def _run_forward_pass(self, input, context, state, context_lengths=None):
        """
        Private helper for running the specific RNN forward pass.
        Must be overriden by all subclasses.
        Args:
            input (LongTensor): a sequence of input tokens tensors
                                of size (len x batch x nfeats).
            context (FloatTensor): output(tensor sequence) from the encoder
                        RNN of size (src_len x batch x hidden_size).
            state (FloatTensor): hidden state from the encoder RNN for
                                 initializing the decoder.
            context_lengths (LongTensor): the source context lengths.
        Returns:
            hidden (Variable): final hidden state from the decoder.
            outputs ([FloatTensor]): an array of output of every time
                                     step from the decoder.
            attns (dict of (str, [FloatTensor]): a dictionary of different
                            type of attention Tensor array of every time
                            step from the decoder.
            coverage (FloatTensor, optional): coverage from the decoder.
        """
        assert not self._copy  # TODO, no support yet.
        assert not self._coverage  # TODO, no support yet.

        # Initialize local and return variables.
        outputs = []
        attns = {"std": []}
        coverage = None

        emb = self.embeddings(input)

        # Run the forward pass of the RNN.
        if isinstance(self.rnn, nn.GRU):
            rnn_output, hidden = self.rnn(emb, state.hidden[0])
        else:
            rnn_output, hidden = self.rnn(emb, state.hidden)
        # Result Check
        input_len, input_batch, _ = input.size()
        output_len, output_batch, _ = rnn_output.size()
        aeq(input_len, output_len)
        aeq(input_batch, output_batch)
        # END Result Check

        # Calculate the attention.
        attn_outputs, attn_scores = self.attn(
            rnn_output.transpose(0, 1).contiguous(),  # (output_len, batch, d)
            context.transpose(0, 1),                  # (contxt_len, batch, d)
            context_lengths=context_lengths
        )
        attns["std"] = attn_scores

        # Calculate the context gate.
        if self.context_gate is not None:
            outputs = self.context_gate(
                emb.view(-1, emb.size(2)),
                rnn_output.view(-1, rnn_output.size(2)),
                attn_outputs.view(-1, attn_outputs.size(2))
            )
            outputs = outputs.view(input_len, input_batch, self.hidden_size)
            outputs = self.dropout(outputs)
        else:
            outputs = self.dropout(attn_outputs)    # (input_len, batch, d)

        # Return result.
        return hidden, outputs, attns, coverage

    def _build_rnn(self, rnn_type, input_size,
                   hidden_size, num_layers, dropout):
        """
        Private helper for building standard decoder RNN.
        """
        # Use pytorch version when available.
        if rnn_type == "SRU":
            return onmt.modules.SRU(
                    input_size, hidden_size,
                    num_layers=num_layers,
                    dropout=dropout)

        return getattr(nn, rnn_type)(
            input_size, hidden_size,
            num_layers=num_layers,
            dropout=dropout)

    @property
    def _input_size(self):
        """
        Private helper returning the number of expected features.
        """
        return self.embeddings.embedding_size


class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (:obj:`EncoderBase`): an encoder object
      decoder (:obj:`RNNDecoderBase`): a decoder object
      multi<gpu (bool): setup for multigpu support
    """
    def __init__(self, encoder, decoder, multigpu=False, opt=None):
        self.multigpu = multigpu
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.opt = opt

    def forward(self, src, tgt, lengths, dec_state=None):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`. however, may be an
                image or other generic input depending on encoder.
            tgt (:obj:`LongTensor`):
                 a target sequence of size `[tgt_len x batch]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
            dec_state (:obj:`DecoderState`, optional): initial decoder state
        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):

                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
                 * final decoder state
        """
        tgt = tgt[:-1].squeeze()  # exclude last target from inputs
        src = src.squeeze()

        encoder_output_tuple = self.encoder(src, lengths)

        if len(encoder_output_tuple) == 3:
            enc_hidden, context, enc_output_dict = encoder_output_tuple
        elif len(encoder_output_tuple) == 2:
            enc_hidden, context = encoder_output_tuple

        enc_state = self.decoder.init_decoder_state(enc_hidden)

        decoder_output_tuple = self.decoder(tgt,
                                            context,
                                            enc_state if dec_state is None else dec_state,
                                            context_lengths=lengths)
        if len(decoder_output_tuple) == 4:
            out, dec_state, attns, dec_output_dict = decoder_output_tuple
        elif len(decoder_output_tuple) == 3:
            out, dec_state, attns = decoder_output_tuple

        if len(encoder_output_tuple) == 3 and len(decoder_output_tuple) == 4:
            return out, attns, dec_state, enc_output_dict, dec_output_dict
        else:
            return out, attns, dec_state


class RevNMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (:obj:`EncoderBase`): an encoder object
      decoder (:obj:`RNNDecoderBase`): a decoder object
      multi<gpu (bool): setup for multigpu support
    """
    def __init__(self, encoder, decoder, multigpu=False, opt=None):
        self.multigpu = multigpu
        super(RevNMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.opt = opt

    def forward(self, src, tgt, src_lengths, tgt_lengths, token_weights, padding_idx, dec_state=None):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`. however, may be an
                image or other generic input depending on encoder.
            tgt (:obj:`LongTensor`):
                 a target sequence of size `[tgt_len x batch]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
            dec_state (:obj:`DecoderState`, optional): initial decoder state
        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):

                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
                 * final decoder state
        """

        batch_size = src.size(1)
        enc_input_seq = src.squeeze()
        dec_input_seq = tgt[:-1].squeeze()
        target_seq = tgt[1:].squeeze()
        enc_lengths = src_lengths
        dec_lengths = tgt_lengths

        hiddens = self.encoder.init_hiddens(batch_size)
        enc_hiddens, saved_hiddens, enc_buffers, main_buf, slice_buf, enc_dict = self.encoder(enc_input_seq, enc_lengths, hiddens)

        enc_seq_length = enc_input_seq.size(0)

        mem_dict = {}

        if self.opt.separate_buffers:
            # IF using separate buffers in encoder/decoder
            loss, num_words, num_correct, attn, dec_main_buf, dec_dict = self.decoder(dec_input_seq, target_seq, enc_hiddens, saved_hiddens, None,
                                                                                      token_weights, padding_idx, dec_lengths, enc_lengths, enc_input_seq, self.encoder)

            if self.encoder.slice_dim == self.encoder.h_size:
                mem_dict['enc_used_bits'] = 1.0
                mem_dict['dec_used_bits'] = dec_main_buf.bit_usage()
                mem_dict['enc_optimal_bits'] = 1.0
                mem_dict['dec_optimal_bits'] = float(dec_dict['optimal_bits'])
                mem_dict['enc_normal_bits'] = 1.0
                mem_dict['dec_normal_bits'] = float(dec_dict['normal_bits'])
            else:
                mem_dict['enc_used_bits'] = main_buf.bit_usage() + slice_buf.bit_usage() + 32 * self.opt.slice_dim * batch_size * enc_seq_length
                mem_dict['dec_used_bits'] = dec_main_buf.bit_usage()
                mem_dict['enc_optimal_bits'] = float(enc_dict['optimal_bits'])
                mem_dict['dec_optimal_bits'] = float(dec_dict['optimal_bits'])
                mem_dict['enc_normal_bits'] = float(enc_dict['normal_bits'])
                mem_dict['dec_normal_bits'] = float(dec_dict['normal_bits'])
        else:
            loss, output, attn, main_buf, dec_dict = self.decoder(dec_input_seq, target_seq, enc_hiddens, saved_hiddens, main_buf,
                                                                  token_weights, dec_lengths, enc_lengths, enc_input_seq, self.encoder)

            # IF using the same buffer in encoder/decoder
            mem_dict['used_bits'] = main_buf.bit_usage() + slice_buf.bit_usage() + 32 * self.opt.slice_dim * batch_size * enc_seq_length
            mem_dict['optimal_bits'] = float(enc_dict['optimal_bits'] + dec_dict['optimal_bits'])
            mem_dict['normal_bits'] = float(enc_dict['normal_bits'] + dec_dict['normal_bits'])

        return loss, num_words, num_correct, attn, enc_dict, dec_dict, mem_dict

    def forward_and_backward(self, src, tgt, src_lengths, tgt_lengths, token_weights, padding_idx, dec_state=None):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`. however, may be an
                image or other generic input depending on encoder.
            tgt (:obj:`LongTensor`):
                 a target sequence of size `[tgt_len x batch]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
            dec_state (:obj:`DecoderState`, optional): initial decoder state
        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):

                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
                 * final decoder state
        """
        batch_size = src.size(1)
        enc_input_seq = src.squeeze()
        dec_input_seq = tgt[:-1].squeeze()
        target_seq = tgt[1:].squeeze()
        enc_lengths = src_lengths
        dec_lengths = tgt_lengths

        hiddens = self.encoder.init_hiddens(batch_size)

        # Forward through encoder.
        enc_hiddens, saved_hiddens, enc_buffers, main_buf, slice_buf, enc_dict = self.encoder.rev_forward(enc_input_seq,
                                                                                                          enc_lengths,
                                                                                                          hiddens)

        enc_seq_length = enc_input_seq.size(0)

        mem_dict = {}

        # Compute memory usage (needs to be done here, otherwise buffers shorten during reverse).
        if self.opt.separate_buffers:
            hiddens, dec_buffers, dec_main_buf, dec_dict = self.decoder.rev_forward(dec_input_seq, enc_hiddens, None, dec_lengths)

            if self.encoder.slice_dim == self.encoder.h_size:
                enc_used_bits = 1.0
                enc_optimal_bits = 1.0
                enc_normal_bits = 1.0
                dec_used_bits = dec_main_buf.bit_usage()
                dec_optimal_bits = float(dec_dict['optimal_bits'])
                dec_normal_bits = float(dec_dict['normal_bits'])
            else:
                enc_used_bits = main_buf.bit_usage() + slice_buf.bit_usage() + 32 * self.opt.slice_dim * batch_size * enc_seq_length
                dec_used_bits = dec_main_buf.bit_usage()
                enc_optimal_bits = float(enc_dict['optimal_bits'])
                dec_optimal_bits = float(dec_dict['optimal_bits'])
                enc_normal_bits = float(enc_dict['normal_bits'])
                dec_normal_bits = float(dec_dict['normal_bits'])

            mem_dict['enc_used_bits'] = enc_used_bits
            mem_dict['dec_used_bits'] = dec_used_bits
            mem_dict['enc_optimal_bits'] = enc_optimal_bits
            mem_dict['dec_optimal_bits'] = dec_optimal_bits
            mem_dict['enc_normal_bits'] = enc_normal_bits
            mem_dict['dec_normal_bits'] = dec_normal_bits
        else:
            hiddens, dec_buffers, dec_main_buf, dec_dict = self.decoder.rev_forward(dec_input_seq, enc_hiddens, main_buf, dec_lengths)
            used_bits = main_buf.bit_usage() + slice_buf.bit_usage() + 32 * self.opt.slice_dim * batch_size * enc_seq_length
            optimal_bits = float(enc_dict['optimal_bits'] + dec_dict['optimal_bits'])
            normal_bits = float(enc_dict['normal_bits'] + dec_dict['normal_bits'])

            mem_dict['used_bits'] = used_bits
            mem_dict['optimal_bits'] = optimal_bits
            mem_dict['normal_bits'] = normal_bits

        # Reverse through decoder.
        loss, num_words, num_correct, hiddens, hidden_grads, saved_hiddens = self.decoder.reverse(dec_input_seq,
                                                                                                  target_seq,
                                                                                                  hiddens,
                                                                                                  saved_hiddens,
                                                                                                  dec_buffers,
                                                                                                  token_weights,
                                                                                                  padding_idx,
                                                                                                  dec_lengths,
                                                                                                  enc_lengths,
                                                                                                  enc_input_seq,
                                                                                                  self.encoder)

        # Reverse through encoder.
        self.encoder.reverse(enc_input_seq, enc_lengths, hiddens, hidden_grads, saved_hiddens, enc_buffers)
        return loss, num_words, num_correct, enc_dict, dec_dict, mem_dict


class DecoderState(object):
    """Interface for grouping together the current state of a recurrent
    decoder. In the simplest case just represents the hidden state of
    the model.  But can also be used for implementing various forms of
    input_feeding and non-recurrent models.

    Modules need to implement this to utilize beam search decoding.
    """
    def detach(self):
        for i in range(len(self._all)):
            if self._all[i] is not None:
                self._all[i] = self._all[i].detach()


class RNNDecoderState(DecoderState):
    def __init__(self, rnnstate):
        """
        Args:
            context (FloatTensor): output from the encoder of size
                                   len x batch x rnn_size.
            hidden_size (int): the size of hidden layer of the decoder.
            rnnstate (Variable): final hidden state from the encoder.
                transformed to shape: layers x batch x (directions*dim).
            input_feed (FloatTensor): output from last layer of the decoder.
            coverage (FloatTensor): coverage output from the decoder.
        """
        if not isinstance(rnnstate, (list, tuple)):
            self.hidden = [rnnstate]
        else:
            self.hidden = rnnstate

    @property
    def _all(self):
        return self.hidden

    def update_state(self, rnnstate):
        if not isinstance(rnnstate, (list, tuple)):
            self.hidden = [rnnstate]
        else:
            self.hidden = rnnstate

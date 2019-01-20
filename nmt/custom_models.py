import sys
import ipdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import onmt.Models
import onmt.modules
from onmt.Utils import aeq
from onmt.Models import RNNDecoderState

sys.path.insert(0, '..')
from language_modelling.locked_dropout import LockedDropout


class MyEncoder(nn.Module):
    def __init__(self, rnn_type, nhid, num_layers, embeddings,
                 context_type='hidden', slice_dim=100,
                 dropoute=0, dropouti=0, dropouth=0, dropouto=0, wdrop=0):

        super(MyEncoder, self).__init__()

        self.embeddings = embeddings

        if rnn_type == 'lstm':
            self.rnns = [MyLSTM(nhid, nhid) for l in range(num_layers)]
        elif rnn_type == 'gru':
            self.rnns = [MyGRU(nhid, nhid) for l in range(num_layers)]

        print(self.rnns)
        self.rnns = torch.nn.ModuleList(self.rnns)

        self.lockdrop = LockedDropout()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.num_layers = num_layers
        self.context_type = context_type
        self.slice_dim = slice_dim
        self.dropoute = dropoute
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropouto = dropouto
        self.wdrop = wdrop

    def forward(self, input, lengths, hidden=None):
        """
        """
        if hidden is None:
            hidden = self.init_hidden(input.size(1))

        emb = self.embeddings(input)
        emb = self.lockdrop(emb, self.dropouti)

        outputs = []
        raw_outputs = []
        new_hidden = []

        raw_output = emb

        for l, rnn in enumerate(self.rnns):
            raw_output, new_h = rnn(raw_output, hidden[l], lengths)
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.num_layers - 1:
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)

        hidden = new_hidden
        output = self.lockdrop(raw_output, self.dropouto)
        outputs.append(output)

        if self.rnn_type == 'lstm':
            # Necessary to reshape hidden and cells state for multiple layers to, e.g., 2 x 64 x 300
            hidden = (torch.cat([hidden[l][0] for l in range(len(hidden))]),
                      torch.cat([hidden[l][1] for l in range(len(hidden))]))
        elif self.rnn_type == 'gru':
            hidden = torch.cat([hidden[l] for l in range(len(hidden))])

        if self.context_type == 'hidden':
            return hidden, output
        elif self.context_type == 'emb':
            return hidden, emb
        elif self.context_type == 'slice':
            return hidden, output[:,:,:self.slice_dim]
        elif self.context_type == 'slice_emb':
            return hidden, torch.cat([emb, output[:,:,:self.slice_dim]], dim=2)
        else:
            return hidden, output

    def init_hidden(self, bsz):
        if self.rnn_type == 'lstm':
            weight = next(self.parameters()).data
            return [(Variable(weight.new(1, bsz, self.nhid).zero_()),
                    Variable(weight.new(1, bsz, self.nhid).zero_()))
                    for l in range(self.num_layers)]
        elif self.rnn_type == 'gru':
            weight = next(self.parameters()).data
            return [Variable(weight.new(1, bsz, self.nhid)).zero_() for l in range(self.num_layers)]


class MyLSTM(nn.Module):
    """Cell for dropconnect RNN."""

    def __init__(self, ninp, nhid):
        super(MyLSTM, self).__init__()

        self.ninp = ninp
        self.nhid = nhid

        self.i2h = nn.Linear(ninp, 4*nhid)
        self.h2h = nn.Linear(nhid, 4*nhid)

    def forward(self, input, hidden, lengths=None):

        hidden_list = []
        nhid = self.nhid
        h, cell = hidden

        if lengths is not None:
            max_length = lengths.max()
        else:
            max_length = len(input)

        # Loop over the indexes in the sequence --> process each index in parallel across items in the batch
        for i in range(max_length):

            h = h.squeeze()
            cell = cell.squeeze()

            x = input[i]

            x_components = self.i2h(x)
            h_components = self.h2h(h)

            preactivations = x_components + h_components

            gates_together = F.sigmoid(preactivations[:, 0:3*nhid])
            forget_gate = gates_together[:, 0:nhid]
            input_gate = gates_together[:, nhid:2*nhid]
            output_gate = gates_together[:, 2*nhid:3*nhid]
            new_cell = F.tanh(preactivations[:, 3*nhid:4*nhid])

            cell_new = forget_gate * cell + input_gate * new_cell
            h_new = output_gate * F.tanh(cell)

            if lengths is not None:
                # Masking step
                mask = (i < lengths).float().unsqueeze(1).expand_as(h)
                mask = Variable(mask.cuda())
                h = h_new * mask + h * (1 - mask)
                cell = cell_new * mask + cell * (1 - mask)
            else:
                h = h_new
                cell = cell_new

            hidden_list.append(h)

        hidden_stacked = torch.stack(hidden_list)
        return hidden_stacked, (h.unsqueeze(0), cell.unsqueeze(0))


class MyGRU(nn.Module):
    def __init__(self, ninp, nhid):
        super(MyGRU, self).__init__()

        self.ninp = ninp
        self.nhid = nhid

        # Input linear layers
        self.Wiz = nn.Linear(ninp, nhid)
        self.Wir = nn.Linear(ninp, nhid)
        self.Wih = nn.Linear(ninp, nhid)

        # Hidden linear layers
        self.Whz = nn.Linear(nhid, nhid)
        self.Whr = nn.Linear(nhid, nhid)
        self.Whh = nn.Linear(nhid, nhid)

    def forward(self, input, h, lengths=None):

        if lengths is not None:
            max_length = lengths.max()
        else:
            max_length = len(input)

        hidden_list = []
        h = h.squeeze(0)

        # Loop over indexes in the sequence; process each index in parallel across all items in the batch
        for i in range(max_length):

            x = input[i]

            z = F.sigmoid(self.Whz(h) + self.Wiz(x))
            r = F.sigmoid(self.Whr(h) + self.Wir(x))
            h_candidate = F.tanh(self.Whh(r * h) + self.Wih(x))

            h_new = (1 - z) * h + z * h_candidate

            if lengths is not None:
                # Masking step
                mask = (i < lengths).float().unsqueeze(1).expand_as(h)
                mask = Variable(mask.cuda())
                h = h_new * mask + h * (1 - mask)
            else:
                h = h_new

            hidden_list.append(h)

        hidden_stacked = torch.stack(hidden_list)
        return hidden_stacked, h.unsqueeze(0)


class MyDecoder(nn.Module):
    """
    Base recurrent attention-based decoder class.
    Specifies the interface used by different decoder types
    and required by :obj:`onmt.Models.NMTModel`.

    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional_encoder (bool) : use with a bidirectional encoder
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       attn_type (str) : see :obj:`onmt.modules.GlobalAttention`
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """
    def __init__(self,
                 rnn_type,
                 num_layers,
                 hidden_size,
                 attn_type="general",
                 context_size=None,
                 dropout=0.0,
                 embeddings=None,
                 dropouti=0,
                 dropouth=0,
                 dropouto=0,
                 wdrop=0):
        super(MyDecoder, self).__init__()

        self.decoder_type = 'rnn'
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.dropout = nn.Dropout(dropout)
        self.attn_type = attn_type
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropouto = dropouto
        self.wdrop = wdrop

        self.generator = nn.Linear(hidden_size, embeddings.num_embeddings)

        if not context_size:
            context_size = hidden_size

        self.rnn_type = rnn_type
        self.input_size = self._input_size

        self.lockdrop = LockedDropout()

        if rnn_type == 'lstm':
            self.rnns = [MyLSTM(self.input_size, self.hidden_size) for l in range(self.num_layers)]
        elif rnn_type == 'gru':
            self.rnns = [MyGRU(self.input_size, self.hidden_size) for l in range(self.num_layers)]

        print(self.rnns)
        self.rnns = torch.nn.ModuleList(self.rnns)

        # Set up the standard attention.
        if attn_type != 'none':
            self.attn = onmt.modules.MultiSizeAttention(
                hidden_size, context_size=context_size, attn_type=attn_type
            )

    def forward(self, input, context, state, context_lengths=None):

        outputs = []
        attns = {"std": []}
        coverage = None

        emb = self.embeddings(input)

        hidden = state.hidden

        input = self.lockdrop(emb, self.dropouti)

        outputs = []
        raw_outputs = []
        new_hidden = []

        raw_output = input

        for (l, rnn) in enumerate(self.rnns):
            if self.rnn_type == 'lstm':
                raw_output, new_h = rnn(raw_output, (hidden[0][l].unsqueeze(0), hidden[1][l].unsqueeze(0)))
            elif self.rnn_type == 'gru':
                raw_output, new_h = rnn(raw_output, hidden[0][l].unsqueeze(0))

            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.num_layers - 1:
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)

        hidden = new_hidden
        output = self.lockdrop(raw_output, self.dropouto)
        outputs.append(output)

        if self.rnn_type == 'lstm':
            # Necessary to reshape hidden and cells state for multiple layers to, e.g., 2 x 64 x 300
            hidden = (torch.cat([hidden[l][0] for l in range(len(hidden))]),
                      torch.cat([hidden[l][1] for l in range(len(hidden))]))
        elif self.rnn_type == 'gru':
            hidden = torch.cat([hidden[l] for l in range(len(hidden))])

        rnn_output = output


        # Calculate the attention.
        if self.attn_type != 'none':
            attn_outputs, attn_scores = self.attn(
                rnn_output.transpose(0, 1).contiguous(),
                context.transpose(0, 1),
                context_lengths=context_lengths
            )
            attns['std'] = attn_scores
        else:
            attns['std'] = Variable(torch.zeros(output_len, output_batch, context.size(0)).cuda())

        # Calculate the context gate.
        if self.attn_type == 'none':
            outputs = self.dropout(rnn_output)
        else:
            outputs = self.dropout(attn_outputs)

        # Update the state with the result.
        # final_output = outputs[-1]
        state.update_state(hidden)

        return outputs, state, attns

    def init_decoder_state(self, enc_hidden):
        return RNNDecoderState(enc_hidden)

    @property
    def _input_size(self):
        """
        Private helper returning the number of expected features.
        """
        return self.embeddings.embedding_dim

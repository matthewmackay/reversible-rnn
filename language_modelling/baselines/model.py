import os, sys
import torch
import torch.nn as nn
from baselines.weight_drop import WeightDrop

from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout

class RNNModel(nn.Module):
    """Container module with an embedding layer, a recurrent module, and a classifier
    output layer."""

    def __init__(self, rnn_type, ntoken, in_size, h_size, nlayers, dropout=0.5,
        dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0, max_forget=-1,
        use_buffers=False):
        
        super(RNNModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        self.drop = nn.Dropout(dropout)
        self.embed = nn.Embedding(ntoken, in_size)
        assert rnn_type in ['lstm', 'gru'], 'RNN type is not supported'
        if rnn_type == 'lstm':
            self.rnns = [torch.nn.LSTM(in_size if l == 0 else h_size, 
                h_size if l != nlayers - 1 else in_size, 1, dropout=0) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        if rnn_type == 'gru':
            self.rnns = [torch.nn.GRU(in_size if l == 0 else h_size,
                h_size if l != nlayers - 1 else in_size, 1, dropout=0) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]

        self.rnns = torch.nn.ModuleList(self.rnns)
        self.out = nn.Linear(h_size, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        self.out.weight = self.embed.weight
        self.init_weights()

        self.rnn_type = rnn_type
        self.in_size = in_size
        self.h_size = h_size
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute

    def init_weights(self):
        initrange = 0.1
        self.out.bias.data.fill_(0)
        self.out.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = embedded_dropout(self.embed, input,
            dropout=self.dropoute if self.training else 0)
        #emb = self.idrop(emb)

        emb = self.lockdrop(emb, self.dropouti)

        raw_output = emb
        new_hidden = []
        #raw_output, hidden = self.rnn(emb, hidden)
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            current_input = raw_output
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)

        decoded = self.out(output.view(output.size(0)*output.size(1), output.size(2)))
        result = decoded.view(output.size(0), output.size(1), decoded.size(1))
        output_dict = {'decoded': result, 'last_h': hidden}
        return output_dict

    def init_hiddens(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'lstm':
            return [(weight.new(1, bsz, self.h_size if l != self.nlayers - 1 else self.in_size).zero_(),
                    weight.new(1, bsz, self.h_size if l != self.nlayers - 1 else self.in_size).zero_())
                    for l in range(self.nlayers)]
        elif self.rnn_type == 'gru':
            return [weight.new(1, bsz, self.h_size if l != self.nlayers - 1 else self.in_size).zero_()
                    for l in range(self.nlayers)]

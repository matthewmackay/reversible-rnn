import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import prod
from usual_training.weight_drop import WeightDrop

sys.path.insert(0, os.path.abspath(".."))
from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout

sys.path.insert(0, os.path.abspath("../.."))
from revlstm import RevLSTM
from revgru import RevGRU
from fixed_util import ConvertToFloat, ConvertToFixed
from buffer import InformationBuffer

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
        self.dropout = dropout
        self.dropouth = dropouth
        self.dropouti = dropouti
        self.dropoute = dropoute

        self.lockdrop = LockedDropout()
        self.embed = nn.Embedding(ntoken, in_size)

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
        if wdrop:
            self.rnns = [WeightDrop(rnn, module_names, wdrop) for rnn in self.rnns]
        self.rnns = nn.ModuleList(self.rnns)

        # Initialize linear transform from hidden states to log probs.
        self.out = nn.Linear(in_size, ntoken)
        self.out.weight = self.embed.weight
        self.init_weights()

    def forward(self, input_seq, hiddens):
        """
        Arguments:
            input_seq (LongTensor): of shape (seq_length, batch_size)
            hiddens (list): list of Tensors of length nlayers

        """
        # Intialize information buffers. To limit unused space at the end of each buffer, 
        # use the same information buffer for all hiddens of the same size.
        seq_length, batch_size = input_seq.size()
        buffers = [None for _ in range(self.nlayers)]
        if self.use_buffers:
            buffers = []
            unique_buffers = []       
            for l in range(self.nlayers):
                if l in [0, self.nlayers-1]:
                    buf_dim = self.h_size//2 if l != self.nlayers-1 else self.in_size//2
                    buf = InformationBuffer(batch_size, buf_dim, input_seq.device)
                    buf_tup = (buf, buf) if self.rnn_type == 'revgru' else (buf, buf, buf, buf)
                    unique_buffers.append(buf)
                else:
                    buf_tup = buffers[l-1]
                buffers.append(buf_tup)

        # Embed input sequence.
        input_seq = embedded_dropout(self.embed, input_seq,
            dropout=self.dropoute if self.training else 0) 
        input_seq = self.lockdrop(input_seq, self.dropouti)

        # Process input sequence through model. Start with finding all hidden states
        # for current layer. Then use these hidden states as inputs to the next layer.
        output_dict = {"optimal_bits": 0}
        last_hiddens = []
        curr_seq = input_seq
        for rnn in self.rnns:
            rnn.set_weights()
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
                curr_seq = self.lockdrop(curr_seq, self.dropouth)

        # Use the last layer hiddens as inputs to our classifier.
        curr_seq = self.lockdrop(curr_seq, self.dropout)
        decoded = self.out(curr_seq.view(curr_seq.size(0) * curr_seq.size(1), -1))
        output_dict['decoded'] = decoded.view(curr_seq.size(0), curr_seq.size(1), -1)
        output_dict['last_h'] = last_hiddens
        
        # Collect stats over entire sequence. 
        if self.use_buffers:
            output_dict['used_bits'] = sum([buf.bit_usage() for buf in unique_buffers])
        output_dict['normal_bits'] = sum([32*seq_length*batch_size*(self.h_size if l != self.nlayers-1 else self.in_size) for l in range(self.nlayers)])
        if self.rnn_type == 'revlstm':
            output_dict['normal_bits'] *= 2
        
        return output_dict 

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
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
import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightDrop(torch.nn.Module):
    def __init__(self, rnn, module_names, dropout=0):
        super(WeightDrop, self).__init__()
        self.rnn = rnn
        self.module_names = module_names
        self.dropout = dropout
        
        recur_mask = next(rnn.parameters()).new_ones(rnn.in_size + rnn.h_size//2)
        recur_mask[rnn.in_size:] = 0
        self.register_buffer('recur_mask', recur_mask)
        self._setup()

    def _setup(self):
        for m_name in self.module_names:
            module = getattr(self.rnn, m_name)
            w = module.weight
            del module._parameters['weight']
            module.register_parameter('raw_weight', nn.Parameter(w.data))

    def set_weights(self):
        for m_name in self.module_names:
            module = getattr(self.rnn, m_name)
            raw_w = module.raw_weight
            w = self.recur_mask * raw_w +\
                (1-self.recur_mask) * F.dropout(raw_w, self.dropout, self.training)
            setattr(module, 'weight', w)

    def forward(self, *args, **kwargs):
        return self.rnn.forward(*args, **kwargs)
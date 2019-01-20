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

    def set_mask(self):
        self.masks = {}
        for m_name in self.module_names:
            module = getattr(self.rnn, m_name)
            raw_w = module.raw_weight
            
            w_mask = raw_w.data.new(raw_w.size()).bernoulli_(1-self.dropout)
            w_mask = w_mask / (1-self.dropout)
            w_ones = torch.ones_like(raw_w)
            w_mask = (1-self.recur_mask)*w_mask + self.recur_mask*w_ones

            self.masks[m_name] = w_mask

    def set_weights(self):
        for m_name in self.module_names:
            module = getattr(self.rnn, m_name)
            raw_w = module.raw_weight.clone()
            if self.training:
                w = self.masks[m_name] * raw_w
                setattr(module, 'weight', w)
            else:
                setattr(module, 'weight', raw_w)

    def forward(self, *args, **kwargs):
        self.set_weights()
        return self.rnn.forward(*args, **kwargs)

    def reverse(self, *args, **kwargs):
        return self.rnn.reverse(*args, **kwargs)
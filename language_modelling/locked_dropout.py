import torch
import torch.nn as nn

class LockedDropout(nn.Module):
    def __init__(self):
        super(LockedDropout, self).__init__()

    def forward(self, x, dropout=0.5):
        """
        Keyword args:
        x (torch.tensor): Should have dimensions [Length, Batch, Dim]

        Applies dropout to elements of x where Bernoulli mask of dimension [Batch, Dim]
        is sampled and used across Length dimension.
        """
        if not self.training or not dropout:
            return x
        mask = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = mask / (1 - dropout)
        mask.requires_grad = False
        mask = mask.expand_as(x)
        return mask * x
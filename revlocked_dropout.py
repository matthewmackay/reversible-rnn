import torch
import torch.nn as nn

class RevLockedDropout(nn.Module):
    def __init__(self, dropout, dim):
        super(RevLockedDropout, self).__init__()
        self.dropout = dropout
        self.dim = dim

    def set_mask(self, batch_size, device):
        mask = torch.ones(batch_size, self.dim).bernoulli_(1 - self.dropout).to(device)
        self.mask = mask / (1 - self.dropout)

    def forward(self, x, dropout=0.5):
        """
        Keyword args:
        x (Tensor): Should have dimensions [Batch, Dim]

        Applies dropout to elements of x where Bernoulli mask of dimension [Batch, Dim]
        is sampled and used across Length dimension.
        """
        if not self.training:
            return x
        else:
            return self.mask.expand_as(x) * x
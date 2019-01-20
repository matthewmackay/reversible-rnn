import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

ln_2 = math.log(2)
hidden_radix = 23
forget_radix = 10

from fixed_util import ConvertToFloat, ConvertToFixed, MaskFixedMultiply, EnsureNotForgetAll
from buffer import FixedMultiply, FixedDivide

class RevGRU(nn.Module):

    def __init__(self, in_size, h_size, max_forget):

        super(RevGRU, self).__init__()
        self.h_size = h_size
        self.in_size = in_size
        self.max_forget = max_forget

        self.ih2_to_zr1 = nn.Linear(in_size + h_size // 2, h_size)
        self.irh2_to_g1 = nn.Linear(in_size + h_size // 2, h_size // 2)
        self.ih1_to_zr2 = nn.Linear(in_size + h_size // 2, h_size)
        self.irh1_to_g2 = nn.Linear(in_size + h_size // 2, h_size // 2)

    def forward(self, input, hidden, buf=None, slice_dim=0, mask=None):
        """
        Arguments:
            input (FloatTensor): Of size (batch_size, in_size)
            hidden (IntTensor): Of size (batch_size, h_size)
            buf (InformationBuffer): Stores bits lost in forgetting
            slice_dim (int): how large of a slice needs to be saved for attention
            mask (IntTensor): Of size (batch_size), used for masking different length
                sequences in NMT
        """
        # Set up.
        if buf is not None:
            buf_h1, buf_h2 = buf
        else:
            buf_h1 = buf_h2 = None
        group_size = self.h_size // 2
        h1 = hidden[:, :(self.h_size // 2)]
        h2 = hidden[:, (self.h_size // 2):]
        mask = torch.ones_like(h1) if mask is None else mask[:, None].expand_as(h1)

        # Compute update/forgetting for h1.
        h2_fl = ConvertToFloat.apply(h2, hidden_radix)
        zr1 = F.sigmoid(self.ih2_to_zr1(torch.cat([input, h2_fl], dim=1)))
        z1, r1 = zr1[:, :group_size], zr1[:, group_size:]
        z1 = self.max_forget * z1 + (1 - self.max_forget)
        
        rh2_fl = r1 * h2_fl
        g1 = F.tanh(self.irh2_to_g1(torch.cat([input, rh2_fl], dim=1)))

        # Apply update/forgetting for h1.
        z1_fix = ConvertToFixed.apply(z1, forget_radix) 
        z1_fix = EnsureNotForgetAll.apply(z1_fix, self.max_forget)
        h1 = FixedMultiply.apply(h1, z1_fix, buf_h1, mask, slice_dim)
        update1 = ConvertToFixed.apply((1 - z1) * g1, hidden_radix)
        h1 = h1 + MaskFixedMultiply.apply(update1, mask)

        # Compute update/forgetting for h2.
        h1_fl = ConvertToFloat.apply(h1, hidden_radix)
        zr2 = F.sigmoid(self.ih1_to_zr2(torch.cat([input, h1_fl], dim=1)))
        z2, r2 = zr2[:, :group_size], zr2[:, group_size:]
        z2 = self.max_forget * z2 + (1 - self.max_forget)

        rh1_fl = r2 * h1_fl
        g2 = F.tanh(self.irh1_to_g2(torch.cat([input, rh1_fl], dim=1)))

        # Apply update/forgetting for h2.
        z2_fix = ConvertToFixed.apply(z2, forget_radix)
        z2_fix = EnsureNotForgetAll.apply(z2_fix, self.max_forget)
        h2 = FixedMultiply.apply(h2, z2_fix, buf_h2, mask)
        update2 = ConvertToFixed.apply((1 - z2) * g2, hidden_radix)
        h2 = h2 + MaskFixedMultiply.apply(update2, mask)

        recurrent_hidden = torch.cat([h1, h2], dim=1)
        output_hidden = ConvertToFloat.apply(recurrent_hidden, hidden_radix)
        hidden_dict = {"recurrent_hidden": recurrent_hidden, "output_hidden": output_hidden}

        nonattn_z1 = z1[:, slice_dim:]
        optimal_bits1 = torch.sum(-torch.log(nonattn_z1.data) / ln_2)
        optimal_bits2 = torch.sum(-torch.log(z2.data) / ln_2)
        stats = {"optimal_bits": optimal_bits1 + optimal_bits2 + 32*slice_dim*input.size(0)}

        return hidden_dict, stats

    def reverse(self, input, hidden, buf, slice_dim=0, saved_hidden=None, mask=None):
        if buf is None:
            return saved_hidden.clone()

        buf_h1, buf_h2 = buf
        group_size = self.h_size // 2
        h1 = hidden[:, :group_size]
        h2 = hidden[:, group_size:]
        mask = torch.ones_like(h1) if mask is None else mask[:, None].expand_as(h1)

        # Compute update/forgetting for h2.
        h1_fl = ConvertToFloat.apply(h1, hidden_radix)
        zr2 = F.sigmoid(self.ih1_to_zr2(torch.cat([input, h1_fl], dim=1)))
        z2, r2 = zr2[:, :group_size], zr2[:, group_size:]
        z2 = self.max_forget * z2 + (1 - self.max_forget)

        rh1_fl = r2 * h1_fl
        g2 = F.tanh(self.irh1_to_g2(torch.cat([input, rh1_fl], dim=1)))

        # Reverse update/forgetting for h2.
        update2 = ConvertToFixed.apply((1 - z2) * g2, hidden_radix)
        h2 = h2 - update2 * mask
        z2_fix = ConvertToFixed.apply(z2, forget_radix)
        z2_fix = EnsureNotForgetAll.apply(z2_fix, self.max_forget)
        h2 = FixedDivide.apply(h2, z2_fix, buf_h2, mask)

        # Compute update/forgetting for h1.
        h2_fl = ConvertToFloat.apply(h2, hidden_radix)
        zr1 = F.sigmoid(self.ih2_to_zr1(torch.cat([input, h2_fl], dim=1)))
        z1, r1 = zr1[:, :group_size], zr1[:, group_size:]
        z1 = self.max_forget * z1 + (1 - self.max_forget)
        
        rh2_fl = r1 * h2_fl
        g1 = F.tanh(self.irh2_to_g1(torch.cat([input, rh2_fl], dim=1)))

        # Reverse update/forgetting for h1.
        update1 = ConvertToFixed.apply((1 - z1) * g1, hidden_radix)
        h1 = h1 - update1 * mask
        z1_fix = ConvertToFixed.apply(z1, forget_radix)
        z1_fix = EnsureNotForgetAll.apply(z1_fix, self.max_forget)
        h1 = FixedDivide.apply(h1, z1_fix, buf_h1, mask, slice_dim)
        if slice_dim > 0:
            h1[:, :slice_dim] = saved_hidden
        hidden = torch.cat([h1, h2], dim=1)

        return hidden

if __name__ == "__main__":
    # Test reversibility.
    torch.manual_seed(3)
    batch_size = 2
    h_size = 200
    slice_dim = 30
    in_size = 100
    seq_length = 100
    
    initial_hidden = ConvertToFixed.apply(torch.randn(batch_size, h_size), hidden_radix)
    input_seq = torch.randn(seq_length, batch_size, in_size)
    masks = [torch.randn(batch_size).bernoulli_(0.5).int() for _ in range(seq_length)]

    from buffer import InformationBuffer 
    buf_h1 = InformationBuffer(batch_size=batch_size, buf_dim=h_size // 2 - slice_dim, device='cpu') 
    buf_h2 = InformationBuffer(batch_size=batch_size, buf_dim=h_size // 2, device='cpu')
    buf = buf_h1, buf_h2
    rnn = RevGRU(in_size, h_size, max_forget=0.96875)

    hidden = initial_hidden
    saved_hiddens = []
    for t in range(seq_length):
        if slice_dim > 0:
            saved_hiddens.append(hidden[:,:slice_dim])
        hidden_dict, _ = rnn(input_seq[t], hidden, buf, slice_dim, masks[t])
        hidden = hidden_dict['recurrent_hidden']

    with torch.no_grad():
        for t in reversed(range(seq_length)):
            if slice_dim > 0:
                hidden = rnn.reverse(input_seq[t], hidden, buf, slice_dim, saved_hiddens[t], mask=masks[t])
            else: 
                hidden = rnn.reverse(input_seq[t], hidden, buf, slice_dim, mask=masks[t])

    print("reconstructed hidden = original hidden")
    print((hidden.data.numpy() == initial_hidden.data.numpy()).all())
import ipdb
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



class RevLSTM(nn.Module):

    def __init__(self, in_size, h_size, max_forget):

        super(RevLSTM, self).__init__()
        self.h_size = h_size
        self.in_size = in_size
        self.max_forget = max_forget
        self.ih2_to_zgfop1 = nn.Linear(in_size + (h_size // 2), 2*h_size + (h_size // 2))
        self.ih1_to_zgfop2 = nn.Linear(in_size + (h_size // 2), 2*h_size + (h_size // 2))

    def forward(self, input, hidden, buf=None, slice_dim=0, mask=None):
        """
        Arguments:
            input (FloatTensor): Of size (batch_size, in_size)
            hidden (IntTensor): Of size (batch_size, 2 * h_size)
        """
        if buf is not None:
            buf_h1, buf_h2, buf_c1, buf_c2 = buf
        else:
            buf_h1 = buf_h2 = buf_c1 = buf_c2 = None

        group_size = self.h_size // 2
        h1 = hidden[:, :group_size]
        h2 = hidden[:, group_size:2*group_size]
        c1 = hidden[:, 2*group_size:3*group_size]
        c2 = hidden[:, 3*group_size:]
        mask = torch.ones_like(h1) if mask is None else mask[:, None].expand_as(h1)

        # Compute concatenated gates required to update h1, c1.
        h2_fl = ConvertToFloat.apply(h2, hidden_radix)
        zgfop1 = self.ih2_to_zgfop1(torch.cat([input, h2_fl], dim=1))

        # Compute gates necessary to update c1.
        z1 = F.sigmoid(zgfop1[:, :group_size])
        z1 = self.max_forget * z1 + (1 - self.max_forget)
        g1 = F.tanh(zgfop1[:, group_size:2*group_size])
        f1 = F.sigmoid(zgfop1[:, 2*group_size:3*group_size])

        # Apply update/forgetting for c1.
        z1_fix = ConvertToFixed.apply(z1, forget_radix) 
        z1_fix = EnsureNotForgetAll.apply(z1_fix, self.max_forget)
        c1 = FixedMultiply.apply(c1, z1_fix, buf_c1, mask)
        update_c1 = ConvertToFixed.apply(f1 * g1, hidden_radix)
        c1 = c1 + MaskFixedMultiply.apply(update_c1, mask)

        # Compute gates necessary to update h1.
        o1 = F.sigmoid(zgfop1[:, 3*group_size:4*group_size])
        p1 = F.sigmoid(zgfop1[:, 4*group_size:])
        p1 = self.max_forget * p1 + (1 - self.max_forget)

        # Apply update/forgetting for h1.
        c1_fl = ConvertToFloat.apply(c1, hidden_radix)
        p1_fix = ConvertToFixed.apply(p1, forget_radix)
        p1_fix = EnsureNotForgetAll.apply(p1_fix, self.max_forget)
        h1 = FixedMultiply.apply(h1, p1_fix, buf_h1, mask, slice_dim)
        update_h1 = ConvertToFixed.apply(o1 * F.tanh(c1_fl), hidden_radix)
        h1 = h1 + MaskFixedMultiply.apply(update_h1, mask)

        # Compute concatenated gates required to update h2, c2.
        h1_fl = ConvertToFloat.apply(h1, hidden_radix)
        zgfop2 = self.ih1_to_zgfop2(torch.cat([input, h1_fl], dim=1))

        # Compute gates necessary to update c2.
        z2 = F.sigmoid(zgfop2[:, :group_size])
        z2 = self.max_forget * z2 + (1 - self.max_forget)
        g2 = F.tanh(zgfop2[:, group_size:2*group_size])
        f2 = F.sigmoid(zgfop2[:, 2*group_size:3*group_size])

        # Apply update/forgetting for c2.
        z2_fix = ConvertToFixed.apply(z2, forget_radix)
        z2_fix = EnsureNotForgetAll.apply(z2_fix, self.max_forget)
        c2 = FixedMultiply.apply(c2, z2_fix, buf_c2, mask)
        update_c2 = ConvertToFixed.apply(f2 * g2, hidden_radix)
        c2 = c2 + MaskFixedMultiply.apply(update_c2, mask)

        # Compute gates necessary to update h2
        o2 = F.sigmoid(zgfop2[:, 3*group_size:4*group_size])
        p2 = F.sigmoid(zgfop2[:, 4*group_size:])
        p2 = self.max_forget * p2 + (1 - self.max_forget)

        # Apply update/forgetting for h2.
        c2_fl = ConvertToFloat.apply(c2, hidden_radix)
        p2_fix = ConvertToFixed.apply(p2, forget_radix)
        p2_fix = EnsureNotForgetAll.apply(p2_fix, self.max_forget)
        h2 = FixedMultiply.apply(h2, p2_fix, buf_h2, mask)
        update_h2 = ConvertToFixed.apply(o2 * F.tanh(c2_fl), hidden_radix)
        h2 = h2 + MaskFixedMultiply.apply(update_h2, mask)

        recurrent_hidden = torch.cat([h1, h2, c1, c2], dim=1)
        output_hidden = ConvertToFloat.apply(torch.cat([h1, h2], dim=1), hidden_radix)
        hidden_dict = {"recurrent_hidden": recurrent_hidden, "output_hidden": output_hidden}

        nonattn_p1 = p1[:, slice_dim:]
        optimal_bits1 = torch.sum(-torch.log(z1.data) / ln_2) + torch.sum(-torch.log(nonattn_p1.data) / ln_2)
        optimal_bits2 = torch.sum(-torch.log(z2.data) / ln_2) + torch.sum(-torch.log(p2.data) / ln_2)
        stats = {"optimal_bits": optimal_bits1 + optimal_bits2 + 32*slice_dim*input.size(0)}

        return hidden_dict, stats

    def reverse(self, input, hidden, buf, slice_dim=0, saved_hidden=None, mask=None):
        if buf is None:
            return saved_hidden.clone() 

        buf_h1, buf_h2, buf_c1, buf_c2 = buf
        group_size = self.h_size // 2
        h1 = hidden[:, :group_size]
        h2 = hidden[:, group_size:2*group_size]
        c1 = hidden[:, 2*group_size:3*group_size]
        c2 = hidden[:, 3*group_size:]
        mask = torch.ones_like(h1) if mask is None else mask[:, None].expand_as(h1)

        # Compute concatenated gates used to update h2, c2.
        h1_fl = ConvertToFloat.apply(h1, hidden_radix)
        zgfop2 = self.ih1_to_zgfop2(torch.cat([input, h1_fl], dim=1))

        # Compute gates used to update h2
        o2 = F.sigmoid(zgfop2[:, 3*group_size:4*group_size])
        p2 = F.sigmoid(zgfop2[:, 4*group_size:])
        p2 = self.max_forget * p2 + (1 - self.max_forget)

        # Reverse update/forgetting for h2.
        c2_fl = ConvertToFloat.apply(c2, hidden_radix)
        update_h2 = ConvertToFixed.apply(o2 * F.tanh(c2_fl), hidden_radix)
        h2 = h2 - update_h2 * mask
        p2_fix = ConvertToFixed.apply(p2, forget_radix)
        p2_fix = EnsureNotForgetAll.apply(p2_fix, self.max_forget)
        h2 = FixedDivide.apply(h2, p2_fix, buf_h2, mask)

        # Compute gates used to update c2.
        z2 = F.sigmoid(zgfop2[:, :group_size])
        z2 = self.max_forget * z2 + (1 - self.max_forget)
        g2 = F.tanh(zgfop2[:, group_size:2*group_size])
        f2 = F.sigmoid(zgfop2[:, 2*group_size:3*group_size])

        # Reverse update/forgetting for c2.
        update_c2 = ConvertToFixed.apply(f2 * g2, hidden_radix)
        c2 = c2 - update_c2 * mask
        z2_fix = ConvertToFixed.apply(z2, forget_radix)
        z2_fix = EnsureNotForgetAll.apply(z2_fix, self.max_forget)
        c2 = FixedDivide.apply(c2, z2_fix, buf_c2, mask)

        # Compute concatenated gates used to update h1, c1.
        h2_fl = ConvertToFloat.apply(h2, hidden_radix)
        zgfop1 = self.ih2_to_zgfop1(torch.cat([input, h2_fl], dim=1))

        # Compute gates used to update h1.
        o1 = F.sigmoid(zgfop1[:, 3*group_size:4*group_size])
        p1 = F.sigmoid(zgfop1[:, 4*group_size:])
        p1 = self.max_forget * p1 + (1 - self.max_forget)

        # Reverse update/forgetting for h1.
        c1_fl = ConvertToFloat.apply(c1, hidden_radix)
        update_h1 = ConvertToFixed.apply(o1 * F.tanh(c1_fl), hidden_radix)
        h1 = h1 - update_h1 * mask
        p1_fix = ConvertToFixed.apply(p1, forget_radix)
        p1_fix = EnsureNotForgetAll.apply(p1_fix, self.max_forget)
        h1 = FixedDivide.apply(h1, p1_fix, buf_h1, mask, slice_dim)
        if slice_dim > 0:
            h1[:, :slice_dim] = saved_hidden

        # Compute gates used to update c1.
        z1 = F.sigmoid(zgfop1[:, :group_size])
        z1 = self.max_forget * z1 + (1 - self.max_forget)
        g1 = F.tanh(zgfop1[:, group_size:2*group_size])
        f1 = F.sigmoid(zgfop1[:, 2*group_size:3*group_size])

        # Apply update/forgetting for c1.
        update_c1 = ConvertToFixed.apply(f1 * g1, hidden_radix)
        c1 = c1 - update_c1 * mask
        z1_fix = ConvertToFixed.apply(z1, forget_radix) 
        z1_fix = EnsureNotForgetAll.apply(z1_fix, self.max_forget)
        c1 = FixedDivide.apply(c1, z1_fix, buf_c1, mask)

        hidden = torch.cat([h1, h2, c1, c2], dim=1)
        return hidden



if __name__ == "__main__":
    # Test reversibility.
    torch.manual_seed(3)
    batch_size = 2
    h_size = 200
    slice_dim = 30
    in_size = 100
    seq_length = 100
    
    initial_hidden = ConvertToFixed.apply(torch.randn(batch_size, 2*h_size), hidden_radix)
    input_seq = torch.randn(seq_length, batch_size, in_size)
    masks = [torch.randn(batch_size).bernoulli_(0.5).int() for _ in range(seq_length)]

    from buffer import InformationBuffer 
    buf_h1 = InformationBuffer(batch_size=batch_size, buf_dim=h_size // 2 - slice_dim, device='cpu') 
    buf_h2 = buf_c2 = buf_c1 = InformationBuffer(batch_size=batch_size, buf_dim=h_size // 2, device='cpu') 
    buf = buf_h1, buf_h2, buf_c1, buf_c2
    rnn = RevLSTM(in_size, h_size, max_forget=0.875)

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
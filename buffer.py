import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy import prod
import math
from fixed_util import convert_to_float

sign_bit = -2**31
ln_2 = math.log(2)

hidden_radix = 23
forget_radix = 10

###############################################################################
# Information Buffer class
###############################################################################
class InformationBuffer():

    def __init__(self, batch_size, buf_dim, device):
        self.batch_size = batch_size
        self.buf_dim = buf_dim
        
        self.overflow_int = 2**(63 - forget_radix) # Can 63 -> 64?
        self.curr_buffer = torch.zeros(batch_size, buf_dim).long().to(device)
        self.past_buffers = None
        self.overflow_detect = torch.zeros(1).byte().to(device)
        self.counter = 0

    def bit_usage(self):
        buf_bits = 64 * prod(tuple(self.curr_buffer.size()))
        if self.past_buffers is not None:
            buf_bits += 64 * prod(tuple(self.past_buffers.size()))
        overflow_bits = 8 * prod(tuple(self.overflow_detect.size()))
        return buf_bits + overflow_bits

    def overflow_mul(self, multiplicand, mask):
        # Check for overflow and record whether it has occurred.
        overflowed = ((self.curr_buffer >= self.overflow_int) * mask.byte()).any()
        if overflowed: 
            if self.past_buffers is None:
                self.past_buffers = self.curr_buffer.unsqueeze(2)
            else:
                self.past_buffers = torch.cat(
                    [self.past_buffers, self.curr_buffer.unsqueeze(2)], dim=2)
            self.curr_buffer = self.curr_buffer.new(self.batch_size, self.buf_dim).zero_()
            self.overflow_detect[self.counter // 8] += (2**(self.counter % 8))
        
        # Increment counter, make new one if needed.
        self.counter += 1
        if self.counter % 8 == 0:
            zero = self.overflow_detect.new(1).zero_()
            self.overflow_detect = torch.cat([self.overflow_detect, zero], dim=0)

        mask = mask.long()
        self.curr_buffer = mask*(self.curr_buffer*multiplicand) + (1-mask)*self.curr_buffer

    def mul(self, multiplicand, mask):
        multiplicand, mask = multiplicand.long(), mask.long()
        self.curr_buffer = mask*(multiplicand*self.curr_buffer) + (1-mask)*self.curr_buffer

    def add(self, summand, mask):
        summand, mask = summand.long(), mask.long()
        self.curr_buffer = mask*(self.curr_buffer+summand) + (1-mask)*self.curr_buffer

    def mod(self, divisor):
        divisor = divisor.long()
        return torch.remainder(self.curr_buffer, divisor).int()

    def div(self, divisor, mask):
        # Assumes the entries of buffer are always positive. They should be 
        # since remainders are always positive and so is z.
        divisor, mask = divisor.long(), mask.long()
        self.curr_buffer = mask*(self.curr_buffer / divisor) + (1-mask)*self.curr_buffer

    def mod_divide(self, forget_radix, mask):
        # Only called in reverse process. 
        mask = mask.long()
        self.counter -= 1

        buf_mod = torch.remainder(self.curr_buffer, 2**forget_radix).int()
        self.curr_buffer = mask*(self.curr_buffer/(2**forget_radix)) + (1-mask)*self.curr_buffer

        overflowed = self.overflow_detect.__and__(2**(self.counter % 8))[self.counter // 8]
        if overflowed:
            self.curr_buffer = self.past_buffers[:,:,-1]
            if self.past_buffers.size(2) > 1:
                self.past_buffers = self.past_buffers[:,:,:-1]

        return buf_mod

###############################################################################
# Multiply/divide fixed point numbers with buffer
###############################################################################
negative_bits = -2**31 + sum([2**(30 - j) for j in range(forget_radix-1)])
class FixedMultiply(torch.autograd.Function):

    @staticmethod
    def forward(ctx, h, z, buf, mask, slice_dim=0):
        ctx.save_for_backward(h, z, mask)

        # Shift buffer left, enlarging if needed, then store modulus of h in buffer.
        if buf is not None:
            h_mod = torch.remainder(h[:, slice_dim:], 2**forget_radix)
            buf.overflow_mul(2**forget_radix, mask[:, slice_dim:])
            buf.add(h_mod, mask[:, slice_dim:])

        # Multiply h by z/(2**forget_radix).
        # Have to do extra work in case h is negative.
        sign_bits = h.__and__(sign_bit)  
        one_bits = negative_bits * -1 * torch.clamp(sign_bits, min=-1)
        h = h.__rshift__(forget_radix * mask) 
        h = h.__or__(one_bits * mask)
        h = mask*h*z + (1-mask)*h

        # Store modulus of buffer in h then divide buffer by z.
        if buf is not None:
            buf_mod = buf.mod(z[:,slice_dim:])
            h[:,slice_dim:] = h[:,slice_dim:] + buf_mod*mask[:, slice_dim:]
            buf.div(z[:,slice_dim:], mask[:, slice_dim:])

        return h

    @staticmethod 
    def backward(ctx, grad_output):
        h, z, mask = ctx.saved_variables
        mask_fl = mask.float()
        h_grad = grad_output * (convert_to_float(z, forget_radix)*mask_fl + (1-mask_fl))
        z_grad = grad_output * (convert_to_float(h, hidden_radix)*mask_fl)
        return h_grad, z_grad, None, None, None

class FixedDivide(torch.autograd.Function):
    # TODO: Check this over

    @staticmethod
    def forward(ctx, h, z, buf, mask, slice_dim=0):
        buf.mul(z[:,slice_dim:], mask[:, slice_dim:])
        h_mod = torch.remainder(h[:,slice_dim:], z[:,slice_dim:])
        buf.add(h_mod, mask[:, slice_dim:])
        h[h<0] = mask[h<0]*(h[h<0]-(z[h<0]-1)) + (1-mask[h<0])*h[h<0]
        h = mask*(h / z) + (1-mask)*h 

        h = mask*(h * (2**forget_radix)) + (1-mask)*h
        buf_mod = buf.mod_divide(forget_radix, mask[:, slice_dim:])
        h[:,slice_dim:] = mask[:, slice_dim:]*(h[:,slice_dim:].__or__(buf_mod)) +\
            (1-mask[:, slice_dim:])*h[:,slice_dim:]

        return h

    @staticmethod 
    def backward(ctx, grad_output):
        # TODO: implement this if necessary
        return grad_output, grad_output, None, None


if __name__ == '__main__':
    # Test reversibility.
    # - Working for (slice_dim=0, mask=1)
    # - Working for slice_dim>0, mask=1
    # - Working for slice_dim=0, mask!=1
    # - Working for slice_dim>0, mask!=1
    from fixed_util import ConvertToFloat, ConvertToFixed
    torch.manual_seed(4)
    batch_size = 30
    h_size = 200
    slice_dim = 20
    seq_length = 100
    
    h = ConvertToFixed.apply(torch.randn(batch_size, h_size), hidden_radix)
    zs = ConvertToFixed.apply(F.sigmoid(torch.randn(seq_length, batch_size, h_size)), forget_radix)
    masks = [torch.randn(batch_size, 1).bernoulli_(0.5).expand_as(h).int() for _ in range(seq_length)]
    # masks = [torch.ones(batch_size, 1).expand_as(h).int() for _ in range(seq_length)]
    buf = InformationBuffer(batch_size=batch_size, buf_dim=h_size-slice_dim, device='cpu')
    
    initial_hidden = h.data.numpy()
    saved_hs = []
    forward_hs = []
    for t in range(seq_length):
        saved_hs.append(h[:,:slice_dim])
        h = FixedMultiply.apply(h, zs[t], buf, masks[t], slice_dim)
        forward_hs.append(h.data.clone().numpy())

    reverse_hs = []
    for t in reversed(range(seq_length)):
        reverse_hs.append(h.data.clone().numpy())
        h = FixedDivide.apply(h, zs[t], buf, masks[t], slice_dim)
        h[:,:slice_dim] = saved_hs[t]

    print("reconstructed hidden = original hidden")
    print((h.data.numpy() == initial_hidden).all())

    all_good = True
    for forward_h, reverse_h in zip(reversed(forward_hs), reverse_hs):
        # print("FORWARD")
        # print(forward_h)
        # print("REVERSE")
        # print(reverse_h)
        all_good = all_good and (forward_h == reverse_h).all()
    print("forward hiddens = reverse hiddens")
    print(all_good)

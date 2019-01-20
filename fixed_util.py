import torch
import torch.nn as nn
import torch.nn.functional as F

###############################################################################
# Utility functions
###############################################################################
def convert_to_fixed(x, radix):
    return torch.round(x * (2**radix)).int()

def convert_to_float(x, radix):
    return (2**(-radix)) * x.float()

###############################################################################
# Custom autograd functions
###############################################################################
class ConvertToFixed(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, radix):
        return convert_to_fixed(input, radix)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.float(), None

class ConvertToFloat(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, radix):
        return convert_to_float(input, radix)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class MaskFixedMultiply(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, mask):
        ctx.save_for_backward(x, mask)
        return x * mask

    @staticmethod
    def backward(ctx, grad_output):
        x, mask = ctx.saved_variables
        return mask.float() * grad_output, None

class EnsureNotForgetAll(torch.autograd.Function):

    @staticmethod
    def forward(ctx, forget_gate, max_forget):
        if max_forget != 1:
            return forget_gate
        forget_gate[forget_gate == 0] = 1
        return forget_gate

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


if __name__ == "__main__":
    pass
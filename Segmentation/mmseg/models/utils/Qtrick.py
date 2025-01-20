import torch
import torch.nn as nn

quant = 4
T = quant


class Quant(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, i, min_value=0, max_value=quant):
        ctx.min = min_value
        ctx.max = max_value
        ctx.save_for_backward(i)
        return torch.round(torch.clamp(i, min=min_value, max=max_value))

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        i, = ctx.saved_tensors
        grad_input[i < ctx.min] = 0
        grad_input[i > ctx.max] = 0
        return grad_input, None, None


class Multispike_norm(nn.Module):
    def __init__(
            self,
            T=T,  # 在T上进行Norm
    ):
        super().__init__()
        self.spike = Quant()
        self.T = T

    def forward(self, x):

        return self.spike.apply(x) / self.T

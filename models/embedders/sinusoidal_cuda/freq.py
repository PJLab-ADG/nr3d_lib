"""
Borrowed from https://github.com/ashawkey/torch-ngp
"""

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.cuda.amp import custom_bwd, custom_fwd 

import nr3d_lib_bindings._freqencoder as _backend

class _freq_encoder(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32) # force float32 for better precision
    def forward(ctx, inputs, n_frequencies, output_dim):
        # inputs: [B, input_dim], float 
        # RETURN: [B, F], float

        if not inputs.is_cuda: inputs = inputs.cuda()
        inputs = inputs.contiguous()

        B, input_dim = inputs.shape # batch size, coord dim
        
        outputs = torch.empty(B, output_dim, dtype=inputs.dtype, device=inputs.device)

        _backend.freq_encode_forward(inputs, B, input_dim, n_frequencies, output_dim, outputs)

        if ctx.needs_input_grad[0]:  # inputs
            ctx.save_for_backward(inputs, outputs)
            ctx.dims = [B, input_dim, n_frequencies, output_dim]

        return outputs
    
    @staticmethod
    @once_differentiable # NOTE: Important !!! Not 2nd-backwardable !!!
    @custom_bwd
    def backward(ctx, grad):
        # grad: [B, C * C]

        grad = grad.contiguous()
        inputs, outputs = ctx.saved_tensors
        B, input_dim, n_frequencies, output_dim = ctx.dims

        grad_inputs = torch.zeros_like(inputs)
        _backend.freq_encode_backward(grad, outputs, B, input_dim, n_frequencies, output_dim, grad_inputs)

        return grad_inputs, None, None
    
def freq_encode(input: torch.Tensor, n_frequencies: int, output_dim: int=None) -> torch.Tensor:
    input_dim = input.shape[-1]
    if output_dim is None: output_dim = input_dim + input_dim * 2 * n_frequencies
    return _freq_encoder.apply(input, n_frequencies, output_dim)

class FreqEncoder(nn.Module):
    def __init__(self, input_dim=3, n_frequencies=4, include_input=True):
        super().__init__()

        assert include_input, "Currently sinusoidal embedder only support `include_input`==True."

        self.in_features = input_dim
        self.n_frequencies = n_frequencies
        self.out_features = input_dim + input_dim * 2 * n_frequencies
        
    def __repr__(self):
        return f"FreqEncoder: input_dim={self.in_features}, output_dim={self.out_features}, n_frequencies={self.n_frequencies} "
    
    def forward(self, inputs, **kwargs) -> torch.Tensor:
        # inputs: [..., input_dim]
        # return: [..., ]
        prefix = inputs.shape[:-1]
        inputs = inputs.flatten(0,-2)

        outputs = freq_encode(inputs, self.n_frequencies, self.out_features).unflatten(0, prefix)

        return outputs

if __name__ == "__main__":
    def test():
        from icecream import ic
        m = FreqEncoder(3, 10)
        ic(m)
        x = torch.randn([7,3]).cuda().requires_grad_(True)
        y = m(x)
        ic(y)
        y.mean().backward()
        ic(x.grad)
    test()
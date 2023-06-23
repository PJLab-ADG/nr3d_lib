"""
Borrowed from https://github.com/ashawkey/torch-ngp
"""

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.cuda.amp import custom_bwd, custom_fwd 

import nr3d_lib_bindings._shencoder as _backend

class _sh_encoder(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32) # force float32 for better precision
    def forward(ctx, inputs, degree, calc_grad_inputs=False):
        # inputs: [B, input_dim], float in [-1, 1]
        # RETURN: [B, F], float

        inputs = inputs.contiguous()
        B, input_dim = inputs.shape # batch size, coord dim
        out_features = degree ** 2
        
        outputs = torch.empty(B, out_features, dtype=inputs.dtype, device=inputs.device)

        if calc_grad_inputs:
            dy_dx = torch.empty(B, input_dim * out_features, dtype=inputs.dtype, device=inputs.device)
        else:
            dy_dx = torch.empty(1, dtype=inputs.dtype, device=inputs.device)

        _backend.sh_encode_forward(inputs, outputs, B, input_dim, degree, calc_grad_inputs, dy_dx)

        if calc_grad_inputs: # inputs
            ctx.save_for_backward(inputs, dy_dx)
            ctx.dims = [B, input_dim, degree]
            ctx.calc_grad_inputs = calc_grad_inputs

        return outputs
    
    @staticmethod
    #@once_differentiable
    @custom_bwd
    def backward(ctx, grad):
        # grad: [B, C * C]

        if ctx.calc_grad_inputs:
            grad = grad.contiguous()
            inputs, dy_dx = ctx.saved_tensors
            B, input_dim, degree = ctx.dims
            grad_inputs = torch.zeros_like(inputs)
            _backend.sh_encode_backward(grad, inputs, B, input_dim, degree, dy_dx, grad_inputs)
            return grad_inputs, None, None
        else:
            return None, None, None

def sh_encode(input: torch.Tensor, degree: int, calc_grad_inputs=False) -> torch.Tensor:
    return _sh_encoder.apply(input, degree, calc_grad_inputs)

class SHEncoder(nn.Module):
    def __init__(self, input_dim=3, degree=4):
        super().__init__()
        self.degree = degree # 0 ~ 4
        self.in_features = input_dim # coord dims, must be 3
        self.out_features = degree ** 2
        assert self.in_features == 3, "SH encoder only support input dim == 3"
        assert self.degree > 0 and self.degree <= 8, "SH encoder only supports degree in [1, 8]"
        
    def __repr__(self):
        return f"SHEncoder: input_dim={self.in_features}, output_dim={self.out_features}, degree={self.degree}"
    
    def forward(self, inputs: torch.Tensor, size=1) -> torch.Tensor:
        # inputs: [..., input_dim], normalized real world positions in [-size, size]
        # return: [..., degree^2]

        prefix = inputs.shape[:-1]
        inputs = (inputs / size).flatten(0, -2) # [-1, 1]

        outputs = sh_encode(inputs, self.degree, inputs.requires_grad).unflatten(0, prefix)
        return outputs

if __name__ == "__main__":
    def test():
        from icecream import ic
        m = SHEncoder()
        ic(m)
        x = torch.randn([7,3]).cuda().requires_grad_(True)
        y = m(x)
        ic(y)
        y.mean().backward()
        ic(x.grad)
    test()
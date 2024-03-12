"""
@file   modulations.py
@author Xinyang Li, Shanghai AI Lab
@brief  Common and generic modulation layers
"""

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from nr3d_lib.utils import torch_dtype
from nr3d_lib.models.layers import DenseLayer

#----------------------------------------------------------------------------

class ConstLayer(nn.Module):
    def __init__(self,
        out_channels,                               # Number of output channels.
        const_size
    ):
        super().__init__()

        self.const = nn.Parameter(torch.randn(out_channels, *const_size))

    def forward(self):
        return self.const.unsqueeze(0)

#----------------------------------------------------------------------------

class ModulatedLayer(nn.Module):
    def __init__(self,
        in_channels,                            # Number of input channels.
        out_channels,                           # Number of output channels.
        w_dim,                                  # Intermediate latent (W) dimensionality.
        kernel_type     = [0, 2, 3][2],         # 0 for linear, 2 for conv2d, 3 for conv3d
        kernel_size     = 3,                    # Convolution kernel size. Only for kernel_type = 2 or 3
        activation      = nn.LeakyReLU(0.2),    # Activation function: 'relu', 'lrelu', etc.
        gain            = np.sqrt(2 / (1 + 0.2 ** 2)),
        fmm_rank        = None,                 # None for style modulation, interger for FMM.
        demodulation    = True,
        equal_lr        = False, 
        device          = None, 
        dtype           = torch.float, 
    ):
        super().__init__()
        self.dtype = torch_dtype(dtype)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim

        assert kernel_type in [0, 2, 3]
        self.kernel_type = kernel_type
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.gain = gain
        self.fmm_rank = fmm_rank
        self.demodulation = demodulation
        
        if fmm_rank is None:
            self.affine = DenseLayer(w_dim, in_channels, equal_lr=equal_lr, bias_init=1, device=device, dtype=self.dtype)
        else:
            if in_channels * fmm_rank + fmm_rank * out_channels <  in_channels * out_channels:
                self.affine = None
                self.affine_o = DenseLayer(w_dim, out_channels * fmm_rank, equal_lr=equal_lr, bias_init=0, device=device, dtype=self.dtype)
                self.affine_i = DenseLayer(w_dim, fmm_rank * in_channels, equal_lr=equal_lr, bias_init=0, device=device, dtype=self.dtype)
            else:
                self.affine = DenseLayer(w_dim, out_channels * in_channels, equal_lr=equal_lr, bias_init=1, device=device, dtype=self.dtype)
        
        # NOTE: Alwarys register parameter in float !
        self.weight = torch.nn.Parameter(torch.empty([out_channels, in_channels, *[kernel_size for _ in range(kernel_type)]], device=device, dtype=torch.float))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels], device=device, dtype=torch.float))
        if equal_lr:
            init.uniform_(self.weight, -1., 1.)
            self.weight_gain = 1 / np.sqrt(self.in_channels * (self.kernel_size ** self.kernel_type))
        else:
            init.kaiming_uniform_(self.weight, a=math.sqrt(5.0))
            self.weight_gain = 1

        # self.up = up
        self.activation = activation

    @property
    def device(self) -> torch.device:
        return self.weight.device

    def forward(self, x, w):
        """
        For conv2d and 3d, x.shape = B C (D) H W
        For linear, x.shape = B, ..., C
        """
        shape = x.shape
        assert w.shape[0] == shape[0]

        if self.fmm_rank is None:
            styles = self.affine(w).unsqueeze(1)
        else:
            if self.affine is None:
                styles_o = self.affine_o(w).view(shape[0], self.out_channels, self.fmm_rank)
                styles_i = self.affine_i(w).view(shape[0], self.fmm_rank, self.in_channels)
                styles = torch.bmm(styles_o, styles_i).mul_(1 / np.sqrt(self.fmm_rank)).add_(1)
            else:
                styles = self.affine(w).view(shape[0], self.out_channels, self.in_channels)
        
        w = self.weight.unsqueeze(0) # [NOIk*]
        w = w * styles.reshape(*styles.shape, *[1 for _ in range(self.kernel_type)]) # [NOIk*]

        if self.demodulation:
            dcoefs = (w.square().sum(dim=[2, *[3 + i for i in range(self.kernel_type)]], keepdim=True) + 1e-8).rsqrt()
            w = w * dcoefs
        else:
            w = w * self.weight_gain

        w = w.mul_(self.gain)

        if self.kernel_type == 0:
            x = x.reshape(shape[0], -1, shape[-1])
            x = torch.bmm(x, w.to(x.dtype).transpose(1, 2))
            x = x.add_(self.bias.unsqueeze(0).unsqueeze(0).to(x.dtype))
            x = x.reshape(*shape[:-1], -1)
        else:
            x = x.reshape(1, -1, *shape[2:])
            w = w.reshape(-1, self.in_channels, *self.weight.shape[2:])
            if self.kernel_type == 2:
                x = F.conv2d(x, w.to(x.dtype), padding=self.padding, groups=shape[0])
            elif self.kernel_type == 3:
                x = F.conv3d(x, w.to(x.dtype), padding=self.padding, groups=shape[0])
            else:
                raise NotImplementedError
            x = x.reshape(shape[0], -1, *x.shape[2:])
            x = x.add_(self.bias.reshape(1, self.out_channels, *[1 for _ in range(self.kernel_type)]).to(x.dtype))

        x = self.activation(x)

        return x

    def extra_repr(self) -> str:
        return f"in_channels={self.in_channels}, out_channels={self.out_channels}, w_dim={self.w_dim}, kernel_type={self.kernel_type}"

#----------------------------------------------------------------------------

class ModulatedBlock(nn.Module):
    def __init__(self,
        in_channels,                            # Number of input channels, 0 = first block.
        out_channels,                           # Number of output channels.
        w_dim,                                  # Intermediate latent (W) dimensionality.
        const_mode      = 'const',
        up              = nn.Upsample(scale_factor=2, mode='trilinear'),
        kernel_type     = [0, 2, 3][2],         # 0 for linear, 2 for conv2d, 3 for conv3d
        kernel_size     = 3,                    # Convolution kernel size. Only for kernel_type = 2 or 3
        activation      = nn.LeakyReLU(0.2),    # Activation function: 'relu', 'lrelu', etc.
        gain            = np.sqrt(2 / (1 + 0.2 ** 2)),
        fmm_rank        = None,                 # None for style modulation, interger for FMM.
        device          = None, 
        dtype           = torch.float, 
    ):
        super().__init__()
        
        self.dtype = torch_dtype(dtype)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.kernel_type = kernel_type

        if in_channels == 0:
            if const_mode == 'const':
                self.const = ConstLayer(out_channels, (4 for _ in range(kernel_type)))
            else:
                raise NotImplementedError()
        else:
            self.up = up
            self.conv0 = ModulatedLayer(
                in_channels, out_channels, w_dim=w_dim, 
                kernel_type=kernel_type, kernel_size=kernel_size, 
                activation=activation, gain=gain, fmm_rank=fmm_rank, 
                device=device, dtype=dtype
            )

        self.conv1 = ModulatedLayer(
            out_channels, out_channels, w_dim=w_dim, 
            kernel_type=kernel_type, kernel_size=kernel_size, 
            activation=activation, gain=gain, fmm_rank=fmm_rank, 
            device=device, dtype=dtype
        )

    @property
    def device(self) -> torch.device:
        return self.conv1.device

    def forward(self, x, w):

        # Input.
        if self.in_channels == 0:
            x = self.const()
            x = x.repeat([w.shape[0], *[1 for _ in x.shape[1:]]])
        else:
            x = self.up(x)
            x = self.conv0(x, w)

        x = self.conv1(x, w)

        return x
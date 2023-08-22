"""
@file   sdf.py
@author Jianfei Guo, Shanghai AI Lab
@brief  MLP-based vanilla SDF.
"""

__all__ = [
    'MlpPESDF'
]

import numpy as np
from typing import Literal

import torch
import torch.nn as nn
from torch import autograd

from nr3d_lib.utils import torch_dtype
from nr3d_lib.config import ConfigDict

from nr3d_lib.models.base import ModelMixin
from nr3d_lib.models.spatial import AABBSpace
from nr3d_lib.models.embedders import get_embedder
from nr3d_lib.models.layers import DenseLayer, get_nonlinearity
from nr3d_lib.models.fields.sdf.utils import pretrain_sdf_sphere

class MlpPESDF(ModelMixin, nn.Module):
    def __init__(
        self, 
        D=8, W=256, Ws=None, skips=[4], input_ch=3, n_rgb_used_output=0, n_frequencies=6, 
        activation={'type': 'softplus', 'beta':100.}, output_activation=None,
        radius_init=0.6, inside_out=False, 
        geo_init_method: Literal['geometric', 'pretrain', 'pretrain_after_geometric']='geometric',
        bounding_size=2.0, # Network's boundary size; coordinates are within [-bounding_size/2, bounding_size/2]
        aabb=None, # Network's boundary; coordinates are within [aabb[0], aabb[1]]
        weight_norm=False, use_tcnn_backend=False, dtype=torch.float, device=torch.device('cuda'), 
        ):
        """
        n_rgb_used_output: 
            1. sdf decoder output width = 1 + n_rgb_used_output
            2. the extra_feat that the radiance net used:
                set to == 0: comes from encoding()
                set to  > 0: comes from output[.., 1:]
        """
        super().__init__()
        self.dtype = torch_dtype(dtype)
        self.device = device

        if Ws is None:
            Ws = [W]*D
        else:
            D = len(Ws)

        nl, gain, init_fn, first_init_fn = get_nonlinearity(activation)
        last_nl, last_gain, last_init_fn, _ = get_nonlinearity(output_activation)

        self.radius_init = radius_init
        self.inside_out = inside_out
        self.geo_init_method = geo_init_method
        self.is_extrafeat_from_output = n_rgb_used_output > 0
        if n_frequencies > 0:
            # NOTE: For now, the CUDA sinusoidal does not support 2nd backward ! Use sinusoidal_legacy instead.
            self.embed_fn, input_ch = get_embedder({'type': 'sinusoidal_legacy', 'n_frequencies': n_frequencies}, 3)
        else:
            self.embed_fn, input_ch = nn.Identity(), 3
        assert radius_init < bounding_size/2., "Half bounding size should be large then init radius"

        self.D = D
        self.skips = skips
        self.n_rgb_used_extrafeat = n_rgb_used_output if self.is_extrafeat_from_output else Ws[-1]

        surface_fc_layers = []
        # NOTE: as in IDR/NeuS, the network's has D+1 layers
        for l in range(D+1):
            # decide out_dim
            if l == D:
                out_dim = 1 + n_rgb_used_output
            elif (l+1) in self.skips:
                out_dim = Ws[l] - input_ch  # recude output dim before the skips layers, as in IDR / NeuS
            else:
                out_dim = Ws[l]
                
            # decide in_dim
            if l == 0:
                in_dim = input_ch
            else:
                in_dim = Ws[l-1]
            
            if l != D:
                layer = DenseLayer(in_dim, out_dim, activation=nl, dtype=self.dtype, device=self.device)
                layer.apply(first_init_fn if (l==0) else init_fn)
            else:
                layer = DenseLayer(in_dim, out_dim, activation=last_nl, dtype=self.dtype, device=self.device)
                layer.apply(last_init_fn)

            # if true preform preform geometric initialization
            if geo_init_method == 'geometric' or geo_init_method == 'pretrain_after_geometric':
                #--------------
                # sphere init, as in SAL / IDR.
                #--------------
                if l == D:
                    nn.init.normal_(layer.weight, mean=np.sqrt(np.pi) / np.sqrt(in_dim), std=0.0001)
                    if not self.inside_out:
                        nn.init.constant_(layer.bias, -1 * radius_init)
                    else:
                        nn.init.constant_(layer.bias, radius_init)
                elif n_frequencies > 0 and l == 0:
                    torch.nn.init.constant_(layer.bias, 0.0)
                    torch.nn.init.constant_(layer.weight[:, 3:], 0.0)   # let the initial weights for octaves to be 0.
                    torch.nn.init.normal_(layer.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif n_frequencies > 0 and l in self.skips:
                    torch.nn.init.constant_(layer.bias, 0.0)
                    torch.nn.init.normal_(layer.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(layer.weight[:, -(input_ch - 3):], 0.0) # NOTE: this contrains the concat order to be  [h, x_embed]
                else:
                    nn.init.constant_(layer.bias, 0.0)
                    nn.init.normal_(layer.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                layer = nn.utils.weight_norm(layer)

            surface_fc_layers.append(layer)

        self.surface_fc_layers = nn.ModuleList(surface_fc_layers)
        
        self.register_buffer('is_pretrained', torch.tensor([False], dtype=torch.bool), persistent=True)
        
        # Valid representing space
        self.space = AABBSpace(bounding_size=bounding_size, aabb=aabb, device=device)

    def initialize(self, config=ConfigDict(), logger=None, log_prefix=None):
        geo_init_method = self.geo_init_method
        if self.geo_init_method == 'pretrain' or ('pretrain_after' in geo_init_method):
            if not self.is_pretrained:
                pretrain_sdf_sphere(self, target_radius=self.radius_init, logger=logger, log_prefix=log_prefix, **config)
                self.is_pretrained = ~self.is_pretrained
                return True
        return False

    def forward(self, x: torch.Tensor, input_normalized=True, return_h=True):
        if not input_normalized:
            x = self.space.normalize_coords(x)
        ret = dict()
        x = self.embed_fn(x)
        
        h = x
        for i in range(self.D):
            if i in self.skips:
                # NOTE: concat order can not change! Special operations are taken in intialization.
                h = torch.cat([h, x], dim=-1) / np.sqrt(2)
            h = self.surface_fc_layers[i](h)
        
        out = self.surface_fc_layers[-1](h)
        
        if self.is_extrafeat_from_output > 0:
            if return_h:
                ret['h'] = out[..., 1:]
            ret['sdf'] = out[..., 0]
        else:
            ret['sdf'] = out.squeeze(-1)
            if return_h:
                ret['h'] = h
        return ret
    
    def forward_sdf(self, x: torch.Tensor, input_normalized=True):
        return self.forward(x, input_normalized=input_normalized, return_h=False)
    
    def forward_sdf_nablas(self,  x: torch.Tensor, has_grad:bool = None, nablas_has_grad:bool=None, input_normalized=True):
        if not input_normalized:
            x = self.space.normalize_coords(x)
        ret = dict()
        has_grad = torch.is_grad_enabled() if has_grad is None else has_grad
        nablas_has_grad = has_grad if nablas_has_grad is None else (nablas_has_grad and has_grad)
        x = x.requires_grad_(True)
        # Force enabling grad for normal calculation
        with torch.enable_grad():
            ret.update(self.forward(x))
            ret['nablas'] = autograd.grad(
                ret['sdf'],
                x,
                torch.ones_like(ret['sdf'], device=x.device),
                create_graph=nablas_has_grad,
                retain_graph=has_grad,
                only_inputs=True)[0]
        if not nablas_has_grad:
            ret['nablas'] = ret['nablas'].detach()
        if not has_grad:
            ret['sdf'] = ret['sdf'].detach()
            ret['h'] = ret['h'].detach()
        return ret

    def forward_in_obj(self, x: torch.Tensor, invalid_sdf: float = 1.1, with_normal=False, return_h=False):
        """
        Forward using coords from object's coordinates
        """
        net_x = self.space.normalize_coords(x)
        # NOTE: 1e-6 is LoTD's clamp eps. TODO: Make it correspondent.
        valid_i = ((net_x > -1+1e-6) & (net_x < 1-1e-6)).all(dim=-1).nonzero(as_tuple=True)
        
        if valid_i[0].numel() > 0:
            if with_normal:
                raw_ret = self.forward_sdf_nablas(net_x[valid_i], input_normalized=True, return_h=return_h)
            else:
                raw_ret = self.forward(net_x[valid_i], input_normalized=True, return_h=return_h)
        
        sdf = torch.full([*x.shape[:-1]], invalid_sdf, dtype=self.dtype, device=self.device)
        if valid_i[0].numel() > 0:
            sdf = sdf.to(raw_ret['sdf'])
            sdf.index_put_(valid_i, raw_ret['sdf'])
        ret = dict(sdf=sdf)
        
        if with_normal:
            nablas = torch.full([*x.shape[:-1], 3], 0., dtype=torch.float, device=self.device)
            if valid_i[0].numel() > 0:
                nablas = nablas.to(raw_ret['nablas'])
                nablas.index_put_(valid_i, raw_ret['nablas'])
            ret.update(nablas=nablas)

        if return_h:
            h = torch.full([*x.shape[:-1], self.n_rgb_used_extrafeat], dtype=self.dtype, device=self.device)
            if valid_i[0].numel() > 0:
                h = h.to(raw_ret['h'])
                h.index_put_(valid_i, raw_ret['h'])
            ret.update(h=h)

        return ret

    def get_weight_reg(self, norm_type: float = 2.0):
        return torch.stack([p.norm(p=norm_type) for n, p in self.surface_fc_layers.named_parameters()])

if __name__ == "__main__":
    def unit_test(dtype=torch.float16, device=torch.device('cuda')):
        from icecream import ic
        m1 = MlpPESDF(
            input_ch=3, D=4, W=64, skips=[], 
            activation={'type': 'softplus', 'beta': 100.}, output_activation=None,
            radius_init=0.5, bounding_size=2.0, 
            weight_norm=False, dtype=dtype).to(device=device)
        ic(m1)
        x = torch.ones([1,3], dtype=dtype, device=device)
        y = m1(x)
        ic(y)
    unit_test()
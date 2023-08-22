"""
@file   lotd_batched.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Basic batched LoTD encoding module.
"""

__all__ = [
    'LoTDBatched', 
    'LoTDBatchedParamWrapper'
]

import re
from math import prod
from typing import Dict

import torch
import torch.nn as nn

from nr3d_lib.config import ConfigDict
from nr3d_lib.utils import import_str, tensor_statistics, torch_dtype

from nr3d_lib.models.spatial import BatchedBlockSpace
from nr3d_lib.models.grids.lotd.lotd import LoTD
from nr3d_lib.models.grids.lotd.lotd_batched_growers import LoTDGrower
from nr3d_lib.models.grids.lotd.lotd_helpers import LoTDAnnealer, level_param_index_shape

class LoTDBatched(nn.Module):
    def __init__(
        self, 
        input_ch, *, 
        bounding_size=2.0, # Network's boundary size; coordinates are within [-bounding_size/2, bounding_size/2]
        grower_cfg: ConfigDict = ...,
        anneal_cfg: ConfigDict = None,
        device='cuda', dtype=torch.float
        ) -> None:
        super().__init__()

        self.device = device
        self.dtype = torch_dtype(dtype)

        #------- Valid representing space
        self.space = BatchedBlockSpace(bounding_size=bounding_size, device=device)

        #------- LoTD param's conditional grower
        grower_cfg.param['world_dim'] = input_ch
        self.lotd_grower: LoTDGrower = import_str(grower_cfg.target)(**grower_cfg.param, device=self.device, dtype=self.dtype)
        
        #------- LoTD Metadata and interface
        self.lotd = LoTD(input_ch, self.lotd_grower.level_res, self.lotd_grower.level_n_feats, self.lotd_grower.lod_types, dtype=self.dtype, device=self.device)
        
        self.z_dim = self.lotd_grower.z_dim
        self.in_features = input_ch
        self.out_features = self.lotd.out_features
        
        #------- LoTD Encoding anneal
        if anneal_cfg is not None:
            self.annealer = LoTDAnnealer(self.lotd.level_n_feats, **anneal_cfg, dtype=self.dtype, device=self.device)
        else:
            self.annealer = None
        # (Optional) float tensor that soft masks encoding's feature.
        self.window: torch.FloatTensor = None 
        # Discard levels that [lvl > max_level], where lvl is in range [0, lotd.n_levels-1]
        # Set max_level=-1 to discard all levels; set to lotd.n_levels-1 to remain all levels
        self.max_level: int = None 

    @property
    def lod_meta(self):
        return self.lotd.meta

    def set_anneal_iter(self, cur_it: int):
        if self.annealer is not None:
            self.max_level, self.window = self.annealer(cur_it)

    def grow(self, z: torch.Tensor, max_level: int = None):
        if z.dim() == 1:
            z = z.unsqueeze(0)
        assert z.dim() == 2, "latent input must be 2D"
        z = z.to(device=self.device, dtype=torch.float) # Should use float32 z input.
        self.B = z.shape[0]
        self.z = z
        self.lod_params = self.lotd_grower(z, max_level=(max_level or self.max_level)).flatten().to(self.dtype) # Convert to float16 param if required.

    def clear(self):
        if hasattr(self, 'lod_params'):
            del self.lod_params
        if hasattr(self, 'z'):
            del self.z

    def forward(self, input: torch.Tensor, batch_inds: torch.Tensor=None, max_level: int=None):
        # NOTE: input must be in range [-1,1]
        if batch_inds is None:
            # Batched forward
            assert input.shape[0] == self.B, f"input should have a batch size of {self.B}"
            return self.lotd(input / 2. + 0.5, self.lod_params, input_batched=True, max_level=(max_level or self.max_level))
        else:
            # Unbatched forward
            assert [*batch_inds.shape] == [*input.shape[:-1]], "batch_inds and input does not match."
            return self.lotd(input / 2. + 0.5, self.lod_params, batch_inds, input_batched=False, max_level=(max_level or self.max_level))
    
    def forward_dydx(self, input: torch.Tensor, batch_inds: torch.Tensor=None, max_level: int=None, need_loss_backward_input=False):
        # NOTE: input must be in range [-1,1]
        if batch_inds is None:
            # Batched forward
            assert input.shape[0] == self.B, f"input should have a batch size of {self.B}"
            return self.lotd.forward_dydx(input / 2. + 0.5, self.lod_params, input_batched=True, 
                                          max_level=(max_level or self.max_level), need_loss_backward_input=need_loss_backward_input)
        else:
            # Unbatched forward
            assert [*batch_inds.shape] == [*input.shape[:-1]], "batch_inds and input does not match."
            return self.lotd.forward_dydx(input / 2. + 0.5, self.lod_params, batch_inds, input_batched=False, 
                                          max_level=(max_level or self.max_level), need_loss_backward_input=need_loss_backward_input)
    
    def backward_dydx(self, dL_dy: torch.Tensor, dy_dx: torch.Tensor, input: torch.Tensor, batch_inds: torch.Tensor=None, max_level: int=None):
        # NOTE: input must be in range [-1,1]
        #       multiply with 0.5 since input is divided with 2 when forward.
        if batch_inds is None:
            # Batched forward
            assert input.shape[0] == self.B, f"input should have a batch size of {self.B}"
            return .5 * self.lotd.backward_dydx(dL_dy, dy_dx, input / 2. + 0.5, self.lod_params, input_batched=True, 
                                                max_level=(max_level or self.max_level))
        else:
            # Unbatched forward
            assert [*batch_inds.shape] == [*input.shape[:-1]], "batch_inds and input does not match."
            return .5 * self.lotd.backward_dydx(dL_dy, dy_dx, input / 2. + 0.5, self.lod_params, batch_inds, input_batched=False, 
                                                max_level=(max_level or self.max_level))

    @torch.no_grad()
    def stat_param(self, with_grad=False, prefix: str='') -> Dict[str, float]:
        prefix_ = prefix + ('.' if prefix and not prefix.endswith('.') else '')
        ret = {}
        ret.update({prefix_ + f"{n}.{k}": v for n, p in self.lotd_grower.named_parameters() for k, v in tensor_statistics(p.data).items()})
        if with_grad:
            ret.update({prefix_ + f"grad.{n}.{k}": v for n, p in self.lotd_grower.named_parameters() if p.grad is not None for k, v in  tensor_statistics(p.grad.data).items() })
        return ret

class LoTDBatchedParamWrapper(object):
    def __init__(
        self, 
        batch_size: int, lod_meta, params: torch.Tensor = None, 
        device=torch.device('cuda'), dtype=torch.float) -> None:
        
        self.B = batch_size
        self.lod_meta = lod_meta
        
        if params is None:
            self.batched_flattened_params = torch.zeros([batch_size, lod_meta.n_params], dtype=dtype, device=device)
        else:
            self.batched_flattened_params = params

        """
        NOTE: Bind lod param getter/setter; possible APIs (getter & setter):
              - lod0, lod1, ..., lodn
              - lod0_vec, lod1_mat, ...
              - lod0_vec0, lod1_mat2, ...
        """
        self.pattern = re.compile(r"^lod(?P<level>[0-9]+)(_(?P<op>[a-z]+)(?P<dim>[0-9]+){0,1}){0,1}")

    def get_level_param(self, l: int, op: str = None, dim: int = None, grad=False):
        index, shape = level_param_index_shape(self.lod_meta, l, op, dim)
        return (self.batched_flattened_params if not grad else self.batched_flattened_params.grad)[index].view(self.B, *shape)

    def set_level_param(self, l: int, op: str = None, dim: int = None, value: torch.Tensor = ...):
        index, shape = level_param_index_shape(self.lod_meta, l, op, dim)
        self.batched_flattened_params[(slice(None), *index)] = value.contiguous().view(self.B, prod(shape))

    def __getattr__(self, name: str) -> torch.Tensor:
        if ('lod' in name) and (hasattr(self, 'pattern')) and (t:=self.pattern.match(name)):
            # l = int(t.group(1))
            ret = t.groupdict()
            l, op, dim = int(ret['level']), ret['op'], ret['dim']
            if dim is None or dim == '':
                dim = None
            else:
                dim = int(dim)
            if op is None or op == '':
                op = None
            return self.get_level_param(l, op, dim)
        else:
            return super().__getattr__(name)
    
    def __setattr__(self, name: str, value: torch.Tensor) -> None:
        if ('lod' in name) and (hasattr(self, 'pattern')) and (t:=self.pattern.match(name)):
            # l = int(t.group(1))
            ret = t.groupdict()
            l, op, dim = int(ret['level']), ret['op'], ret['dim']
            if dim is None or dim == '':
                dim = None
            else:
                dim = int(dim)
            if op is None or op == '':
                op = None
            self.set_level_param(l, op, dim, value)
        else:
            return super().__setattr__(name, value)
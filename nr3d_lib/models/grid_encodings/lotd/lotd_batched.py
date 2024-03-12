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
from typing import Dict, Optional

import torch
import torch.nn as nn

from nr3d_lib.config import ConfigDict
from nr3d_lib.utils import import_str, tensor_statistics, torch_dtype

from nr3d_lib.models.spatial import BatchedBlockSpace
from nr3d_lib.models.grid_encodings.multires_annealer import MultiresAnnealer
from nr3d_lib.models.grid_encodings.lotd.lotd import LoTD
from nr3d_lib.models.grid_encodings.lotd.lotd_batched_growers import LoTDGrower
from nr3d_lib.models.grid_encodings.lotd.lotd_helpers import level_param_index_shape

class LoTDBatched(nn.Module):
    def __init__(
        self, 
        input_ch, *, 
        space: nn.Module = None,
        space_cfg: dict = None, 
        grower_cfg: dict = ...,
        anneal_cfg: dict = None,
        device=None, dtype=torch.float
        ) -> None:
        super().__init__()

        self.dtype = torch_dtype(dtype)

        #------- Valid representing space
        if space is not None:
            # Directly use the externally provided space object definition
            self.space = space
        elif space_cfg is not None:
            space_cfg = space_cfg.copy()
            space_type = space_cfg.pop('type').lower()
            if space_type == 'batched': # Batched AABB
                space = BatchedBlockSpace(**space_cfg)
            elif space_type == 'unbounded' or space_type == 'none':
                space = None
            else:
                raise RuntimeError(f"Invalid space_type={space_type}")
            self.space = space
        else:
            # Do not need input space definition and conversion
            self.space = None

        #------- LoTD param's conditional grower
        grower_cfg['param']['world_dim'] = input_ch
        self.lotd_grower: LoTDGrower = import_str(grower_cfg['target'])(**grower_cfg['param'], device=device, dtype=self.dtype)
        
        #------- LoTD Metadata and interface
        self.lotd = LoTD(input_ch, self.lotd_grower.level_res, self.lotd_grower.level_n_feats, self.lotd_grower.lod_types, dtype=self.dtype, device=device)
        
        self.z_dim = self.lotd_grower.z_dim
        self.in_features = input_ch
        self.out_features = self.lotd.out_features
        
        #------- LoTD Encoding anneal
        if anneal_cfg is not None:
            self.annealer = MultiresAnnealer(self.lotd.level_n_feats, **anneal_cfg, dtype=self.dtype, device=device)
        else:
            self.annealer = None
        # (Optional) float tensor that soft masks encoding's feature.
        self.window: torch.FloatTensor = None 
        # Discard levels that [lvl > max_level], where lvl is in range [0, lotd.n_levels-1]
        # Set max_level=-1 to discard all levels; set to lotd.n_levels-1 to remain all levels
        self.max_level: int = None 

    @property
    def device(self) -> torch.device:
        return self.lotd_grower.device

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
        self.z_per_batch = z
        self.lod_params = self.lotd_grower(z, max_level=(max_level or self.max_level)).flatten().to(self.dtype) # Convert to float16 param if required.

    def clear(self):
        if hasattr(self, 'lod_params'):
            del self.lod_params
        if hasattr(self, 'z_per_batch'):
            del self.z_per_batch

    def forward(self, input: torch.Tensor, bidx: torch.Tensor=None, max_level: int=None):
        # NOTE: input must be in range [-1,1]
        if bidx is None:
            # Batched forward
            assert input.shape[0] == self.B, f"input should have a batch size of {self.B}"
            return self.lotd(input / 2. + 0.5, self.lod_params, input_batched=True, max_level=(max_level or self.max_level))
        else:
            # Unbatched forward
            assert [*bidx.shape] == [*input.shape[:-1]], "bidx and input does not match."
            return self.lotd(input / 2. + 0.5, self.lod_params, bidx, input_batched=False, max_level=(max_level or self.max_level))
    
    def forward_dydx(self, input: torch.Tensor, bidx: torch.Tensor=None, max_level: int=None, need_dL_dinput: Optional[bool]=None):
        # NOTE: input must be in range [-1,1]
        if bidx is None:
            # Batched forward
            assert input.shape[0] == self.B, f"input should have a batch size of {self.B}"
            return self.lotd.forward_dydx(input / 2. + 0.5, self.lod_params, input_batched=True, 
                                          max_level=(max_level or self.max_level), need_dL_dinput=need_dL_dinput)
        else:
            # Unbatched forward
            assert [*bidx.shape] == [*input.shape[:-1]], "bidx and input does not match."
            return self.lotd.forward_dydx(input / 2. + 0.5, self.lod_params, bidx, input_batched=False, 
                                          max_level=(max_level or self.max_level), need_dL_dinput=need_dL_dinput)
    
    def backward_dydx(self, dL_dy: torch.Tensor, dy_dx: torch.Tensor, input: torch.Tensor, bidx: torch.Tensor=None, max_level: int=None):
        # NOTE: input must be in range [-1,1]
        if bidx is None:
            # Batched forward
            assert input.shape[0] == self.B, f"input should have a batch size of {self.B}"
            nablas = self.lotd.backward_dydx(
                dL_dy, dy_dx, input / 2. + 0.5, self.lod_params, 
                input_batched=True, max_level=(max_level or self.max_level))
        else:
            # Unbatched forward
            assert [*bidx.shape] == [*input.shape[:-1]], "bidx and input does not match."
            nablas = self.lotd.backward_dydx(
                dL_dy, dy_dx, input / 2. + 0.5, self.lod_params, bidx, 
                input_batched=False, max_level=(max_level or self.max_level))
        return nablas / 2. # Divided by 2 since `input` is also divided by 2 during the forward_dydx process

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
        dtype=torch.float, device=None) -> None:
        
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
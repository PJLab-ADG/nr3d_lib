"""
@file   lotd_forest.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Basic forest LoTD encoding module.
"""

__all__ = [
    'lotd_forest_encoding', 
    'lotd_forest_fwd_dydx', 
    'lotd_forest_bwd_dydx', 
    'LoTDForestEncoding'
]

from math import prod, sqrt
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from nr3d_lib.models.utils import clip_norm_

from nr3d_lib.utils import tensor_statistics, torch_dtype
from nr3d_lib.config import ConfigDict

from nr3d_lib.models.spatial import ForestBlockSpace
from nr3d_lib.models.grid_encodings.multires_annealer import MultiresAnnealer
from nr3d_lib.models.grid_encodings.lotd.lotd_helpers import level_param_index_shape
from nr3d_lib.models.grid_encodings.lotd.lotd import LoTD, LoTDFunction, LoTDFunctionFwdDydx, LoTDFunctionBwdDydx, LoDType

def lotd_forest_encoding(
    metas, input: torch.Tensor, params: torch.Tensor, block_inds: torch.Tensor = None, block_offsets: torch.Tensor = None, 
    input_batched=False, loss_scale: float = None, max_level: int=None
    ) -> torch.Tensor:
    assert isinstance(metas, tuple) and len(metas) == 2, "`metas` should be a tuple of (lod_meta, forest_meta)"
    if input_batched:
        batch_data_size = prod(input.shape[1:-1])
        block_inds = None
    else:
        batch_data_size = 0
    if loss_scale is None: loss_scale = 128.0 if (params.dtype == torch.float16) else 1.
    output = LoTDFunction.apply(metas, input, params, block_inds, block_offsets, batch_data_size, loss_scale, max_level)
    return output

def lotd_forest_fwd_dydx(
    metas, input: torch.Tensor, params: torch.Tensor, block_inds: torch.Tensor = None, block_offsets: torch.Tensor = None, 
    input_batched=False, loss_scale: float = None, max_level: int=None, need_dL_dinput: Optional[bool]=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    if need_dL_dinput is None:
        need_dL_dinput = torch.is_grad_enabled() and input.requires_grad
    assert isinstance(metas, tuple) and len(metas) == 2, "`metas` should be a tuple of (lod_meta, forest_meta)"
    if input_batched:
        batch_data_size = prod(input.shape[1:-1])
        block_inds = None
    else:
        batch_data_size = 0
    if loss_scale is None: loss_scale = 128.0 if (params.dtype == torch.float16) else 1.
    output, dy_dx = LoTDFunctionFwdDydx.apply(metas, input, params, block_inds, block_offsets, batch_data_size, loss_scale, max_level, need_dL_dinput)
    return output, dy_dx

def lotd_forest_bwd_dydx(
    metas, dL_dy: torch.Tensor, dy_dx: torch.Tensor, input: torch.Tensor, params: torch.Tensor, block_inds: torch.Tensor = None, block_offsets: torch.Tensor = None, 
    input_batched=False, loss_scale: float = None, max_level: int=None, grad_guard=None
    ) -> torch.Tensor:
    assert isinstance(metas, tuple) and len(metas) == 2, "`metas` should be a tuple of (lod_meta, forest_meta)"
    if input_batched:
        batch_data_size = prod(input.shape[1:-1])
        block_inds = None
    else:
        batch_data_size = 0
    if loss_scale is None: loss_scale = 128.0 if (params.dtype == torch.float16) else 1.
    dL_dx = LoTDFunctionBwdDydx.apply(metas, dL_dy, input, params, dy_dx, block_inds, block_offsets, batch_data_size, loss_scale, max_level, grad_guard)
    return dL_dx

class LoTDForestEncoding(nn.Module):
    def __init__(
        self,
        input_ch=3, *, 
        lotd_cfg=dict(),
        anneal_cfg: dict = None,
        param_init_cfg={'type': 'uniform_to_type', 'bound': 1.0e-4}, 
        clip_level_grad_ema_factor: float=0,       
        dtype=torch.half, device=None) -> None:
        super().__init__()
        
        self.dtype = torch_dtype(dtype)
        self.loss_scale = 128.0 if self.dtype == torch.float16 else 1.0

        #------- Valid representing space
        self.space = ForestBlockSpace(dtype=torch.float, device=device)

        #------- LoTD Metadata & param register
        self.lotd = LoTD(input_ch, **lotd_cfg, dtype=self.dtype, device=device)
        # self.lod_meta = generate_meta(3, **encoding_cfg)
        # self.lod_meta = self.lotd.meta
        self.in_features = input_ch
        self.out_features = self.lotd.out_features

        self.register_parameter("forest_flattened_params", None)
        self.clip_level_grad_ema_factor = clip_level_grad_ema_factor
        self.param_init_cfg = param_init_cfg

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

        #------- Ema grad storage
        if self.clip_level_grad_ema_factor > 0:
            self.register_buffer("level_grad_norm_ema", torch.full([self.lotd.n_levels], 0.1, dtype=torch.float, device=device))

        self._register_load_state_dict_pre_hook(self.before_load_state_dict)

    def before_load_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        p = state_dict[prefix + 'forest_flattened_params']
        if [*self.forest_flattened_params.shape] != [*p.shape]:
            del self.forest_flattened_params
            self.register_parameter('forest_flattened_params', nn.Parameter(torch.zeros(p.shape, dtype=p.dtype, device=self.device)))

    @property
    def device(self) -> torch.device:
        return self.space.device

    def populate(self, **kwargs):
        self.space.populate(**kwargs)
        # Populate params
        self._populate_params()

    def _populate_params(self):
        #------------------ Init encoding parameters
        n_trees = self.forest_meta.n_trees
        n_params = self.lod_meta.n_params
        
        # NOTE: Parameters should always be stored as float32. `self.dtype` is only respected when forward.
        p = torch.zeros([n_trees, n_params], dtype=torch.float, device=self.device)
        self.register_parameter('forest_flattened_params', nn.Parameter(p, requires_grad=True))
        
        #------------------ Random initilization
        with torch.no_grad():
            bid = slice(None) # Operate on all blocks
            param_init_cfg = self.param_init_cfg
            param_init_method = param_init_cfg['type']

            if param_init_method == 'uniform':
                bound = param_init_cfg['bound']
                self.forest_flattened_params.uniform_(-bound, bound)

            elif param_init_method == 'normal':
                std = param_init_cfg['std']
                self.forest_flattened_params.normal_(0., std)
            
            elif param_init_method == 'uniform_to_type':
                bound = param_init_cfg['bound']
                for l, tp in enumerate(self.lotd.level_types):
                    tp = LoDType(tp)
                    if tp == LoDType.Dense or tp == LoDType.Hash:
                        b = bound
                    elif tp == LoDType.VectorMatrix:
                        b = sqrt(bound)
                    elif tp == LoDType.NPlaneSum:
                        b = bound / 3.
                    elif tp == LoDType.NPlaneMul or tp == LoDType.CP or tp == LoDType.CPfast:
                        b = bound ** (1/3.)
                    else:
                        raise RuntimeError(f"Invalid tp={tp}")
                    self.get_level_param(bid, l).uniform_(-b, b)
            
            elif param_init_method == 'normal_to_type':
                std = param_init_cfg['std']
                for l, tp in enumerate(self.lotd.level_types):
                    tp = LoDType(tp)
                    if tp == LoDType.Dense or tp == LoDType.Hash:
                        s = std
                    elif tp == LoDType.VectorMatrix:
                        s = sqrt(std)
                    elif tp == LoDType.NPlaneSum:
                        s = std
                    elif tp == LoDType.NPlaneMul or tp == LoDType.CP or tp == LoDType.CPfast:
                        s = std ** (1/3.)
                    else:
                        raise RuntimeError(f"Invalid tp={tp}")
                    self.get_level_param(bid, l).normal_(0., s)
            
            else:
                raise RuntimeError(f"Invalid param_init_method={param_init_method}")

    @property
    def lod_meta(self):
        return self.lotd.meta

    @property
    def forest_meta(self):
        return self.space.meta

    @property
    def metas(self):
        return (self.lotd.meta, self.space.meta)

    @property
    def active_forest_params(self):
        # TODO: Use active_inds 
        return self.forest_flattened_params

    def set_anneal_iter(self, cur_it: int):
        if self.annealer is not None:
            self.max_level, self.window = self.annealer(cur_it)

    def abid2bid(self, active_bid: Union[int, torch.LongTensor]):
        raise NotImplementedError

    def forward(self, block_x: torch.Tensor, block_inds: torch.Tensor = None, block_offsets: torch.Tensor = None, max_level: int=None):
        # NOTE: block_x must be in range [-1,1]
        input_batched = block_inds is None
        output = lotd_forest_encoding(self.metas, block_x / 2. + 0.5, self.active_forest_params.flatten().to(self.dtype), block_inds, block_offsets, 
                                      input_batched=input_batched, max_level=(max_level or self.max_level), loss_scale=self.loss_scale)
        return (output * self.window) if self.window is not None else output

    def forward_dydx(self, block_x: torch.Tensor, block_inds: torch.Tensor = None, block_offsets: torch.Tensor = None, max_level: int=None, need_dL_dinput: Optional[bool]=None):
        # NOTE: block_x must be in range [-1,1]
        input_batched = block_inds is None
        output, dy_dx = lotd_forest_fwd_dydx(self.metas, block_x / 2. + 0.5, self.active_forest_params.flatten().to(self.dtype), block_inds, block_offsets, 
                                             input_batched=input_batched, max_level=(max_level or self.max_level), loss_scale=self.loss_scale, 
                                             need_dL_dinput=need_dL_dinput)
        return ((output * self.window) if self.window is not None else output), dy_dx

    def backward_dydx(self, dL_dy: torch.Tensor, dy_dx: torch.Tensor,  block_x: torch.Tensor, block_inds: torch.Tensor = None, block_offsets: torch.Tensor = None, max_level: int=None):
        # NOTE: block_x must be in range [-1,1]
        input_batched = block_inds is None
        nablas = lotd_forest_bwd_dydx(
            self.metas, dL_dy, dy_dx, block_x / 2. + 0.5, self.active_forest_params.flatten().to(self.dtype), block_inds, block_offsets, 
            input_batched=input_batched, max_level=(max_level or self.max_level), loss_scale=self.loss_scale)
        return nablas / 2.

    def get_level_param(self, bid: Union[int, List[int], slice, torch.Tensor], l: int, op: str = None, dim: int = None, grad=False):
        index, shape = level_param_index_shape(self.lod_meta, l, op, dim)
        prefix = self.forest_flattened_params.data[bid, 0].shape
        return (self.forest_flattened_params if not grad else self.forest_flattened_params.grad)[(bid, *index)].view(*prefix, *shape)

    def set_level_param(self, bid: Union[int, List[int], slice, torch.Tensor], l: int, op: str = None, dim: int = None, value: torch.Tensor = None):
        index, shape = level_param_index_shape(self.lod_meta, l, op, dim)
        prefix = self.forest_flattened_params.data[bid, 0].shape
        self.forest_flattened_params[(bid, *index)] = value.contiguous().view(*prefix, prod(shape))
    
    @torch.no_grad()
    def clip_grad_and_update_ema(self, val: float=None):
        if self.clip_level_grad_ema_factor > 0:
            # gnorm = torch.stack([self.get_level_param(l, grad=True).data.abs().max() for l in range(self.lotd.n_levels)])
            gnorm = torch.stack([self.get_level_param(l, grad=True).data.norm() for l in range(self.lotd.n_levels)])
            
            ema = self.level_grad_norm_ema.copy_(gnorm.lerp(self.level_grad_norm_ema, 0.99))

            for lvl in range(self.lotd.n_levels):
                index, shape = level_param_index_shape(self.lod_meta, lvl)
                
                val = self.clip_level_grad_ema_factor * ema[lvl].item()
                
                # self.forest_flattened_params.grad[index].clip_(-val, val)
                clip_norm_(self.forest_flattened_params.grad[index], val)
    
    @torch.no_grad()
    def stat_param(self, with_grad=False, prefix: str='') -> Dict[str, float]:
        prefix_ = prefix + ('.' if prefix and not prefix.endswith('.') else '')
        ret = {}
        ret.update({prefix_ + f'total.{k}': v for k,v in tensor_statistics(self.forest_flattened_params.data).items()})
        if with_grad and self.forest_flattened_params.grad is not None:
            ret.update({prefix_ + f'grad_total.{k}': v for k,v in tensor_statistics(self.forest_flattened_params.grad.data).items()})
        
        for lvl, tp in enumerate(self.lotd.level_types):
            tp = LoDType(tp)
            if tp == LoDType.VectorMatrix:
                ret.update({prefix_ + f'lv.{lvl}.vec.{k}': v for k,v in tensor_statistics(self.get_level_param(lvl, 'vec').data).items()})
                ret.update({prefix_ + f'lv.{lvl}.mat.{k}': v for k,v in tensor_statistics(self.get_level_param(lvl, 'mat').data).items()})
                if with_grad:
                    ret.update({prefix_ + f'grad.lv.{lvl}.vec.{k}': v for k,v in tensor_statistics(self.get_level_param(lvl, 'vec', grad=True).data).items()})
                    ret.update({prefix_ + f'grad.lv.{lvl}.mat.{k}': v for k,v in tensor_statistics(self.get_level_param(lvl, 'mat', grad=True).data).items()})
            else:
                ret.update({prefix_ + f'lv.{lvl}.{k}': v for k,v in tensor_statistics(self.get_level_param(lvl).data).items()})
                if with_grad:
                    ret.update({prefix_ + f'grad.lv.{lvl}.{k}': v for k,v in tensor_statistics(self.get_level_param(lvl, grad=True).data).items()})
        if self.clip_level_grad_ema_factor > 0:
            ret.update({prefix_ + f'grad.lv.{lvl}.ema': self.level_grad_norm_ema[lvl].item() for lvl in range(self.lotd.n_levels)})
        return ret
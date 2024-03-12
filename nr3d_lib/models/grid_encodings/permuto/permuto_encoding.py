"""
@file   permuto_encoding.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Basic single-block permutohedral-lattice-based encoding module.
"""

__all__ = [
    'PermutoEncoding'
]

from copy import deepcopy
from math import prod
from typing import Any, Dict, Literal, Optional, Union

import torch
import torch.nn as nn

from nr3d_lib.fmt import log
from nr3d_lib.config import ConfigDict
from nr3d_lib.utils import tensor_statistics, torch_dtype

from nr3d_lib.models.utils import clip_norm_
from nr3d_lib.models.spatial import AABBSpace, AABBDynamicSpace, BatchedBlockSpace, BatchedDynamicSpace
from nr3d_lib.models.grid_encodings.multires_annealer import MultiresAnnealer
from nr3d_lib.models.grid_encodings.permuto.permuto import PermutoEncImpl, level_param_index_shape, get_permuto_cfg

class PermutoEncoding(nn.Module):
    def __init__(
        self, 
        input_ch: int, *, 
        permuto_cfg: dict = None, 
        permuto_auto_compute_cfg: dict = None, # (Optional) auto-compute LoTD config from aabb
        pos_scale: Union[float, torch.Tensor] = None, 
        space: nn.Module = None,
        space_cfg: dict = None, 
        anneal_cfg: dict = None,
        param_init_cfg={'type': 'uniform', 'bound': 1.0e-4}, 
        clip_level_grad_ema_factor: float=0, 
        dtype=torch.half, device=None, 
        ) -> None:
        super().__init__()
        
        self.dtype = torch_dtype(dtype)
        self.clip_level_grad_ema_factor = clip_level_grad_ema_factor
        self.param_init_cfg = param_init_cfg
        
        #------- Valid representing space
        if space is not None:
            # Directly use the externally provided space object
            self.space = space
        elif space_cfg is not None:
            space_cfg = space_cfg.copy()
            space_type = space_cfg.pop('type').lower()
            if space_type == 'aabb':
                space = AABBSpace(**space_cfg)
            elif space_type == 'aabb_dynamic':
                space = AABBDynamicSpace(**space_cfg)
            elif space_type == 'batched': # Batched AABB
                space = BatchedBlockSpace(**space_cfg)
            elif space_type == 'batched_dynamic':
                space = BatchedDynamicSpace(**space_cfg)
            elif space_type == 'unbounded' or space_type == 'none':
                space = None
            else:
                raise RuntimeError(f"Invalid space_type={space_type}")
            self.space = space
        else:
            # Do not need input space definition and conversion
            self.space = None
        
        #------- Permutohedral enc metadata
        assert bool(permuto_cfg is not None) != bool(permuto_auto_compute_cfg is not None), "Please specify one and only one of `lotd_cfg` and `lotd_auto_compute_cfg`"
        if permuto_auto_compute_cfg is not None:
            permuto_cfg = get_permuto_cfg(
                **permuto_auto_compute_cfg, 
                input_ch=input_ch, 
                stretch=(self.space.radius3d*2).tolist() if self.space is not None else None, 
            )
        else:
            permuto_cfg = deepcopy(permuto_cfg)
        
        if pos_scale is not None:
            permuto_cfg.update(pos_scale=pos_scale)
        self.permuto_cfg = permuto_cfg.to_dict() if not isinstance(permuto_cfg, dict) else permuto_cfg
        self.permuto = PermutoEncImpl(input_ch, **permuto_cfg, dtype=self.dtype, device=device)
        self.in_features: int = input_ch
        self.out_features: int = self.permuto.out_features
        
        #------- Permutohedral lattice values register
        # NOTE: Parameters should always be stored as float32. `self.dtype` is only respected when forward.
        p = torch.zeros(self.permuto.n_params, device=device, dtype=torch.float)
        self.register_parameter("flattened_params", nn.Parameter(p, requires_grad=True))
        
        #------- Param random initialization
        self.init_lattice_values_random()

        #------- Multi-res Encoding anneal
        if anneal_cfg is not None:
            self.annealer = MultiresAnnealer(self.permuto.level_n_feats, **anneal_cfg, dtype=self.dtype, device=device)
        else:
            self.annealer = None
        # (Optional) float tensor that soft masks encoding's feature.
        self.window: torch.FloatTensor = None 
        # Discard levels that [lvl > max_level], where lvl is in range [0, lotd.n_levels-1]
        # Set `max_level=-1` to discard all levels; set `max_level=lotd.n_levels-1` to remain all levels
        self.max_level: int = None 

    @property
    def device(self) -> torch.device:
        return self.flattened_params.device
    
    @property
    def level_n_feats(self):
        return self.permuto.level_n_feats
    
    @property
    def meta(self):
        return self.permuto.meta
    
    @property
    def inference_param(self):
        return self.flattened_params.data.to(self.dtype)
    
    def set_anneal_iter(self, cur_it: int):
        if self.annealer is not None:
            self.max_level, self.window = self.annealer(cur_it)
    
    def forward(self, input: torch.Tensor, max_level: int=None, need_dL_dinput: Optional[bool]=None) -> torch.Tensor:
        """ Forward

        Args:
            input (torch.Tensor): Relative position/coords in range [-1,1]
            max_level (int, optional): Maximum lotd level bypass. 
                Only levels l <= `max_level` will be used if specify. Defaults to None.

        Returns:
            torch.Tensor: [..., out_features] The interpolated features
        """
        if isinstance(self.space, (AABBSpace, AABBDynamicSpace, BatchedBlockSpace, BatchedDynamicSpace)):
            # Maps input from [-1,1] to [0,1]
            input = input / 2. + 0.5
        
        output = self.permuto.forward(
            input, self.flattened_params, 
            max_level=(max_level or self.max_level), 
            need_dL_dinput=need_dL_dinput)
        return (output * self.window) if self.window is not None else output
    
    def backward_dydx(self, dL_dy: torch.Tensor, input: torch.Tensor, max_level: int=None, max_pos_dims: int=None) -> torch.Tensor:
        """ Caculate `nablas` given `dL_dy` (loss propagated or assigned to feature output)

        Args:
            dL_dy (torch.Tensor): [..., out_features], Loss propagated/assigned to lotd feature output.
            input (torch.Tensor): [..., in_features], Input relative position/coords in range [-1,1]
            max_level (int, optional): Maximum lotd level bypass. Defaults to None.

        Returns:
            torch.Tensor: [..., in_features], Back-propagated nablas
        """
        if isinstance(self.space, (AABBSpace, AABBDynamicSpace, BatchedBlockSpace, BatchedDynamicSpace)):
            # Maps input from [-1,1] to [0,1]
            nablas = self.permuto.backward_dydx(
                dL_dy, input / 2. + 0.5, self.flattened_params, 
                max_level=(max_level or self.max_level), max_pos_dims=max_pos_dims
            ) / 2. # NOTE: # Divided by 2 since `input` is also divided by 2 during the forward process
        else:
            nablas = self.permuto.backward_dydx(
                dL_dy, input, self.flattened_params, 
                max_level=(max_level or self.max_level), max_pos_dims=max_pos_dims)
        return nablas
    
    def get_level_param(self, l: int, grad=False) -> torch.Tensor:
        index, shape = level_param_index_shape(self.meta, l)
        return (self.flattened_params if not grad else self.flattened_params.grad)[index].view(shape)
    
    def set_level_param(self, l: int, value: torch.Tensor = ...):
        index, shape = level_param_index_shape(self.meta, l)
        self.flattened_params[index] = value.contiguous().view(prod(shape))
    
    @torch.no_grad()
    def init_lattice_values_random(self):
        param_init_cfg = self.param_init_cfg
        param_init_method = param_init_cfg['type']
        if param_init_method == 'uniform':
            bound = param_init_cfg['bound']
            self.flattened_params.uniform_(-bound, bound)
        elif param_init_method == 'normal':
            std = param_init_cfg['std']
            self.flattened_params.normal_(0, std)
        else:
            raise RuntimeError(f"Invalid param_init_method={param_init_method}")

    # Backup lotd_cfg just in case
    def get_extra_state(self) -> Any:
        return self.permuto_cfg
    
    def set_extra_state(self, state: Any):
        self.permuto_cfg = state

    @torch.no_grad()
    def clip_grad_and_update_ema(self, val: float=None):
        if self.clip_level_grad_ema_factor > 0:
            # gnorm = torch.stack([self.get_level_param(l, grad=True).data.abs().max() for l in range(self.lotd.n_levels)])
            gnorm = torch.stack([self.get_level_param(l, grad=True).data.norm() for l in range(self.lotd.n_levels)])
            
            ema = self.level_grad_norm_ema.copy_(gnorm.lerp(self.level_grad_norm_ema, 0.99))

            for lvl in range(self.lotd.n_levels):
                index, shape = level_param_index_shape(self.meta, lvl)
                
                val = self.clip_level_grad_ema_factor * ema[lvl].item()
                
                # self.flattened_params.grad[index].clip_(-val, val)
                clip_norm_(self.flattened_params.grad[index], val)

    @torch.no_grad()
    def stat_param(self, with_grad=False, prefix: str='') -> Dict[str, float]:
        prefix_ = prefix + ('.' if prefix and not prefix.endswith('.') else '')
        ret = {}
        
        ret.update({prefix_ + f'total.{k}': v for k,v in tensor_statistics(self.flattened_params.data).items()})
        if with_grad and self.flattened_params.grad is not None:
            ret.update({prefix_ + f'grad_total.{k}': v for k,v in tensor_statistics(self.flattened_params.grad.data).items()})
        for lvl in range(self.permuto.n_levels):
            ret.update({prefix_ + f'lv.{lvl}.{k}': v for k,v in tensor_statistics(self.get_level_param(lvl).data).items()})
            if with_grad and self.flattened_params.grad is not None:
                ret.update({prefix_ + f'grad.lv.{lvl}.{k}': v for k,v in tensor_statistics(self.get_level_param(lvl, grad=True).data).items()})
        return ret

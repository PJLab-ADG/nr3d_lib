"""
@file   permuto.py
@author Jianfei Guo, Shanghai AI Lab
@brief  A re-implementation of the permutohedral encoding.

New features:
- Support half(float16) param dtype
- Support 2 <= n_levels <= 20
- Support n_feats >= 2
- Support different layers using different widths (n_feats)
- Support batched inference with batch inds or batched input

Original: https://github.com/RaduAlexandru/permutohedral_encoding

Citation: 
@inproceedings{rosu2023permutosdf,
    title={PermutoSDF: Fast Multi-View Reconstruction with 
            Implicit Surfaces using Permutohedral Lattices  },
    author={Radu Alexandru Rosu and Sven Behnke},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2023}
}
"""

__all__ = [
    'generate_meta', 
    'level_param_index_shape', 
    'get_permuto_cfg', 
    'PermutoEncFunction', 
    'PermutoEncBwdInputFunction', 
    'permuto_enc_fwd', 
    'permuto_enc_bwd_input', 
    'PermutoEncImpl', 
]

import numpy as np
from math import prod
from numbers import Number
from typing import Any, List, Optional, Union

import torch
import torch.nn as nn
from torch.autograd.function import once_differentiable

from nr3d_lib.utils import check_to_torch
from nr3d_lib.profile import profile
import nr3d_lib.bindings._permuto as _backend

def generate_meta(n_input_dim: int, res_list: List[float], n_feats_list: List[int], hashmap_size: int):
    assert n_input_dim in _backend.supported_n_input_dims, f"n_input_dim={n_input_dim} not in supported list={_backend.supported_n_input_dims}"
    return _backend.PermutoEncMeta(n_input_dim, hashmap_size, res_list, n_feats_list)

def level_param_index_shape(meta, l: int):
    M = meta.level_n_feats[l]
    size = meta.level_sizes[l]
    offset = meta.level_offsets[l]
    offset_next = meta.level_offsets[l+1]
    index = (slice(offset, offset_next),)
    shape = (size, M)
    return index, shape

def get_permuto_cfg(
    type: str, 
    input_ch: int = ..., 
    stretch: Union[float, List[float]]=None, 
    **kwargs
    ) -> dict:
    def multi_res_cfg(
        coarsest_res: float = 10.0, 
        finest_res: float = 1000.0, 
        n_levels: int = 16, 
        n_feats: int = 2, 
        log2_hashmap_size: int = 19, 
        **other_kwargs # See PermutoEncImpl.__init__()
        ):
        res_list = np.geomspace(coarsest_res, finest_res, num=n_levels)
        n_feats_list = [n_feats] * n_levels
        hashmap_size = 2**log2_hashmap_size
        return dict(
            res_list=res_list, 
            n_feats_list=n_feats_list, 
            hashmap_size=hashmap_size, 
            **other_kwargs)
    
    if type == 'multi_res':
        return multi_res_cfg(**kwargs)
    else:
        raise RuntimeError(f"Invalid type={type}")

class PermutoEncFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, 
        # 0   1          2               3                         
        meta, positions, lattice_values, level_random_shifts=None, 
        # 4              5                   6                     7               8              9
        bidx=None, batch_offsets=None, batch_data_size=None, loss_scale=1.0, pos_scale=1.0, max_level=None, 
        # 10
        need_dL_dinput: Optional[bool]=None
        ):
        
        if need_dL_dinput is None:
            need_dL_dinput = torch.is_grad_enabled() and positions.requires_grad
        
        ctx.set_materialize_grads(False)
        prefix = positions.shape[:-1]
        
        bidx = None if bidx is None else bidx.contiguous().long().flatten()
        encoded = _backend.permuto_enc_fwd(
            meta, 
            positions.flatten(0, -2) * pos_scale, 
            lattice_values, 
            level_random_shifts, 
            bidx, batch_offsets, batch_data_size, 
            max_level
        )
        
        if need_dL_dinput or ctx.needs_input_grad[2]:
            ctx.save_for_backward(positions, lattice_values, level_random_shifts, bidx, batch_offsets, pos_scale)
            ctx.meta = meta
            ctx.prefix = prefix
            ctx.batch_data_size = batch_data_size
            ctx.loss_scale = loss_scale
            ctx.max_level = max_level
            ctx.need_dL_dinput = need_dL_dinput
        
        return encoded.unflatten(0, prefix)

    @staticmethod
    @once_differentiable
    def backward(ctx, dL_dy):
        if dL_dy is None:
            dL_dparam = None
            dL_dx = None
        else:
            if ctx.need_dL_dinput or ctx.needs_input_grad[2]:
                positions, lattice_values, level_random_shifts, bidx, batch_offsets, pos_scale = ctx.saved_tensors
                loss_scale = ctx.loss_scale
                dL_dx, dL_dparam = _backend.permuto_enc_bwd(
                    ctx.meta, 
                    dL_dy.flatten(0, -2) * loss_scale, 
                    positions.flatten(0, -2) * pos_scale, 
                    lattice_values, 
                    level_random_shifts, 
                    None if bidx is None else bidx.flatten(0, -1), 
                    batch_offsets, 
                    ctx.batch_data_size, ctx.max_level, None, 
                    # need_input_grad 
                    ctx.need_dL_dinput, 
                    # need_params_grad
                    ctx.needs_input_grad[2]
                )

                dL_dx = None if dL_dx is None else (dL_dx.unflatten(0, ctx.prefix) * (pos_scale / loss_scale))
                dL_dparam = None if dL_dparam is None else (dL_dparam / loss_scale)
            else:
                dL_dx = None
                dL_dparam = None
            
            # NOTE: Should not be used to calculate nablas. Use `PermutoEncBwdInputFunction` instead.
            #       Should only be used for calculating pose gradients, latent's gradients etc.
            ctx.mark_non_differentiable(dL_dx) 
            
            ctx.mark_non_differentiable(dL_dparam)
        # 0:meta,    1:positions, 2:lattice_values, 3,    4,    5,    6,    7,    8,    9,    10
        return None, dL_dx,       dL_dparam,        None, None, None, None, None, None, None, None

class PermutoEncBwdInputFunction(torch.autograd.Function):
    """
    Fwd function that only contains 1st order gradients.
    
    NOTE: Due to torch.autograd issue https://github.com/pytorch/pytorch/issues/56500
          To avoid useless and time-wasting `dy_dparam` calculation when just calculating `dy_dx` for nablas, 
              DO NOT use `PermutoEncFunction`'s backward() or its autograd.grad() to calculate nablas (dydx).
          Use `PermutoEncBwdInputFunction` below instead.
          See nr3d_lib/models/fields/sdf/permuto_sdf.py::forward_sdf_nablas for examples.
    """
    @staticmethod
    def forward(
        ctx,
        # 0   1      2          3               4                    
        meta, dL_dy, positions, lattice_values, level_random_shifts, 
        # 5         6              7                8               9
        bidx, batch_offsets, batch_data_size, loss_scale=1.0, pos_scale=1.0, 
        # 10            11
        max_level=None, max_pos_dims=None
        ):
        
        ctx.set_materialize_grads(False)
        prefix = positions.shape[:-1]
        bidx = None if bidx is None else bidx.contiguous().long().flatten()
        dL_dx, _ = _backend.permuto_enc_bwd(
            meta, 
            dL_dy.flatten(0, -2) * loss_scale, 
            positions.flatten(0, -2) * pos_scale, 
            lattice_values, 
            level_random_shifts, 
            bidx, batch_offsets, batch_data_size, 
            max_level, max_pos_dims, 
            # need_input_grad
            True,              
            # need_params_grad NOTE: Setting this to false explicitly saves time compared to pytorch's autograd when just calculating dydx.
            False
        )
        
        # Only consider sencond-order gradients that backwards on to dL_dy, grid
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[3]:
            ctx.save_for_backward(dL_dy, positions, lattice_values, level_random_shifts, bidx, batch_offsets, pos_scale)
            ctx.meta = meta
            ctx.batch_data_size = batch_data_size
            ctx.loss_scale = loss_scale
            ctx.max_level = max_level
            ctx.max_pos_dims = max_pos_dims
        
        dL_dx = None if dL_dx is None else (dL_dx.unflatten(0, prefix) * (pos_scale / loss_scale))
        return dL_dx
    
    @staticmethod
    @once_differentiable
    def backward(ctx, dL_ddLdx):
        # NOTE: 2nd order gradients. Currently support:
        #       ✓   d(dL_dinput)_d(dL_doutput)  dL_ddLdy_from_dL_ddLdx
        #       ✓   d(dL_dinput)_d(params)      dL_dparam_from_dL_ddLdx
        #       x   d(dL_dinput)_d(input)       dL_dinput_from_dL_ddLdx
        dL_dy, positions, lattice_values, level_random_shifts, bidx, batch_offsets, pos_scale = ctx.saved_tensors
        if dL_ddLdx is None:
            dL_ddLdy_from_dL_ddLdx = None
            dL_dparam_from_dL_ddLdx = None
        else:
            """
            from:   dL_ddLdx
            to:     dL_ddLdy, dL_dparam
            """
            prefix = positions.shape[:-1]
            # NOTE: Be cautious when multiplying and dividing loss_scale:
            #       dL_ddLdy_from_dL_ddLdx  uses dL_ddLdx
            #       dL_dparam_from_dL_ddLdx  uses dL_ddLdx * dL_dy
            loss_scale = ctx.loss_scale
            dL_ddLdy_from_dL_ddLdx, dL_dparam_from_dL_ddLdx = _backend.permuto_enc_bwd_bwd_input(
                ctx.meta, 
                dL_ddLdx.flatten(0, -2).to(positions.dtype) * pos_scale, 
                dL_dy.flatten(0, -2) * loss_scale, 
                positions.flatten(0, -2) * pos_scale, 
                lattice_values, 
                level_random_shifts, 
                None if bidx is None else bidx.flatten(), 
                batch_offsets, ctx.batch_data_size, ctx.max_level, 
                # need_dLdinput_ddLdoutput, need_dLdinput_dparams
                ctx.needs_input_grad[1],    ctx.needs_input_grad[3]
            )
            dL_ddLdy_from_dL_ddLdx = None if dL_ddLdy_from_dL_ddLdx is None else dL_ddLdy_from_dL_ddLdx.unflatten(0, prefix)
            dL_dparam_from_dL_ddLdx = None if dL_dparam_from_dL_ddLdx is None else (dL_dparam_from_dL_ddLdx / loss_scale)
        
        # 0:meta,    1:dL_dy,                2:positions, 3:lattice_values,        4,    5,    6,    7,    8,    9,    10,   11
        return None, dL_ddLdy_from_dL_ddLdx, None,        dL_dparam_from_dL_ddLdx, None, None, None, None, None, None, None, None

def permuto_enc_fwd(
    positions: torch.Tensor, lattice_values: torch.Tensor, level_random_shifts: torch.Tensor = None, 
    bidx: torch.Tensor = None, batch_offsets: torch.Tensor = None, 
    input_batched=False, max_level=None, need_dL_dinput: Optional[bool]=None, pos_scale: float = 1.0, 
    meta=None, n_input_dim: int = None, res_list: List[float] = None, n_feats_list: Union[int, List[int]] = None, hashmap_size: int = None
    ) -> torch.Tensor:
    if need_dL_dinput is None:
        need_dL_dinput = torch.is_grad_enabled() and positions.requires_grad
    if meta is None:
        meta = generate_meta(n_input_dim, res_list, n_feats_list, hashmap_size)
    if input_batched:
        batch_data_size = prod(positions.shape[1:-1])
        bidx = None
    else:
        batch_data_size = 0
    loss_scale = 128.0 if (lattice_values.dtype == torch.float16) else 1.
    encoded = PermutoEncFunction.apply(
        meta, positions.float(), lattice_values, level_random_shifts, 
        bidx, batch_offsets, batch_data_size, loss_scale, pos_scale, max_level, need_dL_dinput)
    return encoded

def permuto_enc_bwd_input(
    meta, dL_dy: torch.Tensor, positions: torch.Tensor, lattice_values: torch.Tensor, level_random_shifts: torch.Tensor = None, 
    bidx: torch.Tensor = None, batch_offsets: torch.Tensor = None, input_batched=False, max_level: int=None, max_pos_dims: int=None, pos_scale: float = 1.0
    ) -> torch.Tensor:
    if input_batched:
        batch_data_size = prod(positions.shape[1:-1])
        bidx = None
    else:
        batch_data_size = 0
    loss_scale = 128.0 if (lattice_values.dtype == torch.float16) else 1.
    dL_dx = PermutoEncBwdInputFunction.apply(
        meta, dL_dy, positions.float(), lattice_values, level_random_shifts, 
        bidx, batch_offsets, batch_data_size, loss_scale, pos_scale, max_level, max_pos_dims)
    return dL_dx

class PermutoEncImpl(nn.Module):
    def __init__(
        self, 
        in_features: int, 
        res_list: List[float], 
        n_feats_list: List[int], 
        hashmap_size: int = None, 
        log2_hashmap_size: int = None, 
        apply_random_shifts_per_level=True, 
        pos_scale: float = 1.0, 
        dtype=torch.half, device=None
        ) -> None:
        
        super().__init__()
        assert dtype == torch.float or dtype == torch.float16, "dtype must be one of torch.float or torch.float16"
        assert bool(log2_hashmap_size is None) != bool(hashmap_size is None), "Please specify one of [hashmap_size, log2_hashmap_size]"
        
        self.loss_scale = 128.0 if dtype == torch.float16 else 1.0
        self.dtype = dtype
        
        if log2_hashmap_size is not None:
            assert hashmap_size is None, "Do not specify `hashmap_size` when `log2_hashmap_size` is already specified."
            hashmap_size = 2 ** log2_hashmap_size
        
        self.meta = generate_meta(in_features, res_list, n_feats_list, hashmap_size)

        if apply_random_shifts_per_level:
            level_random_shfits = 10.0 * torch.randn([self.n_levels, self.in_features], dtype=torch.float, device=device)
        else:
            level_random_shfits = torch.zeros([self.n_levels, self.in_features], dtype=torch.float, device=device)
        self.register_buffer('level_random_shifts', level_random_shfits, persistent=True)
        
        pos_scale = check_to_torch(pos_scale, dtype=torch.float, device=device).expand(in_features).clone()
        self.register_buffer('pos_scale', pos_scale, persistent=True)

        self.params = {
            'in_features': in_features, 
            'res_list': res_list,
            'n_feats_list': n_feats_list,
            'hashmap_size': hashmap_size,
            'log2_hashmap_size': log2_hashmap_size, 
            'apply_random_shifts_per_level': apply_random_shifts_per_level,
            'pos_scale': pos_scale, 
            'dtype': dtype,
            'device': device
        }

    @property
    def in_features(self) -> int:
        """Number of dims to encode"""
        return self.meta.n_dims_to_encode
    
    @property
    def out_features(self) -> int:
        """Number of encoded dims"""
        return self.meta.n_encoded_dims
    
    @property
    def n_levels(self) -> int:
        """Number of actual levels (allow non-equal feature width)"""
        return self.meta.n_levels
    
    @property
    def n_params(self) -> int:
        """Number of total params"""
        return self.meta.n_params
    
    @property
    def level_scales0(self) -> List[float]:
        return self.meta.level_scales0
    
    @property
    def level_sizes(self) -> List[int]:
        """[n_levels] Grid sizes for each actual `level` (not considering feature width)"""
        return self.meta.level_sizes
    
    @property
    def level_offsets(self) -> List[int]:
        """[n_levels+1] Parameter offsets for each actual `level` in a single entry (considering feature width)"""
        return self.meta.level_offsets
    
    @property
    def level_n_feats(self) -> List[int]:
        """[n_levels] Feature width for each actual `level`"""
        return self.meta.level_n_feats
    
    @property
    def level_n_params(self) -> List[int]:
        """[n_levels] Grid parameter sizes for each actual `level` (considering feature width)"""
        return self.meta.level_n_params
    
    @profile
    def forward(
        self, 
        positions: torch.Tensor, lattice_values: torch.Tensor, 
        bidx: torch.Tensor = None, batch_offsets: torch.Tensor = None, input_batched=False, 
        max_level: int=None, need_dL_dinput: Optional[bool]=None) -> torch.Tensor:
        if need_dL_dinput is None:
            need_dL_dinput = torch.is_grad_enabled() and positions.requires_grad
        if input_batched:
            assert bidx is None, 'bidx is only taken care of when input is not batched.'
            batch_data_size = prod(positions.shape[1:-1])
        else:
            batch_data_size = 0
        encoded = PermutoEncFunction.apply(
            self.meta, positions.float(), lattice_values.to(self.dtype), self.level_random_shifts, 
            bidx, batch_offsets, batch_data_size, self.loss_scale, self.pos_scale, max_level, need_dL_dinput)
        return encoded
    
    @profile
    def backward_dydx(
        self, 
        dL_dy: torch.Tensor, positions: torch.Tensor, lattice_values: torch.Tensor, 
        bidx: torch.Tensor = None, batch_offsets: torch.Tensor = None, input_batched=False, 
        max_level: int=None, max_pos_dims: int=None) -> torch.Tensor:
        if input_batched:
            assert bidx is None, 'bidx is only taken care of when input is not batched.'
            batch_data_size = prod(positions.shape[1:-1])
        else:
            batch_data_size = 0
        dL_dx = PermutoEncBwdInputFunction.apply(
            self.meta, dL_dy, positions.float(), lattice_values.to(self.dtype), self.level_random_shifts, 
            bidx, batch_offsets, batch_data_size, self.loss_scale, self.pos_scale, max_level, max_pos_dims)
        return dL_dx
    
    def __getstate__(self):
        self.params['device'] = self.level_random_shifts.device
        self.params['level_random_shifts'] = self.level_random_shifts
        return self.params
    
    def __setstate__(self, state_dict):
        level_random_shifts = state_dict.pop('level_random_shifts')
        self.__init__(**state_dict)
        self.level_random_shifts = level_random_shifts
    
    def extra_repr(self) -> str:
        if self.dtype == torch.float64:
            ele_size = 8
        elif self.dtype == torch.float32:
            ele_size = 4
        elif self.dtype == torch.float16:
            ele_size = 2
        else:
            raise RuntimeError(f"Invalid self.dtype={self.dtype}")
        
        hyperparams1 = {
            'in_dim': self.meta.n_dims_to_encode, 
            'out_dim': self.meta.n_encoded_dims, 
            'num_levels': self.meta.n_levels, 
            'num_params': self.meta.n_params, 
            'params_size': f"{(self.meta.n_params * ele_size) / (1024**2):.3f} MiB", 
            'pos_scale': self.pos_scale, 
            'dtype': self.dtype,
        }
        level_n_params = self.meta.level_n_params
        level_n_params_cumsum = np.cumsum(level_n_params)
        level_n_params_ratio = np.array(level_n_params) / sum(level_n_params)
        level_n_params_ratio = f"[{', '.join([f'{i:.3f}' for i in level_n_params_ratio])}]"
        level_n_params_ratio_cumsum = level_n_params_cumsum / level_n_params_cumsum[-1]
        level_n_params_ratio_cumsum = f"[{', '.join([f'{i:.3f}' for i in level_n_params_ratio_cumsum])}]"
        
        hyperparams2 = {
            'level_scales0': self.meta.level_scales0, 
            'level_n_feats': self.meta.level_n_feats, 
            "level_n_params": level_n_params, 
            "level_n_params_cumsum": level_n_params_cumsum.tolist(), 
            'level_n_params_ratio': level_n_params_ratio, 
            "level_n_params_cumsum_ratio": level_n_params_ratio_cumsum
        }
        
        return ", ".join([f"{k}={v}" for k, v in hyperparams1.items()]) + "\n" + "\n".join([f"{k}={v}" for k, v in hyperparams2.items()])

if __name__ == "__main__":
    def unit_test():
        device = torch.device('cuda')
        enc = PermutoEncImpl(
            12, 
            np.geomspace(16.0, 2048.0, num=16).tolist(), 
            [2]*16, 
            log2_hashmap_size=18, 
            apply_random_shifts_per_level=True, 
            dtype=torch.float16, device=device
        )
        print(enc)
    unit_test()

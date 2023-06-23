"""
@file   lotd.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Basic LoTD python bindings API.
"""

__all__ = [
    'LoDType', 
    'generate_meta', 
    'LoTDFunction', 
    'LoTDFunctionFwdDydx', 
    'LoTDFunctionBwdDydx', 
    'lotd_encoding', 
    'lotd_encoding_fwd_dydx', 
    'lotd_encoding_bwd_dydx', 
    'LoTD'
]

import numpy as np
from enum import Enum
from math import prod
from typing import List, Literal, Tuple, Union

import torch
import torch.nn as nn
from torch.autograd.function import once_differentiable

import nr3d_lib_bindings._lotd as _backend

class LoDType(Enum):
    Dense = int(_backend.LoDType.Dense)
    VectorMatrix = int(_backend.LoDType.VectorMatrix)
    CP = int(_backend.LoDType.CP)
    CPfast = int(_backend.LoDType.CPfast)
    NPlaneMul = int(_backend.LoDType.NPlaneMul)
    NPlaneSum = int(_backend.LoDType.NPlaneSum)
    Hash = int(_backend.LoDType.Hash)

def generate_meta(n_input_dim: Literal[2,3], lod_res: List[int], lod_n_feats: Union[int, List[int]], lod_types: Union[str, List[str]], hashmap_size: int=None, use_smooth_step=False):
    if isinstance(lod_n_feats, int):
        lod_n_feats = [lod_n_feats] * len(lod_res)
    if isinstance(lod_types, str):
        lod_types = [lod_types] * len(lod_res)
    return _backend.LoDMeta(n_input_dim, lod_res, lod_n_feats, lod_types, hashmap_size, use_smooth_step)

class LoTDFunction(torch.autograd.Function):
    """
    Fwd function that only contains 1st order gradients.
    
    NOTE: Due to torch.autograd issue https://github.com/pytorch/pytorch/issues/56500
          To avoid useless and time-wasting `dy_dparam` calculation when just calculating `dy_dx` for nablas, 
              DO NOT use `LoTDFunction`'s backward() or autograd.grad() to calculate nablas (dydx).
          Use LoTDFunctionFwdDydx and LoTDFunctionBwdDydx below instead.
          See nr3d_lib/models/fields/sdf/lotd_sdf.py::forward_sdf_nablas for examples.
    """
    @staticmethod
    def forward(
        ctx,
        # 0   1  2     3                4                   5                     6               7
        meta, x, grid, batch_inds=None, batch_offsets=None, batch_data_size=None, loss_scale=1.0, max_level=None):
        
        ctx.set_materialize_grads(False)
        prefix = x.shape[:-1]
        
        x = x.clamp(1.0e-6, 1-1.0e-6)
        batch_inds = None if batch_inds is None else batch_inds.contiguous().long().flatten()
        y, dy_dx = _backend.lod_fwd(meta, x.flatten(0, -2), grid, batch_inds, batch_offsets, batch_data_size, max_level, ctx.needs_input_grad[1])
        
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            ctx.save_for_backward(x, grid, dy_dx, batch_inds, batch_offsets)
            ctx.meta = meta
            ctx.prefix = prefix
            ctx.batch_data_size = batch_data_size
            ctx.loss_scale = loss_scale
            ctx.max_level = max_level
        
        return y.unflatten(0, prefix)
    
    @staticmethod
    @once_differentiable
    def backward(ctx, dL_dy):
        """
        from:   dL_dy
        to:     dL_dx, dL_dgrid
        """
        if dL_dy is None:
            dL_dgrid = None
            dL_dx = None
        else:            
            if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
                dL_dgrid = None
            
                x, grid, dy_dx, batch_inds, batch_offsets = ctx.saved_tensors
                loss_scale = ctx.loss_scale
                
                dL_dx, dL_dgrid = _backend.lod_bwd(
                    ctx.meta, dL_dy.flatten(0, -2) * loss_scale, x.flatten(0, -2), grid, dy_dx, 
                    None if batch_inds is None else batch_inds.flatten(0, -1), batch_offsets, ctx.batch_data_size, ctx.max_level, 
                    # need_input_grad 
                    ctx.needs_input_grad[1], 
                    # need_params_grad
                    ctx.needs_input_grad[2]
                )

                dL_dx = None if dL_dx is None else dL_dx.unflatten(0, ctx.prefix) / loss_scale
                dL_dgrid = None if dL_dgrid is None else dL_dgrid / loss_scale
            else:
                dL_dx = None
                dL_dgrid = None
            ctx.mark_non_differentiable(dL_dx)
            ctx.mark_non_differentiable(dL_dgrid)
        # 0:meta,    1:x,   2:grid,   3:batch_inds, 4:batch_offsets, 5:batch_data_size, 6:loss_scale, 7:max_level
        return None, dL_dx, dL_dgrid, None,         None,            None,              None,         None

class LoTDFunctionFwdDydx(torch.autograd.Function):
    """
    LoTDFunctionFwdDydx & LoTDFunctionBwdDydx combined to calculate dydx with 2nd order gradients.
    See nr3d_lib/models/fields/sdf/lotd_sdf.py::forward_sdf_nablas for examples.
    NOTE: 
        Due to torch.autograd issue https://github.com/pytorch/pytorch/issues/56500
        To avoid useless and time-wasting `dy_dparam` calculation when just calculating `dy_dx` for nablas, but also allow calculating dy_dparam when training:
            DO NOT use this backward() or autograd.grad() to calculate nablas.
            Use LoTDFunctionBwdDydx.forward() to calculate nablas instead.
    """
    @staticmethod
    def forward(
        ctx, 
        # 0   1  2     3                4                   5                     6
        meta, x, grid, batch_inds=None, batch_offsets=None, batch_data_size=None, loss_scale=1.0, max_level=None, need_loss_backward_input=False):
                
        # If no output gradient is provided, no need to automatically materialize it as torch.zeros.
        ctx.set_materialize_grads(False)
        prefix = x.shape[:-1]
        x = x.clamp(1.0e-6, 1-1.0e-6)
        batch_inds = None if batch_inds is None else batch_inds.contiguous().long().flatten()
        y, dy_dx = _backend.lod_fwd(meta, x.flatten(0, -2), grid, batch_inds, batch_offsets, batch_data_size, max_level, True)
        
        # x, grid
        # if need_loss_backward_input or ctx.needs_input_grad[2]:
        ctx.save_for_backward(x, grid, dy_dx, batch_inds, batch_offsets)
        ctx.prefix = prefix
        ctx.meta = meta
        ctx.batch_data_size = batch_data_size
        ctx.loss_scale = loss_scale
        ctx.max_level = max_level
        ctx.need_loss_backward_input = need_loss_backward_input
        
        ctx.mark_non_differentiable(dy_dx)
        return y.unflatten(0, prefix), dy_dx

    @staticmethod
    @once_differentiable
    def backward(ctx, dL_dy, _1):
        """
        from:   dL_dy, x, grid
        to:     (dL_dx), dL_dgrid # NOTE: only returns dL_dx when `need_loss_backward_input`.
        """
        if dL_dy is None:
            dL_dx = None
            dL_dgrid = None
        else:
            x, grid, dy_dx, batch_inds, batch_offsets = ctx.saved_tensors
            loss_scale = ctx.loss_scale
            dL_dy_flatten = dL_dy.flatten(0, -2) * loss_scale
            x_flatten = x.flatten(0, -2)
            with torch.no_grad():
                dL_dx, dL_dgrid = _backend.lod_bwd(
                    ctx.meta, dL_dy_flatten, x_flatten, grid, dy_dx, None if batch_inds is None else batch_inds.flatten(), batch_offsets, ctx.batch_data_size, ctx.max_level, 
                    # need_input_grad,   NOTE: intentionally overwritten to ctx.need_loss_backward_input, instead of ctx.needs_input_grad[1].
                    ctx.need_loss_backward_input, 
                    # need_params_grad
                    ctx.needs_input_grad[2]
                )
                dL_dx = None if dL_dx is None else dL_dx.unflatten(0, ctx.prefix) / loss_scale
                dL_dgrid = None if dL_dgrid is None else dL_dgrid / loss_scale
        ctx.mark_non_differentiable(dL_dx) # NOTE: If you want second-order gradients for nablas, use LoTDFunctionBwdDydx.forward() instead.
        ctx.mark_non_differentiable(dL_dgrid)
        # 0:meta,    1:x,   2:grid,   3:batch_inds, 4:batch_offsets, 5:batch_data_size, 6:loss_scale, 7:max_level
        return None, dL_dx, dL_dgrid, None,         None,            None,              None,         None, None

class LoTDFunctionBwdDydx(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        # 0   1      2  3     4      5           6              7                8           9,         
        meta, dL_dy, x, grid, dy_dx, batch_inds, batch_offsets, batch_data_size, loss_scale, max_level, grad_guard):
        """
        from: dL_dy, x, grid
        to:   dL_dx
        """
        ctx.set_materialize_grads(False)
        prefix = x.shape[:-1]
        x = x.clamp(1.0e-6, 1-1.0e-6)
        batch_inds = None if batch_inds is None else batch_inds.contiguous().long().flatten()
        dL_dx, _ = _backend.lod_bwd(
            meta, dL_dy.flatten(0, -2) * loss_scale, x.flatten(0, -2), grid, dy_dx, batch_inds, batch_offsets, batch_data_size, max_level, 
            # need_input_grad
            True,              
            # need_params_grad NOTE: Setting this to false explicitly saves time compared to pytorch's autograd when just calculating dydx.
            False
        )
        
        # Only consider sencond-order gradients that backwards on to dL_dy, grid
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[3]:
            ctx.save_for_backward(dL_dy, x, grid, dy_dx, batch_inds, batch_offsets)
            ctx.meta = meta
            ctx.batch_data_size = batch_data_size
            ctx.loss_scale = loss_scale
            ctx.max_level = max_level
            ctx.grad_guard = grad_guard
        
        dL_dx = None if dL_dx is None else dL_dx.unflatten(0, prefix) / loss_scale
        return dL_dx

    @staticmethod
    @once_differentiable
    def backward(ctx, dL_ddLdx):
        # NOTE: 2nd order gradients. Currently support:
        #       ✓   d(dL_dinput)_d(dL_doutput)  dL_ddLdy_from_dL_ddLdx
        #       ✓   d(dL_dinput)_d(input)       dL_dinput_from_dL_ddLdx
        #       ✓   d(dL_dinput)_d(params)      dL_dgrid_from_dL_ddLdx
        dL_dy, x, grid, dy_dx, batch_inds, batch_offsets = ctx.saved_tensors
        if dL_ddLdx is None:
            dL_ddLdy_from_dL_ddLdx = None
            dL_dinput_from_dL_ddLdx = None
            dL_dgrid_from_dL_ddLdx = None
        else:
            """
            from:   dL_ddLdx
            to:     dL_ddLdy, dL_dx, dL_dgrid
            """
            # NOTE: Be cautious when multiplying and dividing loss_scale:
            #       dL_ddLdy_from_dL_ddLdx  uses dL_ddLdx
            #       dL_dgrid_from_dL_ddLdx  uses dL_ddLdx * dL_dy
            #       dL_dinput_from_dL_ddLdx uses dL_ddLdx * dL_dy
            prefix = x.shape[:-1]
            loss_scale = ctx.loss_scale
            dL_ddLdy_from_dL_ddLdx, dL_dgrid_from_dL_ddLdx, dL_dinput_from_dL_ddLdx = _backend.lod_bwd_bwd_input(
                ctx.meta, dL_ddLdx.flatten(0, -2), dL_dy.flatten(0, -2) * loss_scale, x.flatten(0, -2), grid, dy_dx, None if batch_inds is None else batch_inds.flatten(), batch_offsets, ctx.batch_data_size, ctx.max_level, 
                # need_dLdinput_ddLdoutput, need_dLdinput_dparams,   need_dLdinput_dinput, 
                ctx.needs_input_grad[1],    ctx.needs_input_grad[3], False # Usually not needed.
            )
            
            dL_ddLdy_from_dL_ddLdx = None if dL_ddLdy_from_dL_ddLdx is None else dL_ddLdy_from_dL_ddLdx.unflatten(0, prefix)
            # dL_dgrid_from_dL_ddLdx = None if dL_dgrid_from_dL_ddLdx is None else (dL_dgrid_from_dL_ddLdx / loss_scale).clamp_(-5.0e-6, 5.0e-6)
            dL_dgrid_from_dL_ddLdx = None if dL_dgrid_from_dL_ddLdx is None else (dL_dgrid_from_dL_ddLdx / loss_scale)
            dL_dinput_from_dL_ddLdx = None if dL_dinput_from_dL_ddLdx is None else dL_dinput_from_dL_ddLdx.unflatten(0, prefix) / loss_scale
            
            if (ctx.grad_guard is not None) and (dL_dgrid_from_dL_ddLdx is not None or dL_ddLdy_from_dL_ddLdx is not None):
                ctx.grad_guard.custom_grad_clip_step(dL_ddLdx, dy_dx, dL_dgrid_from_dL_ddLdx, dL_ddLdy_from_dL_ddLdx)

        # 0:meta,    1:dL_dy,                2:x,                     3:grid,                 4:dy_dx 5:batch_inds, 6:batch_offsets, 7:batch_data_size, 8:loss_scale, 9:max_level
        return None, dL_ddLdy_from_dL_ddLdx, dL_dinput_from_dL_ddLdx, dL_dgrid_from_dL_ddLdx, None,   None,         None,            None,              None,         None, None

def lotd_encoding(
    input: torch.Tensor, params: torch.Tensor, batch_inds: torch.Tensor = None, batch_offsets: torch.Tensor = None, input_batched=False, max_level=None, 
    meta=None, n_input_dim: Literal[2,3]=None, lod_res: List[int]=None, lod_n_feats: Union[int, List[int]]=None, lod_types: Union[str, List[str]]=None):
    if meta is None:
        meta = generate_meta(n_input_dim, lod_res, lod_n_feats, lod_types) # 20 us
    if input_batched:
        batch_data_size = prod(input.shape[1:-1])
        batch_inds = None
    else:
        batch_data_size = 0
    loss_scale = 128.0 if (params.dtype == torch.float16) else 1.
    output = LoTDFunction.apply(meta, input, params, batch_inds, batch_offsets, batch_data_size, loss_scale, max_level)
    return output

def lotd_encoding_fwd_dydx(
    input: torch.Tensor, params: torch.Tensor, batch_inds: torch.Tensor = None, batch_offsets: torch.Tensor = None, input_batched=False, max_level=None, need_loss_backward_input=False, 
    meta=None, n_input_dim: Literal[2,3]=None, lod_res: List[int]=None, lod_n_feats: Union[int, List[int]]=None, lod_types: Union[str, List[str]]=None):
    if meta is None:
        meta = generate_meta(n_input_dim, lod_res, lod_n_feats, lod_types) # 20 us
    if input_batched:
        batch_data_size = prod(input.shape[1:-1])
        batch_inds = None
    else:
        batch_data_size = 0
    loss_scale = 128.0 if (params.dtype == torch.float16) else 1.
    output, dy_dx = LoTDFunctionFwdDydx.apply(meta, input, params, batch_inds, batch_offsets, batch_data_size, loss_scale, max_level, need_loss_backward_input)
    return output, dy_dx, meta

def lotd_encoding_bwd_dydx(
    meta, dL_dy: torch.Tensor, dy_dx: torch.Tensor, input: torch.Tensor, params: torch.Tensor, batch_inds: torch.Tensor = None, batch_offsets: torch.Tensor = None, input_batched=False, max_level=None):
    if input_batched:
        batch_data_size = prod(input.shape[1:-1])
        batch_inds = None
    else:
        batch_data_size = 0
    loss_scale = 128.0 if (params.dtype == torch.float16) else 1.
    dL_dx = LoTDFunctionBwdDydx.apply(meta, dL_dy, input, params, dy_dx, batch_inds, batch_offsets, batch_data_size, loss_scale, max_level)
    return dL_dx

def lotd_get_grid_index(meta, input: torch.Tensor, batch_inds: torch.Tensor = None, batch_offsets: torch.Tensor = None, input_batched=False, max_level=None) -> torch.LongTensor:
    # NOTE: Only support single / batched for now; forest not supported.
    if input_batched:
        batch_data_size = prod(input.shape[1:-1])
        batch_inds = None
    else:
        batch_data_size = 0
    grid_inds = _backend.lod_get_grid_index(meta, input, batch_inds, batch_offsets, batch_data_size, max_level)
    return grid_inds

class LoTD(nn.Module):
    def __init__(
        self,
        in_features: Literal[2,3], 
        lod_res: Union[List[int], List[List[int]]],
        lod_n_feats: Union[int, List[int]],
        lod_types: Union[str, List[str]],
        hashmap_size: int = None,
        log2_hashmap_size: int = None, 
        use_smooth_step = False,
        dtype=torch.float16, device=torch.device('cuda')
        ):
        super().__init__()
        assert dtype == torch.float or dtype == torch.float16, "dtype must be one of torch.float or torch.float16"
        self.params = {
            'in_features': in_features, 
            'lod_res': lod_res,
            'lod_n_feats': lod_n_feats,
            'lod_types': lod_types,
            'hashmap_size': hashmap_size,
            'log2_hashmap_size': log2_hashmap_size, 
            'use_smooth_step': use_smooth_step,
            'dtype':dtype,
            'device':device
        }

        self.loss_scale = 128.0 if dtype == torch.float16 else 1.0
        self.dtype = dtype
        self.device = device
        
        if log2_hashmap_size is not None:
            assert hashmap_size is None, "Do not specify `hashmap_size` when `log2_hashmap_size` is already specified."
            hashmap_size = 2 ** log2_hashmap_size
        
        self.meta = generate_meta(in_features, lod_res, lod_n_feats, lod_types, hashmap_size, use_smooth_step)
    
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
    def level_res_multi_dim(self) -> List[List[int]]:
        """[n_levels], each [n_dims_to_encode]:		Grid side lengths for each actual `level`"""
        return self.meta.level_res_multi_dim
    
    @property
    def level_res(self) -> List[int]:
        """[n_levels] Grid side lengths for each actual `level` (Only valid when cubic)"""
        # `level_res` only is valid when it's not cuboid
        level_res_multi_dim = np.array(self.meta.level_res_multi_dim)
        if (level_res_multi_dim == level_res_multi_dim[:,0][:,None]).all():
            return level_res_multi_dim[:,0].tolist()
        else:
            return None
    
    @property
    def level_types(self) -> List[LoDType]:
        """[n_levels] lod types for each actual `level`"""
        return [LoDType(tp) for tp in self.meta.level_types]
    
    @property
    def level_types_str(self) -> List[str]:
        """[n_levels] lod type strings for each actual `level`"""
        return self.meta.level_types_str
    
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
    
    def forward(self, input: torch.Tensor, params: torch.Tensor, batch_inds: torch.Tensor = None, batch_offsets: torch.Tensor = None, input_batched=False, max_level: int=None) -> torch.Tensor:
        if input_batched:
            assert batch_inds is None, 'batch_inds is only taken care of when input is not batched.'
            batch_data_size = prod(input.shape[1:-1])
        else:
            batch_data_size = 0
        output = LoTDFunction.apply(self.meta, input, params.to(self.dtype), batch_inds, batch_offsets, batch_data_size, self.loss_scale, max_level)
        return output
    
    def forward_dydx(self, input: torch.Tensor, params: torch.Tensor, batch_inds: torch.Tensor = None, batch_offsets: torch.Tensor = None, input_batched=False, max_level: int=None, need_loss_backward_input=False) -> Tuple[torch.Tensor, torch.Tensor]:
        if input_batched:
            assert batch_inds is None, 'batch_inds is only taken care of when input is not batched.'
            batch_data_size = prod(input.shape[1:-1])
        else:
            batch_data_size = 0
        output, dy_dx = LoTDFunctionFwdDydx.apply(self.meta, input, params.to(self.dtype), batch_inds, batch_offsets, batch_data_size, self.loss_scale, max_level, need_loss_backward_input)
        return output, dy_dx
    
    def backward_dydx(self, dL_dy: torch.Tensor, dy_dx: torch.Tensor, input: torch.Tensor, params: torch.Tensor, batch_inds: torch.Tensor = None, batch_offsets: torch.Tensor = None, input_batched=False, max_level: int=None, grad_guard=None) -> torch.Tensor:
        if input_batched:
            assert batch_inds is None, 'batch_inds is only taken care of when input is not batched.'
            batch_data_size = prod(input.shape[1:-1])
        else:
            batch_data_size = 0
        nablas = LoTDFunctionBwdDydx.apply(self.meta, dL_dy, input, params.to(self.dtype), dy_dx, batch_inds, batch_offsets, batch_data_size, self.loss_scale, max_level, grad_guard)
        return nablas

    def __getstate__(self):
        return self.params
    
    def __setstate__(self, state_dict):
        self.__init__(**state_dict)

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
            'device': self.device, 
            'dtype': self.dtype,
        }
        level_n_params = self.meta.level_n_params
        level_n_params_cumsum = np.cumsum(level_n_params)
        lod_n_params_ratio = np.array(level_n_params) / sum(level_n_params)
        lod_n_params_ratio = f"[{', '.join([f'{i:.3f}' for i in lod_n_params_ratio])}]"
        lod_n_params_ratio_cumsum = level_n_params_cumsum / level_n_params_cumsum[-1]
        lod_n_params_ratio_cumsum = f"[{', '.join([f'{i:.3f}' for i in lod_n_params_ratio_cumsum])}]"
        
        hyperparams2 = {
            'lod_res': self.meta.level_res, 
            'lod_res_multi_dim': self.meta.level_res_multi_dim, 
            'lod_n_feats': self.meta.level_n_feats, 
            'lod_types': self.meta.level_types_str, 
            "lod_n_params": level_n_params, 
            "lod_n_params_cumsum": level_n_params_cumsum.tolist(), 
            'lod_n_params_ratio': lod_n_params_ratio, 
            "lod_n_params_cumsum_ratio": lod_n_params_ratio_cumsum
        }
        
        return ", ".join([f"{k}={v}" for k, v in hyperparams1.items()]) + "\n" + "\n".join([f"{k}={v}" for k, v in hyperparams2.items()])

if __name__ == "__main__":
    def test(device=torch.device('cuda')):
        enc = LoTD(
            3, 
            [34,      55,     90, 140, 230, 370, 600, 1000, 1600, 2600, 4200], 
            [2,       2,      2,  2,   2,   2,   2,   2,    2,    2,    2], 
            ['Dense','Dense','VM','VM','VM','VM','VM','VM','VM', 'CPfast', 'CPfast'], 
            store_params=True, dtype=torch.float16, device=device)
        print(enc)
    test()

"""
@file   lotd_helpers.py
@author Jianfei Guo, Shanghai AI Lab
@brief  LoTD helper functions.
"""

__all__ = [
    'param_vertices', 
    'param_interpolate', 
    'level_param_index_shape', 
    'get_level_param', 
    'get_level_param_batched',  
    'LoTD2ndGradGuard'
]

import numpy as np
from math import prod
from numbers import Number
from typing import Literal, Union, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nr3d_lib.utils import tensor_statistics

from nr3d_lib.models.utils import clip_norm_
from nr3d_lib.models.grid_encodings.utils import gridsample1d_by2d
from nr3d_lib.models.grid_encodings.lotd.lotd import LoDType

def level_param_index_shape(lod_meta, l: int, op: str = None, dim: int = None) -> Tuple[tuple, tuple]:
    """ A helper function for LoTD: 
        To get the param index (slices / offset) and their human-readable shapes of a certain level.

    Args:
        lod_meta (_type_): LoTD meta data
        l (int): The given level number (0 to max_level-1)
        op (str, optional): Optionally for VM to specify `mat` or `vec`. Defaults to None.
        dim (int, optional): Optionally for multi-dim lotd types to specify `dim`. Defaults to None.

    Returns:
        Tuple[tuple, tuple]: The index (slice / offset) and the shape of the required level and op.
    """
    D = lod_meta.n_dims_to_encode
    L = lod_meta.n_levels
    assert 0 <= l < L
    if dim is not None: assert 0 <= dim < D
    
    RS = lod_meta.level_res_multidim[l]
    R0 = RS[0]
    LINE_RCS = [0, *np.cumsum(RS)] # Line size cumsum
    PLANE_RS = [int(prod(RS)//R) for R in RS] # Plane sizes
    PLANE_RCS = [0, *np.cumsum(PLANE_RS)] # Plane size cumsum
    
    M = lod_meta.level_n_feats[l]
    tp = LoDType(int(lod_meta.level_types[l]))
    size = lod_meta.level_sizes[l]
    offset = lod_meta.level_offsets[l]
    offset_next = lod_meta.level_offsets[l+1]
    
    if D == 3:
        if tp == LoDType.Dense:
            if op is None:
                index = (slice(offset, offset_next),)
                shape = (size, M)
            elif op == 'vol':
                index = (slice(offset, offset_next),)
                shape = (*RS, M)
            else:
                raise RuntimeError(f"Invalid op={op}")
        
        elif tp == LoDType.VectorMatrix:
            if op is None:
                index = (slice(offset, offset_next),)
                shape = (size, M)
            elif op == 'line' or op == 'vec':
                if dim is None:
                    index = (slice(offset, offset+LINE_RCS[-1]*M),)
                    # assert all([R==R0 for R in RS]), "Expect equal resolution!"
                    if all([R==R0 for R in RS]):
                        shape = (D, R0, M)
                    else:
                        shape = (LINE_RCS[-1], M)
                else:
                    index = (slice(offset+LINE_RCS[dim]*M, offset+LINE_RCS[dim+1]*M),)
                    shape = (RS[dim], M)
            elif op == 'plane' or op == 'mat':
                if dim is None:
                    index = (slice(offset+LINE_RCS[-1]*M, offset_next),)
                    # assert all([R==R0 for R in RS]), "Expect equal resolution!"
                    if all([R==R0 for R in RS]):
                        shape = (D, R0, R0, M)
                    else:
                        shape = (PLANE_RCS[-1], M)
                else:
                    index = (slice(offset+LINE_RCS[-1]*M+PLANE_RCS[dim]*M, offset+LINE_RCS[-1]*M+PLANE_RCS[dim+1]*M),)
                    shape = (PLANE_RS[dim], M)
            else:
                raise RuntimeError(f"Invalid op={op}")
        
        elif tp == LoDType.NPlaneMul or tp == LoDType.NPlaneSum:
            if op is None:
                index = (slice(offset, offset_next),)
                shape = (size, M)
            elif op == 'plane' or op == 'mat':
                if dim is None:
                    index = (slice(offset, offset_next),)
                    # assert all([R==R0 for R in RS]), "Expect equal resolution!"
                    if all([R==R0 for R in RS]):
                        shape = (D, R0, R0, M)
                    else:
                        shape = (PLANE_RCS[-1], M)
                else:
                    index = (slice(offset+PLANE_RCS[dim]*M, offset+PLANE_RCS[dim+1]*M),)
                    shape = (PLANE_RS[dim], M)
            else:
                raise RuntimeError(f"Invalid op={op}")
        
        elif tp == LoDType.CP or tp == LoDType.CPfast:
            if op is None:
                index = (slice(offset, offset_next),)
                shape = (size, M)
            elif (op == 'line' or op == 'vec'):
                if dim is None:
                    index = (slice(offset, offset_next),)
                    # assert all([R==R0 for R in RS]), "Expect equal resolution!"
                    if all([R==R0 for R in RS]):
                        shape = (D, R0, M)
                    else:
                        shape = (LINE_RCS[-1], M)
                else:
                    index = (slice(offset+LINE_RCS[dim]*M, offset+LINE_RCS[dim+1]*M),)
                    shape = (RS[dim], M)
            else:
                raise RuntimeError(f"Invalid op={op}")
        
        elif tp == LoDType.Hash:
            index = (slice(offset, offset_next),)
            shape = (size, M)
        else:
            raise RuntimeError(f"Invalid tp={tp}")
    elif D == 4:
        if tp == LoDType.Dense:
            if op is None:
                index = (slice(offset, offset_next),)
                shape = (size, M)
            elif op == 'vol':
                index = (slice(offset, offset_next),)
                shape = (*RS, M)
            else:
                raise RuntimeError(f"Invalid op={op}")
        elif tp == LoDType.Hash:
            index = (slice(offset, offset_next),)
            shape = (size, M)
        else:
            raise NotImplementedError
    elif D == 2:
        if tp == LoDType.Dense:
            if op is None:
                index = (slice(offset, offset_next),)
                shape = (size, M)
            elif op == 'vol':
                index = (slice(offset, offset_next),)
                shape = (*RS, M)
            else:
                raise RuntimeError(f"Invalid op={op}")
        elif tp == LoDType.CP or tp == LoDType.CPfast:
            if op is None:
                index = (slice(offset, offset_next),)
                shape = (size, M)
            elif (op == 'line' or op == 'vec'):
                if dim is None:
                    index = (slice(offset, offset_next),)
                    # assert all([R==R0 for R in RS]), "Expect equal resolution!"
                    if all([R==R0 for R in RS]):
                        shape = (D, R0, M)
                    else:
                        shape = (LINE_RCS[-1], M)
                else:
                    index = (slice(offset+LINE_RCS[dim]*M, offset+LINE_RCS[dim+1]*M),)
                    shape = (RS[dim], M)
            else:
                raise RuntimeError(f"Invalid op={op}")
        elif tp == LoDType.Hash:
            index = (slice(offset, offset_next),)
            shape = (size, M)
        else:
            raise NotImplementedError
    elif D == 1:
        if tp == LoDType.Dense:
            if op is None:
                index = (slice(offset, offset_next),)
                shape = (size, M)
            elif op == 'vol':
                index = (slice(offset, offset_next),)
                shape = (*RS, M)
            else:
                raise RuntimeError(f"Invalid op={op}")
        elif tp == LoDType.Hash:
            index = (slice(offset, offset_next),)
            shape = (size, M)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    
    return index, shape

def get_level_param(params: torch.Tensor, lod_meta, l: int, op: str = None, dim: int = None) -> torch.Tensor:
    """ Get the param of the given level and op from the flattened params

    Args:
        params (torch.Tensor): [N,] tensor, Flattened params
        lod_meta (Any): LoTD meta data
        l (int): The given level number (0 to max_level-1)
        op (str, optional): Optionally for VM to specify `mat` or `vec`. Defaults to None.
        dim (int, optional): Optionally for multi-dim lotd types to specify `dim`. Defaults to None.

    Returns:
        torch.Tensor: The sliced / indexed param
    """
    index, shape = level_param_index_shape(lod_meta, l, op, dim)
    return params[index].view(shape)

def get_level_param_batched(params: torch.Tensor, lod_meta, bid: Union[int, List[int], slice, torch.Tensor] = slice(None), l: int = ..., op: str = None, dim: int = None) -> torch.Tensor:
    """ Get the param of the given level and op from the batched flattened params

    Args:
        params (torch.Tensor): [B, N,] tensor, Batched flattened params
        lod_meta (Any): LoTD meta data
        bid (Union[int, List[int], slice, torch.Tensor], optional): The given batch inds. Defaults to slice(None).
        l (int, optional): The given level number (0 to max_level-1). Defaults to ....
        op (str, optional): Optionally for VM to specify `mat` or `vec`. Defaults to None.
        dim (int, optional): Optionally for multi-dim lotd types to specify `dim`. Defaults to None.

    Returns:
        torch.Tensor: The sliced / indexed param
    """
    index, shape = level_param_index_shape(lod_meta, l, op, dim)
    prefix = params.data[bid, 0].shape
    return params[(bid, *index)].view(*prefix, *shape)


def param_vertices(res: Union[int, List[int]], dim: int = 3, is_forest=False, dtype=torch.float, device=None) -> torch.Tensor:
    """ Get the vertices of the given grid features (in normalized [-1,1] coordinates) 

    Args:
        res (Union[int, List[int]]): Grid resolution. Could be multi-dimensional or rectangular / cuboid / hyper-cuboids
        dim (int, optional): Space dimension of the grid. Defaults to 3.
        is_forest (bool, optional): Whether dealing with forest block grids. Defaults to False.
        device (torch.device, optional): Output torch device. Defaults to torch.device('cuda').
        dtype (torch.dtype, optional): Output torch dtype. Defaults to torch.float.

    Returns:
        torch.Tensor: The relative grid cell vertices coords
    """
    res = [res]*dim if isinstance(res, Number) else res
    # Needs to be [xyz]
    ls = [
        torch.linspace(-1, 1, r, device=device, dtype=dtype) * \
            (  (1.+1./(r-2.) ) if not is_forest else ( 1.-1./r)  )
        for r in res
    ]
    dense_coord = torch.stack(torch.meshgrid(ls, indexing='ij'), dim=-1)
    return dense_coord

# def rel_coords_to_param_interpolate_coords(coords: torch.Tensor, res: int, is_forest=False):
#     if not is_forest:
#         scale = (res-2.)/(res-1.)
#         return coords * scale # And, align_corners=True
#     else:
#         return coords # And, align_corners=False

def param_interpolate(param: torch.Tensor, rel_x: torch.Tensor, res: int, is_forest=False) -> torch.Tensor:
    """ Interpolate grid features on given normalized coords

    Args:
        param (torch.Tensor): The grid features. [B,R,M] for 1D, [B,R,R,M] for 2D, [B,R,R,R,M] for 3D
        rel_x (torch.Tensor): The given normalized coords to interpolate on. [B,...,1/2/3]
        res (int): Resolution of the given grid features.
        is_forest (bool, optional): Whether dealing with forest block grid features. Defaults to False.

    Returns:
        torch.Tensor: The interpolated grid features
    """
    dim = rel_x.shape[-1]
    if dim == 1:
        """
        param:  [B,R,M]
        rel_x:  [B,...,1]
        """
        assert param.dim()==3 and param.shape[0]==rel_x.shape[0] and [*param.shape[1:-1]] == [res]
        data_shape = rel_x.shape[1:-1]
        B, N = param.shape[0], prod(data_shape)
        if not is_forest:
            scale = (res-2.)/(res-1.)
            rel_x = (rel_x*scale).view(B,N)
            param = param.permute(0,2,1)
            ret = gridsample1d_by2d(param, rel_x, align_corners=True, padding_mode="zeros")
        else:
            rel_x = rel_x.view(B,N)
            param = param.permute(0,2,1)
            ret = gridsample1d_by2d(param, rel_x, align_corners=False, padding_mode="zeros")
        return ret.view(B,-1,N).permute(0,2,1).unflatten(1,data_shape)
    
    elif dim == 2:
        """
        param:  [B,R,R,M]
        rel_x:  [B,...,2]
        """
        assert param.dim()==4 and param.shape[0]==rel_x.shape[0] and [*param.shape[1:-1]] == [res,res]
        data_shape = rel_x.shape[1:-1]
        B, N = param.shape[0], prod(data_shape)
        if not is_forest:
            scale = (res-2.)/(res-1.)
            # NOTE: param's layout in LoTD-CUDA is y,x, while in Pytorch it's x,y
            rel_x = (rel_x * scale)[..., [1,0]].view(B,1,N,2)
            param = param.permute(0,3,1,2)
            ret = F.grid_sample(param, rel_x, align_corners=True, padding_mode="zeros")
        else:
            # NOTE: param's layout in LoTD-CUDA is y,x, while in Pytorch it's x,y
            rel_x = rel_x[..., [1,0]].view(B,1,N,2)
            param = param.permute(0,3,1,2)
            ret = F.grid_sample(param, rel_x, align_corners=False, padding_mode="zeros")
        return ret.view(B,-1,N).permute(0,2,1).unflatten(1,data_shape)
    
    elif dim == 3:
        """
        param:  [B,R,R,R,M]
        rel_x:  [B,...,3]
        """
        assert param.dim()==5 and param.shape[0]==rel_x.shape[0] and [*param.shape[1:-1]] == [res,res,res]
        data_shape = rel_x.shape[1:-1]
        B, N = param.shape[0], prod(data_shape)
        if not is_forest:
            scale = (res-2.)/(res-1.)
            # NOTE: param's layout in LoTD-CUDA is z,y,x, while in Pytorch it's x,y,z
            rel_x = (rel_x * scale)[..., [2,1,0]].view(B,1,1,N,3)
            param = param.permute(0,4,1,2,3)
            ret = F.grid_sample(param, rel_x, align_corners=True, padding_mode="zeros")
        else:
            # NOTE: param's layout in LoTD-CUDA is z,y,x, while in Pytorch it's x,y,z
            rel_x = rel_x[..., [2,1,0]].view(B,1,1,N,3)
            param = param.permute(0,4,1,2,3)
            ret = F.grid_sample(param, rel_x, align_corners=False, padding_mode="zeros")
        return ret.view(B, -1, N).permute(0,2,1).unflatten(1, data_shape)

class LoTD2ndGradGuard(nn.Module):
    def __init__(self, lod_meta, factor: float = 1.5, log_prefix: str = '', device=None) -> None:
        super().__init__()
        self.lod_meta = lod_meta
        self.factor = factor
        self.log_prefix = log_prefix
        self.logger = None
        self.level_n_feats_cumsum = [0, *np.cumsum(self.lod_meta.level_n_feats).tolist()]
        
        self.register_buffer("dldx_grad_norm_ema", torch.tensor([1.], dtype=torch.float, device=device), persistent=True)
        
        # self.register_buffer("grid_level_grad_max_ema", torch.ones([self.lod_meta.n_levels], dtype=torch.float, device=device), persistent=True)
        # self.register_buffer("dLdy_level_grad_max_ema", torch.ones([self.lod_meta.n_levels], dtype=torch.float, device=device), persistent=True)
        
        self.register_buffer("grid_level_grad_norm_ema", torch.ones([self.lod_meta.n_levels], dtype=torch.float, device=device), persistent=True)
        self.register_buffer("dLdy_level_grad_norm_ema", torch.ones([self.lod_meta.n_levels], dtype=torch.float, device=device), persistent=True)
    
    @torch.no_grad()
    def clip_grad_and_update_ema(self, dL_ddLdx: torch.Tensor, dy_dx: torch.Tensor, dL_dgrid: torch.Tensor, dL_ddLdy: torch.Tensor):        
        dlddldx_norm = dL_ddLdx.norm()
        dlddldx_ema = self.dldx_grad_norm_ema.copy_(dlddldx_norm.lerp(self.dldx_grad_norm_ema, 0.99))
        # if (factor:=self.factor) > 0:
        #     clip_norm_(dL_ddLdx, factor * dlddldx_ema.item())
        
        if dL_dgrid is not None:
            # dldgrid_max = torch.stack([get_level_param(dL_dgrid, self.lod_meta, l).data.abs().max() for l in range(self.lod_meta.n_levels)])
            # dldgrid_ema = self.grid_level_grad_max_ema.copy_(dldgrid_max.lerp(self.grid_level_grad_max_ema, 0.99))
            
            dldgrid_norm = torch.stack([get_level_param(dL_dgrid, self.lod_meta, l).data.norm() for l in range(self.lod_meta.n_levels)])
            dldgrid_ema = self.grid_level_grad_norm_ema.copy_(dldgrid_norm.lerp(self.grid_level_grad_norm_ema, 0.99))
            
            if (factor:=self.factor) > 0:
                for l in range(self.lod_meta.n_levels):
                    index, shape = level_param_index_shape(self.lod_meta, l)
                    val = factor * dldgrid_ema[l].item()
                    
                    # dL_dgrid[index].clip_(-val, val)
                    clip_norm_(dL_dgrid[index], val)
        
        if dL_ddLdy is not None:
            nf_cumsum = self.level_n_feats_cumsum
            
            # dlddldy_max = torch.stack([dL_ddLdy[..., nf_cumsum[l]:nf_cumsum[l+1]].abs().max() for l in range(self.lod_meta.n_levels)])
            # dlddldy_ema = self.dLdy_level_grad_max_ema.copy_(dlddldy_max.lerp(self.dLdy_level_grad_max_ema, 0.99))
            
            dlddldy_norm = torch.stack([dL_ddLdy[..., nf_cumsum[l]:nf_cumsum[l+1]].norm() for l in range(self.lod_meta.n_levels)])
            dlddldy_ema = self.dLdy_level_grad_norm_ema.copy_(dlddldy_norm.lerp(self.dLdy_level_grad_norm_ema, 0.99))
            
            if (factor:=self.factor) > 0:
                for l in range(self.lod_meta.n_levels):
                    val = 2.0 * factor * dlddldy_ema[l].item()

                    # dL_ddLdy[..., nf_cumsum[l]:nf_cumsum[l+1]].clip_(-val, val)
                    
                    clip_norm_(dL_ddLdy[..., nf_cumsum[l]:nf_cumsum[l+1]], val)
        
        if self.logger is not None:
            self.logger.add(self.log_prefix + ".dbg_grad_direct.dl_ddldx", "norm_ema", dlddldx_ema.item(), self.logger.last_step)
            for l in range(self.lod_meta.n_levels):
                self.logger.add(self.log_prefix + ".dbg_grad_direct.dl_dgrid", f"lv.{l}.norm_ema", dldgrid_ema[l].item(), self.logger.last_step)
                self.logger.add(self.log_prefix + ".dbg_grad_direct.dl_ddldy", f"lv.{l}.norm_ema", dlddldy_ema[l].item(), self.logger.last_step)

            self.log_2nd_grad(dL_ddLdx, dy_dx, dL_dgrid, dL_ddLdy)

    @torch.no_grad()
    def log_2nd_grad(self, dL_ddLdx: torch.Tensor, dy_dx: torch.Tensor, dL_dgrid: torch.Tensor, dL_ddLdy: torch.Tensor):
        if self.logger is not None:
            logger = self.logger
            nf_cumsum = self.level_n_feats_cumsum
            logger.add_nested_dict(self.log_prefix + ".dbg_grad_direct.dl_ddldx", "total", tensor_statistics(dL_ddLdx), logger.last_step)
            
            logger.add_nested_dict(self.log_prefix + ".dbg_grad_direct.dydx", "total", tensor_statistics(dy_dx), logger.last_step)
            for l in range(self.lod_meta.n_levels):
                logger.add_nested_dict(self.log_prefix + ".dbg_grad_direct.dydx", f"lv.{l}", tensor_statistics(dy_dx.view(-1, self.lod_meta.n_encoded_dims, 3)[..., nf_cumsum[l]:nf_cumsum[l+1], :]), logger.last_step)
            
            if dL_dgrid is not None:
                logger.add_nested_dict(self.log_prefix + ".dbg_grad_direct.dl_dgrid", "total", tensor_statistics(dL_dgrid), logger.last_step)
                for l, tp in enumerate(self.lod_meta.level_types):
                    tp = LoDType(tp)
                    if tp == LoDType.VectorMatrix:
                        stat_vec_grad = tensor_statistics(get_level_param(dL_dgrid.data, self.lod_meta, l, 'vec'))
                        stat_mat_grad = tensor_statistics(get_level_param(dL_dgrid.data, self.lod_meta, l, 'mat'))
                        logger.add_nested_dict(self.log_prefix + ".dbg_grad_direct.dl_dgrid", f"lv.{l}.vec", stat_vec_grad, logger.last_step)
                        logger.add_nested_dict(self.log_prefix + ".dbg_grad_direct.dl_dgrid", f"lv.{l}.mat", stat_mat_grad, logger.last_step)
                    else:
                        stat_grad = tensor_statistics(get_level_param(dL_dgrid.data, self.lod_meta, l))
                        logger.add_nested_dict(self.log_prefix + ".dbg_grad_direct.dl_dgrid", f"lv.{l}", stat_grad, logger.last_step)
            
            if dL_ddLdy is not None:
                logger.add_nested_dict(self.log_prefix + ".dbg_grad_direct.dl_ddldy", "total", tensor_statistics(dL_ddLdy), logger.last_step)
                for l in range(self.lod_meta.n_levels):
                    logger.add_nested_dict(self.log_prefix + ".dbg_grad_direct.dl_ddldy", f"lv.{l}", tensor_statistics(dL_ddLdy[..., nf_cumsum[l]:nf_cumsum[l+1]]), logger.last_step)

if __name__ == "__main__":
    def unit_test_interpolate(device=torch.device('cuda')):
        B = 7
        R = 13
        M = 4
        data_shape = [5,5]
        param = torch.randn([B, R, R, M], device=device)
        x = torch.rand([B, *data_shape, 2], device=device)
        param_interpolate(param, x, R)
    
    def test_dense_param_cycle_equivalence(device=torch.device('cuda')):
        B = 7
        R = 13
        M = 4
        
        #--------------- 1D param cycle equivalence test
        param = torch.randn([B, R, M], device=device)
        x = param_vertices(R, 1, is_forest=False, device=device).unsqueeze_(0).expand(B,R,1)
        param2 = param_interpolate(param, x, R, is_forest=False)
        print(torch.allclose(param, param2, atol=1.0e-5))
        
        x = param_vertices(R, 1, is_forest=True, device=device).unsqueeze_(0).expand(B,R,1)
        param3 = param_interpolate(param, x, R, is_forest=True)
        print(torch.allclose(param, param3, atol=1.0e-5))
        
        #--------------- 2D param cycle equivalence test
        param = torch.randn([B, R, R, M], device=device)
        x = param_vertices(R, 2, is_forest=False, device=device).unsqueeze_(0).expand(B,R,R,2)
        param2 = param_interpolate(param, x, R, is_forest=False)
        print(torch.allclose(param, param2, atol=1.0e-5))
        
        x = param_vertices(R, 2, is_forest=True, device=device).unsqueeze_(0).expand(B,R,R,2)
        param3 = param_interpolate(param, x, R, is_forest=True)
        print(torch.allclose(param, param3, atol=1.0e-5))

        #--------------- 3D param cycle equivalence test
        param = torch.randn([B, R, R, R, M], device=device)
        x = param_vertices(R, 3, is_forest=False, device=device).unsqueeze_(0).expand(B,R,R,R,3)
        param2 = param_interpolate(param, x, R, is_forest=False)
        print(torch.allclose(param, param2, atol=1.0e-5))
        
        x = param_vertices(R, 3, is_forest=True, device=device).unsqueeze_(0).expand(B,R,R,R,3)
        param3 = param_interpolate(param, x, R, is_forest=True)
        print(torch.allclose(param, param3, atol=1.0e-5))

    def test_nplane_param_cycle_equivalence(device=torch.device('cuda')):
        B = 1
        R = 13
        M = 4
        
        param = torch.randn([B,3,R,R,M], device=device)
        x = param_vertices(R, 2, is_forest=False, device=device).unsqueeze_(0).expand(B,R,R,2)
        # Define as [xyzxyz]
        x = x[..., [0,1,1,0,0,1]].view(B,R,R,2,3)
        # ... Some 3D stuff could happen to x here ...
        x = x.view(B,R,R,6)[..., [4,5,3,2,0,1]].view(B,R,R,3,2).permute(0,3,1,2,4) # [B,3,R,R,2]
        param2 = param_interpolate(param.view(B*3,R,R,M), x.view(B*3,R,R,2), R, is_forest=False)
        print(torch.allclose(param, param2, atol=1.0e-5))
        
        x = param_vertices(R, 2, is_forest=True, device=device).unsqueeze_(0).expand(B,R,R,2)
        # Define as [xyzxyz]
        x = x[..., [0,1,1,0,0,1]].view(B,R,R,2,3)
        # ... Some 3D stuff could happen to x here ...
        x = x.view(B,R,R,6)[..., [4,5,3,2,0,1]].view(B,R,R,3,2).permute(0,3,1,2,4) # [B,3,R,R,2]
        param3 = param_interpolate(param.view(B*3,R,R,M), x.view(B*3,R,R,2), R, is_forest=True)
        print(torch.allclose(param, param3, atol=1.0e-5))
    
    def test_param_getset(device=torch.device('cuda')):
        from icecream import ic
        from nr3d_lib.models.grid_encodings.lotd import generate_meta
        B = 7
        lod_meta = generate_meta(3, [4,8,16,32,64], [4,2,4,16,4], ["Dense","Dense","VM","NPlaneMul","CPfast"])
        ic(lod_meta.level_res)
        ic(lod_meta.level_n_feats)
        ic(lod_meta.level_types)
        ic(lod_meta.level_offsets)
        ic(lod_meta.level_n_params)
        ic(lod_meta.map_levels)
        ic(lod_meta.map_cnt)

        ic(lod_meta.n_levels)
        ic(lod_meta.n_pseudo_levels)
        ic(lod_meta.n_feat_per_pseudo_lvl)
        ic(lod_meta.n_dims_to_encode)
        ic(lod_meta.n_encoded_dims)
        ic(lod_meta.n_params)
        
        ic(lod_meta.interpolation_type)
        
        params = torch.zeros([B, lod_meta.n_params], dtype=torch.float, device=device)
        p1 = get_level_param_batched(params, lod_meta, slice(None), l=2)
        p2 = get_level_param_batched(params, lod_meta, l=4, dim=3)
        ic(p1.shape)
        ic(p2.shape)



    # unit_test_interpolate()
    # test_dense_param_cycle_equivalence()
    # test_nplane_param_cycle_equivalence()
    # test_param_getset()
    # test_generate_lotd_cfg()
    # test_generate_ngp_cfg()
    # test_lotd_annealer()
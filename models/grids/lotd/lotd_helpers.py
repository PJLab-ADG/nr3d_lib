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
    'auto_compute_lotd_cfg_deprecated', 
    'auto_compute_ngp_cfg', 
    'auto_compute_ngp4d_cfg', 
    'LoTD2ndGradGuard', 
    'LoTDAnnealer', 
]

import numpy as np
from math import prod
from numbers import Number
from typing import Literal, Union, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nr3d_lib.fmt import log
from nr3d_lib.fmt import colored_str, GREEN
from nr3d_lib.utils import check_to_torch, tensor_statistics

from nr3d_lib.models.utils import clip_norm_
from nr3d_lib.models.grids.utils import gridsample1d_by2d
from nr3d_lib.models.grids.lotd.lotd import LoDType

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
    
    RS = lod_meta.level_res_multi_dim[l]
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


def param_vertices(res: Union[int, List[int]], dim: int = 3, is_forest=False, device=torch.device('cuda'), dtype=torch.float) -> torch.Tensor:
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

def gen_ngp_cfg(
    min_res: int = 16, dim: int = 3, n_feats: int = 2, log2_hashmap_size: int = 19, 
    per_level_scale: float = 1.382, num_levels: int = 16
):
    hashmap_size = 2**log2_hashmap_size
    level_res = min_res * per_level_scale ** np.arange(num_levels)
    level_res = level_res.astype(int)
    level_n_feats = [n_feats] * num_levels
    level_types = [ ("Dense" if res ** dim <= hashmap_size else "Hash")  for res in level_res]
    return dict(lod_res=level_res.tolist(), lod_n_feats=level_n_feats, lod_types=level_types, hashmap_size=hashmap_size)

def auto_compute_ngp_cfg(
    stretch: Union[float, List[float]], target_num_params: int, 
    *, dim: int = 3, n_feats: int = 2, log2_hashmap_size: int = 19, min_res: int = 4, 
    per_level_scale: float = 1.382, max_num_levels: int = 128
):
    """ Automatically computes LoTD configs for [Dense --> Hash] configs

    Args:
        stretch (Union[float, List[float]], optional): 
            Spatial strech ratio or 3D/4D aspect ratios for cuboid / hyper-cuboid representations. 
            A single number would suggest using cubic representation. Defaults to 1.0.
        target_num_params (int): Target total number of params to generate.
        dim (int, optional): Space dimension. Defaults to 3.
        n_feats (int, optional): Feature width. Defaults to 2.
        log2_hashmap_size (int, optional): log_2^{} of the hashmap size. Defaults to 19.
        min_res (int, optional): Minimum resolution of the shortest side. Defaults to 4.
        per_level_scale (float, optional): Scale factor of the next level's resolution to current level's. Defaults to 1.382.
        max_num_levels (int, optional): Maximum number of layers. Defaults to 128.

    Returns:
        dict: Automatically computed LoTD configs. Composed of `lod_res`, `lod_n_feats`, `lod_types`, `hashmap_size`
        
    Algo:
        1. The parameter growth rate for Dense layers is 1.38^3 ~= 2.62
            -> The total number of parameters in all previous dense layers is roughly equivalent to one hash layer
            -> Determine the number of hash layers based on target_num_params and log2_hashmap_size
        
        2. The number of parameters in the last dense layer is approximately 1 / 2.5~2.6 of the hash layer parameters (i.e., roughly maintaining a constant growth rate for the total number of parameters in each layer)
            -> Determine the resolution of the last dense layer based on log2_hashmap_size
            -> Determine the number of dense layers based on the resolution of the last dense layer and min_res
        
        3. Determine all parameters based on the resolution of the last dense layer, the number of dense layers, the number of hash layers, and per_level_scale
    """
    
    stretch = [stretch]*dim if isinstance(stretch, Number) else stretch
    stretch = np.array(stretch)
    
    hashmap_size = 2**log2_hashmap_size
    hash_num_param = hashmap_size * n_feats
    dense_nparam_factor = (stretch / stretch.min()).prod()
    # per_level_param_scale = per_level_scale ** 3
    
    #----------- The total number of all dense levels should be roughly the same with one hash level    
    dense_last_num_param = hashmap_size / 2.5
    dense_last_min_res = int( (dense_last_num_param / dense_nparam_factor) ** (1/3))
    num_dense_levels = np.exp( np.log(dense_last_min_res/min_res) / per_level_scale)
    num_dense_levels = max(int(num_dense_levels + 1), 1)

    num_hash_levels = target_num_params / hash_num_param - 1
    num_hash_levels = max(int(num_hash_levels + 0.5), 0)

    num_levels = num_dense_levels + num_hash_levels
    if max_num_levels is not None:
        num_levels = min(num_levels, max_num_levels)
        num_hash_levels = num_levels - num_dense_levels
    
    dense_last_res = stretch / (stretch.min() / dense_last_min_res)
    
    ngp_res_dense = dense_last_res[..., None] / (per_level_scale ** np.arange(num_dense_levels))
    ngp_res_dense = ngp_res_dense[:, ::-1].T.astype(int) # NOTE: Convert to int at the last moment
    
    ngp_res_hash = dense_last_res[..., None] * (per_level_scale ** (np.arange(num_hash_levels)+1))
    ngp_res_hash = ngp_res_hash.T.astype(int)
    
    ngp_res_all = np.concatenate([ngp_res_dense, ngp_res_hash], axis=0)
    ngp_type_all = ["Dense"] * num_dense_levels + ["Hash"] * num_hash_levels
    ngp_n_feats_all = [n_feats] * num_levels
    
    ngp_num_param = ngp_res_dense.prod(axis=1).sum() * n_feats + num_hash_levels * hashmap_size * n_feats
    
    log.info(colored_str(f"NGP auto-computed config: layer resolutions: {ngp_res_all.tolist()}", GREEN))
    log.info(colored_str(f"NGP auto-computed config: layer types: {ngp_type_all}", GREEN))
    log.info(colored_str(f"NGP auto-computed config: layer n_feats: {ngp_n_feats_all}", GREEN))
    log.info(colored_str(f"NGP auto-computed config: expected num_params={target_num_params}; generated: {ngp_num_param} [{ngp_num_param/target_num_params:.2f}x]", GREEN))
    return dict(lod_res=ngp_res_all.tolist(), lod_n_feats=ngp_n_feats_all, lod_types=ngp_type_all, hashmap_size=hashmap_size)

def auto_compute_ngp4d_cfg(dim: int = 4, n_feats: int = 2, stretch: Union[float, List[float]] = 1.0,
                           target_num_params: int = 2**32, max_layers = 128,
                           min_dense_layers: int = 0, log2_hashmap_size: int = 19,
                           min_res_xyz: int = 4, min_res_w: int = 4, per_level_scale: float = 1.382
    ):
    """ Automatically computes LoTD configs for [Dense --> Hash] configs for NeRF++

    Args:
        dim (int, optional): Space dimension. Defaults to 4.
        n_feats (int, optional): Feature width. Defaults to 2.
        stretch (Union[float, List[float]], optional): 
            Spatial strech ratio or 3D/4D aspect ratios for cuboid / hyper-cuboid representations. 
            A single number would suggest using cubic representation. Defaults to 1.0.
        target_num_params (int, optional): Target total number of params to generate. Defaults to 2**32.
        max_layers (int, optional): Maximum number of layers. Defaults to 128.
        min_dense_layers (int, optional): Minimum number of dense layers. Defaults to 0.
        log2_hashmap_size (int, optional): log_2^{} of the hashmap size. Defaults to 19.
        min_res_xyz (int, optional): Minimum resolution of the shortest side of xyz part of xyzw. Defaults to 4.
        min_res_w (int, optional): Minimum resolution of w part of xyzw. Defaults to 4.
        per_level_scale (float, optional): Scale factor of the next level's resolution to current level's. Defaults to 1.382.

    Returns:
        dict: Automatically computed LoTD configs. Composed of `lod_res`, `lod_n_feats`, `lod_types`, `hashmap_size`
    """
    hashmap_size = 2**log2_hashmap_size
    stretch = np.array([stretch] * dim if isinstance(stretch, Number) else stretch)
    base_res = np.concatenate([
        min_res_xyz * stretch / stretch.min(),
        np.array([min_res_w], dtype=np.float32)
    ])
    num_params = 0
    layers_res = []
    layers_type = []

    for i in range(max_layers):
        layer_res = np.ceil(base_res).astype(np.int64)
        layer_num_grids = layer_res.prod()
        if layer_num_grids > hashmap_size and i >= min_dense_layers:
            layer_type = "Hash"
            layer_num_params = hashmap_size * n_feats
        else:
            layer_type = "Dense"
            layer_num_params = layer_num_grids * n_feats
        if num_params + layer_num_params > target_num_params:
            break
        layers_res.append(layer_res.tolist())
        layers_type.append(layer_type)
        num_params += layer_num_params
        base_res *= per_level_scale

    layers_n_feats = [n_feats] * len(layers_res)

    log.info(colored_str(f"NGP-4D auto-computed config: layer resolutions: {layers_res}", GREEN))
    log.info(colored_str(f"NGP-4D auto-computed config: layer types: {layers_type}", GREEN))
    log.info(colored_str(f"NGP-4D auto-computed config: layer n_feats: {layers_n_feats}", GREEN))
    log.info(colored_str(f"NGP-4D auto-computed config: totally {num_params} parameters " f"[{num_params/target_num_params:.2f}x]", GREEN))
    return dict(lod_res=layers_res, lod_n_feats=layers_n_feats, lod_types=layers_type,
                hashmap_size=hashmap_size)

def auto_compute_lotd_cfg_deprecated(
    stretch: Union[float, List[float]], target_num_params: int, 
    *, dim: int = 3, min_res: int = 4, max_n_feat: int = 8, min_n_feat: int = 2, 
    level_scale: float = 2, ratio_last_nparam: float = 0.3, ratio_dense_nparam: float = 0.2, ratio_dense_nlevel: float = 0.6) -> dict:
    """ Automatically computes LoTD configs for [Dense --> VM] configs

    Args:
        stretch (Union[float, List[float]]): 
            Spatial strech ratio or 3D/4D aspect ratios for cuboid / hyper-cuboid representations. 
            A single number would suggest using cubic representation.
        target_num_params (int): Target total number of params to generate.
        dim (int, optional): Space dimension. Defaults to 3.
        min_res (int, optional): Minimum resolution of the shortest side. Defaults to 4.
            TODO: Perhaps this should be changed to the resolution of the longest side, which might be more accurate to control? To be verified.
                But the previous configurations should also run correctly under these circumstances, so some LoTD configurations should be saved in the ckpt.
                Make this function a backup for the next run, and then start a new function.
            TODO: The resolution of smaller dense layers should be deduced from the last dense layer.
        max_n_feat (int, optional): Maximum feature width. Defaults to 8.
        min_n_feat (int, optional): Minimum feature width. Defaults to 2.
        level_scale (float, optional): Scale factor of the next level's resolution to current level's. Defaults to 2.
        ratio_last_nparam (float, optional): Proportion of n_params of the last VM level. Defaults to 0.3.
        ratio_dense_nparam (float, optional): Proportion of n_params of the dense levels. Defaults to 0.2.
        ratio_dense_nlevel (float, optional): Proportion of n_levels of the dense levels. Defaults to 0.6.

    Returns:
        dict: Automatically computed LoTD configs. Composed of lod_res, lod_n_feats, lod_types

    Algo:
        NOTE: For now, usually generates 0.7x of the requested `target_num_param`
        
        1. Assume the first layer LoDType == Dense, determine the smallest grid resolution and its number of parameters through min_res.
        2. Assume the last layer LoDType == VM, determine the required grid resolution for the last VM layer based on the set `ratio_last_nparam` (proportion of the number of parameters of the largest level).
        3. Determine the number of levels and the grid resolutions of each level based on the resolution of the last level (now only the lodtype and n_feat of each level need to be determined).
        4. Determine the number of layers of dense type and their configurations based on the set ratio_dense_nparam (proportion of the number of parameters of the dense layers).
        5. Place the remaining VM layers.
    """
    stretch = [stretch]*dim if isinstance(stretch, Number) else stretch
    stretch = np.array(stretch)
    
    plane_stretches = np.array([prod(stretch) / s for s in stretch])
    
    target = int(target_num_params * 1.5) # Usally generates 0.6x of requested `target`
    
    grid_size_lv0 = stretch.min() / min_res # stretch per grid
    res_lv0 = (stretch / grid_size_lv0).astype(int)
    
    #----------- Max level [LoDType==VM] consumes up to 40% of param pool
    # - The plane resolutions of grid: plane_stretches / grid_size_lvn^2
    # - To solve: grid_size_lvn
    # - Equation: sum(plane_stretches) / grid_size_lvn^2 * min_n_feat = 0.4 * target
    grid_size_lvn = np.sqrt(plane_stretches.sum() * min_n_feat / (ratio_last_nparam * target))
    
    up_factor = grid_size_lv0 / grid_size_lvn
    # NOTE: log_{ `level_scale` }^{ `up_factor` }
    #       Empirically, +0 will leads to only 0.4x~0.7x of num_params, +0.8 is just OK
    # num_levels = int(np.log(up_factor) / np.log(level_scale) + 0.8) + 1
    num_levels = max(int(np.log(up_factor) / np.log(level_scale)+0.5) + 1, 1)
    level_scale = up_factor ** (1./(num_levels-1))
    
    lotd_res_all = (res_lv0[..., None] * (level_scale ** np.arange(num_levels))).astype(int).T
    lotd_n_feats = [np.nan] * num_levels
    lotd_type_all = ["TODO"] * num_levels
    lotd_type_all[-1] = "VM"
    lotd_n_feats[-1] = min_n_feat

    accumu_num_param = 0
    accumu_lvl = 0

    #----------- [LoDType==Dense] consumes up to 20% of param pool
    for lvl in range(0, num_levels-1, 1):
        cur_res = lotd_res_all[lvl]
        
        cur_n_feat = (target*ratio_dense_nparam - accumu_num_param) / prod(cur_res) / 2
        cur_n_feat = np.clip(cur_n_feat, min_n_feat, max_n_feat)
        cur_n_feat = int( max(cur_n_feat // min_n_feat,1) * min_n_feat)
        
        cur_n_param = prod(cur_res) * cur_n_feat
        
        if (cur_n_param + accumu_num_param) > (ratio_dense_nparam * target) or \
            (lvl+1)/num_levels > ratio_dense_nlevel:
            break
        
        accumu_num_param += cur_n_param
        lotd_n_feats[lvl] = cur_n_feat
        lotd_type_all[lvl] = "Dense"
        accumu_lvl += 1
    
    
    #----------- [LoDType==VM]
    lvn_plane_size = [prod(lotd_res_all[-1])//r for r in lotd_res_all[-1]]
    lvn_n_param = sum(lvn_plane_size) * min_n_feat
    accumu_num_param += lvn_n_param
    
    for lvl in range(accumu_lvl, num_levels-1, 1):
        cur_res = lotd_res_all[lvl]
        
        plane_size = [prod(cur_res) // r for r in cur_res]
        
        cur_n_feat = (target - accumu_num_param) / sum(plane_size) / 2
        cur_n_feat = np.clip(cur_n_feat, min_n_feat, max_n_feat)
        cur_n_feat = int(max(cur_n_feat // min_n_feat,1) * min_n_feat)
        
        cur_n_param = sum(plane_size) * cur_n_feat
        
        if (cur_n_param + accumu_num_param) > target:
            break
        
        accumu_num_param += cur_n_param
        lotd_n_feats[lvl] = cur_n_feat
        lotd_type_all[lvl] = "VM"
        accumu_lvl += 1
    
    #----------- Fill all the remaining levels with VM (if any) 
    # Only happens when the above loop already exceeds the target num_params
    for lvl in range(accumu_lvl, num_levels-1, 1):
        cur_res = lotd_res_all[lvl]
        cur_n_feat = min_n_feat
        plane_size = [prod(cur_res) // r for r in cur_res]
        cur_n_param = sum(plane_size) * cur_n_feat
        
        accumu_num_param += cur_n_param
        lotd_n_feats[lvl] = cur_n_feat
        lotd_type_all[lvl] = "VM"

    print(colored_str(f"Expected num_params={target_num_params}; auto-computed LoTD config makes {accumu_num_param} [{accumu_num_param/target_num_params:.2f}x]", GREEN))
    return dict(lod_res=lotd_res_all.tolist(), lod_n_feats=lotd_n_feats, lod_types=lotd_type_all)

class LoTD2ndGradGuard(nn.Module):
    def __init__(self, lod_meta, factor: float = 1.5, log_prefix: str = '', device=torch.device('cuda')) -> None:
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
    def custom_grad_clip_step(self, dL_ddLdx: torch.Tensor, dy_dx: torch.Tensor, dL_dgrid: torch.Tensor, dL_ddLdy: torch.Tensor):        
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

class LoTDAnnealer(nn.Module):
    def __init__(
        self, 
        lod_n_feats: List[int], 
        type: Literal['hardmask', 'cosine'], 
        stop_it: int, start_it: int = 0, update_every: int=1, start_level: int = 0, 
        dtype=None, device=None # NOTE: dtype should be the same with lotd's output dtype
        ) -> None:
        super().__init__()
        
        lod_n_feats = check_to_torch(lod_n_feats, dtype=dtype, device=device)
        self.num_levels = len(lod_n_feats)
        self.register_buffer('lod_n_feats', lod_n_feats, persistent=False)
        self.register_buffer('window', torch.ones([lod_n_feats.numel()], dtype=dtype, device=device), persistent=False)
        self.register_buffer('level_arange', torch.arange(self.num_levels, dtype=dtype, device=device), persistent=False)
        
        self.start_it = int(start_it)
        self.stop_it = int(stop_it)
        self.update_every = int(update_every)
        self.it = self.stop_it # At anneal stop state by default.
        self.total_stages = (self.stop_it - self.start_it) // self.update_every
        # NOTE: At least -1 (no levels used;)
        #       max_level=0 means the 1st level will be used; max_level=-1 means no level used.
        self.start_level = max(min(int(start_level), self.num_levels-1), -1)
        if type == 'hardmask':
            self.window_fn = self._window_fn_hardmask
        elif type == 'cosine':
            self.window_fn = self._window_fn_cosine
        else:
            raise RuntimeError(f'Invalid anneal_type={type}')

    def set_iter(self, it: int):
        self.it = it

    def forward(self, it:int = None) -> Tuple[int, Union[None, torch.Tensor]]:
        it = self.it if it is None else it
        cur_stage = (it - self.start_it) // self.update_every
        alpha = min(1.0, max(0.0, cur_stage / self.total_stages))
        max_level, window = self.window_fn(alpha)
        return max_level, window

    def _window_fn_hardmask(self, alpha: float = 1.0):
        # From self.start_level (at least=-1) to self.num_levels-1
        length = (self.num_levels-1) - self.start_level
        max_level = self.start_level + min(int(alpha * length), length)
        return max_level, None
    
    def _window_fn_cosine(self, alpha: float = 1.0):
        # NOTE: when alpha=0, only the minimum level is active
        length = (self.num_levels-1) - self.start_level
        raw = self.start_level + alpha * length - self.level_arange + 1
        window = 0.5 * (1 + torch.cos(np.pi * torch.clip(raw, 0.0, 1.0) + np.pi))
        window = window.repeat_interleave(self.lod_n_feats)
        # From self.start_level (at least=-1) to self.num_levels-1
        max_level = (raw > 0).sum().item()-1
        return max_level, window

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
        from nr3d_lib.models.grids.lotd import generate_meta
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
    
    def test_generate_lotd_cfg(device=torch.device('cuda')):
        # 32 MiB of float params
        from icecream import ic
        from nr3d_lib.models.grids.lotd import LoTD
        lotd_cfg = auto_compute_lotd_cfg_deprecated([100., 20., 20.], dim=3, target_num_params=16*(2**20)//4, max_n_feat=8, min_res=16, min_n_feat=2)
        m = LoTD(3, **lotd_cfg, device=device, dtype=torch.float)
        print(m)
        
        lotd_cfg = auto_compute_lotd_cfg_deprecated([20., 20., 20.], dim=3, target_num_params=16*(2**20)//4, max_n_feat=8, min_res=8, min_n_feat=2, ratio_dense_nparam=0.5)
        m = LoTD(3, **lotd_cfg, device=device, dtype=torch.float)
        print(m)

        lotd_cfg = auto_compute_lotd_cfg_deprecated([20., 20., 20.], dim=3, target_num_params=16*(2**10)//4, max_n_feat=8, min_res=8, min_n_feat=2, ratio_dense_nparam=0.5)
        m = LoTD(3, **lotd_cfg, device=device, dtype=torch.float)
        print(m)

    def test_generate_ngp_cfg(device=torch.device('cuda')):
        from icecream import ic
        from nr3d_lib.models.grids.lotd import LoTD
        ngp_cfg = auto_compute_ngp_cfg([100., 20., 20.], 16*(2**20)//4, dim=3, n_feats=2, log2_hashmap_size=18, min_res=16)
        m = LoTD(3, **ngp_cfg, device=device, dtype=torch.float)
        ic(m)

        ngp_cfg = auto_compute_ngp_cfg([100., 20., 20.], 16*(2**20)//2, dim=3, n_feats=2, log2_hashmap_size=19, min_res=16)
        m = LoTD(3, **ngp_cfg, device=device, dtype=torch.float16)
        ic(m)

        ngp_cfg = auto_compute_ngp_cfg([149.458, 92.742, 39.2999], 40*(2**20)//2, dim=3, n_feats=2, log2_hashmap_size=20, min_res=16)
        m = LoTD(3, **ngp_cfg, device=device, dtype=torch.float16)
        ic(m)
        
        ngp_cfg = auto_compute_ngp4d_cfg(4, 2, [149.458, 92.742, 39.2999], 16*(2**20), min_res_xyz=16, min_res_w=4, log2_hashmap_size=20, per_level_scale=1.382)
        m = LoTD(4, **ngp_cfg, device=device, dtype=torch.float16)
        ic(m)
    
    def test_generate_cuboid_vs_cubic(device=torch.device('cuda')):
        from icecream import ic
        from nr3d_lib.models.grids.lotd import LoTD
        cfg1 = auto_compute_ngp_cfg([20, 4, 1], 16*(2**20), dim=3, n_feats=2, log2_hashmap_size=20, min_res=16)
        cfg2 = auto_compute_ngp_cfg(1, 16*(2**20), dim=3, n_feats=2, log2_hashmap_size=20, min_res=16)
        ic(len(cfg1['lod_n_feats']), len(cfg2['lod_n_feats']))
        print(np.array([200., 40., 10.]) / np.array(cfg1['lod_res'][-1]))
        print(np.array([200., 40., 10.]) / np.array(cfg2['lod_res'][-1]))

    def test_lotd_annealer(device=torch.device('cuda')):
        import matplotlib.pyplot as plt
        an1 = LoTDAnnealer([2, 4, 2, 3, 5], 'hardmask', start_it=10, stop_it=433, start_level=1, device=device)
        an2 = LoTDAnnealer([2, 4, 2, 3, 5], 'hardmask', start_it=10, stop_it=334, start_level=0, device=device)
        an3 = LoTDAnnealer([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'cosine', start_it=100, stop_it=443, start_level=2, device=device)
        
        lst_maxlvl_1 = []
        lst_maxlvl_2 = []
        lst_maxlvl_3 = []
        windows_3 = []
        iters = range(0, 1000)
        for it in iters:
            max_level, _ = an1(it)
            lst_maxlvl_1.append(max_level)
            
            max_level, _ = an2(it)
            lst_maxlvl_2.append(max_level)

            max_level, window = an3(it)
            lst_maxlvl_3.append(max_level)
            windows_3.append(window.data.cpu().numpy())
        windows_3 = np.stack(windows_3, axis=1)
        
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.plot(iters, lst_maxlvl_1, label='an1')
        plt.plot(iters, lst_maxlvl_2, label='an2')
        plt.plot(iters, lst_maxlvl_3, label='an3')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        for i, w in enumerate(windows_3):
            plt.plot(iters, w, label=f"lod{i}")
        plt.legend()
        plt.show()

    # unit_test_interpolate()
    # test_dense_param_cycle_equivalence()
    # test_nplane_param_cycle_equivalence()
    # test_param_getset()
    # test_generate_lotd_cfg()
    # test_generate_ngp_cfg()
    test_generate_cuboid_vs_cubic()
    # test_lotd_annealer()
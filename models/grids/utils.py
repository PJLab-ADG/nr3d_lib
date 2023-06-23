"""
@file   utils.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Utilities for (voxel)-grid-based neural network modules.
"""

import numpy as np
from enum import Enum
from typing import Union
from numbers import Number

import torch
import torch.nn.functional as F

def gridsample1d_by2d(input: torch.Tensor, grid: torch.Tensor, padding_mode: str = 'zeros', align_corners=False) -> torch.Tensor:
    """ 1D version of F.grid_sample by using 2D F.grid_sample
    Modifed from https://github.com/luo3300612/grid_sample1d

    Args:
        input (torch.Tensor): [B, C, M] 
        grid (torch.Tensor): [B, N]
        padding_mode (str, optional): `padding_mode` defined the same way in F.grid_sample. Defaults to 'zeros'.
        align_corners (bool, optional): `align_cornders` defined the same way in F.grid_sample. Defaults to False.

    Returns:
        torch.Tensor: [B, C, N]
    """
    shape = grid.shape
    input = input.unsqueeze(-1)  # batch_size * C * L_in * 1
    grid = grid.unsqueeze(1)  # batch_size * 1 * L_out
    grid = torch.stack([grid.new_zeros(grid.shape), grid], dim=-1) # NOTE: pseudo dim should be zeros !
    z = F.grid_sample(input, grid, padding_mode=padding_mode, align_corners=align_corners)
    C = input.shape[1]
    z = z.view(shape[0], C, shape[1])  # batch_size * C * L_out
    return z


#----------------------------------------------------
#--------------- Dealing with voxels ----------------
#----------------------------------------------------
class VoxelMode(Enum):
    XYZ=0
    ZYX=1
    PT3D=2

def offset_voxel(ref_coords: torch.Tensor=None, _0: float=0., _1: float=1., mode: VoxelMode=VoxelMode.ZYX) -> torch.Tensor:
    """ Calculate vertex coords from given voxel center / corner / etc.

    Args:
        ref_coords (torch.Tensor, optional): An optional input coords in voxel, could be center or corner or anything. Defaults to None.
        _0 (float, optional): Lower-bound offset value. Defaults to 0.
        _1 (float, optional): Upper-bound offset value. Defaults to 1.
        mode (VoxelMode, optional): Specific pattern of voxel vertex arrangement. Defaults to VoxelMode.ZYX.

    Returns:
        torch.Tensor: Voxel vertices
    """    
    if mode==VoxelMode.XYZ:
        """
        (row-major)row-col-depth  ==  x-y-z; 
        
         ↖ z
          `.
        (1) +---------+. (5)
            | ` .     |  ` .
            | (0) o---+-----+ (4) -> x
            |     |   |     |
        (3) +-----+---+. (7)|
            ` .   |     ` . |
            (2) ` +---------+ (6)
                  |
                  v y
        """
        off = torch.tensor([
            [_0, _0, _0],
            [_0, _0, _1],
            [_0, _1, _0],
            [_0, _1, _1],
            [_1, _0, _0],
            [_1, _0, _1],
            [_1, _1, _0],
            [_1, _1, _1]
        ])
    elif mode==VoxelMode.PT3D:
        """
         ↖ z
          `.
        (4) +---------+. (5)
            | ` .     |  ` .
            | (0) o---+-----+ (1) -> x
            |     |   |     |
        (7) +-----+---+. (6)|
            ` .   |     ` . |
            (3) ` +---------+ (2)
                  |
                  v y
        """
        off = torch.tensor([
            [_0, _0, _0],
            [_1, _0, _0],
            [_1, _1, _0],
            [_0, _1, _0],
            [_0, _0, _1],
            [_1, _0, _1],
            [_1, _1, _1],
            [_0, _1, _1],
        ])
    elif mode==VoxelMode.ZYX:
        """
        (row-major)row-col-depth  ==  z-y-x
        
         ↖ z
          `.
        (4) +---------+. (5)
            | ` .     |  ` .
            | (0) o---+-----+ (1) -> x
            |     |   |     |
        (6) +-----+---+. (7)|
            ` .   |     ` . |
            (2) ` +---------+ (3)
                  |
                  v y
        """
        off = torch.tensor([
            [_0, _0, _0],
            [_1, _0, _0],
            [_0, _1, _0],
            [_1, _1, _0],
            [_0, _0, _1],
            [_1, _0, _1],
            [_0, _1, _1],
            [_1, _1, _1]
        ])
    else:
        raise NotImplementedError
    if ref_coords is None:
        return off
    else:
        return off.to(ref_coords).reshape([*[1]*(ref_coords.dim()-1), 8, 3]) 

def offset_voxel_unflatten(ref_coords: torch.Tensor=None):
    off = torch.tensor(
        [[[[0, 0, 0],
          [0, 0, 1]],
         [[0, 1, 0],
          [0, 1, 1]]],
        [[[1, 0, 0],
          [1, 0, 1]],
         [[1, 1, 0],
          [1, 1, 1]]]])
    if ref_coords is None:
        return off
    else:
        return off.to(ref_coords).reshape([*[1]*(ref_coords.dim()-1), 2,2,2, 3]) 

def trilinear_voxel(rel_coords: torch.Tensor, verts_feats: torch.Tensor, _0: float=0., _1: float=1., mode=VoxelMode.ZYX) -> torch.Tensor:
    """ Trilinear interpolation given relative position and vertex features

    rel_coords: [*prefix, 3], range [0->1]
    verts_feats:[*prefix, 8, F]
    
    return:     [*prefix, F]

    Args:
        rel_coords (torch.Tensor): [..., 3] in range [_0,_1], relative 3D position in voxel
        verts_feats (torch.Tensor): [..., 8, F] voxel vertex features 
        _0 (float, optional): Lower bound of rel_coords. Defaults to 0.
        _1 (float, optional): Upper bound of rel_coords. Defaults to 1.
        mode (VoxelMode, optional): Specific pattern of voxel vertex arrangement of `verts_feats`. Defaults to VoxelMode.ZYX.

    Returns:
        torch.Tensor: [..., F] The interpolated features on the given relative position
    """
    p = rel_coords.unsqueeze(-2)
    q = offset_voxel(rel_coords, _0=_0, _1=_1, mode=mode)
    
    # [*prefix, 8, 1]
    weights = (p * q + (1 - p) * (1 - q)).prod(dim=-1, keepdim=True)
    feats = (weights * verts_feats).sum(-2)
    return feats

def points_to_corners(
    points: Union[torch.Tensor, np.ndarray], 
    _0: Number=None, _1: Number=None, spacing: Union[torch.Tensor, np.ndarray, Number]=None):
    
    if _0 is None:
        _0 = 0
    if _1 is None:
        _1 = 1
    """
    (row-major)row-col-depth  ==  x-y-z; 
    
    ↖ z
      `.
    (1) +---------+. (5)
        | ` .     |  ` .
        | (0) o---+-----+ (4) -> x
        |     |   |     |
    (3) +-----+---+. (7)|
        ` .   |     ` . |
        (2) ` +---------+ (6)
              |
              v y
    """
    if isinstance(points, torch.Tensor):
        off = torch.tensor([
            [_0, _0, _0],
            [_0, _0, _1],
            [_0, _1, _0],
            [_0, _1, _1],
            [_1, _0, _0],
            [_1, _0, _1],
            [_1, _1, _0],
            [_1, _1, _1]
        ], device=points.device, dtype=points.dtype)
    elif isinstance(points, np.ndarray):
        off = np.array([
            [_0, _0, _0],
            [_0, _0, _1],
            [_0, _1, _0],
            [_0, _1, _1],
            [_1, _0, _0],
            [_1, _0, _1],
            [_1, _1, _0],
            [_1, _1, _1]
        ], dtype=points.dtype)
    else:
        raise RuntimeError(f"Invalid `points` type={points}")
    
    if spacing is not None:
        off = off * spacing
    return points[..., None, :] + off

# def offset_patch(ref_coords: torch.Tensor=None):
#     off = torch.tensor([
#         [0., 0.],
#         [0., 1.],
#         [1., 0.],
#         [1., 1.],
#     ])
#     if ref_coords is None:
#         return off
#     else:
#         return off.to(ref_coords).reshape([*[1]*(ref_coords.dim()-1), 4, 3])  

# def bilinear_patch(rel_coords: torch.Tensor, verts_feats: torch.Tensor):
#     """
#     rel_coords: [*prefix, 2], range [0->1]
#     verts_feats:[*prefix, 4, F]
    
#     return:     [*prefix, F]
#     """
#     p = rel_coords.unsqueeze(-2)
#     q = offset_patch(rel_coords)

#     # [*prefix, 4, 1]
#     weights = (p * q + (1 - p) * (1 - q)).prod(dim=-1, keepdim=True)
#     feats = (weights * verts_feats).sum(-2)
#     return feats   


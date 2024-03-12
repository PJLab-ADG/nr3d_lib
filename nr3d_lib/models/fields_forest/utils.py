"""
@file   utils.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Utilities functions for forest of fields.
"""

import numpy as np
from math import sqrt
from typing import Union

import torch

from nr3d_lib.utils import check_to_torch
from nr3d_lib.graphics.cameras import pinhole_view_frustum, sphere_inside_frustum
from nr3d_lib.models.grid_encodings.utils import points_to_corners

def prepare_dense_grids(
    aabb: Union[torch.Tensor, np.ndarray], 
    *, block_radius: float = None, should_force_to_power_of_two=True, split_level: int = None, dilate_ratio: float = 0.5, overlap: float = 0
    ):
    if overlap != 0:
        raise NotImplementedError("`overlap` is currently not supported")
    
    aabb = check_to_torch(aabb)
    
    if block_radius is None:
        assert should_force_to_power_of_two and (split_level is not None)
        block_radius = float((aabb[1] - aabb[0]).max().item()) / (2**split_level + 2*dilate_ratio)
    
    world_block_size = block_radius * 2
    world_origin = aabb[0] - world_block_size * dilate_ratio
    aabb_max = aabb[1] + world_block_size * dilate_ratio
    
    resolution = ((aabb_max - world_origin) / (block_radius * 2) + 0.5).long() # Round to the closest integer
    # resolution = ((aabb_max - world_origin) / (block_radius * 2)).ceil().long() # Round to the ceiling integer
    if should_force_to_power_of_two:
        # Find the next power of two. 
        dim = int(torch.argmax(aabb_max - world_origin))
        level = int(np.ceil(np.log2(resolution[dim].item()))) if split_level is None else split_level
        world_block_size = float((aabb_max[dim] - world_origin[dim]).item() / (2**level))
        
        # Set new world_origin, aabb_max, resolution
        world_origin = aabb[0] - world_block_size * dilate_ratio
        aabb_max = aabb[1] + world_block_size * dilate_ratio
        resolution = ((aabb_max - world_origin) / world_block_size + 0.5).long() # Round to the closest integer
        # resolution = ((aabb_max - world_origin) / world_block_size).ceil().long() # Round to the ceiling integer
        resolution[dim] = 2**level # In case of floating number punctuations
    else:
        level = None
        world_block_size = (block_radius * 2)
    
    return resolution, world_origin, world_block_size, level

def split_block_on_continuous_waypoint_2d(
    tracks: np.ndarray, 
    *, block_radius: float = None, should_force_to_power_of_two=True, split_level: int = None, overlap: float = 0, device=None):
    assert tracks.shape[-1] == 2 and len(tracks.shape)==2, f"Expect tracks to be an array of shape [N, 2], while current={tracks.shape}"  

def split_block_on_continuous_waypoint_3d(
    tracks: np.ndarray, 
    *, block_radius: float = None, should_force_to_power_of_two=True, split_level: int = None, overlap: float = 0, device=None):
    assert tracks.shape[-1] == 3 and len(tracks.shape)==2, f"Expect tracks to be an array of shape [N, 3], while current={tracks.shape}"

def split_block_on_waypoints(
    tracks: Union[np.ndarray, torch.Tensor], 
    *, block_radius: float = None, should_force_to_power_of_two=True, split_level: int = None, overlap: float = 0, device=None):
    """
    TODO:
    - What if block_radius is so small such that the waypoint can miss some of the occ grid (might only be solved with continuous_waypoint)
    - Consider overlap and output data structure
    - Need to consider frustum. Only waypoints is not OK
    """
    
    tracks = check_to_torch(tracks, device=device, dtype=torch.float)
    aabb = torch.stack([tracks.min(0).values, tracks.max(0).values], 0)
    
    resolution, world_origin, world_block_size, level = prepare_dense_grids(
        aabb, block_radius=block_radius, should_force_to_power_of_two=should_force_to_power_of_two, split_level=split_level, overlap=overlap)
    
    occ_grid = torch.zeros(resolution.tolist(), dtype=torch.bool)
    
    inds = points_to_corners(((tracks-world_origin) / world_block_size), -0.5, 0.5).reshape(-1,tracks.shape[-1]).long().clip(resolution.new_tensor([0]), resolution-1)
    occ_grid[tuple(inds.t())] = 1

    block_ks = occ_grid.nonzero().long()
    
    return block_ks, world_origin, world_block_size, level

def split_block_on_cameras_3d(
    c2ws: Union[np.ndarray, torch.Tensor], intrs: Union[np.ndarray, torch.Tensor], far_clip: float, 
    *, block_radius: float = None, should_force_to_power_of_two=True, split_level: int = None, overlap: float = 0, device=None):

    if overlap != 0:
        raise NotImplementedError("`overlap` is currently not supported")

    c2ws = check_to_torch(c2ws, device=device, dtype=torch.float)
    intrs = check_to_torch(intrs, device=device, dtype=torch.float)
    fx, fy, cx, cy = intrs[..., 0, 0], intrs[..., 1, 1], intrs[..., 0, 2], intrs[..., 1, 2]
    
    # [N_frames, N_planes, 4]
    frustums = pinhole_view_frustum(c2ws, cx, cy, fx, fy, far_clip=far_clip)
    
    tracks = c2ws[..., :3, 3]
    resolution, world_origin, world_block_size, level = prepare_dense_grids(
        torch.stack([tracks.min(0).values-far_clip, tracks.max(0).values+far_clip], 0), 
        block_radius=block_radius, should_force_to_power_of_two=should_force_to_power_of_two, split_level=split_level, overlap=overlap)
    
    gidx = torch.stack(torch.meshgrid([torch.arange(r, device=device) for r in resolution], indexing='ij'), -1).reshape(-1,3)
    block_centers = world_origin + world_block_size * (gidx+0.5)
    
    # [N_blocks, 4]
    block_bounding_spheres = torch.zeros([gidx.shape[0], 4], device=device, dtype=torch.float)
    block_bounding_spheres[:, :3] = block_centers
    block_bounding_spheres[:, 3] = sqrt(3) * world_block_size/2.
    
    # [1, N_blocks, 4] [N_frames, N_planes, 4] -> [N_blocks, N_frames]
    inside = sphere_inside_frustum(block_bounding_spheres.unsqueeze(0), frustums, holistic=False, normalized=True)
    valid_block = inside.any(1)
    valid_block_inds = valid_block.nonzero().long()[..., 0]
    
    block_ks = gidx[valid_block_inds]
    return block_ks, world_origin, world_block_size, level
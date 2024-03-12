"""
@file   batched.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Handles one batch of block space
"""

__all__ = [
    'BatchedBlockSpace', 
]

import numpy as np
from typing import List, Tuple, Union

import torch
import torch.nn as nn

from nr3d_lib.utils import check_to_torch
from nr3d_lib.graphics.raytest import ray_box_intersection_fast_float_nocheck

class BatchedBlockSpace(nn.Module):
    def __init__(
        self, 
        bounding_size: float = None, aabb = None, 
        dtype=torch.float, device=None) -> None:
        super().__init__()
        
        self.dtype = dtype
        
        if aabb is None:
            # `bounding_size` is only taken care of when `aabb` is not specified
            bounding_size = 2.0 if bounding_size is None else bounding_size
            hbs = bounding_size / 2.
            aabb = [[-hbs, -hbs, -hbs], [hbs, hbs, hbs]]
        
        self.register_buffer('aabb', check_to_torch(aabb, dtype=dtype, device=device), persistent=True)

    @property
    def device(self) -> torch.device:
        return self.aabb.device

    @property
    def center(self):
        return (self.aabb[1] + self.aabb[0]) / 2.

    @property
    def radius3d(self):
        return (self.aabb[1] - self.aabb[0]) / 2.

    @property
    def radius3d_original(self):
        return self.radius3d # Never shrink

    def get_bounding_volume(self) -> torch.Tensor:
        return torch.cat([self.center, self.radius3d], dim=-1)

    def set_condition(self, *args, **kwargs):
        pass
    
    def clean_condition(self):
        pass

    """ Current batched operations """
    def cur_batch__unnormalize_coords(self, coords: torch.Tensor, bidx: torch.LongTensor = None):
        return coords * self.radius3d + self.center

    def cur_batch__normalize_coords(self, world_coords: torch.Tensor, bidx: torch.LongTensor = None):
        # To [-1,1] range.
        # if (aabb:=self.aabb) is not None:
        #     coords = (coords - (aabb[0]+aabb[1])/2.) / ((aabb[1]-aabb[0])/2.)
        coords = (world_coords - self.center) / self.radius3d
        return coords

    def cur_batch__normalize_rays(self, rays_o: torch.Tensor, rays_d: torch.Tensor):
        """
        Such that [new_rays_o + new_rays_d * real_depth] is directly in range [-1,1]
        NOTE: the norm of rays_d has changed.
        """
        # if (aabb:=self.aabb) is not None:
        #     scene_origin, scene_scale = (aabb[0]+aabb[1])/2., ((aabb[1]-aabb[0])/2.)
        #     rays_o, rays_d = (rays_o - scene_origin) / scene_scale, rays_d / scene_scale
        rays_o, rays_d = (rays_o - self.center) / self.radius3d, rays_d / self.radius3d
        return rays_o, rays_d

    def cur_batch__sample_pts_uniform(
        self, batch_size: int, num_pts_per_batch: int) -> Tuple[torch.Tensor, torch.LongTensor]:
        x = torch.empty([batch_size, num_pts_per_batch, 3], dtype=self.dtype, device=self.device).uniform_(-1,1)
        bidx = torch.arange(batch_size, dtype=torch.long, device=self.device).unsqueeze(-1).expand(batch_size,num_pts_per_batch)
        return x, bidx

    def cur_batch__ray_test(
        self, rays_o: torch.Tensor, rays_d: torch.Tensor, near=None, far=None, 
        return_rays=True, normalized=False, compact_batch=False, **extra_ray_data):
        assert rays_o.dim() == rays_d.dim() == 3 # [B, N_rays, 3]
        if not normalized: rays_o, rays_d = self.cur_batch__normalize_rays(rays_o, rays_d)
        """
        Ray intersection with valid regions

        rays_o:     [B, N_rays, 3]
        rays_d:     [B, N_rays, 3]
        near:       [B, N_rays] or float or None
        far:        [B, N_rays] or float or None
        """
        with torch.no_grad():
            B, N_rays, _ = rays_o.shape
            device = rays_o.device
            near_, far_ = ray_box_intersection_fast_float_nocheck(rays_o, rays_d, -1., 1.)
            if near is not None: near_.clamp_min_(near) # = max(t_near, near)
            if far is not None: far_.clamp_max_(far) # = min(t_far, far)
            check1, check2 = (far_ > near_), (far_ > (0 if near is None else near))
            mask = check1 & check2 & (True if far is None else near_ < far)
        
        # NOTE: Make sure that the returned `ridx` is consecutive. 
        #       - Necessary for rendering multiple batched objects in a single scene
        if not compact_batch:
            mask = mask.t()
            ridx, bidx = mask.nonzero(as_tuple=True)
            num_rays = ridx.numel()
            full_bidx = bidx
            ret = dict(
                num_rays=num_rays,
                rays_inds=ridx, 
                rays_bidx=bidx, 
                full_bidx_map=torch.arange(B, device=device, dtype=torch.long), # From compact bidx to full
                rays_full_bidx=full_bidx, 
            )
        else:
            used_batches = mask.any(dim=-1)
            full_bidx_map = used_batches.nonzero().long()[..., 0]
            ridx, bidx = mask[full_bidx_map].t().nonzero(as_tuple=True)
            num_rays = ridx.numel()
            # mask = mask.t()
            full_bidx = full_bidx_map[bidx]
            ret = dict(
                num_rays=num_rays,
                rays_inds=ridx, 
                rays_bidx=bidx,
                full_bidx_map=full_bidx_map,  # From compact bidx to full
                rays_full_bidx=full_bidx
            )
        
        inds = (full_bidx,ridx)
        ret.update(near=near_[inds], far=far_[inds])
        ret.update({k: v[inds] if isinstance(v, torch.Tensor) else v for k, v in extra_ray_data.items()})
        if return_rays:
            ret.update(rays_o=rays_o[inds], rays_d=rays_d[inds])
        return ret

"""
TODO    In future, consider replacing the `representations` in `neuralgen` with the format in `attributes`;
        Pass `pose` in as an argument;
    [*] Or specify that: 
        `space` is only responsible for the valid region of the network representation of an object after `scale`, `pose`, etc.; 
        The handling of object-level `pose`, `scale`, etc. is delegated to the scene graph level scene organizer; 
"""
# class BatchedBlockSpaceInputPoseScale(object):

#     def __init__(self) -> None:
#         pass

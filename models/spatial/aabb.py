"""
@file   aabb.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Handles a single block AABB (Axis-Aligned-Bounding-Box) space
"""

from typing import Literal

import torch
import torch.nn as nn

from nr3d_lib.utils import check_to_torch
from nr3d_lib.render.raytest import ray_box_intersection_fast_float_nocheck
from nr3d_lib.render.raysample import batch_sample_step_linear, interleave_sample_step_linear_in_nearfar, interleave_sample_step_wrt_depth_in_nearfar

class AABBSpace(nn.Module):
    """
    Single block AABB space
    """
    def __init__(
        self, 
        bounding_size: float = None, aabb = None, 
        dtype=torch.float, device=torch.device('cuda')) -> None:
        super().__init__()
        
        self.dtype = dtype
        self.device = device
        
        if aabb is None:
            # `bounding_size` is only taken care of when `aabb` is not specified
            bounding_size = 2.0 if bounding_size is None else bounding_size
            hbs = bounding_size / 2.
            aabb = [[-hbs, -hbs, -hbs], [hbs, hbs, hbs]]
        
        self.register_buffer('aabb', check_to_torch(aabb, dtype=dtype, device=device), persistent=True)
        self.register_buffer('scale0', self.scale.data.clone(), persistent=True) # Record the original scale in case of shrinking

    @property
    def origin(self):
        return (self.aabb[1] + self.aabb[0]) / 2.

    @property
    def scale(self):
        # To [-1,1] range.
        return self.extent / 2.
    
    @property
    def extent(self):
        return self.aabb[1] - self.aabb[0]

    @property
    def shrink_scale(self):
        # This is to keep nablas unchanged if shrink happened in the middle of a training.
        return self.scale / self.scale0

    @property
    def stretch(self):
        aabb = self.aabb
        return aabb[1] - aabb[0]

    @property
    def diameter(self):
        aabb = self.aabb
        return (aabb[1] - aabb[0]).norm(dim=-1) # Full length of diagonal

    @property
    def radius(self):
        # return gmean(self.scale).item()
        aabb = self.aabb
        return (aabb[1] - aabb[0]).norm(dim=-1)/2. # Half length of diagonal

    def unnormalize_coords(self, coords: torch.Tensor):
        return coords * self.scale + self.origin

    def normalize_coords(self, world_coords: torch.Tensor):
        # To [-1,1] range.
        # if (aabb:=self.aabb) is not None:
        #     coords = (coords - (aabb[0]+aabb[1])/2.) / ((aabb[1]-aabb[0])/2.)
        coords = (world_coords - self.origin) / self.scale
        return coords

    def normalize_rays(self, rays_o: torch.Tensor, rays_d: torch.Tensor):
        """
        Such that [new_rays_o + new_rays_d * real_depth] is directly in range [-1,1]
        NOTE: the norm of rays_d has changed.
        """
        # if (aabb:=self.aabb) is not None:
        #     scene_origin, scene_scale = (aabb[0]+aabb[1])/2., ((aabb[1]-aabb[0])/2.)
        #     rays_o, rays_d = (rays_o - scene_origin) / scene_scale, rays_d / scene_scale
        rays_o, rays_d = (rays_o - self.origin) / self.scale, rays_d / self.scale
        return rays_o, rays_d

    def uniform_sample_points(self, num_samples: int):
        return torch.empty([num_samples, 3], dtype=self.dtype, device=self.device).uniform_(-1,1)

    def ray_test(
        self, rays_o: torch.Tensor, rays_d: torch.Tensor, near=None, far=None,
        return_rays=True, normalized=False, **extra_ray_data):
        if not normalized: rays_o, rays_d = self.normalize_rays(rays_o, rays_d)
        with torch.no_grad():
            near_, far_ = ray_box_intersection_fast_float_nocheck(rays_o, rays_d, -1., 1.)
            if near is not None: near_.clamp_min_(near) # = max(t_near, near)
            if far is not None: far_.clamp_max_(far) # = min(t_far, far)
            check1, check2 = (far_ > near_), (far_ > (0 if near is None else near))
            mask = check1 & check2 & (True if far is None else near_ < far)
            ridx = mask.nonzero().long()[..., 0]
            ret = dict(num_rays=ridx.shape[0], ray_inds=ridx, near=near_[ridx], far=far_[ridx])
            ret.update({k: v[ridx] if isinstance(v, torch.Tensor) else v for k, v in extra_ray_data.items()})
        if return_rays:
            ret.update(rays_o=rays_o[ridx], rays_d=rays_d[ridx])
        return ret
    
    def contains(self, pts: torch.Tensor):
        return torch.logical_and(pts >= self.aabb[0], pts < self.aabb[1]).all(-1)
    
    def _ray_march_ray_batch(
        self, 
        rays_o: torch.Tensor, rays_d: torch.Tensor, near: torch.Tensor, far: torch.Tensor, *, 
        perturb=False, step_mode: Literal['linear', 'depth'], num_marched=256, **step_kwargs):
        """
        near, far is the ray-tested near, fars.
        """
        if step_mode == 'linear':
            depth_samples, deltas = batch_sample_step_linear(near, far, num_samples=num_marched, perturb=perturb, return_dt=True, **step_kwargs)
        elif step_mode == 'depth':
            depth_samples, deltas = batch_sample_step_linear(near, far, num_samples=num_marched, perturb=perturb, return_dt=True, **step_kwargs)
        else:
            raise RuntimeError(f"Invalid step_mode={step_mode}")
        samples = torch.addcmul(rays_o.unsqueeze(-2), rays_d.unsqueeze(-2), depth_samples.unsqueeze(-1))
        return samples, depth_samples, deltas
    
    def _ray_march_ray_step_in_nearfar(
        rays_o: torch.Tensor, rays_d: torch.Tensor, near: torch.Tensor, far: torch.Tensor, *, 
        perturb=False, step_mode: Literal['linear', 'depth'], **step_kwargs):
        
        if step_mode == 'linear':
            depth_samples, deltas, ridx, pack_infos = interleave_sample_step_linear_in_nearfar(near, far, perturb=perturb, **step_kwargs)
        elif step_mode == 'depth':
            depth_samples, deltas, ridx, pack_infos = interleave_sample_step_wrt_depth_in_nearfar(near, far, perturb=perturb, **step_kwargs)
        else:
            raise RuntimeError(f"Invalid step_mode={step_mode}")
        
        samples = torch.addcmul(rays_o.index_select(0, ridx), rays_d.index_select(0, ridx), depth_samples.unsqueeze(-1))
        return samples, depth_samples, deltas, ridx, pack_infos

    @torch.no_grad()
    def rescale_volume(self, new_aabb: torch.Tensor, reset_scale=False):
        self.aabb = new_aabb
        if reset_scale:
            self.scale0 = self.scale
    
    def extra_repr(self) -> str:
        return f"aabb={self.aabb}"
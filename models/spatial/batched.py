"""
@file   batched.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Handles one batch of block space
"""

import torch
import torch.nn as nn

from nr3d_lib.geometry import gmean
from nr3d_lib.render.raytest import ray_box_intersection_fast_float_nocheck

class BatchedBlockSpace(nn.Module):
    def __init__(
        self, 
        bounding_size: float = None, aabb = None, 
        device=torch.device('cuda'), dtype=torch.float) -> None:

        super().__init__()

        self.dtype = dtype
        self.device = device
        assert (bounding_size is None) or (aabb is None), "Only specify one of `bounding_size` and `aabb`"
        
        if bounding_size is not None:
            hbs = bounding_size / 2.
            aabb = [[-hbs, -hbs, -hbs], [hbs, hbs, hbs]]
        elif aabb is not None:
            pass
        else:
            aabb = [[-1,-1,-1], [1,1,1]]

        # NOTE: Code convention: if any of the three is to be changed, the other two should also be correspondingly adjusted.
        self.register_buffer('aabb', torch.tensor(aabb, dtype=dtype, device=device), persistent=True) 
        # To [-1,1]
        self.scale = (self.aabb[1] - self.aabb[0])/2. 
        self.origin = (self.aabb[1] + self.aabb[0])/2.

    @property
    def radius(self):
        return gmean(self.scale).item()

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

    def uniform_sample_points(self, B: int, num_samples: int):
        return torch.empty([B, num_samples, 3], dtype=self.dtype, device=self.device).uniform_(-1,1)

    def ray_test(
        self, rays_o: torch.Tensor, rays_d: torch.Tensor, near=None, far=None, 
        return_rays=True, normalized=False, compact_batch=False, **extra_ray_data):
        assert rays_o.dim() == rays_d.dim() == 3 # [B, N_rays, 3]
        if not normalized: rays_o, rays_d = self.normalize_rays(rays_o, rays_d)
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
            #       - Necessary when rendering multiple batched objects in a single scene
            if not compact_batch:
                mask = mask.t()
                ridx, bidx = mask.nonzero(as_tuple=True)
                num_rays = ridx.numel()
                full_bidx = bidx
                ret = dict(
                    num_rays=num_rays,
                    ray_inds=ridx, batch_inds=bidx, 
                    full_batch_ind_map=torch.arange(B, device=device, dtype=torch.long), # From compact bidx to full
                    full_batch_inds=full_bidx, 
                )
            else:
                used_batches = mask.any(dim=-1)
                full_batch_ind_map = used_batches.nonzero().long()[..., 0]
                ridx, bidx = mask[full_batch_ind_map].t().nonzero(as_tuple=True)
                num_rays = ridx.numel()
                # mask = mask.t()
                full_bidx = full_batch_ind_map[bidx]
                ret = dict(
                    num_rays=num_rays,
                    ray_inds=ridx, batch_inds=bidx,
                    full_batch_ind_map=full_batch_ind_map,  # From compact bidx to full
                    full_batch_inds=full_bidx
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
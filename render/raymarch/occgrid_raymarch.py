"""
@file   occgrid_raymarch.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Ray marching on occcupancy grids.
        CUDA Backends are modified from https://github.com/KAIR-BAIR/nerfacc
"""

from enum import Enum
from operator import itemgetter
from typing import Literal, Union

import torch

from nr3d_lib.fmt import log
from nr3d_lib.render.pack_ops import get_pack_infos_from_boundary, packed_diff, mark_pack_boundaries
from nr3d_lib.render.raymarch import dataclass_raymarch_ret, dataclass_batched_raymarch_ret, dataclass_forest_raymarch_ret

import nr3d_lib_bindings._occ_grid as _backend

class ContractionType(Enum):
    AABB = int(_backend.ContractionType.AABB)
    UN_BOUNDED_TANH = int(_backend.ContractionType.UN_BOUNDED_TANH)
    UN_BOUNDED_SPHERE = int(_backend.ContractionType.UN_BOUNDED_SPHERE)

def occgrid_raymarch(
    occ_grid: torch.BoolTensor,
    rays_o: torch.Tensor, rays_d: torch.Tensor, near: Union[torch.Tensor, float], far: Union[torch.Tensor, float], 
    *, constraction: Literal['aabb', 'tanh', 'sphere'] = 'aabb', 
    perturb=False, perturb_before_march=False, roi: torch.Tensor=None, 
    step_size: float = 1e-3, max_step_size: float = 1e10, dt_gamma: float = 0.0, max_steps: int = 512, step_size_factor=1.0
) -> dataclass_raymarch_ret:
    """ Ray marching on the given binary occupancy grid.

    Args:
        occ_grid (torch.BoolTensor): [res_x, res_y, res_z] binary occupancy grid
        rays_o (torch.Tensor): Ray origins.
        rays_d (torch.Tensor): Ray directions.
        near (Union[torch.Tensor, float]): Near depth of marching.
        far (Union[torch.Tensor, float]): Far depth of marching.
        constraction (Literal['aabb', 'tanh', 'sphere'], optional): 
            Type of space contraction. Defaults to 'aabb'.
        perturb (bool, optional): Whether to randomize the marching process. Defaults to False.
        perturb_before_march (bool, optional): 
            [DEPRECATED] This is just for debugging, should always be False. Defaults to False.
        roi (torch.Tensor, optional): Region of interest for marching. 
            Only use if rays_o, rays_d are not normalized or not ray-tested in advance. Defaults to None.
        step_size (float, optional): Initial and minimum size of each marching step. Defaults to 1e-3.
        max_step_size (float, optional): Maximum marching step. Only respected when `dt_gamma>0`. Defaults to 1e10.
        dt_gamma (float, optional): The ratio of [increase in step length of the next step] w.r.t. [current depth]. Defaults to 0.0.
        max_steps (int, optional): Maximum number of steps on each ray. Defaults to 512.
        step_size_factor (float, optional): A factor multiplied to `step_size` and `dt_gamma` in advance. Defaults to 1.0.

    Returns:
        dataclass_raymarch_ret: A structure containg all information of the marched result.
    """
    
    step_size *= step_size_factor
    dt_gamma *= step_size_factor
    
    device, dtype = rays_o.device, rays_o.dtype
    near = rays_o.new_full(rays_o.shape[:-1], near) if not isinstance(near, torch.Tensor) else near
    far = rays_o.new_full(rays_o.shape[:-1], far) if not isinstance(far, torch.Tensor) else far
    
    if roi is None:
        roi = torch.tensor([-1,-1,-1,1,1,1], dtype=dtype, device=device)

    constraction = constraction.lower()
    if constraction == 'aabb':
        contraction_type = _backend.ContractionType(ContractionType.AABB.value)
    elif constraction == 'sphere':
        contraction_type = _backend.ContractionType(ContractionType.UN_BOUNDED_SPHERE.value)
    elif constraction == 'tanh':
        contraction_type = _backend.ContractionType(ContractionType.UN_BOUNDED_TANH.value)
    else:
        raise RuntimeError(f"Invalid constraction={constraction}")

    if perturb and perturb_before_march:
        near = near + step_size * torch.rand_like(near, device=device, dtype=dtype)
    
    pack_infos, t_starts, t_ends, ridx, gidx = _backend.ray_marching(
        rays_o, rays_d, near, far, roi, occ_grid, contraction_type, 
        step_size, max_step_size, dt_gamma, max_steps, True)
    
    ridx, gidx = ridx.long(), gidx.long()
    ridx_hit = pack_infos[...,1].nonzero().long()[..., 0].contiguous()
    
    if ridx_hit.numel() != 0:
        pack_infos = pack_infos[ridx_hit].contiguous().long()

        # ridx = ridx_hit[interleave_arange_simple(pack_infos[...,1].contiguous(), return_idx=True)[1]] # 126 us @ 1M pts
        # ridx = torch.repeat_interleave(ridx_hit, pack_infos[...,1].contiguous()) # 79 us @ 1M pts

        # NOTE: Meant to be squeezed in-place.
        deltas = t_ends.squeeze_(-1) - t_starts.squeeze_(-1)
        
        if perturb and not perturb_before_march:
            noise = torch.rand_like(deltas, dtype=deltas.dtype, device=deltas.device)
            t_samples = torch.addcmul(t_starts, noise, deltas)
            deltas = packed_diff(t_samples, pack_infos) # The last item of a pack is defaulted to be zero
            # NOTE: Perturbing after march could lead to errorneous calculation of delta, compared to original non-perturbed ones' delta
            #       Consider adding random in CUDA's marching, or use the `gidx` returned by CUDA for marking consecutiveness.
            # Update 2023.02.16: Perturbing `near` and `far` before marching (by passing perturb_before_march=True) can solve the problem, 
            #       which is also the current solution in nerfacc.
        else:
            t_samples = t_starts

        samples = torch.addcmul(rays_o.index_select(0, ridx), rays_d.index_select(0, ridx), t_starts.unsqueeze(-1))
        return dataclass_raymarch_ret(ridx_hit, samples, t_starts, deltas, ridx, pack_infos, gidx, None)
    else:
        return dataclass_raymarch_ret(None, None, None, None, None, None, None, None)

def batched_occgrid_raymarch(
    occ_grid: torch.Tensor, # [B, res_x, res_y, res_z]
    rays_o: torch.Tensor, rays_d: torch.Tensor, 
    near: Union[torch.Tensor, float], far: Union[torch.Tensor, float], batch_inds: torch.Tensor=None, 
    *, constraction: Literal['aabb', 'tanh', 'sphere'] = 'aabb', 
    perturb=False, perturb_before_march=False, roi: torch.Tensor=None, 
    step_size: float = 1e-3, max_step_size: float = 1e10, dt_gamma: float = 0.0, max_steps: int = 512, step_size_factor=1.0
) -> dataclass_batched_raymarch_ret:
    """ Ray marching on the given batched binary occupancy grid.

    Args:
        occ_grid (torch.Tensor): [B, res_x, res_y, res_z] batched binary occupancy grid
        rays_o (torch.Tensor): Ray origins.
            If not given `batch_inds`, should be of shape [B, N, 3]. 
            If given `batch_inds`, should be of shape [N, 3], which is the same with `batch_inds`.
        rays_d (torch.Tensor): Ray directions.
            If not given `batch_inds`, should be of shape [B, N, 3]. 
            If given `batch_inds`, should be of shape [N, 3], which is the same with `batch_inds`.
        near (Union[torch.Tensor, float]): Near depth of marching.
        far (Union[torch.Tensor, float]): Far depth of marching.
        batch_inds (torch.Tensor, optional): Optional batch indices corresponding to each input. Defaults to None.
        constraction (Literal['aabb', 'tanh', 'sphere'], optional): 
            Type of space contraction. Defaults to 'aabb'.
        perturb (bool, optional): Whether to randomize the marching process. Defaults to False.
        perturb_before_march (bool, optional): 
            [DEPRECATED] This is just for debugging, should always be False. Defaults to False.
        roi (torch.Tensor, optional): Region of interest for marching. 
            Only use if rays_o, rays_d are not normalized or not ray-tested in advance. Defaults to None.
        step_size (float, optional): Initial and minimum size of each marching step. Defaults to 1e-3.
        max_step_size (float, optional): Maximum marching step. Only respected when `dt_gamma>0`. Defaults to 1e10.
        dt_gamma (float, optional): The ratio of [increase in step length of the next step] w.r.t. [current depth]. Defaults to 0.0.
        max_steps (int, optional): Maximum number of steps on each ray. Defaults to 512.
        step_size_factor (float, optional): A factor multiplied to `step_size` and `dt_gamma` in advance. Defaults to 1.0.

    Returns:
        dataclass_batched_raymarch_ret: A structure containg all information of the marched result.
    """

    step_size *= step_size_factor
    dt_gamma *= step_size_factor

    assert occ_grid.dim() == 4, "Requires batched occ grid input of shape [B,Nx,Ny,Nz]"
    B = occ_grid.shape[0]
    
    device, dtype = rays_o.device, rays_o.dtype
    near = rays_o.new_full(rays_o.shape[:-1], near) if not isinstance(near, torch.Tensor) else near
    far = rays_o.new_full(rays_o.shape[:-1], far) if not isinstance(far, torch.Tensor) else far

    if batch_inds is None:
        assert rays_o.dim() == 3 and rays_o.shape[0] == B, "When not given batch_inds, inputs should be batched"
        batch_data_size = rays_o.shape[1]
        rays_o, rays_d = rays_o.flatten(0, -2), rays_d.flatten(0, -2)
        near, far = near.flatten(), far.flatten()
    else:
        # Flattened rays from different batches specified by `batch_inds`` input.
        assert rays_o.dim() == 2 and [*rays_o.shape[:-1]] == [*batch_inds.shape], "When given batch_inds, inputs should have the same size with batch_inds"
        batch_inds = batch_inds.int().contiguous()
        batch_data_size = 0
    
    if roi is None:
        roi = torch.tensor([-1,-1,-1,1,1,1], dtype=dtype, device=device).tile(B,1)
    elif roi.dim() == 1:
        roi = roi.tile(B,1)
    else:
        assert roi.dim() == 2 and roi.shape[0] == B

    constraction = constraction.lower()
    if constraction == 'aabb':
        contraction_type = _backend.ContractionType(ContractionType.AABB.value)
    elif constraction == 'sphere':
        contraction_type = _backend.ContractionType(ContractionType.UN_BOUNDED_SPHERE.value)
    elif constraction == 'tanh':
        contraction_type = _backend.ContractionType(ContractionType.UN_BOUNDED_TANH.value)
    else:
        raise RuntimeError(f"Invalid constraction={constraction}")

    if perturb and perturb_before_march:
        near = near + step_size * torch.rand_like(near, device=device, dtype=dtype)
    
    pack_infos, t_starts, t_ends, ridx, bidx, gidx = _backend.batched_ray_marching(
        rays_o, rays_d, near, far, batch_inds, batch_data_size, roi, occ_grid, 
        contraction_type, step_size, max_step_size, dt_gamma, max_steps, True)

    ridx, bidx, gidx = ridx.long(), bidx.long(), gidx.long()
    ridx_hit = pack_infos[...,1].nonzero().long()[..., 0].contiguous()
    
    if ridx_hit.numel() != 0:
        pack_infos = pack_infos[ridx_hit].contiguous().long()
        
        # Meant to be inplaced.
        deltas = t_ends.squeeze_(-1) - t_starts.squeeze_(-1)
        
        if perturb and not perturb_before_march:
            noise = torch.rand_like(deltas, dtype=deltas.dtype, device=deltas.device)
            t_samples = torch.addcmul(t_starts, noise, deltas)
            deltas = packed_diff(t_samples, pack_infos) # The last item of a pack is defaulted to be zero
        else:
            t_samples = t_starts
        
        samples = torch.addcmul(rays_o.index_select(0, ridx), rays_d.index_select(0, ridx), t_starts.unsqueeze(-1))
        
        return dataclass_batched_raymarch_ret(ridx_hit, samples, t_samples, deltas, ridx, bidx, pack_infos, gidx, None)
    else:
        return dataclass_batched_raymarch_ret(None, None, None, None, None, None, None, None, None)

def forest_occgrid_raymarch() -> dataclass_forest_raymarch_ret:
    # To be released.
    pass

if __name__ == "__main__":
    def unit_test(device=torch.device('cuda'), H=1000, W=1000):
        import numpy as np
        from torch.utils.benchmark import Timer
        from nr3d_lib.render.cameras import pinhole_get_rays
        from nr3d_lib.geometry import look_at_opencv
        from nr3d_lib.render.raytest import ray_box_intersection_fast_tensor
        roi = torch.tensor([-1,-1,-1,1,1,1], device=device, dtype=torch.float)
        c2w = torch.tensor(look_at_opencv(np.array([-4,0,0]), point=np.array([0,0,0])), dtype=torch.float, device=device)
        intr = torch.zeros([3,3], dtype=torch.float, device=device)
        intr[0,0] = intr[1,1] = (H+W)/2.
        intr[0,2] = W/2.
        intr[1,2] = H/2.
        intr[2,2] = 1.
        rays_o, rays_d = pinhole_get_rays(c2w, intr, H=H, W=W)
        # near, far, mask_intersect = ray_box_intersection_fast_float(rays_o, rays_d, -1, 1)
        near, far, mask_intersect = ray_box_intersection_fast_tensor(rays_o, rays_d, roi[0:3], roi[3:6])
        ray_inds_hit = mask_intersect.nonzero().long()[...,0]
        
        rays_o, rays_d, near, far = rays_o[ray_inds_hit], rays_d[ray_inds_hit], near[ray_inds_hit], far[ray_inds_hit]
        occ_grid = torch.rand([32, 32, 32], device=device, dtype=torch.float) > 0.5
        ret = occgrid_raymarch(occ_grid, rays_o, rays_d, near, far, step_size=0.01, roi=roi)

    def unit_test_batched(device=torch.device('cuda'), H=500, W=500, B=7):
        import numpy as np
        import torch.nn.functional as F
        from torch.utils.benchmark import Timer
        from nr3d_lib.render.cameras import pinhole_get_rays
        from nr3d_lib.geometry import look_at_opencv
        from nr3d_lib.render.raytest import ray_box_intersection_fast_tensor
        
        occ_grid = torch.rand([B, 32, 32, 32], device=device, dtype=torch.float) > 0.5
        roi = torch.tensor([-1,-1,-1,1,1,1], device=device, dtype=torch.float)
        
        cam_loc = F.normalize(torch.randn([B, 3], device=device, dtype=torch.float), dim=-1) * 4
        c2w = torch.tensor(look_at_opencv(cam_loc.data.cpu().numpy(), point=np.zeros([B,3])), dtype=torch.float, device=device)
        intr = torch.zeros([3,3], dtype=torch.float, device=device)
        intr[0,0] = intr[1,1] = (H+W)/2.
        intr[0,2] = W/2.
        intr[1,2] = H/2.
        intr[2,2] = 1.
        intr = intr.tile(B,1,1)
        
        rays_o, rays_d = pinhole_get_rays(c2w, intr, H=H, W=W)
        
        # NOTE: batch inds are consecutive; ray inds are not
        # near, far, mask_intersect = ray_box_intersection_fast_tensor(rays_o, rays_d, roi[0:3], roi[3:6])
        # batch_inds, ray_inds = mask_intersect.nonzero(as_tuple=True)

        # NOTE: ray inds are consecutive; batch inds are not
        # near, far, mask_intersect = ray_box_intersection_fast_tensor(rays_o.transpose(1,0), rays_d.transpose(1,0), roi[0:3], roi[3:6])
        # ray_inds, batch_inds = mask_intersect.nonzero(as_tuple=True)
        # # u, cnt = torch.unique_consecutive(ray_inds, return_counts=True)

        # NOTE: ray inds are consecutive; batch inds are not (the same as above, but easier to use)
        near, far, mask_intersect = ray_box_intersection_fast_tensor(rays_o, rays_d, roi[0:3], roi[3:6])
        ray_inds, batch_inds = mask_intersect.t().nonzero(as_tuple=True)
        # u, cnt = torch.unique_consecutive(ray_inds, return_counts=True)

        #------------------ Regularly batched input (`batch_inds` is None)
        ret1 = batched_occgrid_raymarch(
            occ_grid, rays_o, rays_d, rays_o.new_full(rays_o.shape, 2), rays_o.new_full(rays_o.shape, 6), 
            step_size=0.01, roi=roi)
        
        #------------------ Bulks of batched input (`batch_inds` is not None)
        ret2 = batched_occgrid_raymarch(
            occ_grid, rays_o[batch_inds,ray_inds], rays_d[batch_inds,ray_inds], near[batch_inds,ray_inds], far[batch_inds,ray_inds], 
            batch_inds=batch_inds.int(), step_size=0.01, roi=roi)

    unit_test()
    unit_test_batched()
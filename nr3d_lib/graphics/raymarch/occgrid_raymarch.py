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
from nr3d_lib.graphics.pack_ops import get_pack_infos_from_boundary, packed_diff, mark_pack_boundaries
from nr3d_lib.graphics.raymarch import RaymarchRetSingle, RaymarchRetBatched, RaymarchRetForest

import nr3d_lib.bindings._occ_grid as _backend

class ContractionType(Enum):
    AABB = int(_backend.ContractionType.AABB)
    UN_BOUNDED_TANH = int(_backend.ContractionType.UN_BOUNDED_TANH)
    UN_BOUNDED_SPHERE = int(_backend.ContractionType.UN_BOUNDED_SPHERE)

def occgrid_raymarch(
    occ_grid: torch.BoolTensor,
    rays_o: torch.Tensor, rays_d: torch.Tensor, 
    near: Union[torch.Tensor, float], far: Union[torch.Tensor, float], 
    *, 
    constraction: Literal['aabb', 'tanh', 'sphere'] = 'aabb', 
    perturb=False, perturb_before_march=False, roi: torch.Tensor=None, 
    step_size: float = 1e-3, max_step_size: float = 1e10, dt_gamma: float = 0.0, max_steps: int = 512, step_size_factor=1.0
) -> RaymarchRetSingle:
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
        RaymarchRetSingle: A structure containg all information of the marched result.
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
        return RaymarchRetSingle(ridx_hit.numel(), ridx_hit, samples, t_starts, deltas, ridx, pack_infos, gidx, None)
    else:
        return RaymarchRetSingle(0, None, None, None, None, None, None, None, None)

def occgrid_raymarch_batched(
    occ_grid: torch.Tensor, # [B, res_x, res_y, res_z]
    rays_o: torch.Tensor, rays_d: torch.Tensor, rays_bidx: torch.Tensor=None, 
    near: Union[torch.Tensor, float] = ..., far: Union[torch.Tensor, float] = ..., 
    *, 
    constraction: Literal['aabb', 'tanh', 'sphere'] = 'aabb', 
    perturb=False, perturb_before_march=False, roi: torch.Tensor=None, 
    step_size: float = 1e-3, max_step_size: float = 1e10, dt_gamma: float = 0.0, max_steps: int = 512, step_size_factor=1.0
) -> RaymarchRetBatched:
    """ Ray marching on the given batched binary occupancy grid.

    Args:
        occ_grid (torch.Tensor): [B, res_x, res_y, res_z] batched binary occupancy grid
        rays_o (torch.Tensor): Ray origins.
            If not given `rays_bidx`, should be of shape [B, N, 3]. 
            If given `rays_bidx`, should be of shape [N, 3], which is the same with `rays_bidx`.
        rays_d (torch.Tensor): Ray directions.
            If not given `rays_bidx`, should be of shape [B, N, 3]. 
            If given `rays_bidx`, should be of shape [N, 3], which is the same with `rays_bidx`.
        rays_bidx (torch.Tensor, optional): Optional batch indices corresponding to each input. \
            Defaults to None.
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
        RaymarchRetBatched: A structure containg all information of the marched result.
    """

    step_size *= step_size_factor
    dt_gamma *= step_size_factor

    assert occ_grid.dim() == 4, "Requires batched occ grid input of shape [B,Nx,Ny,Nz]"
    B = occ_grid.shape[0]
    
    device, dtype = rays_o.device, rays_o.dtype
    near = rays_o.new_full(rays_o.shape[:-1], near) if not isinstance(near, torch.Tensor) else near
    far = rays_o.new_full(rays_o.shape[:-1], far) if not isinstance(far, torch.Tensor) else far

    if rays_bidx is None:
        assert rays_o.dim() == 3 and rays_o.shape[0] == B, \
            "When not given rays_bidx, inputs should be batched"
        batch_data_size = rays_o.shape[1]
        rays_o, rays_d = rays_o.flatten(0, -2), rays_d.flatten(0, -2)
        near, far = near.flatten(), far.flatten()
    else:
        # Flattened rays from different batches specified by `rays_bidx`` input.
        assert rays_o.dim() == 2 and [*rays_o.shape[:-1]] == [*rays_bidx.shape], \
            "When given rays_bidx, inputs should have the same size with rays_bidx"
        rays_bidx = rays_bidx.int().contiguous()
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
        rays_o, rays_d, near, far, rays_bidx, batch_data_size, roi, occ_grid, 
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
        
        return RaymarchRetBatched(ridx_hit.numel(), ridx_hit, samples, t_samples, deltas, ridx, pack_infos, bidx, gidx, None)
    else:
        return RaymarchRetBatched(0, None, None, None, None, None, None, None, None, None)

def occgrid_raymarch_forest(
    forest_meta, occ_grid: torch.Tensor, # [B, res_x, res_y, res_z]
    rays_o: torch.Tensor, rays_d: torch.Tensor, 
    near: Union[torch.Tensor, float], far: Union[torch.Tensor, float], # [num_rays]
    seg_block_inds: torch.Tensor, 
    seg_entries: torch.Tensor, 
    seg_exits: torch.Tensor, # [num_segments] ray intersected segments with forest blocks
    seg_pack_infos: torch.Tensor, # [num_rays, 2] Pack info of ray intersected segments with forest blocks
    *, perturb=False, perturb_before_march=False, 
    step_size: float = 1e-3, max_step_size: float = 1e10, dt_gamma: float = 0.0, max_steps: int = 512, step_size_factor=1.0
):
    
    step_size *= step_size_factor
    dt_gamma *= step_size_factor
    
    device, dtype = rays_o.device, rays_o.dtype
    near = rays_o.new_full(rays_o.shape[:-1], near) if not isinstance(near, torch.Tensor) else near
    far = rays_o.new_full(rays_o.shape[:-1], far) if not isinstance(far, torch.Tensor) else far
    
    if perturb and perturb_before_march:
        near = near + step_size * torch.rand_like(near, device=device, dtype=dtype)

    ray_pack_infos, t_starts, t_ends, ridx, blidx, _ = _backend.forest_ray_marching(
        forest_meta, rays_o, rays_d, near, far, 
        seg_block_inds.int().contiguous(), seg_entries.contiguous(), seg_exits.contiguous(), seg_pack_infos.int().contiguous(), 
        occ_grid, step_size, max_step_size, dt_gamma, max_steps, False)
    
    ridx, blidx = ridx.long(), blidx.long()
    ridx_hit = ray_pack_infos[...,1].nonzero().long()[..., 0].contiguous()
    
    if ridx_hit.numel() != 0:
        ray_pack_infos = ray_pack_infos[ridx_hit].contiguous().long()
        
        # Meant to be inplaced.
        deltas = t_ends.squeeze_(-1) - t_starts.squeeze_(-1)
        
        if perturb and not perturb_before_march:
            noise = torch.rand_like(deltas, dtype=deltas.dtype, device=deltas.device)
            t_samples = torch.addcmul(t_starts, noise, deltas)
            deltas = packed_diff(t_samples, ray_pack_infos) # The last item of a pack is defaulted to be zero
        else:
            t_samples = t_starts
        
        blidx_pack_infos = get_pack_infos_from_boundary(mark_pack_boundaries(blidx))
        
        samples = torch.addcmul(rays_o.index_select(0, ridx), rays_d.index_select(0, ridx), t_starts.unsqueeze(-1))
        
        return RaymarchRetForest(ridx_hit.numel(), ridx_hit, samples, t_samples, deltas, ridx, ray_pack_infos, blidx, blidx_pack_infos, None, None)
    else:
        return RaymarchRetForest(0, None, None, None, None, None, None, None, None, None, None)

if __name__ == "__main__":
    def unit_test(device=torch.device('cuda'), H=1000, W=1000):
        import numpy as np
        from torch.utils.benchmark import Timer
        from nr3d_lib.graphics.cameras import pinhole_get_rays
        from nr3d_lib.maths import look_at_np
        from nr3d_lib.graphics.raytest import ray_box_intersection_fast_tensor
        roi = torch.tensor([-1,-1,-1,1,1,1], device=device, dtype=torch.float)
        c2w = torch.tensor(look_at_np(np.array([-4,0,0]), point=np.array([0,0,0])), dtype=torch.float, device=device)
        intr = torch.zeros([3,3], dtype=torch.float, device=device)
        intr[0,0] = intr[1,1] = (H+W)/2.
        intr[0,2] = W/2.
        intr[1,2] = H/2.
        intr[2,2] = 1.
        rays_o, rays_d = pinhole_get_rays(c2w, intr, H=H, W=W)
        # near, far, mask_intersect = ray_box_intersection_fast_float(rays_o, rays_d, -1, 1)
        near, far, mask_intersect = ray_box_intersection_fast_tensor(rays_o, rays_d, roi[0:3], roi[3:6])
        rays_inds_hit = mask_intersect.nonzero().long()[...,0]
        
        rays_o, rays_d, near, far = rays_o[rays_inds_hit], rays_d[rays_inds_hit], near[rays_inds_hit], far[rays_inds_hit]
        occ_grid = torch.rand([32, 32, 32], device=device, dtype=torch.float) > 0.5
        ret = occgrid_raymarch(occ_grid, rays_o, rays_d, near, far, step_size=0.01, roi=roi)

    def unit_test_batched(device=torch.device('cuda'), H=500, W=500, B=7):
        import numpy as np
        import torch.nn.functional as F
        from torch.utils.benchmark import Timer
        from nr3d_lib.graphics.cameras import pinhole_get_rays
        from nr3d_lib.maths import look_at_np
        from nr3d_lib.graphics.raytest import ray_box_intersection_fast_tensor
        
        occ_grid = torch.rand([B, 32, 32, 32], device=device, dtype=torch.float) > 0.5
        roi = torch.tensor([-1,-1,-1,1,1,1], device=device, dtype=torch.float)
        
        cam_loc = F.normalize(torch.randn([B, 3], device=device, dtype=torch.float), dim=-1) * 4
        c2w = torch.tensor(look_at_np(cam_loc.data.cpu().numpy(), point=np.zeros([B,3])), dtype=torch.float, device=device)
        intr = torch.zeros([3,3], dtype=torch.float, device=device)
        intr[0,0] = intr[1,1] = (H+W)/2.
        intr[0,2] = W/2.
        intr[1,2] = H/2.
        intr[2,2] = 1.
        intr = intr.tile(B,1,1)
        
        rays_o, rays_d = pinhole_get_rays(c2w, intr, H=H, W=W)
        
        # NOTE: batch inds are consecutive; ray inds are not
        # near, far, mask_intersect = ray_box_intersection_fast_tensor(rays_o, rays_d, roi[0:3], roi[3:6])
        # rays_bidx, rays_inds = mask_intersect.nonzero(as_tuple=True)

        # NOTE: ray inds are consecutive; batch inds are not
        # near, far, mask_intersect = ray_box_intersection_fast_tensor(rays_o.transpose(1,0), rays_d.transpose(1,0), roi[0:3], roi[3:6])
        # rays_inds, rays_bidx = mask_intersect.nonzero(as_tuple=True)
        # # u, cnt = torch.unique_consecutive(rays_inds, return_counts=True)

        # NOTE: ray inds are consecutive; batch inds are not (the same as above, but easier to use)
        near, far, mask_intersect = ray_box_intersection_fast_tensor(rays_o, rays_d, roi[0:3], roi[3:6])
        rays_inds, rays_bidx = mask_intersect.t().nonzero(as_tuple=True)
        # u, cnt = torch.unique_consecutive(rays_inds, return_counts=True)

        #------------------ Regularly batched input (`rays_bidx` is None)
        ret1 = occgrid_raymarch_batched(
            occ_grid, rays_o, rays_d, rays_o.new_full(rays_o.shape, 2), rays_o.new_full(rays_o.shape, 6), 
            step_size=0.01, roi=roi)
        
        #------------------ Bulks of batched input (`rays_bidx` is not None)
        ret2 = occgrid_raymarch_batched(
            occ_grid, rays_o[rays_bidx,rays_inds], rays_d[rays_bidx,rays_inds], near[rays_bidx,rays_inds], far[rays_bidx,rays_inds], 
            bidx=rays_bidx.int(), step_size=0.01, roi=roi)
    
    def unit_test_forest(device=torch.device('cuda'), H=500, W=500, B=7):
        import numpy as np
        from torch.utils.benchmark import Timer
        from kaolin.ops.spc import unbatched_points_to_octree
        from nr3d_lib.models.spatial import ForestBlockSpace
        from nr3d_lib.graphics.cameras import pinhole_get_rays
        from nr3d_lib.maths import look_at_np

        forest_level = 3
        forest_block_coords = (torch.rand([2**forest_level]*3, device=device) > 0.5).nonzero().short()
        num_forest_blocks = forest_block_coords.shape[0]
        octree = unbatched_points_to_octree(forest_block_coords, level=forest_level, sorted=False)
        forest = ForestBlockSpace(device=device)
        forest.reset(octree, level=forest_level)
        
        forest_occ_grid = torch.rand([num_forest_blocks, 32, 32, 32], device=device) > 0.5
        
        c2w = torch.tensor(look_at_np(np.array([-4,0,0]), point=np.array([0,0,0])), dtype=torch.float, device=device)
        intr = torch.zeros([3,3], dtype=torch.float, device=device)
        intr[0,0] = intr[1,1] = (H+W)/2.
        intr[0,2] = W/2.
        intr[1,2] = H/2.
        intr[2,2] = 1.
        rays_o, rays_d = pinhole_get_rays(c2w, intr, H=H, W=W)
        ray_tested = forest.ray_test(rays_o, rays_d, return_rays=True)
        
        ret = occgrid_raymarch_forest(
            forest.meta, forest_occ_grid, ray_tested['rays_o'], ray_tested['rays_d'], 0., np.sqrt(3) * 2**forest_level,  
            ray_tested['seg_block_inds'], ray_tested['seg_entries'], ray_tested['seg_exits'], ray_tested['seg_pack_infos'], 
            perturb=False, step_size=0.001, dt_gamma=0, max_steps=4096)

        ret = occgrid_raymarch_forest(
            forest.meta, forest_occ_grid, ray_tested['rays_o'], ray_tested['rays_d'], 0., np.sqrt(3) * 2**forest_level,  
            ray_tested['seg_block_inds'], ray_tested['seg_entries'], ray_tested['seg_exits'], ray_tested['seg_pack_infos'], 
            perturb=True, step_size=0.001, dt_gamma=0, max_steps=4096)

        ret = occgrid_raymarch_forest(
            forest.meta, forest_occ_grid, ray_tested['rays_o'], ray_tested['rays_d'], 0., np.sqrt(3) * 2**forest_level,  
            ray_tested['seg_block_inds'], ray_tested['seg_entries'], ray_tested['seg_exits'], ray_tested['seg_pack_infos'], 
            perturb=True, perturb_before_march=True, step_size=0.001, dt_gamma=0, max_steps=4096)

    def test_forest_ray_march(device=torch.device('cuda')):
        from icecream import ic
        from nr3d_lib.config import ConfigDict
        from nr3d_lib.graphics.pack_ops import interleave_arange
        from nr3d_lib.models.spatial import ForestBlockSpace
        from nr3d_lib.models.accelerations import OccGridAccelForest
        
        forest_level = 3
        occ_grid_resolution = [16,16,16]
        
        forest = ForestBlockSpace(device=device)
        # occ_grid = torch.rand([2**forest_level]*3, device=device, dtype=torch.float) > 0.5
        # torch.save({'forest_space_occ': occ_grid}, './dev_test/dbg_forest.pt')
        occ_grid = torch.load('./dev_test/dbg_forest.pt')['forest_space_occ']
        corners = occ_grid.nonzero().long()
        forest.populate_from_corners(corners=corners, level=forest_level, world_origin=[-1,-2,-1], world_block_size=[0.3,0.3,0.3])
        
        rays_o_0 = torch.tensor([[-2,-2,-2]], device=device, dtype=torch.float)
        rays_d_0 = torch.tensor([[1, 1, 1]], device=device, dtype=torch.float)
        ray_tested = forest.ray_test(rays_o_0, rays_d_0)
        
        rays_o, rays_d, near, far, num_rays, rays_inds, seg_pack_infos, seg_block_inds, seg_entries, seg_exits = \
            itemgetter('rays_o', 'rays_d', 'near', 'far', 'num_rays', 'rays_inds', 'seg_pack_infos', 'seg_block_inds', 'seg_entries', 'seg_exits')(ray_tested)

        # from nr3d_lib.graphics.raytest import octree_raytrace, ray_box_intersection_fast_float_nocheck
        # from kaolin.render.spc import unbatched_raytrace
        # world_origin, world_block_size = forest.world_origin.to(rays_o_0.dtype), forest.world_block_size.to(rays_d_0.dtype)
        # world_scale = world_block_size * (2**forest.level) / 2.
        # rays_o_forest, rays_d_forest = (rays_o_0-world_origin)/world_scale - 1., rays_d_0/world_scale
        # ridx, pidx, depth = octree_raytrace(forest.spc.octrees, forest.spc.point_hierarchies, forest.spc.pyramids[0], forest.spc.exsum, rays_o_forest, rays_d_forest, level=forest.level, tmin=None, tmax=None)

        # # _near, _far = ray_box_intersection_fast_float_nocheck(rays_o_forest, rays_d_forest, -1., 1.)
        # ridx0, pidx0, depth0 = unbatched_raytrace(forest.spc.octrees, forest.spc.point_hierarchies, forest.spc.pyramids[0], forest.spc.exsum, rays_o_forest, rays_d_forest, level=forest.level, with_exit=True)
        # ridx1, pidx1, depth1 = unbatched_raytrace(forest.spc.octrees, forest.spc.point_hierarchies, forest.spc.pyramids[0], forest.spc.exsum, rays_o_forest, rays_d_forest, level=forest.level, with_exit=False)


        num_blocks = len(forest.block_ks)
        batched_occ_val_grid = torch.rand([num_blocks,*occ_grid_resolution], device=device, dtype=torch.float)
        # batched_occ_grid = batched_occ_val_grid > 0.9
        # batched_occ_grid = torch.ones([num_blocks,*occ_grid_resolution], device=device, dtype=torch.bool)
        # ret = occgrid_raymarch_forest(forest.meta, batched_occ_grid, rays_o, rays_d, near, far, seg_block_inds, seg_entries, seg_exits, seg_pack_infos, step_size=0.01)

        accel = OccGridAccelForest(
            space=forest, 
            resolution=occ_grid_resolution, 
            occ_val_fn_cfg=ConfigDict(type='sdf', inv_s=256.0), 
            occ_thre=0.3, ema_decay=0.95, init_cfg={'mode':'from_net'}, update_from_net_cfg={}, update_from_samples_cfg={}, 
            n_steps_between_update=16, n_steps_warmup=256
        )
        accel.populate()
        accel.occ.occ_val_grid = batched_occ_val_grid
        accel.occ.occ_grid = accel.occ.occ_val_grid > accel.occ.occ_thre
        
        ret = accel.ray_march(rays_o, rays_d, near, far, seg_block_inds, seg_entries, seg_exits, seg_pack_infos, step_size=0.01)
        
        # test_depth_samples = torch.linspace(0, far.item(), 1000, device=device)
        test_depth_samples = interleave_arange(near, far, 0.01, return_idx=False)
        test_samples = torch.addcmul(rays_o[0].unsqueeze(-2), rays_d[0].unsqueeze(-2), test_depth_samples.unsqueeze(-1))
        test_samples_occupied = accel.query_world(test_samples)
        
        # npts, blidx = accel.space.normalize_coords(test_samples)
        
        ic(test_depth_samples[test_samples_occupied])
        ic(ret.depth_samples)        
        
        from vedo import Box, Plotter, Volume, show, Points, Arrow
        forest_actors = accel.debug_vis(draw=False, boundary=True, draw_occ_grid=False)
        ray_arrow = Arrow(rays_o[0].data.cpu().numpy(), (rays_o+rays_d)[0].data.cpu().numpy(), s=0.01)
        plt = Plotter(axes=1)
        sample_pts = Points(ret.samples.data.cpu().numpy(), r=12.0)
        plt.show(*forest_actors, ray_arrow, sample_pts)
        plt.interactive().close()
        

    unit_test()
    unit_test_batched()
    unit_test_forest()
    test_forest_ray_march()

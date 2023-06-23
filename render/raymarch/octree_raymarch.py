"""
@file   octree_raymarch.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Ray marching on an octree. Ray-trace in advance, then march directly on the traced results.
        - Benefits from and based on the output of the efficient ray-tracing on octrees (via Spc of Kaolin lib).
        - Impirically found to be slower than march directly on occupancy grids without tracing in advance.
          - TODO: List experiment benchmark data here.
"""

from typing import Literal, Dict

import torch
from torch.utils.benchmark import Timer

try:
    from kaolin.rep.spc import Spc
    from kaolin.ops.spc import unbatched_query
    from kaolin.render.spc import mark_pack_boundaries
except:
    from nr3d_lib.fmt import log
    log.warning("kaolin is not installed. OctreeAS / ForestAS disabled.")

from nr3d_lib.render.raymarch import dataclass_raymarch_ret
from nr3d_lib.render.raytest import octree_raytrace_fixed_from_kaolin
from nr3d_lib.render.pack_ops import get_pack_infos_from_boundary, get_pack_infos_from_first, get_pack_infos_from_n, interleave_linstep, octree_mark_consecutive_segments, packed_diff, merge_two_packs_sorted_a_includes_b, merge_two_packs_sorted_aligned
from nr3d_lib.render.raysample import batch_sample_step_linear, batch_sample_step_wrt_depth, batch_sample_step_wrt_sqrt_depth, interleave_sample_step_linear, interleave_sample_step_linear_in_packed_segments, interleave_sample_step_wrt_depth_clamped, interleave_sample_step_wrt_depth_in_packed_segments

def octree_raytrace(spc: Spc, rays_o: torch.Tensor, rays_d: torch.Tensor, near: torch.Tensor=None, far: torch.Tensor=None, level: int = None):
    if level is None: level = spc.pyramids[0].shape[-1] - 2
    ridx, pidx, depth = octree_raytrace_fixed_from_kaolin(
        spc.octrees, spc.point_hierarchies, spc.pyramids[0], spc.exsum, 
        rays_o, rays_d, level=level, tmin=near.min().item() if near is not None else 0., tmax=far.max().item() if far is not None else None)
    return dict(ridx=ridx.contiguous().long(), pidx=pidx.contiguous().long(), depth=depth.contiguous())

def octree_query(spc: Spc, level: int, x: torch.Tensor, with_parents=False):
    if x.dim() != 2:
        prefix = x.shape[:-1]
        return unbatched_query(spc.octrees, spc.exsum, x.flatten(0,-2), level, with_parents).unflatten(0, prefix)
    else:
        return unbatched_query(spc.octrees, spc.exsum, x, level, with_parents)

def octree_raymarch(
    spc: Spc, level: int, 
    rays_o: torch.Tensor, rays_d: torch.Tensor, near=None, far=None, *, perturb=False, 
    march_mode='segment_step', **march_kwargs
    ):
    if march_mode == 'segment_step':
        return octree_raymarch_segment_step(spc, level, rays_o, rays_d, near, far, perturb=perturb, **march_kwargs)
    elif march_mode == 'voxel_step':
        return octree_raymarch_voxel_step(spc, level, rays_o, rays_d, near, far, perturb=perturb, **march_kwargs)
    elif march_mode == 'ray_step_in_segment':
        return octree_raymarch_ray_step_in_segment(spc, level, rays_o, rays_d, near, far, perturb=perturb, **march_kwargs)
    elif march_mode == 'voxel_batch':
        return octree_raymarch_voxel_batch(spc, level, rays_o, rays_d, near, far, perturb=perturb, **march_kwargs)
    elif march_mode == 'ray_batch_in_voxel':
        return octree_raymarch_ray_batch_in_voxel(spc, level, rays_o, rays_d, near, far, perturb=perturb, **march_kwargs)
    elif march_mode == 'two_level':
        spc2, level2 = march_kwargs.pop('spc2'), march_kwargs.pop('level2')
        return octree_raymarch_two_level(spc, level, spc2, level2, rays_o, rays_d, near, far, perturb=perturb, **march_kwargs)
    else:
        raise RuntimeError(f"Invalid march_mode={march_mode}")

# @profile
def octree_raymarch_segment_step(
    spc: Spc, level: int, 
    rays_o: torch.Tensor, rays_d: torch.Tensor, near=None, far=None, *, perturb=False, 
    ray_traced: Dict[str,torch.Tensor]=None, 
    step_mode: Literal['linear', 'depth'], step_size_factor=1.0, **step_kwargs
    ) -> dataclass_raymarch_ret:
    device, num_rays = rays_o.device, rays_o.shape[0]
    # Assume rays are normalized
    if ray_traced is None:
        ray_traced = octree_raytrace(spc, rays_o, rays_d, near=near, far=far, level=level)

    # [num_nuggets], [num_nuggets], [num_nuggets, 2]
    nugget_ridx, nugget_pidx, nugget_depths = ray_traced['ridx'], ray_traced['pidx'], ray_traced['depth']
    
    if nugget_ridx.numel() > 0:
        nugget_boundary = mark_pack_boundaries(nugget_ridx)
        nugget_pack_infos = get_pack_infos_from_boundary(nugget_boundary)

        # [num_rays_hit]
        ridx_hit = nugget_ridx[nugget_pack_infos[...,0]] # 6.69 us @ 195k pts

        # [num_nuggets]
        mark_start, mark_end = octree_mark_consecutive_segments(nugget_pidx, nugget_pack_infos, spc.point_hierarchies)
        # [num_segments]
        seg_entry_nidx, seg_exit_nidx = mark_start.nonzero().long()[..., 0], mark_end.nonzero().long()[..., 0] # 70 us @ 180k pts , 300 us @ 10M pts
        seg_entry, seg_exit, seg_ridx = nugget_depths[seg_entry_nidx,0], nugget_depths[seg_exit_nidx,1], nugget_ridx[seg_entry_nidx] # 27 us @ 180k pts, 270 us @ 10M pts

        # [num_samples]
        if step_mode == 'linear':
            depth_samples, deltas, sidx, seg_pack_infos = interleave_sample_step_linear(seg_entry, seg_exit, perturb=perturb, step_size_factor=step_size_factor, **step_kwargs)
        elif step_mode == 'depth':
            depth_samples, deltas, sidx, seg_pack_infos = interleave_sample_step_wrt_depth_clamped(seg_entry, seg_exit, perturb=perturb, step_size_factor=step_size_factor, **step_kwargs)
        else:
            raise RuntimeError(f"Invalid step_mode={step_mode}")
        
        # [num_samples]
        ridx = seg_ridx[sidx] # 6.41 us @ 195k pts

        # [num_samples, 3]
        samples = torch.addcmul(rays_o.index_select(0, ridx), rays_d.index_select(0, ridx), depth_samples.unsqueeze(-1))

        # TOTAL: 72 us @ 195k pts, 189 us @ 3.5M pts
        boundary = mark_pack_boundaries(ridx) # 14.03 us @ 195k pts
        pack_infos = get_pack_infos_from_boundary(boundary)

        return dataclass_raymarch_ret(ridx_hit, samples, depth_samples, deltas, ridx, pack_infos, sidx, seg_pack_infos)
    else:
        return dataclass_raymarch_ret(None, None, None, None, None, None, None, None)

# @profile
def octree_raymarch_voxel_step(
    spc: Spc, level: int, 
    rays_o: torch.Tensor, rays_d: torch.Tensor, near=None, far=None, *, perturb=False, 
    ray_traced: Dict[str,torch.Tensor]=None, 
    step_mode: Literal['linear', 'depth'], step_size_factor=1.0, **step_kwargs
    ) -> dataclass_raymarch_ret:
    
    device, num_rays = rays_o.device, rays_o.shape[0]
    # Assume rays are normalized
    if ray_traced is None:
        ray_traced = octree_raytrace(spc, rays_o, rays_d, near=near, far=far, level=level)

    # [num_nuggets], [num_nuggets], [num_nuggets, 2]
    nugget_ridx, nugget_pidx, nugget_depths = ray_traced['ridx'], ray_traced['pidx'], ray_traced['depth']
    if nugget_ridx.numel() > 0:
        nugget_boundary = mark_pack_boundaries(nugget_pidx)

        # [num_samples]        
        if step_mode == 'linear':
            depth_samples, deltas, nidx, pack_infos = interleave_sample_step_linear(nugget_depths[:,0], nugget_depths[:,1], perturb=perturb, step_size_factor=step_size_factor, **step_kwargs)
        elif step_mode == 'depth':
            depth_samples, deltas, nidx, pack_infos = interleave_sample_step_wrt_depth_clamped(nugget_depths[:,0], nugget_depths[:,1], perturb=perturb, step_size_factor=step_size_factor,  **step_kwargs)
        else:
            raise RuntimeError(f"Invalid step_mode={step_mode}")
        
        # [num_samples]        
        ridx, pidx = nugget_ridx[nidx], nugget_pidx[nidx]
        
        # [num_samples, 3]
        samples = torch.addcmul(rays_o.index_select(0, ridx), rays_d.index_select(0, ridx), depth_samples.unsqueeze(-1))
        
        # [num_rays_hit]
        ridx_hit = nugget_ridx[nugget_boundary]
        
        point_pack_infos = get_pack_infos_from_boundary(mark_pack_boundaries(pidx))
        return dataclass_raymarch_ret(ridx_hit, samples, depth_samples, deltas, ridx, pack_infos, pidx, point_pack_infos)
    else:
        return dataclass_raymarch_ret(None, None, None, None, None, None, None, None)

# @profile
def octree_raymarch_ray_step_in_segment(
    spc: Spc, level: int, 
    rays_o: torch.Tensor, rays_d: torch.Tensor, near=None, far=None, *, perturb=False, 
    ray_traced: Dict[str,torch.Tensor]=None, 
    step_mode: Literal['linear', 'depth'], step_size_factor=1.0, **step_kwargs
    ) -> dataclass_raymarch_ret:

    device, num_rays = rays_o.device, rays_o.shape[0]
    # Assume rays are normalized
    if ray_traced is None:
        ray_traced = octree_raytrace(spc, rays_o, rays_d, near=near, far=far, level=level)

    # [num_nuggets], [num_nuggets], [num_nuggets, 2]
    nugget_ridx, nugget_pidx, nugget_depths = ray_traced['ridx'], ray_traced['pidx'], ray_traced['depth']
    if nugget_ridx.numel() > 0:
        nugget_boundary = mark_pack_boundaries(nugget_ridx)
        nugget_pack_infos = get_pack_infos_from_boundary(nugget_boundary)

        # [num_rays_hit]
        ridx_hit = nugget_ridx[nugget_pack_infos[...,0]] # 6.69 us @ 195k pts
        # ridx_hit = nugget_ridx[nugget_boundary] # 33 us @ 195k pts
        # ridx_hit = ridx[boundary] # 58.6 us @ 195k pts

        # [num_nuggets]
        mark_start, mark_end = octree_mark_consecutive_segments(nugget_pidx, nugget_pack_infos, spc.point_hierarchies)
        # [num_segments]
        seg_entry_nidx, seg_exit_nidx = mark_start.nonzero().long()[..., 0], mark_end.nonzero().long()[..., 0] # 70 us @ 180k pts , 300 us @ 10M pts
        seg_entry, seg_exit, seg_ridx = nugget_depths[seg_entry_nidx,0], nugget_depths[seg_exit_nidx,1], nugget_ridx[seg_entry_nidx]
        input_seg_infos = get_pack_infos_from_boundary(nugget_boundary[seg_entry_nidx].contiguous())

        # [num_samples]
        if step_mode == 'linear':
            depth_samples, deltas, _ridx, pack_infos, sidx, seg_pack_infos = interleave_sample_step_linear_in_packed_segments(near[ridx_hit], far[ridx_hit], seg_entry, seg_exit, input_seg_infos, perturb=perturb, step_size_factor=step_size_factor, **step_kwargs)
        elif step_mode == 'depth':
            depth_samples, deltas, _ridx, pack_infos, sidx, seg_pack_infos = interleave_sample_step_wrt_depth_in_packed_segments(near[ridx_hit], far[ridx_hit], seg_entry, seg_exit, input_seg_infos, perturb=perturb, step_size_factor=step_size_factor, **step_kwargs)
        else:
            raise RuntimeError(f"Invalid step_mode={step_mode}")
        ridx = ridx_hit[_ridx]
        
        samples = torch.addcmul(rays_o.index_select(0, ridx), rays_d.index_select(0, ridx), depth_samples.unsqueeze(-1))

        # boundary = mark_pack_boundaries(ridx) # 14.03 us @ 195k pts
        # pack_infos = get_pack_infos_from_boundary(boundary)

        # NOTE: only for debug purposes
        # return ridx_hit, samples, depth_samples, deltas, ridx, pack_infos, sidx, seg_pack_infos, nugget_depths, nugget_pack_indices
        return dataclass_raymarch_ret(ridx_hit, samples, depth_samples, deltas, ridx, pack_infos, sidx, seg_pack_infos)
    else:
        return dataclass_raymarch_ret(None, None, None, None, None, None, None, None)

# @profile
def octree_raymarch_voxel_batch(
    spc: Spc, level: int, 
    rays_o: torch.Tensor, rays_d: torch.Tensor, near=None, far=None, *, perturb=False, 
    ray_traced: Dict[str,torch.Tensor]=None, 
    step_mode: Literal['linear', 'depth', 'sqrt_depth'], num_marched: int=16, **step_kwargs
    ) -> dataclass_raymarch_ret:
    device, num_rays = rays_o.device, rays_o.shape[0]
    # Assume rays are normalized
    if ray_traced is None:
        ray_traced = octree_raytrace(spc, rays_o, rays_d, near=near, far=far, level=level)

    # [num_nuggets], [num_nuggets], [num_nuggets, 2]
    nugget_ridx, nugget_pidx, nugget_depths = ray_traced['ridx'], ray_traced['pidx'], ray_traced['depth']
    
    if nugget_ridx.numel() > 0:
        nugget_boundary = mark_pack_boundaries(nugget_ridx)
        nugget_pack_infos = get_pack_infos_from_boundary(nugget_boundary)
        
        # [num_rays_hit]
        ridx_hit = nugget_ridx[nugget_pack_infos[...,0]]
        
        # [num_rays_hit, num_marched]
        if step_mode == 'linear':
            depth_samples, deltas = batch_sample_step_linear(nugget_depths[:,0], nugget_depths[:,1], perturb=perturb, num_samples=num_marched, return_dt=True, **step_kwargs)
        elif step_mode == 'depth':
            depth_samples, deltas = batch_sample_step_wrt_depth(nugget_depths[:,0], nugget_depths[:,1], perturb=perturb, num_samples=num_marched, return_dt=True, **step_kwargs)
        elif step_mode == 'sqrt_depth':
            depth_samples, deltas = batch_sample_step_wrt_sqrt_depth(nugget_depths[:,0], nugget_depths[:,1], perturb=perturb, num_samples=num_marched, return_dt=True, **step_kwargs)
        else:
            raise RuntimeError(f"Invalid step_mode={step_mode}")

        # [num_rays_hit*num_marched, 3]
        samples = torch.addcmul(
            rays_o.index_select(0, nugget_ridx).unsqueeze(-2), 
            rays_d.index_select(0, nugget_ridx).unsqueeze(-2), 
            depth_samples.unsqueeze(-1)
        ).view(-1,3)

        # [num_rays_hit*num_marched]
        depth_samples, deltas = depth_samples.view(-1), deltas.view(-1)

        # [num_rays_hit]
        pack_infos = get_pack_infos_from_first(nugget_pack_infos[...,0] * num_marched)
        
        # [num_rays_hit*num_marched]
        # ridx = ridx.repeat_interleave(num_marched).view(-1) # NOTE: slow
        ridx = nugget_ridx.view(-1,1).tile(1, num_marched).view(-1).contiguous()
        pidx = nugget_pidx.view(-1,1).tile(1, num_marched).view(-1).contiguous()
        point_boundary = mark_pack_boundaries(pidx)
        point_pack_infos = get_pack_infos_from_boundary(point_boundary)
        
        return dataclass_raymarch_ret(ridx_hit, samples, depth_samples, deltas, ridx, pack_infos, pidx, point_pack_infos)
    else:
        return dataclass_raymarch_ret(None, None, None, None, None, None, None, None)

# @profile
def octree_raymarch_ray_batch_in_voxel(
    spc: Spc, level: int, 
    rays_o: torch.Tensor, rays_d: torch.Tensor, near=None, far=None, *, perturb=False, 
    ray_traced: Dict[str,torch.Tensor]=None, 
    step_mode: Literal['linear', 'depth', 'sqrt_depth'], num_marched: int=256, **step_kwargs
    ) -> dataclass_raymarch_ret:
    device, dtype, num_rays = rays_o.device, rays_o.dtype, rays_o.shape[0]
    
    # [num_rays, num_marched]
    if step_mode == 'linear':
        depth_samples, deltas = batch_sample_step_linear(near, far, perturb=perturb, prefix_shape=(num_rays,), device=device, dtype=dtype, num_samples=num_marched, return_dt=True, **step_kwargs)
    elif step_mode == 'depth':
        depth_samples, deltas = batch_sample_step_wrt_depth(near, far, perturb=perturb, prefix_shape=(num_rays,), device=device, dtype=dtype, num_samples=num_marched, return_dt=True, **step_kwargs)
    elif step_mode == 'sqrt_depth':
        depth_samples, deltas = batch_sample_step_wrt_sqrt_depth(near, far, perturb=perturb, prefix_shape=(num_rays,), device=device, dtype=dtype, num_samples=num_marched, return_dt=True, **step_kwargs)
    else:
        raise RuntimeError(f"Invalid step_mode={step_mode}")

    # [num_rays, num_marched, 3]
    samples = torch.addcmul(rays_o.unsqueeze(-2), rays_d.unsqueeze(-2), depth_samples.unsqueeze(-1))
    
    # [num_rays, num_marched]
    pidx = octree_query(spc, level, samples)
    mask = pidx > -1
    
    # [num_masked, ...]
    ridx = torch.nonzero(mask).long()[..., 0]
    if ridx.numel() > 0:
        pidx, depth_samples, deltas, samples = pidx[ridx], depth_samples[ridx], deltas[ridx], samples[ridx]
        
        boundary = mark_pack_boundaries(ridx)
        # [num_rays_hit]
        pack_infos = get_pack_infos_from_boundary(boundary)
        ridx_hit = ridx[pack_infos[...,0]]
        
        point_boundary = mark_pack_boundaries(pidx)
        point_pack_infos = get_pack_infos_from_boundary(point_boundary)
        
        return dataclass_raymarch_ret(ridx_hit, samples, depth_samples, deltas, ridx, pack_infos, pidx, point_pack_infos)
    else:
        return dataclass_raymarch_ret(None, None, None, None, None, None, None, None)

def octree_raymarch_two_level(
    spc: Spc, level: int, spc2: Spc, level2: int,
    rays_o: torch.Tensor, rays_d: torch.Tensor, near=None, far=None, *, perturb=False, 
    lvl1_march_cfg: dict = {}, lvl2_march_cfg: dict = {}
    ) -> dataclass_raymarch_ret:
    assert level < level2, f"level_1={level} should be less than leve_2={level2}"
    
    # TODO: These two marches can totally be parallel (marybe use different streams)
    ret1 = octree_raymarch(spc, level, rays_o, rays_d, near, far, perturb=perturb, **lvl1_march_cfg)
    ret2 = octree_raymarch(spc2, level2, rays_o, rays_d, near, far, perturb=perturb, **lvl2_march_cfg)
    
    pidx1, pidx2, pack_infos = merge_two_packs_sorted_a_includes_b(ret1.depth_samples, ret1.pack_infos, ret1.ridx_hit, ret2.depth_samples, ret2.pack_infos, ret2.ridx_hit)
    
    numel = ret1.depth_samples.numel(), ret2.depth_samples.numel()
    depth_samples = ret1.depth_samples.new_zeros([numel])
    samples = ret1.samples.new_zeros([numel, 3])
    ridx = ret1.ridx.new_zeros([numel])
    
    depth_samples[pidx1], depth_samples[pidx2] = ret1.depth_samples, ret2.depth_samples
    samples[pidx1], samples[pidx2] = ret1.samples, ret2.samples
    ridx[pidx1], ridx[pidx2] = ret1.ridx, ret2.ridx
    deltas = packed_diff(depth_samples, pack_infos)
    
    return dataclass_raymarch_ret(ret1.ridx_hit, samples, depth_samples, deltas, ridx, pack_infos)
    
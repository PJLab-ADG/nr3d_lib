"""
@file   forest.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Handles a forest of blocks space
"""

__all__ = [
    'ForestBlockSpace'
]

import numpy as np
from numbers import Number
from typing import List, Literal, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.benchmark import Timer

from nr3d_lib.utils import check_to_torch
from nr3d_lib.graphics.raymarch import RaymarchRetForest
from nr3d_lib.graphics.raytest import ray_box_intersection, octree_raytrace_fixed_from_kaolin
from nr3d_lib.graphics.pack_ops import get_pack_infos_from_first, get_pack_infos_from_boundary, packed_sort_inplace, mark_pack_boundaries
from nr3d_lib.graphics.raysample import interleave_sample_step_linear_in_packed_segments, interleave_sample_step_wrt_depth_in_packed_segments
from nr3d_lib.models.spatial.utils import create_octree_dense, create_octree_root_only, octree_to_spc_ins

from nr3d_lib.bindings._forest import ForestMeta

try:
    from kaolin.ops.spc import unbatched_points_to_octree, unbatched_query
except:
    from nr3d_lib.fmt import log
    log.warning("kaolin is not installed. OctreeAS / ForestAS disabled.")

class ForestBlockSpace(nn.Module):
    def __init__(
        self, 
        continuity_enabled=True, 
        dtype=torch.float, device=None
        ) -> None:
        super().__init__()
        self.dtype = dtype
        self.continuity_enabled = continuity_enabled
        
        world_origin = torch.zeros([3,], dtype=dtype, device=device)
        self.register_buffer('world_origin', world_origin, persistent=True) # Block [0,0,0]'s position in world
        self.register_buffer('world_origin0', world_origin.clone(), persistent=True) # Block [0,0,0]'s position in world
        
        world_block_size = torch.ones([3,], dtype=dtype, device=device)
        self.register_buffer('world_block_size', world_block_size, persistent=True) # Block size in world
        self.register_buffer('world_block_size0', world_block_size.clone(), persistent=True) # Block size in world
        
        self.register_buffer('_octree', torch.empty([], dtype=torch.uint8, device=device), persistent=True) # SPC data structure of forest
        self.register_buffer('_max_length', torch.tensor([0,], dtype=torch.long, device=device), persistent=True) # Integer coords range
        # self.register_buffer('_level', torch.tensor([1], dtype=torch.long, device=device), persistent=True)
        
        self.meta = None
        self._register_load_state_dict_pre_hook(self.before_load_state_dict)

    @property
    def device(self) -> torch.device:
        return self.world_origin.device

    def before_load_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # Load octree in advance
        octree = state_dict[prefix + '_octree']
        # level = state_dict.pop(prefix + '_level').item()

        if (prefix + '_max_length') not in state_dict:
            # NOTE: Temorary fix for backward compatibility
            level = octree_to_spc_ins(octree).pyramids.shape[-1] - 2
            max_length = 2**level
            state_dict[prefix + '_max_length'] = check_to_torch([max_length], ref=octree)
        else:
            max_length = state_dict[prefix + '_max_length'].item()

        self.reset(octree, max_length=max_length)
        # self.reset(octree, level=level)

    def get_aabb(self) -> torch.Tensor:
        aabb_min = self.world_origin + self.world_block_size * self.block_ks.min(dim=0).values
        aabb_max = self.world_origin + self.world_block_size * (self.block_ks.max(dim=0).values+1)
        return torch.stack([aabb_min, aabb_max], dim=0)

    def get_bounding_volume(self) -> torch.Tensor:
        aabb = self.get_aabb()
        center = (aabb[1] + aabb[0]) / 2.
        radius3d = (aabb[1] - aabb[0]) / 2.
        return torch.cat([center, radius3d], dim=-1)

    @property
    def n_trees(self) -> int:
        return self.meta.n_trees

    @property
    def level(self) -> int:
        return self.meta.level

    @property
    def block_ks(self) -> torch.Tensor:
        return self.meta.block_ks

    @torch.no_grad()
    def to_occgrid(self):
        max_length = self._max_length.item()
        occ_grid = torch.zeros([max_length]*3, dtype=torch.bool, device=self.device)
        occ_grid[tuple(self.block_ks.long().t())] = 1
        return occ_grid

    @torch.no_grad()
    def set_enable_continuity(self, enabled=True):
        self.meta.continuity_enabled = enabled

    @torch.no_grad()
    def reset(self, octree: torch.Tensor, level: int=None, max_length: int=None, world_origin=None, world_block_size=None):
        #------- Reset world_origin and world_block_size
        if world_origin is not None:
            if isinstance(world_origin, Number):
                world_origin = [world_origin] * 3
            if isinstance(world_origin, (list, tuple, np.ndarray)):
                world_origin = torch.tensor(world_origin, dtype=self.dtype, device=self.device)
            elif isinstance(world_origin, torch.Tensor):
                world_origin = world_origin.to(dtype=self.dtype, device=self.device)
            else:
                raise RuntimeError(f"Invalid type of world_origin={type(world_origin)}")
            self.world_origin = world_origin
        if world_block_size is not None:
            if isinstance(world_block_size, Number):
                world_block_size = [world_block_size] * 3
            if isinstance(world_block_size, (list, tuple, np.ndarray)):
                world_block_size = torch.tensor(world_block_size, dtype=self.dtype, device=self.device)
            elif isinstance(world_block_size, torch.Tensor):
                world_block_size = world_block_size.to(dtype=self.dtype, device=self.device)
            else:
                raise RuntimeError(f"Invalid type of world_block_size={type(world_block_size)}")
            self.world_block_size = world_block_size

        #------- Reset forest meta
        spc = self.spc = octree_to_spc_ins(octree)
        if level is None: level = spc.pyramids.shape[-1] - 2
        if max_length is None: max_length = 2**level
        # This level only
        block_ks = spc.point_hierarchies[spc.pyramids[0,1,level].item() : spc.pyramids[0,1,level+1].item()].contiguous()

        meta = ForestMeta()
        meta.n_trees = block_ks.shape[0]
        meta.level = level
        meta.level_poffset = spc.pyramids[0,1,level].item()
        meta.world_block_size = self.world_block_size.tolist()
        meta.world_origin = self.world_origin.tolist()
        meta.resolution = [max_length]*3
        meta.octree = spc.octrees
        meta.exsum = spc.exsum
        meta.block_ks = block_ks
        meta.continuity_enabled = self.continuity_enabled
        self.meta = meta

        self._octree.resize_(octree.shape)
        self._octree[:] = octree
        self._max_length[:] = check_to_torch([max_length], ref=self._max_length)

    @torch.no_grad()
    def populate(self, mode: Literal[''], **kwargs):
        if mode == 'single_block':
            self.populate_single_block(**kwargs)
        elif mode == 'dense':
            self.populate_dense(**kwargs)
        elif mode == 'from_corners':
            self.populate_from_corners(**kwargs)
        else:
            raise RuntimeError(f"Invalid mode={mode}")
        
        # NOTE: Only set during population. 
        #       For the rest of the lifecycle, this should remain constant,  a fixed value stored and loaded alongside the network parameters.
        # NOTE: This serves a similar purpose to 'radius3d_original' in AABB. It stores the original coordinate scale from initialization (pretraining), \
        #       which remains unchanges despite potentional parameter space scaling caused by rescale_volume / block downscaling, 
        #       ensures consistency in the definition of `nablas` and `sdf` before and after rescale_volume / block downscaling.
        # `radius3d_original = world_block_size0 / 2.`
        self.world_block_size0 = self.world_block_size
        self.world_origin0 = self.world_origin

    @torch.no_grad()
    def populate_single_block(self, world_origin=None, world_block_size=None):
        octree = create_octree_root_only(devide=self.device)
        self.reset(octree, level=0, world_origin=world_origin, world_block_size=world_block_size)

    @torch.no_grad()
    def populate_dense(self, level, world_origin=None, world_block_size=None):
        octree = create_octree_dense(level, device=self.device)
        self.reset(octree, level=level, world_origin=world_origin, world_block_size=world_block_size)

    @torch.no_grad()
    def populate_from_corners(
        self, corners: torch.Tensor, *, 
        level: int=None, world_origin=None, world_block_size=None):
        """
        Initialize from integer octree corners
        """
        corners = check_to_torch(corners, dtype=torch.short, device=self.device).contiguous()
        if level is None:
            level = int(np.log2(corners.max().item()))+1
            max_length = corners.max().item()+1
        else:
            max_length = 2**level
        octree = unbatched_points_to_octree(corners, level=level, sorted=False)
        self.reset(octree, level=level, max_length=max_length, world_origin=world_origin, world_block_size=world_block_size)

    @torch.no_grad()
    def populate_from_mesh(self, level: int, ):
        """
        This functionality should be in the split_block.py of each dataset, or given when creating the dataset, not here.
        """
        # TODO: Perhaps the level is determined automatically.
        raise NotImplementedError
    
    @torch.no_grad()
    def populate_from_waypoints(self, pts: torch.Tensor, dilate_size: float):
        """
        Dilate a forest from camera tracks
        This functionality should be in the split_block.py of each dataset, or given when creating the dataset, not here.
        Args:
            tracks: [n_pts, 3]
        """
        pass

    @torch.no_grad()
    def populate_from_pinhole_cameras(self, c2ws: torch.Tensor, intrs: torch.Tensor, far: float):
        """
        This functionality should be in the split_block.py of each dataset, or given when creating the dataset, not here.
        """
        pass

    def pidx2blidx_unsafe(self, pidx: torch.Tensor):
        return pidx - self.spc.pyramids[0,1,self.level]

    def pidx2blidx(self, pidx: torch.Tensor):
        invalid_i = (pidx==-1).nonzero(as_tuple=True)
        blidx = pidx - self.spc.pyramids[0,1,self.level]
        blidx[invalid_i] = -1
        return blidx

    def blidx2pidx_unsafe(self, blidx: torch.LongTensor):
        return blidx + self.spc.pyramids[0,1,self.level]

    def normalize_coords_01(self, world_coords: torch.Tensor, block_inds: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        dtype = world_coords.dtype
        world_origin, world_block_size = self.world_origin.to(dtype), self.world_block_size.to(dtype)
        coords_ = (world_coords - world_origin).div_(world_block_size) # [0 ~ num_blocks-1]
        if block_inds is None:
            prefix_shape = coords_.shape[:-1]
            # NOTE: blidx = -1 for non-valid blocks.
            pidx = unbatched_query(self.spc.octrees, self.spc.exsum, coords_.reshape(-1, 3).int(),
                                   self.level, with_parents=False)
            blidx = self.pidx2blidx(pidx)
            blidx = blidx.reshape(prefix_shape)
        else:
            blidx = block_inds
        # To per-block [0,1]
        # NOTE: If there are -1 in blidx, the corresponding coords should not be used!
        coords_in_block = coords_ - self.block_ks[blidx].to(dtype) #[0 ~ 1]
        return coords_in_block, blidx

    def normalize_coords(self, world_coords: torch.Tensor, block_inds: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        coords_in_block, blidx = self.normalize_coords_01(world_coords, block_inds)
        # To per-block [-1,1]
        return coords_in_block.mul_(2.).subtract_(1.), blidx

    def unnormalize_coords(self, coords_in_block: torch.Tensor, block_inds: Union[torch.Tensor, int]) -> torch.Tensor:
        # `coords_in_block` should be in range [-1,1]
        dtype = coords_in_block.dtype
        world_origin, world_block_size = self.world_origin.to(dtype), self.world_block_size.to(dtype)
        coords_ = (coords_in_block.add(1.).div_(2.) + self.block_ks[block_inds].to(dtype))
        world_coords = coords_.mul_(world_block_size).add_(world_origin)
        return world_coords

    # def normalize_rays_01(self, rays_o: torch.Tensor, rays_d: torch.Tensor, block_inds: torch.Tensor):
    #     """
    #     Such that [new_rays_o + new_rays_d * real_depth] is directly in block per-block range [0,1]
    #     NOTE: the norm of rays_d has changed.
    #     """
    #     dtype = rays_o.dtype
    #     world_origin, world_block_size = self.world_origin.to(dtype), self.world_block_size.to(dtype)
    #     block_coords = self.block_ks[block_inds].to(dtype) * world_block_size + world_origin
    #     return (rays_o - block_coords) / world_block_size, rays_d / world_block_size

    # def normalize_rays(self, rays_o: torch.Tensor, rays_d: torch.Tensor, block_inds: torch.Tensor):
    #     """
    #     Such that [new_rays_o + new_rays_d * real_depth] is directly in block per-block range [-1,1]
    #     NOTE: the norm of rays_d has changed.
    #     """
    #     dtype = rays_o.dtype
    #     world_origin, world_block_size = self.world_origin.to(dtype), self.world_block_size.to(dtype)
    #     block_coords = self.block_ks[block_inds].to(dtype) * world_block_size + world_origin
    #     return (rays_o-block_coords)/(world_block_size/2.) - 1., rays_d/(world_block_size/2.)

    def sample_pts_uniform(self, num_pts: int = None, num_pts_per_block: int = None) -> Union[torch.Tensor, torch.Tensor]:
        if num_pts_per_block is None:
            num_pts_per_block = int(num_pts // self.n_trees) + 1
        block_x = torch.empty([self.n_trees, num_pts_per_block, 3], dtype=self.dtype, device=self.device).uniform_(-1,1)
        blidx = torch.arange(self.n_trees, device=self.device).unsqueeze_(-1).expand(-1, num_pts_per_block).contiguous()
        return block_x, blidx

    """
    NOTE: kaolin's octree ray trace is very buggy right now. 
          For now, we directly use ray-boxes intersections instead; see below.
    def ray_test(
        self, rays_o: torch.Tensor, rays_d: torch.Tensor, near: float=None, far: float=None, return_rays=True):
        # TODO: Here we should rescale based on the difference between the product of the current actual self._max_length and 2**level.
        # To forest [-1,1]
        world_origin, world_block_size = self.world_origin.to(rays_o.dtype), self.world_block_size.to(rays_o.dtype)
        world_scale = world_block_size * self._max_length / 2.
        rays_o_forest, rays_d_forest = (rays_o-world_origin)/world_scale - 1., rays_d/world_scale
        # Octree ray trace
        with torch.no_grad():
            ridx, pidx, depth = octree_raytrace_fixed_from_kaolin(
                self.spc.octrees, self.spc.point_hierarchies, self.spc.pyramids[0], self.spc.exsum, rays_o_forest, rays_d_forest, level=self.level, 
                return_depth=True, with_exit=True, include_head=True)
            
            if near is not None or far is not None:
                segment_pack_infos_0 = get_pack_infos_from_boundary(mark_pack_boundaries(ridx))
                if isinstance(near, torch.Tensor):
                    near = torch.repeat_interleave(near, segment_pack_infos_0[:,1], dim=0)
                if isinstance(far, torch.Tensor):
                    far = torch.repeat_interleave(far, segment_pack_infos_0[:,1], dim=0)
                if near is not None and far is not None:
                    nidx = (depth[:,1] > near) & (depth[:,0] < far)
                elif near is not None:
                    nidx = depth[:,1] > near
                elif far is not None:
                    nidx = depth[:,0] < far
            
                ridx, pidx, depth = ridx[nidx].long().contiguous(), pidx[nidx].long().contiguous(), depth[nidx].clamp_min_(0).contiguous()
            
            boundary = mark_pack_boundaries(ridx)
            first_inds = boundary.nonzero().long()[..., 0]
            segment_pack_infos = get_pack_infos_from_first(first_inds, boundary.numel())
            
            blidx = self.pidx2blidx_unsafe(pidx)
            ridx_hit = ridx[first_inds]
            num_rays = ridx_hit.numel()
            # Set to the first entry if not specified
            near = depth[first_inds, 0].contiguous() if near is None else (rays_o.new_full([num_rays,], near) if not isinstance(near, torch.Tensor) else near)
            # Set to the last exit if not specified
            far = depth[segment_pack_infos.sum(-1).sub_(1), 1].contiguous() if far is None else (rays_o.new_full([num_rays], far) if not isinstance(far, torch.Tensor) else far)
            ret = dict(
                num_rays=num_rays, rays_inds=ridx_hit.long().contiguous(), near=near, far=far, 
                seg_pack_infos=segment_pack_infos.long().contiguous(), seg_block_inds=blidx.long().contiguous(), 
                seg_entries=depth[:, 0].contiguous(), seg_exits=depth[:, 1].contiguous())
        if return_rays:
            ret.update(rays_o=rays_o.index_select(0, ridx_hit), rays_d=rays_d.index_select(0, ridx_hit))
        return ret
    """

    def ray_test(
        self, rays_o: torch.Tensor, rays_d: torch.Tensor, near: float=None, far: float=None, 
        return_rays=True, **extra_ray_data):
        world_origin, world_block_size = self.world_origin.to(rays_o.dtype), self.world_block_size.to(rays_o.dtype)
        with torch.no_grad():
            blocks_aabb_min = self.block_ks * world_block_size + world_origin
            blocks_aabb_max = blocks_aabb_min + world_block_size
            # [N_rays, N_blocks]; The `nonzero()` function returns `ridx` and `blidx`, where `ridx` is consecutive.
            t_near, t_far, check = ray_box_intersection(
                rays_o.unsqueeze(1), rays_d.unsqueeze(1), aabb_min=blocks_aabb_min.unsqueeze(0), aabb_max=blocks_aabb_max.unsqueeze(0), 
                t_min_cons=0. if near is None else near, t_max_cons=far)
            ridx, blidx = check.nonzero(as_tuple=True)
            boundary = mark_pack_boundaries(ridx)
            first_inds = boundary.nonzero().long()[..., 0]
            ridx_hit = ridx[first_inds]
            num_rays = first_inds.numel()
        if num_rays > 0:
            segment_pack_infos = get_pack_infos_from_first(first_inds, boundary.numel())
            
            seg_entries, seg_exits = t_near[ridx, blidx].contiguous(), t_far[ridx, blidx].contiguous()
            # Sort segments according to entry depth of each segment pack
            indices = packed_sort_inplace(seg_entries, segment_pack_infos)
            seg_exits, blidx = seg_exits[indices], blidx[indices]
            
            # Set to the first entry if not specified
            near = seg_entries[first_inds].contiguous() if near is None else (rays_o.new_full([num_rays,], near) if not isinstance(near, torch.Tensor) else near[ridx_hit])
            # Set to the last exit if not specified
            far = seg_exits[segment_pack_infos.sum(-1).sub_(1)].contiguous() if far is None else (rays_o.new_full([num_rays], far) if not isinstance(far, torch.Tensor) else far[ridx_hit])
            ret = dict(
                num_rays=num_rays, rays_inds=ridx_hit.long().contiguous(), near=near, far=far, 
                seg_pack_infos=segment_pack_infos.long().contiguous(), seg_block_inds=blidx.long().contiguous(), 
                seg_entries=seg_entries.contiguous(), seg_exits=seg_exits.contiguous())
            ret.update({k: v[ridx_hit] if isinstance(v, torch.Tensor) else v for k, v in extra_ray_data.items()})
        else:
            ret = dict(num_rays=num_rays, rays_inds=None)
        if return_rays:
            ret.update(rays_o=rays_o.index_select(0, ridx_hit), rays_d=rays_d.index_select(0, ridx_hit))
        return ret

    # TODO: The complete set of octree raymarching can be moved here appropriately.
    #       Consider whether this should be an inheritance of octree, or if the raymarching functions of octree should be abstracted into some common functions.
    # NOTE: update 221110: Directly use some utility functions in octree_raymarch for the octree of the forest, including segment sampling, etc., if it is necessary to directly march at the forest level.
    
    def ray_step_coarse(
        self, 
        rays_o: torch.Tensor, rays_d: torch.Tensor, near: Union[torch.Tensor, float], far: Union[torch.Tensor, float], # [num_rays]
        seg_block_inds: torch.Tensor, seg_entries: torch.Tensor, seg_exits: torch.Tensor, # [num_segments] ray intersected segments with forest blocks
        seg_pack_infos: torch.Tensor, # [num_rays, 2] Pack info of ray intersected segments with forest blocks
        *, step_mode: Literal['linear', 'depth'], **step_kwargs):
        
        if step_mode == 'linear':
            depths, deltas, ridx, pack_infos, sidx, blidx_pack_infos = interleave_sample_step_linear_in_packed_segments(
                near, far, seg_entries, seg_exits, seg_pack_infos, **step_kwargs)
        elif step_mode == 'depth':
            depths, deltas, ridx, pack_infos, sidx, blidx_pack_infos = interleave_sample_step_wrt_depth_in_packed_segments(
                near, far, seg_entries, seg_exits, seg_pack_infos, **step_kwargs)
        else:
            raise RuntimeError(f"Invalid step_mode={step_mode}")

        ridx_hit = ridx[pack_infos[:,0]]
        samples = torch.addcmul(rays_o[ridx], rays_d[ridx], depths.unsqueeze(-1))
        blidx = seg_block_inds[sidx].long().contiguous()
        return RaymarchRetForest(ridx_hit.numel(), ridx_hit, samples, depths, deltas, ridx, pack_infos, blidx, blidx_pack_infos)

    #------------------------------------------------------
    #--------------- DEBUG Functionalities ----------------
    #------------------------------------------------------
    @torch.no_grad()
    def debug_vis(self, draw_lines=False, draw_mesh=True, show=True,  **kwargs):
        from nr3d_lib.plot import vis_occgrid_voxels_o3d
        return vis_occgrid_voxels_o3d(
            self.to_occgrid(), 
            origin=self.world_origin, block_size=self.world_block_size, draw_lines=draw_lines, draw_mesh=draw_mesh, show=show, **kwargs)
        # from vedo import Volume, show
        # from nr3d_lib.models.grid_encodings.utils import voxel_verts
        # max_length = self._max_length.item()
        # spacing = (self.world_block_size / max_length).tolist()
        # origin = self.world_origin.tolist()
        # val_grid = torch.zeros([max_length]*3, dtype=torch.float, device=self.device)
        # val_grid[ tuple((voxel_verts(self.block_ks) + self.block_ks[:,None,:]).flatten(0,-2).t().long().clamp_(0, max_length-1)) ] = 1.
        # vol = Volume(val_grid.data.cpu().numpy(), c=['white','b','g','r'], mapper='gpu', origin=origin, spacing=spacing)
        # vox = vol.legosurface(vmin=0.5, boundary=False)
        # show(vox, axes=1, viewup='z').close()

    def extra_repr(self) -> str:
        world_origin_str = '[' + ', '.join([f"{s:.3f}" for s in self.world_origin.tolist()]) + ']'
        world_block_size_str = '[' + ', '.join([f"{s:.3f}" for s in self.world_block_size.tolist()]) + ']'
        max_length = self._max_length.item()
        return f"n_trees={self.n_trees}, level={self.level}, idx_max={self.block_ks.max(0).values.tolist()}, "+\
                f"max_length={max_length}, world_origin={world_origin_str}, world_block_size={world_block_size_str}"

if __name__ == "__main__":
    def unit_test(device=torch.device('cuda')):
        forest_level = 5
        space = ForestBlockSpace(device=device)
        occ_grid = torch.rand([2**forest_level]*3, device=device) > 0.5
        corners = occ_grid.nonzero().short()
        space.populate('from_corners', level=forest_level, corners=corners)
        space.debug_vis()
    unit_test()
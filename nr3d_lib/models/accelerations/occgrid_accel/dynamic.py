___all___ = [
    'OccGridAccelDynamic', 
    'OccGridAccelStaticAndDynamic'
]

import math
import numpy as np
from numbers import Number
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn

from nr3d_lib.logger import Logger
from nr3d_lib.config import ConfigDict
from nr3d_lib.utils import check_to_torch, tensor_statistics, torch_consecutive_nearest1d

from nr3d_lib.models.spatial import AABBDynamicSpace, AABBSpace
from nr3d_lib.models.accelerations.occgrid import OccGridEmaBatched, OccGridEma
from nr3d_lib.models.accelerations.occgrid_accel.single import OccGridAccel

from nr3d_lib.graphics.raymarch import RaymarchRetDynamic
from nr3d_lib.graphics.raymarch.occgrid_raymarch import occgrid_raymarch_batched

class OccGridAccelDynamic(nn.Module):
    def __init__(
        self, 
        space: AABBDynamicSpace, 
        ts_keyframes: torch.Tensor, 
        resolution: Union[int, List[int], torch.Tensor] = None, vox_size: float = None, 
        dtype=torch.float, device=None, 
        **occ_kwargs) -> None:
        super().__init__()
        
        #------- Valid representing space 
        assert isinstance(space, AABBDynamicSpace), f"{self.__class__.__name__} expects space of AABBDynamicSpace"
        self.space: AABBDynamicSpace = space
        self.dtype = dtype
        
        #------- Temporal information
        self.num_frames = len(ts_keyframes)
        ts_keyframes = check_to_torch(ts_keyframes, device=device)
        self.register_buffer('ts_keyframes', ts_keyframes, persistent=True)
        
        #------- Occupancy information
        assert bool(resolution is not None) != bool(vox_size is not None), \
            "Please specify `vox_size` or `resolution` for OccGridAccel."
        
        if resolution is None:
            resolution = (self.space.radius3d * 2 / vox_size).long()
        
        self.occ = OccGridEmaBatched(
            num_batches=self.num_frames, 
            resolution=resolution, **occ_kwargs, 
            dtype=self.dtype, device=device
        )
        
        self.training_granularity = 0.0 # No level/resolution annealing

    @property
    def device(self) -> torch.device:
        return self.space.device

    @property
    def NUM_DIM(self) -> int:
        return self.occ.NUM_DIM
    
    @property
    def resolution(self):
        return self.occ.resolution
    
    """ Updating and using occ """
    def get_occ_grid(self):
        return self.occ.occ_grid
    
    @torch.no_grad()
    def init(self, val_query_fn_normalized_x_ts, logger: Logger = None):
        def val_query_fn_converted(x, *, bidx=...):
            ts = self.ts_keyframes[bidx] # `bidx` actually represents `fidx`
            return val_query_fn_normalized_x_ts(x, ts=ts)
        updated = self.occ.init(val_query_fn_converted, logger=logger)
    
    @torch.no_grad()
    def step(self, cur_it: int, val_query_fn_normalized_x_ts, logger: Logger = None):
        def val_query_fn_converted(x, *, bidx=...):
            ts = self.ts_keyframes[bidx] # `bidx` actually represents `fidx`
            return val_query_fn_normalized_x_ts(x, ts=ts)
        updated = self.occ.step(cur_it, val_query_fn_converted, logger=logger)
    
    @torch.no_grad()
    def collect_samples(self, pts: torch.Tensor, ts: torch.Tensor, val: torch.Tensor, normalized=True):
        """
        NOTE: `collect_samples` should be invoked like a forward-hook function.
        """
        if self.training:
            if not normalized:
                pts = self.space.normalize_coords(pts)
            # NOTE: For now, using simple nearest solution.
            fidx = torch_consecutive_nearest1d(self.ts_keyframes, ts)[0]
            # Pass `fidx` to `bidx`
            self.occ.collect_samples(pts, fidx, val)
    
    @torch.no_grad()
    def sample_pts_in_occupied(self, num_pts: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pts, fidx = self.occ.sample_pts_in_occupied(num_pts)
        ts = self.ts_keyframes[fidx]
        return pts, ts
    
    @torch.no_grad()
    def query_occupancy(self, pts: torch.Tensor, ts: torch.Tensor) -> torch.BoolTensor:
        # Expect pts in range [-1,1]
        # NOTE: For now, using simple nearest solution.
        fidx = torch_consecutive_nearest1d(self.ts_keyframes, ts)[0]
        return self.occ.query(pts, fidx)
    
    """ ray marching """
    def ray_march(
        self, 
        rays_o: torch.Tensor, rays_d: torch.Tensor, rays_ts: torch.Tensor, near=None, far=None, *, perturb=False, normalized=True, 
        step_size: float = 1e-3, max_step_size: float = 1e10, dt_gamma: float = 0.0, max_steps: int = 512
        ) -> RaymarchRetDynamic:
        if not normalized: 
            rays_o, rays_d = self.space.normalize_rays(rays_o, rays_d)
        
        # Get the nearest frame ind for each ray
        rays_fidx = torch_consecutive_nearest1d(self.ts_keyframes, rays_ts)[0]
        
        # March multi-frame occ grid by viewing them as multi-batch versions, with each batch=each frame
        ret = occgrid_raymarch_batched(
            self.get_occ_grid(), rays_o, rays_d, rays_fidx, near, far, 
            perturb=perturb, step_size=step_size, max_step_size=max_step_size, dt_gamma=dt_gamma, max_steps=max_steps
        )
        
        # Re-assemble the returns
        if ret.bidx is not None:
            ret_ts = self.ts_keyframes[ret.bidx]
        else:
            ret_ts = None
        ret = RaymarchRetDynamic(
            num_hit_rays=ret.num_hit_rays, 
            ridx_hit=ret.ridx_hit, 
            samples=ret.samples, 
            depth_samples=ret.depth_samples, 
            deltas=ret.deltas, 
            ridx=ret.ridx, 
            pack_infos=ret.pack_infos, 
            ts=ret_ts, 
            gidx=ret.gidx, 
            gidx_pack_infos=ret.gidx_pack_infos
        )
        return ret

    """ DEBUG Functionalities """
    @torch.no_grad()
    def debug_stats(self) -> Dict[str, Number]:
        occ_grid = self.get_occ_grid()
        #---- 360 us for [20,32,32,32]
        num_occupied = occ_grid.sum().item()
        frac_occupied = num_occupied / occ_grid.numel()
        num_occupied_per_frame = occ_grid.sum(dim=tuple(range(1,self.NUM_DIM+1,1)))
        num_empty_frame = (num_occupied_per_frame == 0).sum().item()
        num_occupied_per_nonempty_frame = num_occupied_per_frame[num_occupied_per_frame != 0]
        frac_occupied_per_nonempty_frame = num_occupied_per_nonempty_frame / math.prod(occ_grid.shape[1:])
        return {
            'num_occ': num_occupied, 
            'frac_occ': frac_occupied, 
            'num_empty_frame': num_empty_frame, 
            **tensor_statistics(num_occupied_per_nonempty_frame, 'per_frame.nonempty.num_occupied', metrics=['mean', 'min', 'max', 'std']), 
            **tensor_statistics(frac_occupied_per_nonempty_frame, 'per_frame.nonempty.frac_occupied', metrics=['mean', 'min', 'max', 'std']), 
        }

    @torch.no_grad()
    def debug_vis(self, draw=True):
        from vedo import Volume, show
        batched_val_grid = self.occ.occ_val_grid.data.cpu().numpy()
        actors = []
        val_thre = self.occ.occ_thre
        origin0 = self.space.aabb[0].tolist()
        radius3d = self.space.radius3d_original
        spacing = ((self.space.aabb[1]-self.space.aabb[0]) / self.resolution).tolist()
        for fidx, val_grid in enumerate(batched_val_grid[::4]):
            if not (val_grid > val_thre).any():
                continue
            # NOTE: Different bidx on y-axis
            origin = [origin0[0], origin0[1] + fidx * radius3d[0] * 3, origin0[2]]
            
            # Old API
            # vol = Volume(val_grid, c=['white','b','g','r'], mapper='gpu', origin=origin, spacing=spacing)
            # New API
            vol = Volume(val_grid, origin=origin, spacing=spacing).color(['white','b','g','r'])
            vol.mapper = 'gpu'
            
            vox = vol.legosurface(vmin=val_thre, vmax=None, boundary=True)
            vox.cmap('GnBu', on='cells', vmin=val_thre, vmax=1.).add_scalarbar()
            actors.append(vox)
        if draw:
            show(*actors, __doc__, axes=1, viewup='z').close()
        else:
            return actors

class OccGridAccelStaticAndDynamic(nn.Module):
    """
    Combined occgrid (static + dynamic)
    """
    def __init__(
        self, 
        space: AABBDynamicSpace, 
        ts_keyframes: torch.Tensor, 
        resolution: Union[int, List[int], torch.Tensor] = None, vox_size: float = None, 
        occ_static_cfg: dict = ..., occ_dynamic_cfg: dict = ..., 
        dtype=torch.float, device=None, 
        **occ_common_kwargs
        ) -> None:        
        super().__init__()
        
        #------- Valid representing space 
        assert isinstance(space, AABBDynamicSpace), f"{self.__class__.__name__} expects space of AABBDynamicSpace"
        self.space: AABBDynamicSpace = space
        self.dtype = dtype
        
        #------- Temporal information
        self.num_frames = len(ts_keyframes)
        ts_keyframes = check_to_torch(ts_keyframes, device=device)
        self.register_buffer('ts_keyframes', ts_keyframes, persistent=True)
        
        #------- Occupancy information
        assert bool(resolution is not None) != bool(vox_size is not None), \
            "Please specify `vox_size` or `resolution` for OccGridAccel."
        
        if resolution is None:
            resolution = (self.space.radius3d * 2 / vox_size).long()
        
        self.occ = None
        # For static
        self.occ_static = OccGridEma(
            resolution=resolution, 
            **occ_static_cfg, **occ_common_kwargs, 
            dtype=self.dtype, device=device
        )
        # For dynamic
        self.occ_dynamic = OccGridEmaBatched(
            num_batches=self.num_frames, 
            resolution=resolution, 
            **occ_dynamic_cfg, **occ_common_kwargs, 
            dtype=self.dtype, device=device
        )
        
        self.training_granularity = 0.0 # No level/resolution annealing

    @property
    def device(self) -> torch.device:
        return self.space.device

    @property
    def NUM_DIM(self) -> int:
        return self.occ_static.NUM_DIM
    
    @property
    def resolution(self):
        return self.occ_static.resolution

    """ Updating and using occ """
    def get_occ_grid(self):
        occ_grid_static = self.occ_static.occ_grid
        occ_grid_dynamic = self.occ_dynamic.occ_grid
        occ_grid = occ_grid_dynamic | occ_grid_static.unsqueeze(0).expand_as(occ_grid_dynamic)
        return occ_grid
    
    @torch.no_grad()
    def init(
        self, 
        val_query_fn_static_normalized_x, 
        val_query_fn_dynamic_normalized_x_ts, 
        logger: Logger = None):
        def val_query_fn_converted(x, *, bidx=...):
            ts = self.ts_keyframes[bidx] # `bidx` actually represents `fidx`
            return val_query_fn_dynamic_normalized_x_ts(x, ts=ts)
        updated = self.occ_dynamic.init(val_query_fn_converted, logger=logger)
        updated |= self.occ_static.init(val_query_fn_static_normalized_x, logger=logger)
    
    @torch.no_grad()
    def step(
        self, cur_it: int, 
        val_query_fn_static_normalized_x, 
        val_query_fn_dynamic_normalized_x_ts, 
        logger: Logger = None):
        def val_query_fn_converted(x, *, bidx=...):
            ts = self.ts_keyframes[bidx] # `bidx` actually represents `fidx`
            return val_query_fn_dynamic_normalized_x_ts(x, ts=ts)
        updated = self.occ_dynamic.step(cur_it, val_query_fn_converted, logger=logger)
        updated |= self.occ_static.step(cur_it, val_query_fn_static_normalized_x, logger=logger)
    
    @torch.no_grad()
    def collect_samples(
        self, pts: torch.Tensor, ts: torch.Tensor, 
        val_static: torch.Tensor, val_dynamic: torch.Tensor, normalized=True):
        """
        NOTE: `collect_samples` should be invoked like a forward-hook function.
        """
        if self.training:
            if not normalized:
                pts = self.space.normalize_coords(pts)
            # NOTE: For now, using simple nearest solution.
            fidx = torch_consecutive_nearest1d(self.ts_keyframes, ts)[0]
            # Pass `fidx` to `bidx`
            self.occ_dynamic.collect_samples(pts, fidx, val_dynamic)
            self.occ_static.collect_samples(pts, val_static)
    
    @torch.no_grad()
    def sample_pts_in_occupied(self, num_pts: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
        pts, fidx = self.occ_dynamic.sample_pts_in_occupied(num_pts)
        ts = self.ts_keyframes[fidx]
        pts = self.occ_static.sample_pts_in_occupied(num_pts)
        return pts, ts
    
    @torch.no_grad()
    def query_occupancy(self, pts: torch.Tensor, ts: torch.Tensor) -> torch.BoolTensor:
        # Expect pts in range [-1,1]
        # NOTE: For now, using simple nearest solution.
        fidx = torch_consecutive_nearest1d(self.ts_keyframes, ts)[0]
        pts_occupied_dynamic = self.occ_dynamic.query(pts, fidx)
        pts_occupied_static = self.occ_static.query(pts)
        pts_occupied = pts_occupied_static | pts_occupied_dynamic
        return pts_occupied

    """ ray marching """
    def ray_march(
        self, 
        rays_o: torch.Tensor, rays_d: torch.Tensor, rays_ts: torch.Tensor, near=None, far=None, *, perturb=False, normalized=True, 
        step_size: float = 1e-3, max_step_size: float = 1e10, dt_gamma: float = 0.0, max_steps: int = 512
        ) -> RaymarchRetDynamic:
        if not normalized: 
            rays_o, rays_d = self.space.normalize_rays(rays_o, rays_d)
        
        occ_grid = self.get_occ_grid()
        
        # Get the nearest frame ind for each ray
        rays_fidx = torch_consecutive_nearest1d(self.ts_keyframes, rays_ts)[0]
        
        # March multi-frame occ grid by viewing them as multi-batch versions, with each batch=each frame
        ret = occgrid_raymarch_batched(
            occ_grid, rays_o, rays_d, rays_fidx, near, far, 
            perturb=perturb, step_size=step_size, max_step_size=max_step_size, dt_gamma=dt_gamma, max_steps=max_steps
        )
        
        # Re-assemble the returns
        if ret.bidx is not None:
            ret_ts = self.ts_keyframes[ret.bidx]
        else:
            ret_ts = None
        ret = RaymarchRetDynamic(
            num_hit_rays=ret.num_hit_rays, 
            ridx_hit=ret.ridx_hit, 
            samples=ret.samples, 
            depth_samples=ret.depth_samples, 
            deltas=ret.deltas, 
            ridx=ret.ridx, 
            pack_infos=ret.pack_infos, 
            ts=ret_ts, 
            gidx=ret.gidx, 
            gidx_pack_infos=ret.gidx_pack_infos
        )
        return ret

    """ Seperate static / dynamic model """
    def get_accel_static(self):
        space = AABBSpace(aabb=self.space.aabb, dtype=self.dtype, device=self.device)
        accel = OccGridAccel(space=space, resolution=self.resolution, dtype=self.dtype, device=self.device)
        return accel
    
    def get_accel_dynamic(self):
        accel = OccGridAccelDynamic(space=self.space, resolution=self.resolution, dtype=self.dtype, device=self.device)
        return accel

    """ DEBUG Functionalities """
    @torch.no_grad()
    def debug_stats(self) -> Dict[str, Number]:
        ret = {}
        occ_grid_static = self.occ_static.occ_grid
        num_occupied = occ_grid_static.sum().item()
        frac_occupied = num_occupied / occ_grid_static.numel()
        ret.update({
            'static.num_occupied': num_occupied, 
            'static.frac_occupied': frac_occupied, 
        })
        occ_grid_dynamic = self.occ_dynamic.occ_grid
        num_occupied = occ_grid_dynamic.sum().item()
        frac_occupied = num_occupied / occ_grid_dynamic.numel()
        num_occupied_per_frame = occ_grid_dynamic.sum(dim=tuple(range(1,self.NUM_DIM+1,1)))
        num_empty_frame = (num_occupied_per_frame == 0).sum().item()
        num_occupied_per_nonempty_frame = num_occupied_per_frame[num_occupied_per_frame != 0]
        frac_occupied_per_nonempty_frame = num_occupied_per_nonempty_frame / math.prod(occ_grid_dynamic.shape[1:])
        ret.update({
            'dynamic.num_occ': num_occupied, 
            'dynamic.frac_occ': frac_occupied, 
            'dynamic.num_empty_frame': num_empty_frame, 
            **tensor_statistics(num_occupied_per_nonempty_frame, 'dynamic.per_frame.nonempty.num_occupied', metrics=['mean', 'min', 'max', 'std']), 
            **tensor_statistics(frac_occupied_per_nonempty_frame, 'dynamic.per_frame.nonempty.frac_occupied', metrics=['mean', 'min', 'max', 'std']), 
        })
        return ret

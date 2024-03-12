"""
@file   occgrid_batch_accel.py
@author Jianfei Guo, Shanghai AI Lab
@brief  For a batched-blocks space; Acceleration struct based on occupancy grid.
"""

__all__ = [
    'OccGridAccelBatched_Base', 
    'OccGridAccelBatched_Ema', 
    'OccGridAccelBatched_Getter'
]

import math
from numbers import Number
from typing import Dict, List, Literal, Tuple, Union
from nr3d_lib.utils import tensor_statistics

import torch
import torch.nn as nn

from nr3d_lib.config import ConfigDict
from nr3d_lib.logger import Logger

from nr3d_lib.models.spatial import BatchedBlockSpace
from nr3d_lib.models.accelerations.occgrid import OccGridGetter, OccGridEmaBatched, \
    sample_pts_in_voxels, err_msg_empty_occ

from nr3d_lib.graphics.raymarch import RaymarchRetBatched
from nr3d_lib.graphics.raymarch.occgrid_raymarch import occgrid_raymarch_batched

class OccGridAccelBatched_Base(nn.Module):
    @torch.no_grad()
    def cur_batch__sample_pts_in_occupied(self, num_pts: int) -> Tuple[torch.Tensor, torch.LongTensor]:
        """ Uniformly sample points in the occupied region of *current batch*

        Args:
            num_pts (int): _description_

        Returns:
            Tuple[torch.Tensor, torch.LongTensor]: _description_
        """
        assert self.occ_grid_per_batch is not None, \
            "Please call set_condition() first before sample_pts_in_occupied()"
        idx_nonempty = self.occ_grid_per_batch.nonzero().long()
        assert idx_nonempty.numel() > 0, err_msg_empty_occ
        bidx_nonempty, gidx_nonempty = idx_nonempty[..., 0], idx_nonempty[..., 1:]
        pts, vidx = sample_pts_in_voxels(gidx_nonempty, num_pts, resolution=self.resolution, dtype=self.dtype)
        bidx = bidx_nonempty[vidx]
        # bidx = self.ins_inds_per_batch[bidx] # For global bidx
        return pts, bidx

    @torch.no_grad()
    def cur_batch__query_occupancy(self, pts: torch.Tensor, bidx: torch.LongTensor) -> torch.BoolTensor:
        """_summary_

        Args:
            pts (torch.Tensor): Expected to be in range [-1,1]
            bidx (torch.LongTensor): The batch indices of current batch

        Returns:
            torch.BoolTensor: _description_
        """
        assert self.occ_grid_per_batch is not None, \
            "Please call set_condition() first before query_occupancy()"
        gidx = ((pts/2.+0.5) * self.resolution).long().clamp(self.resolution.new_tensor([0]), self.resolution-1)
        return self.occ_grid_per_batch[(bidx,)+tuple(gidx.movedim(-1,0))]

    def cur_batch__ray_march(
        self, 
        rays_o: torch.Tensor, rays_d: torch.Tensor, rays_bidx: torch.Tensor = None, # per ray
        *, near=None, far=None, perturb=False, 
        step_size: float = 1e-3, max_step_size: float = 1e10, dt_gamma: float = 0.0, max_steps: int = 512
        ) -> RaymarchRetBatched:
        assert self.occ_grid_per_batch is not None, \
            "Please call set_condition() first before ray_march()"
        # March multi-ins occ grid
        ret = occgrid_raymarch_batched(
            self.occ_grid_per_batch, 
            rays_o, rays_d, rays_bidx, 
            near, far, perturb=perturb, 
            step_size=step_size, max_step_size=max_step_size, dt_gamma=dt_gamma, max_steps=max_steps
        )
        return ret

class OccGridAccelBatched_Ema(OccGridAccelBatched_Base):
    def __init__(
        self, 
        space: BatchedBlockSpace, 
        num_batches: int, 
        resolution: Union[int, List[int], torch.Tensor] = None, 
        dtype=torch.float, device=None, 
        **occ_kwargs) -> None:

        super().__init__()
        self.training_granularity = 0.0 # No level/resolution annealing

        #------- Valid representing space 
        assert isinstance(space, BatchedBlockSpace), f"{self.__class__.__name__} expects space of BatchedBlockSpace"
        self.space: BatchedBlockSpace = space
        self.dtype = dtype
        
        #------- Occupancy information
        self.num_batches = num_batches
        self.occ = OccGridEmaBatched(**occ_kwargs, num_batches=num_batches, resolution=resolution, dtype=self.dtype, device=device)
        self.clean_condition()

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
    
    def set_condition(
        self, batch_size: int, *, 
        ins_inds_per_batch: torch.LongTensor = None, 
        val_query_fn_normalized_x_bi = None):
        
        assert ins_inds_per_batch is not None, f"`ins_inds_per_batch` is required for {type(self)}"
        self.batch_size = batch_size
        self.ins_inds_per_batch = ins_inds_per_batch
        self.occ_grid_per_batch = self.get_occ_grid()[ins_inds_per_batch].contiguous()

    def clean_condition(self):
        self.batch_size = None
        self.ins_inds_per_batch = None
        self.occ_grid_per_batch = None

    @torch.no_grad()
    def init(self, val_query_fn_normalized_x_bi, logger: Logger = None):
        assert (self.batch_size is not None) and (self.batch_size == self.num_batches), \
            "Before init(), should set_condition() on all batches (i.e. all latents)."
        # NOTE: In this case, the bidx passed here represents the global batch inds
        updated = self.occ.init(val_query_fn_normalized_x_bi, logger=logger)

    @torch.no_grad()
    def cur_batch__step(self, cur_it: int, val_query_fn_normalized_x_bi, logger: Logger = None):
        assert self.ins_inds_per_batch is not None, \
            "Please call set_condition() first before step()"
        updated = self.occ.step(cur_it, val_query_fn_normalized_x_bi, within_bi=self.ins_inds_per_batch, logger=logger)

    @torch.no_grad()
    def cur_batch__collect_samples(self, pts: torch.Tensor, bidx: torch.LongTensor, val: torch.Tensor, normalized=True):
        """
        NOTE: Should be invoked like a forward-hook function.
        Args:
            pts (torch.Tensor): _description_
            bidx (torch.LongTensor): _description_
            val (torch.Tensor): _description_
            normalized (bool, optional): _description_. Defaults to True.
        """
        if self.training:
            assert self.ins_inds_per_batch is not None, \
                "Please call set_condition() first before collect_samples()"
            if not normalized:
                pts = self.space.cur_batch__normalize_coords(pts, bidx)
            # NOTE: `bidx` is actually the local bidx
            global_bidx = self.ins_inds_per_batch[bidx]
            self.occ.collect_samples(pts, global_bidx, val)

    """ DEBUG Functionalities """
    @torch.no_grad()
    def debug_stats(self) -> Dict[str, Number]:
        occ_grid = self.get_occ_grid()
        #---- 360 us for [20,32,32,32]
        num_occupied = occ_grid.sum().item()
        frac_occupied = num_occupied / occ_grid.numel()
        num_occupied_per_ins = occ_grid.sum(dim=tuple(range(1,self.NUM_DIM+1,1)))
        num_empty_ins = (num_occupied_per_ins == 0).sum().item()
        num_occupied_per_nonempty_ins = num_occupied_per_ins[num_occupied_per_ins != 0]
        frac_occupied_per_nonempty_ins = num_occupied_per_nonempty_ins / math.prod(occ_grid.shape[1:])
        return {
            'num_occ': num_occupied, 
            'frac_occ': frac_occupied, 
            'num_empty_ins': num_empty_ins, 
            **tensor_statistics(num_occupied_per_nonempty_ins, 'per_ins.nonempty.num_occupied', metrics=['mean', 'min', 'max', 'std']), 
            **tensor_statistics(frac_occupied_per_nonempty_ins, 'per_ins.nonempty.frac_occupied', metrics=['mean', 'min', 'max', 'std']), 
        }

    @torch.no_grad()
    def debug_vis(self, draw=True):
        from vedo import Volume, show
        # NOTE: Primary drawing function
        # from vedo.applications import IsosurfaceBrowser
        
        # batched_val_grid = self.occ.occ_val_grid[self.ins_inds_per_batch].data.cpu().numpy()
        # actors = []
        # val_thre = self.occ.occ_thre
        # origin0 = self.space.aabb[0].tolist()
        # # radius3d = self.space.radius3d_original
        # batched_radius3d = self.radius3d
        # mean_radius3d = self.radius3d.mean(0)
        # for bidx, val_grid in enumerate(batched_val_grid):
        #     if not (val_grid > val_thre).any():
        #         continue
        #     # NOTE: Different bidx on x-axis
        #     spacing = (2 * batched_radius3d[bidx] / self.resolution).tolist()
        #     origin = [origin0[0] + bidx * mean_radius3d[0].item() * 3, origin0[1], origin0[2]]
        #     vol = Volume(val_grid, c=['white','b','g','r'], mapper='gpu', origin=origin, spacing=spacing)
        #     vox = vol.legosurface(vmin=val_thre, vmax=1., boundary=True)
        #     vox.cmap('GnBu', on='cells', vmin=val_thre, vmax=1.).add_scalarbar()
        #     actors.append(vox)
        # show(*actors, __doc__, axes=1, viewup='z').close()
        
        batched_val_grid = self.occ.occ_val_grid.data.cpu().numpy()
        actors = []
        val_thre = self.occ.occ_thre
        origin0 = self.space.aabb[0].tolist()
        radius3d = self.space.radius3d_original
        spacing = ((self.space.aabb[1]-self.space.aabb[0]) / self.resolution).tolist()
        for bidx, val_grid in enumerate(batched_val_grid):
            if not (val_grid > val_thre).any():
                continue
            # NOTE: Different bidx on x-axis
            origin = [origin0[0] + bidx * radius3d[0] * 3, origin0[1], origin0[2]]
            vol = Volume(val_grid, c=['white','b','g','r'], mapper='gpu', origin=origin, spacing=spacing)
            vox = vol.legosurface(vmin=val_thre, vmax=1., boundary=True)
            vox.cmap('GnBu', on='cells', vmin=val_thre, vmax=1.).add_scalarbar()
            actors.append(vox)
        if draw:
            show(*actors, __doc__, axes=1, viewup='z').close()
        else:
            return actors

class OccGridAccelBatched_Getter(OccGridAccelBatched_Base):
    def __init__(
        self, 
        space: BatchedBlockSpace, 
        resolution: Union[int, List[int], torch.Tensor] = None, 
        dtype=torch.float, device=None, 
        **occ_kwargs) -> None:
        
        super().__init__()
        self.training_granularity = 0.0 # No level/resolution annealing
        
        #------- Valid representing space 
        assert isinstance(space, BatchedBlockSpace), f"{self.__class__.__name__} expects space of BatchedBlockSpace"
        self.space: BatchedBlockSpace = space
        self.dtype = dtype
        
        #------- Occupancy information getter
        self.occ_getter = OccGridGetter(**occ_kwargs, resolution=resolution, dtype=self.dtype, device=device)
        self.occ_grid_per_batch = None
        self.batch_size = None

    @property
    def device(self) -> torch.device:
        return self.space.device

    @property
    def resolution(self):
        return self.occ_getter.reslution

    """ Updating and using occ """
    def set_condition(
        self, batch_size: int, *, 
        ins_inds_per_batch: torch.LongTensor = None, 
        val_query_fn_normalized_x_bi = None):
        assert val_query_fn_normalized_x_bi is not None, \
            f"`val_query_fn_normalized_x_bi` is required for {type(self)}"
        self.batch_size = batch_size
        self.occ_grid_per_batch = self.occ_getter.occ_grid_from_net_batched_v2(
            batch_size, val_query_fn_normalized_x_bi)

    def clean_condition(self):
        self.batch_size = None
        self.occ_grid_per_batch = None

    @torch.no_grad()
    def init(self, val_query_fn_normalized_x_bi, logger: Logger = None):
        pass

    @torch.no_grad()
    def cur_batch__step(self, cur_it: int, val_query_fn_normalized_x_bi, logger: Logger = None):
        pass

    @torch.no_grad()
    def cur_batch__collect_samples(self, pts: torch.Tensor, bidx: torch.LongTensor, val: torch.Tensor, normalized=True):
        pass

    """ DEBUG Functionalities """
    @torch.no_grad()
    def debug_stats(self) -> Dict[str, Number]:
        return {}

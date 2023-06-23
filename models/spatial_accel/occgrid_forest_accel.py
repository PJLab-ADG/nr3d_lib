"""
@file   occgrid_forest_accel.py
@author Jianfei Guo, Shanghai AI Lab
@brief  For a forest of blocks; Acceleration structure based on occupancy grid.
"""

__all__ = [
    'OccupancyGridForestAS'
]

from numbers import Number
from typing import Dict, Literal

import torch
import torch.nn as nn

from nr3d_lib.fmt import log
from nr3d_lib.logger import Logger
from nr3d_lib.config import ConfigDict

from nr3d_lib.models.spatial import ForestBlockSpace
from nr3d_lib.models.spatial_accel.occgrid import OccupancyGridEMABatched
from nr3d_lib.render.raymarch.occgrid_raymarch import occgrid_raymarch, forest_occgrid_raymarch

try:
    import vedo
    from vedo import Box, Plotter, Volume, show
except ImportError:
    log.info("vedo not installed. Some of the visualizations are disabled.")

DEBUG_OCCGRID = False
DEBUG_OCCGRID_OFFSCREEN = False 
DEBUG_OCCGRID_INTERACTIVE = False # This will not allow the training to run.

class OccupancyGridForestAS(nn.Module):
    def __init__(
        self, space: ForestBlockSpace, occ_cfg: ConfigDict, 
        dtype=torch.float, device=torch.device('cuda') ) -> None:
        super().__init__()
        
        #------- Valid representing space 
        assert isinstance(space, ForestBlockSpace),  f"{self.__class__.__name__} expects space of ForestBlockSpace"
        self.space: ForestBlockSpace = space
        self.dtype = dtype
        
        #------- Occupancy information
        self.occ = None
        self.occ_cfg = occ_cfg

        self.training_granularity = 0.0 # No level/resolution annealing

    @property
    def device(self):
        return self.space.device

    def populate(self):
        # Populate occupancy grid AFTER forest space is populated
        self.occ = OccupancyGridEMABatched(num_batches=self.space.n_trees, **self.occ_cfg)
        
    @torch.no_grad()
    def gather_samples(self, pts: torch.Tensor, bidx: torch.Tensor = None, val: torch.Tensor = ..., normalized=True):
        if self.training:
            if not normalized:
                pts, bidx = self.space.normalize_coords(pts, bidx)
            # NOTE: Filter out invalid pts where bidx==-1
            valid = (bidx>=0).nonzero(as_tuple=True)
            self.occ.gather_samples(pts[valid], bidx[valid], val[valid])
    
    @torch.no_grad()
    def preprocess_per_train_step(self, cur_it: int, query_fn_block_x_with_bidx, logger: Logger = None):
        """
        `query_fn_block_x_with_bidx`:
            A callable function that expects two input tensor and returns one tensor:
            Input:
                block_x:    [-1,1] normalized coordinates in block
                bidx:       The corresponding block inds of every point
            Return:
                sdf:        The queried SDF value of current block.
        """
        updated = self.occ.preprocess_per_train_step(cur_it, query_fn_block_x_with_bidx, logger=logger)

        if DEBUG_OCCGRID:
            if cur_it == 0 or not hasattr(self, 'plt'):
                vedo.settings.allow_interaction = True
                self.plt = Plotter(axes=1, interactive=DEBUG_OCCGRID_INTERACTIVE, offscreen=DEBUG_OCCGRID_OFFSCREEN)
                self.forest_actors = self.debug_vis(draw=False)
                self.plt.show(*self.forest_actors, __doc__,  viewup="z")
                self.plt.camera.Elevation(15.0)
                if logger is not None:
                    img = self.plt.screenshot(asarray=True)
                    logger.add_imgs(img, "dbg_occgrid", cur_it)
            elif updated:
                if not self.plt.escaped:
                    self.debug_vis_tick()
                    if logger is not None:
                        img = self.plt.screenshot(asarray=True)
                        logger.add_imgs(img, "dbg_occgrid", cur_it)
                else:
                    self.plt.interactive().close()

    @torch.no_grad()
    def postprocess_per_train_step(self, cur_it: int, query_fn_block_x_with_bidx, logger: Logger = None):
        """
        `query_fn_block_x_with_bidx`:
            A callable function that expects two input tensor and returns one tensor:
            Input:
                block_x:    [-1,1] normalized coordinates in block
                bidx:       The corresponding block inds of every point
            Return:
                sdf:        The queried SDF value of current block.
        """
        pass
    
    def ray_march(
        self, 
        rays_o: torch.Tensor, rays_d: torch.Tensor, near=None, far=None, 
        seg_block_inds: torch.Tensor = ..., seg_entries: torch.Tensor = ...,  seg_exits: torch.Tensor = ..., seg_pack_infos: torch.Tensor = ...,  
        *, perturb=False, step_size: float = 1e-3, max_step_size: float = 1e10, dt_gamma: float = 0.0, max_steps: int = 512
        ):
        # NOTE: Expects rays_o, rays_d in world space.
        return forest_occgrid_raymarch(
            self.space.meta, self.occ.occ_grid, rays_o, rays_d, near, far, 
            seg_block_inds, seg_entries, seg_exits, seg_pack_infos, 
            perturb=perturb, step_size=step_size, max_step_size=max_step_size, dt_gamma=dt_gamma, max_steps=max_steps)
    
    def ray_march_simple_step_segment(
        self, 
        rays_o: torch.Tensor, rays_d: torch.Tensor, near=None, far=None, 
        seg_block_inds: torch.Tensor = ..., seg_entries: torch.Tensor = ...,  seg_exits: torch.Tensor = ..., seg_pack_infos: torch.Tensor = ...,  
        *, perturb=False, step_mode: Literal['linear', 'depth'], **step_kwargs
        ):
        # NOTE: Expects rays_o, rays_d in world space.
        return self.space.ray_step_coarse(
            rays_o, rays_d, near, far, seg_block_inds, seg_entries, seg_exits, seg_pack_infos, 
            step_mode=step_mode, perturb=perturb, **step_kwargs)
    
    def ray_march_simple_step_nearfar(
        self, 
        rays_o: torch.Tensor, rays_d: torch.Tensor, near=None, far=None, 
        *, perturb=False, step_size: float = 1e-3, max_step_size: float = 1e10, dt_gamma: float = 0.0, max_steps: int = 512
        ):
        # NOTE: Expects rays_o, rays_d in world space.
        aabb = self.space.get_aabb()
        occ_grid = self.space.to_occgrid()
        return occgrid_raymarch(
            occ_grid, rays_o, rays_d, near, far, 
            perturb=perturb, step_size=step_size, max_step_size=max_step_size, dt_gamma=dt_gamma, max_steps=max_steps, roi=aabb)

    def query_world(self, pts: torch.Tensor):
        pts, bidx = self.space.normalize_coords(pts)
        valid = (bidx >= 0).nonzero(as_tuple=True)
        occupied = torch.zeros(bidx.shape, dtype=torch.bool, device=pts.device)
        if valid[0].numel() > 0:
            occupied[valid] = self.occ.query(pts[valid], bidx[valid])
        return occupied

    #------------------------------------------------------
    #--------------- DEBUG Functionalities ----------------
    #------------------------------------------------------
    @torch.no_grad()
    def debug_stats(self) -> Dict[str, Number]:
        return {
            'num_occupied': self.occ.occ_grid.sum().item(), 
            'frac_occupied': self.occ.occ_grid.sum().item() / self.occ.occ_grid.numel()
        }
    
    @torch.no_grad()
    def debug_vis(self, draw=True, boundary=True, draw_occ_grid=True):
        # NOTE: Primary drawing function
        spacing = (self.space.world_block_size / self.occ.resolution).tolist()
        world_block_size = self.space.world_block_size.data.cpu().numpy()
        aabb = self.space.aabb
        world = Box(((aabb[0] + aabb[1])/2.).data.cpu().numpy(), *(aabb[1]-aabb[0]).tolist()).wireframe()
        forest_actors = [world]
        for bidx, block_idx in enumerate(self.space.block_ks):
            origin = (self.space.world_origin + block_idx * self.space.world_block_size).data.cpu().numpy()
            box = Box(origin + world_block_size/2., *world_block_size.tolist()).wireframe()
            forest_actors.append(box)
            occ_val_grid = self.occ.occ_val_grid[bidx].data.cpu().numpy()
            if not draw_occ_grid or not (occ_val_grid > self.occ.occ_thre).any():
                # NOTE: Skip empty occ_grid
                continue
            vol = Volume(occ_val_grid, c=['white','b','g','r'], mapper='gpu', origin=origin, spacing=spacing)
            vox = vol.legosurface(vmin=self.occ.occ_thre, vmax=1., boundary=boundary)
            vox.cmap('GnBu', on='cells', vmin=self.occ.occ_thre, vmax=1.)
            forest_actors.append(vox)
        if draw:
            show(*forest_actors, __doc__, axes=1, viewup='z').close()
        else:
            return forest_actors

    @torch.no_grad()
    def debug_vis_tick(self):
        self.plt.clear(self.forest_actors)
        self.forest_actors = self.debug_vis(draw=False)
        self.plt.add(*self.forest_actors, resetcam=False)

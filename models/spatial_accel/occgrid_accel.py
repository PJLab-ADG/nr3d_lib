"""
@file   occgrid_accel.py
@author Jianfei Guo, Shanghai AI Lab
@brief  For a single-block space; Acceleration struct based on occupancy grid.
"""

__all__ = [
    'OccupancyGridAS'
]

from numbers import Number
from typing import Dict, List, Union

import torch
import torch.nn as nn

from nr3d_lib.fmt import log
from nr3d_lib.logger import Logger
from nr3d_lib.config import ConfigDict
from nr3d_lib.plot import create_occ_grid_mesh_o3d

from nr3d_lib.models.spatial import AABBSpace
from nr3d_lib.models.spatial_accel.occgrid import OccupancyGridEMA
from nr3d_lib.render.raymarch.occgrid_raymarch import occgrid_raymarch

try:
    import vedo
    from vedo import Box, Plotter, Volume, show
except ImportError:
    log.info("vedo not installed. Some of the visualizations are disabled.")

DEBUG_OCCGRID = False # [True] = Visualizing occgrid using vedo
DEBUG_OCCGRID_OFFSCREEN = False # [True] = No prompt vedo window
DEBUG_OCCGRID_INTERACTIVE = False # [True] = Allow mouse dragging on the vedo window, but will block training.

class OccupancyGridAS(nn.Module):
    def __init__(
        self, 
        space: AABBSpace, 
        dtype=torch.float, 
        device=torch.device('cuda'), 
        resolution: Union[int, List[int], torch.Tensor] = None, 
        vox_size: float = None, 
        **occ_kwargs) -> None:
        super().__init__()
        
        #------- Valid representing space 
        assert isinstance(space, AABBSpace), f"{self.__class__.__name__} expects space of AABBSpace"
        self.space: AABBSpace = space
        self.dtype = dtype
        # self.device = device # NOTE: Directly use the device of self.space
        
        #------- Occupancy information
        assert bool(resolution is not None) != bool(vox_size is not None), \
            "Please specify `vox_size` or `resolution` for OccupancyGridAS."
        
        if resolution is None:
            resolution = (self.space.stretch / vox_size).long()
        self.occ = OccupancyGridEMA(**occ_kwargs, resolution=resolution, dtype=self.dtype, device=self.device)
        
        self.training_granularity = 0.0 # No level/resolution annealing

    @property
    def device(self):
        return self.space.device

    @torch.no_grad()
    def gather_samples(self, pts: torch.Tensor, val: torch.Tensor, normalized=True):
        """
        NOTE: `gather_samples` should be invoked like a forward-hook function.
        """
        if self.training:
            if not normalized:
                pts = self.space.normalize_coords(pts)
            self.occ.gather_samples(pts, val)
    
    @torch.no_grad()
    def preprocess_per_train_step(self, cur_it: int, query_fn, logger: Logger = None):
        updated = self.occ.preprocess_per_train_step(cur_it, query_fn, logger=logger)
        
        if DEBUG_OCCGRID:
            if cur_it == 0 or not hasattr(self, 'plt'):
                vedo.settings.allow_interaction = True
                self.world = Box(self.space.origin.data.cpu().numpy(), *(self.space.scale*2).tolist()).wireframe()
                self.plt = Plotter(axes=1, interactive=DEBUG_OCCGRID_INTERACTIVE, offscreen=DEBUG_OCCGRID_OFFSCREEN)
                self.vox = self.debug_vis(draw=False)
                self.plt.show(self.world, self.vox, __doc__,  viewup="z")
                self.plt.camera.Elevation(15.0)
                # self.plt.screenshot(os.path.join(self.dbg_dir, f"{cur_it:08d}.png"))
                if logger is not None:
                    img = self.plt.screenshot(asarray=True)
                    logger.add_imgs(img, "dbg_occgrid", cur_it)
            elif updated:
                self.debug_vis_tick()
                # self.plt.screenshot(os.path.join(self.dbg_dir, f"{cur_it:08d}.png"))
                if logger is not None:
                    img = self.plt.screenshot(asarray=True)
                    logger.add_imgs(img, "dbg_occgrid", cur_it)
                # NOTE: if self.plt.escaped (this is for older versions)
                # self.plt.interactive().close()
    
    @torch.no_grad()
    def postprocess_per_train_step(self, cur_it: int, query_fn, logger: Logger = None):
        pass

    @torch.no_grad()
    def rescale_volume(self, new_aabb: torch.Tensor):
        old_aabb = self.space.aabb.clone()
        self.occ.rescale_volume(old_aabb, new_aabb)

    @torch.no_grad()
    def try_shrink(self) -> torch.Tensor:
        old_aabb = self.space.aabb.clone()
        new_aabb = self.occ.try_shrink(old_aabb)
        return new_aabb

    def ray_march(
        self, 
        rays_o: torch.Tensor, rays_d: torch.Tensor, near=None, far=None, *, perturb=False, normalized=True, 
        step_size: float = 1e-3, max_step_size: float = 1e10, dt_gamma: float = 0.0, max_steps: int = 512
        ):
        if not normalized: 
            rays_o, rays_d = self.space.normalize_rays(rays_o, rays_d)
        return occgrid_raymarch(
            self.occ.occ_grid, rays_o, rays_d, near, far, 
            perturb=perturb, step_size=step_size, max_step_size=max_step_size, dt_gamma=dt_gamma, max_steps=max_steps)

    #------------------------------------------------------
    #--------------- DEBUG Functionalities ----------------
    #------------------------------------------------------
    @torch.no_grad()
    def num_occupied(self) -> int:
        return self.occ.occ_grid.sum().item()
    
    @torch.no_grad()
    def frac_occupied(self) -> float:
        return self.occ.occ_grid.sum().item() / self.occ.occ_grid.numel()
    
    @torch.no_grad()
    def debug_stats(self) -> Dict[str, Number]:
        return {
            'num_occupied': self.num_occupied(), 
            'frac_occupied': self.frac_occupied()
        }
    
    @torch.no_grad()
    def debug_get_mesh(self):
        return create_occ_grid_mesh_o3d(self.occ.occ_grid)

    @torch.no_grad()
    def debug_vis_legacy(self, show=True, dbg_save=False, **kwargs):
        from nr3d_lib.plot import vis_occgrid_voxels_o3d
        return vis_occgrid_voxels_o3d(self.occ.occ_grid, show=show, **kwargs)
    
    @torch.no_grad()
    def debug_vis(self, draw=True):
        # NOTE: Primary drawing function
        # from vedo.applications import IsosurfaceBrowser
        occ_val_grid = self.occ.occ_val_grid.data.cpu().numpy()
        spacing = ((self.space.aabb[1]-self.space.aabb[0]) / self.occ.resolution).tolist()
        origin = self.space.aabb[0].tolist()
        vol = Volume(occ_val_grid, c=['white','b','g','r'], mapper='gpu', origin=origin, spacing=spacing)
        vox = vol.legosurface(vmin=self.occ.occ_thre, vmax=1., boundary=True)
        vox.cmap('GnBu', on='cells', vmin=self.occ.occ_thre, vmax=1.).add_scalarbar()
        if draw:
            show(vox, __doc__, axes=1, viewup='z').close()
        else:
            return vox
        
        # plt = IsosurfaceBrowser(vol, lego=True, cmap='seismic') # Plotter instance
        # plt.show(axes=7, bg2='lb').close()
    
    @torch.no_grad()
    def debug_vis_tick(self):
        self.plt.remove(self.world, self.vox)
        self.vox = self.debug_vis(draw=False)
        self.world = Box(self.space.origin.data.cpu().numpy(), *(self.space.scale*2).tolist()).wireframe()
        self.plt.show(self.vox, self.world, resetcam=False)
"""
@file   occgrid_accel.py
@author Jianfei Guo, Shanghai AI Lab
@brief  For a single-block space; Acceleration struct based on occupancy grid.
"""

__all__ = [
    'OccGridAccel'
]

from numbers import Number
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn

from nr3d_lib.fmt import log
from nr3d_lib.logger import Logger
from nr3d_lib.config import ConfigDict
from nr3d_lib.plot import create_occ_grid_mesh_o3d

from nr3d_lib.models.spatial import AABBSpace
from nr3d_lib.models.accelerations.occgrid import OccGridEma
from nr3d_lib.graphics.raymarch.occgrid_raymarch import occgrid_raymarch

try:
    import vedo
    from vedo import Box, Plotter, Volume, show
except ImportError:
    log.info("vedo not installed. Some of the visualizations are disabled.")

DEBUG_OCCGRID = False # [True] = Visualizing occgrid using vedo
DEBUG_OCCGRID_OFFSCREEN = False # [True] = No prompt vedo window
DEBUG_OCCGRID_INTERACTIVE = False # [True] = Allow mouse dragging on the vedo window, but will block training.

class OccGridAccel(nn.Module):
    def __init__(
        self, 
        space: AABBSpace, 
        resolution: Union[int, List[int], torch.Tensor] = None, vox_size: float = None, 
        dtype=torch.float, device=None, 
        **occ_kwargs) -> None:
        super().__init__()
        
        #------- Valid representing space 
        assert isinstance(space, AABBSpace), f"{self.__class__.__name__} expects space of AABBSpace"
        self.space: AABBSpace = space
        self.dtype = dtype
        
        #------- Occupancy information
        assert bool(resolution is not None) != bool(vox_size is not None), \
            "Please specify `vox_size` or `resolution` for OccGridAccel."
        
        if resolution is None:
            resolution = (self.space.radius3d * 2 / vox_size).long()
        self.occ = OccGridEma(
            resolution=resolution, **occ_kwargs, 
            dtype=self.dtype, device=self.device
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
    def init(self, val_query_fn, logger: Logger = None):
        updated = self.occ.init(val_query_fn, logger=logger)

    @torch.no_grad()
    def step(self, cur_it: int, val_query_fn, logger: Logger = None):
        updated = self.occ.step(cur_it, val_query_fn, logger=logger)
        
        if DEBUG_OCCGRID:
            if not hasattr(self, 'plt'):
                vedo.settings.allow_interaction = True
                self.world = Box(self.space.center.data.cpu().numpy(), *(self.space.radius3d*2).tolist()).wireframe()
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
    def collect_samples(self, pts: torch.Tensor, val: torch.Tensor, normalized=True):
        """
        NOTE: `collect_samples` should be invoked like a forward-hook function.
        """
        if self.training:
            if not normalized:
                pts = self.space.normalize_coords(pts)
            self.occ.collect_samples(pts, val)

    @torch.no_grad()
    def sample_pts_in_occupied(self, num_pts: int) -> torch.Tensor:
        pts = self.occ.sample_pts_in_occupied(num_pts)
        return pts
    
    @torch.no_grad()
    def query_occupancy(self, pts: torch.Tensor) -> torch.BoolTensor:
        # Expect pts in range [-1,1]
        return self.occ.query(pts)

    """ Ray marching """
    def ray_march(
        self, 
        rays_o: torch.Tensor, rays_d: torch.Tensor, near=None, far=None, *, perturb=False, normalized=True, 
        step_size: float = 1e-3, max_step_size: float = 1e10, dt_gamma: float = 0.0, max_steps: int = 512
        ):
        if not normalized: 
            rays_o, rays_d = self.space.normalize_rays(rays_o, rays_d)
        ret = occgrid_raymarch(
            self.get_occ_grid(), rays_o, rays_d, near, far, 
            perturb=perturb, step_size=step_size, max_step_size=max_step_size, dt_gamma=dt_gamma, max_steps=max_steps)
        return ret

    """ Shrink volume """
    @torch.no_grad()
    def rescale_volume(self, new_aabb: torch.Tensor):
        old_aabb = self.space.aabb.clone()
        self.occ.rescale_volume(old_aabb, new_aabb)

    @torch.no_grad()
    def try_shrink(self) -> torch.Tensor:
        old_aabb = self.space.aabb.clone()
        new_aabb = self.occ.try_shrink(old_aabb)
        return new_aabb

    """ DEBUG Functionalities """
    @torch.no_grad()
    def num_occupied(self) -> int:
        occ_grid = self.get_occ_grid()
        return occ_grid.sum().item()
    
    @torch.no_grad()
    def frac_occupied(self) -> float:
        occ_grid = self.get_occ_grid()
        return occ_grid.sum().item() / occ_grid.numel()
    
    @torch.no_grad()
    def debug_stats(self) -> Dict[str, Number]:
        return {
            'num_occupied': self.num_occupied(), 
            'frac_occupied': self.frac_occupied()
        }
    
    @torch.no_grad()
    def debug_get_mesh(self):
        return create_occ_grid_mesh_o3d(self.get_occ_grid())

    @torch.no_grad()
    def debug_vis_legacy(self, show=True, dbg_save=False, **kwargs):
        from nr3d_lib.plot import vis_occgrid_voxels_o3d
        return vis_occgrid_voxels_o3d(self.get_occ_grid(), show=show, **kwargs)
    
    @torch.no_grad()
    def debug_vis(self, draw=True):
        # NOTE: Primary drawing function
        # from vedo.applications import IsosurfaceBrowser
        occ_val_grid = self.occ.occ_val_grid.data.cpu().numpy()
        if (occ_val_grid > self.occ.occ_thre).any():
            spacing = ((self.space.aabb[1]-self.space.aabb[0]) / self.resolution).tolist()
            origin = self.space.aabb[0].tolist()
            vol = Volume(occ_val_grid, c=['white','b','g','r'], mapper='gpu', origin=origin, spacing=spacing)
            vox = vol.legosurface(vmin=self.occ.occ_thre, vmax=1., boundary=True)
            vox.cmap('GnBu', on='cells', vmin=self.occ.occ_thre, vmax=1.).add_scalarbar()
        else:
            vox = None
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
        self.world = Box(self.space.center.data.cpu().numpy(), *(self.space.radius3d*2).tolist()).wireframe()
        self.plt.show(self.vox, self.world, resetcam=False)

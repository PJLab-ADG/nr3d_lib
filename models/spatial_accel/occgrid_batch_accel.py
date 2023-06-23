"""
@file   occgrid_batch_accel.py
@author Jianfei Guo, Shanghai AI Lab
@brief  For a batched-blocks space; Acceleration struct based on occupancy grid.
"""

__all__ = [
    'OccupancyGridBatchAS'
]

import torch
import torch.nn as nn
from nr3d_lib.config import ConfigDict

from nr3d_lib.models.spatial import BatchedBlockSpace
from nr3d_lib.models.spatial_accel.occgrid import OccupancyGridGetter
from nr3d_lib.render.raymarch.occgrid_raymarch import batched_occgrid_raymarch

class OccupancyGridBatchAS(nn.Module):
    def __init__(
        self, 
        space: BatchedBlockSpace, occ_cfg: ConfigDict, 
        dtype=torch.float, device=torch.device('cuda')
        ) -> None:
        super().__init__()
        
        #------- Valid representing space 
        assert isinstance(space, BatchedBlockSpace), f"{self.__class__.__name__} expects space of BatchedBlockSpace"
        self.space: BatchedBlockSpace = space
        self.dtype = dtype
        
        #------- Occupancy information getter
        self.occ_getter = OccupancyGridGetter(**occ_cfg, dtype=self.dtype, device=self.device)
        
        self.training_granularity = 0.0 # No level/resolution annealing

    @property
    def device(self):
        return self.space.device

    def ray_march(
        self, occ_grid: torch.Tensor, 
        rays_o: torch.Tensor, rays_d: torch.Tensor, near=None, far=None, batch_inds: torch.Tensor = None,  *, perturb=False, 
        step_size: float = 1e-3, max_step_size: float = 1e10, dt_gamma: float = 0.0, max_steps: int = 512
        ):
        return batched_occgrid_raymarch(
            occ_grid, rays_o, rays_d, near, far, batch_inds, perturb=perturb, 
            step_size=step_size, max_step_size=max_step_size, dt_gamma=dt_gamma, max_steps=max_steps
        )
__all__ = [
    'OccGridGetter'
]

import numpy as np
from math import prod
from tqdm import trange
from icecream import ic
from typing import List, Literal, Union

import torch
import torch.nn as nn
from torch_scatter import scatter_max

from nr3d_lib.fmt import log
# from nr3d_lib.profile import profile
from .utils import *

class OccGridGetter(nn.Module):
    NUM_DIM: int = 3
    def __init__(
        self, 
        resolution: Union[int, List[int], torch.Tensor] = 128,
        occ_val_fn_cfg=dict(type='density'), occ_val_fn = None, occ_thre: float = 0.01, 
        occ_thre_consider_mean=False, # Whether consider average value as threshold when binarizing occ_val
        num_steps=4, num_pts_per_batch: int=2**18, 
        num_pts: int = None,
        dtype=torch.float, device=None
        ) -> None:
        super().__init__()
        
        self.dtype = dtype
        
        # if num_pts is not None:
        #     num_pts_per_batch = num_pts
        #     log.warn(f"Warning: `num_pts` in batched occ grid getter is deprecated. Use `num_pts_per_batch` instead.")
        
        if isinstance(resolution, int):
            resolution = [resolution] * self.NUM_DIM
        if isinstance(resolution, (list, tuple, np.ndarray)):
            resolution = torch.tensor(resolution, dtype=torch.int32, device=device)
        elif isinstance(resolution, torch.Tensor):
            resolution = resolution.to(dtype=torch.int32, device=device)
        else:
            raise RuntimeError(f"Invalid type of resolution={type(resolution)}")

        self.register_buffer("resolution", resolution, persistent=False)
        gidx_full = torch.stack(
            torch.meshgrid(
                [torch.arange(res, device=device) for res in resolution.tolist()], indexing='ij'
            ), dim=-1
        ).view(-1,self.NUM_DIM)
        self.register_buffer("gidx_full", gidx_full, persistent=False)

        self.occ_thre = occ_thre
        self.occ_val_fn = get_occ_val_fn(**occ_val_fn_cfg) if occ_val_fn is None else occ_val_fn
        self.occ_thre_consider_mean = occ_thre_consider_mean

        self.num_steps = num_steps
        self.num_pts = num_pts
        self.num_pts_per_batch = num_pts_per_batch
    
    @property
    def device(self) -> torch.device:
        return self.resolution.device
    
    @torch.no_grad()
    def occ_grid_from_net(self, val_query_fn, progress=False) -> torch.Tensor:
        """
        Always uniformly sample in all voxels
        """
        num_steps, num_pts, gidx_full = self.num_steps, self.num_pts, self.gidx_full
        occ_grid = torch.zeros(self.resolution.tolist(), dtype=torch.bool, device=self.device)
        for _ in trange(num_steps, disable=not progress):
            n_per_vox = int(num_pts // gidx_full.shape[0]) + 1
            offsets = torch.rand([gidx_full.shape[0], n_per_vox, self.NUM_DIM], device=self.device, dtype=self.dtype)
            pts = ((gidx_full[:,None,:] + offsets) / self.resolution) * 2 -1
            val = val_query_fn(pts)
            occ_val = self.occ_val_fn(val.flatten()).view(gidx_full.shape[0], n_per_vox)
            occ_grid_iter = binarize(occ_val, self.occ_thre, self.occ_thre_consider_mean).any(dim=-1).view(self.resolution.tolist())
            occ_grid |= occ_grid_iter
        return occ_grid

    @torch.no_grad()
    def occ_grid_from_net_v2(self, val_query_fn, progress=False, verbose=False) -> torch.Tensor:
        """
        Each step, only uniformly sample in voxels that remains empty at the current step
        """
        num_steps, num_pts, gidx_full = self.num_steps, self.num_pts, self.gidx_full
        resolution = self.resolution.tolist()
        occ_grid = torch.zeros(resolution, dtype=torch.bool, device=self.device)
        for _ in trange(num_steps, disable=not progress):
            # Only sample in currently empty voxels
            idx = (~occ_grid).nonzero().long()
            num_voxels = idx.shape[0]
            n_per_vox = int(num_pts // num_voxels) + 1
            if verbose:
                ic(num_voxels, n_per_vox)
            
            # [num_voxels, n_per_vox, ...]
            offsets = torch.rand([num_voxels, n_per_vox, self.NUM_DIM], device=self.device, dtype=self.dtype)
            pts = ((gidx_full.view(*resolution, self.NUM_DIM)[tuple(idx.t())].unsqueeze(-2) + offsets) / self.resolution)*2-1
            val = val_query_fn(pts)
            occ_val = self.occ_val_fn(val.flatten()).view(num_voxels, n_per_vox).max(-1).values
            
            # Scatter max in currently empty voxels
            # [(1)*(2)*1, (2)*1, 1]
            idx_ravel = (idx * idx.new_tensor([resolution[1] * resolution[2], resolution[2], 1])).sum(-1)
            occ_val_grid_iter, _ = scatter_max(occ_val, idx_ravel, dim=0, dim_size=prod(occ_grid.shape))
            occ_grid |= binarize(occ_val_grid_iter, self.occ_thre, self.occ_thre_consider_mean).view(occ_grid.shape)
        return occ_grid

    @torch.no_grad()
    def occ_grid_from_net_batched_v1(self, B: int, val_query_fn_batched, progress=False) -> torch.Tensor:
        """
        Batched net that does NOT support bidx input
        """
        num_steps, num_pts_per_batch, gidx_full = self.num_steps, self.num_pts_per_batch, self.gidx_full
        occ_grid = torch.zeros([B, *self.resolution.tolist()], dtype=torch.bool, device=self.device)
        for _ in trange(num_steps, disable=not progress):
            n_per_vox = int(num_pts_per_batch // gidx_full.shape[0]) + 1
            offsets = torch.rand([B, gidx_full.shape[0], n_per_vox, self.NUM_DIM], device=self.device, dtype=self.dtype)
            pts = ((gidx_full[None,:,None,:] + offsets) / self.resolution) * 2 -1
            val = val_query_fn_batched(pts)
            occ_val = self.occ_val_fn(val.flatten()).view(gidx_full.shape[0], n_per_vox)
            occ_grid_iter = binarize(occ_val, self.occ_thre, self.occ_thre_consider_mean).any(dim=-1).view(occ_grid.shape)
            occ_grid |= occ_grid_iter
        return occ_grid

    @torch.no_grad()
    def occ_grid_from_net_batched_v2(self, B: int, val_query_fn_normalized_x_bi, progress=False) -> torch.Tensor:
        """
        Each step, only uniformly sample in voxels that remains empty at the current step
        """
        num_steps, num_pts_per_batch, gidx_full = self.num_steps, self.num_pts_per_batch, self.gidx_full
        resolution = self.resolution.tolist()
        occ_grid = torch.zeros([B, *resolution], dtype=torch.bool, device=self.device)
        for _ in trange(num_steps, disable=not progress):
            # Only sample in currently empty voxels
            idx = (~occ_grid).nonzero().long()
            bidx, gidx = idx[..., 0], idx[..., 1:]
            num_voxels = idx.shape[0]
            n_per_vox = int(num_pts_per_batch // num_voxels) + 1
            
            # [num_voxels, n_per_vox, ...]
            offsets = torch.rand([num_voxels, n_per_vox, self.NUM_DIM], device=self.device, dtype=self.dtype)
            pts = ((gidx_full.view(*resolution, self.NUM_DIM)[tuple(gidx.t())].unsqueeze(-2) + offsets) / self.resolution)*2-1
            val = val_query_fn_normalized_x_bi(pts, bidx=bidx.view(num_voxels, 1).expand(num_voxels, n_per_vox))
            occ_val = self.occ_val_fn(val.flatten()).view(num_voxels, n_per_vox).max(-1).values
            
            # Scatter max in currently empty voxels
            # [(0)*(1)*(2)*1, (1)*(2)*1, (2)*1, 1]
            idx_ravel = (idx * idx.new_tensor([prod(resolution), resolution[1] * resolution[2], resolution[2], 1])).sum(-1)
            occ_val_grid_iter, _ = scatter_max(occ_val, idx_ravel, dim=0, dim_size=prod(occ_grid.shape))
            occ_grid |= binarize(occ_val_grid_iter, self.occ_thre, self.occ_thre_consider_mean).view(occ_grid.shape)
        return occ_grid

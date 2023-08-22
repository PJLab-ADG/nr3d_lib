"""
@file   occgrid.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Occupancy memory and updater.
"""

__all__ = [
    'OccupancyGridEMA', 
    'OccupancyGridEMABatched', 
    'OccupancyGridGetter'
]

import functools
import numpy as np
from math import prod
from tqdm import trange
from icecream import ic
from typing import List, Literal, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max

from nr3d_lib.fmt import log
from nr3d_lib.logger import Logger
from nr3d_lib.config import ConfigDict
from nr3d_lib.models.annealers import get_anneal_val
from nr3d_lib.models.utils import normalized_logistic_density

"""
The key is how we define `occupancy`; 
ideally, everything should be converted to a unified concept of "occupancy" and `threshold`
"""
def sdf_to_occ(sdf: torch.Tensor, *, inv_s: float = None, inv_s_anneal_cfg: ConfigDict = None):
    if inv_s_anneal_cfg is not None:
        inv_s = get_anneal_val(**inv_s_anneal_cfg)
    else:
        assert inv_s is not None, "Need config `inv_s`"
    return normalized_logistic_density(sdf, inv_s)

def get_occ_val_fn(type: Literal['sdf', 'density', 'occ'] = 'sdf', **kwargs):
    if type == 'sdf':
        return functools.partial(sdf_to_occ, **kwargs)
    elif type == "raw_sdf":
        return lambda sdf: 1.0 - torch.abs(sdf)
    elif type == 'occ':
        return nn.Identity()
    elif type == 'density':
        return nn.Identity()
    else:
        raise RuntimeError(f"Invalid type={type}")

def update_occ_val_grid_idx_(occ_val_grid: torch.Tensor, gidx: torch.LongTensor, occ_val: torch.Tensor, ema_decay: float = 1.0):
    resolution_l = occ_val_grid.shape
    occ_val = occ_val.flatten().to(occ_val_grid)
    gidx_ravel = (gidx * gidx.new_tensor([resolution_l[1] * resolution_l[2], resolution_l[2], 1])).sum(-1)
    # 267 us @ 2M pts
    # NOTE: `out` argument also participates in `maximum` reduce.
    #       This is similar to `scatter_reduce_`'s behavior in pytorch 1.12 with `include_self=True`
    # NOTE: EMA update of val grid; only on selected indices.
    occ_val_new, _ = scatter_max(occ_val, gidx_ravel, out=ema_decay * occ_val_grid.flatten())
    occ_val_grid.index_put_(tuple(gidx.t()), occ_val_new[gidx_ravel])

def update_occ_val_grid_(occ_val_grid: torch.Tensor, pts: torch.FloatTensor, occ_val: torch.Tensor, ema_decay: float = 1.0):
    device, dtype, resolution_l = occ_val_grid.device, occ_val_grid.dtype, occ_val_grid.shape
    resolution = torch.tensor([resolution_l], device=device)
    pts, occ_val = pts.flatten(0, -2), occ_val.flatten().to(occ_val_grid)
    gidx = ((pts/2. + 0.5) * resolution).long().clamp(resolution.new_tensor([0]), resolution-1)
    update_occ_val_grid_idx_(occ_val_grid, gidx, occ_val, ema_decay=ema_decay)

def update_batched_occ_val_grid_idx_(
    occ_val_grid: torch.Tensor, bidx: torch.Tensor = None, gidx: torch.LongTensor = ..., occ_val: torch.Tensor = ..., ema_decay: float = 1.0):
    num_batches, resolution_l = occ_val_grid.shape[0], occ_val_grid.shape[1:]
    if bidx is not None:
        # Not batched
        bidx, occ_val = bidx.flatten(), occ_val.flatten().to(occ_val_grid)
        idx_ravel = bidx * prod(resolution_l) + (gidx * gidx.new_tensor([resolution_l[1] * resolution_l[2], resolution_l[2], 1])).sum(-1)
        occ_val_new, _ = scatter_max(occ_val, idx_ravel, dim=0, out=ema_decay * occ_val_grid.flatten())
        occ_val_grid.index_put_((bidx,) + tuple(gidx.t()), occ_val_new[idx_ravel])
    else:
        # Batched
        occ_val = occ_val.flatten(1, -1).to(occ_val_grid)
        gidx_ravel = (gidx * gidx.new_tensor([resolution_l[1] * resolution_l[2], resolution_l[2], 1])).sum(-1) # [B, num_pts]
        occ_val_new, _ = scatter_max(occ_val, gidx_ravel, dim=1, out=ema_decay * occ_val_grid.flatten(1,-1)) # [B, num_grid_voxels]
        occ_val_grid.view(num_batches, -1).scatter_(1, gidx_ravel, occ_val_new.gather(1, gidx_ravel))

def update_batched_occ_val_grid_(
    occ_val_grid: torch.Tensor, pts: torch.FloatTensor, bidx: torch.Tensor = None, occ_val: torch.Tensor = ..., ema_decay: float = 1.0):
    resolution_l, device = occ_val_grid.shape[1:], occ_val_grid.device
    resolution = torch.tensor(resolution_l, device=device)
    gidx = ((pts/2. + 0.5) * resolution).long().clamp(resolution.new_tensor([0]), resolution-1)
    update_batched_occ_val_grid_idx_(occ_val_grid, bidx, gidx, occ_val, ema_decay=ema_decay)

def binarize(occ_val: torch.FloatTensor, occ_threshold: float, consider_mean=False, eps=1e-5) -> torch.BoolTensor:
    # NOTE: (-eps) is to due with all same values.
    threshold = occ_threshold if not consider_mean else (occ_val.mean() - eps).clamp_max_(occ_threshold)
    return occ_val > threshold

class OccupancyGridEMA(nn.Module):
    """
    Coordinates here are assumed to be normalized; in [-1,1] range.
    """
    NUM_DIM: int = 3
    def __init__(
        self, 
        resolution: Union[int, List[int], torch.Tensor] = 128,
        occ_val_fn_cfg=ConfigDict(type='density'), occ_val_fn = None, occ_thre: float = 0.01, 
        occ_thre_consider_mean=False, # Whether consider average value as threshold when binarizing occ_val
        ema_decay: float = 0.95, n_steps_between_update: int = 16, n_steps_warmup: int = 256,
        init_cfg=ConfigDict(), acquire_from_net_cfg = ConfigDict(), acquire_from_samples_cfg = ConfigDict(),
        dtype=torch.float, device=torch.device('cuda')) -> None:
        super().__init__()
    
        self.dtype = dtype
        self.device = device
        self.ema_decay = ema_decay
        
        if isinstance(resolution, int):
            resolution = [resolution] * self.NUM_DIM
        if isinstance(resolution, (list, tuple, np.ndarray)):
            resolution = torch.tensor(resolution, dtype=torch.int32, device=device)
        elif isinstance(resolution, torch.Tensor):
            resolution = resolution.to(dtype=torch.int32, device=device)
        else:
            raise RuntimeError(f"Invalid type of resolution={type(resolution)}")
        
        self.register_buffer("resolution", resolution, persistent=False)
        self.register_buffer("occ_grid", torch.zeros(resolution.tolist(), dtype=torch.bool, device=self.device), persistent=True)
        self.register_buffer("occ_val_grid", torch.zeros(resolution.tolist(), dtype=self.dtype, device=self.device), persistent=True)
        grid_coords = torch.stack(torch.meshgrid([torch.arange(res, device=self.device) for res in resolution.tolist()], indexing='ij'), dim=-1).view(-1,3)
        self.register_buffer("grid_coords", grid_coords, persistent=False)
        
        self._register_load_state_dict_pre_hook(self.load_state_dict_hook)
        
        self.init_cfg = init_cfg
        self.acquire_from_net_cfg = acquire_from_net_cfg
        self.acquire_from_samples_cfg = acquire_from_samples_cfg
        self.should_gather_samples: bool = acquire_from_samples_cfg is not None
        
        self.occ_thre = occ_thre
        self.occ_val_fn = get_occ_val_fn(**occ_val_fn_cfg) if occ_val_fn is None else occ_val_fn
        self.occ_thre_consider_mean = occ_thre_consider_mean
        
        self.n_steps_between_update = n_steps_between_update
        self.n_steps_warmup = n_steps_warmup
        
        if self.should_gather_samples:
            # To gather samples collected during forward & uniform sampling; and use them to update when it's time to update.
            self.register_buffer('_occ_val_grid_pcl', torch.zeros(resolution.tolist(), dtype=self.dtype, device=self.device), persistent=False)
    
    def load_state_dict_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # NOTE: Legacy support; To be removed soon.
        if prefix + 'val_grid' in state_dict:
            state_dict[prefix+'occ_val_grid'] = state_dict.pop(prefix+'val_grid')
        
        val_grid = state_dict[prefix + 'occ_val_grid']
        self.occ_val_grid.resize_(val_grid.shape)
        occ_grid = state_dict[prefix + 'occ_grid']
        self.occ_grid.resize_(occ_grid.shape)
        
        self.resolution[:] = self.resolution.new_tensor(occ_grid.shape)
        grid_coords = torch.stack(torch.meshgrid([torch.arange(res, device=self.device) for res in self.resolution.tolist()], indexing='ij'), dim=-1).view(-1,3)
        self.grid_coords = grid_coords

    @torch.no_grad()
    def initialize(self, *, mode: Literal['constant', 'from_net'], val_query_fn=None, **kwargs):
        if mode == 'constant':
            self._init_constant(**kwargs)
        elif mode == 'from_net':
            self._init_from_net(val_query_fn, **kwargs)
        else:
            raise RuntimeError(f"Invalid init_mode={mode}")

    @torch.no_grad()
    def preprocess_per_train_step(self, cur_it: int, val_query_fn, logger: Logger=None):
        updated = False
        if cur_it == 0:
            #------- Initialize
            self.initialize(val_query_fn=val_query_fn, **self.init_cfg)
            updated = True
        # TODO: When training on multiple GPU, repsect the relationship between the `it` passed in here and the control steps here; 
        #       also respect the warmup steps.
        elif cur_it % self.n_steps_between_update == 0:
            self._update_from_net_per_iter(cur_it, val_query_fn, **self.acquire_from_net_cfg)
            updated = True
        return updated
    
    @torch.no_grad()
    def postprocess_per_train_step(self, cur_it: int, val_query_fn):
        pass
    
    @torch.no_grad()
    def gather_samples(self, pts: torch.Tensor, val: torch.Tensor=None):
        """
        NOTE: `gather_samples` should be invoked like a forward-hook function.
        """
        if self.training and self.should_gather_samples:
            self._gather_samples(pts, val, **self.acquire_from_samples_cfg)

    @torch.no_grad()
    def _gather_samples(self, pts: torch.Tensor, val: torch.Tensor):
        update_occ_val_grid_(self._occ_val_grid_pcl, pts, self.occ_val_fn(val), ema_decay=1.0)

    @torch.no_grad()
    def _sample_pts_uniform(self, num_pts: int):
        grid_coords = self.grid_coords
        num_voxels = grid_coords.shape[0]
        if num_pts / num_voxels < 2.0:
            pts = torch.empty([num_pts, self.NUM_DIM], device=self.device, dtype=self.dtype).uniform_(-1, 1)
        else:
            n_per_vox = int(num_pts // num_voxels) + 1
            offsets = torch.rand([num_voxels, n_per_vox, self.NUM_DIM], device=self.device, dtype=self.dtype)
            pts = ((grid_coords[:,None,:] + offsets) / self.resolution).view(-1,3) * 2 -1
        return pts

    @torch.no_grad()
    def _sample_pts_selected(self, gidx: torch.Tensor, num_pts: int):
        num_voxels = gidx.shape[0]
        if num_pts / num_voxels < 2.0:
            inds = torch.randint(num_voxels, [num_pts, ], device=self.device)
            offsets = torch.rand([num_pts, self.NUM_DIM], device=self.device, dtype=self.dtype)
            pts = ((gidx[inds] + offsets) / self.resolution) * 2 - 1
        else:
            n_per_vox = int(num_pts // num_voxels) + 1
            offsets = torch.rand([num_voxels, n_per_vox, self.NUM_DIM], device=self.device, dtype=self.dtype)
            pts = ((gidx[:,None,:] + offsets) / self.resolution).view(-1,3) * 2 - 1
        return pts

    @torch.no_grad()
    def sample_pts_in_occupied(self, num_pts: int):
        gidx_nonempty = self.occ_grid.nonzero().long() 
        assert gidx_nonempty.numel() > 0, "Occupancy grid becomes empty during training. Your model/algorithm/training settings might be incorrect. Please check configs and tensorboard."
        
        return self._sample_pts_selected(gidx_nonempty, num_pts)

    # if constant_value is not None:
    #     self.occ_val_grid.fill_(constant_value)
    #     self.occ_grid = binarize(self.occ_val_grid, self.occ_thre, self.occ_thre_consider_mean)
    #     return

    @torch.no_grad()
    def _init_constant(self, constant_value: float):
        self.occ_val_grid.fill_(constant_value)
        self.occ_grid = binarize(self.occ_val_grid, self.occ_thre, self.occ_thre_consider_mean)

    @torch.no_grad()
    def _init_from_net(self, val_query_fn, *, num_steps=4, num_pts: int=2**18):
        for _ in trange(num_steps, desc="Init OCC", leave=False):
            # Sample in non-occupied voxels only (usally its all voxels at the first round).
            gidx_empty = self.occ_grid.logical_not().nonzero().long()
            if gidx_empty.shape[0] > 0:
                pts = self._sample_pts_selected(gidx_empty, num_pts)
                val = val_query_fn(pts)
                # No ema here. (ema=1.0)
                update_occ_val_grid_(self.occ_val_grid, pts, self.occ_val_fn(val), ema_decay=1.0)
                self.occ_grid = binarize(self.occ_val_grid, self.occ_thre, self.occ_thre_consider_mean)

    @torch.no_grad()
    def _update_grids_per_iter(self, pts: torch.Tensor, val: torch.Tensor):
        pts, occ_val = pts.flatten(0, -2), self.occ_val_fn(val.flatten())
        gidx = ((pts/2. + 0.5) * self.resolution).long().clamp(self.resolution.new_tensor([0]), self.resolution-1)
        if self.should_gather_samples:
            idx_pcl = self._occ_val_grid_pcl.nonzero().long()
            if idx_pcl.numel() > 0:
                occ_val_pcl = self._occ_val_grid_pcl[tuple(idx_pcl.t())]
                gidx = torch.cat([gidx, idx_pcl], dim=0)
                occ_val = torch.cat([occ_val, occ_val_pcl], dim=0)
            self._occ_val_grid_pcl.zero_()
        update_occ_val_grid_idx_(self.occ_val_grid, gidx, occ_val, ema_decay=self.ema_decay)
        self.occ_grid = binarize(self.occ_val_grid, self.occ_thre, self.occ_thre_consider_mean)

    @torch.no_grad()
    def _update_from_net_per_iter(
        self, cur_it: int, val_query_fn, *, num_steps=4, num_pts: int=2**18):
        if cur_it < self.n_steps_warmup:
            pts_list, vals_list = [], []
            for _ in range(num_steps):
                pts = self._sample_pts_uniform(num_pts)
                val = val_query_fn(pts)
                pts_list.append(pts)
                vals_list.append(val)
            pts_list, vals_list = torch.cat(pts_list, 0), torch.cat(vals_list, 0)
        else:
            pts_list, vals_list = [], []
            n_uniform = int(num_pts // 2)
            n_in_empty = int(num_pts // 4)
            n_in_non_empty = int(num_pts // 4)
            
            gidx_nonempty = self.occ_grid.nonzero().long() 
            gidx_empty = self.occ_grid.logical_not().nonzero().long() 
            assert gidx_nonempty.numel() > 0, "Occupancy grid becomes empty during training. Your model/algorithm/training settings might be incorrect. Please check configs and tensorboard."
            
            for _ in range(num_steps):
                pts1 = self._sample_pts_uniform(n_uniform)
                pts2 = self._sample_pts_selected(gidx_empty, n_in_empty) if gidx_empty.numel() > 0 else torch.empty([0, 3], device=self.device, dtype=self.dtype)
                pts3 = self._sample_pts_selected(gidx_nonempty, n_in_non_empty) if gidx_nonempty.numel() > 0 else torch.empty([0, 3], device=self.device, dtype=self.dtype)
                pts = torch.cat([pts1, pts2, pts3], dim=0)
                val = val_query_fn(pts)
                pts_list.append(pts)
                vals_list.append(val)
            pts_list, vals_list = torch.cat(pts_list, 0), torch.cat(vals_list, 0)
        
        self._update_grids_per_iter(pts_list, vals_list)

    @torch.no_grad()
    def try_shrink(self, old_aabb: torch.Tensor) -> torch.Tensor:
        origin, scale = (old_aabb[1]+old_aabb[0])/2., (old_aabb[1] - old_aabb[0])/2.
        idx = self.occ_grid.nonzero()
        min_idx, max_idx = idx.min(dim=0).values, idx.max(dim=0).values
        new_aabb = torch.stack([min_idx - 1, max_idx + 1], 0).clamp_(max_idx.new_tensor([0]), self.resolution-1)
        new_aabb = ((new_aabb/self.resolution)*2-1) * scale + origin
        return new_aabb

    @torch.no_grad()
    def rescale_volume(self, old_aabb: torch.Tensor, new_aabb: torch.Tensor):
        new_aabb = new_aabb.view(2,3)
        #----------- Rescale val grid
        origin, scale = (old_aabb[1]+old_aabb[0])/2., (old_aabb[1] - old_aabb[0])/2.
        new_origin, new_scale = (new_aabb[1]+new_aabb[0])/2., (new_aabb[1] - new_aabb[0])/2.
        # Vertices in normalized new aabb
        v = (self.grid_coords / self.resolution) * 2. - 1.
        # Vertices in world
        v = v * new_scale + new_origin
        # Vertices in old aabb
        v = (v - origin) / scale
        # Val grid in old aabb
        occ_val_grid = self.occ_val_grid
        # Upsampled val grid in new aabb
        occ_val_grid_new = F.grid_sample(
            occ_val_grid.view([1,*self.resolution.tolist(),1]).permute(0,4,1,2,3), 
            v.view(1,*self.resolution.tolist(),3)[..., [2,1,0]], align_corners=True, padding_mode='zeros').squeeze(0).squeeze(0)
        
        #----------- Debug visualize
        # import vedo
        # from vedo import Box, Plotter, Volume, show
        # plt_spacing = ((new_aabb[1]-new_aabb[0]) / self.resolution).tolist()
        # plt_origin = new_aabb[0].tolist()
        # world_scale = (new_aabb[1]-new_aabb[0])
        # world_origin = (new_aabb[1]+new_aabb[0])/2.
        # vol = Volume(val_grid_new.data.cpu().numpy(), c=['white','b','g','r'], mapper='gpu', origin=plt_origin, spacing=plt_spacing)
        # vox = vol.legosurface(vmin=self.occ_thre, vmax=1., boundary=True)
        # vox.cmap('GnBu', on='cells', vmin=self.occ_thre, vmax=1.).add_scalarbar()
        # world = Box(world_origin.data.cpu().numpy(), *world_scale.tolist()).wireframe()
        # show(world, vox, __doc__, axes=1, viewup='z').close()

        #----------- Set new occ_val grid
        self.occ_val_grid = occ_val_grid_new
        self.occ_grid = binarize(self.occ_val_grid, self.occ_thre, self.occ_thre_consider_mean)

    @torch.no_grad()
    def query(self, pts: torch.Tensor):
        # Expect pts in range [-1,1]
        gidx = ((pts/2.+0.5) * self.resolution).long().clamp(self.resolution.new_tensor([0]), self.resolution-1)
        return self.occ_grid[tuple(gidx.movedim(-1,0))]

class OccupancyGridEMABatched(nn.Module):
    NUM_DIM: int = 3
    def __init__(
        self, num_batches: int, # Number of occ_grid 
        resolution: Union[int, List[int], torch.Tensor] = 128,
        occ_val_fn_cfg=ConfigDict(type='density'), occ_val_fn = None, occ_thre: float = 0.01, 
        occ_thre_consider_mean=False, # Whether consider average value as threshold when binarizing occ_val
        ema_decay: float = 0.95, n_steps_between_update: int = 16, n_steps_warmup: int = 256,
        init_cfg=ConfigDict(), acquire_from_net_cfg=ConfigDict(), acquire_from_samples_cfg=ConfigDict(),
        dtype=torch.float, device=torch.device('cuda')
        ) -> None:
        super().__init__()
        
        self.num_batches = num_batches
        self.dtype = dtype
        self.device = device
        self.ema_decay = ema_decay
        
        if isinstance(resolution, int):
            resolution = [resolution] * self.NUM_DIM
        if isinstance(resolution, (list, tuple, np.ndarray)):
            resolution = torch.tensor(resolution, dtype=torch.int32, device=device)
        elif isinstance(resolution, torch.Tensor):
            resolution = resolution.to(dtype=torch.int32, device=device)
        else:
            raise RuntimeError(f"Invalid type of resolution={type(resolution)}")
        
        self.register_buffer("resolution", resolution, persistent=False)
        self.register_buffer("occ_grid", torch.zeros([num_batches, *resolution.tolist()], dtype=torch.bool, device=self.device), persistent=True)
        self.register_buffer("occ_val_grid", torch.zeros([num_batches, *resolution.tolist()], dtype=self.dtype, device=self.device), persistent=True)
        grid_coords = torch.stack(torch.meshgrid([torch.arange(res, device=self.device) for res in resolution.tolist()], indexing='ij'), dim=-1).view(-1,3)
        self.register_buffer("grid_coords", grid_coords, persistent=False)

        self._register_load_state_dict_pre_hook(self.load_state_dict_hook)

        self.init_cfg = init_cfg
        self.acquire_from_net_cfg = acquire_from_net_cfg
        self.acquire_from_samples_cfg = acquire_from_samples_cfg
        self.should_gather_samples: bool = acquire_from_samples_cfg is not None
        
        self.occ_thre = occ_thre
        self.occ_val_fn = get_occ_val_fn(**occ_val_fn_cfg) if occ_val_fn is None else occ_val_fn
        self.occ_thre_consider_mean = occ_thre_consider_mean
        
        self.n_steps_between_update = n_steps_between_update
        self.n_steps_warmup = n_steps_warmup
        
        if self.should_gather_samples:
            # To gather samples collected during forward & uniform sampling; and use them to update when it's time to update.
            self.register_buffer('_occ_val_grid_pcl', torch.zeros([num_batches, *resolution.tolist()], dtype=self.dtype, device=self.device), persistent=False)

    def load_state_dict_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        occ_val_grid = state_dict[prefix + 'occ_val_grid']
        self.occ_val_grid.resize_(occ_val_grid.shape)
        occ_grid = state_dict[prefix + 'occ_grid']
        self.occ_grid.resize_(occ_grid.shape)
        
        self.num_batches = occ_grid.shape[0]
        self.resolution[:] = self.resolution.new_tensor(occ_grid.shape[1:])
        grid_coords = torch.stack(torch.meshgrid([torch.arange(res, device=self.device) for res in self.resolution.tolist()], indexing='ij'), dim=-1).view(-1,3)
        self.grid_coords = grid_coords

    @torch.no_grad()
    def initialize(
        self, *, mode: Literal['constant', 'from_net'], 
        val_query_fn_normalized_x_with_bidx=None, **kwargs):
        if mode == 'constant':
            self._init_constant(**kwargs)
        elif mode == 'from_net':
            self._init_from_net(val_query_fn_normalized_x_with_bidx, **kwargs)
        else:
            raise RuntimeError(f"Invalid init_mode={mode}")

    @torch.no_grad()
    def preprocess_per_train_step(self, cur_it: int, val_query_fn_normalized_x_with_bidx, logger: Logger=None):
        updated = False
        if cur_it == 0:
            self.initialize(**self.init_cfg, val_query_fn_normalized_x_with_bidx=val_query_fn_normalized_x_with_bidx)
            updated = True
            
        # TODO: When training on multiple GPU, repsect the relationship between the `it` passed in here and the control steps here; 
        #       also respect the warmup steps.
        elif cur_it % self.n_steps_between_update == 0:
            self._update_from_net_per_iter(cur_it, val_query_fn_normalized_x_with_bidx, **self.acquire_from_net_cfg)
            updated = True
        return updated

    @torch.no_grad()
    def postprocess_per_train_step(self, cur_it: int, val_query_fn):
        pass
    
    @torch.no_grad()
    def gather_samples(self, pts: torch.Tensor, bidx: torch.Tensor = None, val: torch.Tensor = ...):
        """
        NOTE: `gather_samples` should be invoked like a forward-hook function.
        """
        if self.training and self.should_gather_samples:
            self._gather_samples(pts.flatten(0,-2), bidx.flatten(), val.flatten(), **self.acquire_from_samples_cfg)

    @torch.no_grad()
    def _gather_samples(self, pts: torch.Tensor, bidx: torch.Tensor = None, val: torch.Tensor = ...):
        # NOTE: No ema decay. We are gathering all possible samples into `_occ_val_grid_pcl`. Only decay when udpate.
        update_batched_occ_val_grid_(self._occ_val_grid_pcl, pts, bidx, self.occ_val_fn(val), ema_decay=1.0)

    @torch.no_grad()
    def _sample_pts_uniform(self, num_pts: int, return_bidx=True):
        grid_coords = self.grid_coords
        num_pts = num_pts // self.num_batches
        num_voxels = grid_coords.shape[0]
        if num_pts / num_voxels < 2.0:
            pts = torch.empty([self.num_batches, num_pts, self.NUM_DIM], device=self.device, dtype=self.dtype).uniform_(-1, 1)
            if return_bidx:
                bidx = torch.arange(self.num_batches, device=self.device).unsqueeze_(-1).expand(-1,num_pts)
                return pts.reshape(-1, 3), bidx.reshape(-1).contiguous()
            else:
                return pts
        else:
            n_per_vox = int(num_pts // num_voxels) + 1
            offsets = torch.rand([self.num_batches, num_voxels, n_per_vox, self.NUM_DIM], device=self.device, dtype=self.dtype)
            pts = ((grid_coords.unsqueeze(-2) + offsets) / self.resolution).view(self.num_batches,-1,3) * 2 -1
            if return_bidx:
                bidx = torch.arange(self.num_batches, device=self.device).unsqueeze_(-1).expand(-1,num_voxels*n_per_vox)
                return pts.reshape(-1,3), bidx.reshape(-1).contiguous()
            else:
                return pts

    @torch.no_grad()
    def _sample_pts_selected(self, bidx: torch.Tensor, gidx: torch.Tensor, num_pts: int):
        num_voxels = gidx.shape[0]
        if num_pts / num_voxels < 2.0:
            inds = torch.randint(num_voxels, [num_pts, ], device=self.device)
            offsets = torch.rand([num_pts, self.NUM_DIM], device=self.device, dtype=self.dtype)
            pts = ((gidx[inds] + offsets) / self.resolution) * 2 - 1
            return pts, bidx[inds]
        else:
            n_per_vox = int(num_pts // num_voxels) + 1
            offsets = torch.rand([num_voxels, n_per_vox, self.NUM_DIM], device=self.device, dtype=self.dtype)
            pts = ((gidx[:,None,:] + offsets) / self.resolution).view(-1,3) * 2 - 1
            return pts, bidx[:,None].expand(num_voxels, n_per_vox).reshape(-1).contiguous()

    @torch.no_grad()
    def _init_constant(self, constant_value: float):
        self.occ_val_grid.fill_(constant_value)
        self.occ_grid = binarize(self.occ_val_grid, self.occ_thre, self.occ_thre_consider_mean)

    @torch.no_grad()
    def _init_from_net(
        self, val_query_fn_normalized_x_with_bidx, *, num_steps=4, num_pts_per_batch: int=2**18, num_pts: int = None):
        if num_pts is None:
            num_pts = num_pts_per_batch * self.num_batches
        for _ in range(num_steps):
            # Sample in non-occupied voxels only (usally its all voxels at the first round).
            idx_empty = self.occ_grid.logical_not().nonzero().long()
            bidx_empty, gidx_empty = idx_empty[..., 0], idx_empty[..., 1:]
            if gidx_empty.shape[0] > 0:
                pts, bidx = self._sample_pts_selected(bidx_empty, gidx_empty, num_pts)
                val = val_query_fn_normalized_x_with_bidx(pts, bidx)
                # No ema here. (ema=1.0)
                update_batched_occ_val_grid_(self.occ_val_grid, pts, bidx, self.occ_val_fn(val), ema_decay=1.0)
                self.occ_grid = binarize(self.occ_val_grid, self.occ_thre, self.occ_thre_consider_mean)

    @torch.no_grad()
    def _update_grids_per_iter(self, pts: torch.Tensor, bidx: torch.Tensor = None, val: torch.Tensor = ...):
        if self.should_gather_samples:
            idx_pcl_all = self._occ_val_grid_pcl.nonzero().long()
            if idx_pcl_all.numel() > 0:
                bidx_pcl, gidx_pcl = idx_pcl_all[..., 0], idx_pcl_all[..., 1:]
                occ_val_pcl = self._occ_val_grid_pcl[(bidx_pcl, ) + tuple(gidx_pcl.t())]
            else:
                bidx_pcl = None
            self._occ_val_grid_pcl.zero_()
        else:
            bidx_pcl = None
        
        if bidx is not None:
            # Not batched
            pts, bidx, occ_val = pts.flatten(0, -2), bidx.flatten(), self.occ_val_fn(val).flatten()
            gidx = ((pts/2. + 0.5) * self.resolution).long().clamp(self.resolution.new_tensor([0]), self.resolution-1)
            if bidx_pcl is not None:
                bidx = torch.cat([bidx, bidx_pcl], dim=0)
                gidx = torch.cat([gidx.view(-1,3), gidx_pcl], dim=0)
                occ_val = torch.cat([occ_val.view(-1), occ_val_pcl], dim=0)
        else:
            # Batched
            pts, occ_val = pts.flatten(1, -2), self.occ_val_fn(val).flatten(1, -1)
            gidx = ((pts/2. + 0.5) * self.resolution).long().clamp(self.resolution.new_tensor([0]), self.resolution-1)
            if bidx_pcl is not None:
                # should_gather_samples, change batched mode to non-batched mode by calculating an auxillary bidx
                bidx = torch.arange(self.num_batches, device=self.device).view(-1,1).expand(-1, pts.shape[1]).reshape(-1)
                bidx = torch.cat([bidx, bidx_pcl], dim=0)
                gidx = torch.cat([gidx.view(-1,3), gidx_pcl], dim=0)
                occ_val = torch.cat([occ_val.view(-1), occ_val_pcl], dim=0)

        update_batched_occ_val_grid_idx_(self.occ_val_grid, bidx, gidx, occ_val, ema_decay=self.ema_decay)
        self.occ_grid = binarize(self.occ_val_grid, self.occ_thre, self.occ_thre_consider_mean)

    @torch.no_grad()
    def _update_from_net_per_iter(
        self, cur_it: int, val_query_fn_normalized_x_with_bidx, *, num_steps=4, num_pts_per_batch: int=2**18, num_pts: int = None):
        if num_pts is None: 
            num_pts = num_pts_per_batch * self.num_batches
        if cur_it < self.n_steps_warmup:
            pts_list, bidx_list, vals_list = [], [], []
            for _ in range(num_steps):
                pts, bidx = self._sample_pts_uniform(num_pts)
                val = val_query_fn_normalized_x_with_bidx(pts, bidx)
                pts_list.append(pts)
                bidx_list.append(bidx)
                vals_list.append(val)
            pts_list, bidx_list, vals_list = torch.cat(pts_list, 0), torch.cat(bidx_list, 0), torch.cat(vals_list, 0)
        else:
            pts_list, bidx_list, vals_list = [], [], []
            n_uniform = int(num_pts // 2)
            n_in_empty = int(num_pts // 4)
            n_in_non_empty = int(num_pts // 4)
            
            idx_nonempty = self.occ_grid.nonzero().long()
            bidx_nonempty, gidx_nonempty = idx_nonempty[..., 0], idx_nonempty[..., 1:]
            idx_empty = self.occ_grid.logical_not().nonzero().long()
            bidx_empty, gidx_empty = idx_empty[..., 0], idx_empty[..., 1:]
            assert idx_nonempty.numel() > 0, "Occupancy grid becomes empty during training. Your model/algorithm/training setting might be incorrect. Please check."
            
            for _ in range(num_steps):
                pts1, bidx1 = self._sample_pts_uniform(n_uniform)
                pts2, bidx2 = self._sample_pts_selected(bidx_empty, gidx_empty, n_in_empty) if gidx_empty.numel() > 0 else (torch.empty([0, 3], device=self.device, dtype=self.dtype), torch.empty([0], device=self.device, dtype=torch.long))
                pts3, bidx3 = self._sample_pts_selected(bidx_nonempty, gidx_nonempty, n_in_non_empty) if gidx_nonempty.numel() > 0 else (torch.empty([0, 3], device=self.device, dtype=self.dtype), torch.empty([0], device=self.device, dtype=torch.long))
                pts = torch.cat([pts1, pts2, pts3], dim=0)
                bidx = torch.cat([bidx1, bidx2, bidx3], dim=0)
                val = val_query_fn_normalized_x_with_bidx(pts, bidx)
                pts_list.append(pts)
                bidx_list.append(bidx)
                vals_list.append(val)
            pts_list, bidx_list, vals_list = torch.cat(pts_list, 0), torch.cat(bidx_list, 0), torch.cat(vals_list, 0)
        
        self._update_grids_per_iter(pts_list, bidx_list, vals_list)

    @torch.no_grad()
    def query(self, pts: torch.Tensor, bidx: torch.Tensor):
        # Expect pts in range [-1,1]
        gidx = ((pts/2.+0.5) * self.resolution).long().clamp(self.resolution.new_tensor([0]), self.resolution-1)
        return self.occ_grid[(bidx,)+tuple(gidx.movedim(-1,0))]

class OccupancyGridGetter(nn.Module):
    NUM_DIM: int = 3
    def __init__(
        self, 
        resolution: Union[int, List[int], torch.Tensor] = 128,
        occ_val_fn_cfg=ConfigDict(type='density'), occ_val_fn = None, occ_thre: float = 0.01, 
        occ_thre_consider_mean=False, # Whether consider average value as threshold when binarizing occ_val
        num_steps=4, num_pts_per_batch: int=2**18, 
        num_pts: int = None,
        dtype=torch.float, device=torch.device('cuda')
        ) -> None:
        super().__init__()
        
        self.dtype = dtype
        self.device = device
        
        if num_pts is not None:
            num_pts_per_batch = num_pts
            log.warn(f"Warning: `num_pts` in batched occ grid getter is deprecated. Use `num_pts_per_batch` instead.")
        
        if isinstance(resolution, int):
            resolution = [resolution] * self.NUM_DIM
        if isinstance(resolution, (list, tuple, np.ndarray)):
            resolution = torch.tensor(resolution, dtype=torch.int32, device=device)
        elif isinstance(resolution, torch.Tensor):
            resolution = resolution.to(dtype=torch.int32, device=device)
        else:
            raise RuntimeError(f"Invalid type of resolution={type(resolution)}")

        self.register_buffer("resolution", resolution, persistent=False)
        grid_coords = torch.stack(torch.meshgrid([torch.arange(res, device=self.device) for res in resolution.tolist()], indexing='ij'), dim=-1).view(-1,3)
        self.register_buffer("grid_coords", grid_coords, persistent=False)

        self.occ_thre = occ_thre
        self.occ_val_fn = get_occ_val_fn(**occ_val_fn_cfg) if occ_val_fn is None else occ_val_fn
        self.occ_thre_consider_mean = occ_thre_consider_mean

        self.num_steps = num_steps
        self.num_pts = num_pts
        self.num_pts_per_batch = num_pts_per_batch
        
    @torch.no_grad()
    def occ_grid_from_net(self, val_query_fn, progress=False) -> torch.Tensor:
        """
        Always uniformly sample in all voxels
        """
        num_steps, num_pts, grid_coords = self.num_steps, self.num_pts, self.grid_coords
        occ_grid = torch.zeros(self.resolution.tolist(), dtype=torch.bool, device=self.device)
        for _ in trange(num_steps, disable=not progress):
            n_per_vox = int(num_pts // grid_coords.shape[0]) + 1
            offsets = torch.rand([grid_coords.shape[0], n_per_vox, self.NUM_DIM], device=self.device, dtype=self.dtype)
            pts = ((grid_coords[:,None,:] + offsets) / self.resolution) * 2 -1
            val = val_query_fn(pts)
            occ_val = self.occ_val_fn(val.flatten()).view(grid_coords.shape[0], n_per_vox)
            occ_grid_iter = binarize(occ_val, self.occ_thre, self.occ_thre_consider_mean).any(dim=-1).view(self.resolution.tolist())
            occ_grid |= occ_grid_iter
        return occ_grid

    @torch.no_grad()
    def occ_grid_from_net_v2(self, val_query_fn, progress=False, verbose=False) -> torch.Tensor:
        """
        Each step, only uniformly sample in currently empty voxels
        """
        num_steps, num_pts, grid_coords = self.num_steps, self.num_pts, self.grid_coords
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
            pts = ((grid_coords.view(*resolution, 3)[tuple(idx.t())].unsqueeze(-2) + offsets) / self.resolution)*2-1
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
        Batched net that does NOT support batch_inds input
        """
        num_steps, num_pts_per_batch, grid_coords = self.num_steps, self.num_pts_per_batch, self.grid_coords
        occ_grid = torch.zeros([B, *self.resolution.tolist()], dtype=torch.bool, device=self.device)
        for _ in trange(num_steps, disable=not progress):
            n_per_vox = int(num_pts_per_batch // grid_coords.shape[0]) + 1
            offsets = torch.rand([B, grid_coords.shape[0], n_per_vox, self.NUM_DIM], device=self.device, dtype=self.dtype)
            pts = ((grid_coords[None,:,None,:] + offsets) / self.resolution) * 2 -1
            val = val_query_fn_batched(pts)
            occ_val = self.occ_val_fn(val.flatten()).view(grid_coords.shape[0], n_per_vox)
            occ_grid_iter = binarize(occ_val, self.occ_thre, self.occ_thre_consider_mean).any(dim=-1).view(occ_grid.shape)
            occ_grid |= occ_grid_iter
        return occ_grid

    @torch.no_grad()
    def occ_grid_from_net_batched_v2(self, B: int, val_query_fn_normalized_x_with_bidx, progress=False) -> torch.Tensor:
        """
        Batched net that supports batch_inds input
        """
        num_steps, num_pts_per_batch, grid_coords = self.num_steps, self.num_pts_per_batch, self.grid_coords
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
            pts = ((grid_coords.view(*resolution, 3)[tuple(gidx.t())].unsqueeze(-2) + offsets) / self.resolution)*2-1
            val = val_query_fn_normalized_x_with_bidx(pts, bidx.view(num_voxels, 1).expand(num_voxels, n_per_vox))
            occ_val = self.occ_val_fn(val.flatten()).view(num_voxels, n_per_vox).max(-1).values
            
            # Scatter max in currently empty voxels
            # [(0)*(1)*(2)*1, (1)*(2)*1, (2)*1, 1]
            idx_ravel = (idx * idx.new_tensor([prod(resolution), resolution[1] * resolution[2], resolution[2], 1])).sum(-1)
            occ_val_grid_iter, _ = scatter_max(occ_val, idx_ravel, dim=0, dim_size=prod(occ_grid.shape))
            occ_grid |= binarize(occ_val_grid_iter, self.occ_thre, self.occ_thre_consider_mean).view(occ_grid.shape)
        return occ_grid

if __name__ == "__main__":
    def unit_test(device=torch.device('cuda')):
        num_pts = 4396
        occ = OccupancyGridEMA([32, 32, 32], {'type':'sdf', 's': 64.0}, occ_thre=0.3, device=device)
        
        def dummy_sdf_query(x: torch.Tensor):
            # Expect x to be of range [-1,1]
            return x.norm(dim=-1) - 0.5 # A sphere of radius=0.5
        
        # Init from net
        occ._init_from_net(dummy_sdf_query, num_steps=4, num_pts=2**18)
        
        # Pure update
        pts = torch.rand([num_pts, 3], device=device) * 2 - 1
        val = dummy_sdf_query(pts)
        occ._update_grids_per_iter(pts, val)
        
        # Gather sample
        pts = torch.rand([num_pts, 3], device=device) * 2 - 1
        val = dummy_sdf_query(pts)
        occ.gather_samples(pts, val)
        # Update after gather sample
        pts = torch.rand([num_pts, 3], device=device) * 2 - 1
        val = dummy_sdf_query(pts)
        occ._update_grids_per_iter(pts, val)
        
        # Update from net (during warmup)
        occ._update_from_net_per_iter(0, dummy_sdf_query, num_steps=4, num_pts=2**18)
        
        # Update from net
        occ._update_from_net_per_iter(10000, dummy_sdf_query, num_steps=4, num_pts=2**18)
        
        # Visulization
        from vedo import Volume, show
        occ_val_grid = occ.occ_val_grid.data.cpu().numpy()
        aabb = torch.tensor([[-1,-1,-1], [1,1,1]], dtype=torch.float, device=device)
        spacing = ((aabb[1]-aabb[0]) / occ.resolution).tolist()
        origin = aabb[0].tolist()
        vol = Volume(occ_val_grid, c=['white','b','g','r'], mapper='gpu', origin=origin, spacing=spacing)
        vox = vol.legosurface(vmin=occ.occ_thre, vmax=1., boundary=True)
        vox.cmap('GnBu', on='cells', vmin=occ.occ_thre, vmax=1.).add_scalarbar()
        show(vox, __doc__, axes=1, viewup='z').close()
    
    def unit_test_batched(device=torch.device('cuda')):
        num_batch = 7
        num_pts = 4396
        occ = OccupancyGridEMABatched(num_batch, [32, 32, 32], {'type':'sdf', 's': 64.0}, occ_thre=0.3, device=device)
        
        dummy_radius = torch.empty([num_batch], device=device, dtype=torch.float).uniform_(0.3, 0.7)
        def dummy_sdf_query(x: torch.Tensor, bidx: torch.Tensor = None):
            # Expect x to be of range [-1,1]
            if bidx is not None:
                radius = dummy_radius[bidx]
                return x.norm(dim=-1) - radius # A sphere of radius
            else:
                return x.norm(dim=-1) - dummy_radius.view(-1, *[1]*(x.dim()-2))
        
        # Init from net
        occ._init_from_net(dummy_sdf_query, num_steps=4, num_pts=2**18)
        
        # Batched (pure update)
        pts = torch.rand([num_batch, num_pts, 3], device=device) * 2 - 1
        val = dummy_sdf_query(pts)
        occ._update_grids_per_iter(pts, None, val)
        
        # Not batched (pure update)
        pts = torch.rand([num_pts, 3], device=device) * 2 - 1
        bidx = torch.randint(num_batch, [num_pts, ], device=device)
        val = dummy_sdf_query(pts, bidx)
        occ._update_grids_per_iter(pts, bidx, val)
        
        # Batched (gather sample)
        pts = torch.rand([num_batch, num_pts, 3], device=device) * 2 - 1
        val = dummy_sdf_query(pts)
        occ.gather_samples(pts, None, val)
        # Batched (update after gather sample)
        pts = torch.rand([num_batch, num_pts, 3], device=device) * 2 - 1
        val = dummy_sdf_query(pts)
        occ._update_grids_per_iter(pts, None, val)

        # Not bathced (gather sample)
        pts = torch.rand([num_pts, 3], device=device) * 2 - 1
        bidx = torch.randint(num_batch, [num_pts, ], device=device)
        val = dummy_sdf_query(pts, bidx)
        occ.gather_samples(pts, bidx, val)
        # Not batched (update after gather sample)
        pts = torch.rand([num_pts, 3], device=device) * 2 - 1
        bidx = torch.randint(num_batch, [num_pts, ], device=device)
        val = dummy_sdf_query(pts, bidx)
        occ._update_grids_per_iter(pts, bidx, val)
        
        # Update from net (during warmup)
        occ._update_from_net_per_iter(0, dummy_sdf_query, num_steps=4, num_pts=2**18)
        
        # Update from net
        occ._update_from_net_per_iter(10000, dummy_sdf_query, num_steps=4, num_pts=2**18)

        # Batched Visulization
        from vedo import Volume, show
        batched_val_grid = occ.occ_val_grid.data.cpu().numpy()
        vox_actors = []
        aabb = torch.tensor([[-1,-1,-1], [1,1,1]], dtype=torch.float, device=device)
        spacing = ((aabb[1]-aabb[0]) / occ.resolution).tolist()
        for i, occ_val_grid in enumerate(batched_val_grid):
            origin = (aabb[0] + torch.tensor([2. * i, 0., 0.], device=device)).tolist()
            vol = Volume(occ_val_grid, c=['white','b','g','r'], mapper='gpu', origin=origin, spacing=spacing)
            vox = vol.legosurface(vmin=occ.occ_thre, vmax=1., boundary=True)
            vox.cmap('GnBu', on='cells', vmin=occ.occ_thre, vmax=1.).add_scalarbar()
            vox_actors.append(vox)
        show(*vox_actors, __doc__, axes=1, viewup='z').close()
    
    unit_test()
    unit_test_batched()
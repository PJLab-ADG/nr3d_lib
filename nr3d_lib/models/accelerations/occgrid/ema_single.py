__all__ = [
    'OccGridEma'
]

import numpy as np
from tqdm import trange
from copy import deepcopy
from typing import List, Literal, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from nr3d_lib.logger import Logger
from .utils import *

class OccGridEma(nn.Module):
    """
    Coordinates here are assumed to be normalized; in [-1,1] range.
    """
    NUM_DIM: int = 3
    def __init__(
        self, 
        resolution: Union[int, List[int], torch.Tensor] = 128,
        occ_val_fn_cfg=dict(type='density'), occ_val_fn = None, occ_thre: float = 0.01, 
        occ_thre_consider_mean=False, # Whether consider average value as threshold when binarizing occ_val
        ema_decay: float = 0.95, n_steps_between_update: int = 16, n_steps_warmup: int = 256,
        init_cfg=dict(), update_from_net_cfg = dict(), update_from_samples_cfg = dict(),
        dtype=torch.float, device=None) -> None:
        super().__init__()
    
        self.dtype = dtype
        
        if isinstance(resolution, int):
            resolution = [resolution] * self.NUM_DIM
        if isinstance(resolution, (list, tuple, np.ndarray)):
            resolution = torch.tensor(resolution, dtype=torch.int32, device=device)
        elif isinstance(resolution, torch.Tensor):
            resolution = resolution.to(dtype=torch.int32, device=device)
        else:
            raise RuntimeError(f"Invalid type of resolution={type(resolution)}")
        
        self.register_buffer('is_initialized', torch.tensor([False], dtype=torch.bool), persistent=True)
        self.register_buffer("resolution", resolution, persistent=False)
        self.register_buffer("occ_grid", torch.zeros(resolution.tolist(), dtype=torch.bool, device=device), persistent=True)
        self.register_buffer("occ_val_grid", torch.zeros(resolution.tolist(), dtype=self.dtype, device=device), persistent=True)
        gidx_full = torch.stack(
            torch.meshgrid(
                [torch.arange(res, device=device) for res in resolution.tolist()], 
                indexing='ij'
            ), dim=-1
        ).view(-1,self.NUM_DIM)
        self.register_buffer("gidx_full", gidx_full, persistent=False)
        
        self._register_load_state_dict_pre_hook(self.before_load_state_dict)
        
        self.ema_decay = ema_decay
        self.init_cfg = init_cfg
        self.update_from_net_cfg = update_from_net_cfg
        self.update_from_samples_cfg = update_from_samples_cfg
        self.should_collect_samples: bool = update_from_samples_cfg is not None
        
        self.occ_thre = occ_thre
        self.occ_val_fn = get_occ_val_fn(**occ_val_fn_cfg) if occ_val_fn is None else occ_val_fn
        self.occ_thre_consider_mean = occ_thre_consider_mean
        
        self.n_steps_between_update = n_steps_between_update
        self.n_steps_warmup = n_steps_warmup
        
        if self.should_collect_samples:
            # To gather samples collected during forward & uniform sampling; and use them to update when it's time to update.
            self.register_buffer('_occ_val_grid_pcl', torch.zeros(resolution.tolist(), dtype=self.dtype, device=device), persistent=False)
    
    @property
    def device(self) -> torch.device:
        return self.resolution.device
    
    def before_load_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # TODO: Just for compatibility. To be removed later.
        if prefix + 'val_grid' in state_dict:
            state_dict[prefix+'occ_val_grid'] = state_dict.pop(prefix+'val_grid')
        
        val_grid = state_dict[prefix + 'occ_val_grid']
        self.occ_val_grid.resize_(val_grid.shape)
        occ_grid = state_dict[prefix + 'occ_grid']
        self.occ_grid.resize_(occ_grid.shape)
        
        self.resolution[:] = self.resolution.new_tensor(occ_grid.shape)
        gidx_full = torch.stack(
            torch.meshgrid(
                [torch.arange(res, device=self.device) for res in self.resolution.tolist()], 
                indexing='ij'
            ), dim=-1
        ).view(-1,self.NUM_DIM)
        self.gidx_full = gidx_full

    """ Init """
    @torch.no_grad()
    def init(self, val_query_fn=None, logger: Logger=None) -> bool:
        updated = False
        if not self.is_initialized:
            init_cfg = deepcopy(self.init_cfg)
            init_mode = init_cfg.pop('mode')
            if init_mode == 'constant':
                self._init_from_constant(**init_cfg)
            elif init_mode == 'from_net':
                self._init_from_net(val_query_fn, **init_cfg)
            else:
                raise RuntimeError(f"Invalid init_mode={init_mode}")
            self.is_initialized.fill_(True)
            updated = True
        return updated

    @torch.no_grad()
    def _init_from_constant(self, constant_value: float):
        self.occ_val_grid.fill_(constant_value)
        self.occ_grid = binarize(self.occ_val_grid, self.occ_thre, self.occ_thre_consider_mean)

    @torch.no_grad()
    def _init_from_net(self, val_query_fn, *, num_steps=4, num_pts: int=2**18):
        for _ in trange(num_steps, desc="Init OCC", leave=False):
            # Sample in non-occupied voxels only (usally its all voxels at the first round).
            gidx_empty = self.occ_grid.logical_not().nonzero().long()
            if gidx_empty.shape[0] > 0:
                pts = sample_pts_in_voxels(gidx_empty, num_pts, resolution=self.resolution, dtype=self.dtype)[0]
                val = val_query_fn(pts)
                # No ema here. (ema=1.0)
                update_occ_val_grid_(self.occ_val_grid, pts, self.occ_val_fn(val), ema_decay=1.0)
                self.occ_grid = binarize(self.occ_val_grid, self.occ_thre, self.occ_thre_consider_mean)

    """ Step per iter """
    @torch.no_grad()
    def step(self, cur_it: int, val_query_fn, logger: Logger=None):
        assert self.is_initialized, f"{type(self)} should init() first before step(cur_it={cur_it})"
        # NOTE: Skip cur_it==0
        updated = False
        if (cur_it > 0) and (cur_it % self.n_steps_between_update == 0):
            self._step(cur_it, val_query_fn, **self.update_from_net_cfg)
            updated = True
        return updated
    
    @torch.no_grad()
    def _step(
        self, cur_it: int, val_query_fn, *, num_steps=4, num_pts: int=2**18):
        if cur_it < self.n_steps_warmup:
            pts_list, vals_list = [], []
            for _ in range(num_steps):
                pts = sample_pts_in_voxels(self.gidx_full, num_pts, resolution=self.resolution, dtype=self.dtype)[0]
                val = val_query_fn(pts)
                pts_list.append(pts)
                vals_list.append(val)
            pts_list, vals_list = torch.cat(pts_list, 0), torch.cat(vals_list, 0)
        else:
            pts_list, vals_list = [], []
            n_uniform = int(num_pts // 2)
            n_in_empty = int(num_pts // 4)
            n_in_nonempty = int(num_pts // 4)
            
            gidx_nonempty = self.occ_grid.nonzero().long() 
            gidx_empty = self.occ_grid.logical_not().nonzero().long() 
            assert gidx_nonempty.numel() > 0, "Occupancy grid becomes empty during training. Your model/algorithm/training settings might be incorrect. Please check configs and tensorboard."
            
            for _ in range(num_steps):
                _pts_list = [sample_pts_in_voxels(self.gidx_full, n_uniform, resolution=self.resolution, dtype=self.dtype)[0]]
                if gidx_empty.numel() > 0:
                    _pts_list.append(sample_pts_in_voxels(gidx_empty, n_in_empty, resolution=self.resolution, dtype=self.dtype)[0])
                if gidx_nonempty.numel() > 0:
                    _pts_list.append(sample_pts_in_voxels(gidx_nonempty, n_in_nonempty, resolution=self.resolution, dtype=self.dtype)[0])
                pts = torch.cat(_pts_list, dim=0)
                val = val_query_fn(pts)
                pts_list.append(pts)
                vals_list.append(val)
            pts_list, vals_list = torch.cat(pts_list, 0), torch.cat(vals_list, 0)
        
        self._step_update_occ(pts_list, vals_list)

    @torch.no_grad()
    def _step_update_occ(self, pts: torch.Tensor, val: torch.Tensor):
        pts, occ_val = pts.flatten(0, -2), self.occ_val_fn(val.flatten())
        gidx = ((pts/2. + 0.5) * self.resolution).long().clamp(self.resolution.new_tensor([0]), self.resolution-1)
        if self.should_collect_samples:
            idx_pcl = self._occ_val_grid_pcl.nonzero().long()
            if idx_pcl.numel() > 0:
                occ_val_pcl = self._occ_val_grid_pcl[tuple(idx_pcl.t())]
                gidx = torch.cat([gidx, idx_pcl], dim=0)
                occ_val = torch.cat([occ_val, occ_val_pcl], dim=0)
            self._occ_val_grid_pcl.zero_()
        update_occ_val_grid_idx_(self.occ_val_grid, gidx, occ_val, ema_decay=self.ema_decay)
        self.occ_grid = binarize(self.occ_val_grid, self.occ_thre, self.occ_thre_consider_mean)

    """ (Optional) Collect samples from ray march or rendering process """
    @torch.no_grad()
    def collect_samples(self, pts: torch.Tensor, val: torch.Tensor=None):
        """
        NOTE: `collect_samples` should be invoked like a forward-hook function.
        """
        if self.training and self.should_collect_samples:
            self._collect_samples(pts, val, **self.update_from_samples_cfg)

    @torch.no_grad()
    def _collect_samples(self, pts: torch.Tensor, val: torch.Tensor):
        update_occ_val_grid_(self._occ_val_grid_pcl, pts, self.occ_val_fn(val), ema_decay=1.0)

    """ Sampling or querying from the occ grid """
    @torch.no_grad()
    def sample_pts_in_occupied(self, num_pts: int) -> torch.Tensor:
        gidx_nonempty = self.occ_grid.nonzero().long() 
        assert gidx_nonempty.numel() > 0, "Occupancy grid becomes empty during training. Your model/algorithm/training settings might be incorrect. Please check configs and tensorboard."
        pts, _ = sample_pts_in_voxels(gidx_nonempty, num_pts, resolution=self.resolution, dtype=self.dtype)
        return pts

    @torch.no_grad()
    def query(self, pts: torch.Tensor) -> torch.BoolTensor:
        # Expect pts in range [-1,1]
        gidx = ((pts/2.+0.5) * self.resolution).long().clamp(self.resolution.new_tensor([0]), self.resolution-1)
        return self.occ_grid[tuple(gidx.movedim(-1,0))]

    """ Volume shriking (rescaling) """
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
        new_aabb = new_aabb.view(2,self.NUM_DIM)
        #----------- Rescale val grid
        origin, scale = (old_aabb[1]+old_aabb[0])/2., (old_aabb[1] - old_aabb[0])/2.
        new_origin, new_scale = (new_aabb[1]+new_aabb[0])/2., (new_aabb[1] - new_aabb[0])/2.
        # Vertices in normalized new aabb
        v = (self.gidx_full / self.resolution) * 2. - 1.
        # Vertices in world
        v = v * new_scale + new_origin
        # Vertices in old aabb
        v = (v - origin) / scale
        # Val grid in old aabb
        occ_val_grid = self.occ_val_grid
        # Upsampled val grid in new aabb
        occ_val_grid_new = F.grid_sample(
            occ_val_grid.view([1,*self.resolution.tolist(),1]).movedim(-1,1), 
            v.view(1,*self.resolution.tolist(),self.NUM_DIM)[..., list(range(self.NUM_DIM-1,-1,-1))], # Reversed view
            align_corners=True, padding_mode='zeros'
        ).squeeze(0).squeeze(0)
        
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

    def extra_repr(self) -> str:
        occ_grid_shape_str = '[' + ','.join([str(s) for s in self.occ_grid.shape]) + ']'
        return f"occ_grid={occ_grid_shape_str}"

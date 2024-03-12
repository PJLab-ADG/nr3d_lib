__all__ = [
    'OccGridEmaBatched'
]

import numpy as np
from tqdm import trange
from copy import deepcopy
from typing import Callable, List, Literal, Tuple, Union

import torch
import torch.nn as nn

from nr3d_lib.logger import Logger
from .utils import *

class OccGridEmaBatched(nn.Module):
    NUM_DIM: int = 3
    def __init__(
        self, num_batches: int, # Number of occ_grid 
        resolution: Union[int, List[int], torch.Tensor] = 128,
        occ_val_fn_cfg=dict(type='density'), occ_val_fn = None, occ_thre: float = 0.01, 
        occ_thre_consider_mean=False, # Whether consider average value as threshold when binarizing occ_val
        ema_decay: float = 0.95, n_steps_between_update: int = 16, n_steps_warmup: int = 256,
        init_cfg=dict(), update_from_net_cfg=dict(), update_from_samples_cfg=dict(),
        dtype=torch.float, device=None
        ) -> None:
        super().__init__()
        self.num_batches = num_batches
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
        self.register_buffer("occ_grid", torch.zeros([num_batches, *resolution.tolist()], dtype=torch.bool, device=device), persistent=True)
        self.register_buffer("occ_val_grid", torch.zeros([num_batches, *resolution.tolist()], dtype=self.dtype, device=device), persistent=True)
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
            self.register_buffer('_occ_val_grid_pcl', torch.zeros([num_batches, *resolution.tolist()], dtype=self.dtype, device=device), persistent=False)

    @property
    def device(self) -> torch.device:
        return self.resolution.device

    def before_load_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        occ_val_grid = state_dict[prefix + 'occ_val_grid']
        self.occ_val_grid.resize_(occ_val_grid.shape)
        occ_grid = state_dict[prefix + 'occ_grid']
        self.occ_grid.resize_(occ_grid.shape)
        
        self.num_batches = occ_grid.shape[0]
        self.resolution[:] = self.resolution.new_tensor(occ_grid.shape[1:])
        gidx_full = torch.stack(
            torch.meshgrid(
                [torch.arange(res, device=self.device) for res in self.resolution.tolist()]
                , indexing='ij'
            ), dim=-1
        ).view(-1,self.NUM_DIM)
        self.gidx_full = gidx_full

    """ Init """
    def init(
        self, 
        val_query_fn_normalized_x_bi: Callable[[torch.Tensor,torch.LongTensor],torch.Tensor]=None, 
        logger: Logger=None) -> bool:
        updated = False
        if not self.is_initialized:
            init_cfg = deepcopy(self.init_cfg)
            init_mode = init_cfg.pop('mode')
            if init_mode == 'constant':
                self._init_from_constant(**init_cfg)
            elif init_mode == 'from_net' or init_mode == 'net':
                self._init_from_net(val_query_fn_normalized_x_bi, **init_cfg)
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
    def _init_from_net(
        self, val_query_fn_normalized_x_bi, *, 
        num_steps=4, num_pts_per_batch: int=2**18, num_pts: int = None):
        
        if num_pts is None:
            num_pts = num_pts_per_batch * self.num_batches
        for _ in trange(num_steps, desc="Init OCC", leave=False):
            # Sample in non-occupied voxels only (usally its all voxels at the first round).
            idx_empty = self.occ_grid.logical_not().nonzero().long()
            bidx_empty, gidx_empty = idx_empty[..., 0], idx_empty[..., 1:]
            if gidx_empty.shape[0] > 0:
                pts, vidx = sample_pts_in_voxels(gidx_empty, num_pts, self.resolution, dtype=self.dtype)
                bidx = bidx_empty[vidx]
                val = val_query_fn_normalized_x_bi(pts, bidx=bidx)
                # No ema here. (ema=1.0)
                update_batched_occ_val_grid_(self.occ_val_grid, pts, bidx, self.occ_val_fn(val), ema_decay=1.0)
                self.occ_grid = binarize(self.occ_val_grid, self.occ_thre, self.occ_thre_consider_mean)

    """ Step per iter """
    @torch.no_grad()
    def step(
        self, cur_it: int, 
        val_query_fn_normalized_x_bi: Callable[[torch.Tensor,torch.LongTensor],torch.Tensor], 
        within_bi: torch.LongTensor = None, 
        logger: Logger=None) -> bool:
        assert self.is_initialized, f"{type(self)} should init() first before step(cur_it={cur_it})"
        # NOTE: Skip cur_it==0
        updated = False
        if (cur_it > 0) and (cur_it % self.n_steps_between_update == 0):
            self._step(cur_it, val_query_fn_normalized_x_bi, within_bi=within_bi, 
                       **self.update_from_net_cfg)
            updated = True
        return updated

    @torch.no_grad()
    def _step(
        self, 
        cur_it: int, 
        val_query_fn_normalized_x_bi, 
        *, 
        within_bi: torch.LongTensor = None, 
        num_steps=4, num_pts_per_batch: int=2**18, num_pts: int = None):
        
        if num_pts is None: 
            num_pts = num_pts_per_batch * self.num_batches

        if within_bi is not None:
            num_batches = len(within_bi)
        else:
            num_batches = self.num_batches

        if cur_it < self.n_steps_warmup:
            pts_list, bidx_list, vals_list = [], [], []
            for _ in range(num_steps):
                # Uniform sample
                pts, _ = sample_pts_in_voxels(self.gidx_full, num_pts, self.resolution, dtype=self.dtype)
                bidx = torch.randint(num_batches, size=(len(pts),), dtype=torch.long, device=self.device)
                val = val_query_fn_normalized_x_bi(pts, bidx=bidx)
                pts_list.append(pts)
                bidx_list.append(bidx)
                vals_list.append(val)
            pts_list, bidx_list, vals_list = torch.cat(pts_list, 0), torch.cat(bidx_list, 0), torch.cat(vals_list, 0)
        else:
            if within_bi is not None:
                occ_grid = self.occ_grid[within_bi].contiguous()
            else:
                occ_grid = self.occ_grid
            
            pts_list, bidx_list, vals_list = [], [], []
            n_uniform = int(num_pts // 2)
            n_in_empty = int(num_pts // 4)
            n_in_nonempty = int(num_pts // 4)
            
            idx_nonempty = occ_grid.nonzero().long()
            bidx_nonempty, gidx_nonempty = idx_nonempty[..., 0], idx_nonempty[..., 1:]
            idx_empty = occ_grid.logical_not().nonzero().long()
            bidx_empty, gidx_empty = idx_empty[..., 0], idx_empty[..., 1:]
            assert idx_nonempty.numel() > 0, "Occupancy grid becomes empty during training. Your model/algorithm/training setting might be incorrect. Please check."
            
            for _ in range(num_steps):
                # Uniform sample
                pts, _ = sample_pts_in_voxels(self.gidx_full, n_uniform, self.resolution, dtype=self.dtype)
                bidx = torch.randint(num_batches, size=(len(pts),), dtype=torch.long, device=self.device)
                _pts_list = [pts]
                _bidx_list = [bidx]
                # Sample in empty
                if gidx_empty.numel() > 0:
                    pts, _vidx = sample_pts_in_voxels(gidx_empty, n_in_empty, self.resolution, dtype=self.dtype)
                    _pts_list.append(pts)
                    _bidx_list.append(bidx_empty[_vidx])
                # Sample in non-empty
                if gidx_nonempty.numel() > 0:
                    pts, _vidx = sample_pts_in_voxels(gidx_nonempty, n_in_nonempty, self.resolution, dtype=self.dtype)
                    _pts_list.append(pts)
                    _bidx_list.append(bidx_nonempty[_vidx])
                # Cat all three and query val
                pts = torch.cat(_pts_list, dim=0)
                bidx = torch.cat(_bidx_list, dim=0)
                val = val_query_fn_normalized_x_bi(pts, bidx=bidx)
                pts_list.append(pts)
                bidx_list.append(bidx)
                vals_list.append(val)
            pts_list, bidx_list, vals_list = torch.cat(pts_list, 0), torch.cat(bidx_list, 0), torch.cat(vals_list, 0)
        
        if within_bi is not None:
            # From local bidx to global bidx
            bidx_list = within_bi[bidx_list]
        self._step_update_occ(pts_list, bidx_list, vals_list)

    @torch.no_grad()
    def _step_update_occ(
        self, pts: torch.Tensor, bidx: torch.LongTensor = None, val: torch.Tensor = ...):
        
        if self.should_collect_samples:
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
                gidx = torch.cat([gidx.view(-1,self.NUM_DIM), gidx_pcl], dim=0)
                occ_val = torch.cat([occ_val.view(-1), occ_val_pcl], dim=0)
        else:
            # Batched
            pts, occ_val = pts.flatten(1, -2), self.occ_val_fn(val).flatten(1, -1)
            gidx = ((pts/2. + 0.5) * self.resolution).long().clamp(self.resolution.new_tensor([0]), self.resolution-1)
            if bidx_pcl is not None:
                # should_collect_samples, change batched mode to non-batched mode by calculating an auxillary bidx
                bidx = torch.arange(self.num_batches, device=self.device).view(-1,1).expand(-1, pts.shape[1]).reshape(-1)
                bidx = torch.cat([bidx, bidx_pcl], dim=0)
                gidx = torch.cat([gidx.view(-1,self.NUM_DIM), gidx_pcl], dim=0)
                occ_val = torch.cat([occ_val.view(-1), occ_val_pcl], dim=0)

        update_batched_occ_val_grid_idx_(self.occ_val_grid, bidx, gidx, occ_val, ema_decay=self.ema_decay)
        self.occ_grid = binarize(self.occ_val_grid, self.occ_thre, self.occ_thre_consider_mean)

    """ (Optional) Collect samples from ray march or rendering process """
    @torch.no_grad()
    def collect_samples(self, pts: torch.Tensor, bidx: torch.LongTensor = None, val: torch.Tensor = ...):
        """
        NOTE: `collect_samples` should be invoked like a forward-hook function.
        """
        if self.training and self.should_collect_samples:
            self._collect_samples(pts.flatten(0,-2), bidx.flatten(), val.flatten(), **self.update_from_samples_cfg)

    @torch.no_grad()
    def _collect_samples(self, pts: torch.Tensor, bidx: torch.LongTensor = None, val: torch.Tensor = ...):
        # NOTE: No ema decay. We are gathering all possible samples into `_occ_val_grid_pcl`. Only decay when udpate.
        update_batched_occ_val_grid_(self._occ_val_grid_pcl, pts, bidx, self.occ_val_fn(val), ema_decay=1.0)

    """ Sampling or querying from the occ grid """
    @torch.no_grad()
    def sample_pts_in_occupied(self, num_pts: int, within_bi: torch.LongTensor = None) -> Tuple[torch.Tensor, torch.LongTensor]:
        if within_bi is not None:
            idx_nonempty = self.occ_grid[within_bi].nonzero().long()
            assert idx_nonempty.numel() > 0, err_msg_empty_occ
            local_bidx_nonempty, gidx_nonempty = idx_nonempty[..., 0], idx_nonempty[..., 1:]
            pts, vidx = sample_pts_in_voxels(gidx_nonempty, num_pts, self.resolution, dtype=self.dtype)
            bidx = local_bidx_nonempty[vidx]
        else:
            idx_nonempty = self.occ_grid.nonzero().long()
            assert idx_nonempty.numel() > 0, err_msg_empty_occ
            bidx_nonempty, gidx_nonempty = idx_nonempty[..., 0], idx_nonempty[..., 1:]
            pts, vidx = sample_pts_in_voxels(gidx_nonempty, num_pts, self.resolution, dtype=self.dtype)
            bidx = bidx_nonempty[vidx]
        return pts, bidx

    @torch.no_grad()
    def query(self, pts: torch.Tensor, bidx: torch.LongTensor) -> torch.BoolTensor:
        """_summary_

        Args:
            pts (torch.Tensor): Expected to be in range [-1,1]
            bidx (torch.LongTensor): The global batch idx (in range [0,self.num_batches-1])
        Returns:
            torch.BoolTensor: _description_
        """
        # Expect pts in range [-1,1]
        gidx = ((pts/2.+0.5) * self.resolution).long().clamp(self.resolution.new_tensor([0]), self.resolution-1)
        return self.occ_grid[(bidx,)+tuple(gidx.movedim(-1,0))]

    def extra_repr(self) -> str:
        occ_grid_shape_str = '[' + ','.join([str(s) for s in self.occ_grid.shape]) + ']'
        return f"occ_grid={occ_grid_shape_str}"

__all__ = [
    'OccGridAccelBatchedDynamic_Base', 
    'OccGridAccelBatchedDynamic_Ema', 
    'OccGridAccelBatchedDynamic_Getter'
]

import math
import numpy as np
from tqdm import tqdm
from numbers import Number
from typing import Dict, List, Literal, Tuple, Union

import torch
import torch.nn as nn

from nr3d_lib.logger import Logger
from nr3d_lib.utils import check_to_torch, tensor_statistics, torch_consecutive_nearest1d
from nr3d_lib.models.spatial import BatchedDynamicSpace
from nr3d_lib.models.accelerations.occgrid import OccGridEmaBatched, OccGridGetter, \
    sample_pts_in_voxels, err_msg_empty_occ
from nr3d_lib.graphics.raymarch import RaymarchRetBatchedDynamic
from nr3d_lib.graphics.raymarch.occgrid_raymarch import occgrid_raymarch_batched

class OccGridAccelBatchedDynamic_Base(nn.Module):
    @torch.no_grad()
    def cur_batch__sample_pts_in_occupied(self, num_pts: int) -> Tuple[torch.Tensor, torch.LongTensor, torch.Tensor]:
        """ Uniformly sample points in the occupied region of *current batch*

        Args:
            num_pts (int): _description_

        Returns:
            Tuple[torch.Tensor, torch.LongTensor, torch.Tensor]: x, bidx, ts
        """
        assert self.tmp_flat_occ_grid is not None, \
            "Please call set_condition() first before sample_pts_in_occupied()"
        idx_nonempty = self.tmp_flat_occ_grid.nonzero().long()
        assert idx_nonempty.numel() > 0, err_msg_empty_occ
        tmp_flat_bidx_nonempty, gidx_nonempty = idx_nonempty[..., 0], idx_nonempty[..., 1:]
        pts, vidx = sample_pts_in_voxels(gidx_nonempty, num_pts, resolution=self.resolution, dtype=self.dtype)
        tmp_flat_bidx = tmp_flat_bidx_nonempty[vidx]
        
        bidx = torch.div(tmp_flat_bidx, self.num_frames, rounding_mode='floor').long()
        # global_bidx = self.ins_inds_per_batch[bidx]
        fidx = tmp_flat_bidx - bidx * self.num_frames
        ts = self.ts_keyframes[fidx]
        return pts, bidx, ts

    @torch.no_grad()
    def cur_batch__query_occupancy(
        self, pts: torch.Tensor, bidx: torch.LongTensor, ts: torch.Tensor
        ) -> torch.BoolTensor:
        """_summary_

        Args:
            pts (torch.Tensor): Expected to be in range [-1,1]
            bidx (torch.LongTensor): The batch indices of current batch

        Returns:
            torch.BoolTensor: _description_
        """
        assert self.tmp_flat_occ_grid is not None, \
            "Please call set_condition() first before query_occupancy()"
        gidx = ((pts/2.+0.5) * self.resolution).long().clamp(self.resolution.new_tensor([0]), self.resolution-1)
        fidx = torch_consecutive_nearest1d(self.ts_keyframes, ts)[0]
        flat_bidx = bidx * self.num_frames + fidx
        return self.tmp_flat_occ_grid[(flat_bidx,)+tuple(gidx.movedim(-1,0))]
    
    def cur_batch__ray_march(
        self, 
        rays_o: torch.Tensor, rays_d: torch.Tensor, rays_bidx: torch.Tensor, rays_ts: torch.Tensor, # per ray
        *, near=None, far=None,  perturb=False, 
        step_size: float = 1e-3, max_step_size: float = 1e10, dt_gamma: float = 0.0, max_steps: int = 512
        ) -> RaymarchRetBatchedDynamic:
        assert self.tmp_flat_occ_grid is not None, \
            "Please call set_condition() first before ray_march()"
        # Get the nearest frame ind for each ray
        rays_fidx = torch_consecutive_nearest1d(self.ts_keyframes, rays_ts)[0]
        # Form tmp holistic batch inds for each ray
        rays_flat_bi = rays_bidx * self.num_frames + rays_fidx
        
        # March multi-ins multi-frame occ grid by viewing it as multi-flat versions
        ret = occgrid_raymarch_batched(
            self.tmp_flat_occ_grid, 
            rays_o, rays_d, rays_flat_bi, near, far, 
            perturb=perturb, step_size=step_size, max_step_size=max_step_size, 
            dt_gamma=dt_gamma, max_steps=max_steps
        )
        
        # Re-assemble the returns
        if ret.bidx is not None:
            ret_bi = torch.div(ret.bidx, self.num_frames, rounding_mode='floor').long()
            ret_fi = ret.bidx - ret_bi * self.num_frames
            ret_ts = self.ts_keyframes[ret_fi]
        else:
            ret_bi, ret_fi, ret_ts = None, None, None
        ret = RaymarchRetBatchedDynamic(
            num_hit_rays=ret.num_hit_rays, 
            ridx_hit=ret.ridx_hit, 
            samples=ret.samples, 
            depth_samples=ret.depth_samples, 
            deltas=ret.deltas, 
            ridx=ret.ridx, 
            pack_infos=ret.pack_infos, 
            bidx=ret_bi, 
            ts=ret_ts, 
            gidx=ret.gidx, 
            gidx_pack_infos=ret.gidx_pack_infos
        )
        return ret

class OccGridAccelBatchedDynamic_Ema(OccGridAccelBatchedDynamic_Base):
    def __init__(
        self, 
        space: BatchedDynamicSpace, 
        num_batches: int, 
        ts_keyframes: torch.Tensor, 
        resolution: Union[int, List[int], torch.Tensor] = None, 
        dtype=torch.float, device=None, 
        **occ_kwargs) -> None:
        super().__init__()

        #------- Valid representing space 
        assert isinstance(space, BatchedDynamicSpace), f"{self.__class__.__name__} expects space of BatchedDynamicSpace"
        self.space: BatchedDynamicSpace = space
        self.dtype = dtype
        
        #------- Batched / Temporal information
        self.num_frames = len(ts_keyframes)
        self.num_batches_real = num_batches
        self.register_buffer('ts_keyframes', ts_keyframes, persistent=True)

        #------- Occupancy information
        self.occ = OccGridEmaBatched(
            resolution=resolution, **occ_kwargs, 
            num_batches=self.num_batches_real * self.num_frames, 
            dtype=self.dtype, device=device
        )
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
        # ts_per_batch: torch.Tensor, 
        val_query_fn_normalized_x_bi_ts = None):

        assert ins_inds_per_batch is not None, f"`ins_inds_per_batch` is required for {type(self)}"
        self.batch_size = batch_size
        self.ins_inds_per_batch = ins_inds_per_batch
        tmp_flat_bidx = torch.arange(self.num_frames, device=self.device, dtype=torch.long)
        # [batch_size, 1] + [1, num_frames] -> [batch_size, num_frames] -> [batch_size * num_frames]
        tmp_flat_bidx = self.ins_inds_per_batch.unsqueeze(1) * self.num_frames + tmp_flat_bidx.unsqueeze(0)
        self.tmp_flat_bidx = tmp_flat_bidx.flatten()
        
        occ_grid_unflat = self.get_occ_grid().unflatten(0, (self.num_batches_real, self.num_frames))
        self.tmp_flat_occ_grid = occ_grid_unflat[ins_inds_per_batch, :].flatten(0,1).contiguous()
    
    def clean_condition(self):
        self.batch_size = None
        self.ins_inds_per_batch = None
        self.tmp_flat_bidx = None
        self.tmp_flat_occ_grid = None

    @torch.no_grad()
    def init(self, val_query_fn_normalized_x_bi_ts, logger: Logger = None):
        assert (self.batch_size is not None) and (self.batch_size == self.num_batches_real), \
            "Before init(), should set_condition() on all batches (i.e. all latents)."
        def val_query_fn_converted(x, *, bidx=...):
            # NOTE: `bidx` is the bidx of `self.tmp_flat_bidx`
            real_bidx = torch.div(bidx, self.num_frames, rounding_mode='floor').long()
            fidx = bidx - real_bidx * self.num_frames
            # global_bidx = bidx
            ts = self.ts_keyframes[fidx]
            return val_query_fn_normalized_x_bi_ts(x, bidx=real_bidx, ts=ts)
        updated = self.occ.init(val_query_fn_converted, logger=logger)
        if updated:
            pass

    @torch.no_grad()
    def cur_batch__step(self, cur_it: int, val_query_fn_normalized_x_bi_ts, logger: Logger = None):
        assert self.ins_inds_per_batch is not None, \
            "Please call set_condition() first before step()"
        def val_query_fn_converted(x, *, bidx=...):
            # NOTE: `bidx` is the bidx of `self.tmp_flat_bidx`
            real_bidx = torch.div(bidx, self.num_frames, rounding_mode='floor').long()
            fidx = bidx - real_bidx * self.num_frames
            # global_bidx = self.ins_inds_per_batch[bidx]
            ts = self.ts_keyframes[fidx]
            return val_query_fn_normalized_x_bi_ts(x, bidx=real_bidx, ts=ts)
        updated = self.occ.step(cur_it, val_query_fn_converted, within_bi=self.tmp_flat_bidx, logger=logger)
        if updated:
            pass

    @torch.no_grad()
    def cur_batch__collect_samples(
        self, pts: torch.Tensor, bidx: torch.LongTensor, ts: torch.Tensor, val: torch.Tensor, 
        normalized=True):
        """
        NOTE: Should be invoked like a forward-hook function.
        Args:
            pts (torch.Tensor): _description_
            bidx (torch.LongTensor): _description_
            ts (torch.Tensor): _description_
            val (torch.Tensor): _description_
            normalized (bool, optional): _description_. Defaults to True.
        """
        if self.training:
            assert self.ins_inds_per_batch is not None, \
                "Please call set_condition() first before collect_samples()"
            if not normalized:
                pts = self.space.cur_batch__normalize_coords(pts, bidx)
            global_bidx = self.ins_inds_per_batch[bidx]
            fidx = torch_consecutive_nearest1d(self.ts_keyframes, ts)[0]
            global_flat_bidx = global_bidx * self.num_frames + fidx
            self.occ.collect_samples(pts, global_flat_bidx, val)

    """ DEBUG Functionalities """
    @torch.no_grad()
    def debug_stats(self) -> Dict[str, Number]:
        occ_grid = self.get_occ_grid()
        #---- 530 us for [112, 95, 8, 7, 16]
        # [num_batches, num_frames, NX, NY, NZ]
        unflat_occ_grid = occ_grid.data.unflatten(0, (self.num_batches_real, self.num_frames))
        
        num_occupied_per_ins_per_frame = unflat_occ_grid.sum(dim=tuple(range(2,self.NUM_DIM+2,1)))
        num_occupied_per_ins = num_occupied_per_ins_per_frame.sum(dim=1)
        num_occupied = num_occupied_per_ins.sum().item()
        
        frac_occupied = num_occupied / occ_grid.numel()
        num_empty_ins = (num_occupied_per_ins == 0).sum().item()
        num_occupied_per_nonempty_ins = num_occupied_per_ins[num_occupied_per_ins != 0]
        frac_occupied_per_nonempty_ins = num_occupied_per_nonempty_ins / math.prod(unflat_occ_grid.shape[1:])
        return {
            'num_occ': num_occupied, 
            'frac_occ': frac_occupied, 
            'num_empty_ins': num_empty_ins, 
            **tensor_statistics(num_occupied_per_nonempty_ins, 'per_ins.nonempty.num_occupied', metrics=['mean', 'min', 'max', 'std']), 
            **tensor_statistics(frac_occupied_per_nonempty_ins, 'per_ins.nonempty.frac_occupied', metrics=['mean', 'min', 'max', 'std']), 
        }

    @torch.no_grad()
    def debug_vis(self, draw=True):
        # NOTE: Primary drawing function
        # from vedo.applications import IsosurfaceBrowser
        from vedo import Volume, show
        # [num_batches, num_frames, NX, NY, NZ]
        unflat_val_grid_tensor = self.occ.occ_val_grid.data.unflatten(0, (self.num_batches_real, self.num_frames))
        unflat_val_grid = unflat_val_grid_tensor[self.ins_inds_per_batch[0:20], ::3].cpu().numpy()
        
        actors = []
        val_thre = self.occ.occ_thre
        spacing = ((self.space.aabb[1]-self.space.aabb[0]) / self.resolution.float().mean()).tolist()
        # spacing = ((self.space.aabb[1]-self.space.aabb[0]) / self.resolution).tolist()
        radius3d = self.space.radius3d_original.tolist()
        origin0 = self.space.aabb[0].tolist()
        for bidx, val_grid_all_frames in enumerate(tqdm(unflat_val_grid, 'Processing batches')):
            if not (val_grid_all_frames > val_thre).any():
                continue
            for fidx, val_grid in enumerate(val_grid_all_frames):
                if not (val_grid > val_thre).any():
                    continue
                # NOTE: Different bidx on x-axis, different fidx on y-axis
                origin = [origin0[0] + bidx * radius3d[0] * 3, 
                          origin0[1] + fidx * radius3d[1] * 3, 
                          origin0[2]]
                vol = Volume(val_grid, c=['white','b','g','r'], mapper='gpu', origin=origin, spacing=spacing)
                vox = vol.legosurface(vmin=val_thre, vmax=1., boundary=True)
                vox.cmap('GnBu', on='cells', vmin=val_thre, vmax=1.).add_scalarbar()
                actors.append(vox)
        show(*actors, __doc__, axes=1, viewup='z').close()
        
        if draw:
            show(*actors, __doc__, axes=1, viewup='z').close()
        else:
            return actors

class OccGridAccelBatchedDynamic_Getter(OccGridAccelBatchedDynamic_Base):
    def __init__(
        self, 
        space: BatchedDynamicSpace, 
        resolution: Union[int, List[int], torch.Tensor] = None, 
        n_jump_frame: int = 1, 
        dtype=torch.float, device=None, 
        **occ_kwargs) -> None:
        super().__init__()

        #------- Valid representing space 
        assert isinstance(space, BatchedDynamicSpace), f"{self.__class__.__name__} expects space of BatchedDynamicSpace"
        self.space: BatchedDynamicSpace = space
        self.dtype = dtype
        
        #------- Batched / Temporal information
        ts_keyframes = self.space.ts_keyframes.data[::n_jump_frame]
        self.num_frames = len(ts_keyframes)
        ts_keyframes = check_to_torch(ts_keyframes, device=device)
        self.register_buffer('ts_keyframes', ts_keyframes, persistent=True)

        #------- Occupancy information getter
        self.occ_getter = OccGridGetter(**occ_kwargs, resolution=resolution, dtype=self.dtype, device=device)
        self.clean_condition()

    @property
    def device(self) -> torch.device:
        return self.space.device

    @property
    def resolution(self):
        return self.occ_getter.resolution

    def set_condition(
        self, 
        batch_size: int, 
        ins_inds_per_batch: torch.LongTensor, 
        # ts_per_batch: torch.Tensor, 
        val_query_fn_normalized_x_bi_ts = ...):
        
        """
        We need to discuss two scenarios depending on whether ts_per_batch is provided; 
        If it is provided, each batch can be further subdivided on the time dimension; 
        If not provided, the time dimension can only be viewed separately.
        
        Similarly, whether ins_inds_per_batch is provided also needs to be discussed separately.
        
        -> Specifically, it depends on whether rays_bidx and rays_ts can be provided during raymarch; 
            **Among them, rays_ts can be ignored, the main concern here is that ins_inds_per_batch must be provided**
        """
        
        # TODO: Here, decide more flexibly based on whether ins_inds_per_batch and ts_per_batch are provided
        #       We can assume that ins_inds_per_batch is always provided, ts_per_batch is not necessarily provided / according to the uniform setting, ts_per_batch is not provided by default
        #       Or directly use rays_ts to get_occ again during march
        def val_query_fn_converted(x, bidx):
            real_bidx = torch.div(bidx, self.num_frames, rounding_mode='floor').long()
            fidx = bidx - real_bidx * self.num_frames
            ts = self.ts_keyframes[fidx]
            return val_query_fn_normalized_x_bi_ts(x, bidx=real_bidx, ts=ts)
        
        self.tmp_flat_occ_grid = self.occ_getter.occ_grid_from_net_batched_v2(
            batch_size * self.num_frames, val_query_fn_converted)
        
        # self.cur_batch_represents: Literal['']

    def clean_condition(self):
        self.batch_size = None
        self.tmp_flat_occ_grid = None

    @torch.no_grad()
    def init(self, val_query_fn_normalized_x_bi_ts, logger: Logger = None):
        pass

    @torch.no_grad()
    def cur_batch__step(self, cur_it: int, val_query_fn_normalized_x_bi_ts, logger: Logger = None):
        pass

    @torch.no_grad()
    def cur_batch__collect_samples(
        self, pts: torch.Tensor, bidx: torch.LongTensor, ts: torch.Tensor, val: torch.Tensor, 
        normalized=True):
        pass

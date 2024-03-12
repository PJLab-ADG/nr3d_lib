"""
@file   batched.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Handles one batch of block space
"""

__all__ = [
    'BatchedDynamicSpace', 
    'BatchedDynamicSpaceNormalizeTs', 
]

import numpy as np
from typing import List, Tuple, Union

import torch
import torch.nn as nn

from nr3d_lib.utils import check_to_torch

from .batched import BatchedBlockSpace

class BatchedDynamicSpace(BatchedBlockSpace):
    """ Current batched operations """
    def cur_batch__normalize_ts(self, ts: torch.Tensor, bidx: torch.LongTensor = None):
        return ts

    def cur_batch__unnormalize_ts(self, ts: torch.Tensor, bidx: torch.LongTensor = None):
        return ts

    def cur_batch__sample_pts_uniform(
        self, batch_size: int, num_pts_per_batch: int
        ) -> Tuple[torch.Tensor, torch.LongTensor, torch.Tensor]:
        x, bidx = super().cur_batch__sample_pts_uniform(batch_size, num_pts_per_batch)
        ts = torch.empty([batch_size,num_pts_per_batch], dtype=torch.float, device=self.device).uniform_(-1,1)
        return x, bidx, ts

    def cur_batch__ray_test(
        self, rays_o: torch.Tensor, rays_d: torch.Tensor, near=None, far=None, 
        return_rays=True, normalized=False, compact_batch=False, **extra_ray_data):
        ret = super().cur_batch__ray_test(rays_o, rays_d, near, far, return_rays, normalized, compact_batch, **extra_ray_data)
        return ret

class BatchedDynamicSpaceNormalizeTs(BatchedBlockSpace):
    """
    NOTE: This is only required if the scene's timestamps are not already normalized when _get_scenario()
    """
    def __init__(
        self, *args,
        t_scale: float, # A common universal dt scaling for all instances
        all_ts_offset: Union[np.ndarray, torch.Tensor], # ts offset for each instances
        dtype=torch.float, device=None, 
        **kwargs) -> None:
        super().__init__(*args, **kwargs, dtype=dtype, device=device)
        
        ts_scale = check_to_torch(t_scale, dtype=dtype, device=device)
        all_ts_offset = check_to_torch(all_ts_offset, dtype=dtype, device=device)

        self.register_buffer('all_ts_offset', all_ts_offset, persistent=True)
        self.register_buffer('ts_scale', ts_scale, persistent=True)
    
    def set_condition(self, ins_inds_per_batch: torch.LongTensor = None, ts_offset_per_batch: torch.Tensor = None):
        if ts_offset_per_batch is None:
            ts_offset_per_batch = self.all_ts_offset[ins_inds_per_batch]
        self.ts_offset_per_batch = ts_offset_per_batch
    
    def clean_condition(self):
        self.ts_offset_per_batch = None
    
    """ Current batched operations """
    def cur_batch__normalize_ts(self, ts: torch.Tensor, bidx: torch.LongTensor):
        """ From raw to [-1,1] """
        assert self.ts_offset_per_batch is not None, \
            f"Please call set_condition() before cur_batch__normalize_ts() for {type(self)}"
        ts = (ts - self.ts_offset_per_batch[bidx]) * self.ts_scale
        return ts
    
    def cur_batch__unnormalize_ts(self, ts: torch.Tensor, bidx: torch.LongTensor):
        """ From [-1,1] to raw """
        assert self.ts_offset_per_batch is not None, \
            f"Please call set_condition() before cur_batch__unnormalize_ts() for {type(self)}"
        ts = ts / self.ts_scale + self.ts_offset_per_batch[bidx]
        return ts

    def cur_batch__sample_pts_uniform(
        self, batch_size: int, num_pts_per_batch: int
        ) -> Tuple[torch.Tensor, torch.LongTensor, torch.Tensor]:
        x, bidx = super().cur_batch__sample_pts_uniform(batch_size, num_pts_per_batch)
        ts = torch.empty([batch_size,num_pts_per_batch], dtype=torch.float, device=self.device).uniform_(-1,1)
        return x, bidx, ts

    def cur_batch__ray_test(
        self, rays_o: torch.Tensor, rays_d: torch.Tensor, near=None, far=None, 
        return_rays=True, normalized=False, compact_batch=False, **extra_ray_data):
        ret = super().cur_batch__ray_test(rays_o, rays_d, near, far, return_rays, normalized, compact_batch, **extra_ray_data)
        ret['rays_ts'] = self.cur_batch__normalize_ts(ret['rays_ts'], ret['rays_bidx'])
        return ret

    @staticmethod
    def normalize_all_ts_keyframes(all_ts_keyframes: Union[List[np.ndarray], List[torch.Tensor]]):
        all_ts_keyframes = [kf.tolist() for kf in all_ts_keyframes]
        
        # Quick set for waymo: 2.0 / 200 frames = 0.01
        if t_scale is None:
            all_len = [len(kf) for kf in all_ts_keyframes]
            all_dt = [(kf[-1]-kf[0])/(len(kf)-1) for kf in all_ts_keyframes]
            dt = np.array(all_dt).mean()
            # We want to spread the keyframes which has the max extent to [-1,1] uniform temporal space
            normalized_dt = 2. / (max(all_len)-1)
            t_scale = normalized_dt / dt
        
        if all_ts_offset is None:
            # Usually it is the ts of the middle frame
            all_mid_ts = [kf[len(kf)//2] for kf in all_ts_keyframes]
            all_ts_offset = all_mid_ts
        
        return all_ts_offset, t_scale

# class BatchedDynamicSpace(BatchedBlockSpace):
#     def __init__(
#         self, *args,
#         ts_keyframes: Union[np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]] = ..., 
#         **kwargs) -> None:
#         super().__init__(*args, **kwargs)
        
#         if isinstance(ts_keyframes, (list, tuple)):
#             # Multiple ts-keyframes with potential different offset
#             # TODO: This must be used in conjunction with id indexing, just like the batched auto-decoder needs to provide ins_inds, this also needs to provide ins_inds.
#             pass
#         else:
#             pass
        
#         ts_keyframes = check_to_torch(ts_keyframes, dtype=self.dtype, device=self.device)
#         self.register_buffer('ts_keyframes', ts_keyframes, persistent=True)
    
#     def normalize_ts(self, ts: torch.Tensor, ins_inds: torch.LongTensor = None):
#         return ts

#     def sample_pts_uniform(self, batch_size: int, num_pts_per_batch: int) -> Tuple[torch.Tensor, torch.LongTensor, torch.Tensor]:
#         x, bidx = super().sample_pts_uniform(batch_size, num_pts_per_batch)
#         ts_w = torch.rand([batch_size, num_pts_per_batch], dtype=torch.float, device=self.device)
#         ts_i = torch.randint(len(self.ts_keyframes)-1, [batch_size, num_pts_per_batch], dtype=torch.long, device=self.device)
#         ts = torch.lerp(self.ts_keyframes[ts_i], self.ts_keyframes[ts_i+1], ts_w)
#         return x, bidx, ts
    
#     def extra_repr(self) -> str:
#         extra_repr = super().extra_repr()
#         extra_repr += \
#             f", num_frames={len(self.ts_keyframes)}"\
#             f", ts_from={self.ts_keyframes[0].item():.3f}"\
#             f", ts_to={self.ts_keyframes[-1].item():.3f}"
#         return extra_repr


"""
@file   aabb.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Handles a single block AABB (Axis-Aligned-Bounding-Box) space
"""

__all__ = [
    'AABBDynamicSpace', 
]

import numpy as np
from typing import Literal, Tuple, Union

import torch
import torch.nn as nn

from nr3d_lib.utils import check_to_torch

from .aabb import AABBSpace

class AABBDynamicSpace(AABBSpace):
    # def __init__(
    #     self, *args, 
    #     ts_keyframes: Union[np.ndarray, torch.Tensor] = ..., 
    #     **kwargs) -> None:
    #     super().__init__(*args, **kwargs)
    #     ts_keyframes = check_to_torch(ts_keyframes, dtype=self.dtype, device=self.device)
    #     self.register_buffer('ts_keyframes', ts_keyframes, persistent=True)
    
    def normalize_ts(self, ts: torch.Tensor):
        return ts
    
    def unnormalize_ts(self, ts: torch.Tensor):
        return ts
    
    def sample_pts_uniform(self, num_pts: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = super().sample_pts_uniform(num_pts)
        ts = torch.empty([num_pts,], dtype=torch.float, device=self.device).uniform_(-1,1)
        return x, ts 
    
    # def sample_pts_uniform(self, num_pts: int) -> Tuple[torch.Tensor, torch.Tensor]:
    #     x = super().sample_pts_uniform(num_pts)
    #     ts_w = torch.rand([num_pts], dtype=torch.float, device=self.device)
    #     ts_i = torch.randint(len(self.ts_keyframes)-1, dtype=torch.long, device=self.device)
    #     ts = torch.lerp(self.ts_keyframes[ts_i], self.ts_keyframes[ts_i+1], ts_w)
    #     return x, ts 
    
    # def extra_repr(self) -> str:
    #     extra_repr = super().extra_repr()
    #     extra_repr += \
    #         f", num_frames={len(self.ts_keyframes)}"\
    #         f", ts_from={self.ts_keyframes[0].item():.3f}"\
    #         f", ts_to={self.ts_keyframes[-1].item():.3f}"
    #     return extra_repr

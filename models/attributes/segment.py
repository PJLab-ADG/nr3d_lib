"""
@file   segment.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Attr log sequences related definitions
"""

__all__ = [
    "Valid", 
    "AttrSegment"
]

import functools
import numpy as np
from typing import Union

import torch
import torch.nn as nn

from nr3d_lib.utils import is_scalar

from .attr import *

@AttrBase
class Valid(Attr):
    """
    Mark validness of each frame of scene members
    """
    default = torch.ones([], dtype=torch.bool)
    def value(self):
        if len(self.prefix) == 0:
            return self.tensor.item()
        else:
            return self.tensor

class AttrSegment(object):
    """
    Using `start_frame`, `stop_frame` to mark segments
    """
    def __init__(self, **kwargs):
        self.subattr = kwargs
        self.n_frames = 0
        self.start_frame = None
        self.stop_frame = None
    def is_valid(self, i: Union[slice, int, torch.Tensor, np.ndarray]):
        if isinstance(i, slice):
            if i.start is None:
                i.start = 0
            if i.stop is None:
                raise "Can not decide validness if given stop_frame is None."
            if (i.stop <= self.start_frame) or (i.start >= self.stop_frame):
                return False
            else:
                return True
        elif is_scalar(i):
            return i >= self.start_frame and i < self.stop_frame
        elif isinstance(i, (torch.Tensor, np.ndarray)):
            return ((i >= self.start_frame) & (i < self.stop_frame)).all()
        else:
            raise ValueError(f"Invalid input type(i)={type(i)}")
    def __len__(self):
        return self.n_frames
    def __getitem__(self, index):
        # TODO: Use float timestamp to do interpolation or nearest neighbor search
        return {k: v[index] for k,v in self.subattr.items()}
    def __repr__(self) -> str:
        return f"{type(self).__name__}(" +\
            ",\n".join(
                [f"start_frame={self.start_frame}", f"n_frames={self.n_frames}"] +\
                    [f"{k}={repr(v)}" for k, v in self.subattr.items()]
                    ) + "\n)"
    @functools.wraps(nn.Module.to)
    def to(self, *args, **kwargs):
        self.subattr = {k:v.to(*args, **kwargs) for k,v in self.subattr.items()}
    @functools.wraps(nn.Module._apply)
    def _apply(self, fn):
        self.subattr = {k:v._apply(fn) for k,v in self.subattr.items()}
        return self
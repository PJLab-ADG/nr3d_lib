
import math
import numpy as np
from numbers import Number
from typing import List, Literal, Union

import torch
import torch.nn as nn

from nr3d_lib.utils import check_to_torch

class SeqEmbedder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, ts: torch.Tensor):
        pass

class UniformSeqEmbedder(SeqEmbedder):
    def __init__(self, start: Number, stop: Number) -> None:
        super().__init__()
        self.start = start
        self.stop = stop

class SinusoidalSeqEmbedder(SeqEmbedder):
    def __init__(
        self, 
        sin_or_cos: Literal['sin', 'cos', 'sincos'] = 'cos', 
        amp: float = 1.0, offset: float = 0.0, 
        periods: Union[Number, List, np.ndarray] = 1, 
        shifts: Union[Number, List, np.ndarray] = 0, 
        learnable=False, dtype=torch.float, device=None
        ) -> None:
        super().__init__()
        
        bands = 1 if isinstance(periods, Number) else len(periods)
        periods = check_to_torch(periods, dtype=dtype, device=device).view([bands])
        shifts = check_to_torch(shifts, dtype=dtype, device=device).view([bands])
        
        dim_per_band = 2 if sin_or_cos == 'sincos' else 1
        
        self.embedding_dim = dim_per_band * bands
        self.sin_or_cos = sin_or_cos

        freqs = (math.pi * 2.) / periods

        self.amp = amp
        self.offset = offset
        if learnable:
            self.register_parameter('freqs', nn.Parameter(freqs, requires_grad=True))
            self.register_parameter('shifts', nn.Parameter(shifts, requires_grad=True))
        else:
            self.register_buffer('freqs', freqs, persistent=True)
            self.register_buffer('shifts', shifts, persistent=True)
    
    @property
    def device(self) -> torch.device:
        return self.freqs.device
    
    @property
    def dtype(self):
        return self.freqs.dtype
    
    def forward(self, ts: torch.Tensor):
        angles = (ts.unsqueeze(-1) - self.shifts) * self.freqs
        if self.sin_or_cos == 'sincos':
            angles = torch.stack([angles, angles + math.pi/2.], dim=-1).flatten(-2, -1)
        elif self.sin_or_cos == 'cos':
            angles = angles + math.pi/2.
        elif self.sin_or_cos == 'sin':
            pass
        else:
            raise RuntimeError(f"Invalid self.sin_or_cos={self.sin_or_cos}")
        out = self.amp * torch.sin(angles) + self.offset
        return out
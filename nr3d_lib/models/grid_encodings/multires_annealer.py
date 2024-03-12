"""
@file   multires_annealer.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Common feature annealers for multi-resolution encoded features
"""

__all__ = [
    'MultiresAnnealer'
]

import numpy as np
from typing import Literal, Union, List, Tuple

import torch
import torch.nn as nn

from nr3d_lib.utils import check_to_torch

class MultiresAnnealer(nn.Module):
    def __init__(
        self, 
        level_n_feats: List[int], 
        type: Literal['hardmask', 'cosine'], 
        stop_it: int, start_it: int = 0, update_every: int=1, start_level: int = 0, 
        dtype=torch.float, device=None # NOTE: dtype should be the same with encoding's output dtype
        ) -> None:
        super().__init__()
        
        level_n_feats = check_to_torch(level_n_feats, dtype=dtype, device=device)
        self.num_levels = len(level_n_feats)
        self.register_buffer('level_n_feats', level_n_feats, persistent=False)
        self.register_buffer('window', torch.ones([level_n_feats.numel()], dtype=dtype, device=device), persistent=False)
        self.register_buffer('level_arange', torch.arange(self.num_levels, dtype=dtype, device=device), persistent=False)
        
        self.start_it = int(start_it)
        self.stop_it = int(stop_it)
        self.update_every = int(update_every)
        self.it = self.stop_it # At anneal stop state by default.
        self.total_stages = (self.stop_it - self.start_it) // self.update_every
        # NOTE: At least -1 (no levels used;)
        #       max_level=0 means the 1st level will be used; max_level=-1 means no level used.
        self.start_level = max(min(int(start_level), self.num_levels-1), -1)
        if type == 'hardmask':
            self.window_fn = self._window_fn_hardmask
        elif type == 'cosine':
            self.window_fn = self._window_fn_cosine
        else:
            raise RuntimeError(f'Invalid anneal_type={type}')

    def set_iter(self, it: int):
        self.it = it

    def forward(self, it:int = None) -> Tuple[int, Union[None, torch.Tensor]]:
        it = self.it if it is None else it
        cur_stage = (it - self.start_it) // self.update_every
        alpha = min(1.0, max(0.0, cur_stage / self.total_stages))
        max_level, window = self.window_fn(alpha)
        return max_level, window

    def _window_fn_hardmask(self, alpha: float = 1.0):
        # From self.start_level (at least=-1) to self.num_levels-1
        length = (self.num_levels-1) - self.start_level
        max_level = self.start_level + min(int(alpha * length), length)
        return max_level, None
    
    def _window_fn_cosine(self, alpha: float = 1.0):
        # NOTE: when alpha=0, only the minimum level is active
        length = (self.num_levels-1) - self.start_level
        raw = self.start_level + alpha * length - self.level_arange + 1
        window = 0.5 * (1 + torch.cos(np.pi * torch.clip(raw, 0.0, 1.0) + np.pi))
        window = window.repeat_interleave(self.level_n_feats)
        # From self.start_level (at least=-1) to self.num_levels-1
        max_level = (raw > 0).sum().item()-1
        return max_level, window

if __name__ == "__main__":
    def test_multires_annealer(device=torch.device('cuda')):
        import matplotlib.pyplot as plt
        an1 = MultiresAnnealer([2, 4, 2, 3, 5], 'hardmask', start_it=10, stop_it=433, start_level=1, device=device)
        an2 = MultiresAnnealer([2, 4, 2, 3, 5], 'hardmask', start_it=10, stop_it=334, start_level=0, device=device)
        an3 = MultiresAnnealer([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'cosine', start_it=100, stop_it=443, start_level=2, device=device)
        
        lst_maxlvl_1 = []
        lst_maxlvl_2 = []
        lst_maxlvl_3 = []
        windows_3 = []
        iters = range(0, 1000)
        for it in iters:
            max_level, _ = an1(it)
            lst_maxlvl_1.append(max_level)
            
            max_level, _ = an2(it)
            lst_maxlvl_2.append(max_level)

            max_level, window = an3(it)
            lst_maxlvl_3.append(max_level)
            windows_3.append(window.data.cpu().numpy())
        windows_3 = np.stack(windows_3, axis=1)
        
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.plot(iters, lst_maxlvl_1, label='an1')
        plt.plot(iters, lst_maxlvl_2, label='an2')
        plt.plot(iters, lst_maxlvl_3, label='an3')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        for i, w in enumerate(windows_3):
            plt.plot(iters, w, label=f"lv.{i}")
        plt.legend()
        plt.show()
    test_multires_annealer()
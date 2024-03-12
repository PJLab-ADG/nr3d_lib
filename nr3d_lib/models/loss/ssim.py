"""
@file   ssim.py
@author Qiusheng Huang, Shanghai AI Lab
@brief  ssim loss

The source code is adapted from:
https://github.com/Po-Hsun-Su/pytorch-ssim
Reference:
[1] Wang Z, Bovik A C, Sheikh H R, et al.
    Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing
"""

__all__ = [
    'ssim', 
    'ssim_module'
]

import numpy as np
from math import exp
from numbers import Number

import torch
import torch.nn as nn
import torch.nn.functional as F

def create_gaussian(window_size: int, sigma: Number, dtype=torch.float) -> torch.Tensor:
    gauss = torch.tensor([exp(-(ws - window_size//2)**2 / float(2*sigma**2)) for ws in range(window_size)], dtype=dtype)
    return gauss / gauss.sum()

def create_window(window_size: int, channel: int, dtype=torch.float) -> torch.Tensor:
    window_1d = create_gaussian(window_size, 1.5, dtype=dtype).unsqueeze(1)
    window_2d = window_1d.mm(window_1d.t())[None, None, ...].expand(channel, 1, window_size, window_size).contiguous()
    return window_2d

def compute_ssim(
    img1: torch.Tensor, img2: torch.Tensor, window: torch.Tensor, 
    window_size: int, channel: int, size_average = True, stride: int=None, 
    C1: float = 0.01**2, C2: float = 0.03**2) -> torch.Tensor:
    
    mu1 = F.conv2d(img1, window, padding=(window_size-1)//2, groups=channel, stride=stride)
    mu2 = F.conv2d(img2, window, padding=(window_size-1)//2, groups=channel, stride=stride)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=(window_size-1)//2, groups=channel, stride=stride) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=(window_size-1)//2, groups=channel, stride=stride) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=(window_size-1)//2, groups=channel, stride=stride) - mu1_mu2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

#---- Function to directly compute SSIM
def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel).to(img1)
    return compute_ssim(img1, img2, window, window_size, channel, size_average)

#---- Module storage for SSIM
class ssim_module(nn.Module):
    def __init__(
        self, 
        window_size: int = 3, 
        size_average = True, 
        stride: int = 3, 
        channel: int = 3, 
        dtype=torch.float, device=None):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.stride = stride
        window = create_window(window_size, self.channel, dtype=dtype).to(device=device)
        self.register_buffer('window', window, persistent=False)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor):
        """
        img1, img2: torch.Tensor([b,c,h,w])
        """
        (_, channel, _, _) = img1.size()
        assert self.channel == channel, f"Input channel does not match (should be [B, {self.channel}, H, W], but got {list(img1.shape)})"
        return compute_ssim(img1, img2, self.window.to(img1), self.window_size, channel, self.size_average, stride=self.stride)

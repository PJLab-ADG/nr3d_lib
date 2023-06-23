"""
@file   utils.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Utilities for rendering.
"""

import os
import math
import struct
import functools
import numpy as np
from enum import Enum
from typing import Dict, Tuple, Union
from scipy.ndimage.filters import convolve1d

import torch
import torch.nn.functional as F

from nr3d_lib.utils import load_rgb
from nr3d_lib.models.loss.utils import reduce
from nr3d_lib.models.loss.recon import mse_loss
from nr3d_lib.render.pack_ops import interleave_arange_simple

#----------------------------------------------------
#----------------- Image related --------------------
#----------------------------------------------------
def srgb_to_linear(img):
	limit = 0.04045
	return np.where(img > limit, np.power((img + 0.055) / 1.055, 2.4), img / 12.92)

def linear_to_srgb(img):
	limit = 0.0031308
	return np.where(img > limit, 1.055 * (img ** (1.0 / 2.4)) - 0.055, 12.92 * img)

def read_image(file):
	if os.path.splitext(file)[1] == ".bin":
		with open(file, "rb") as f:
			bytes = f.read()
			h, w = struct.unpack("ii", bytes[:8])
			img = np.frombuffer(bytes, dtype=np.float16, count=h*w*4, offset=8).astype(np.float32).reshape([h, w, 4])
	else:
		img = load_rgb(file)
		if img.shape[2] == 4:
			img[...,0:3] = srgb_to_linear(img[...,0:3])
			# Premultiply alpha
			img[...,0:3] *= img[...,3:4]
		else:
			img = srgb_to_linear(img)
	return img

def luminance(x):
	x = np.maximum(0, x)**0.4545454545
	return 0.2126 * x[:,:,0] + 0.7152 * x[:,:,1] + 0.0722 * x[:,:,2]

def SSIM(x, y, mask = None, only_in_mask=False):
    def blur(m):
        k = np.array([0.120078, 0.233881, 0.292082, 0.233881, 0.120078])
        x = convolve1d(m, k, axis=0)
        return convolve1d(x, k, axis=1)
    if isinstance(x, torch.Tensor):
        x = x.data.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.data.cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.data.cpu().numpy()
    if mask is not None:
        mask = mask.reshape(*x.shape[:-1])
        # x = x * mask # x should not be masked !
        y = y * mask[...,  None]
        
    x = luminance(x)
    y = luminance(y)
    mA = blur(x)
    mB = blur(y)
    sA = blur(x*x) - mA**2
    sB = blur(y*y) - mB**2
    sAB = blur(x*y) - mA*mB
    c1 = 0.01**2
    c2 = 0.03**2
    p1 = (2.0*mA*mB + c1)/(mA*mA + mB*mB + c1)
    p2 = (2.0*sAB + c2)/(sA + sB + c2)
    error = p1 * p2
    
    if (mask is not None) and only_in_mask:
        return error[mask].sum() / mask.sum().clip(1e-5)
    else:
        return error.mean()

def PSNR(x: torch.Tensor, y: torch.Tensor, mask: torch.BoolTensor=None, only_in_mask=False):
    """
    Calculates the Peak-signal-to-noise ratio between tensors `x` and `y`.
    """
    
    if (mask is not None):
        mask = mask.view(*x.shape[:-1])
        if only_in_mask:
            mse = ((x - y)**2)[mask].sum() / mask.sum().clip(1e-5)
        else:
            # NOTE: x should not be masked; 
            #       and the convergence of mask will also affect performance here (which is expected)
            mse = ((x - y * mask.unsqueeze(-1)) ** 2).mean() 
            # mse = ((x - y)**2 * mask).mean() # x should not be masked !
    else:
        mse = ((x - y)**2).mean()
    psnr = -10.0 * torch.log10(mse)
    return psnr

#----------------------------------------------------
#----------- Image processing related ---------------
#----------------------------------------------------
def lin2img(tensor: torch.Tensor, H: int, W: int, batched=False, B=None) -> torch.Tensor:
    *_, num_samples, channels = tensor.shape
    assert num_samples == H * W
    if batched:
        if B is None:
            B = tensor.shape[0]
        else:
            tensor = tensor.view([B, num_samples//B, channels])
        return tensor.permute(0, 2, 1).view([B, channels, H, W])
    else:
        return tensor.permute(1, 0).view([channels, H, W])

#----------------------------------------------------
#----------------- Index related --------------------
#----------------------------------------------------
def torch_ravel(sep_inds: torch.Tensor, prefix) -> torch.Tensor:
    # NOTE: the same with np.ravel
    # NOTE: holistic indices
    #       e.g. (3,4,5,6) -> a0*(4*5*6*1) + a1*(5*6*1) + a2*(6*1) + a3*(1)
    device = sep_inds.device
    _pre = list(reversed([*prefix[1:], 1]))
    _pre = torch.tensor(list(reversed(np.cumprod(_pre).tolist())), device=device)
    return torch.sum(sep_inds * _pre[:,None], dim=0)
    
def torch_unravel(holistic_inds: torch.Tensor, prefix) -> torch.Tensor:
    # NOTE: holistic indices
    #       e.g. (3,4,5,6) -> a0*(4*5*6*1) + a1*(5*6*1) + a2*(6*1) + a3*(1)
    device = holistic_inds.device
    _pre = list(reversed([*prefix[1:], 1]))
    _pre = torch.tensor(list(reversed(np.cumprod(_pre).tolist())), device=device)
    ri_s = []
    for p in _pre[:-1]:
        # ri = holistic_inds // p
        ri = holistic_inds.floor_divide(p)
        holistic_inds = holistic_inds - ri * p
        ri_s.append(ri)
    ri_s.append(holistic_inds)
    return torch.stack(ri_s)

def search_index(item: torch.Tensor, list_search_within: torch.Tensor):
    """
    TODO    Consider implementing this in CUDA for better performance/speed.
            For elements that do not exist, return an index of -1.
    """
    prefix = item.shape[:-1]
    prefix_dim = len(prefix)
    match = (list_search_within[(slice(None),)+(None,)*prefix_dim]==item.unsqueeze(0)).all(dim=-1)
    idx = torch.nonzero(match)
    return idx[..., 0]

def unique_consecutive_cumucount(inds: torch.Tensor):
    """
    NOTE: This function requires inds to be consecutive;
    """
    assert inds.dim() == 1, 'Only works for 1D Tensors.'
    u, cnt = torch.unique_consecutive(inds, return_counts=True)
    cumu_count = interleave_arange_simple(cnt, return_idx=False)
    return u, cnt, cumu_count

@functools.lru_cache(maxsize=128)
def create_ray_step_id_cuda(shape: Tuple[int,int]):
    ray_id = torch.arange(shape[0], device=torch.device('cuda')).view(-1,1).expand(shape).flatten()
    step_id = torch.arange(shape[1], device=torch.device('cuda')).view(1,-1).expand(shape).flatten()
    return ray_id, step_id

@functools.lru_cache(maxsize=128)
def create_ray_id_cuda(num_rays: int):
    ray_id = torch.arange(num_rays, device=torch.device('cuda'))
    return ray_id

if __name__ == "__main__":
    def test_cumu_count(device=torch.device('cuda')):
        from torch.utils.benchmark import Timer
        ridx = torch.arange(4096, device=device).repeat_interleave(torch.randint(32,96, [4096], device=device))
        # @torch.jit.script
        def cumu_count_v2(inds: torch.Tensor):
            u, cnt = torch.unique_consecutive(inds, return_counts=True)
            first_inds = cnt.cumsum(0).roll(1); first_inds[0] = 0
            cumu_count = torch.arange(inds.shape[0], device=inds.device) - torch.repeat_interleave(first_inds, cnt)
            return u, cnt, cumu_count
        # @torch.jit.script
        def cumu_count_v3(inds: torch.Tensor):
            u, cnt = torch.unique_consecutive(inds, return_counts=True)
            first_inds = cnt.cumsum(0) - cnt
            cumu_count = torch.arange(inds.shape[0], device=inds.device) - torch.repeat_interleave(first_inds, cnt)
            return u, cnt, cumu_count
        y2 = cumu_count_v2(ridx)[-1]
        y3 = cumu_count_v3(ridx)[-1]
        y4 = unique_consecutive_cumucount(ridx)[-1] # v4
        
        print(torch.allclose(y2, y3))
        print(torch.allclose(y2, y4))
        
        # 137 us
        print(Timer(
            stmt="cumu_count_v2(ridx)",
            globals={'cumu_count_v2': cumu_count_v2, 'ridx':ridx}
        ).blocked_autorange())
        # 107 us
        print(Timer(
            stmt="cumu_count_v3(ridx)",
            globals={'cumu_count_v3': cumu_count_v3, 'ridx':ridx}
        ).blocked_autorange())
        # 56 us
        print(Timer(
            stmt="unique_consecutive_cumucount(ridx)",
            globals={'unique_consecutive_cumucount': unique_consecutive_cumucount, 'ridx':ridx}
        ).blocked_autorange())

    test_cumu_count()
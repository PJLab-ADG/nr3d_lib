import functools
from math import prod
from typing import Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max

from nr3d_lib.models.annealers import get_anneal_val
from nr3d_lib.maths import normalized_logistic_density

err_msg_empty_occ = "Occupancy grid becomes empty during training. "\
    "Your model/algorithm/training settings might be incorrect. "\
    "Please check configs and tensorboard."

""" Sample """
def sample_pts_in_voxels(
    gidx: torch.LongTensor, num_pts: int, resolution: torch.LongTensor, 
    device=None, dtype=torch.float) -> Tuple[torch.Tensor, torch.LongTensor]:
    """
    Returns the normalized points in range [-1,1] and the voxel index of each sampled point.
    """
    assert gidx.dim() == 2, f"Only support gidx with shape [N,num_dim]"
    
    device = device or gidx.device
    num_dim = gidx.shape[-1]
    num_voxels = gidx.shape[0]
    if num_pts / num_voxels < 2.0:
        vidx = torch.randint(num_voxels, [num_pts, ], device=device)
        offsets = torch.rand([num_pts, num_dim], device=device, dtype=dtype)
        pts = ((gidx[vidx] + offsets) / resolution.float()) * 2 - 1
    else:
        n_per_vox = int(num_pts // num_voxels) + 1
        offsets = torch.rand([num_voxels, n_per_vox, num_dim], device=device, dtype=dtype)
        pts = ((gidx[:,None,:] + offsets) / resolution.float()).view(-1,num_dim) * 2 - 1
        vidx = torch.arange(num_voxels, device=device, dtype=torch.long)
        vidx = vidx.unsqueeze(-1).expand(num_voxels, n_per_vox).reshape(-1).contiguous()
    return pts, vidx

# def sample_pts_uniform(
#     num_pts: int, resolution: torch.LongTensor, 
#     gidx_full: torch.LongTensor, # [R0*R1*R2, 3]
#     device=None, dtype=torch.float
#     ):
#     device = device or gidx_full.device
#     assert gidx_full.dim() == 2, f"Only support gidx with shape [N,num_dim]"
#     num_dim = resolution.dim()
#     num_voxels = len(gidx_full)
#     # NOTE: Should be equal to sample_pts_in_voxels(gidx_full)
#     if num_pts / num_voxels < 2.0:
#         pts = torch.empty([num_pts, num_dim], device=device, dtype=dtype).uniform_(-1, 1)
#     else:
#         n_per_vox = int(num_pts // num_voxels) + 1
#         offsets = torch.rand([num_voxels, n_per_vox, num_dim], device=device, dtype=dtype)
#         pts = ((gidx_full[:,None,:] + offsets) / self.resolution).view(-1,num_dim) * 2 -1
#     return pts

"""
The key is how we define `occupancy`; 
ideally, everything should be converted to a unified concept of "occupancy" and `threshold`
"""
def sdf_to_occ_val(sdf: torch.Tensor, *, inv_s: float = None, inv_s_anneal_cfg: dict = None):
    if inv_s_anneal_cfg is not None:
        inv_s = get_anneal_val(**inv_s_anneal_cfg)
    else:
        assert inv_s is not None, "Need config `inv_s`"
    return normalized_logistic_density(sdf, inv_s)

def get_occ_val_fn(type: Literal['sdf', 'density', 'occ'] = 'sdf', **kwargs):
    if type == 'sdf':
        return functools.partial(sdf_to_occ_val, **kwargs)
    elif type == "raw_sdf":
        return lambda sdf: 1.0 - torch.abs(sdf)
    elif type == 'occ':
        return nn.Identity()
    elif type == 'density':
        return nn.Identity()
    else:
        raise RuntimeError(f"Invalid type={type}")

def binarize(occ_val: torch.FloatTensor, occ_threshold: float, consider_mean=False, eps=1e-5) -> torch.BoolTensor:
    # NOTE: (-eps) is to deal with all same values.
    threshold = occ_threshold if not consider_mean else (occ_val.mean() - eps).clamp_max_(occ_threshold)
    return occ_val > threshold

"""
The trailing underscore `_` means that the functions modifies input `occ_val_grid` in place
"""

""" Single update """
def update_occ_val_grid_idx_(occ_val_grid: torch.Tensor, gidx: torch.LongTensor, occ_val: torch.Tensor, ema_decay: float = 1.0):
    resolution_l = occ_val_grid.shape
    occ_val = occ_val.flatten().to(occ_val_grid)
    gidx_ravel = (gidx * gidx.new_tensor([resolution_l[1] * resolution_l[2], resolution_l[2], 1])).sum(-1)
    # 267 us @ 2M pts
    # NOTE: `out` argument also participates in `maximum` reduce.
    #       This is similar to `scatter_reduce_`'s behavior in pytorch 1.12 with `include_self=True`
    # NOTE: EMA update of val grid; only on selected indices.
    occ_val_new, _ = scatter_max(occ_val, gidx_ravel, out=ema_decay * occ_val_grid.flatten())
    occ_val_grid.index_put_(tuple(gidx.t()), occ_val_new[gidx_ravel])

def update_occ_val_grid_(occ_val_grid: torch.Tensor, pts: torch.FloatTensor, occ_val: torch.Tensor, ema_decay: float = 1.0):
    device, dtype, resolution_l = occ_val_grid.device, occ_val_grid.dtype, occ_val_grid.shape
    resolution = torch.tensor([resolution_l], device=device)
    pts, occ_val = pts.flatten(0, -2), occ_val.flatten().to(occ_val_grid)
    gidx = ((pts/2. + 0.5) * resolution).long().clamp(resolution.new_tensor([0]), resolution-1)
    update_occ_val_grid_idx_(occ_val_grid, gidx, occ_val, ema_decay=ema_decay)

""" Batched update """
def update_batched_occ_val_grid_idx_(
    occ_val_grid: torch.Tensor, bidx: torch.LongTensor = None, gidx: torch.LongTensor = ..., occ_val: torch.Tensor = ..., ema_decay: float = 1.0):
    num_batches, resolution_l = occ_val_grid.shape[0], occ_val_grid.shape[1:]
    if bidx is not None:
        # Update given non-batched input `occ_val` and the correponding batch inds `bidx`
        bidx, occ_val = bidx.flatten(), occ_val.flatten().to(occ_val_grid)
        idx_ravel = bidx * prod(resolution_l) + (gidx * gidx.new_tensor([resolution_l[1] * resolution_l[2], resolution_l[2], 1])).sum(-1)
        occ_val_new, _ = scatter_max(occ_val, idx_ravel, dim=0, out=ema_decay * occ_val_grid.flatten())
        occ_val_grid.index_put_((bidx,) + tuple(gidx.t()), occ_val_new[idx_ravel])
    else:
        # Update given batched input `occ_val`
        occ_val = occ_val.flatten(1, -1).to(occ_val_grid)
        gidx_ravel = (gidx * gidx.new_tensor([resolution_l[1] * resolution_l[2], resolution_l[2], 1])).sum(-1) # [B, num_pts]
        occ_val_new, _ = scatter_max(occ_val, gidx_ravel, dim=1, out=ema_decay * occ_val_grid.flatten(1,-1)) # [B, num_grid_voxels]
        occ_val_grid.view(num_batches, -1).scatter_(1, gidx_ravel, occ_val_new.gather(1, gidx_ravel))

def update_batched_occ_val_grid_(
    occ_val_grid: torch.Tensor, pts: torch.FloatTensor, bidx: torch.LongTensor = None, occ_val: torch.Tensor = ..., ema_decay: float = 1.0):
    resolution_l, device = occ_val_grid.shape[1:], occ_val_grid.device
    resolution = torch.tensor(resolution_l, device=device)
    gidx = ((pts/2. + 0.5) * resolution).long().clamp(resolution.new_tensor([0]), resolution-1)
    update_batched_occ_val_grid_idx_(occ_val_grid, bidx, gidx, occ_val, ema_decay=ema_decay)


"""
@file   utils.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Utility funcitons for NeuS model and rendering. Implemented on both batched-rays and packed-rays.

Below we provide two implementations of alpha calculation.
- `diff impl` stands for our implementation
- `nablas + estimate impl` stands for neus original implmentation

For more details on differences of these two implementations, please refer to nr3d_lib/models/fields/neus/renderer_mixin.py
"""

__all__ = [
    # NeuS fundamental funcs
    'neus_pdf', 
    'neus_cdf', 
    # NeuS on ray batch (diff impl)
    'neus_ray_cdf_to_alpha', 
    'neus_ray_sdf_to_tau', 
    'neus_ray_sdf_to_alpha', 
    'neus_ray_sdf_to_vw', 
    # NeuS on ray packs (diff impl)
    'neus_packed_cdf_to_alpha', 
    'neus_packed_sdf_to_tau', 
    'neus_packed_sdf_to_alpha', 
    'neus_packed_sdf_to_vw', 
    # NeuS (nablas + estimate impl)
    'neus_estimate_sdf_nablas_to_alpha', 
    # NeuS (upsample in official solution)
    'neus_packed_sdf_to_upsample_alpha', 
    'neus_ray_sdf_to_upsample_alpha', 
]

from typing import Union

import torch
import torch.nn.functional as F

from nr3d_lib.models.utils import logistic_density

from nr3d_lib.render.pack_ops import packed_diff
from nr3d_lib.render.volume_graphics import  tau_to_alpha, ray_tau_to_vw
from nr3d_lib.render.volume_graphics import packed_alpha_to_vw, ray_alpha_to_vw

#--------------------------------------------------------------------
#-------------------- NeuS fundamental funcs ------------------------
#--------------------------------------------------------------------
# @torch.jit.script
def neus_pdf(x: torch.Tensor, inv_s: torch.Tensor):
    return logistic_density(x, inv_s)

# @torch.jit.script
def neus_cdf(x, inv_s):
    return torch.sigmoid(x*inv_s)

#--------------------------------------------------------------------
#---------------- NeuS on ray batch (diff impl) ---------------------
#--------------------------------------------------------------------
# @torch.jit.script
def neus_ray_cdf_to_alpha(cdf: torch.Tensor, append_cdf_1=False):
    if append_cdf_1:
        alpha = -1 * cdf.diff(append=cdf.new_full((*cdf.shape[:-1],1), 1.)) / (cdf + 1e-5)
    else:
        alpha = -1 * cdf.diff() / (cdf[..., :-1] + 1e-5)
    return alpha.clamp_min_(0)

# @torch.jit.script
def neus_ray_sdf_to_tau(sdf: torch.Tensor, inv_s, append_cdf_1=False):
    logcdf = F.logsigmoid(sdf*inv_s)
    if append_cdf_1:
        tau = -1 * torch.diff(logcdf, append=sdf.new_full((*sdf.shape[:-1],1), 0.))
    else:
        tau = -1 * torch.diff(logcdf)
    return tau.clamp_min_(0)

# @torch.jit.script
def neus_ray_sdf_to_alpha(sdf: torch.Tensor, inv_s, append_cdf_1=False):
    return neus_ray_cdf_to_alpha(torch.sigmoid(sdf*inv_s), append_cdf_1=append_cdf_1)

# @torch.jit.script
def neus_ray_sdf_to_vw(sdf: torch.Tensor, inv_s, append_cdf_1=False):
    alpha = neus_ray_cdf_to_alpha(torch.sigmoid(sdf*inv_s), append_cdf_1=append_cdf_1)
    vw = ray_alpha_to_vw(alpha)
    return vw

#--------------------------------------------------------------------
#---------------- NeuS on ray packs (diff impl) ---------------------
#--------------------------------------------------------------------
def neus_packed_cdf_to_alpha(cdf: torch.Tensor, pack_infos: torch.LongTensor, append_cdf_1=False, pack_cdf_appends: torch.Tensor=None):
    if append_cdf_1:
        cdf_diff = -1 * packed_diff(cdf, pack_infos, pack_appends=cdf.new_full((pack_infos.shape[0],), 1.))
    elif pack_cdf_appends is not None:
        cdf_diff = -1 * packed_diff(cdf, pack_infos, pack_appends=pack_cdf_appends)
    else:
        cdf_diff = -1 * packed_diff(cdf, pack_infos)
    alpha = (cdf_diff / (cdf+1e-5)).clamp_min_(0)
    return alpha

def neus_packed_sdf_to_tau(sdf: torch.Tensor, inv_s, pack_infos: torch.LongTensor, append_cdf_1=False, pack_sdf_appends: torch.Tensor = None):
    logcdf = F.logsigmoid(sdf*inv_s)
    if append_cdf_1:
        # log(cdf=1) = 0
        tau = -1 * packed_diff(logcdf, pack_infos, pack_appends=sdf.new_full((pack_infos.shape[0],), 0.))
    elif pack_sdf_appends:
        logcdf_pack_appends = F.logsigmoid(pack_sdf_appends*inv_s)
        tau = -1 * packed_diff(logcdf, pack_infos, pack_appends=logcdf_pack_appends)
    else:
        tau = -1 * packed_diff(logcdf, pack_infos)
    return tau.clamp_min_(0)

def neus_packed_sdf_to_alpha(sdf: torch.Tensor, inv_s, pack_infos: torch.LongTensor, append_cdf_1=False, pack_sdf_appends: torch.Tensor = None):
    return neus_packed_cdf_to_alpha(torch.sigmoid(sdf * inv_s), pack_infos, append_cdf_1=append_cdf_1, pack_cdf_appends=(torch.sigmoid(pack_sdf_appends * inv_s) if pack_sdf_appends is not None else None))

def neus_packed_sdf_to_vw(sdf: torch.Tensor, inv_s, pack_infos: torch.LongTensor, append_cdf_1=False, pack_sdf_appends: torch.Tensor = None):
    alpha = neus_packed_cdf_to_alpha(torch.sigmoid(sdf * inv_s), pack_infos, append_cdf_1=append_cdf_1, pack_cdf_appends=(torch.sigmoid(pack_sdf_appends * inv_s) if pack_sdf_appends is not None else None))
    vw = packed_alpha_to_vw(alpha)
    return vw

#--------------------------------------------------------------------
#------------------ NeuS (nablas + estimate impl) -------------------
#--------------------------------------------------------------------
def neus_estimate_sdf_nablas_to_alpha(
    sdf: torch.Tensor, deltas: torch.Tensor, nablas: torch.Tensor, dirs: torch.Tensor, inv_s: Union[float, torch.Tensor], dir_scales: torch.Tensor=1, 
    ratio: float=1, delta_max: float=1e+10) -> torch.Tensor:
    """ NeuS alpha calculation via differences of 
    
        NOTE: Modified from https://github.com/Totoro97/NeuS

    Args:
        sdf (torch.Tensor): [..., ], SDF value of current points
        deltas (torch.Tensor): [..., ], Interval length from current points to the next points
        nablas (torch.Tensor): [..., 3], nablas value of current points
        dirs (torch.Tensor): [..., 3], Normalized ray direction vectors.
            NOTE: Must be normalized !!!
        inv_s (Union[float, torch.Tensor]): `inv_s` parameter
        dir_scales (torch.Tensor, optional): [..., ], Optional deltas length scale. Defaults to 1.
        ratio (float, optional): Ratio of true cosine (for training stability). Defaults to 1.
        delta_max (float, optional): Maximum assumed interval length. Defaults to 0.01.

    Returns:
        torch.Tensor: [..., ] The calculated alpha values
    """
    
    true_cos = (dirs * nablas).sum(-1, keepdim=True)
    # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
    # the cos value "not dead" at the beginning training iterations, for better convergence.
    if ratio == 1:
        iter_cos = -F.relu_(-true_cos)
    elif ratio == 0:
        iter_cos = -F.relu_(-true_cos * 0.5 + 0.5)
    else:
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - ratio) + F.relu(-true_cos) * ratio)
    # [..., 2]
    prev_next_half_deltas = (deltas.new_tensor([-0.5, 0.5]) * (deltas * dir_scales).unsqueeze_(-1).clamp_max_(delta_max))
    estimated_prev_next_sdf = torch.addcmul(sdf.unsqueeze(-1), iter_cos, prev_next_half_deltas)
    prev_next_cdf = torch.sigmoid(estimated_prev_next_sdf * inv_s)
    # [...]
    alpha = (prev_next_cdf[..., 0] - prev_next_cdf[..., 1]) / (prev_next_cdf[..., 0]+1e-5)
    return alpha.clamp_min_(0)

#--------------------------------------------------------------------
#-------------- NeuS (upsample in official solution) ----------------
#--------------------------------------------------------------------
@torch.no_grad()
def neus_packed_sdf_to_upsample_alpha(
    sdf: torch.Tensor, depth_samples: torch.Tensor, inv_s: torch.Tensor, pack_infos: torch.LongTensor, 
    ) -> torch.Tensor:
    """ Pack-wise: Re-estimate prev,next sdfs and calculates interval alpha values

    Args:
        sdf (torch.Tensor): [num_packed_pts, ] The given interval boundary points.
        depth_samples (torch.Tensor): [num_packed_pts, ] The given interval boundary depth values
        inv_s (torch.Tensor): `inv_s` parameter
        pack_infos (torch.LongTensor): [num_packs, 2] pack_infos for sdf & depth_samples

    Returns:
        torch.Tensor: [num_packed_pts, ] The calculated alpha value for each interval (with trailing zero on each pack)
    """
    sdf_diff = packed_diff(sdf, pack_infos) # Each pack with trailing zero
    deltas = packed_diff(depth_samples, pack_infos) # Each pack with trailing zero
    mid_sdf = sdf + sdf_diff * 0.5
    dot_val = sdf_diff / (deltas + 1e-5)
    prev_dot_val = dot_val.roll(1).index_fill_(0, pack_infos[:,0], 0) # Can be replaced by packed shift
    dot_val = torch.minimum(prev_dot_val, dot_val).clamp_(-10,0)
    
    prev_next_esti_sdf = torch.addcmul(mid_sdf.unsqueeze(-1).to(depth_samples.dtype), dot_val.unsqueeze(-1), deltas.unsqueeze(-1) * deltas.new_tensor([-0.5, 0.5]))
    prev_next_cdf = torch.sigmoid(prev_next_esti_sdf * inv_s)
    alpha = (prev_next_cdf[..., 0] - prev_next_cdf[..., 1]) / (prev_next_cdf[..., 0]+1e-5)
    return alpha.clamp_min_(0)

@torch.no_grad()
def neus_ray_sdf_to_upsample_alpha(
    sdf: torch.Tensor, depth_samples: torch.Tensor, inv_s: torch.Tensor
    ) -> torch.Tensor:
    """ Batch-wise: Re-estimate prev,next sdfs and calculates interval alpha values

    Args:
        sdf (torch.Tensor): [..., num_pts+1] The given interval boundary points.
            Number of interval boundary points = number of intervals + 1
        depth_samples (torch.Tensor): [..., num_pts+1] The given interval boundary depth values
        inv_s (torch.Tensor): `inv_s` parameter

    Returns:
        torch.Tensor: [..., num_pts] The calculated alpha value
    """
    
    prev_sdf, next_sdf = sdf[..., :-1], sdf[..., 1:] # [..., num_pts]
    prev_z_vals, next_z_vals = depth_samples[..., :-1], depth_samples[..., 1:] # [..., num_pts]
    deltas = (next_z_vals - prev_z_vals) # [..., num_pts]
    mid_sdf = (prev_sdf + next_sdf) * 0.5 # [..., num_pts]
    dot_val = (next_sdf - prev_sdf) / (deltas + 1e-5) # [..., num_pts]
    prev_dot_val = torch.cat([dot_val.new_zeros([*prev_sdf.shape[:-1], 1]), dot_val[..., :-1]], -1) # [..., num_pts], prev_slope, right shifted
    dot_val = torch.minimum(prev_dot_val, dot_val).clamp_(-10.0, 0.0) # [..., num_pts]
    
    prev_next_esti_sdf = torch.addcmul(mid_sdf.unsqueeze(-1), dot_val.unsqueeze(-1), deltas.unsqueeze(-1) * deltas.new_tensor([-0.5, 0.5])) # [..., num_pts, 2]
    prev_next_cdf = torch.sigmoid(prev_next_esti_sdf * inv_s) # [..., num_pts, 2]
    alpha = (prev_next_cdf[..., 0] - prev_next_cdf[..., 1]) / (prev_next_cdf[..., 0]+1e-5) # [..., num_pts]
    return alpha.clamp_min_(0)

if __name__ == "__main__":
    def test_tau_alpha(device=torch.device('cuda')):
        from icecream import ic
        sdf = torch.randn(7, 13, device=device)
        inv_s = torch.tensor([10.0], device=device)
        logcdf = F.logsigmoid(sdf*inv_s)
        cdf1 = torch.sigmoid(sdf*inv_s)
        cdf2 = torch.exp(logcdf)
        # ic(cdf1, cdf2)
        print(torch.allclose(cdf1, cdf2))
        
        tau = neus_ray_sdf_to_tau(sdf, inv_s)
        
        alpha1 = neus_ray_sdf_to_alpha(sdf, inv_s)
        alpha2 = tau_to_alpha(tau)
        # ic(alpha1, alpha2)
        print(torch.allclose(alpha1, alpha2, atol=1e-2, rtol=1e-2)) # Will be slightly different but acceptable
        
        vw1 = neus_ray_sdf_to_vw(sdf, inv_s)
        vw2 = ray_tau_to_vw(tau)
        # ic(vw1, vw2)
        print(torch.allclose(vw1, vw2, atol=1e-2, rtol=1e-1))
        
        # Test grad
        sdf = torch.randn(7, 13, device=device, requires_grad=True)
        inv_s = torch.tensor([10.0], device=device, requires_grad=True)
        color = torch.rand(7, 12, 3, device=device, requires_grad=True)
        c_gt = torch.rand(7, 3, device=device)
        
        vw = ray_tau_to_vw(neus_ray_sdf_to_tau(sdf, inv_s))
        c = (vw.unsqueeze(-1) * color).sum(dim=-2)
        loss = F.mse_loss(c, c_gt)
        loss.backward()
        
        sdf_grad1, s_grad1, color_grad1 = sdf.grad, inv_s.grad, color.grad
        
        sdf.grad, inv_s.grad, color.grad = None, None, None
        vw = neus_ray_sdf_to_vw(sdf, inv_s)
        c = (vw.unsqueeze(-1) * color).sum(dim=-2)
        loss = F.mse_loss(c, c_gt)
        loss.backward()
        
        sdf_grad2, s_grad2, color_grad2 = sdf.grad, inv_s.grad, color.grad
        
        print(torch.allclose(sdf_grad1, sdf_grad2, rtol=1e-1, atol=1e-4)) # Most of the values are the same; a little has larger difference 
        print(torch.allclose(s_grad1, s_grad2, rtol=1e-3))
        print(torch.allclose(color_grad1, color_grad2, rtol=1e-2, atol=1e-4))

    test_tau_alpha()
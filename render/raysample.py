"""
@file   raysample.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Ray sampling untility functions. Implemented on both batched-rays and packed-rays.
"""

__all__ = [
    #---- Sample points from packs
    'packed_sample_cdf', 
    'packed_sample_pdf', 
    'interleave_sample_step_linear', 
    'interleave_sample_step_wrt_depth_clamped', 
    # 'interleave_sample_step_wrt_sqrt_depth', 
    'interleave_sample_step_linear_in_nearfar', 
    'interleave_sample_step_wrt_depth_in_nearfar', 
    'interleave_sample_step_linear_in_packed_segments', 
    'interleave_sample_step_wrt_depth_in_packed_segments', 
    #---- Sample points from batched-ray
    'batch_sample_cdf', 
    'batch_sample_pdf', 
    'batch_sample_step_linear', 
    'batch_sample_step_wrt_depth_unsafe', 
    'batch_sample_step_wrt_depth', 
    'batch_sample_step_wrt_sqrt_depth', 
]

from typing import Tuple, Union

import torch
from torch.utils.benchmark import Timer

from nr3d_lib.render.pack_ops import get_pack_infos_from_n, interleave_arange_simple, interleave_linstep, packed_cumsum, packed_diff, packed_div, packed_invert_cdf, interleave_sample_step_wrt_depth_clamped, interleave_sample_step_wrt_depth_in_packed_segments

#----------------------------------------------------
#-------- Sampling points from packs ----------------
#----------------------------------------------------
@torch.no_grad()
def packed_sample_cdf(bins: torch.Tensor, cdfs: torch.Tensor, pack_infos: torch.LongTensor, num_to_sample: int, perturb=False) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Pack-wise inverse cdf sampling

    Args:
        bins (torch.Tensor): [num_pts] Packed tensor, intervals' boundary points
        cdfs (torch.Tensor): [num_pts] Packed tensor, CDF value corresponding to each boundary point (with leading zero on each pack)
        pack_infos (torch.LongTensor): [num_packs, 2] pack_infos for bins/cdfs
        num_to_sample (int): Number of samples on each pack
        perturb (bool, optional): Whether to randomize. Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            t_samples: [num_packs, num_to_sample] Sampled `t` values on each pack
            bin_idx: [num_packs, num_to_sample] The global bin indices corresponding to each `t` sample point
                NOTE: Use `bin_idx - pack_infos[:,0].unsqueeze(-1)` to get "local" bin indices on each pack.
    """
    num_packs, device, dtype = pack_infos.shape[0], bins.device, bins.dtype
    if not perturb:
        u = torch.linspace(0., 1., num_to_sample+2, device=device, dtype=dtype)[1:-1].expand((num_packs, num_to_sample))
    else:
        # NOTE: No need to sort as already sorted.
        u = batch_sample_step_linear(bins.new_zeros((num_packs)), bins.new_ones((num_packs)), num_to_sample, perturb=True)
    t_samples, bin_idx = packed_invert_cdf(bins, cdfs.to(dtype), u.contiguous(), pack_infos)
    return t_samples, bin_idx

@torch.no_grad()
def packed_sample_pdf(bins: torch.Tensor, weights: torch.Tensor, pack_infos: torch.LongTensor, num_to_sample: int, perturb=False) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Pack-wise weighted sampling based on inverse cdf sampling

    Args:
        bins (torch.Tensor): [num_packed_pts] Packed tensor, intervals' boundary points
        weights (torch.Tensor): [num_packed_pts] Packed tensor, PDF value corresponding to each interval (with trailing zero on each pack)
        pack_infos (torch.LongTensor): [num_packs, 2] pack_infos for bins/pdfs
        num_to_sample (int): Number of samples on each pack
        perturb (bool, optional): Whether to randomize. Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            t_samples: [num_packs, num_to_sample] Sampled `t` values on each pack
            bin_idx: [num_packs, num_to_sample] The global bin indices corresponding to each `t` sample point
                NOTE: Use `bin_idx - pack_infos[:,0].unsqueeze(-1)` to get "local" bin indices on each pack.
    """
    # NOTE: Right-shifted cumsum, resulting preceding zero CDF
    cdfs = packed_cumsum(weights, pack_infos, exclusive=True)
    last_cdf = cdfs[pack_infos.sum(-1).sub_(1)]
    cdfs = packed_div(cdfs, last_cdf.clamp_min(1e-5), pack_infos)
    return packed_sample_cdf(bins, cdfs, pack_infos, num_to_sample=num_to_sample, perturb=perturb)

@torch.no_grad()
def interleave_sample_step_linear(
    near: torch.Tensor, far: torch.Tensor, 
    step_size: float=0.01, min_steps: int=2, step_size_factor: float=1.0, perturb=False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    step_size = step_size * step_size_factor
    n_per_pack = ((far-near) / step_size).round_().long().clamp_min_(min_steps) # 27 us
    cumsum = n_per_pack.cumsum(0) # 11 us
    last_inds, pack_indices = cumsum-1, cumsum-n_per_pack # 13 us
    if not perturb:
        # range: {0, 1, ..., n_per_pack-1}
        indexing, ridx = interleave_arange_simple(n_per_pack) # 32 us
        dt = (far-near) / (n_per_pack-1) # 20 us
        deltas = dt[ridx] # 10 us
        t_samples = torch.addcmul(near[ridx], indexing.to(dtype=near.dtype), deltas) # 23 us
    else:
        # range: rand(0,1) + {0, 1, ..., n_per_pack-1}
        indexing, ridx = interleave_arange_simple(n_per_pack) # 33 us
        # NOTE: Important: add noise to each already-determined interval
        indexing = indexing.to(near.dtype) + torch.rand_like(indexing, dtype=near.dtype, device=near.device) # 28 us
        dt = (far-near) / n_per_pack # 13 us
        t_samples = torch.addcmul(near[ridx], indexing.to(dtype=near.dtype), dt[ridx]) # 33 us
        deltas = t_samples.diff(append=t_samples.new_empty([1])).index_put_((last_inds,), dt) # 36 us
    pack_infos = torch.stack([pack_indices, n_per_pack], -1)
    return t_samples, deltas, ridx, pack_infos

# @torch.no_grad()
# def interleave_sample_step_wrt_sqrt_depth(
#     near: torch.Tensor, far: torch.Tensor, max_steps: int=512,
#     dt_gamma: float=0.01, step_size_factor: float=1.0, perturb=False
#     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#     pass

@torch.no_grad()
def interleave_sample_step_linear_in_nearfar(
    near: torch.Tensor, far: torch.Tensor, step_size: float=0.01, step_size_factor: float=1.0, min_steps: int=2, perturb=False):
    step_size = step_size * step_size_factor
    if not perturb:
        entry_ind = near.div_(step_size, rounding_mode='trunc').long()
        exit_ind = far.div_(step_size, rounding_mode='trunc').long().add_(1)
        
        n_per_pack = (exit_ind - entry_ind).clamp_min_(min_steps)
        pack_infos = torch.stack([n_per_pack.cumsum(0) - n_per_pack, n_per_pack], 1)
        
        indexing, ridx = interleave_linstep(entry_ind, n_per_pack, 1, return_idx=True)
        
        t_samples = indexing.to(near.dtype) * step_size
        deltas = near.new_full(ridx.shape, step_size)
    else:
        entry_ind = near.div_(step_size, rounding_mode='trunc').long()
        exit_ind = far.div_(step_size, rounding_mode='trunc').long()
        
        n_per_pack = (exit_ind - entry_ind).clamp_min_(min_steps)
        pack_infos = torch.stack([n_per_pack.cumsum(0) - n_per_pack, n_per_pack], 1)

        indexing, ridx = interleave_linstep(entry_ind, n_per_pack, 1, return_idx=True)
        indexing = indexing.to(near.dtype) + torch.rand_like(indexing, dtype=near.dtype, device=near.device) # 40 us @ 1M pts
        
        t_samples = indexing * step_size
        last_inds = pack_infos[..., 0] + pack_infos[..., 1] - 1
        deltas = packed_diff(t_samples, pack_infos).index_fill_(0, last_inds, step_size)
    
    return t_samples, deltas, ridx, pack_infos

@torch.no_grad()
def interleave_sample_step_wrt_depth_in_nearfar(
    near: torch.Tensor, far: torch.Tensor, common_start: float = 0.1, 
    max_steps: int=512, dt_gamma: float=0.01, min_step_size: float=0.01, max_step_size: float=1.0, step_size_factor: float=1.0, perturb=False):
    """
    `common_start`: The "common starting point" for all rays -> A setting shared among all rays at different times
    """
    common_start = near.new_full(near.shape, common_start)
    raise NotImplementedError

@torch.no_grad()
def interleave_sample_step_linear_in_packed_segments_v1(
    near: Union[torch.Tensor, float], far: Union[torch.Tensor, float], entry: torch.Tensor, exit: torch.Tensor, seg_pack_infos: torch.Tensor,
    step_size: float=0.01, step_size_factor: float=1.0, min_steps: int=2, max_steps: int = 512, perturb=False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    
    num_rays = seg_pack_infos.shape[0]
    near = entry.new_full([num_rays], near) if not isinstance(near, torch.Tensor) else near
    far = entry.new_full([num_rays], far) if not isinstance(far, torch.Tensor) else far
    
    step_size = step_size * step_size_factor
    _, seg_ridx = interleave_arange_simple(seg_pack_infos[...,1].contiguous()) # 31 us @ 130k pts; 146 us @ 2M pts
    _near = near[seg_ridx] # 6.8 us @ 130k pts; 38.7 us @ 2M pts
    # seg_ridx = torch.arange(seg_pack_infos.shape[0], device=seg_pack_infos.device, dtype=seg_pack_infos.dtype).repeat_interleave(n_seg_per_pack) # 200 us @ 500k segments -> 2M pts
    # _near = near.repeat_interleave(n_seg_per_pack) # 171 us @ 500k segments -> 2M pts

    if not perturb:
        entry_ind = (entry - _near).div_(step_size, rounding_mode='trunc').long() # 17 us @ 180k pts; 90 us @ 2M pts
        exit_ind = (exit - _near).div_(step_size, rounding_mode='trunc').long().add_(1) # 32 us @ 130k pts; 222 us @ 2M pts
        n_per_seg = (exit_ind - entry_ind).clamp_(min_steps, max_steps)
        out_seg_pack_infos = get_pack_infos_from_n(n_per_seg)
        
        # seg_in_far = (entry < far).nonzero().long()[..., 0]
        indexing, sidx = interleave_linstep(entry_ind, n_per_seg, 1, return_idx=True)
        
        t_samples = _near[sidx] + indexing.to(near.dtype) * step_size
        deltas = near.new_full(sidx.shape, step_size)
    else:
        entry_ind = (entry - _near).div_(step_size, rounding_mode='trunc').long()
        exit_ind = (exit - _near).div_(step_size, rounding_mode='trunc').long()
        n_per_seg = (exit_ind - entry_ind).clamp_(min_steps, max_steps)
        out_seg_pack_infos = get_pack_infos_from_n(n_per_seg)
        
        indexing, sidx = interleave_linstep(entry_ind, n_per_seg, 1, return_idx=True)
        indexing = indexing.to(near.dtype) + torch.rand_like(indexing, dtype=near.dtype, device=near.device) # 40 us @ 1M pts
        
        t_samples = _near[sidx] + indexing * step_size
        last_inds = out_seg_pack_infos[...,0] + out_seg_pack_infos[...,1] - 1
        deltas = packed_diff(t_samples, out_seg_pack_infos).index_fill_(0, last_inds, step_size)
    
    ridx = seg_ridx[sidx]
    n_per_ray = seg_pack_infos.new_zeros([num_rays])
    n_per_ray.index_add_(0, seg_ridx, n_per_seg)
    ray_pack_infos = get_pack_infos_from_n(n_per_ray)
    
    return t_samples, deltas, ridx, ray_pack_infos, sidx, out_seg_pack_infos

@torch.no_grad()
def interleave_sample_step_linear_in_packed_segments_v2(
    near: Union[torch.Tensor, float], far: Union[torch.Tensor, float], entry: torch.Tensor, exit: torch.Tensor, seg_pack_infos: torch.Tensor,
    step_size: float=0.01, step_size_factor: float=1.0, max_steps: int=512, perturb=False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return interleave_sample_step_wrt_depth_in_packed_segments(
        near, far, entry, exit, seg_pack_infos, perturb=perturb, max_steps=max_steps, dt_gamma=0, min_step_size=step_size, step_size_factor=step_size_factor)

interleave_sample_step_linear_in_packed_segments = interleave_sample_step_linear_in_packed_segments_v1

#----------------------------------------------------
#-------- Sampling points from ray ------------------
#----------------------------------------------------
@torch.no_grad()
def batch_sample_cdf(bins: torch.Tensor, cdf: torch.Tensor, num_to_sample: int, perturb=False, eps: float=1e-5) -> torch.Tensor:
    """ Batch-wise inverse cdf sampling

    Args:
        bins (torch.Tensor): [..., num_pts+1] Interval boundary points
        cdf (torch.Tensor): [..., num_pts+1] Interval CDFs with leading zero
        num_to_sample (int): Number of points to sample on each batch
        perturb (bool, optional): Whether to randomize. Defaults to False.
        eps (float, optional): Numeric tolerance. Defaults to 1e-5.

    Returns:
        torch.Tensor: [..., num_to_sample] The samples `t` values
    """
    dtype, device, prefix = bins.dtype, bins.device, bins.shape[:-1]
    # Sample in CDF range (0,1)
    if not perturb:
        u = torch.linspace(0., 1., steps=num_to_sample+2, device=device, dtype=dtype)[..., 1:-1].expand((*prefix, num_to_sample))
    else:
        # u = torch.rand([*cdf.shape[:-1], num_to_sample], device=device).clamp_(1.0e-4, 1-1.0e-4)
        # NOTE: No need to sort as already sorted.
        u = batch_sample_step_linear(bins.new_zeros(prefix), bins.new_ones(prefix), num_to_sample, perturb=True)
    u = u.contiguous()

    # Invert CDF
    inds = torch.searchsorted(cdf.detach(), u, right=False)

    below, above = torch.clamp_min(inds-1, 0), torch.clamp_max(inds, cdf.shape[-1]-1)
    inds_g = torch.stack([below, above], -1)

    matched_shape = [*inds_g.shape[:-1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(-2).expand(matched_shape), -1, inds_g)
    bins_g = torch.gather(bins.unsqueeze(-2).expand(matched_shape), -1, inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom[denom<eps] = 1
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples

@torch.no_grad()
def batch_sample_pdf(bins: torch.Tensor, weights: torch.Tensor, num_to_sample: int, perturb=False, eps=1e-5) -> torch.Tensor:
    """ Batch-wise weighted sampling based on inverse cdf sampling

    Args:
        bins (torch.Tensor): [..., num_pts+1] Interval boundary points
        weights (torch.Tensor): [..., num_pts] Interval weights
        num_to_sample (int): Number of sample on each batch
        perturb (bool, optional): Whether to randomize. Defaults to False.
        eps (_type_, optional): Numeric tolerance. Defaults to 1e-5.

    Returns:
        torch.Tensor: [..., num_to_sample] The sampled `t` values
    """    
    # OPTION1:
    pdf = weights / torch.sum(weights, -1, keepdim=True).clamp_min_(eps)
    # # OPTION2: will introduce inf
    # weights = weights + 1e-5  # prevent nans 
    # pdf = weights / weights.sum(-1, keepdim=True)
    cdf = torch.cumsum(pdf, -1) # Accumulated pdf = cdf
    cdf = torch.cat([cdf.new_zeros([*cdf.shape[:-1],1]), cdf], -1) # [..., num_pts+1] with leading zero
    return batch_sample_cdf(bins, cdf, num_to_sample, perturb, eps)

@torch.no_grad()
def batch_sample_step_linear(
    near: torch.Tensor, far: torch.Tensor, num_samples: int, prefix_shape=None, perturb=False, return_dt=False):
    
    if prefix_shape is None: 
        prefix_shape = [1] if ([*near.shape] == [1]) else near.squeeze().shape
    near, far = near.squeeze().expand(prefix_shape).unsqueeze_(-1), far.squeeze().expand(prefix_shape).unsqueeze_(-1)
    device, dtype = near.device, near.dtype
    
    if not perturb:
        # [num_samples,] range: {0, 1, ..., num_samples-1}
        indexing = torch.arange(num_samples, device=device)
        dt = (far-near) / (num_samples-1)
    else:
        # [num_samples,] range: rand(0,1) + {0, 1, ..., num_samples-1}
        indexing = torch.arange(num_samples, device=device) + torch.rand([*prefix_shape, num_samples], dtype=dtype, device=device)
        dt = (far-near) / num_samples
    t_samples = torch.addcmul(near, indexing.to(dtype=dtype), dt)
    if not return_dt:
        return t_samples
    else:
        if not perturb:
            deltas = dt.expand((*prefix_shape, num_samples))
        else:
            deltas = t_samples.new_zeros(t_samples.shape)
            deltas[..., :-1], deltas[..., -1] = torch.diff(t_samples, dim=-1), (far[...,0]-near[...,0])/num_samples
        return t_samples, deltas

@torch.no_grad()
def batch_sample_step_wrt_depth_unsafe(
    near: torch.Tensor, far: torch.Tensor, num_samples: int, prefix_shape=None, perturb=False, return_dt=False):
    
    if prefix_shape is None: 
        prefix_shape = [1] if ([*near.shape] == [1]) else near.squeeze().shape
    near, far = near.squeeze().expand(prefix_shape).unsqueeze_(-1), far.squeeze().expand(prefix_shape).unsqueeze_(-1)
    device, dtype = near.device, near.dtype
    
    ratio = far/near # NOTE: Unsafe!
    if not perturb:
        logk = torch.log(ratio) / (num_samples - 1)
        # [num_samples,] range: {1-num_samples, 2-num_samples, ..., -1, 0}
        indexing = torch.arange(1-num_samples, 1, 1, device=device)
    else:
        logk = torch.log(ratio) / (num_samples)
        prefix_shape = logk.shape[:-1]
        # [num_samples,] range: rand(0,1) + {-num_samples, 1-num_samples, 2-num_samples, ..., -1}
        indexing = torch.arange(-num_samples, 0, 1, device=device) + torch.rand((*prefix_shape, num_samples), dtype=dtype, device=device)
    logr = logk * indexing
    t_samples = far * torch.exp(logr)
    if not return_dt:
        return t_samples
    else:
        deltas = t_samples.new_zeros(t_samples.shape)
        deltas[..., :-1]  = torch.diff(t_samples, dim=-1)
        deltas[..., -1] = deltas[..., -2]
        return t_samples, deltas

@torch.no_grad()
def batch_sample_step_wrt_depth(
    near: torch.Tensor, far: torch.Tensor, num_samples: int, prefix_shape=None, perturb=False, return_dt=False):
    
    if prefix_shape is None: 
        prefix_shape = [1] if ([*near.shape] == [1]) else near.squeeze().shape
    near, far = near.squeeze().expand(prefix_shape).unsqueeze_(-1), far.squeeze().expand(prefix_shape).unsqueeze_(-1)
    device, dtype = near.device, near.dtype

    ratio = (far/near).clamp_max_(num_samples)
    near_ = far / ratio
    t_samples = batch_sample_step_wrt_depth_unsafe(near_, far, num_samples=num_samples, prefix_shape=prefix_shape, perturb=perturb)
    t_samples = (t_samples-near_) * ((far-near)/(far-near_)) + near

    if not return_dt:
        return t_samples
    else:
        deltas = t_samples.new_zeros(t_samples.shape)
        deltas[..., :-1]  = torch.diff(t_samples, dim=-1)
        deltas[..., -1] = deltas[..., -2]
        return t_samples, deltas

@torch.no_grad()
def batch_sample_step_wrt_sqrt_depth(
    near: torch.Tensor, far: torch.Tensor, num_samples: int, prefix_shape=None, perturb=False, return_dt=False):
    
    if prefix_shape is None: 
        prefix_shape = [1] if ([*near.shape] == [1]) else near.squeeze().shape
    near, far = near.squeeze().expand(prefix_shape).unsqueeze_(-1), far.squeeze().expand(prefix_shape).unsqueeze_(-1)
    device, dtype = near.device, near.dtype
    
    c = (4*near).sqrt()
    if not perturb:
        k = ((4*far).sqrt()-c) / (num_samples-1)
        # [num_samples,] range: {0, 1, ..., num_samples-1}
        indexing = torch.arange(num_samples, device=device)
    else:
        k = ((4*far).sqrt()-c) / num_samples
        # [num_samples,] range: rand(0,1) + {0, 1, ..., num_samples-1}
        indexing = torch.arange(num_samples, device=device) + torch.rand((*prefix_shape, num_samples), dtype=dtype, device=device)
    t_samples = 0.25 * ((k*indexing+c).square_())
    if not return_dt:
        return t_samples
    else:
        deltas = t_samples.new_zeros(t_samples.shape)
        deltas[..., :-1] = torch.diff(t_samples, dim=-1)
        deltas[..., -1] = deltas[..., -2]
        return t_samples, deltas

if __name__ == "__main__":
    def test_batch_sample_cdf(device=torch.device('cuda')):
        N_rays = 64
        N_pts = 32
        from icecream import ic
        pdfs = torch.rand([N_rays, N_pts], dtype=torch.float, device=device)
        bins = batch_sample_step_linear(
            -torch.randn([N_rays], dtype=torch.float, device=device).abs()-1, 
            torch.randn([N_rays], dtype=torch.float, device=device).abs()+1, 
            N_pts+1, perturb=False)
        batch_sample_pdf(bins, pdfs, 7, perturb=False)
    
    def test_sample_step_wrt_depth(device=torch.device('cuda')):
        from icecream import ic
        import numpy as np
        import matplotlib.pyplot as plt
        from nr3d_lib.utils import check_to_torch
        t = batch_sample_step_wrt_depth(torch.tensor([0.5, 0.1, 0.01], device=device), torch.tensor([250., 250., 250.], device=device), num_samples=512)
        
        # NOTE: This sampling mode is not very effective, mainly because there are too many near-field sampling points.
        t = batch_sample_step_wrt_depth(
            check_to_torch([0.01], device=device), check_to_torch([250.0], device=device), 
            num_samples=4096, perturb=True, prefix_shape=())
        # should be all the same.
        dt = t[..., 2:] - t[..., 1:-1]
        dt_r = dt / t[..., 1:-1]
        ic(dt_r.std(dim=-1))
        fig = plt.figure()
        plt.hist(t.data.cpu().numpy(), bins=np.arange(250), label='sample_density')
        ax = plt.gca()
        ax2 = ax.twinx()
        ax2.plot(t[...,1:-1].data.cpu().numpy(), dt.data.cpu().numpy(), 'brown', label='dt')
        ax2.plot(t[...,1:-1].data.cpu().numpy(), dt_r.data.cpu().numpy(), 'green', label='dt_r')
        fig.legend(loc='center right', bbox_to_anchor=(1, 0.5), bbox_transform=ax.transAxes)
        plt.show()
    
    def test_sample_step_wrt_depth_clamp(device=torch.device('cuda')):
        from icecream import ic
        from math import ceil
        def sample_step_wrt_depth_clamp_min_v1(
            near: torch.Tensor, far: torch.Tensor, num_samples: int, 
            prefix_shape=(), dtype=None, device=None, perturb=False,
            min_step_size: float=0.01):
            """
            batch_sample_step_wrt_depth with given `min_step_size`
            Behaves terrible when `num_samples` is small or `min_step_size` is too small.
            """
            near, far = near.expand(*prefix_shape,1), far.expand(*prefix_shape,1)
            
            num_uniform0 = (((far-near)/10.+near)/min_step_size).clamp_max_(int(0.2*num_samples)).max().ceil().item()
            t_begin_cone0 = near + num_uniform0*min_step_size
            t_begin_cone = torch.maximum(far / ((1+min_step_size / t_begin_cone0)**(num_samples-num_uniform0)), near+num_uniform0*min_step_size)
            
            # This is the major part where this function is compromised, when `min_step_size` and `num_samples` are too small combined.
            num_uniform = int(min( ceil( ((t_begin_cone-near) / min_step_size).max().item()) , 0.5*num_samples )) 
            
            num_cone = num_samples - num_uniform
            
            t_samples_uniform = batch_sample_step_linear(near, t_begin_cone, num_uniform, prefix_shape=prefix_shape, perturb=perturb)
            t_samples_cone = batch_sample_step_wrt_depth_unsafe(t_begin_cone, far, num_cone, prefix_shape=prefix_shape, perturb=perturb)
            t_samples = torch.cat([t_samples_uniform, t_samples_cone], dim=-1)
            return t_samples
        t = sample_step_wrt_depth_clamp_min_v1(
            torch.tensor([0.], device=device), torch.tensor([250.], device=device), 1024, 
            min_step_size=0.01, device=device)
        dt = t[..., 1:] - t[..., :-1]
        dt_r = dt / t[..., :-1]
        ic(t)
        
    def test_sample_step_wrt_depth_sqrt(device=torch.device('cuda')):
        import numpy as np
        from icecream import ic
        import matplotlib.pyplot as plt
        from nr3d_lib.utils import check_to_torch
        t = batch_sample_step_wrt_sqrt_depth(check_to_torch(0.5, device=device), check_to_torch(250, device=device), 4096, prefix_shape=())
        dt = t[..., 2:] - t[..., 1:-1]
        dt_r = dt / t[..., 1:-1]
        
        fig = plt.figure()
        plt.hist(t.data.cpu().numpy(), bins=np.arange(250), label='sample_density')
        ax = plt.gca()
        ax2 = ax.twinx()
        ax2.plot(t[...,1:-1].data.cpu().numpy(), dt.data.cpu().numpy(), 'brown', label='dt')
        ax2.plot(t[...,1:-1].data.cpu().numpy(), dt_r.data.cpu().numpy(), 'green', label='dt_r')
        fig.legend(loc='center right', bbox_to_anchor=(1, 0.5), bbox_transform=ax.transAxes)
        plt.show()

    def test_sample_step_linear(device=torch.device('cuda')):
        import numpy as np
        from icecream import ic
        import matplotlib.pyplot as plt
        from nr3d_lib.utils import check_to_torch
        t = batch_sample_step_linear(check_to_torch(0.5, device=device), check_to_torch(250, device=device), 4096, prefix_shape=())
        dt = t[..., 2:] - t[..., 1:-1]
        dt_r = dt / t[..., 1:-1]
        
        fig = plt.figure()
        plt.hist(t.data.cpu().numpy(), bins=np.arange(250), label='sample_density')
        ax = plt.gca()
        ax2 = ax.twinx()
        ax2.plot(t[...,1:-1].data.cpu().numpy(), dt.data.cpu().numpy(), 'brown', label='dt')
        ax2.plot(t[...,1:-1].data.cpu().numpy(), dt_r.data.cpu().numpy(), 'green', label='dt_r')
        fig.legend(loc='center right', bbox_to_anchor=(1, 0.5), bbox_transform=ax.transAxes)
        plt.show()
        
        batch_sample_step_linear(torch.tensor([0.], device=device), torch.tensor([1.], device=device), 2, perturb=True)

    def test_interleave_sample_step_linear(device=torch.device('cuda')):
        from torch.utils.benchmark import Timer
        from nr3d_lib.render.pack_ops import interleave_linstep
        near = -torch.randn([4096, ], dtype=torch.float, device=device).abs_()
        far = torch.randn([4096,], dtype=torch.float, device=device).abs_()
        
        def interleave_sample_step_linear_v1(
            near: torch.Tensor, far: torch.Tensor, 
            step_size: float, min_steps: int, perturb=False):
            num_samples = ((far-near) / step_size).clamp_min_(min_steps).long() # 29 us
            dt = (far-near) / (num_samples-1) # 19 us
            t_samples, ridx = interleave_linstep(near, num_samples, dt) # 40 us
            return t_samples, ridx
        
        t_samples1, ridx1 = interleave_sample_step_linear_v1(near, far, 0.01, 2)
        t_samples, deltas, ridx, pack_infos = interleave_sample_step_linear(near, far, 0.01, 2)
        # print(torch.allclose(t_samples1, t_samples, atol=1.0e-4))
        # print(torch.allclose(ridx1, ridx))
        
        # 128 threads: 90 us
        # 1024 threads: 409.56 us
        print(Timer(
            stmt="interleave_sample_step_linear_v1(near, far, 0.01, 2)",
            globals={'interleave_sample_step_linear_v1':interleave_sample_step_linear_v1, 'near': near, 'far': far}
        ).blocked_autorange())

        # 128 threads: 137 us
        # 1024 threads: 459.78 us
        print(Timer(
            stmt="interleave_sample_step_linear(near, far, 0.01, 2)",
            globals={'interleave_sample_step_linear':interleave_sample_step_linear, 'near': near, 'far': far}
        ).blocked_autorange())

        # 128 threads: 176 us
        print(Timer(
            stmt="interleave_sample_step_linear(near, far, 0.01, 2, perturb=True)",
            globals={'interleave_sample_step_linear':interleave_sample_step_linear, 'near': near, 'far': far}
        ).blocked_autorange())

    def test_interleave_sample_step_linear_in_packed_segments(device=torch.device('cuda')):
        import numpy as np
        import matplotlib.pyplot as plt
        from torch.utils.benchmark import Timer
        from nr3d_lib.render.pack_ops import get_pack_infos_from_first
        near = torch.tensor([0.5], dtype=torch.float, device=device)
        far = torch.tensor([250.], dtype=torch.float, device=device)
        seg_entry = torch.tensor([1.0, 10.0, 90.0,  200.0], dtype=torch.float, device=device)
        seg_exit =  torch.tensor([5.0, 14.0, 100.0, 260.0], dtype=torch.float, device=device)
        seg_pack_indices = torch.tensor([0], dtype=torch.long, device=device)
        seg_pack_infos = get_pack_infos_from_first(seg_pack_indices, seg_entry.numel())
        t_samples, deltas, sidx, ridx, pack_infos = interleave_sample_step_linear_in_packed_segments(
            near, far, seg_entry, seg_exit, seg_pack_infos, step_size=0.15)
        fig = plt.figure()
        plt.hist(t_samples.flatten().data.cpu().numpy(), bins=np.arange(250), label='sample_density')
        ax = plt.gca()
        ax2 = ax.twinx()
        ax2.plot(t_samples.data.cpu().numpy(), deltas.data.cpu().numpy(), 'rx-', label='dt')
        ax2.plot(t_samples.data.cpu().numpy(), (deltas.flatten() / t_samples.flatten()).data.cpu().numpy(), 'gx--', label='dt_r')
        fig.legend(loc='center right', bbox_to_anchor=(1, 0.5), bbox_transform=ax.transAxes)
        plt.show()

    def test_noise(device=torch.device('cuda')):
        from torch.utils.benchmark import Timer
        a = torch.rand([4096, 512], device=device)
        delta = a.diff(dim=-1)
        # 13.02 us
        print(Timer(
            stmt="torch.rand([4096, 512], device=device)",
            globals={'device':device}
        ).blocked_autorange())
        # 23.10 us
        print(Timer(
            stmt="a.diff(dim=-1)",
            globals={'a':a}
        ).blocked_autorange())

    def test_packed_sample_cdf(device=torch.device('cuda')):
        from torch.utils.benchmark import Timer
        from nr3d_lib.render.volume_graphics import packed_tau_to_vw
        from nr3d_lib.render.pack_ops import packed_searchsorted, interleave_linspace
        num_to_sample = 40
        # cdfs = torch.tensor([0.1, 0.6, 0.9, 1.0, 0.8, 1.0, 0.0, 0.5, 1.0], device=device, dtype=torch.float)

        # NOTE: The start (cdf=0) and end (cdf=1) as well as their corresponding start and end bin points must be included, 
        #           so that invert cdf sampling can correctly be conducted in all intervals.
        #       This implies that:
        #           1. When calculating `cdf` from `pdf` using cumsum, `exclusive=True` must be used, which will produce a leading zero.
        #               Or to put it another way, the `cdf` of the current point comes from the sumation of all previous points and not including current point's pdf.
        #           2. The `cdf` needs to pack-wise normalized so that the trailing value is 1. 
        #               (This is already the case in common practice of upsampling)
        cdfs = torch.tensor([0.0,0.1,0.6,0.9,1.0,  0.0,0.8,1.0,  0.0,0.0,0.5,1.0], device=device, dtype=torch.float)
        bins = torch.tensor([0.0,0.1,0.2,0.3,0.4,  0.0,0.1,0.2,  0.0,0.1,0.2,0.3], device=device, dtype=torch.float)
        n_per_pack = torch.tensor([5, 3, 4], dtype=torch.long, device=device)
        pack_infos = get_pack_infos_from_n(n_per_pack)
        u = torch.linspace(0., 1., num_to_sample+2, device=device, dtype=torch.float)[1:-1].expand([3, num_to_sample]).contiguous()
        # NOTE: to check, the expected behavior of each pack:
        #       `pack[0]`: Most points are between 0.1 and 0.2, very few points are between 0.2 and 0.3.
        #       `pack[1]`: Most points are between 0 and 0.1.
        #       `pack[2]`: No points are between 0 and 0.1, equal number of points are between 0.1 to 0.2 and 0.2 to 
        t_samples, pidx1 = packed_invert_cdf(bins, cdfs, u, pack_infos)
        pidx2 = packed_searchsorted(cdfs, u, pack_infos)
        print(torch.allclose(pidx1 - pack_infos[:,0].unsqueeze(-1), pidx2 - pack_infos[:,0].unsqueeze(-1)))
        
        #------------------------- Benchmark
        num_to_sample = 4
        n_per_pack = torch.randint(32,96, [4096], device=device)
        pack_infos = get_pack_infos_from_n(n_per_pack)
        starts = -torch.randn([4096,], device=device).abs_()
        ends = torch.randn([4096,], device=device).abs_()
        bins, ridx = interleave_linspace(starts, ends, n_per_pack, return_idx=True)
        deltas = packed_diff(bins, pack_infos, pack_last_fill=((ends-starts)/(n_per_pack-1)))
        
        last_inds = n_per_pack.cumsum(0) - 1
        sigma = torch.empty_like(bins, dtype=torch.float, device=device).uniform_(0, 100)
        tau = sigma * deltas
        vw = packed_tau_to_vw(tau, pack_infos)
        cdfs = packed_cumsum(vw, pack_infos, exclusive=True)
        # normalize
        cdfs = packed_div(cdfs, cdfs[last_inds], pack_infos)
        u = torch.rand([4096,num_to_sample], device=device)
        
        pidx2 = packed_searchsorted(cdfs, u, pack_infos)
        # 9 us, for 256k pts
        print(Timer(
            stmt="packed_searchsorted(cdfs, u, pack_infos)",
            globals={'packed_searchsorted':packed_searchsorted, 'cdfs':cdfs, 'u':u, 'pack_infos':pack_infos}
        ).blocked_autorange())
        
        t_samples, pidx = packed_invert_cdf(bins, cdfs, u, pack_infos)
        print(torch.allclose(pidx, pidx2))
        
        # 12.29 us, for 4 sample @ 256k pts
        print(Timer(
            stmt="packed_invert_cdf(bins, cdfs, u, pack_infos)",
            globals={'packed_invert_cdf':packed_invert_cdf, 'bins':bins, 'cdfs': cdfs, 'u': u, 'pack_infos': pack_infos}
        ).blocked_autorange())
        
        #---------------------------- Function
        # 32 us, for 8 sample @ 256k pts
        t_samples, pidx = packed_sample_cdf(bins, cdfs, pack_infos, num_to_sample=8)
        print(Timer(
            stmt="packed_sample_cdf(bins, cdfs, pack_infos, num_to_sample=8)",
            globals={'packed_sample_cdf':packed_sample_cdf, 'bins':bins, 'cdfs': cdfs, 'pack_infos': pack_infos}
        ).blocked_autorange())
        
        # 80 us, for 8 sample @ 256k pts
        t_samples, pidx = packed_sample_cdf(bins, cdfs, pack_infos, num_to_sample=8, perturb=True)
        print(Timer(
            stmt="packed_sample_cdf(bins, cdfs, pack_infos, num_to_sample=8, perturb=True)",
            globals={'packed_sample_cdf':packed_sample_cdf, 'bins':bins, 'cdfs': cdfs, 'pack_infos': pack_infos}
        ).blocked_autorange())

    def some_backups():
        def get_dists_on_partially_valid_t(t_all: torch.Tensor, last_dist_val: float = 1e10) -> torch.Tensor:
            """
            calculate distance on t_vals that is partially valid (and partially invalid; 
            in this case, the invalid values are torch.isinf(t_all)
            """
            device = t_all.device
            
            dists = torch.zeros_like(t_all, device=device)
                                    
            invalid_mask = torch.isinf(t_all)
            valid_mask = ~invalid_mask
            
            # NOTE: =True, when there is at least one valid point along rays
            once_valid_raymask = valid_mask.sum(dim=-1) > 0
            
            # NOTE: only operate on rays that has least one valid point 
            invalid_mask = invalid_mask[once_valid_raymask]
            
            # NOTE: =True, when there is at least one invalid point along rays (rays of at least one valid point along the ray)
            once_invalid_raymask = invalid_mask.sum(dim=-1) > 0
            
            # NOTE: the mask operation's output must be 2-D, and max(dim=-1) make output exactly a 1-D vector
            first_invalid_idx = invalid_mask[once_invalid_raymask].max(dim=-1).indices
            
            # NOTE: must be 2-D
            dists_on_once_valid_ray = torch.zeros_like(t_all[once_valid_raymask], device=device)
            dists_on_once_valid_ray[:, :-1] = t_all[once_valid_raymask][:, 1:] - t_all[once_valid_raymask][:, :-1]
            
            # NOTE: the most meaningful code: fill the last but one valid dist with `last_dist_val` (original dist is   inf - some_val = inf  )
            # NOTE: [first_invalid_idx-1] index is only illegal on rays that contains at least one valid t (which is why 'dists_on_once_valid_ray' is needed)
            dists_on_once_valid_ray[once_invalid_raymask] = \
                dists_on_once_valid_ray[once_invalid_raymask].index_fill_(dim=1, index=first_invalid_idx-1, value=last_dist_val)
            dists_on_once_valid_ray[torch.isnan(dists_on_once_valid_ray)] = 0
            
            dists[once_valid_raymask] = dists_on_once_valid_ray
            
            return dists

    test_batch_sample_cdf()
    # test_noise()
    # test_sample_step_wrt_depth()
    # test_sample_step_wrt_depth_clamp()
    # test_sample_step_linear()
    # test_sample_step_wrt_depth_sqrt()
    # test_interleave_sample_step_linear()
    # test_interleave_sample_step_linear_in_packed_segments()
    test_packed_sample_cdf()
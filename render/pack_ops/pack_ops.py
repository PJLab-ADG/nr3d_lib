"""
@file   pack_ops.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Pack-wise operation. Extends pytorch ops for packed tensors which has inequal batch data sizes.
"""

from numbers import Number
from typing import Tuple, Union

import torch
from torch.autograd.function import once_differentiable

import nr3d_lib_bindings._pack_ops as _backend

__all__ = [
    #---- packed sort
    'packed_sort_inplace', 
    'packed_sort', 
    #---- packed searchsorted
    'packed_searchsorted', 
    'packed_searchsorted_packed_vals', 
    #---- packed reduce
    'packed_sum', 
    'packed_mean', 
    #---- per-pack math
    'packed_cumprod', 
    'packed_cumsum', 
    'packed_diff', 
    #---- per-pack rendering 
    'packed_invert_cdf', 
    'packed_alpha_to_vw', 
    'packed_volume_render_compression', 
    #---- packed arithmetic
    'packed_add', 
    'packed_sub', 
    'packed_mul', 
    'packed_div', 
    'packed_matmul', 
    #---- packed logics
    'packed_gt', 
    'packed_geq', 
    'packed_lt', 
    'packed_leq', 
    'packed_eq', 
    'packed_neq', 
    #---- interleave production of packed tensors
    'interleave_arange_simple', 
    'interleave_arange', 
    'interleave_linstep', 
    'interleave_linspace', 
    'interleave_sample_step_wrt_depth_clamped', 
    'interleave_sample_step_wrt_depth_in_packed_segments', 
    #---- merging two sorted packs
    'merge_two_packs_sorted_aligned', 
    'merge_two_packs_sorted_a_includes_b', 
    'merge_two_packs_sorted', 
    #---- merging two batch (into packs)
    'merge_two_batch_a_includes_b', 
    'merge_two_batch', 
    #---- miscs
    'get_pack_infos_from_boundary', 
    'get_pack_infos_from_first', 
    'get_pack_infos_from_n', 
    'get_pack_infos_from_batch', 
    'octree_mark_consecutive_segments', 
    'mark_pack_boundaries', # borrowed from kaolin
    'expand_pack_boundary', 
    'torch_intersect1d_unique', 
]

@torch.no_grad()
def packed_sort_inplace(vals: torch.Tensor, pack_infos: torch.LongTensor, return_idx=True) -> torch.Tensor:
    # return _backend.packed_sort_thrust(vals, pack_infos, return_idx)
    return _backend.packed_sort_qsort(vals.contiguous(), pack_infos, return_idx)

def packed_sort(vals: torch.Tensor, pack_infos: torch.LongTensor):
    indices = packed_sort_inplace(vals.data.clone(), pack_infos, return_idx=True)
    return vals[indices], indices

@torch.no_grad()
def packed_searchsorted(bins: torch.Tensor, vals: torch.Tensor, pack_infos: torch.LongTensor) -> torch.Tensor:
    # Search a batch (vals) in a sorted pack (bins)
    return _backend.packed_searchsorted(bins.contiguous(), vals.contiguous(), pack_infos)

@torch.no_grad()
def packed_searchsorted_packed_vals(bins: torch.Tensor, pack_infos: torch.LongTensor, vals: torch.Tensor, u_pack_infos: torch.Tensor) -> torch.Tensor:
    # Search a pack (vals,u_pack_infos) in a sorted pack (bins)
    return _backend.packed_searchsorted_packed_vals(bins.contiguous(), pack_infos.contiguous(), vals.contiguous(), u_pack_infos)

@torch.no_grad()
def packed_invert_cdf(bins: torch.Tensor, cdfs: torch.Tensor, u_vals: torch.Tensor, pack_infos: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
    samples, bin_idx = _backend.packed_invert_cdf(bins.contiguous(), cdfs.contiguous(), u_vals.contiguous(), pack_infos)
    return samples, bin_idx

class PackedSum(torch.autograd.Function):
    # Use packed kernels and remove everytime .nonzero() calculation
    @staticmethod
    def forward(ctx, feats, pack_infos):
        if ctx.needs_input_grad[0]: # feats
            ctx.save_for_backward(pack_infos)
        return _backend.packed_sum(feats, pack_infos)
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        pack_infos = ctx.saved_tensors[0]
        grad_feats = None
        if ctx.needs_input_grad[0]:
            grad_feats = grad_output.repeat_interleave(pack_infos[...,1], dim=0)
        return grad_feats, None
def packed_sum(feats: torch.Tensor, pack_infos: torch.LongTensor) -> torch.Tensor:
    if feats.requires_grad:
        return PackedSum.apply(feats.contiguous(), pack_infos)
    else:
        return _backend.packed_sum(feats.contiguous(), pack_infos)

def packed_mean(feats: torch.Tensor, pack_infos: torch.LongTensor) -> torch.Tensor:
    return packed_sum(feats, pack_infos) / (pack_infos[:, 1]+1e-8)

class PackedCumprod(torch.autograd.Function):
    # Remove everytime .nonzero() calculation
    @staticmethod
    def forward(ctx, feats, pack_infos, exclusive, reverse):
        prod = _backend.packed_cumprod(feats, pack_infos, exclusive, reverse)
        if ctx.needs_input_grad[0]:  # feats
            ctx.save_for_backward(feats, pack_infos, prod)
            ctx.flags = (exclusive, reverse)
        return prod
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        prod = ctx.saved_tensors
        exclusive, reverse = ctx.flags
        grad_feats = None
        if ctx.needs_input_grad[0]:
            feats, pack_infos, prod = ctx.saved_tensors
            out = _backend.packed_cumsum(prod * grad_output, pack_infos, exclusive, not reverse)
            # Approximate gradient (consistent with TensorFlow)
            grad_feats = out / feats
            grad_feats[grad_feats.isnan()] = 0
        return grad_feats, None, None, None
def packed_cumprod(feats: torch.Tensor, pack_infos: torch.LongTensor, exclusive: bool=False, reverse: bool=False) -> torch.Tensor:
    """ Pack-wise cumulative production

    Args:
        feats (torch.Tensor): [num_packed_pts(, feat_dim)]
        pack_infos (torch.LongTensor): [num_packs, 2]
        exclusive (bool, optional): If true: right-shifted cumprod with and a preceding 1. Defaults to False.
        reverse (bool, optional): If true: reversed order. Defaults to False.

    Returns:
        torch.Tensor: [num_packed_pts(, feat_dim)]
    """
    if feats.requires_grad:
        return PackedCumprod.apply(feats.contiguous(), pack_infos, exclusive, reverse)
    else:
        return _backend.packed_cumprod(feats.contiguous(), pack_infos, exclusive, reverse)

class PackedCumsum(torch.autograd.Function):
    # Remove everytime .nonzero() calculation
    @staticmethod
    def forward(ctx, feats, pack_infos, exclusive, reverse):
        if ctx.needs_input_grad[0]: # feats
            ctx.save_for_backward(pack_infos)
            ctx.flags = (exclusive, reverse)
        cumsum = _backend.packed_cumsum(feats, pack_infos, exclusive, reverse)
        return cumsum
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        pack_infos, = ctx.saved_tensors
        exclusive, reverse = ctx.flags
        cumsum = _backend.packed_cumsum(grad_output.contiguous(), pack_infos, exclusive, not reverse)
        return cumsum, None, None, None
def packed_cumsum(feats: torch.Tensor, pack_infos: torch.LongTensor, exclusive: bool=False, reverse: bool=False) -> torch.Tensor:
    """ Pack-wise cumulative summation

    Args:
        feats (torch.Tensor): [num_packed_pts(, feat_dim)]
        pack_infos (torch.LongTensor): [num_packs, 2]
        exclusive (bool, optional): If true: right-shifted cumsum with and a preceding 0. Defaults to False.
        reverse (bool, optional): If true: reversed order. Defaults to False.

    Returns:
        torch.Tensor: [num_packed_pts(, feat_dim)]
    """
    if feats.requires_grad:
        return PackedCumsum.apply(feats.contiguous(), pack_infos, exclusive, reverse)
    else:
        return _backend.packed_cumsum(feats.contiguous(), pack_infos, exclusive, reverse)

class PackedDiff(torch.autograd.Function):
    # NOTE: Forward diff. diff_i = feat_{i+1} - feat_i
    # Differentiable
    @staticmethod
    def forward(ctx, feats, pack_infos, pack_appends, pack_last_fill):
        if ctx.needs_input_grad[0]: # feats
            ctx.save_for_backward(pack_infos)
            ctx.flags = (pack_appends is not None, pack_last_fill is not None)
        diff = _backend.packed_diff(feats, pack_infos, pack_appends, pack_last_fill)
        return diff
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        has_append, has_last_fill = ctx.flags
        pack_infos, = ctx.saved_tensors
        first_inds, n_per_pack = pack_infos[..., 0], pack_infos[..., 1]
        last_inds = first_inds + n_per_pack - 1
        grad_feat = None
        if ctx.needs_input_grad[0]:
            # [-dL0, dL0-dL1, dL1-dL2, ..., dL(n-2)-dL(n-1)]
            grad_feat = -1 * _backend.packed_backward_diff(grad_output, pack_infos, None, grad_output[first_inds])
            if not has_append:
                second_last = grad_output[last_inds-1]
                grad_feat[last_inds] = torch.where(n_per_pack.view(second_last.shape) > 1, second_last, grad_output.new_tensor([0.]))
        grad_append = None if (not has_append or not ctx.needs_input_grad[2]) else grad_output[last_inds]
        grad_last_fill = None if (not has_last_fill or not ctx.needs_input_grad[3]) else grad_output[last_inds]
        return grad_feat, None, grad_append, grad_last_fill
def packed_diff(feats: torch.Tensor, pack_infos: torch.LongTensor, pack_appends: torch.Tensor=None, pack_last_fill: torch.Tensor=None) -> torch.Tensor:
    if feats.requires_grad:
        delta = PackedDiff.apply(feats.contiguous(), pack_infos, pack_appends, pack_last_fill)
    else:
        delta = _backend.packed_diff(feats.contiguous(), pack_infos, pack_appends, pack_last_fill)
    return delta

class PackedBackwardDiff(torch.autograd.Function):
    # NOTE: Backward diff. diff_i = feat_i - feat_{i-1}
    # Differentiable
    @staticmethod
    def forward(ctx, feats, pack_infos, pack_prepends, pack_fist_fill):
        if ctx.needs_input_grad[0]: # feats
            ctx.save_for_backward(pack_infos)
            ctx.flags = (pack_prepends is not None, pack_fist_fill is not None)
        diff = _backend.packed_backward_diff(feats, pack_infos, pack_prepends, pack_fist_fill)
        return diff
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        has_prepend, has_first_fill = ctx.flags
        pack_infos, = ctx.saved_tensors
        first_inds, n_per_pack = pack_infos[..., 0], pack_infos[..., 1]
        last_inds = first_inds + n_per_pack - 1
        grad_feat = None
        if ctx.needs_input_grad[0]:
            grad_feat = -1 * _backend.packed_diff(grad_output, pack_infos, None, -grad_output[last_inds])
            if not has_prepend:
                second = grad_output[first_inds+1]
                grad_feat[first_inds] = torch.where(n_per_pack.view(second.shape) > 1, -second, grad_output.new_tensor([0.]))
        grad_prepend = None if (not has_prepend or not ctx.needs_input_grad[2]) else -grad_output[first_inds]
        grad_first_fill = None if (not has_first_fill or not ctx.needs_input_grad[3]) else grad_output[first_inds]
        return grad_feat, None, grad_prepend, grad_first_fill

def packed_backward_diff(feats: torch.Tensor, pack_infos: torch.LongTensor, pack_prepends: torch.Tensor=None, pack_first_fill: torch.Tensor=None) -> torch.Tensor:
    if feats.requires_grad:
        delta = PackedBackwardDiff.apply(feats.contiguous(), pack_infos, pack_prepends, pack_first_fill)
    else:
        delta = _backend.packed_backward_diff(feats.contiguous(), pack_infos, pack_prepends, pack_first_fill)
    return delta

class PackedAlphaToVW(torch.autograd.Function):
    # Modified from https://github.com/KAIR-BAIR/nerfacc
    @staticmethod
    def forward(ctx, alphas, pack_infos, early_stop_eps, alpha_thre):
        weights = _backend.packed_alpha_to_vw_forward(alphas, pack_infos, early_stop_eps, alpha_thre, False)[0]
        if ctx.needs_input_grad[0]:  # alphas
            ctx.save_for_backward(alphas, pack_infos, weights)
            ctx.early_stop_eps = early_stop_eps
            ctx.alpha_thre = alpha_thre
        return weights
    @staticmethod
    def backward(ctx, grad_weights):
        early_stop_eps = ctx.early_stop_eps
        alpha_thre = ctx.alpha_thre
        alphas, pack_infos, weights = ctx.saved_tensors
        grad_alphas = _backend.packed_alpha_to_vw_backward(weights, grad_weights, alphas, pack_infos, early_stop_eps, alpha_thre)
        return grad_alphas, None, None, None

def packed_alpha_to_vw(alpha: torch.Tensor, pack_infos: torch.LongTensor, early_stop_eps: float = 1e-4, alpha_thre: float = 0.0) -> torch.Tensor:
    if alpha.requires_grad:
        return PackedAlphaToVW.apply(alpha, pack_infos, early_stop_eps, alpha_thre)
    else:
        return _backend.packed_alpha_to_vw_forward(alpha, pack_infos, early_stop_eps, alpha_thre, False)[0]

@torch.no_grad()
def packed_volume_render_compression(alpha: torch.Tensor, pack_infos: torch.LongTensor, early_stop_eps: float = 1e-4, alpha_thre: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, compact_pack_infos, compact_selector = _backend.packed_alpha_to_vw_forward(alpha, pack_infos, early_stop_eps, alpha_thre, True)
    pidx = compact_selector.nonzero().long()[..., 0]
    nidx_useful = (compact_pack_infos[:, 1] > 0).nonzero()[..., 0]
    compact_pack_infos = compact_pack_infos[nidx_useful].long()
    return nidx_useful, compact_pack_infos, pidx

class PackedAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feats, other, pack_infos):
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            ctx.shape_in = feats.shape
            ctx.shape_other = other.shape
            ctx.save_for_backward(pack_infos)
        return _backend.packed_add(feats, other, pack_infos)
    @staticmethod
    def backward(ctx, grad_out):
        if grad_out is None:
            return None, None, None
        grad_in = None
        grad_other = None
        pack_infos, *_ = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_in = grad_out
        if ctx.needs_input_grad[1]:
            grad_other = PackedSum.apply(grad_out, pack_infos)
        return grad_in, grad_other, None
def packed_add(feats: torch.Tensor, other: torch.Tensor, pack_infos: torch.LongTensor):
    return PackedAdd.apply(feats.contiguous(), other.contiguous(), pack_infos.contiguous())

class PackedSub(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feats, other, pack_infos):
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            ctx.save_for_backward(pack_infos)
        return _backend.packed_sub(feats, other, pack_infos)
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        if grad_out is None:
            return None, None, None
        grad_in = None
        grad_other = None
        pack_infos, *_ = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_in = grad_out
        if ctx.needs_input_grad[1]:
            grad_other = -1 * _backend.packed_sum(grad_out, pack_infos)
        return grad_in, grad_other, None
def packed_sub(feats: torch.Tensor, other: torch.Tensor, pack_infos: torch.LongTensor):
    return PackedSub.apply(feats.contiguous(), other.contiguous(), pack_infos.contiguous())

class PackedMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feats, other, pack_infos):
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            ctx.save_for_backward(feats, other, pack_infos)
        return _backend.packed_mul(feats, other, pack_infos)
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        if grad_out is None:
            return None, None, None
        grad_in = None
        grad_other = None
        feats, other, pack_infos = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_in = _backend.packed_mul(grad_out, other, pack_infos)
        if ctx.needs_input_grad[1]:
            grad_other = _backend.packed_sum(grad_out * feats, pack_infos)
        return grad_in, grad_other, None
def packed_mul(feats: torch.Tensor, other: torch.Tensor, pack_infos: torch.LongTensor):
    return PackedMul.apply(feats.contiguous(), other.contiguous(), pack_infos.contiguous())

class PackedDiv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feats, other, pack_infos):
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            ctx.save_for_backward(feats, other, pack_infos)
        return _backend.packed_div(feats, other, pack_infos)
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        if grad_out is None:
            return None, None, None
        grad_in = None
        grad_other = None
        feats, other, pack_infos = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_in = _backend.packed_div(grad_out, other, pack_infos)
        if ctx.needs_input_grad[1]:
            # -feats/(other * other)
            grad_other = _backend.packed_div(-grad_out*feats, other*other, pack_infos)
            grad_other = _backend.packed_sum(grad_other, pack_infos)
        return grad_in, grad_other, None
def packed_div(feats: torch.Tensor, other: torch.Tensor, pack_infos: torch.LongTensor):
    """ Calculate pack-wise division: feats / other

    Args:
        feats (torch.Tensor): [num_feats(, feat_dim)]
        other (torch.Tensor): [num_packs(, feat_dim)]
        pack_infos (torch.LongTensor): [num_packs, 2]

    Returns:
        torch.Tensor: Pack-wise division resutls
    """
    return PackedDiv.apply(feats.contiguous(), other.contiguous(), pack_infos.contiguous())

def packed_matmul(feats: torch.Tensor, other: torch.Tensor, pack_infos: torch.LongTensor) -> torch.Tensor:
    """ Calculate pack-wise left-multiplication: results = other @ feat

    Args:
        feats (torch.Tensor): [num_feats, feat_dim]
        other (torch.Tensor): [num_packs, out_feat_dim, feat_dim]
        pack_infos (torch.LongTensor): [num_packs, 2]

    Returns:
        torch.Tensor: Pack-wise matmul results
    """
    # return _backend.packed_matmul(feats, other, pack_infos) # NOTE: non-differentiable
    return (feats.unsqueeze(-2) * torch.repeat_interleave(other, pack_infos[:,1], dim=0)).sum(-1)

def packed_gt(feats: torch.Tensor, other: torch.Tensor, pack_infos: torch.LongTensor):
    # return feats > torch.repeat_interleave(other, pack_infos[:,1], dim=0)
    return _backend.packed_gt(feats.contiguous(), other.contiguous(), pack_infos.contiguous())

def packed_geq(feats: torch.Tensor, other: torch.Tensor, pack_infos: torch.LongTensor):
    # return feats >= torch.repeat_interleave(other, pack_infos[:,1], dim=0)
    return _backend.packed_geq(feats.contiguous(), other.contiguous(), pack_infos.contiguous())

def packed_lt(feats: torch.Tensor, other: torch.Tensor, pack_infos: torch.LongTensor):
    # return feats < torch.repeat_interleave(other, pack_infos[:,1], dim=0)
    return _backend.packed_lt(feats.contiguous(), other.contiguous(), pack_infos.contiguous())

def packed_leq(feats: torch.Tensor, other: torch.Tensor, pack_infos: torch.LongTensor):
    # return feats <= torch.repeat_interleave(other, pack_infos[:,1], dim=0)
    return _backend.packed_leq(feats.contiguous(), other.contiguous(), pack_infos.contiguous())

def packed_eq(feats: torch.Tensor, other: torch.Tensor, pack_infos: torch.LongTensor):
    # return feats == torch.repeat_interleave(other, pack_infos[:,1], dim=0)
    return _backend.packed_eq(feats.contiguous(), other.contiguous(), pack_infos.contiguous())

def packed_neq(feats: torch.Tensor, other: torch.Tensor, pack_infos: torch.LongTensor):
    # return feats != torch.repeat_interleave(other, pack_infos[:,1], dim=0)
    return _backend.packed_neq(feats.contiguous(), other.contiguous(), pack_infos.contiguous())

@torch.no_grad()
def interleave_arange_simple(stop: torch.Tensor, return_idx: bool=True) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    ret = _backend.interleave_arange(stop.contiguous(), return_idx)
    return ret if return_idx else ret[0]

@torch.no_grad()
def interleave_arange(start: torch.Tensor, stop: torch.Tensor, step_size: Union[torch.Tensor, Number], return_idx: bool=True):
    num_steps = stop.subtract(start).div(step_size).ceil().long()
    return interleave_linstep(start, num_steps, step_size, return_idx)

@torch.no_grad()
def interleave_linstep(start: torch.Tensor, num_steps: torch.Tensor, step_size: Union[torch.Tensor, Number], return_idx: bool=True):
    ret = _backend.interleave_linstep(start.contiguous(), num_steps.contiguous(), step_size.contiguous() if isinstance(step_size, torch.Tensor) else step_size, return_idx)
    return ret if return_idx else ret[0]

@torch.no_grad()
def interleave_linspace(start: torch.Tensor, stop: torch.Tensor, num_steps: Union[torch.Tensor, Number], return_idx: bool=True):
    step_size = (stop-start)/(num_steps-1)
    num_steps = num_steps if isinstance(num_steps, torch.Tensor) else torch.full(start.shape, num_steps, device=start.device, dtype=torch.long)
    return interleave_linstep(start, num_steps, step_size, return_idx=return_idx)

@torch.no_grad()
def interleave_sample_step_wrt_depth_clamped(
    near: torch.Tensor, far: torch.Tensor, max_steps: int=512,
    dt_gamma: float=0.01, min_step_size: float=0.01, max_step_size: float=1.0, step_size_factor: float=1.0, perturb=False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dt_gamma = dt_gamma * step_size_factor
    min_step_size = min_step_size * step_size_factor
    max_step_size = max_step_size * step_size_factor
    t_samples, deltas, ridx, pack_infos = \
        _backend.interleave_sample_step_wrt_depth_clamped(near, far, max_steps, dt_gamma, min_step_size, max_step_size)
    if not perturb:
        return t_samples, deltas, ridx, pack_infos
    else:
        noise = torch.rand_like(deltas, dtype=deltas.dtype, device=deltas.device)
        # NOTE: Important: add noise to each already-determined interval
        t_samples = torch.addcmul(t_samples, noise, deltas)
        last_inds = pack_infos[...,0]+pack_infos[...,1]-1
        deltas = t_samples.diff(append=t_samples.new_empty([1])).index_put_((last_inds,), deltas[last_inds])
        # deltas = packed_diff(t_samples, pack_indices).index_put_((last_inds,), deltas[last_inds])
        # deltas = packed_diff(t_samples, pack_indices, pack_last_fill=deltas[pack_indices+n_per_pack-1])
        return t_samples, deltas, ridx, pack_infos

@torch.no_grad()
def interleave_sample_step_wrt_depth_in_packed_segments(
    near: Union[torch.Tensor, float], far: Union[torch.Tensor, float], entry: torch.Tensor, exit: torch.Tensor, seg_pack_infos: torch.Tensor,
    max_steps: int=512, dt_gamma: float=0.01, min_step_size: float=0.01, max_step_size: float=1e10, step_size_factor: float=1.0, perturb=False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    
    num_rays = seg_pack_infos.shape[0]
    near = entry.new_full([num_rays], near) if not isinstance(near, torch.Tensor) else near
    far = entry.new_full([num_rays], far) if not isinstance(far, torch.Tensor) else far
    
    dt_gamma = dt_gamma * step_size_factor
    min_step_size = min_step_size * step_size_factor
    max_step_size = max_step_size * step_size_factor
    t_samples, deltas, sidx, ridx, ray_pack_infos = \
        _backend.interleave_sample_step_wrt_depth_in_packed_segments(near, far, entry, exit, seg_pack_infos, max_steps, dt_gamma, min_step_size, max_step_size)
    out_seg_pack_infos = get_pack_infos_from_boundary(mark_pack_boundaries(sidx))
    if not perturb:
        return t_samples, deltas, ridx, ray_pack_infos, sidx, out_seg_pack_infos
    else:
        noise = torch.rand_like(deltas, dtype=deltas.dtype, device=deltas.device)
        # NOTE: Important: add noise to each already-determined interval
        t_samples = torch.addcmul(t_samples, noise, deltas)
        last_inds = ray_pack_infos[...,0] + ray_pack_infos[...,1] - 1
        deltas = t_samples.diff(append=t_samples.new_empty([1])).index_put_((last_inds,), deltas[last_inds])
        return t_samples, deltas, ridx, ray_pack_infos, sidx, out_seg_pack_infos

# from kaolin.render.spc import mark_pack_boundaries
def mark_pack_boundaries(pack_ids):
    return _backend.mark_pack_boundaries_cuda(pack_ids.contiguous()).bool()

@torch.no_grad()
def octree_mark_consecutive_segments(pidx: torch.Tensor, pack_infos: torch.LongTensor, point_hierarchies: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Mark consecutive segments of the ray intersection outputs with a kaolin-SPC.
    mark_start, mark_end = _backend.octree_mark_consecutive_segments(pidx.int().contiguous(), pack_infos, point_hierarchies)
    return mark_start, mark_end

@torch.no_grad()
def torch_intersect1d_unique(t1: torch.Tensor, t2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Jianfei Guo's answer: https://stackoverflow.com/a/74330048/11121534
    """
    assert t1.dim() == t2.dim() == 1, "Requires t1, t2 to be unique 1D Tensors."
    num_t1, num_t2 = t1.numel(), t2.numel()
    u, inv, cnt = torch.unique(torch.cat([t1,t2]), return_counts=True, return_inverse=True)

    inv_t1, inv_t2 = inv[:num_t1].contiguous(), inv[num_t1:].contiguous()
    m_t1, m_t2 = (cnt[inv_t1] == 2), (cnt[inv_t2] == 2)
    inds_t1, inds_t1_exclusive, inds_t2, inds_t2_exclusive = m_t1.nonzero()[..., 0], (~m_t1).nonzero()[..., 0], m_t2.nonzero()[..., 0], (~m_t2).nonzero()[..., 0]

    # intersection = t1[inds_t1] # t2[inds_t2]
    # t1_exclusive = t1[inds_t1_exclusive]
    # t2_exclusive = t2[inds_t2_exclusive]
    return u, inv_t1, inv_t2, inds_t1, inds_t2, inds_t1_exclusive, inds_t2_exclusive

def merge_two_packs_sorted_aligned(
    vals_a: torch.Tensor, pack_infos_a: torch.Tensor, 
    vals_b: torch.Tensor, pack_infos_b: torch.Tensor, 
    b_sorted=True, return_val=False) -> Union[ Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] :
    """
    Merge and sort two sorted packs. (vals_b can be un-sorted)
    NOTE: Requirements:
    - (sorted) vals_a must be sorted; vals_b can be un-sorted
    - (aligned) two packs' nuggets must be aligned.
    """
    pidx_a, pidx_b, pack_infos = _backend.try_merge_two_packs_sorted_aligned(vals_a, pack_infos_a, vals_b, pack_infos_b, b_sorted)
    if return_val:
        val = vals_a.new_empty([vals_a.numel() + vals_b.numel()])
        val[pidx_a], val[pidx_b] = vals_a, vals_b # Differetiable indexing, if vals has grad
        return val, pack_infos
    else:
        return pidx_a, pidx_b, pack_infos

def merge_two_packs_sorted_a_includes_b(
    vals_a: torch.Tensor, pack_infos_a: torch.Tensor, nidx_a: torch.Tensor, 
    vals_b: torch.Tensor, pack_infos_b: torch.Tensor, nidx_b: torch.Tensor, 
    b_sorted=True, return_val=False) -> Union[ Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Merge and sort two sorted packs. (vals_b can be un-sorted)
    NOTE: Requirements:
    - (sorted) vals_a must be sorted; vals_b can be un-sorted
    - (inclusive) `nidx_a` needs not be aligned with `nidx_b`, but should includes `nidx_b`
    - nidx_a, nidx_b must be sorted & unique (no duplicated values)
    """
    
    device = vals_a.device
    assert vals_a.dim() == vals_b.dim() == 1, "Expect batched inputs with dim()==1"
    assert (pack_infos_a.shape[0] == nidx_a.numel()) and (pack_infos_b.shape[0] == nidx_b.numel()), "pack_infos and their nugget indices should have the same length"

    n_a, n_b = nidx_a.numel(), nidx_b.numel()
    
    if (n_a == n_b) and torch.equal(nidx_a, nidx_b):
        # Merge two aligned packs
        return merge_two_packs_sorted_aligned(vals_a, pack_infos_a, vals_b, pack_infos_b, b_sorted=b_sorted, return_val=return_val)
    else:
        with torch.no_grad():
            inds_b_in_a = torch.searchsorted(nidx_a, nidx_b)
            mask = torch.ones([n_a], dtype=torch.bool, device=device)
            mask[inds_b_in_a] = False
            inds_a_exlusive = mask.nonzero().long()[..., 0]
            
            n_per_pack = pack_infos_a[:,1].contiguous()
            n_per_pack.index_add_(0, inds_b_in_a, pack_infos_b[:,1])
            pack_infos = get_pack_infos_from_n(n_per_pack)

            pidx_a = pack_infos_a.new_full([vals_a.numel()], -1)

            # Union part of two packs
            i_a_union = interleave_linstep(pack_infos_a[inds_b_in_a,0], pack_infos_a[inds_b_in_a,1], 1, return_idx=False)
            vals_a_union = vals_a[i_a_union]
            pinfo_a_union = get_pack_infos_from_n(pack_infos_a[inds_b_in_a,1].contiguous())
            
            pidx_a_union, pidx_b_union, pack_infos_union = merge_two_packs_sorted_aligned(vals_a_union, pinfo_a_union, vals_b, pack_infos_b, b_sorted=b_sorted)
            offset_union = pack_infos[inds_b_in_a, 0] - pack_infos_union[:, 0] # From union's local pack indices to global merged pack indices
            pidx_a[i_a_union] = packed_add(pidx_a_union, offset_union, pinfo_a_union) 
            pidx_b = packed_add(pidx_b_union, offset_union, pack_infos_b)
            
            # Distinct part of `a`
            pidx_a_exclusive, _nidx_a_exclusive = interleave_arange_simple(pack_infos_a[inds_a_exlusive,1], return_idx=True)
            idx_a_exclusive = inds_a_exlusive[_nidx_a_exclusive]
            pidx_a[pidx_a_exclusive + pack_infos_a[idx_a_exclusive, 0]] = pidx_a_exclusive + pack_infos[idx_a_exclusive, 0]
        
        if return_val:
            val = vals_a.new_zeros([vals_a.numel() + vals_b.numel()])
            val[pidx_a], val[pidx_b] = vals_a, vals_b
            return val, pack_infos
        else:
            return pidx_a, pidx_b, pack_infos

def merge_two_packs_sorted(
    vals_a: torch.Tensor, pack_infos_a: torch.Tensor, nidx_a: torch.Tensor, 
    vals_b: torch.Tensor, pack_infos_b: torch.Tensor, nidx_b: torch.Tensor, 
    return_val=False) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Merge and sort two sorted packs (two packs need not to be aligned).
    
    NOTE: 
    - nidx_a, nidx_b must be sorted & unique (no duplicated values)
    - vals_a, vals_b must be sorted
    
    PERF:
    - 1.1 ms @ 4k packs & 400k pts
    """
    if (nidx_a.numel() == nidx_b.numel()) and torch.equal(nidx_a, nidx_b):
        return merge_two_packs_sorted_aligned(vals_a, pack_infos_a, vals_b, pack_infos_b, b_sorted=True, return_val=return_val)
    else:
        with torch.no_grad():
            # 368 us @ 4k packs
            u, inv_a, inv_b, inds_a, inds_b, inds_a_exlusive, inds_b_exclusive = torch_intersect1d_unique(nidx_a, nidx_b)

            n_per_pack = pack_infos_a.new_zeros([u.numel()])
            n_per_pack[inv_a] += pack_infos_a[:,1]
            n_per_pack[inv_b] += pack_infos_b[:,1]
            pack_infos = get_pack_infos_from_n(n_per_pack)
            
            pidx_a = pack_infos_a.new_full([vals_a.numel()], -1)
            pidx_b = pack_infos_b.new_full([vals_b.numel()], -1)

            # Union part of two
            if inds_a.numel() > 0:
                i_a_union, _nidx_a_union = interleave_linstep(pack_infos_a[inds_a,0], pack_infos_a[inds_a,1], 1, return_idx=True)
                vals_a_union = vals_a[i_a_union]
                pinfo_a_union = get_pack_infos_from_n(pack_infos_a[inds_a,1].contiguous())
                
                i_b_union, _nidx_b_union = interleave_linstep(pack_infos_b[inds_b,0], pack_infos_b[inds_b,1], 1, return_idx=True)
                vals_b_union = vals_b[i_b_union]
                pinfo_b_union = get_pack_infos_from_n(pack_infos_b[inds_b,1].contiguous())

                # 200 us @ 4k nuggets & 400k pts
                pidx_a_union, pidx_b_union, pack_infos_union = merge_two_packs_sorted_aligned(vals_a_union, pinfo_a_union, vals_b_union, pinfo_b_union)
                
                pack_inds_union = inv_a[inds_a] # The same with inv_b[inds_b]
                offset_union = pack_infos[pack_inds_union, 0] - pack_infos_union[:, 0] # From union's local pack indices to global merged pack indices
                
                pidx_a[i_a_union] = pidx_a_union + offset_union[_nidx_a_union]
                pidx_b[i_b_union] = pidx_b_union + offset_union[_nidx_b_union]
            
            # Distinct part of `a`
            if inds_a_exlusive.numel() > 0:
                pinfo_a_exclusive = pack_infos_a[inds_a_exlusive]
                pidx_a_exclusive, _nidx_a_exclusive = interleave_arange_simple(pinfo_a_exclusive[:,1], return_idx=True)
                pidx_a[pidx_a_exclusive + pinfo_a_exclusive[_nidx_a_exclusive, 0]] = pidx_a_exclusive + pack_infos[inv_a[inds_a_exlusive][_nidx_a_exclusive], 0]
            
            # Distinct part of `b`
            if inds_b_exclusive.numel() > 0:
                pinfo_b_exclusive = pack_infos_b[inds_b_exclusive]
                pidx_b_exclusive, _nidx_b_exclusive = interleave_arange_simple(pinfo_b_exclusive[:,1], return_idx=True)
                pidx_b[pidx_b_exclusive + pinfo_b_exclusive[_nidx_b_exclusive, 0]] = pidx_b_exclusive + pack_infos[inv_b[inds_b_exclusive][_nidx_b_exclusive], 0]
            
        if return_val:
            val = vals_a.new_zeros([vals_a.numel() + vals_b.numel()])
            val[pidx_a], val[pidx_b] = vals_a, vals_b
            return val, pack_infos
        else:
            return pidx_a, pidx_b, pack_infos

def merge_two_batch_a_includes_b(vals_a: torch.Tensor, nidx_a: torch.Tensor, vals_b: torch.Tensor, nidx_b: torch.Tensor, a_sorted=True, return_val=False):
    # NOTE: requires nidx_a, nidx_b to be unique & sorted
    device = vals_a.device
    assert vals_a.dim() == vals_b.dim() == 2, "Expect batched inputs with dim()==2"
    assert (vals_a.shape[0] == nidx_a.numel()) and (vals_b.shape[0] == nidx_b.numel()), "Values and their nugget indices should have the same length"
    
    n_a, n_b = nidx_a.numel(), nidx_b.numel()
    bds_a, bds_b = vals_a.shape[-1], vals_b.shape[-1]
    
    if (n_a == n_b) and torch.equal(nidx_a, nidx_b):
        # Merge two aligned batches
        vals, sorted_indices = torch.cat([vals_a, vals_b], -1).sort(-1)
        n_per_pack = nidx_a.new_full([n_a], bds_a+bds_b)
        pack_infos = get_pack_infos_from_n(n_per_pack)
        if return_val:
            return vals, pack_infos
        else:
            idx_union = sorted_indices.argsort(-1)
            pidx_a = pack_infos[:, 0].unsqueeze(-1) + idx_union[:, :bds_a]
            pidx_b = pack_infos[:, 0].unsqueeze(-1) + idx_union[:, bds_a:]
            return pidx_a, pidx_b, pack_infos
    else:
        inds_b_in_a = torch.searchsorted(nidx_a, nidx_b)
        mask = torch.ones([n_a], dtype=torch.bool, device=device)
        mask[inds_b_in_a] = False
        inds_a_exlusive = mask.nonzero().long()[..., 0]
        
        n_per_pack = nidx_a.new_full([n_a], bds_a)
        n_per_pack.index_fill_(0, inds_b_in_a, bds_a + bds_b)
        pack_infos = get_pack_infos_from_n(n_per_pack)

        # Union part of two batches
        pidx_a, pidx_b = nidx_a.new_full(vals_a.shape, -1), nidx_a.new_full(vals_b.shape, -1)
        idx_union = torch.cat([vals_a[inds_b_in_a], vals_b], dim=1).argsort(-1).argsort(-1)

        pack_indices_union = pack_infos[inds_b_in_a, 0]
        pidx_a[inds_b_in_a] = pack_indices_union.unsqueeze(-1) + idx_union[:, :bds_a]
        pidx_b = pack_indices_union.unsqueeze(-1) + idx_union[:, bds_a:]
        
        # Distinct part of batch `a`
        if inds_a_exlusive.numel() > 0:
            pidx_a[inds_a_exlusive] = pack_infos[inds_a_exlusive, 0].unsqueeze(-1) + \
                (torch.arange(bds_a, device=device).unsqueeze_(0) if a_sorted else vals_a[inds_a_exlusive].argsort(-1).argsort(-1))

        if return_val:
            vals = vals_a.new_empty([vals_a.numel() + vals_b.numel()])
            vals[pidx_a], vals[pidx_b] = vals_a, vals_b
            return vals, pack_infos
        else:
            return pidx_a, pidx_b, pack_infos

def merge_two_batch(vals_a: torch.Tensor, nidx_a: torch.Tensor, vals_b: torch.Tensor, nidx_b: torch.Tensor):
    raise NotImplementedError

@torch.no_grad()
def expand_pack_boundary(pack_boundary: torch.Tensor, num_samples: int):
    bigpack_boundary = torch.zeros(pack_boundary.shape[0]*num_samples, device=pack_boundary.device, dtype=torch.bool)
    bigpack_boundary[pack_boundary.nonzero().long() * num_samples] = 1
    return bigpack_boundary

@torch.no_grad()
def get_pack_infos_from_boundary(boundary: torch.Tensor):
    first_inds = boundary.nonzero().long()[..., 0]
    return get_pack_infos_from_first(first_inds, boundary.numel()) 

@torch.no_grad()
def get_pack_infos_from_first(first_inds: torch.Tensor, numel: int):
    return torch.stack([first_inds, first_inds.diff(append=first_inds.new_tensor([numel]))], 1)

@torch.no_grad()
def get_pack_infos_from_n(n_per_pack: torch.Tensor):
    return torch.stack([n_per_pack.cumsum(0)-n_per_pack, n_per_pack], 1)

@torch.no_grad()
def get_pack_infos_from_batch(n_batches: int, batch_data_size: int, device=torch.device('cuda')):
    return torch.stack([
        torch.arange(0, n_batches * batch_data_size, batch_data_size, device=device, dtype=torch.long), 
        torch.full([n_batches], batch_data_size, device=device, dtype=torch.long)], 1)

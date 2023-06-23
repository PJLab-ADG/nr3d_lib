"""
Borrowed from CO3D & tiny-cuda-nn
"""

__all__ = [
    'mse_loss', 
    'l1_loss', 
    'smooth_l1_loss', 
    'relative_l1_loss', 
    'mape_loss', 
    'smape_loss', 
    'l2_loss', 
    'relative_l2_loss', 
    'relative_l2_luminance_loss', 
    'huber_loss'
]

from typing import Optional

import torch
import torch.nn.functional as F

from .utils import reduce

def mse_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: Optional[torch.Tensor] = None, 
    reduction = 'mean'
) -> torch.Tensor:
    """
    Calculates the mean square error between tensors `x` and `y`.
    """
    if mask is None:
        return F.mse_loss(x, y, reduction=reduction)
    else:
        loss = F.mse_loss(x, y, reduction='none')
        return reduce(loss, mask, reduction)

def l1_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction = 'mean'
) -> torch.Tensor:
    """
    Calculates the mean square error between tensors `x` and `y`.
    """
    if mask is None:
        return F.l1_loss(x, y, reduction=reduction)
    else:
        loss = F.l1_loss(x, y, reduction='none')
        return reduce(loss, mask, reduction)

def smooth_l1_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction = 'mean'
) -> torch.Tensor:
    """
    Calculates the mean square error between tensors `x` and `y`.
    """
    if mask is None:
        return F.smooth_l1_loss(x, y, reduction=reduction)
    else:
        loss = F.smooth_l1_loss(x, y, reduction='none')
        return reduce(loss, mask, reduction)

def relative_l1_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction = 'mean'
) -> torch.Tensor:
    """
    Relative L1 loss normalized by the network prediction.
    """
    loss = (x - y).abs() / (x.abs() + 1.0e-2)
    return reduce(loss, mask, reduction)

def mape_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction = 'mean'
) -> torch.Tensor:
    """
    Mean absolute percentage error (MAPE). 
    The same as Relative L1, but normalized by the target.
    """
    loss = (x - y).abs() / (y.abs() + 1.0e-2)
    return reduce(loss, mask, reduction)

def smape_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction = 'mean'
) -> torch.Tensor:
    """
    Symmetric mean absolute percentage error (SMAPE). 
    The same as Relative L1, but normalized by the mean of the prediction and the target.
    """
    loss = (x - y).abs() / (0.5 * (x.abs() + y.abs()) + 1.0e-2)
    return reduce(loss, mask, reduction)

def l2_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction = 'mean'
) -> torch.Tensor:
    """
    Standard L2 loss.
    """
    return mse_loss(x, y, mask=mask, reduction=reduction)

def relative_l2_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction = 'mean'
) -> torch.Tensor:
    """
    Relative L2 loss normalized by the network prediction [Lehtinen et al. 2018].
    """
    loss = (x-y)**2 / (x**2+1e-2)
    return reduce(loss, mask, reduction)

def relative_l2_luminance_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction = 'mean'
) -> torch.Tensor:
    """
    Same as relative_l2, but normalized by the luminance of the network prediction. 
    Only applicable when network prediction is RGB.
    Neural Radiance Caching [Müller et al. 2021].
    """
    assert x.shape[-1] == 3
    luminance = (x * x.new_tensor([0.299, 0.587, 0.114])).sum(dim=-1,keepdim=True)
    loss = (x-y)**2 / (luminance**2+1e-2)
    return reduce(loss, mask, reduction)

def safe_sqrt(A: torch.Tensor, eps: float = float(1e-4)) -> torch.Tensor:
    """
    Performs safe differentiable sqrt
    """
    return (torch.clamp(A, float(0)) + eps).sqrt()

def huber(dfsq: torch.Tensor, scaling: float = 0.03) -> torch.Tensor:
    """
    Calculates the huber function of the input squared error `dfsq`.
    The function smoothly transitions from a region with unit gradient
    to a hyperbolic function at `dfsq=scaling`.
    """
    loss = (safe_sqrt(1 + dfsq / (scaling * scaling), eps=1e-4) - 1) * scaling
    return loss

def huber_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    alpha: float = 0.1, 
    reduction = 'mean'
) -> torch.Tensor:
    diff = x - y
    abs_diff = diff.abs()
    sq = (0.5/alpha) * (diff**2)
    loss = torch.where(abs_diff > alpha, abs_diff - 0.5 * alpha, sq)
    return reduce(loss, mask, reduction)
"""
@file   utils.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Utilities for loss calculations
"""

import torch
import numpy as np
from typing import Literal, Union

def reduce(
    loss: Union[torch.Tensor, np.ndarray], 
    mask: Union[torch.Tensor, np.ndarray] = None, 
    reduction: Literal['mean', 'mean_in_mask', 'sum', 'max', 'min', 'none']='mean'):

    if mask is not None:
        if mask.dim() == loss.dim() - 1:
            mask = mask.view(*loss.shape[:-1], 1).expand_as(loss)
        assert loss.dim() == mask.dim(), f"Expects loss.dim={loss.dim()} to be equal to mask.dim()={mask.dim()}"
    
    if reduction == 'mean':
        return loss.mean() if mask is None else (loss * mask).mean()
    elif reduction == 'mean_in_mask':
        return loss.mean() if mask is None else (loss * mask).sum() / mask.sum().clip(1e-5)
    elif reduction == 'sum':
        return loss.sum() if mask is None else (loss * mask).sum()
    elif reduction == 'max':
        return loss.max() if mask is None else loss[mask].max()
    elif reduction == 'min':
        return loss.min() if mask is None else loss[mask].min()
    elif reduction == 'none':
        return loss if mask is None else loss * mask
    else:
        raise RuntimeError(f"Invalid reduction={reduction}")
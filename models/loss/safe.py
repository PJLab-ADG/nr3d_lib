"""
@file   safe.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Custom clipped BCE and mse losses
"""

__all__ = [
    'SafeBCE', 
    'safe_binary_cross_entropy', 
    'ClippedMSE', 
    'safe_mse_loss'
]

from numbers import Number
import numpy as np
from typing import Literal, Tuple, Union

import torch
from torch import autograd

from nr3d_lib.models.loss.utils import reduce

class SafeBCE(autograd.Function):
    """ Perform clipped BCE without disgarding gradients (preserve clipped gradients)
        This function is equivalent to torch.clip(x, limit), 1-limit) before BCE, 
        BUT with grad existing on those clipped values.
        
    NOTE: pytorch original BCELoss implementation is equivalent to limit = np.exp(-100) here.
        see doc https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
    """
    @staticmethod
    def forward(ctx, x, y, limit):
        assert (torch.where(y!=1, y+1, y)==1).all(), u'target must all be {0,1}'
        ln_limit = ctx.ln_limit = np.log(limit)
        # ctx.clip_grad = clip_grad
        
        # NOTE: for example, torch.log(1-torch.tensor([1.000001])) = nan
        x = torch.clip(x, 0, 1)
        y = torch.clip(y, 0, 1)
        ctx.save_for_backward(x, y)
        return -torch.where(y==0, torch.log(1-x).clamp_min_(ln_limit), torch.log(x).clamp_min_(ln_limit))
        # return -(y * torch.log(x).clamp_min_(ln_limit) + (1-y)*torch.log(1-x).clamp_min_(ln_limit))
    
    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        ln_limit = ctx.ln_limit
        
        # NOTE: for y==0, do not clip small x; for y==1, do not clip small (1-x)
        limit = np.exp(ln_limit)
        # x = torch.clip(x, eclip, 1-eclip)
        x = torch.where(y==0, torch.clip(x, 0, 1-limit), torch.clip(x, limit, 1))
        
        grad_x = grad_y = None
        if ctx.needs_input_grad[0]:
            # ttt = torch.where(y==0, 1/(1-x), -1/x) * grad_output * (~(x==y))
            # with open('grad.txt', 'a') as fp:
            #     fp.write(f"{ttt.min().item():.05f}, {ttt.max().item():.05f}\n")
            # NOTE: " * (~(x==y))" so that those already match will not generate gradients.
            grad_x = torch.where(y==0, 1/(1-x), -1/x) * grad_output * (~(x==y))
            # grad_x = ( (1-y)/(1-x) - y/x ) * grad_output
        if ctx.needs_input_grad[1]:
            grad_y = (torch.log(1-x) - torch.log(x)) * grad_output * (~(x==y))
        #---- x, y, limit
        return grad_x, grad_y, None

def safe_binary_cross_entropy(input: torch.Tensor, target: torch.Tensor, limit: float = 0.1, reduction="mean") -> torch.Tensor:
    loss = SafeBCE.apply(input, target, limit)
    return reduce(loss, None, reduction=reduction)

class ClippedMSE(autograd.Function):
    @staticmethod
    def forward(ctx, input, target, limit):
        # Safer when there are extreme values, while still with gradients on those clipped extreme values.
        lmin, lmax = limit
        err = (input - target).clamp_(lmin, lmax)
        ctx.save_for_backward(err)
        return err.square()
    @staticmethod
    def backward(ctx, grad):
        err, *_ = ctx.saved_tensors
        grad_input = 2 * grad * err
        #---- input, target, limit
        return grad_input, None, None

def safe_mse_loss(input: torch.Tensor, target: torch.Tensor, reduction:Literal['mean', 'none'] = 'none', limit: Union[float, Tuple[float,float]] = 1.) -> torch.Tensor:
    if isinstance(limit, Number):
        limit = (-limit, limit)
    loss = ClippedMSE.apply(input, target, limit)
    return reduce(loss, None, reduction=reduction)

if __name__ == "__main__":
    def test_bce():
        import torch.nn.functional as F
        pred = torch.tensor([-0.00001,  0, 0.00001, -0.00001,0, 0.00012, 0.0012, 0.012, 0.1, 0.89, 0.99, 0.989, 0.9989, 0.99989, 1, 1.00001, 0.99999, 1,  1.00001], requires_grad=True)
        gt = torch.tensor(  [0,         0, 0,       1,       1, 1,       1,      1,     1,   1,    1,    0,     0,      0,       0, 0,       1,       1,  1]).float()
        bce_loss = F.binary_cross_entropy(pred.clip(0,1), gt, reduction='none')
        print('pytorch BCE'.center(80, '='))
        print(bce_loss)
        print(autograd.grad(bce_loss.mean(), pred, retain_graph=True)[0])

        safe_bce_loss = safe_binary_cross_entropy(pred, gt, np.exp(-100))
        print(f'safe_bce_loss(ln_limit={-100})'.center(80, '='))
        print(safe_bce_loss)
        # NOTE: slightly different when grad is too large or too small (close to 1e+10 or -1e+10)
        print(autograd.grad(safe_bce_loss.mean(), pred, retain_graph=True)[0])

        # 0.1 -> -2.3
        # 0.01 -> -4.6
        
        safe_bce_loss = safe_binary_cross_entropy(pred, gt, 0.01) 
        print(f'safe_bce_loss(ln_limit={np.log(0.01)})'.center(80, '='))
        print(safe_bce_loss)
        # NOTE: has grad on the clipped values
        print(autograd.grad(safe_bce_loss.mean(), pred, retain_graph=True)[0])

        safe_bce_loss = safe_binary_cross_entropy(pred, gt, 0.03) 
        print(f'safe_bce_loss(ln_limit={np.log(0.03)})'.center(80, '='))
        print(safe_bce_loss)
        # NOTE: has grad on the clipped values
        print(autograd.grad(safe_bce_loss.mean(), pred, retain_graph=True)[0])

        safe_bce_loss = safe_binary_cross_entropy(pred, gt, 0.1) 
        print(f'safe_bce_loss(ln_limit={np.log(0.1)})'.center(80, '='))
        print(safe_bce_loss)
        # NOTE: has grad on the clipped values
        print(autograd.grad(safe_bce_loss.mean(), pred, retain_graph=True)[0])

    def test_mse(device=torch.device('cuda')):
        from torch.utils.benchmark import Timer
        import torch.nn.functional as F
        from icecream import ic
        x = 2 ** torch.randn([7])
        x[4] = 1e+9
        y = torch.ones([7])

        x1 = x.clone().requires_grad_(True)
        l1 = F.mse_loss(x1, y, reduction='mean')
        l1.backward()
        
        x2 = x.clone().requires_grad_(True)
        l2 = safe_mse_loss(x2, y, 'mean', 10.0)
        l2.backward()
        
        ic(x1.grad)
        ic(x2.grad)

        x = 2 ** torch.randn([2**18], device=device)
        y = torch.ones([2**18], device=device)
        # 16.6 us
        print(Timer(
            stmt="F.mse_loss(x, y, reduction='mean')", 
            globals={'F':F, "x": x, 'y': y}
        ).blocked_autorange())

        # 29.0 us
        print(Timer(
            stmt="safe_mse_loss(x, y, reduction='mean')", 
            globals={'safe_mse_loss':safe_mse_loss, "x": x, 'y': y}
        ).blocked_autorange())

    # test_bce()
    test_mse()
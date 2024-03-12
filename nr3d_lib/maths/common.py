__all__ = [
    'geometric_mean', 
    'normalize', 
    'strip_lowerdiag', 
    'strip_symmetric', 
    'skew_symmetric', 
    'divide_no_nan',  
    'dot',  
    'mix',  
    'schlick_fresnel', 
    'inverse_sigmoid', 
    'logistic_density', 
    'normalized_logistic_density', 
    'TruncExp', 
    'trunc_exp', 
    'TruncSoftplus', 
    'trunc_softplus'
]

import math
import numpy as np
from typing import Union
from numbers import Number
from torch.cuda.amp import custom_bwd, custom_fwd

import torch
import torch.nn as nn
import torch.nn.functional as F

def geometric_mean(input: Union[torch.Tensor, np.ndarray], **kwargs):
    if isinstance(input, torch.Tensor):
        return input.log().mean(**kwargs).exp()
    elif isinstance(input, np.ndarray):
        return np.exp(np.log(input).mean(**kwargs))
    else:
        raise RuntimeError(f"Invalid input type={type(input)}")

def normalize(input: Union[torch.Tensor, np.ndarray], dim=-1):
    if isinstance(input, torch.Tensor):
        return F.normalize(input, dim=dim)
    elif isinstance(input, np.ndarray):
        return input / (np.linalg.norm(input, axis=dim, keepdims=True) + 1e-9)
    else:
        raise RuntimeError(f"Invalid input type={type(input)}")

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def skew_symmetric(vector: torch.Tensor):
    ss_matrix = torch.zeros([*vector.shape[:-1],3,3], device=vector.device, dtype=vector.dtype)
    ss_matrix[..., 0, 1] = -vector[..., 2]
    ss_matrix[..., 0, 2] =  vector[..., 1]
    ss_matrix[..., 1, 0] =  vector[..., 2]
    ss_matrix[..., 1, 2] = -vector[..., 0]
    ss_matrix[..., 2, 0] = -vector[..., 1]
    ss_matrix[..., 2, 1] =  vector[..., 0]
    return ss_matrix

def inverse_sigmoid(x: torch.Tensor):
    return torch.log(x / (1 - x))

def divide_no_nan(a: torch.Tensor, b: torch.Tensor):
    return (a / b).nan_to_num(0., 0., 0.)

def dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    _summary_

    :param a `Tensor(N..., C)`: _description_
    :param b `Tensor(N..., C)`: _description_
    :return `Tensor(N..., 1)`: _description_
    """
    return (a * b).sum(-1, keepdim=True)

def mix(a: Union[float, torch.Tensor], b: Union[float, torch.Tensor], t: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
    """
    _summary_

    :param a `float|Tensor(N..., C)`: _description_
    :param b `float|Tensor(N..., C)`: _description_
    :param t `float|Tensor(N..., 1)`: _description_
    :return `float|Tensor(N..., C)`: _description_
    """
    return a + (b - a) * t

def schlick_fresnel(u: torch.Tensor) -> torch.Tensor:
    """
    _summary_

    :param u `Tensor(N...)`: _description_
    :return `Tensor(N...)`: _description_
    """
    return (1. - u) ** 5

def inverse_sigmoid(x):
    return torch.log(x / (1 - x))

def logistic_density(x: torch.Tensor, inv_s: Union[torch.Tensor, Number]) -> torch.Tensor:
    """ Logistic density function
    Source: https://en.wikipedia.org/wiki/Logistic_distribution

    Args:
        x (torch.Tensor): Input
        inv_s (Union[torch.Tensor, Number]): The reciprocal of the distribution scaling factor.

    Returns:
        torch.Tensor: Output
    """
    return 0.25*inv_s / (torch.cosh(inv_s*x/2.).clamp_(-20, 20)**2) 

def normalized_logistic_density(x: torch.Tensor, inv_s: Union[torch.Tensor, Number]) -> torch.Tensor:
    """ Normalized logistic density function (with peak value = 1.0)
    Source: https://en.wikipedia.org/wiki/Logistic_distribution

    Args:
        x (torch.Tensor): Input
        inv_s (Union[torch.Tensor, Number]): The reciprocal of the distribution scaling factor.

    Returns:
        torch.Tensor: Output
    """
    return (1./torch.cosh((inv_s*x/2.).clamp_(-20, 20)))**2

class _TruncExp(torch.autograd.Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        # NOTE: For torch.float16, exp(x > 11.08) = inf
        return g * torch.exp(torch.clamp(x, max=11))

trunc_exp = _TruncExp.apply

class TruncExp(nn.Module):
    def __init__(self, offset: float = 0) -> None:
        super().__init__()
        self.offset = offset
    def forward(self, input: torch.Tensor):
        return trunc_exp(input + self.offset)

class _TruncSoftplus(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x, beta=1.0, threshold=20.0):
        ctx.beta = beta
        ctx.threshold = threshold
        y = F.softplus(x, beta, threshold)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        y = ctx.saved_tensors[0]
        beta = ctx.beta
        threshold = ctx.threshold
        y_beta = y * beta
        # NOTE: For torch.float16, exp(x > 11.08) = inf
        z = torch.exp(torch.clamp(y_beta, max=11))
        grad_input = torch.where(y_beta > threshold, g, g * (z - 1) / z)
        return grad_input, None, None

trunc_softplus = _TruncSoftplus.apply

class TruncSoftplus(nn.Softplus):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return trunc_softplus(input, self.beta, self.threshold)

if __name__ == "__main__":
    def test_trunc_exp():
        import torch.autograd
        from torch.autograd.anomaly_mode import set_detect_anomaly
        set_detect_anomaly(True)
        device = torch.device('cuda')
        x0 = torch.tensor([0., 5., 10., 15., 20., 30., 50.], 
                          dtype=torch.half, device=device)
        softplus = nn.Softplus()
        
        #----
        x = x0.clone().requires_grad_(True)
        with torch.autocast(device_type='cuda', dtype=torch.half):
            y = torch.exp(x)
            grad00 = torch.autograd.grad(y.sum(), x, only_inputs=True)[0]

        #----
        x = x0.clone().requires_grad_(True)
        with torch.autocast(device_type='cuda', dtype=torch.half):
            y = torch.exp(x)
            y.sum().backward()
        grad01 = x.grad.data.clone()

        #----
        x = x0.clone().requires_grad_(True)
        with torch.autocast(device_type='cuda', dtype=torch.half):
            y = torch.exp(x.float())
            y.sum().backward()
        grad02 = x.grad.data.clone()

        #----
        x = x0.clone().requires_grad_(True)
        with torch.autocast(device_type='cuda', dtype=torch.half):
            y = torch.exp(x)
        y.sum().backward()
        grad03 = x.grad.data.clone()

        #----
        x = x0.clone().requires_grad_(True)
        with torch.autocast(device_type='cuda', dtype=torch.half):
            y = torch.exp(x.float())
        y.sum().backward()
        grad04 = x.grad.data.clone()

        #----
        x = x0.clone().requires_grad_(True)
        y = torch.exp(x)
        y.sum().backward()
        grad05 = x.grad.data.clone()

        #----
        x = x0.clone().requires_grad_(True)
        y = torch.exp(x.float())
        y.sum().backward()
        grad06 = x.grad.data.clone()

        #----
        x = x0.clone().requires_grad_(True)
        y = torch.exp(x)
        y.float().sum().backward()
        grad07 = x.grad.data.clone()

        #----
        x = x0.clone().requires_grad_(True)
        with torch.autocast(device_type='cuda', dtype=torch.half):
            y = trunc_exp(x)
            grad10 = torch.autograd.grad(y.sum(), x, only_inputs=True)[0] # NOTE: no error, float16, inf

        #----
        x = x0.clone().requires_grad_(True)
        with torch.autocast(device_type='cuda', dtype=torch.half):
            y = trunc_exp(x.float())
            grad11 = torch.autograd.grad(y.sum(), x, only_inputs=True)[0] # NOTE: no error, float16, inf

        #----
        x = x0.clone().requires_grad_(True)
        with torch.autocast(device_type='cuda', dtype=torch.half):
            y = trunc_exp(x)
            y.sum().backward()
        grad12 = x.grad.data.clone()

        #----
        x = x0.clone().requires_grad_(True)
        with torch.autocast(device_type='cuda', dtype=torch.half):
            y = trunc_exp(x.float())
            y.sum().backward()
        grad13 = x.grad.data.clone()

        #----
        x = x0.clone().requires_grad_(True)
        with torch.autocast(device_type='cuda', dtype=torch.half):
            y = trunc_exp(x)
        y.sum().backward()
        grad14 = x.grad.data.clone()

        #----
        x = x0.clone().requires_grad_(True)
        with torch.autocast(device_type='cuda', dtype=torch.half):
            y = trunc_exp(x.float())
        y.sum().backward()
        grad15 = x.grad.data.clone()


        #----
        x = x0.clone().requires_grad_(True)
        y = trunc_exp(x)
        grad20 = torch.autograd.grad(y.sum(), x, only_inputs=True)[0] # NOTE: no error, float16, inf

        #----
        x = x0.clone().requires_grad_(True)
        y = trunc_exp(x.float())
        grad21 = torch.autograd.grad(y.sum(), x, only_inputs=True)[0] # NOTE: no error, float16, inf
        
        #----
        x = x0.clone().requires_grad_(True)
        y = trunc_exp(x)
        y.sum().backward()
        grad22 = x.grad.data.clone()

        #----
        x = x0.clone().requires_grad_(True)
        y = trunc_exp(x.float())
        y.sum().backward()
        grad23 = x.grad.data.clone()
        

        #----
        x = x0.clone().requires_grad_(True)
        y = softplus(x.float())
        y.sum().backward()
        grad30 = x.grad.data.clone()

        #----
        x = x0.clone().requires_grad_(True)
        with torch.autocast(device_type='cuda', dtype=torch.half):
            y = softplus(x)
        y.sum().backward()
        grad31 = x.grad.data.clone()

        #----
        x = x0.clone().requires_grad_(True)
        y = softplus(x)
        y.sum().backward() # BUG: Error: Function 'SoftplusBackward0' returned nan values in its 0th output.
        grad32 = x.grad.data.clone()
    
    def test_trunc_softpus():
        from nr3d_lib.fmt import log
        import torch.autograd
        from torch.autograd.anomaly_mode import set_detect_anomaly
        set_detect_anomaly(True)
        from torch.utils.benchmark import Timer
        
        device = torch.device('cuda')
        """
        NOTE: All x > 11.08 will raise nan grad to softplus (inf / inf = nan)
        >>> torch.exp(torch.tensor([11], dtype=torch.half, device='cuda')) ->  59782.
        >>> torch.exp(torch.tensor([12], dtype=torch.half, device='cuda')) ->  inf
        
        """
        batch_size = 5432100
        x0 = torch.rand([batch_size], dtype=torch.float, device=device) * 40.
        x1 = torch.rand([batch_size], dtype=torch.float, device=device) * (10**torch.randint(-6, 0, [batch_size,], device=device))
        x0 = torch.cat((x0, x1)).half()
        
        try:
            x = x0.clone().requires_grad_(True)
            y = F.softplus(x)
            y.mean().backward()
            grad0 = x.grad.data.clone() # BUG: nan when x > 11.08
            log.warning("F.softplus(x), " + "pass")
        except Exception as e:
            log.error("F.softplus(x)\n" + repr(e))

        try:
            x = x0.clone().requires_grad_(True)
            with torch.autocast(device_type='cuda', dtype=torch.half):
                y = F.softplus(x) # No error
            y.mean().backward()
            grad1 = x.grad.data.clone() # NOTE: No nan
            log.warning("cast, F.softplus(x), " + "pass")
        except Exception as e:
            log.error("cast, F.softplus(x)\n" + repr(e))
        
        try:
            x = x0.clone().requires_grad_(True)
            y = nn.Softplus()(x)
            y.mean().backward()
            grad2 = x.grad.data.clone() # BUG: nan when x > 11.08
            log.warning("nn.Softplus()(x), " + "pass")
        except Exception as e:
            log.error("nn.Softplus()(x)\n" + repr(e))

        try:
            x = x0.clone().requires_grad_(True)
            y = trunc_softplus(x)
            y.mean().backward()
            grad3 = x.grad.data.clone() # NOTE: No nan
            log.warning("trunc_softplus(x), " + "pass")
        except Exception as e:
            log.error("trunc_softplus(x)\n" + repr(e))

        try:
            x = x0.clone().requires_grad_(True)
            y = TruncSoftplus()(x)
            y.mean().backward()
            grad4 = x.grad.data.clone() # NOTE: No nan
            log.warning("TruncSoftplus()(x), " + "pass")
        except Exception as e:
            log.error("TruncSoftplus()(x)\n" + repr(e))

        # 174 us
        def fn0():
            x = x0.clone().requires_grad_(True)
            y = F.softplus(x)
            return y.mean()

        # 319 us
        def fn1():
            x = x0.clone().requires_grad_(True)
            y = F.softplus(x)
            y.mean().backward()
            return x.grad

        # 718 us
        def fn2():
            x = x0.clone().requires_grad_(True)
            with torch.autocast(device_type='cuda', dtype=torch.half):
                y = F.softplus(x)
            y.mean().backward()
            return x.grad

        # 838 us
        def fn3():
            x = x0.clone().requires_grad_(True)
            y = trunc_softplus(x)
            y.mean().backward()
            return x.grad

        # 85 us
        print(Timer(stmt="F.softplus(x0)", globals={'F':F, 'x0': x0}).blocked_autorange())
        # 77 us
        print(Timer(stmt="trunc_softplus(x0)", globals={'trunc_softplus':trunc_softplus, 'x0': x0}).blocked_autorange())
        
        # 174 us
        print(Timer(stmt="fn0()", globals={'fn0':fn0}).blocked_autorange())
        # 319 us
        print(Timer(stmt="fn1()", globals={'fn1':fn1}).blocked_autorange())
        # 718 us
        print(Timer(stmt="fn2()", globals={'fn2':fn2}).blocked_autorange())
        # 838 us
        print(Timer(stmt="fn3()", globals={'fn3':fn3}).blocked_autorange())

        _  = 1

    # test_trunc_exp()
    test_trunc_softpus()
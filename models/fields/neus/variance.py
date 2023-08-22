"""
@file   variance.py
@author Jianfei Guo, Shanghai AI Lab
@brief  NeuS variance controller
"""

__all__ = [
    'get_neus_var_ctrl'
]

from typing import Literal
import numpy as np

import torch
import torch.nn as nn

from nr3d_lib.config import ConfigDict
from nr3d_lib.models.annealers import AnnealerLinear, get_annealer

def get_neus_var_ctrl(
    ctrl_type: Literal['constant', 'learned', 'manual', 'ln_manual', 
                       'mix_linear', 'mix_halflife'] = 'learned', 
    **kwargs):
    if ctrl_type == 'constant':
        return VarConstant(**kwargs)
    elif ctrl_type == 'learned':
        return VarSingleLearned(**kwargs)
    elif ctrl_type == 'manual':
        return VarSingleManual(**kwargs)
    elif ctrl_type == 'ln_manual':
        return VarSingleLnManual(**kwargs)
    elif ctrl_type == 'mix_linear':
        return VarSingleMixLinear(**kwargs)
    elif ctrl_type == 'mix_linear_manual':
        return VarSingleMixLinearManual(**kwargs)
    elif ctrl_type == 'mix_halflife':
        return VarSingleMixHalfLife(*kwargs)
    else:
        raise RuntimeError(f"Invalid ctrl_type={ctrl_type}")

class VarConstant(nn.Module):
    def __init__(self, inv_s: float, device=None) -> None:
        super().__init__()
        self.inv_s = inv_s
    def set_iter(self, it: int):
        self.it = it
    def forward(self, it: int = None):
        if it is not None: self.set_iter(it)
        return self.inv_s

class VarSingleLearned(nn.Module):
    def __init__(self, ln_inv_s_init: float, ln_inv_s_factor: float = 10.0, device=None) -> None:
        super().__init__()
        self.device = device
        self.ln_inv_s_init = ln_inv_s_init
        self.ln_inv_s_factor = ln_inv_s_factor
        self.ln_inv_s = nn.Parameter(data=torch.tensor([self.ln_inv_s_init], device=self.device, dtype=torch.float), requires_grad=True)
    def set_iter(self, it: int):
        self.it = it
    def forward(self, it: int = None) -> torch.Tensor:
        if it is not None: self.set_iter(it)
        return torch.exp(self.ln_inv_s * self.ln_inv_s_factor)

class VarSingleManual(nn.Module):
    def __init__(self, inv_s_anneal_cfg: ConfigDict, device=None) -> None:
        super().__init__()
        self.inv_s_annealer = get_annealer(**inv_s_anneal_cfg)
    def set_iter(self, it: int):
        self.it = it
        self.inv_s_annealer.set_iter(it)
    def forward(self, it: int = None) -> float:
        if it is not None: self.set_iter(it)
        return self.inv_s_annealer.get_val()

class VarSingleLnManual(nn.Module):
    def __init__(self, ln_inv_s_anneal_cfg: ConfigDict, ln_inv_s_factor: float = 10.0, device=None) -> None:
        super().__init__()
        self.ln_inv_s_factor = ln_inv_s_factor
        self.ln_inv_s_annealer = get_annealer(**ln_inv_s_anneal_cfg)
    def set_iter(self, it: int):
        self.it = it
        self.ln_inv_s_annealer.set_iter(it)
    def forward(self, it: int = None) -> float:
        if it is not None: self.set_iter(it)
        ln_inv_s = self.ln_inv_s_annealer.get_val()
        return np.exp(ln_inv_s * self.ln_inv_s_factor)

class VarSingleMixLinearManual(nn.Module):
    def __init__(
        self, 
        start_inv_s: float = 20.0, final_inv_s: float = 2048., 
        transition_ratio: float = 0.1, transition_it: int = ..., stop_it: int = ...) -> None:
        """ 
        An exponential growth starting from a certain constant (e.g., 20.0) \
        transitioning to a linear growth ending at another constant (e.g., 1000.0). 
        """
        super().__init__()
        self.start_inv_s = start_inv_s
        self.final_inv_s = final_inv_s
        self.stop_it = stop_it
        self.transition_it = transition_it
        
        # NOTE: When it==transition_it (i.e. alpha==0), we want the normalized output to be transition_ratio
        self.beta = np.log(2.) / transition_ratio
        
        # NOTE: The actual minimum (close to zero but not zero)
        self.min_softplus = 1./self.beta * np.log(1+np.exp(self.beta * (-self.transition_it)/(self.stop_it-self.transition_it)))
        
    def set_iter(self, it: int):
        self.it = it
    
    def forward(self, it: int = None):
        if it is not None: 
            self.set_iter(it)
        else:
            it = self.it
        alpha = (it - self.transition_it) / (self.stop_it - self.transition_it)
        softplus = 1./self.beta * np.log(1+np.exp(self.beta * alpha))
        softplus_fix = (softplus - self.min_softplus) / (1-self.min_softplus)
        inv_s = (1-softplus_fix) * self.start_inv_s + softplus_fix * self.final_inv_s
        return inv_s

class VarSingleMixLinear(nn.Module):
    def __init__(
        self, 
        ln_inv_s_init: float, ln_inv_s_factor: float = 10.0, 
        stop_it: int = ..., start_it: int = 0, final_inv_s: float = 2048., 
        device=None) -> None:
        super().__init__()
        self.device = device
        self.ln_inv_s_init = ln_inv_s_init
        self.ln_inv_s_factor = ln_inv_s_factor
        self.ln_inv_s = nn.Parameter(data=torch.tensor([self.ln_inv_s_init], device=self.device, dtype=torch.float), requires_grad=True)
        self.w_annealer = AnnealerLinear(stop_it=stop_it, start_it=start_it, start_val=0., stop_val=1.)
        self.final_inv_s = final_inv_s
    def set_iter(self, it: int):
        self.it = it
        self.w_annealer.set_iter(it)
    def forward(self, it: int = None):
        if it is not None: self.set_iter(it)
        inv_s0 = torch.exp(self.ln_inv_s * self.ln_inv_s_factor)
        w = self.w_annealer.get_val()
        return (1-w) * inv_s0 + w * self.final_inv_s

class VarSingleMixHalfLife(nn.Module):
    def __init__(
        self, 
        ln_inv_s_init: float, ln_inv_s_factor: float = 10.0, 
        stop_it: int = ..., start_it: int=0, final_log2_inv_s: float=14., half_log2_inv_s: float=9., update_every: int=1, 
        device=None) -> None:
        super().__init__()
        self.device = device
        self.ln_inv_s_init = ln_inv_s_init
        self.ln_inv_s_factor = ln_inv_s_factor
        self.ln_inv_s = nn.Parameter(data=torch.tensor([self.ln_inv_s_init], device=self.device, dtype=torch.float), requires_grad=True)
        self.start_it = int(start_it)
        self.stop_it = int(stop_it)
        self.update_every = update_every
        self.total_stages = (self.stop_it - self.start_it) // self.update_every
        self.it = None
        self.final_log2_inv_s = final_log2_inv_s
        self.init_log2_inv_s = half_log2_inv_s - (final_log2_inv_s-half_log2_inv_s)
    def set_iter(self, it: int):
        self.it = it
    def forward(self, it: int = None):
        if it is not None: self.set_iter(it)
        inv_s0 = torch.exp(self.ln_inv_s * self.ln_inv_s_factor)
        w0 = 1 - self.it / self.stop_it
        if self.it < self.start_it:
            inv_s1 = 2 ** self.init_log2_inv_s
        else:
            cur_stage = (self.it-self.start_it) // self.update_every
            w1 = cur_stage / self.total_stages
            log2_inv_s = (1-w1) * self.init_log2_inv_s + w1 * self.final_log2_inv_s
            inv_s1 = 2 ** log2_inv_s
        return w0 * inv_s0 + (1-w0) * inv_s1

if __name__ == "__main__":
    # def test_mix_halflife():
    #     import numpy as np
    #     import matplotlib.pyplot as plt
    #     inv_s0_ = np.linspace(0,1,300) * 200
    #     inv_s_ = []
    #     ctrl = VarSingleMixHalfLife(
    #         ln_inv_s_init=0.3, ln_inv_s_factor=10.0, 
    #         stop_it=300, start_it=50, 
    #         final_log2_inv_s=9, half_log2_inv_s=8)
    #     for it, inv_s0 in enumerate(inv_s0_):
    #         ctrl.set_iter(it)
    #         ctrl.ln_inv_s.data.fill_(np.log(inv_s0) / ctrl.ln_inv_s_factor)
    #         inv_s = ctrl.forward().item()
    #         inv_s_.append(inv_s)
    #     inv_s_ = np.array(inv_s_)
    #     # plt.plot(np.arange(300), np.log(inv_s0_+1), label='inv_s0')
    #     # plt.plot(np.arange(300), np.log(inv_s_+1), label='inv_s')
    #     plt.plot(np.arange(300), inv_s0_, label='inv_s0')
    #     plt.plot(np.arange(300), inv_s_, label='inv_s')
    #     plt.legend()
    #     plt.show()
    def test_mix_manual(num_iters: int = 7500):
        import numpy as np
        import matplotlib.pyplot as plt
        ctrl = VarSingleMixLinearManual(start_inv_s=20.0, final_inv_s=2000.0, transition_it=1000, stop_it=num_iters, transition_ratio=0.03)
        inv_s_ = []
        for it in range(num_iters):
            ctrl.set_iter(it)
            inv_s = ctrl.forward()
            inv_s_.append(inv_s)
        inv_s_ = np.array(inv_s_)
        plt.plot(np.arange(num_iters), inv_s_, label='inv_s')
        plt.legend()
        plt.show()
    # test_mix_halflife()
    test_mix_manual()
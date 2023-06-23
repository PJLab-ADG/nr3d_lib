"""
@file   volume_graphics.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Common volume graphics helper functions. Implemented on both batched-rays & packed-rays.
"""

__all__ = [
    'packed_alpha_to_vw',
    'packed_volume_render_compression',
    'packed_tau_to_vw',
    'packed_tau_alpha_to_vw',
    'ray_alpha_to_vw',
    'ray_tau_to_vw',
    'ray_tau_alpha_to_vw',
]

import torch

from nr3d_lib.render.pack_ops import packed_cumsum, packed_cumprod, packed_alpha_to_vw, packed_volume_render_compression

# @torch.jit.script
def tau_to_alpha(tau: torch.Tensor) -> torch.Tensor:
    return 1-torch.exp(-tau)

#----------------------------------------------------
#---------- Volume graphics on packs ----------------
#----------------------------------------------------
# NOTE:
#       When in eval:   v1 vs. v2 =  73 us vs. 178 us @ 1M pts.
#       When in train:  v1 vs. v2 = 900 us vs. 186 us @ 110k pts
def packed_alpha_to_vw_v1(alpha: torch.Tensor, pack_infos: torch.LongTensor) -> torch.Tensor:
    """ Pack-wise volume rendering

    Args:
        alpha (torch.Tensor): [num_packed_pts, ] Alpha value of each interval
        pack_infos (torch.LongTensor): [num_packs, 2] pack_infos for packed intervals

    Returns:
        torch.Tensor: [num_packed_pts, ] Visibility weitghts of each interval
    """
    alpha = alpha.view(-1,1)
    # Shifted cumprod
    transmittance = packed_cumprod((1+1e-10)-alpha, pack_infos, exclusive=True)
    return (alpha * transmittance).squeeze_(-1)

def packed_alpha_to_vw_v2(alpha: torch.Tensor, pack_infos: torch.LongTensor, early_stop_eps: float = 1e-4, alpha_thre: float = 0.0) -> torch.Tensor:
    """ Pack-wise volume rendering with early_stop_eps and alpha threshold
        Modified from https://github.com/KAIR-BAIR/nerfacc

    Args:
        alpha (torch.Tensor): [num_packed_pts, ] Alpha value of each interval
        pack_infos (torch.LongTensor): [num_packs, 2] pack_infos for packed intervals
        early_stop_eps (float, optional): Points after the T that reaches `T<early_stop_eps` will be ignored. Defaults to 1e-4.
        alpha_thre (float, optional): alhpas <= alpha_thre will be ignored. Defaults to 0.0.

    Returns:
        torch.Tensor: [num_packed_pts, ] Visibility weitghts of each interval
    """
    return packed_alpha_to_vw(alpha, pack_infos, early_stop_eps, alpha_thre)

def packed_tau_to_vw(tau: torch.Tensor, pack_infos: torch.LongTensor) -> torch.Tensor:
    """ Pack-wise volume rendering

    Args:
        tau (torch.Tensor): [num_packed_pts, ] Optical depth of each interval
        pack_infos (torch.LongTensor): [num_packs, 2] pack_infos for packed intervals

    Returns:
        torch.Tensor: [num_packed_pts, ] Visibility weitghts of each interval
    """
    tau = tau.view(-1,1)
    alpha = 1-torch.exp(-tau)
    # Shifted cumsum
    transmittance = torch.exp(-1*packed_cumsum(tau, pack_infos, exclusive=True))
    return (alpha * transmittance).squeeze_(-1)

def packed_tau_alpha_to_vw(tau: torch.Tensor, alpha: torch.Tensor, pack_infos: torch.LongTensor) -> torch.Tensor:
    """ Pack-wise volume rendering

    Args:
        tau (torch.Tensor): [num_packed_pts, ] Alpha value of each interval
        alpha (torch.Tensor): [num_packed_pts, ] Optical depth of each interval
        pack_infos (torch.LongTensor): [num_packs, 2] pack_infos for packed intervals

    Returns:
        torch.Tensor: [num_packed_pts, ] Visibility weitghts of each interval
    """
    tau, alpha = tau.view(-1,1), alpha.view(-1,1)
    # Shifted cumsum
    transmittance = torch.exp(-1*packed_cumsum(tau, pack_infos, exclusive=True))
    return (alpha * transmittance).squeeze_(-1)

#----------------------------------------------------
#---------- Volume graphics on ray ------------------
#----------------------------------------------------
# @torch.jit.script
def ray_alpha_to_vw(alpha: torch.Tensor) -> torch.Tensor:
    """ Batch-wise volume rendering

    Args:
        alpha (torch.Tensor): [..., num_pts], Alpha value of each interval

    Returns:
        torch.Tensor: [..., num_pts], Visibility weights of each interval
    """
    # Shifted cumprod
    shifted_transparency = torch.roll((1+1e-10)-alpha, 1, dims=-1)
    shifted_transparency[..., 0] = 1
    return alpha * torch.cumprod(shifted_transparency, dim=-1)

# @torch.jit.script
def ray_tau_to_vw(tau: torch.Tensor) -> torch.Tensor:
    """ Batch-wise volume rendering

    Args:
        tau (torch.Tensor): [..., num_pts], Optical depth of each interval

    Returns:
        torch.Tensor: [..., num_pts], Visibility weights of each interval
    """
    alpha = 1-torch.exp(-tau)
    # Shifted cumsum
    transmittance = torch.roll(torch.exp(-1*torch.cumsum(tau, dim=-1)), 1)
    transmittance[..., 0] = 1
    return alpha * transmittance

# @torch.jit.script
def ray_tau_alpha_to_vw(tau: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """ Batch-wise volume rendering

    Args:
        tau (torch.Tensor): [..., num_pts], Optical depth of each interval
        alpha (torch.Tensor): [..., num_pts], Alpha value of each interval

    Returns:
        torch.Tensor: [..., num_pts], Visibility weights of each interval
    """
    # Shifted cumsum
    transmittance = torch.roll(torch.exp(-1*torch.cumsum(tau, dim=-1)), 1)
    transmittance[..., 0] = 1
    return alpha * transmittance

# def ray_volume_render(vw: torch.Tensor, depths: torch.Tensor, radiances: torch.Tensor=None, nablas: torch.Tensor=None):
#     pass

# def packed_volume_render(vw: torch.Tensor, pack_infos: torch.LongTensor, depths: torch.Tensor, radiances: torch.Tensor=None, nablas: torch.Tensor=None):
#     pass

if __name__ == "__main__":
    def test_boundary_vw(device=torch.device('cuda')):
        from kaolin.render.spc import cumsum, cumprod
        def boundary_alpha_to_vw(alpha: torch.Tensor, boundary: torch.Tensor) -> torch.Tensor:
            alpha = alpha.view(-1,1)
            # Shifted cumprod
            transmittance = cumprod((1+1e-10)-alpha, boundary, exclusive=True)
            return (alpha * transmittance).squeeze_(-1)

        def boundary_tau_to_vw(tau: torch.Tensor, boundary: torch.Tensor) -> torch.Tensor:
            tau = tau.view(-1,1)
            alpha = 1-torch.exp(-tau)
            # Shifted cumsum
            transmittance = torch.exp(-1*cumsum(tau, boundary, exclusive=True))
            return (alpha * transmittance).squeeze_(-1)

        def boundary_tau_alpha_to_vw(tau: torch.Tensor, alpha: torch.Tensor, boundary: torch.Tensor) -> torch.Tensor:
            tau, alpha = tau.view(-1,1), alpha.view(-1,1)
            # Shifted cumsum
            transmittance = torch.exp(-1*cumsum(tau, boundary, exclusive=True))
            return (alpha * transmittance).squeeze_(-1)
        
        sigma = torch.empty([7,13], dtype=torch.float, device=device).uniform_(30, 50)
        deltas = torch.full([7,13], 0.01, dtype=torch.float, device=device)
        tau = sigma * deltas
        alpha = tau_to_alpha(tau)
        vw1 = ray_tau_to_vw(tau)
        
        boundary = torch.zeros([7,13], dtype=torch.bool, device=device)
        boundary[...,0] = True
        boundary = boundary.flatten()
        vw2 = boundary_tau_to_vw(tau.flatten(), boundary)
        vw3 = boundary_alpha_to_vw(alpha.flatten(), boundary)
        vw4 = boundary_tau_alpha_to_vw(tau.flatten(), alpha.flatten(), boundary)
        
        print(torch.allclose(vw1.flatten(), vw2.flatten()))
        print(torch.allclose(vw1.flatten(), vw3.flatten()))
        print(torch.allclose(vw1.flatten(), vw4.flatten()))
        
        # Test grad
        sigma = torch.empty([7,13], dtype=torch.float, device=device).uniform_(30, 50).requires_grad_(True)
        deltas = torch.full([7,13], 0.01, dtype=torch.float, device=device)
        tau = sigma * deltas
        vw1 = ray_tau_to_vw(tau)
        vw1.mean().backward()
        grad1 = sigma.grad
        sigma.grad = None
        
        tau = sigma * deltas
        boundary = torch.zeros([7,13], dtype=torch.bool, device=device)
        boundary[...,0] = True
        boundary = boundary.flatten()
        vw2 = boundary_tau_to_vw(tau.flatten(), boundary)
        vw2.mean().backward()
        grad2 = sigma.grad
        sigma.grad = None
        
        tau = sigma * deltas
        alpha = tau_to_alpha(tau)
        vw3 = boundary_alpha_to_vw(alpha.flatten(), boundary)
        vw3.mean().backward()
        grad3 = sigma.grad
        sigma.grad = None
        
        tau = sigma * deltas
        alpha = tau_to_alpha(tau)
        vw4 = boundary_tau_alpha_to_vw(tau.flatten(), alpha.flatten(), boundary)
        vw4.mean().backward()
        grad4 = sigma.grad
        sigma.grad = None
        
        print(torch.allclose(grad1.flatten(), grad2.flatten()))
        print(torch.allclose(grad1.flatten(), grad3.flatten()))
        print(torch.allclose(grad1.flatten(), grad4.flatten()))

    def test_packed_vw(device=torch.device('cuda')):
        from nr3d_lib.render.pack_ops import get_pack_infos_from_first
        sigma = torch.empty([7,13], dtype=torch.float, device=device).uniform_(30, 50)
        deltas = torch.full([7,13], 0.01, dtype=torch.float, device=device)
        tau = sigma * deltas
        alpha = tau_to_alpha(tau)
        vw1 = ray_tau_to_vw(tau)
        
        first_inds = torch.arange(7, device=device, dtype=torch.long) * 13
        pack_infos = get_pack_infos_from_first(first_inds)
        vw2 = packed_tau_to_vw(tau.flatten(), pack_infos)
        vw3 = packed_alpha_to_vw_v1(alpha.flatten(), pack_infos)
        vw4 = packed_tau_alpha_to_vw(tau.flatten(), alpha.flatten(), pack_infos)
        
        print(torch.allclose(vw1.flatten(), vw2.flatten()))
        print(torch.allclose(vw1.flatten(), vw3.flatten()))
        print(torch.allclose(vw1.flatten(), vw4.flatten()))
        
        # Test grad
        sigma = torch.empty([7,13], dtype=torch.float, device=device).uniform_(30, 50).requires_grad_(True)
        deltas = torch.full([7,13], 0.01, dtype=torch.float, device=device)

        tau = sigma * deltas
        vw1 = ray_tau_to_vw(tau)
        vw1.mean().backward()
        grad1 = sigma.grad
        sigma.grad = None
        
        tau = sigma * deltas
        first_inds = torch.arange(7, device=device, dtype=torch.long) * 13
        pack_infos = get_pack_infos_from_first(first_inds)
        vw2 = packed_tau_to_vw(tau.flatten(), pack_infos)
        vw2.mean().backward()
        grad2 = sigma.grad
        sigma.grad = None
        
        tau = sigma * deltas
        alpha = tau_to_alpha(tau)
        vw3 = packed_alpha_to_vw_v1(alpha.flatten(), pack_infos)
        vw3.mean().backward()
        grad3 = sigma.grad
        sigma.grad = None
        
        tau = sigma * deltas
        alpha = tau_to_alpha(tau)
        vw4 = packed_tau_alpha_to_vw(tau.flatten(), alpha.flatten(), pack_infos)
        vw4.mean().backward()
        grad4 = sigma.grad
        sigma.grad = None
        
        print(torch.allclose(grad1.flatten(), grad2.flatten()))
        print(torch.allclose(grad1.flatten(), grad3.flatten()))
        print(torch.allclose(grad1.flatten(), grad4.flatten()))

    def benchmark_packed_vw(device=torch.device('cuda')):
        """
        NOTE: 
        
        When in eval:
            Early stopping is actually 2.4x SLOWER than non-early stopping ops.
            Possible guess is that the compare ops in early-stopping costs more.
            
            178 us vs. 73 us @ 1M pts.
            669 us vs. 429 us @ 4M pts.
            
            <torch.utils.benchmark.utils.common.Measurement object at 0x7f5d9ebaff10>
            PackedAlphaToVW.apply(alpha, pack_infos, early_stop_eps, alpha_thre)
            178.42 us
            1 measurement, 10000 runs , 1 thread
            
            <torch.utils.benchmark.utils.common.Measurement object at 0x7f5d6ef52640>
            packed_alpha_to_w_old(alpha, pack_infos)
            Median: 73.20 us
            3 measurements, 1000 runs per measurement, 1 thread
        When in train:
            186 us vs. 900 us @ 110k pts
            
            <torch.utils.benchmark.utils.common.Measurement object at 0x7fb124589cd0>
            packed_alpha_to_vw_v2(alpha, pack_infos)
            Median: 186.72 us
            2 measurements, 1000 runs per measurement, 1 thread
            <torch.utils.benchmark.utils.common.Measurement object at 0x7fb124589ca0>
            packed_alpha_to_vw_v1(alpha, pack_infos)
            906.45 us
            1 measurement, 1000 runs , 1 thread
        """

        alpha = ...
        pack_infos = ...

        with torch.no_grad():
            from torch.utils.benchmark import Timer
            print(Timer(
                stmt="packed_alpha_to_vw_v1(alpha, pack_infos)", 
                globals={'packed_alpha_to_vw_v1':packed_alpha_to_vw_v1, 'alpha':alpha, 'pack_infos':pack_infos}
            ).blocked_autorange())
            print(Timer(
                stmt="packed_alpha_to_vw_v2(alpha, pack_infos)", 
                globals={'packed_alpha_to_vw_v2':packed_alpha_to_vw_v2, 'alpha':alpha, 'pack_infos':pack_infos}
            ).blocked_autorange())

    # test_boundary_vw()
    test_packed_vw()
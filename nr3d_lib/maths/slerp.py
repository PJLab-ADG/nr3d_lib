"""
Modified from roma (https://github.com/naver/roma)

@inproceedings{bregier2021deepregression,
    title={Deep Regression on Manifolds: a {3D} Rotation Case Study},
    author={Br{\'e}gier, Romain},
    journal={2021 International Conference on 3D Vision (3DV)},
    year={2021}
}
"""

__all__ = [
    'unitquat_slerp', 
    'unitquat_slerp_fast', 
    'axis_angle_slerp', 
    'matrix_slerp'
]

import torch
import torch.nn.functional as F

# from .transforms_from_pytorch3d import *
from nr3d_lib.maths.transforms import *

def unitquat_slerp(q0: torch.Tensor, q1: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Spherical linear interpolation between two unit quaternions.

    Args:
        q0 (torch.Tensor): [..., 4], unit quatertions; real-part first (WXYZ-convention)
        q1 (torch.Tensor): [..., 4], unit quatertions; real-part first (WXYZ-convention)
        weights (torch.Tensor): [...] interpolation weights, within range [0,1]\
            0.0 corresponding to q0 and 1.0 to q1 (B may contain multiple dimensions).

    Returns:
        torch.Tensor: Interpolated quaternions; real-part first (WXYZ-convention)
    """
    weights = weights.view(q0.shape[:-1])
    # Relative rotation
    rel_q = quat_raw_multiply(quat_invert(q0), q1)
    rel_rotvec = quat_to_axis_angle(standardize_quat(rel_q))
    # Relative rotations to apply
    rel_rotvecs = weights.unsqueeze(-1) * rel_rotvec
    rots = axis_angle_to_quat(rel_rotvecs) # Already unit quaterion
    interpolated_q = quat_raw_multiply(q0, rots)
    return interpolated_q

def unitquat_slerp_fast(q0: torch.Tensor, q1: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Spherical linear interpolation between two unit quaternions.
    This function requires less computations than :func:`unitquat_slerp`,
    but is **unsuitable for extrapolation (i.e.** ``weights`` **must be within [0,1])**.

    Args:
        q0 (torch.Tensor): [..., 4], unit quatertions; real-part first (WXYZ-convention)
        q1 (torch.Tensor): [..., 4], unit quatertions; real-part first (WXYZ-convention)
        weights (torch.Tensor): [...] interpolation weights, within range [0,1]\
            0.0 corresponding to q0 and 1.0 to q1 (B may contain multiple dimensions).

    Returns:
        torch.Tensor: Interpolated quaternions; real-part first (WXYZ-convention)
    """
    weights = weights.view(q0.shape[:-1])
    cos_omega = torch.sum(q0 * q1, dim=-1)
    if True: # shortest arc
        q1 = torch.where(cos_omega.unsqueeze(-1) < 0, -q1, q1)
        cos_omega = cos_omega.abs()
    # True when q0 and q1 are close.
    nearby_quaternions = cos_omega > (1.0 - 1e-3)
    
    # General approach
    omega = torch.acos(cos_omega)
    alpha = torch.sin((1-weights) * omega)
    beta = torch.sin(weights * omega)
    # Use linear interpolation for nearby quaternions
    alpha[nearby_quaternions] = 1 - weights[nearby_quaternions]
    beta[nearby_quaternions] = weights[nearby_quaternions]
    # Interpolation
    q = alpha.unsqueeze(-1) * q0 + beta.unsqueeze(-1) * q1
    q = F.normalize(q, dim=-1)
    return q

def axis_angle_slerp(aa0: torch.Tensor, aa1: torch.Tensor, weights: torch.Tensor):
    weights = weights.view(aa0.shape[:-1])
    q0 = axis_angle_to_quat(aa0)
    q1 = axis_angle_to_quat(aa1)
    interpolated_q = unitquat_slerp(q0, q1, weights)
    return quat_to_axis_angle(interpolated_q)

def matrix_slerp(R0: torch.Tensor, R1: torch.Tensor, weights: torch.Tensor):
    weights = weights.view(R0.shape[:-2])
    q0 = rot_to_quat(R0)
    q1 = rot_to_quat(R1)
    interpolated_q = unitquat_slerp(q0, q1, weights)
    return quat_to_rot(interpolated_q)

if __name__ == "__main__":
    def test_quaternion_slerp():
        from nr3d_lib.maths.internal.noma_transforms import unitquat_slerp as unitquat_slerp_
        from nr3d_lib.maths.internal.noma_transforms import quat_product, quat_conjugation, unitquat_to_rotvec, rotvec_to_unitquat
        device = torch.device('cuda')
        q0 = normalize_quat(torch.randn([7,4], dtype=torch.float, device=device))
        q1 = normalize_quat(torch.randn([7,4], dtype=torch.float, device=device))
        q0.requires_grad_(True)
        q1.requires_grad_(True)
        weights = torch.rand([7], dtype=torch.float, device=device)
        
        out = unitquat_slerp(q0, q1, weights)
        out.mean().backward()
        q0g = q0.grad.data.clone()
        q1g = q1.grad.data.clone()
        q0.grad = None
        q1.grad = None
        
        q0_ = q0[:, [1,2,3,0]]
        q1_ = q1[:, [1,2,3,0]]
        # out_ = unitquat_slerp_(q0_, q1_, weights) # NOTE: Wrong shape and op
        # out_ = out_[:, [3,0,1,2]]
        rel_q_ = quat_product(quat_conjugation(q0_), q1_)
        rel_rotvec_ = unitquat_to_rotvec(rel_q_, shortest_arc=True)
        # Relative rotations to apply
        rel_rotvecs_ = weights.unsqueeze(-1) * rel_rotvec_
        rots_ = rotvec_to_unitquat(rel_rotvecs_)
        out_ = quat_product(q0_, rots_)
        out_ = out_[:, [3,0,1,2]]
        out_.mean().backward()
        q0g_ = q0.grad.data.clone()
        q1g_ = q1.grad.data.clone()


        out2 = unitquat_slerp_fast(q0.data, q1.data, weights)
        
        _ = 1

    test_quaternion_slerp()
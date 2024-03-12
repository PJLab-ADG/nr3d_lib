__all__ = [
    # Self maintained
    'quat_to_rot_naive', 
    'rot_to_quat_naive', 
    'quat_scaling_to_matrix', 
    'inverse_transform_matrix', 
    'inverse_transform_matrix_np', 
    'get_rot_np', 
    'get_transform_np', 
    
    # Borrowed from pytorch3d
    'quat_invert', 
    'standardize_quat', 
    'normalize_quat', 
    'axis_angle_to_quat', 
    'quat_raw_multiply', 
    'quat_apply', 
    'quat_to_axis_angle', 
    'axis_angle_to_rot', 
    'quat_to_rot', 
    'rot6d_to_rot', 
    'rot_to_rot6d', 
    'rot_to_quat', 
]

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from typing import List, Literal, Union

import torch
import torch.nn.functional as F

from .common import normalize

# import pytorch3d.transforms as tr3d
def quat_invert(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Given a quaternion representing rotation, get the quaternion representing
    its inverse.

    Args:
        quaternion: Quaternions as tensor of shape (..., 4), with real part
            first, which must be versors (unit quaternions).

    Returns:
        The inverse, a tensor of quaternions of shape (..., 4).
    """

    scaling = torch.tensor([1, -1, -1, -1], device=quaternion.device)
    return quaternion * scaling

def standardize_quat(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)

def normalize_quat(quaternions: torch.Tensor) -> torch.Tensor:
    return standardize_quat(F.normalize(quaternions, dim=-1))

def axis_angle_to_quat(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions

def quat_to_rot(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def axis_angle_to_rot(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quat_to_rot(axis_angle_to_quat(axis_angle))

def quat_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)

def quat_apply(quaternion: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    """
    Apply the rotation given by a quaternion to a 3D point.
    Usual torch rules for broadcasting apply.

    Args:
        quaternion: Tensor of quaternions, real part first, of shape (..., 4).
        point: Tensor of 3D points of shape (..., 3).

    Returns:
        Tensor of rotated points of shape (..., 3).
    """
    if point.size(-1) != 3:
        raise ValueError(f"Points are not in 3D, {point.shape}.")
    real_parts = point.new_zeros(point.shape[:-1] + (1,))
    point_as_quaternion = torch.cat((real_parts, point), -1)
    out = quat_raw_multiply(
        quat_raw_multiply(quaternion, point_as_quaternion),
        quat_invert(quaternion),
    )
    return out[..., 1:]

def rot6d_to_rot(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

def quat_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles

def rot_to_rot6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def rot_to_quat(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :  # pyre-ignore[16]
    ].reshape(batch_dim + (4,))

def rot_to_quat_naive(R: torch.Tensor):
    # Rotation matrices to quaternion vectors
    batch_size, _,_ = R.shape
    q = torch.ones((batch_size, 4)).to(R.device)

    R00 = R[..., 0,0]
    R01 = R[..., 0, 1]
    R02 = R[..., 0, 2]
    R10 = R[..., 1, 0]
    R11 = R[..., 1, 1]
    R12 = R[..., 1, 2]
    R20 = R[..., 2, 0]
    R21 = R[..., 2, 1]
    R22 = R[..., 2, 2]

    q[..., 0] = torch.sqrt(1.0 + R00 + R11 + R22)/2 # qw
    q[..., 1] = (R21 - R12) / (4 * q[:, 0]) # qx
    q[..., 2] = (R02 - R20) / (4 * q[:, 0]) # qy
    q[..., 3] = (R10 - R01) / (4 * q[:, 0]) # qz
    return q

def quat_to_rot_naive(q: torch.Tensor):
    # Quaternion vectors to rotation matrices
    prefix, _ = q.shape[:-1]
    q = F.normalize(q, dim=-1)
    R = torch.ones([*prefix, 3, 3]).to(q.device)
    qr = q[... ,0]
    qi = q[..., 1]
    qj = q[..., 2]
    qk = q[..., 3]
    R[..., 0, 0]=1-2 * (qj**2 + qk**2)
    R[..., 0, 1] = 2 * (qj *qi -qk*qr)
    R[..., 0, 2] = 2 * (qi * qk + qr * qj)
    R[..., 1, 0] = 2 * (qj * qi + qk * qr)
    R[..., 1, 1] = 1-2 * (qi**2 + qk**2)
    R[..., 1, 2] = 2*(qj*qk - qi*qr)
    R[..., 2, 0] = 2 * (qk * qi-qj * qr)
    R[..., 2, 1] = 2 * (qj*qk + qi*qr)
    R[..., 2, 2] = 1-2 * (qi**2 + qj**2)
    return R

# @profile
def inverse_transform_matrix(input: torch.Tensor) -> torch.Tensor:
    """ Inverse given transformation matrices
        NOTE: Must be in left-multiply conventions

    Args:
        input (torch.Tensor): [..., 4,4] The given transformation matrices

    Returns:
        torch.Tensor: [..., 4,4] Inversed transformation matrices
    """
    prefix = input.shape[0:-2]
    R_inv = input[..., :3, :3].transpose(-1,-2)
    # t_inv = -torch.bmm(R_inv, t.unsqueeze(-1))
    # t_inv = torch.einsum('...ij,...j->...i', R_inv, -input[..., :3, 3])
    t_inv = -(R_inv * input[..., None, :3, 3]).sum(-1) # NOTE: einsum too slow.
    
    inv = torch.zeros([*prefix, 4, 4], device=input.device, dtype=input.dtype)
    inv[..., :3, :3], inv[..., :3, 3], inv[..., 3, 3] = R_inv, t_inv, 1.

    assert input.shape==inv.shape
    return inv

def inverse_transform_matrix_np(input: np.ndarray) -> np.ndarray:
    """ Inverse given transformation matrices
        NOTE: Must be in left-multiply conventions

    Args:
        input (np.ndarray): [..., 4,4] The given transformation matrices

    Returns:
        np.ndarray: [..., 4,4] Inversed transformation matrices
    """
    prefix = input.shape[0:-2]
    R_inv = np.moveaxis(input[..., :3, :3], -2, -1)
    t_inv = -(R_inv * input[..., None, :3, 3]).sum(-1) # NOTE: einsum too slow.
    inv = np.zeros([*prefix, 4, 4], dtype=input.dtype)
    inv[..., :3, :3], inv[..., :3, 3], inv[..., 3, 3] = R_inv, t_inv, 1.
    return inv

def get_transform_np(
    t: np.ndarray, *, 
    ox: Union[List, np.ndarray]=None, 
    oy: Union[List, np.ndarray]=None, 
    oz: Union[List, np.ndarray]=None, 
    preserve: Literal['x','y','z']=None) -> np.ndarray:
    """ Calcuate the transform matrix given translation `t` and direction vectors (two of ox/oy/oz) of current pose

    Args:
        t (np.ndarray): [3,] Given translation
        ox (Union[List, np.ndarray], optional): [3,] Given x-axis' orientation vector in the original coords. Defaults to None.
        oy (Union[List, np.ndarray], optional): [3,] Given y-axis' orientation vector in the original coords. Defaults to None.
        oz (Union[List, np.ndarray], optional): [3,] Given z-axis' orientation vector in the original coords. Defaults to None.
        preserve (Literal['x','y','z'], optional): 
            Which axis's original direction is preserved between the given to axes' vectors.
            Choose one among x/y/z. Defaults to None.

    Returns:
        np.ndarray: [4,4] The transformation matrix or pose matrix (From original coords to current coords)
    """
    rot = get_rot_np(ox=ox, oy=oy, oz=oz, preserve=preserve)
    mat = np.concatenate((rot, np.asarray(t)[..., None]), axis=-1)
    hom_vec = np.array([[0., 0., 0., 1.]])
    if len(mat.shape) > 2:
        hom_vec = np.tile(hom_vec, [mat.shape[0], 1, 1])
    mat = np.concatenate((mat, hom_vec), axis=-2)
    return mat

def get_rot_np(
    *, 
    ox: Union[List, np.ndarray]=None, 
    oy: Union[List, np.ndarray]=None, 
    oz: Union[List, np.ndarray]=None, 
    preserve: Literal['x','y','z']=None, 
    left_handed=False) -> np.ndarray:
    """Calculate the rotation matrix based on the direction vectors of the rotated coordinate axes in the original coordinate system.
        NOTE: Works iff given 2 known values in [ox,oy,oz].

    Args:
        ox (Union[List, np.ndarray], optional): [3,] Given x-axis' orientation vector in the original coords. Defaults to None.
        oy (Union[List, np.ndarray], optional): [3,] Given y-axis' orientation vector in the original coords. Defaults to None.
        oz (Union[List, np.ndarray], optional): [3,] Given z-axis' orientation vector in the original coords. Defaults to None.
        preserve (Literal['x','y','z'], optional): 
            Which axis's original direction is preserved between the given to axes' vectors.
            Choose one among x/y/z. Defaults to None.
        left_handed (bool, optional): Whether to calculate a left-handed coords. Defaults to False.

    Returns:
        np.ndarray: [3,3] The rotation matrix to rotate vectors from the original coords to the new coords.
    """
    assert int(ox is None) + int(oy is None) + int(oz is None) == 1, "Please specify only two of the ox,oy,oz vectors"
    # NOTE: Right handed coordinate systemcs
    #       ox = oy X oz
    #       oy = oz X ox
    #       oz = ox X oy
    if left_handed:
        sgn = -1
    else:
        sgn = 1
    
    if ox is None:
        oy = np.array(oy)
        oz = np.array(oz)
        if preserve == 'y':
            oy = normalize(oy)
            ox = normalize(np.cross(oy, oz) * sgn)
            oz = normalize(np.cross(ox, oy) * sgn)
        elif preserve == 'z':
            oz = normalize(oz)
            ox = normalize(np.cross(oy, oz) * sgn)
            oy = normalize(np.cross(oz, ox) * sgn)
        else:
            raise ValueError(f"Invalid preserve={preserve}. should be one of [y,z]")
    elif oy is None:
        ox = np.array(ox)
        oz = np.array(oz)
        if preserve == 'x':
            ox = normalize(ox)
            oy = normalize(np.cross(oz, ox) * sgn)
            oz = normalize(np.cross(ox, oy) * sgn)
        elif preserve == 'z':
            oz = normalize(oz)
            oy = normalize(np.cross(oz, ox) * sgn)
            ox = normalize(np.cross(oy, oz) * sgn)
        else:
            raise ValueError(f"Invalid preserve={preserve}. should be one of [x,z]")
    elif oz is None:
        ox = np.array(ox)
        oy = np.array(oy)
        if preserve == 'x':
            ox = normalize(ox)
            oz = normalize(np.cross(ox, oy) * sgn)
            oy = normalize(np.cross(oz, ox) * sgn)
        elif preserve == 'y':
            oy = normalize(oy)
            oz = normalize(np.cross(ox, oy) * sgn)
            ox = normalize(np.cross(oy, oz) * sgn)
        else:
            raise ValueError(f"Invalid preserve={preserve}. should be one of [x,y]")
    return np.stack((ox, oy, oz), axis=-1)

def quat_scaling_to_matrix(scaling: torch.Tensor, quat: torch.Tensor):
    """
    First scaling, then rotation
    """
    L = torch.diagflat(scaling)
    R = quat_to_rot(quat)
    
    # return R @ L
    return (R.unsqueeze(-1) * L.unsqueeze(-3)).sum(-2) # More accurate matrix multiplication
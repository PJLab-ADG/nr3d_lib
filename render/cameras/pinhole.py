"""
@file   pinhole.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Standard pinhole camera math ops.
"""

__all__ = [
    'camera_mat_from_hwf', 
    'pinhole_lift', 
    'pinhole_lift_cf', 
    'pinhole_view_frustum', 
    'pinhole_get_rays', 
    'pinhole_get_rays_np'
]

import numpy as np
from typing import Tuple, Union

import torch
import torch.nn.functional as F

from nr3d_lib.geometry import inverse_transform_matrix

def camera_mat_from_hwf(
    H: int, W: int, fx: float, fy: float=None, *, prefix=(), 
    device=torch.device('cuda'), dtype=torch.float) -> torch.Tensor:
    """ Calculate camera intrinsics matrix from H, W, focal (fx and fy)
    NOTE: Assume principle point to be at the image center

    Args:
        H (int): Image height in pixels
        W (int): Image width in pixels
        fx (float): Focal - x
        fy (float, optional): Focal - y. Defaults to None.
        prefix (tuple, optional): Optional prefix-batch-dims. Defaults to ().
        device (torch.device, optional): Output torch.device. Defaults to torch.device('cuda').
        dtype (torch.dtype, optional): Output torch.dtype. Defaults to torch.float.

    Returns:
        torch.Tensor: [*prefix, 4, 4] The calculated intrinsics mat
    """
    if fy is None: fy = fx
    intr = torch.eye(4, device=device, dtype=dtype)
    intr[0,0] = fx
    intr[1,1] = fy
    intr[0,2] = W/2.+0.5
    intr[1,2] = H/2.+0.5
    return intr.tile((*prefix,1,1))

def pinhole_lift(u: torch.Tensor, v: torch.Tensor, d: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """ Lift 2D pixel locations to 3D points in pinhole cameras

    Args:
        u (torch.Tensor): Given pixel locations, in range [0,W)
        v (torch.Tensor): Given pixel locations, in range [0,H)
        d (torch.Tensor): Given pixel depth
        intrinsics (torch.Tensor): Pinhole camera intrinsics mat

    Returns:
        torch.Tensor: The lifted 3D points
    """
    device = u.device
    # parse intrinsics
    intrinsics = intrinsics.to(device)
    fx = intrinsics[..., 0, 0]
    fy = intrinsics[..., 1, 1]
    cx = intrinsics[..., 0, 2]
    cy = intrinsics[..., 1, 2]
    sk = intrinsics[..., 0, 1]

    x_lift = (u - cx + cy*sk/fy - sk*v/fy) / fx * d
    y_lift = (v - cy) / fy * d
    # Homogeneous
    return torch.stack((x_lift, y_lift, d, d.new_ones(d.shape)), dim=-1)

def pinhole_lift_cf(
    u: torch.Tensor, v: torch.Tensor, d: torch.Tensor, 
    cx: Union[float, torch.Tensor], cy: Union[float, torch.Tensor], 
    fx: Union[float, torch.Tensor], fy: Union[float, torch.Tensor]) -> torch.Tensor:
    """ Lift 2D pixel locations to 3D points in pinhole cameras

    Args:
        u (torch.Tensor): Given pixel locations, in range [0,W)
        v (torch.Tensor): Given pixel locations, in range [0,H)
        d (torch.Tensor): Given pixel depth
        cx (Union[float, torch.Tensor]): Pinhole camera principle point-x in pixels
        cy (Union[float, torch.Tensor]): Pinhole camera principle point-y in pixels
        fx (Union[float, torch.Tensor]): Pinhole camera focal-x in pixels
        fy (Union[float, torch.Tensor]): Pinhole camera focal-y in pixels

    Returns:
        torch.Tensor: The lifted 3D points
    """
    x_lift = (u - cx) / fx * d
    y_lift = (v - cy) / fy * d
    # Homogeneous
    return torch.stack((x_lift, y_lift, d, d.new_ones(d.shape)), dim=-1)

def pinhole_view_frustum(
    c2w: torch.Tensor, 
    cx: torch.Tensor, cy: torch.Tensor, fx: torch.Tensor, fy: torch.Tensor, 
    near_clip: float=None, far_clip: float=None) -> torch.Tensor:
    """ Calculate pinhole camera frustum planes 

    Args:
        c2w (torch.Tensor): [..., 4, 4] camera to world transform matrix
        cx (torch.Tensor): [...] principle point - x
        cy (torch.Tensor): [...] principle point - y
        fx (torch.Tensor): [...] focal - x
        fy (torch.Tensor): [...] focal - y
        near_clip (float, optional): Near clip value. Defaults to None.
        far_clip (float, optional): Far clip value. Defaults to None.

    Returns:
        torch.Tensor: [..., N_plane, 4]. Already normalized plane repr. The last dim (4) = normals (3) + distances (1)
    """
    
    prefix = c2w.shape[:-2]
    w2c = inverse_transform_matrix(c2w)
    
    # NOTE: We live in an OpenCV world.
    vx, vy, vw = (fx / cx).unsqueeze(-1) * w2c[..., 0, :], (fy / cy).unsqueeze(-1) * w2c[..., 1, :], w2c[..., 2, :]
    # plane_right, plane_left, plane_bottom, plane_top
    planes = [vw - vx, vw + vx, vw - vy, vw + vy]
    
    vz = c2w[..., :3, 2]
    if near_clip is not None:
        # NOTE: Einsum too slow. directly use mul and sum.
        near_center_pt = (c2w[..., :3,:3] * c2w.new_tensor([0.,0.,near_clip])).sum(-1)+c2w[..., :3,3]
        plane_near = vz.new_zeros([*prefix,4])
        plane_near[..., :3] = vz
        # Near clip plane pass through near center point
        plane_near[..., 3] = -(near_center_pt * vz).sum(-1)
        planes.append(plane_near)
    if far_clip is not None:
        # NOTE: Einsum too slow. directly use mul and sum.
        far_center_pt = (c2w[..., :3,:3] * c2w.new_tensor([0.,0.,far_clip])).sum(-1)+c2w[..., :3,3]
        plane_far = vz.new_zeros([*prefix,4])
        plane_far[..., :3] = -vz
        # Far clip plane pass through far center point
        plane_far[..., 3] = -(far_center_pt * -vz).sum(-1)
        planes.append(plane_far)
    
    planes = torch.stack(planes, -2)
    planes[..., :4, :] /= planes[..., :4, :3].norm(dim=-1, keepdim=True) # NOTE: Already normalized
    return planes

def pinhole_get_rays(c2w: torch.Tensor, intrinsics: torch.Tensor, H: int, W: int, N_rays=-1) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Pinhole camera get all rays

        < opencv / colmap convention, standard pinhole camera >
        the camera is facing [+z] direction, x right, y downwards
                    z
                   ↗
                  /
                 /
                o------> x
                |
                |
                |
                ↓ 
                y

    Args:
        c2w (torch.Tensor): [..., 3/4, 4] camera to world matrices
        intrinsics (torch.Tensor): [..., 3, 3] pinhole camera intrinsics
        H (int): Image height in pixels
        W (int): Image width in pixels
        N_rays (int, optional): Optionally uniformaly sample some rays. Defaults to -1.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: [..., H*W, 3], Lifted rays_o, rays_d
    """

    prefix, device = c2w.shape[:-2], c2w.device
    i, j = torch.meshgrid(torch.linspace(0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device), indexing='ij')
    i, j = i.t().reshape([*[1]*len(prefix), H*W]).expand([*prefix, H*W]) + 0.5, \
        j.t().reshape([*[1]*len(prefix), H*W]).expand([*prefix, H*W]) + 0.5 # Add half pixel

    if N_rays > 0:
        N_rays = min(N_rays, H*W)
        #----------- option 1: full image uniformly randomize
        # select_inds = torch.tensor(np.random.choice(H*W, size=[*prefix, N_rays], replace=False), device=device)
        select_inds = torch.randint(0, H*W, size=[N_rays], device=device).expand([*prefix, N_rays]) # may duplicate
        #----------- option 2: H/W seperately randomize
        # select_hs = torch.randint(0, H, size=[N_rays], device=device)
        # select_ws = torch.randint(0, W, size=[N_rays], device=device)
        # select_inds = select_hs * W + select_ws
        # select_inds = select_inds.expand([*prefix, N_rays])

        i, j = torch.gather(i, -1, select_inds), torch.gather(j, -1, select_inds)
    else:
        # select_inds = torch.arange(H*W, device=device).expand([*prefix, H*W])
        select_inds = None

    lifted_directions = pinhole_lift(i, j, torch.ones_like(i, device=device), intrinsics=intrinsics.unsqueeze(-3))
    rays_d = F.normalize((c2w[..., None, :3, :3] * lifted_directions[..., None, :3]).sum(-1), dim=-1)
    rays_o = c2w[..., None, :3, 3].expand_as(rays_d)

    if N_rays > 0:
        return rays_o, rays_d, select_inds
    else:
        return rays_o, rays_d

def pinhole_get_rays_np(c2w: np.ndarray, intrinsics:np.ndarray, H: int, W: int) -> Tuple[np.ndarray, np.ndarray]:
    """ Pinhole camera get all rays, numpy impl

    < opencv / colmap convention, standard pinhole camera >
    the camera is facing [+z] direction, x right, y downwards
                z
               ↗
              /
             /
            o------> x
            |
            |
            |
            ↓ 
            y

    Args:
        c2w (np.ndarray): [..., 3/4, 4] camera to world matrices
        intrinsics (np.ndarray): [..., 3, 3] pinhole camera intrinsics
        H (int): Image height in pixels
        W (int): Image width in pixels

    Returns:
        Tuple[np.ndarray, np.ndarray]: [..., H*W, 3], Lifted rays_o, rays_d
    """
    prefix = c2w.shape[:-2] # [...]

    # [H, W]
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    # [H*W]
    u = u.reshape(-1).astype(dtype=np.float32) + 0.5    # Add half pixel
    v = v.reshape(-1).astype(dtype=np.float32) + 0.5
    
    # [3, H*W]
    pixels = np.stack((u, v, np.ones_like(u)), axis=0)  

    # [3, H*W]
    rays_d = np.matmul(np.linalg.inv(intrinsics[:3, :3]), pixels)
    
    # [..., 3, H*W] = [..., 3, 3] @ [1,1,...,  3, H*W], with broadcasting
    rays_d = np.matmul(c2w[..., :3, :3], rays_d.reshape([*len(prefix)*[1], 3, H*W]))
    # [..., H*W, 3]
    rays_d = np.moveaxis(rays_d, -1, -2)

    # [..., 1, 3] -> [..., H*W, 3]
    rays_o = np.tile(c2w[..., None, :3, 3], [*len(prefix)*[1], H*W, 1])

    return rays_o, rays_d

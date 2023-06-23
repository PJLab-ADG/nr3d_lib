"""
@file   fisheye.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Fisheye camera math ops.
        Refer to: https://docs.opencv.org/4.7.0/db/d58/group__calib3d__fisheye.html
"""

__all__ = [
    'fisheye_distort_points_cpu', 
    'fisheye_undistort_points_cpu'
]

import cv2
import numpy as np
from typing import Optional

import torch

from nr3d_lib.utils import check_to_torch

def fisheye_distort_points_cpu(
    points: torch.Tensor, # Pixel coords of input undistorted image
    K: torch.Tensor, # Camera matrix for output distorted points
    dist: torch.Tensor, 
    new_K: Optional[torch.Tensor] = None, # additional K for input undistorted points (to normalize input points)
) -> torch.Tensor:
    if new_K is None:
        new_K = K
    
    # Convert 2D points from pixels to normalized camera coordinates
    new_cx: torch.Tensor = new_K[..., 0:1, 2]  # princial point in x (Bx1)
    new_cy: torch.Tensor = new_K[..., 1:2, 2]  # princial point in y (Bx1)
    new_fx: torch.Tensor = new_K[..., 0:1, 0]  # focal in x (Bx1)
    new_fy: torch.Tensor = new_K[..., 1:2, 1]  # focal in y (Bx1)
    
    # This is equivalent to K^-1 [u,v,1]^T
    x: torch.Tensor = (points[..., 0] - new_cx) / new_fx
    y: torch.Tensor = (points[..., 1] - new_cy) / new_fy
    
    points = torch.stack([x,y], dim=-1)
    
    distorted = cv2.fisheye.distortPoints(
        points.data.cpu().numpy(), # Normalized pixel coords of input undistorted image
        K.data.cpu().numpy(), # Camera matrix for output distorted points
        dist.data.cpu().numpy()
    )
    
    return check_to_torch(distorted, ref=points)

def fisheye_distort_points(
    points: torch.Tensor, # Pixel coords of input undistorted image
    K: torch.Tensor, # Camera matrix for output distorted points
    dist: torch.Tensor, 
    new_K: Optional[torch.Tensor] = None, # additional K for input undistorted points (to normalize input points)
) -> torch.Tensor:
    if new_K is None:
        new_K = K
    raise NotImplementedError

def fisheye_undistort_points_cpu(
    points: torch.Tensor, # Pixel coords of input distorted image
    K: torch.Tensor, # Camera matrix for input distorted points
    dist: torch.Tensor, 
    new_K: Optional[torch.Tensor] = None, # additional K for output undistorted points
) -> torch.Tensor:
    prefix = points.shape[0:-2]
    # NOTE: K and dist should be equal for all
    # K = K.expand([*prefix,3,3])
    # dist = dist.expand([*prefix,dist.shape[-1]])
    if new_K is None:
        new_K = K
    undistorted = cv2.fisheye.undistortPoints(
        points.flatten(0,-3).data.cpu().numpy(), # Normalized pixel coords of input distorted image
        K.data.cpu().numpy(), # Camera matrix for input distorted points
        dist.data.cpu().numpy(), 
        None, 
        new_K.data.cpu().numpy()
    )
    undistorted = check_to_torch(undistorted, ref=points)
    undistorted = undistorted.unflatten(0, prefix)
    return undistorted

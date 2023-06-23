import cv2
import numpy as np
from typing import Literal, Tuple, Union

import torch
import torch.nn.functional as F

def chamfer_distance_pytorch(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    NOTE: Not memory efficient, OOM when x,y are large.
    """
    assert (x.dim() == y.dim()) and (x.dim() >= 2) and x.shape[-1] == y.shape[-1]
    x_i = x.unsqueeze(-2) # [..., N1, 1,  D]
    y_j = x.unsqueeze(-3) # [..., 1,  N2, D]
    D_ij = ((x_i - y_j)**2).sum(dim=-1) # [..., N1, N2]
    cham_x = D_ij.min(dim=-1).values
    cham_y = D_ij.min(dim=-2).values    
    return cham_x, cham_y
    
def chamfer_distance_pt3d(x: torch.Tensor, y: torch.Tensor, norm: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Chamfer distance between two pointclouds x and y.

    Args:
        x (torch.Tensor): [(B, )N1, 3], point clouds or a batch of point clouds.
        y (torch.Tensor): [(B, )N2, 3], point clouds or a batch of point clouds.
        norm (int, optional): the norm used for the distance. Supports 1 for L1 and 2 for L2. Defaults to 2.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: [(B, )N1, 3], [(B, )N2, 3], the distances between the pointclouds
    """
    from pytorch3d.ops.knn import knn_points
    if not ((norm == 1) or (norm == 2)):
        raise ValueError("Support for 1 or 2 norm.")

    _x = x[None] if len(x.shape) == 2 else x
    _y = y[None] if len(y.shape) == 2 else y

    if _y.shape[0] != _x.shape[0] or _y.shape[2] != _x.shape[2]:
        raise ValueError("y does not have the correct shape.")

    x_nn = knn_points(_x, _y, norm=norm, K=1)
    y_nn = knn_points(_y, _x, norm=norm, K=1)

    cham_x = x_nn.dists[..., 0]  # (N, P1)
    cham_y = y_nn.dists[..., 0]  # (N, P2)
    cham_x = cham_x[0] if len(x.shape) == 2 else cham_x
    cham_y = cham_y[0] if len(y.shape) == 2 else cham_y
    return cham_x, cham_y

def chamfer_distance_borrowed_from_pt3d(x: torch.Tensor, y: torch.Tensor, norm: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Chamfer distance between two pointclouds x and y. (Borrowed from pytorch3d)

    Args:
        x (torch.Tensor): [(B, )N1, 3], point clouds or a batch of point clouds.
        y (torch.Tensor): [(B, )N2, 3], point clouds or a batch of point clouds.
        norm (int, optional): the norm used for the distance. Supports 1 for L1 and 2 for L2. Defaults to 2.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: [(B, )N1, 3], [(B, )N2, 3], the distances between the pointclouds
    """
    from nr3d_lib.geometry.knn_from_pytorch3d import knn_points
    if not ((norm == 1) or (norm == 2)):
        raise ValueError("Support for 1 or 2 norm.")

    _x = x[None] if len(x.shape) == 2 else x
    _y = y[None] if len(y.shape) == 2 else y

    if _y.shape[0] != _x.shape[0] or _y.shape[2] != _x.shape[2]:
        raise ValueError("y does not have the correct shape.")

    x_nn = knn_points(_x, _y, norm=norm, K=1)
    y_nn = knn_points(_y, _x, norm=norm, K=1)

    cham_x = x_nn.dists[..., 0]  # (N, P1)
    cham_y = y_nn.dists[..., 0]  # (N, P2)
    cham_x = cham_x[0] if len(x.shape) == 2 else cham_x
    cham_y = cham_y[0] if len(y.shape) == 2 else cham_y
    return cham_x, cham_y
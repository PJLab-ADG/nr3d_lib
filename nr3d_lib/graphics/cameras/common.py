
__all__ = [
    'decompose_intr_c2w_from_proj_np', 
    'look_at_np', 
    'intr_to_glProj', 
    'box_inside_frustum', 
    'sphere_inside_frustum', 
    'pts_inside_frustum', 
]

import cv2
import numpy as np
from typing import Literal, Tuple, Union

import torch
import torch.nn.functional as F

from nr3d_lib.maths import normalize, get_transform_np

def decompose_intr_c2w_from_proj_np(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """ Load intr, c2w matrices from given projection matrix P
    Modified from IDR https://github.com/lioryariv/idr

    Args:
        P (np.ndarray): The given projection matrix = K @ w2c

    Returns:
        Tuple[np.ndarray, np.ndarray]: [4,4], [4,4] pinhole intr matrix and camera to world transform matrix
    """
    out = cv2.decomposeProjectionMatrix(P[:3, :4])
    K = out[0]
    R = out[1]
    t = out[2]

    K = K/K[2,2]
    intrinsics = np.eye(4, dtype=np.float32)
    intrinsics[:3, :3] = K

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R.transpose()
    c2w[:3,3] = (t[:3] / t[3])[:,0]

    return intrinsics, c2w

def look_at_np(
    cam_location: Union[np.ndarray, list], 
    point: Union[np.ndarray, list], 
    up: Union[np.ndarray, list]=np.array([0., -1., 0.]),   # opencv convention, 
    convention: Literal['opencv'] = 'opencv'
    ) -> np.ndarray:
    """ Calculate the camera to world matrix given camera location, focus point and vertical-up direction vector
        NOTE: openCV convention:    facing [+z] direction
        - get_transform_np() assumes that "forward" is +z
        - get_transform_np() assumes that "down" is +y ("up" is -y)

    Args:
        cam_location (np.ndarray): [3,] Camera location
        point (np.ndarray): [3,] The point that the camera is gazing at.
        up (np.ndarray, optional): The vertical-up direction vector. Defaults to np.array([0., -1., 0.])#opencvconvention.

    Returns:
        np.ndarray: [4,4] The camera to world transformation matrix (i.e. the pose matrix of the camera in world coordinate system)
    """
    cam_location, point, up = np.asarray(cam_location), np.asarray(point), np.asarray(up)
    forward = normalize(point - cam_location)
    return get_transform_np(cam_location, oz=forward, oy=-up, preserve='z')

def intr_to_glProj(intr: torch.Tensor, near=None, far=None):
    """
    From pinhole intrinsics matrix to openGL's projection matrix
    """
    if near is None: near = 0.
    device = intr.device
    dtype = intr.dtype
    prefix = intr.shape[:-2]
    proj = torch.zeros([*prefix, 4, 4], device=device, dtype=dtype)
    proj[..., 0, 0] = intr[..., 0, 0] / intr[..., 0, 2] # fx/cx
    proj[..., 1, 1] = intr[..., 1, 1] / intr[..., 1, 2] # fy/cy
    proj[..., 2, 2] = -1 if far is None else -(far+near)/(far-near)
    proj[..., 2, 3] = 0 if far is None else -2*far*near/(far-near)
    proj[..., 3, 2] = -1
    return proj

def box_inside_frustum(box_verts: torch.Tensor, frustum_planes_nd: torch.Tensor, *, holistic=False, normalized=True) -> torch.BoolTensor:
    """ Check whether N boxes are inside P planes

    Args:
        box_verts (torch.Tensor): [N, 8, 3] The given N boxes vertices
        frustum_planes_nd (torch.Tensor): [..., P, 4] The given P planes representation.
        holistic (bool, optional): Whether the holistic box body must be in the planes (T), or any part counts (F). Defaults to False.
        normalized (bool, optional): Whether the plane repr is already normalized. Defaults to True.

    Raises:
        NotImplementedError: Not implemented for now

    Returns:
        torch.BoolTensor: [..., N] Inside checks of the given N boxes w.r.t. all given P planes
    """
    raise NotImplementedError("jianfei: box_inside_frustum not implemented correctly. Should change to IoU calculation.")
    
    if not normalized:
        frustum_planes_nd = frustum_planes_nd / frustum_planes_nd[..., :3].norm(dim=-1, keepdim=True)
    normals = frustum_planes_nd[..., :3]   # [P, 3]
    distances = frustum_planes_nd[..., 3]  # [P,]
    # [N, 8, 1, 3] * [P, 3] -> [N, 8, P, 3] -> [N, 8, P] + [P] -> [N, 8, P] --(all P planes)--> [N, 8]
    check_inside = ((normals * box_verts.unsqueeze(-2)).sum(-1) + distances > 0).all(dim=-1)
    if not holistic:
        # [N, 8] --(any of 8 points)--> [N]
        return check_inside.any(dim=-1)
    else:
        # [N, 8] --(all of 8 points)--> [N]
        return check_inside.all(dim=-1)

def sphere_inside_frustum(sph_centers_radius: torch.Tensor, frustum_planes_nd: torch.Tensor, *, holistic=False, normalized=True) -> torch.BoolTensor:
    """ Check whether N given spheres are inside all P given planes
        NOTE: `...` means arbitary prefix-batch-dims

    Args:
        sph_centers_radius (torch.Tensor): [..., N, 4] The given N spheres representation, center (3) + radius (1)
        frustum_planes_nd (torch.Tensor): [..., P, 4] The given P planes representation. The last dim (4) = normals (3) + distance (1)
        holistic (bool, optional): Whether the holistic sphere body must be in the planes (T), or any part counts (F). Defaults to False.
        normalized (bool, optional): Whether the plane repr is already normalized. Defaults to True.

    Returns:
        torch.BoolTensor: [..., N] Inside checks of the given N spheres w.r.t. all given P planes
    """
    if not normalized:
        frustum_planes_nd = frustum_planes_nd / frustum_planes_nd[..., :3].norm(dim=-1, keepdim=True).clamp_min(1e-5)
    normals = frustum_planes_nd[..., :3]   # [..., P, 3]
    distances = frustum_planes_nd[..., 3]  # [..., P,]
    sph_centers = sph_centers_radius[..., :3]   # [..., N, 3]
    sph_radius = sph_centers_radius[..., 3]     # [..., N]
    if not holistic:       
        #---- Allow only partially inside
        # [..., 1, P, 3] * [..., N, 1, 3] -> [..., N, P, 3] -> [..., N, P] + [..., 1, P] + [..., N, 1] -> [..., N, P] --(all P planes)--> [..., N]
        check_inside = ((normals.unsqueeze(-3) * sph_centers.unsqueeze(-2)).sum(-1) + distances.unsqueeze(-2) + sph_radius.unsqueeze(-1) > 0).all(dim=-1)
        
        # [..., P, 3] * [N, ..., 1, 3] -> [N, ..., P, 3] -> [N, ..., P] + [..., P] + [N, ..., 1] -> [N, ..., P] --(all P planes)--> [N, ...]
        # check_inside = ((normals * sph_centers.unsqueeze(-2)).sum(-1) + distances + sph_radius.unsqueeze(-1) > 0).all(dim=-1)
    else:
        #---- All of the sphere must be inside the planes
        check_inside = ((normals.unsqueeze(-3) * sph_centers.unsqueeze(-2)).sum(-1) + distances.unsqueeze(-2) - sph_radius.unsqueeze(-1) >= 0).all(dim=-1)
        # check_inside = ((normals * sph_centers.unsqueeze(-2)).sum(-1) + distances - sph_radius.unsqueeze(-1) >= 0).all(dim=-1)
    return check_inside

def pts_inside_frustum(pts: torch.Tensor, frustum_planes_nd: torch.Tensor, *, normalized=True) -> torch.BoolTensor:
    """ Check whether N given pts inside all P given planes
        NOTE: `...` means arbitary prefix-batch-dims

    Args:
        pts (torch.Tensor): [..., N, 3] The given N pts
        frustum_planes_nd (torch.Tensor): [..., P, 4] The given P planes representation.
            The last dim (4) = normals (3) + distance (1)
        normalized (bool, optional): Whether the plane repr is already normalized. Defaults to True.

    Returns:
        torch.BoolTensor: [..., N]
    """
    if not normalized:
        frustum_planes_nd = frustum_planes_nd / frustum_planes_nd[..., :3].norm(dim=-1, keepdim=True)
    normals = frustum_planes_nd[..., :3]   # [P, 3]
    distances = frustum_planes_nd[..., 3]  # [P,]
    # [..., 1, P, 3] * [..., N, 1, 3] -> [..., N, P, 3] -> [..., N, P] + [..., 1, P] -> [..., N, P] --(all P Planes)--> [..., N]
    check_inside = ((normals.unsqueeze(-3) * pts.unsqueeze(-2)).sum(-1) + distances.unsqueeze(-2) > 0).all(dim=-1)

    # [..., P, 3] * [N, ..., 1, 3] -> [N, ..., P, 3] -> [N, ..., P] + [..., P] -> [N, ..., P] --(all P Planes)--> [N, ...]
    # check_inside = ((normals * pts.unsqueeze(-2)).sum(-1) + distances > 0).all(dim=-1)
    return check_inside

if __name__ == "__main__":
    def unit_test(device=torch.device('cuda')):
        from icecream import ic
        prefix = (17, 14, 13)
        planes = torch.randn([*prefix, 8, 4], device=device)
        spheres = torch.randn([*prefix, 9, 4], device=device)
        inside = sphere_inside_frustum(spheres, planes, holistic=False, normalized=False)
        ic(inside.shape)
        
        if (dims:=inside.dim()) > 1:
            # Dealing with multiple cameras or batched frames: if any camera suffices, then this drawable node suffices.
            inside = inside.any(dim=0) if dims == 2 else (inside.sum(dim=list(range(0,dims-1))) > 0)

        ic(inside.shape)

        planes = torch.randn([*prefix, 8, 4], device=device)
        pts = torch.randn([*prefix, 1000, 3], device=device)
        inside = pts_inside_frustum(pts, planes, normalized=False)
        ic(inside.shape)

    unit_test()
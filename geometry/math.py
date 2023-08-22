"""
@file   math.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Some common geometry helpers and algebraic math functions.
"""

__all__ = [
    'decompose_K_Rt_from_P', 
    'normalize', 
    'get_rotation_matrix', 
    'get_transformation_matrix', 
    'look_at_opencv', 
    'gmean', 
    'rot_to_quat', 
    'quat_to_rot', 
    'inverse_transform_matrix', 
    'inverse_transform_matrix_np', 
    'skew_symmetric', 
    'intr_to_gl_proj', 
    'box_inside_planes', 
    'sphere_inside_planes', 
    'pts_inside_planes'
]

import cv2
import numpy as np
from typing import List, Literal, Tuple, Union

import torch
import torch.nn.functional as F

def decompose_K_Rt_from_P(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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

def normalize(vec):
    return vec / (np.linalg.norm(vec, axis=-1, keepdims=True) + 1e-9)

def get_rotation_matrix(
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

def get_transformation_matrix(
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
    rot = get_rotation_matrix(ox=ox, oy=oy, oz=oz, preserve=preserve)
    mat = np.concatenate((rot, np.asarray(t)[..., None]), axis=-1)
    hom_vec = np.array([[0., 0., 0., 1.]])
    if len(mat.shape) > 2:
        hom_vec = np.tile(hom_vec, [mat.shape[0], 1, 1])
    mat = np.concatenate((mat, hom_vec), axis=-2)
    return mat

def look_at_opencv(
    cam_location: np.ndarray, 
    point: np.ndarray, 
    up=np.array([0., -1., 0.])  # opencv convention
    ) -> np.ndarray:
    """ Calculate the camera to world matrix given camera location, focus point and vertical-up direction vector
        NOTE: openCV convention:    facing [+z] direction
        - get_transformation_matrix() assumes that "forward" is +z
        - get_transformation_matrix() assumes that "down" is +y ("up" is -y)

    Args:
        cam_location (np.ndarray): [3,] Camera location
        point (np.ndarray): [3,] The point that the camera is gazing at.
        up (np.ndarray, optional): The vertical-up direction vector. Defaults to np.array([0., -1., 0.])#opencvconvention.

    Returns:
        np.ndarray: [4,4] The camera to world transformation matrix (i.e. the pose matrix of the camera in world coordinate system)
    """
    cam_location, point = np.array(cam_location), np.array(point)
    forward = normalize(point - cam_location)
    return get_transformation_matrix(cam_location, oz=forward, oy=-up, preserve='z')

def gmean(input_x: Union[torch.Tensor, np.ndarray], **kwargs):
    if isinstance(input_x, torch.Tensor):
        return input_x.log().mean(**kwargs).exp()
    elif isinstance(input_x, np.ndarray):
        return np.exp(np.log(input_x).mean(**kwargs))

def rot_to_quat(R: torch.Tensor) -> torch.Tensor:
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

    q[...,0]=torch.sqrt(1.0+R00+R11+R22)/2
    q[..., 1]=(R21-R12)/(4*q[:,0])
    q[..., 2] = (R02 - R20) / (4 * q[:, 0])
    q[..., 3] = (R10 - R01) / (4 * q[:, 0])
    return q

def quat_to_rot(q: torch.Tensor) -> torch.Tensor:
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

def skew_symmetric(vector: torch.Tensor):
    ss_matrix = torch.zeros([*vector.shape[:-1],3,3], device=vector.device, dtype=vector.dtype)
    ss_matrix[..., 0, 1] = -vector[..., 2]
    ss_matrix[..., 0, 2] =  vector[..., 1]
    ss_matrix[..., 1, 0] =  vector[..., 2]
    ss_matrix[..., 1, 2] = -vector[..., 0]
    ss_matrix[..., 2, 0] = -vector[..., 1]
    ss_matrix[..., 2, 1] =  vector[..., 0]
    return ss_matrix

def intr_to_gl_proj(intr: torch.Tensor, near=None, far=None):
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

def box_inside_planes(box_verts: torch.Tensor, planes_nd: torch.Tensor, *, holistic=False, normalized=True) -> torch.BoolTensor:
    """ Check whether N boxes are inside P planes

    Args:
        box_verts (torch.Tensor): [N, 8, 3] The given N boxes vertices
        planes_nd (torch.Tensor): [..., P, 4] The given P planes representation.
        holistic (bool, optional): Whether the holistic box body must be in the planes (T), or any part counts (F). Defaults to False.
        normalized (bool, optional): Whether the plane repr is already normalized. Defaults to True.

    Raises:
        NotImplementedError: Not implemented for now

    Returns:
        torch.BoolTensor: [..., N] Inside checks of the given N boxes w.r.t. all given P planes
    """
    raise NotImplementedError("jianfei: box_inside_planes not implemented correctly. Should change to IoU calculation.")
    
    if not normalized:
        planes_nd = planes_nd / planes_nd[..., :3].norm(dim=-1, keepdim=True)
    normals = planes_nd[..., :3]   # [P, 3]
    distances = planes_nd[..., 3]  # [P,]
    # [N, 8, 1, 3] * [P, 3] -> [N, 8, P, 3] -> [N, 8, P] + [P] -> [N, 8, P] --(all P planes)--> [N, 8]
    check_inside = ((normals * box_verts.unsqueeze(-2)).sum(-1) + distances > 0).all(dim=-1)
    if not holistic:
        # [N, 8] --(any of 8 points)--> [N]
        return check_inside.any(dim=-1)
    else:
        # [N, 8] --(all of 8 points)--> [N]
        return check_inside.all(dim=-1)

def sphere_inside_planes(
    sph_centers_radius: torch.Tensor, 
    planes_nd: torch.Tensor, *, 
    holistic=False, normalized=True) -> torch.BoolTensor:
    """ Check whether N given spheres are inside all P given planes
        NOTE: `...` means arbitary prefix-batch-dims

    Args:
        sph_centers_radius (torch.Tensor): [..., N, 4] The given N spheres representation, center (3) + radius (1)
        planes_nd (torch.Tensor): [..., P, 4] The given P planes representation. The last dim (4) = normals (3) + distance (1)
        holistic (bool, optional): Whether the holistic sphere body must be in the planes (T), or any part counts (F). Defaults to False.
        normalized (bool, optional): Whether the plane repr is already normalized. Defaults to True.

    Returns:
        torch.BoolTensor: [..., N] Inside checks of the given N spheres w.r.t. all given P planes
    """
    if not normalized:
        planes_nd = planes_nd / planes_nd[..., :3].norm(dim=-1, keepdim=True).clamp_min(1e-5)
    normals = planes_nd[..., :3]   # [..., P, 3]
    distances = planes_nd[..., 3]  # [..., P,]
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

def pts_inside_planes(pts: torch.Tensor, planes_nd: torch.Tensor, *, normalized=True) -> torch.BoolTensor:
    """ Check whether N given pts inside all P given planes
        NOTE: `...` means arbitary prefix-batch-dims

    Args:
        pts (torch.Tensor): [..., N, 3] The given N pts
        planes_nd (torch.Tensor): [..., P, 4] The given P planes representation.
            The last dim (4) = normals (3) + distance (1)
        normalized (bool, optional): Whether the plane repr is already normalized. Defaults to True.

    Returns:
        torch.BoolTensor: [..., N]
    """
    if not normalized:
        planes_nd = planes_nd / planes_nd[..., :3].norm(dim=-1, keepdim=True)
    normals = planes_nd[..., :3]   # [P, 3]
    distances = planes_nd[..., 3]  # [P,]
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
        inside = sphere_inside_planes(spheres, planes, holistic=False, normalized=False)
        ic(inside.shape)
        
        if (dims:=inside.dim()) > 1:
            # Dealing with multiple cameras or batched frames: if any camera suffices, then this drawable node suffices.
            inside = inside.any(dim=0) if dims == 2 else (inside.sum(dim=list(range(0,dims-1))) > 0)

        ic(inside.shape)

        planes = torch.randn([*prefix, 8, 4], device=device)
        pts = torch.randn([*prefix, 1000, 3], device=device)
        inside = pts_inside_planes(pts, planes, normalized=False)
        ic(inside.shape)

    unit_test()
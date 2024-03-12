"""
@file   normalize_views.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Helper functions for normalizing multiple views
"""

__all__ = [
    'solve_focus_center', 
    'normalize_multi_view'
]

import numpy as np
from numbers import Number
from typing import List, Literal, Tuple, Union

import scipy
import scipy.optimize
from scipy.spatial.transform import Rotation as R

from nr3d_lib.maths import inverse_transform_matrix_np
from nr3d_lib.graphics.cameras import decompose_intr_c2w_from_proj_np

def solve_focus_center(
    c2ws: np.ndarray, intrs: np.ndarray, 
    H: Union[int, List[int]], W: Union[int, List[int]], 
    verbose=False) -> np.ndarray:
    """ Given a set of non-normalized object-centric multi views, estimate a focus center point.

    Args:
        c2ws (np.ndarray): [N, 4, 4], Multi-view camera-to-world transform mat (camera pose mat)
        intrs (np.ndarray): [N, 3, 3] Multi-view camera intrinsics mat
        H (Union[int, List[int]]): Multi-view image height(s) in pixels
        W (Union[int, List[int]]): Multi-view image width(s) in pixels
        verbose (bool, optional): Whether print verbose info. Defaults to False.

    Raises:
        RuntimeError: The optimization can fail. Not guaranteed to converge.

    Returns:
        np.ndarray: [3,], The estimated 3D focus center points
    """
    
    Hs = np.asarray([H] * len(c2ws) if isinstance(H, Number) else H)
    Ws = np.asarray([W] * len(c2ws) if isinstance(W, Number) else W)
    intrs = np.tile(intrs, (len(c2ws), 1, 1)) if len(intrs.shape)==2 else intrs
    assert len(c2ws) == len(intrs) == len(Hs) == len(Ws), "Input should all have the same length"
    
    w2cs = inverse_transform_matrix_np(c2ws)
    
    pix_centers = np.concatenate([Ws[:,None]/2.0, Hs[:,None]/2.0], axis=-1)
    # A scale factor to avoid influences caused by various image sizes when optimizing
    pix_scale_factor = min(Hs.min(), Ws.min())
    
    # Projection matrices
    Ps = (intrs[..., :3, :3] @ w2cs[..., :3, :4])[..., :3, :4]
    
    # All translation of camera poses
    trans = c2ws[:,:3,3]
    
    # Initial guess: average c2w translation
    X_init = trans.mean(0)
    
    # NOTE: Objective function: 
    # The average 2D pixel distance between \
    #   the 3D center point's projections in each image and \
    #   the 2D pixel center points of each image
    def objective(X):
        uvd = Ps @ np.concatenate([X,[1]], axis=0)
        uv = uvd[:,:2] / uvd[:,2:]
        return ((((uv-pix_centers)/pix_scale_factor)**2).sum(axis=-1)).mean()

    if verbose:
        print(f"=> Start optimization: init function value = {objective(X_init):.6f}")
    ret = scipy.optimize.minimize(objective, X_init, options=dict(disp=verbose))
    if not ret.success:
        raise RuntimeError(f"Solve for focusing center failed. Optimization returns: \n{str(ret)}")
    
    # The estimated focus center of all views
    focus_center = ret.x
    
    return focus_center

def normalize_multi_view(
    c2ws: np.ndarray, intrs: np.ndarray, 
    H: Union[int, List[int]], W: Union[int, List[int]], 
    focus_center: np.ndarray = np.zeros(3,), convert: np.ndarray = np.eye(3,), 
    estimate_focus_center: Literal['solve', 'average']=None, 
    normalize_scale: Literal['average', 'max']=None, 
    normalize_scale_factor: float = 1.0, 
    normalize_rotation=True, 
    verbose=False)-> Tuple[np.ndarray, np.ndarray]:
    """ Normlize a given set of non-normalized object-centric multi views

    Args:
        c2ws (np.ndarray): [N, 4, 4], Multi-view camera-to-world transform mat (camera pose mat)
        intrs (np.ndarray): [N, 3, 3] Multi-view camera intrinsics mat
        H (Union[int, List[int]]): Multi-view image height(s) in pixels
        W (Union[int, List[int]]): Multi-view image width(s) in pixels
        focus_center (np.ndarray, optional): 3D focus center point. Defaults to np.zeros(3,).
            Set to None if you want to automatically estimate a focus center
        convert (np.ndarray, optional): An optional coordinate sys convert mat. Defaults to np.eye(3,).
            This mat converts vectors in new coords sys to vectors in original coords sys.
        estimate_focus_center (Literal['average', 'max'], optional): 
            Optionally determines the method to estimate focus center.
            - 'solve': Solve for focus center using `solve_focus_center`
            - 'average': Use the average tranlation as the focus center
            - None: Do not alter the focus center. Use the original coordinate origin.
            Defaults to None.
        normalize_scale (Literal['average', 'max'], optional): 
            Optionally determines the method to normalize world scale.
            - 'average': Will make the new average focus distance to be 1.0
            - 'max': Will make the new maximum focus distance to be 1.0
            - None: Do not rescale.
            Defaults to None.
        normalize_rotation (bool, optional): Whether to automatically normalize rotation. Defaults to True.
        verbose (bool, optional): Whether print verbose info. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            [4, 4], The normalization mat
            [N, 4, 4], The normalized c2ws
    """

    Hs = np.asarray([H] * len(c2ws) if isinstance(H, Number) else H)
    Ws = np.asarray([W] * len(c2ws) if isinstance(W, Number) else W)
    intrs = np.tile(intrs, (len(c2ws), 1, 1)) if len(intrs.shape)==2 else intrs
    assert len(c2ws) == len(intrs) == len(Hs) == len(Ws), "Input should all have the same length"

    if estimate_focus_center == 'solve':
        focus_center = solve_focus_center(c2ws, intrs, Hs, Ws, verbose=verbose)
    elif estimate_focus_center == 'average':
        focus_center = c2ws[:, :3, 3].mean(axis=0)
    else:
        raise RuntimeError(f"Invalid estimate_focus_center={estimate_focus_center}")
    
    w2cs = inverse_transform_matrix_np(c2ws)
    
    # All translation of camera poses
    trans = c2ws[:,:3,3]
    
    # Estimate the scale of the world
    focus_center_dis = np.linalg.norm((trans-focus_center[None,:]), axis=-1)
    if normalize_scale is None:
        scale = 1.
    elif normalize_scale == 'average':
        scale = focus_center_dis.mean() * normalize_scale_factor
    elif normalize_scale == 'max':
        scale = focus_center_dis.max() * normalize_scale_factor
    else:
        raise RuntimeError(f"Invalid normalize_scale={normalize_scale}")    
    
    # Calculate average rotation of camera poses
    rots = c2ws[:,:3,:3]
    Rs = R.from_matrix(rots)
    mean_rot = Rs.mean().as_matrix()

    # An example of converting coordinate system in advance
    # # openCV to [x forward, y left-ward, z upward]
    # convert = np.zeros([3,3])
    # convert[0,2] = 1; convert[1,0] = -1; convert[2, 1] = -1
    
    mean_rot_ex = np.eye(4)
    if normalize_rotation:
        mean_rot_ex[:3,:3] = mean_rot @ convert.T
    else:
        mean_rot_ex[:3,:3] = convert.T
    
    # Normalization mat
    normalization = np.diagflat([scale,scale,scale,1])
    normalization[:3,3] = focus_center
    normalization = normalization @ mean_rot_ex # NOTE: To normalize rotations as well
    
    # Example usage of the normalization mat
    new_Ps = intrs @ (w2cs @ normalization[None,...])
    _, new_c2ws = [*zip(*[decompose_intr_c2w_from_proj_np(P) for P in new_Ps])] # NOTE: intrinsics are not changed.
    new_c2ws = np.array(new_c2ws)
    
    return normalization, new_c2ws
"""
@file   shapenet_to_srn.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Helper functions for coordinate conversion among different versions of Shapenet dataset 
    and SRN (Scene representation network, V.Sitzmann et al) dataset
"""

import copy
import numpy as np

def from_shapenet_to_srn(points, normalize_dict: dict, conversion='v2_to_srn'):
    """ Align points from shapenet models to SRN renderings
    Args:
        points: [N, 3]
        normalize_dict: loaded dict of the model_normalized.json file
        conversion: choose among:
            v2_to_srn: From shapenet v2 to SRN
            v1_to_srn: From shapenet v1 to SRN
            v2_to_sitzmann_rendering: From shapenet v2 to https://github.com/vsitzmann/shapenet_renderer
            v1_to_sitzmann_rendering: From shapenet v1 to https://github.com/vsitzmann/shapenet_renderer
    Return: [N,3]
    """

    bmax = np.array(normalize_dict['max'])
    bmin = np.array(normalize_dict['min'])
    centroid = np.array(normalize_dict['centroid'])
    norm = np.linalg.norm(bmax-bmin)
    center = (bmax + bmin) / 2.

    points = copy.deepcopy(points)
    #---------------
    # offset the model, so that the center of the bounding box is at the origin.
    #   for shapenet v1, it's already satisfied.
    #---------------
    if conversion == 'v2_to_srn':
        points[:, :3] = (points[:, :3] + (centroid - center) / norm)

    if conversion == 'v1_to_srn' or conversion == 'v2_to_srn':
        #---------------
        # rescale the model, so that the max value of the bouding box size equals 1.
        #---------------
        points[:, :3] = points[:, :3] / ((bmax-bmin).max()/norm)

    #---------------
    # different choices of rotations
    #----------------
    if conversion == 'v2_to_srn':
        #----------------
        # From shapenet v2 to SRN
        R2srn = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    
    elif conversion == 'v1_to_srn':
        #----------------
        # From shapenet v1 to SRN
        R2srn = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])

    elif conversion == 'v1_to_sitzmann_rendering':
        #-----------------
        # From shapenet v1 to sitzmann's rendrings scipts
        # R2srn = np.array([[0, 0, -1], [0, 1, 0], [-1, 0, 0]])
        # R2srn = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
        R2srn = np.eye(3)
    elif conversion == 'v2_to_sitzmann_rendering':
        #------------------
        # From shapenet v2 to sitzmann's rendrings scipts 
        # # no rotation is needed
        # R2srn = np.eye(3)
        R2srn = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    else:
        raise RuntimeError("please specify conversion type")

    points[:, :3] = np.matmul(R2srn[None, ...], points[:, :3, None])[...,0]

    return points

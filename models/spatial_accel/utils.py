"""
@file   utils.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Utility functions for acceleration structs.
"""

__all__ = [
    'expand_idx', 
    'expand_points'
]

from itertools import product

import torch

from nr3d_lib.utils import check_to_torch

def expand_idx(idx: torch.LongTensor, dilation: int=1):
    cube_3x3x3 = list(product(*zip([-1, -1, -1], [0, 0, 0], [1, 1, 1]))) # 1 us
    cube_3x3x3 = check_to_torch(cube_3x3x3, ref=idx)
    idx = idx.unsqueeze(-2) + cube_3x3x3 * dilation
    return idx

def expand_points(points: torch.Tensor, dilation: float):
    """
    Modified from neucon-w
    A naive version of the sparse dilation.
    
    Args:
        points: [..., 3]
    
    Returns:
        [..., 27, 3]
    """
    # [27, 3] A cube with size=3 and step=1.
    cube_3x3x3 = list(product(*zip([-1, -1, -1], [0, 0, 0], [1, 1, 1]))) # 1 us
    cube_3x3x3 = check_to_torch(cube_3x3x3, ref=points)
    points = points.unsqueeze(-2) + cube_3x3x3 * dilation
    return points

if __name__ == "__main__":
    def unit_test(device=torch.device('cuda')):
        pts = torch.randn([4096, 3], device=device, dtype=torch.float)
        expand_points(pts, 0.25)
    unit_test()
"""
Modified from https://github.com/kujason/ip_basic.git
Uses kornia tensor operations as pytorch alternatives for cv2 operations.

@inproceedings{ku2018defense,
  title={In Defense of Classical Image Processing: Fast Depth Completion on the CPU},
  author={Ku, Jason and Harakeh, Ali and Waslander, Steven L},
  booktitle={2018 15th Conference on Computer and Robot Vision (CRV)},
  pages={16--22},
  year={2018},
  organization={IEEE}
}
"""

__all__ = [
    'depth_fill_in_fast_pytorch'
]

from typing import Literal
import kornia
import torch

# Full kernels
FULL_KERNEL_3 = torch.ones((3, 3))
FULL_KERNEL_5 = torch.ones((5, 5))
FULL_KERNEL_7 = torch.ones((7, 7))
FULL_KERNEL_9 = torch.ones((9, 9))
FULL_KERNEL_31 = torch.ones((31, 31))

# 3x3 cross kernel
CROSS_KERNEL_3 = torch.tensor(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ])

# 5x5 cross kernel
CROSS_KERNEL_5 = torch.tensor(
    [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ])

# 5x5 diamond kernel
DIAMOND_KERNEL_5 = torch.tensor(
    [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ])

# 7x7 cross kernel
CROSS_KERNEL_7 = torch.tensor(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ])

# 7x7 diamond kernel
DIAMOND_KERNEL_7 = torch.tensor(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ])

def depth_fill_in_fast_pytorch(
    depth_map: torch.Tensor, 
    max_depth: float = None, eps: float = 0.1, 
    custom_kernel=DIAMOND_KERNEL_5, 
    # extrapolate=False, 
    blur_type='bilateral', 
    engine: Literal['convolution', 'unfold'] = 'convolution'
    ):
    """Fast depth completion.

    Args:
        depth_map: [B,H,W] or [H,W] projected depths
        max_depth: max depth value for inversion
        eps: any depth smaller than this value will be assumed non-exist.
        custom_kernel: kernel to apply initial dilation
        extrapolate: whether to extrapolate by extending depths to top of
            the frame, and applying a 31x31 full kernel dilation
        blur_type:
            'bilateral' - preserves local structure (recommended)
            'gaussian' - provides lower RMSE
        engine: in [convolution,unfold]; the type of kornia morphology backend.

    Returns:
        depth_map: dense depth map
    """
    if max_depth is None:
        max_depth = depth_map.max()
    
    shape0 = list(depth_map.shape)
    *_, H, W = depth_map.shape
    depth_map = depth_map.view(-1,1,H,W)
    
    #---- Invert
    depth_map = torch.where(depth_map > eps, max_depth - depth_map, depth_map)
    
    #---- Dilate
    depth_map = kornia.morphology.dilation(depth_map, custom_kernel.to(depth_map), engine=engine)
    
    #---- Hole closing
    depth_map = kornia.morphology.closing(depth_map, FULL_KERNEL_5.to(depth_map), engine=engine)
    
    #---- Fill empty spaces with dilated values
    dilated = kornia.morphology.dilation(depth_map, FULL_KERNEL_7.to(depth_map), engine=engine)
    depth_map = torch.where(depth_map < eps, dilated, depth_map)
    
    #---- Median blur
    depth_map = kornia.filters.median_blur(depth_map, 5)
    
    #---- Bilateral or Gaussian blur
    if blur_type == 'bilateral':
        # Bilateral blur
        depth_map = kornia.filters.bilateral_blur(depth_map, 5, 1.5, (2.0, 2.0))
    elif blur_type == 'gaussian':
        # Gaussian blur
        blurred = kornia.filters.gaussian_blur2d(depth_map, (5,5), (0,0))
        depth_map = torch.where(depth_map > eps, blurred, depth_map)
        # valid_pixels_inds = (depth_map > eps).nonzero(as_tuple=True)
        # depth_map[valid_pixels_inds] = blurred[valid_pixels_inds]
    
    #---- Invert
    depth_map = torch.where(depth_map > eps, max_depth - depth_map, depth_map)
    
    depth_map = depth_map.view(*shape0)
    return depth_map
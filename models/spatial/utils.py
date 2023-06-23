"""
@file   utils.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Utility functions for acceleration structs.
"""

__all__ = [
    'create_dense_grid', 
    'create_octree_dense', 
    'create_octree_root_only', 
    'octree_to_spc_ins'
]
import torch

try:
    from kaolin.rep.spc import Spc
    from kaolin.ops.spc import unbatched_points_to_octree
except:
    from nr3d_lib.fmt import log
    log.warning("kaolin is not installed. OctreeAS / ForestAS disabled.")
 
def create_dense_grid(level: int, device=torch.device('cuda')):
    coords = torch.stack(
        torch.meshgrid(
            [torch.arange(2**level, device=device, dtype=torch.short) for _ in range(3)], 
            indexing='ij'), 
        dim=-1).reshape(-1, 3)
    return coords
 
def create_octree_dense(level: int, device=torch.device('cuda')):
    assert level > 0, "level must be > 0 during creation of octree."
    coords = create_dense_grid(level, device=device)
    octree = unbatched_points_to_octree(coords, level)
    return octree

def create_octree_root_only(device=torch.device('cuda')):
    return torch.tensor([255], device=device, dtype=torch.uint8)

def octree_to_spc_ins(octree):
    lengths = torch.tensor([len(octree)], dtype=torch.int32)
    spc = Spc(octree, lengths)
    spc._apply_scan_octrees()
    spc._apply_generate_points()
    return spc
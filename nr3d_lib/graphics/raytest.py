"""
@file   raytest.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Ray intersection untility functions.
"""

__all__ = [
    'ray_sphere_intersection_rough', 
    'ray_sphere_intersection', 
    'get_dvals_from_radius', 
    'ray_box_intersection', 
    'ray_box_intersection_fast_float', 
    'ray_box_intersection_fast_tensor', 
    'ray_box_intersection_fast_float_nocheck', 
    'ray_box_intersection_fast_tensor_nocheck', 
    'octree_raytrace_fixed_from_kaolin', 
]

from typing import Tuple, Union

import torch

def ray_sphere_intersection_rough(
    rays_o: torch.Tensor, rays_d: torch.Tensor, *,
    r=1.0, keepdim=True, t_min_cons: float=0.0, t_max_cons: float=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    NOTE: Modified from https://github.com/Totoro97/NeuS
    rays_o: camera center's coordinate
    rays_d: camera rays' directions. already normalized.
    """
    dir_scale = rays_d.norm(dim=-1, keepdim=keepdim).clamp_min_(1e-10)
    # NOTE: (minus) the length of the line projected from [the line from camera to sphere center] to [the line of camera rays]
    #       scale: In the scaled space.
    mid = -torch.sum(rays_o * rays_d / dir_scale.view(*rays_d.shape[:-1], 1), dim=-1, keepdim=keepdim)
    # NOTE: a convservative approximation of the half chord length from ray intersections with the sphere.
    #       all half chord length < r
    #       scale: In the original un-scaled space.
    near, far = ((mid - r)/dir_scale).clamp_min_(t_min_cons), ((mid + r)/dir_scale).clamp_(r/dir_scale, t_max_cons)
    return near, far

def ray_sphere_intersection(
    rays_o: torch.Tensor, rays_d: torch.Tensor, *, 
    r = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    NOTE: Modified from IDR. https://github.com/lioryariv/idr
    rays_o: camera center's coordinate
    rays_d: camera rays' directions. already normalized.
    """
    assert (rays_d.norm(dim=-1)-1).abs().max() < 1e-4, "Jianfei: this function has some BUG now. Will be fixed before 20220930."
    rayso_norm_square = torch.sum(rays_o**2, dim=-1, keepdim=True)
    # (minus) the length of the line projected from [the line from camera to sphere center] to [the line of camera rays]
    ray_cam_dot = torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
    
    # accurate ray-sphere intersections
    near = torch.zeros([*rays_o.shape[:-1], 1]).to(rays_o.device)
    far = torch.zeros([*rays_o.shape[:-1], 1]).to(rays_o.device)
    under_sqrt = ray_cam_dot ** 2  + r ** 2 - rayso_norm_square
    mask_intersect = under_sqrt > 0
    sqrt = torch.sqrt(under_sqrt[mask_intersect])
    near[mask_intersect] = - sqrt - ray_cam_dot[mask_intersect]
    far[mask_intersect] = sqrt - ray_cam_dot[mask_intersect]
    return near.clamp_min_(0.0), far.clamp_min_(0.0), mask_intersect

def get_dvals_from_radius(
    rays_o: torch.Tensor, rays_d: torch.Tensor, *, 
    rs: torch.Tensor, far_end=True) -> torch.Tensor:
    """
    rays_o: camera center's coordinate
    rays_d: camera rays' directions. already normalized.
    rs: the distance to the origin
    far_end: whether the point is on the far-end of the ray or on the near-end of the ray
    """
    rayso_norm_square = torch.sum(rays_o**2, dim=-1, keepdim=True)
    # NOTE: (minus) the length of the line projected from [the line from camera to sphere center] to [the line of camera rays]
    ray_cam_dot = torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        
    under_sqrt = rs**2 - (rayso_norm_square - ray_cam_dot ** 2)
    assert (under_sqrt > 0).all()
    sqrt = torch.sqrt(under_sqrt)
    
    if far_end:
        d_vals = -ray_cam_dot + sqrt
    else:
        d_vals = (-ray_cam_dot - sqrt).clamp_min_(0.)
    
    return d_vals

def ray_box_intersection(
    rays_o: torch.Tensor, rays_d: torch.Tensor, *,
    aabb_min: Union[torch.Tensor,float]=-0.5, aabb_max: Union[torch.Tensor,float]=0.5, 
    t_min_cons: float=None, t_max_cons: float=None) -> Tuple[torch.Tensor, torch.Tensor, torch.BoolTensor]:
    """ Caculate intersections between rays and boxes
        By default, assume box frames are scaled to vertices between [-0.5, -0.5, -0.5] and [0.5, 0.5, 0.5]

    Args:
        rays_o (torch.Tensor): [..., 3],    Origin of the ray in each box frame
        rays_d (torch.Tensor): [..., 3],    Direction vector of each ray in each box frame
        aabb_min (Union[torch.Tensor,float], optional): Vertex of a 3D bounding box. Defaults to -0.5.
        aabb_max (Union[torch.Tensor,float], optional): Vertex of a 3D bounding box. Defaults to 0.5.
        t_min_cons (float, optional): [...]. Optional minimum depth constraint. Defaults to None.
        t_max_cons (float, optional): [...]. Optional maximum depth constraint. Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.BoolTensor]:
            t_near:         [...], Entry depth of intersection.
            t_far:          [...], Exit depth of intersection.
            mask_intersect: [...], T/F intersection marks
    """
    # Source: https://medium.com/@bromanz/another-view-on-the-classic-ray-aabb-intersection-algorithm-for-bvh-traversal-41125138b525
    # https://gamedev.stackexchange.com/questions/18436/most-efficient-aabb-vs-ray-collision-algorithms

    t_min = (aabb_min - rays_o) / rays_d
    t_max = (aabb_max - rays_o) / rays_d
    t_near = torch.minimum(t_min, t_max).max(dim=-1).values
    t_far = torch.maximum(t_min, t_max).min(dim=-1).values
    del t_min, t_max
    
    # NOTE: This won't break intersection checks
    #       when t_far > t_near:
    #           when t_min_cons > t_far, no intersection.
    #           when t_max_cons < t_near, no intersection.
    #           in other cases, there will still be intersection with near,far clipped.
    #       when t_far < t_near:
    #           
    if t_min_cons is not None: t_near.clamp_min_(t_min_cons) # = max(t_near, t_min_cons)
    if t_max_cons is not None: t_far.clamp_max_(t_max_cons) # = min(t_far, t_max_cons)

    # Check if rays are inside boxes
    check1 = t_far > t_near
    # Check that boxes are in front of the ray origin
    check2 = t_far > (0 if t_min_cons is None else t_min_cons)
    mask_intersect = check1 & check2 & (True if t_max_cons is None else t_near < t_max_cons)
    return t_near, t_far, mask_intersect

# @torch.jit.script
def ray_box_intersection_fast_float(rays_o: torch.Tensor, rays_d: torch.Tensor, aabb_min: float, aabb_max: float):
    t_min = (aabb_min - rays_o) / rays_d
    t_max = (aabb_max - rays_o) / rays_d
    t_near = torch.minimum(t_min, t_max).max(dim=-1).values
    t_far = torch.maximum(t_min, t_max).min(dim=-1).values
    # Check if rays are inside boxes
    check1 = t_far > t_near
    # Check that boxes are in front of the ray origin
    check2 = t_far > 0
    mask_intersect = check1 & check2
    return t_near, t_far, mask_intersect

# @torch.jit.script
def ray_box_intersection_fast_tensor(rays_o: torch.Tensor, rays_d: torch.Tensor, aabb_min: torch.Tensor, aabb_max: torch.Tensor):
    t_min = (aabb_min - rays_o) / rays_d
    t_max = (aabb_max - rays_o) / rays_d
    t_near = torch.minimum(t_min, t_max).max(dim=-1).values
    t_far = torch.maximum(t_min, t_max).min(dim=-1).values
    # Check if rays are inside boxes
    check1 = t_far > t_near
    # Check that boxes are in front of the ray origin
    check2 = t_far > 0
    mask_intersect = check1 & check2
    return t_near, t_far, mask_intersect

# @torch.jit.script
def ray_box_intersection_fast_float_nocheck(rays_o: torch.Tensor, rays_d: torch.Tensor, aabb_min: float, aabb_max: float):
    t_min = (aabb_min - rays_o) / rays_d
    t_max = (aabb_max - rays_o) / rays_d
    t_near = torch.minimum(t_min, t_max).max(dim=-1).values
    t_far = torch.maximum(t_min, t_max).min(dim=-1).values
    return t_near, t_far

# @torch.jit.script
def ray_box_intersection_fast_tensor_nocheck(rays_o: torch.Tensor, rays_d: torch.Tensor, aabb_min: torch.Tensor, aabb_max: torch.Tensor):
    t_min = (aabb_min - rays_o) / rays_d
    t_max = (aabb_max - rays_o) / rays_d
    t_near = torch.minimum(t_min, t_max).max(dim=-1).values
    t_far = torch.maximum(t_min, t_max).min(dim=-1).values
    return t_near, t_far

"""
NOTE: Problem with kaolin's SPC v0.12.0 
        > no/wrong intersection when rays_o inside AABBs of any level.
        > See more in https://github.com/NVIDIAGameWorks/kaolin/issues/490
        > not yet fixed in the latest PR https://github.com/NVIDIAGameWorks/kaolin/pull/634
"""
# from kaolin.render.spc import unbatched_raytrace
# def octree_raytrace_allow_inside(octree, point_hierarchies, pyramids, exsum, rays_o, rays_d, level, tmin: float=0., tmax: float=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     _near, _far = ray_box_intersection_fast_float_nocheck(rays_o, rays_d, -1., 1.)
#     if (_near < 0).any():
#         # _near.unsqueeze_(-1).add_(-1)
#         __near = _near.min().item()-1
#         rays_o_modified = rays_o + __near * rays_d # Move camera backwards; a proxy rays_o that is outside of [-1,1] range.
#         ridx, pidx, depth = unbatched_raytrace(octree, point_hierarchies, pyramids, exsum, rays_o_modified, rays_d, level, with_exit=True)
#         depth += __near # From proxy depths to real intersection depth.
#     else:
#         ridx, pidx, depth = unbatched_raytrace(octree, point_hierarchies, pyramids, exsum, rays_o, rays_d, level, with_exit=True)
#     tmin = 0. if tmin is None else tmin
#     nidx = (depth[:,1] > tmin).nonzero().long().squeeze_(-1) if tmax is None else ((depth[:,1] > tmin) & (depth[:,0] < tmax)).nonzero().long().squeeze_(-1)
#     return ridx[nidx].long().contiguous(), pidx[nidx].long().contiguous(), depth[nidx].clamp_min_(0).contiguous()

from nr3d_lib.bindings._forest import raytrace_cuda_fixed
def octree_raytrace_fixed_from_kaolin(
    octree, point_hierarchy, pyramid, exsum, origin, direction, level,
    return_depth=True, with_exit=True, include_head=True):
    # Borrowed and fixed from https://github.com/NVIDIAGameWorks/kaolin
    output = raytrace_cuda_fixed(
        octree.contiguous(), point_hierarchy.contiguous(), pyramid.contiguous(),
        exsum.contiguous(), origin.contiguous(), direction.contiguous(),
        level, return_depth, with_exit, include_head)
    nuggets = output[0]
    ray_index = nuggets[..., 0]
    point_index = nuggets[..., 1]
    if return_depth:
        return ray_index, point_index, output[1]
    else:
        return ray_index, point_index

if __name__ == "__main__":
    def unit_test(device=torch.device('cuda')):
        from kaolin.ops.spc import unbatched_points_to_octree
        from nr3d_lib.models.spatial import octree_to_spc_ins
        import torch.nn.functional as F
        level = 5
        occ_grid = torch.rand([2**level]*3, device=device) > 0.5
        coords = occ_grid.nonzero().short()
        octree = unbatched_points_to_octree(coords, level=level)
        spc = octree_to_spc_ins(octree)
        rays_o = torch.zeros([4096, 3], device=device, dtype=torch.float)
        rays_d = F.normalize(torch.randn([4096, 3], device=device, dtype=torch.float).abs(), dim=-1)
        octree_raytrace_fixed_from_kaolin(octree, spc.point_hierarchies, spc.pyramids[0], spc.exsum, rays_o, rays_d, level=level, tmin=0.1, tmax=2.0)

    def test_inside_simple():
        
        from kaolin.rep.spc import Spc
        # from kaolin.render.spc import unbatched_raytrace
        unbatched_raytrace = octree_raytrace_fixed_from_kaolin
        from kaolin.ops.spc import unbatched_points_to_octree
        
        
        device = torch.device('cuda')
        #---------------- level = 1
        level = 1
        octree = torch.tensor([255], dtype=torch.uint8, device=device)
        spc = Spc(octree, torch.tensor([len(octree)], dtype=torch.int32))
        spc._apply_scan_octrees()
        spc._apply_generate_points()
        
        
        rays_o = torch.tensor([[0.,  0.,  0.]], dtype=torch.float, device=device)
        rays_d = torch.tensor([[0,   1,   0]], dtype=torch.float, device=device)
        ridx, pidx, depth = unbatched_raytrace(octree, spc.point_hierarchies, spc.pyramids[0], spc.exsum, rays_o, rays_d, level=level, with_exit=True, return_depth=True)

        rays_o = torch.tensor([[0.,  -1.0e-3, 0.]], dtype=torch.float, device=device)
        rays_d = torch.tensor([[0,   1,       0]], dtype=torch.float, device=device)
        ridx, pidx, depth = unbatched_raytrace(octree, spc.point_hierarchies, spc.pyramids[0], spc.exsum, rays_o, rays_d, level=level, with_exit=True, return_depth=True)

        rays_o = torch.tensor([[0.,  1.0e-3, 0.]], dtype=torch.float, device=device)
        rays_d = torch.tensor([[0,   1,      0]], dtype=torch.float, device=device)
        ridx, pidx, depth = unbatched_raytrace(octree, spc.point_hierarchies, spc.pyramids[0], spc.exsum, rays_o, rays_d, level=level, with_exit=True, return_depth=True)



        rays_o = torch.tensor([[1.0e-3, 1.0e-3, 1.0e-3]], dtype=torch.float, device=device)
        rays_d = torch.tensor([[0,      1,      0]], dtype=torch.float, device=device)
        ridx, pidx, depth = unbatched_raytrace(octree, spc.point_hierarchies, spc.pyramids[0], spc.exsum, rays_o, rays_d, level=level, with_exit=True, return_depth=True)

        rays_o = torch.tensor([[0., -1.+1e-3, 0.]], dtype=torch.float, device=device)
        rays_d = torch.tensor([[0.,  1., 0.]], dtype=torch.float, device=device)
        ridx, pidx, depth = unbatched_raytrace(octree, spc.point_hierarchies, spc.pyramids[0], spc.exsum, rays_o, rays_d, level=level, with_exit=True, return_depth=True)

        rays_o = torch.tensor([[0., -1., 0.]], dtype=torch.float, device=device)
        rays_d = torch.tensor([[0.,  1., 0.]], dtype=torch.float, device=device)
        ridx, pidx, depth = unbatched_raytrace(octree, spc.point_hierarchies, spc.pyramids[0], spc.exsum, rays_o, rays_d, level=level, with_exit=True, return_depth=True)

        rays_o = torch.tensor([[0.1, -0.1, 0.1]], dtype=torch.float, device=device)
        rays_d = torch.tensor([[0.,  1., 0.]], dtype=torch.float, device=device)
        ridx, pidx, depth = unbatched_raytrace(octree, spc.point_hierarchies, spc.pyramids[0], spc.exsum, rays_o, rays_d, level=level, with_exit=True, return_depth=True)

        #---------------- level = 2
        level = 2
        occ_grid = torch.ones([2**level]*3, device=device, dtype=torch.bool)
        coords = occ_grid.nonzero().short()
        octree = unbatched_points_to_octree(coords, level=level)
        spc = Spc(octree, torch.tensor([len(octree)], dtype=torch.int32))
        spc._apply_scan_octrees()
        spc._apply_generate_points()

        rays_o = torch.tensor([[0., -1., 0.]], dtype=torch.float, device=device)
        rays_d = torch.tensor([[0.,  1., 0.]], dtype=torch.float, device=device)
        ridx, pidx, depth = unbatched_raytrace(octree, spc.point_hierarchies, spc.pyramids[0], spc.exsum, rays_o, rays_d, level=level, with_exit=True, return_depth=True)

        rays_o = torch.tensor([[0., -1.+1e-3, 0.]], dtype=torch.float, device=device)
        rays_d = torch.tensor([[0.,  1., 0.]], dtype=torch.float, device=device)
        ridx, pidx, depth = unbatched_raytrace(octree, spc.point_hierarchies, spc.pyramids[0], spc.exsum, rays_o, rays_d, level=level, with_exit=True, return_depth=True)

        rays_o = torch.tensor([[0., -0.5, 0.]], dtype=torch.float, device=device)
        rays_d = torch.tensor([[0.,  1.,  0.]], dtype=torch.float, device=device)
        ridx, pidx, depth = unbatched_raytrace(octree, spc.point_hierarchies, spc.pyramids[0], spc.exsum, rays_o, rays_d, level=level, with_exit=True, return_depth=True)

        rays_o = torch.tensor([[0.,  0.,  0.]], dtype=torch.float, device=device)
        rays_d = torch.tensor([[0,   1,   0]], dtype=torch.float, device=device)
        ridx, pidx, depth = unbatched_raytrace(octree, spc.point_hierarchies, spc.pyramids[0], spc.exsum, rays_o, rays_d, level=level, with_exit=True, return_depth=True)

        rays_o = torch.tensor([[0.,  -1.0e-3, 0.]], dtype=torch.float, device=device)
        rays_d = torch.tensor([[0,   1,       0]], dtype=torch.float, device=device)
        ridx, pidx, depth = unbatched_raytrace(octree, spc.point_hierarchies, spc.pyramids[0], spc.exsum, rays_o, rays_d, level=level, with_exit=True, return_depth=True)

        rays_o = torch.tensor([[0.,  1.0e-3,  0.]], dtype=torch.float, device=device)
        rays_d = torch.tensor([[0,   1,   0]], dtype=torch.float, device=device)
        ridx, pidx, depth = unbatched_raytrace(octree, spc.point_hierarchies, spc.pyramids[0], spc.exsum, rays_o, rays_d, level=level, with_exit=True, return_depth=True)

        rays_o = torch.tensor([[1.0e-3, 1.0e-3, 1.0e-3]], dtype=torch.float, device=device)
        rays_d = torch.tensor([[0,      1,      0]], dtype=torch.float, device=device)
        ridx, pidx, depth = unbatched_raytrace(octree, spc.point_hierarchies, spc.pyramids[0], spc.exsum, rays_o, rays_d, level=level, with_exit=True, return_depth=True)

    # unit_test()
    test_inside_simple()
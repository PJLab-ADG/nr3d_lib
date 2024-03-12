import torch
import argparse
from icecream import ic
from nr3d_lib.config import ConfigDict
from nr3d_lib.graphics.pack_ops import interleave_arange
from nr3d_lib.models.spatial import ForestBlockSpace
from nr3d_lib.models.accelerations import OccGridAccelForest
from nr3d_lib.graphics.raytest import octree_raytrace_fixed_from_kaolin
from kaolin.render.spc import unbatched_raytrace

parser = argparse.ArgumentParser()
parser.add_argument("--ptvsd", action="store_true", help="Whether to start ptvsd debugging.")
args = parser.parse_args()
if args.ptvsd:
    import ptvsd
    ptvsd.enable_attach(address =('127.0.0.1', 10010), redirect_output=True)
    ptvsd.wait_for_attach()

forest_level = 3
occ_grid_resolution = [16, 16, 16]

forest = ForestBlockSpace(device=torch.device('cuda'))
# occ_grid = torch.rand([2**forest_level]*3, device=torch.device('cuda'), dtype=torch.float) > 0.5
# torch.save({'forest_space_occ': occ_grid}, './dev_test/dbg_forest.pt')
occ_grid = torch.load('/home/dengnianchen/Work/neuralsim/nr3d_lib/tests/dbg_forest.pt')['forest_space_occ']
corners = occ_grid.nonzero().long()
forest.populate_from_corners(corners=corners, level=forest_level,
                             world_origin=[-1, -2, -1], world_block_size=[0.3, 0.3, 0.3])

rays_o_0 = torch.tensor([[-2, -2, -2]], device=torch.device('cuda'), dtype=torch.float)
rays_d_0 = torch.tensor([[1, 1, 1]], device=torch.device('cuda'), dtype=torch.float)
ray_tested = forest.ray_test(rays_o_0, rays_d_0)

rays_o, rays_d, near, far, num_rays, rays_inds, seg_pack_infos, seg_block_inds, seg_entries, seg_exits = \
    ray_tested['rays_o'], ray_tested['rays_d'], ray_tested['near'], ray_tested['far'], ray_tested['num_rays'], ray_tested['rays_inds'], \
    ray_tested['seg_pack_infos'], ray_tested['seg_block_inds'], ray_tested['seg_entries'], ray_tested['seg_exits']

print("forest ray_test in:", seg_entries.tolist())
print("forest ray_test out:", seg_exits.tolist())
print("forest ray_test in position:", (rays_o + rays_d * seg_entries[:, None]).tolist())

world_origin, world_block_size = forest.world_origin.to(
    rays_o_0.dtype), forest.world_block_size.to(rays_d_0.dtype)
world_scale = world_block_size * (2**forest.level) / 2.
rays_o_forest, rays_d_forest = (rays_o_0 - world_origin) / world_scale - 1., rays_d_0 / world_scale
ridx, pidx, depth = octree_raytrace_fixed_from_kaolin(forest.spc.octrees, forest.spc.point_hierarchies,
                                                 forest.spc.pyramids[0], forest.spc.exsum, rays_o_forest, rays_d_forest, level=forest.level, tmin=None, tmax=None)

# _near, _far = ray_box_intersection_fast_float_nocheck(rays_o_forest, rays_d_forest, -1., 1.)
ridx0, pidx0, depth0 = unbatched_raytrace(forest.spc.octrees, forest.spc.point_hierarchies,
                                          forest.spc.pyramids[0], forest.spc.exsum, rays_o_forest,
                                          rays_d_forest, level=forest.level, with_exit=True)
ridx1, pidx1, depth1 = unbatched_raytrace(forest.spc.octrees, forest.spc.point_hierarchies,
                                          forest.spc.pyramids[0], forest.spc.exsum, rays_o_forest,
                                          rays_d_forest, level=forest.level, with_exit=False)
ridx2, pidx2, depth2 = unbatched_raytrace(forest.spc.octrees, forest.spc.point_hierarchies,
                                          forest.spc.pyramids[0], forest.spc.exsum, rays_o_forest,
                                          rays_d_forest, level=forest.level, with_exit=True,
                                          include_head=True)
print("kaolin raytrace with exit: ", depth0.tolist())
print("kaolin raytrace w/o exit: ", depth1.tolist())
print("kaolin raytrace with exit and include head: ", depth2.tolist())

num_blocks = len(forest.block_ks)
batched_occ_val_grid = torch.rand([num_blocks, *occ_grid_resolution], device=torch.device('cuda'), dtype=torch.float)
# batched_occ_grid = batched_occ_val_grid > 0.9
# batched_occ_grid = torch.ones([num_blocks,*occ_grid_resolution], device=torch.device('cuda'), dtype=torch.bool)
# ret = occgrid_raymarch_forest(forest.meta, batched_occ_grid, rays_o, rays_d, near, far, seg_block_inds, seg_entries, seg_exits, seg_pack_infos, step_size=0.01)

accel = OccGridAccelForest(
    space=forest, 
    resolution=occ_grid_resolution,
    occ_val_fn_cfg=ConfigDict(type='sdf', inv_s=256.0),
    occ_thre=0.3, ema_decay=0.95, init_cfg={'mode':'from_net'}, update_from_net_cfg={},
    update_from_samples_cfg={}, n_steps_between_update=16, n_steps_warmup=256
)
accel.populate()
accel.occ.occ_val_grid = batched_occ_val_grid
accel.occ.occ_grid = accel.occ.occ_val_grid > accel.occ.occ_thre

ret = accel.ray_march(rays_o, rays_d, near, far, seg_block_inds,
                      seg_entries, seg_exits, seg_pack_infos, step_size=0.01)

# test_depth_samples = torch.linspace(0, far.item(), 1000, device=torch.device('cuda'))
test_depth_samples = interleave_arange(near, far, 0.01, return_idx=False)
test_samples = torch.addcmul(rays_o[0].unsqueeze(-2),
                             rays_d[0].unsqueeze(-2), test_depth_samples.unsqueeze(-1))
test_samples_occupied = accel.query_world(test_samples)

# npts, bidx = accel.space.normalize_coords(test_samples)

ic(test_depth_samples[test_samples_occupied])
ic(ret.depth_samples)

exit()
from vedo import Box, Plotter, Volume, show, Points, Arrow
forest_actors = accel.debug_vis(draw=False, boundary=False, draw_occ_grid=False)
ray_arrow = Arrow(rays_o[0].data.cpu().numpy(), (rays_o + rays_d)[0].data.cpu().numpy(), s=0.01)
plt = Plotter(axes=1)
sample_pts = Points(ret.samples.data.cpu().numpy(), r=12.0)
plt.show(*forest_actors, ray_arrow, sample_pts)
plt.interactive().close()

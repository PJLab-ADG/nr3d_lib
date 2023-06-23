import faulthandler; faulthandler.enable()

import math
import torch
import numpy as np
from icecream import ic
from torch.utils.benchmark import Timer
import nr3d_lib_bindings._lotd as _backend
from nr3d_lib_bindings._forest import ForestMeta

from kaolin.ops.spc import unbatched_points_to_octree
from kaolin.rep.spc import Spc

torch.cuda.manual_seed_all(42)

device = torch.device("cuda")
input_dtype = torch.float
param_dtype = torch.float16

# lod_meta = _backend.LoDMeta(3, [4,8,16,32,64], [4,2,4,16,4], ["Dense","Dense","VM","NPlaneMul","CPfast"])

max_level = None
# max_level = None

lod_meta = _backend.LoDMeta(
    3, 
    [34,      55,     90, 140, 230, 370, 600, 1000, 1600], 
    [2,       2,      2,  2,   2,   2,   2,   2,    2], 
    ['Dense','Dense','VM','VM','VM','VM','VM','VM','VM']
)

# lod_meta = _backend.LoDMeta(
#     3, 
#     [34,      55,     90, 140, 230, 370, 600, 1000, 1600, 2600, 4200], 
#     [2,       2,      2,  2,   2,   2,   2,   2,    2,    2,    2], 
#     ['Dense','Dense','VM','VM','VM','VM','VM','VM','VM', 'CPfast', 'CPfast']
# )

# lod_meta = _backend.LoDMeta(
#     3, 
#     [34,      55,     90, 140, 230, 370, 600, 1000, 1600, 2600,   4200], 
#     [2,       2,      2,  2,   2,   2,   2,   2,    2,    2,      2], 
#     ['Dense','Dense','VM','VM','VM','VM','VM','VM','VM', 'hash', 'hash'], 
#     1600**2
# )

# lod_meta = _backend.LoDMeta(
#     3, 
#     [34,      55,     90, 140, 230, 370, 600], 
#     [2,       2,      2,  2,   2,   2,   2], 
#     ['Dense','Dense','VM','VM','VM','VM','VM']
# )

def generate_meta():
    return _backend.LoDMeta(
        3, 
        [34,      55,     90, 140, 230, 370, 600, 1000, 1600, 2600, 4200], 
        [2,       2,      2,  2,   2,   2,   2,   2,    2,    2,    2], 
        ['Dense','Dense','VM','VM','VM','VM','VM','VM','VM', 'CPfast', 'CPfast']
    )

def generate_forest():
    level = 3
    pts = torch.tensor(
         [[1,1,0], [1,1,1], [1,1,2], [2,2,2], [3,2,2], [4,2,2]], dtype=torch.int16, device=device)
    octree = unbatched_points_to_octree(pts, level)
    lengths = torch.tensor([len(octree)], dtype=torch.int32)
    spc = Spc(octree, lengths)
    spc._apply_scan_octrees()
    spc._apply_generate_points()
    block_ks = spc.point_hierarchies[spc.pyramids[0,1,level].item(): spc.pyramids[0,1,level+1].item()].contiguous()
    forest = ForestMeta()

    forest.n_trees = block_ks.shape[0]
    forest.level = level
    forest.level_poffset = spc.pyramids[0,1,level].item()
    forest.world_block_size = [1.0, 1.0, 1.0]
    forest.world_origin = [0.0, 0.0, 0.0]
    forest.octree = spc.octrees
    forest.exsum = spc.exsum
    forest.block_ks = block_ks

    return forest

forest = generate_forest()
metas = (lod_meta, forest)

ic(lod_meta.level_res)
ic(lod_meta.level_n_feats)
ic(lod_meta.level_types)
ic(lod_meta.level_offsets)
ic(lod_meta.level_n_params)
ic(lod_meta.map_levels)
ic(lod_meta.map_cnt)

ic(lod_meta.n_levels)
ic(lod_meta.n_pseudo_levels)
ic(lod_meta.n_feat_per_pseudo_lvl)
ic(lod_meta.n_dims_to_encode)
ic(lod_meta.n_encoded_dims)
ic(lod_meta.n_params)

ic(forest.n_trees)
ic(forest.level)
ic(forest.level_poffset)
ic(forest.world_block_size)
ic(forest.world_origin)
ic(forest.octree)
ic(forest.exsum)
ic(forest.block_ks)

params = torch.randn([lod_meta.n_params * forest.n_trees], device=device, dtype=param_dtype) / 1.0e+2
# x = torch.rand([365365, 3], device=device, dtype=input_dtype)
# x = torch.rand([3653653, 3], device=device, dtype=input_dtype)
x = torch.tensor(np.load('./dev_test/test_lotd/input.npz')['x'], device=device, dtype=input_dtype)/2+0.5 # [3.6M]
bidx = torch.randint(forest.n_trees, size=x.shape[:-1], device=device, dtype=torch.long)

y, dydx = _backend.lod_fwd(metas, x, params, bidx, None, None, None, True)
grad = torch.randn_like(y, device=device, dtype=param_dtype) / 1.0e+4
grad_input = torch.randn_like(x, device=device, dtype=input_dtype)
dLdx, dLdparam = _backend.lod_bwd(metas, grad, x, params, dydx, bidx, None, None, None, True, True)
dLddLdy, dLdparam_i, dLdx_i = _backend.lod_bwd_bwd_input(metas, grad_input, grad, x, params, dydx, bidx, None, None, None, True, True, True)

ic(y.shape, y.dtype)
ic(dydx.shape, dydx.dtype)
ic(dLdx.shape, dLdx.dtype) # half,half->half; float,half->float, float,float->float
ic(dLdparam.shape, dLdparam.dtype)
ic(dLddLdy.shape, dLddLdy.dtype)
ic(dLdparam_i.shape, dLdparam_i.dtype)
ic(dLdx_i.shape, dLdx_i.dtype)

# 20 us 
print(Timer(
    stmt='generate_meta()',
    globals={'generate_meta':generate_meta}
).blocked_autorange())

# 4.54 ms
print(Timer(
    stmt='generate_forest()',
    globals={'generate_forest':generate_forest}
).blocked_autorange())

#  ms @ float,half + 360k rand pts
# 9.43 ms @ float,half + 3.6M real pts
print(Timer(
    stmt='_backend.lod_fwd(metas, x, params, bidx, None, None, max_level, False)',
    globals={'_backend':_backend, 'metas': metas, 'x':x, 'params':params, 'bidx':bidx, 'max_level': max_level}
).blocked_autorange())

#  ms @ float,half + 360k rand pts
# 39.18 ms @ float,half + 3.6M real pts
print(Timer(
    stmt='_backend.lod_fwd(metas, x, params, bidx, None, None, max_level, True)',
    globals={'_backend':_backend, 'metas': metas, 'x':x, 'params':params, 'bidx':bidx, 'max_level': max_level}
).blocked_autorange())

#  ms @ float,half + 360k rand pts
# 9.07 ms @ float,half + 3.6M real pts
print(Timer(
    stmt='_backend.lod_bwd(metas, grad, x, params, dydx, bidx, None, None, max_level, True, False)',
    globals={'_backend':_backend, 'metas': metas, 'x':x, 'params':params, 'dydx':dydx, 'grad': grad, 'bidx':bidx, 'max_level': max_level}
).blocked_autorange())

#  ms @ float,half + 360k rand pts
# 79.05 ms @ float,half + 3.6M real pts
print(Timer(
    stmt='_backend.lod_bwd(metas, grad, x, params, dydx, bidx, None, None, max_level, False, True)',
    globals={'_backend':_backend, 'metas': metas, 'x':x, 'params':params, 'dydx':dydx, 'grad': grad, 'bidx':bidx, 'max_level': max_level}
).blocked_autorange())

#  ms @ float,half + 360k rand pts
# 86.58 ms @ float,half + 3.6M real pts
print(Timer(
    stmt='_backend.lod_bwd(metas, grad, x, params, dydx, bidx, None, None, max_level, True, True)',
    globals={'_backend':_backend, 'metas': metas, 'x':x, 'params':params, 'dydx':dydx, 'grad': grad, 'bidx':bidx, 'max_level': max_level}
).blocked_autorange())

#  ms @ float,half + 360k rand pts
# 238 ms @ float,half + 3.6M real pts
print(Timer(
    stmt='_backend.lod_bwd_bwd_input(metas, grad_input, grad, x, params, dydx, bidx, None, None, max_level, False, True, False)', 
    globals={'_backend':_backend, 'metas': metas, 'x':x, 'params':params, 'dydx':dydx, 'grad': grad, 'grad_input':grad_input, 'bidx':bidx, 'max_level': max_level}
).blocked_autorange())

#  ms @ float,half + 360k rand pts
# 281 ms @ float,half + 3.6M real pts
print(Timer(
    stmt='_backend.lod_bwd_bwd_input(metas, grad_input, grad, x, params, dydx, bidx, None, None, max_level, True, True, False)', 
    globals={'_backend':_backend, 'metas': metas, 'x':x, 'params':params, 'dydx':dydx, 'grad': grad, 'grad_input':grad_input, 'bidx':bidx, 'max_level': max_level}
).blocked_autorange())

#  ms @ float,half + 360k rand pts
# 330 ms @ float,half + 3.6M real pts
print(Timer(
    stmt='_backend.lod_bwd_bwd_input(metas, grad_input, grad, x, params, dydx, bidx, None, None, max_level, True, True, True)', 
    globals={'_backend':_backend, 'metas': metas, 'x':x, 'params':params, 'dydx':dydx, 'grad': grad, 'grad_input':grad_input, 'bidx':bidx, 'max_level': max_level}
).blocked_autorange())
import faulthandler; faulthandler.enable()

import math
import torch
import numpy as np
from icecream import ic
from torch.utils.benchmark import Timer
import nr3d_lib_bindings._lotd as _backend

torch.cuda.manual_seed_all(42)

device = torch.device("cuda")
input_dtype = torch.float
param_dtype = torch.float16

max_level = None
# lod_meta = _backend.LoDMeta(3, [4,8,16,32,64], [4,2,4,16,4], ["Dense","Dense","VM","NPlaneMul","CPfast"])
lod_meta = _backend.LoDMeta(
    3, 
    [34,      55,     90, 140, 230, 370, 600, 1000, 1600], 
    [2,       2,      2,  2,   2,   2,   2,   2,    2], 
    ['Dense','Dense','VM','VM','VM','VM','VM','VM','VM'], 
    None, 
    False # use_smooth_step
)
# lod_meta = _backend.LoDMeta(3, [8], [4], ["VecZMatXoY"])
# lod_meta = _backend.LoDMeta(3, [8], [4], ["VM"])

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

ic(lod_meta.interpolation_type)

params = torch.randn([lod_meta.n_params], device=device, dtype=param_dtype) / 1.0e+2
# x = torch.rand([365365, 3], device=device, dtype=input_dtype)
# x = torch.rand([3653653, 3], device=device, dtype=input_dtype)
x = torch.tensor(np.load('./dev_test/test_lotd/input.npz')['x'], device=device, dtype=input_dtype)/2+0.5 # [3.6M]


y, dydx = _backend.lod_fwd(lod_meta, x, params, None, None, None, max_level, True)
grad = torch.randn_like(y, device=device, dtype=param_dtype) / 1.0e+4
grad_input = torch.randn_like(x, device=device, dtype=input_dtype)
dLdx, dLdparam = _backend.lod_bwd(lod_meta, grad, x, params, dydx, None, None, None, max_level, True, True)
dLddLdy, dLdparam_i, dLdx_i = _backend.lod_bwd_bwd_input(lod_meta, grad_input, grad, x, params, dydx, None, None, None, max_level, True, True, True)

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

# 2.59 ms @ float,half + 360k rand pts
# 5.14 ms @ float,half + 3.6M real pts
print(Timer(
    stmt='_backend.lod_fwd(lod_meta, x, params, None, None, None, max_level, False)',
    globals={'_backend':_backend, 'lod_meta':lod_meta, 'x':x, 'params':params, 'max_level': max_level}
).blocked_autorange())

# 10.7 ms @ float,half + 360k rand pts
# 20.2 ms @ float,half + 3.6M real pts
print(Timer(
    stmt='_backend.lod_fwd(lod_meta, x, params, None, None, None, max_level, True)',
    globals={'_backend':_backend, 'lod_meta':lod_meta, 'x':x, 'params':params, 'max_level': max_level}
).blocked_autorange())

# 0.904 ms @ float,half + 360k rand pts
# 9.15 ms @ float,half + 3.6M real pts
print(Timer(
    stmt='_backend.lod_bwd(lod_meta, grad, x, params, dydx, None, None, None, max_level, True, False)',
    globals={'_backend':_backend, 'lod_meta':lod_meta, 'x':x, 'params':params, 'dydx':dydx, 'grad': grad, 'max_level': max_level}
).blocked_autorange())

# 12.5 ms @ float,half + 360k rand pts
# 208 ms @ float,half + 3.6M real pts
print(Timer(
    stmt='_backend.lod_bwd(lod_meta, grad, x, params, dydx, None, None, None, max_level, False, True)',
    globals={'_backend':_backend, 'lod_meta':lod_meta, 'x':x, 'params':params, 'dydx':dydx, 'grad': grad, 'max_level': max_level}
).blocked_autorange())

# 13.5 ms @ float,half + 360k rand pts
# 230 ms @ float,half + 3.6M real pts
print(Timer(
    stmt='_backend.lod_bwd(lod_meta, grad, x, params, dydx, None, None, None, max_level, True, True)',
    globals={'_backend':_backend, 'lod_meta':lod_meta, 'x':x, 'params':params, 'dydx':dydx, 'grad': grad, 'max_level': max_level}
).blocked_autorange())

# 36.8 ms @ float,half + 360k rand pts
# 618 ms @ float,half + 3.6M real pts
print(Timer(
    stmt='_backend.lod_bwd_bwd_input(lod_meta, grad_input, grad, x, params, dydx, None, None, None, max_level, False, True, False)', 
    globals={'_backend':_backend, 'lod_meta':lod_meta, 'x':x, 'params':params, 'dydx':dydx, 'grad': grad, 'grad_input':grad_input, 'max_level': max_level}
).blocked_autorange())

# 41.7 ms @ float,half + 360k rand pts
# 678 ms @ float,half + 3.6M real pts
print(Timer(
    stmt='_backend.lod_bwd_bwd_input(lod_meta, grad_input, grad, x, params, dydx, None, None, None, max_level, True, True, False)', 
    globals={'_backend':_backend, 'lod_meta':lod_meta, 'x':x, 'params':params, 'dydx':dydx, 'grad': grad, 'grad_input':grad_input, 'max_level': max_level}
).blocked_autorange())

# 56.1 ms @ float,half + 360k rand pts
# 681 ms @ float,half + 3.6M real pts
print(Timer(
    stmt='_backend.lod_bwd_bwd_input(lod_meta, grad_input, grad, x, params, dydx, None, None, None, max_level, True, True, True)', 
    globals={'_backend':_backend, 'lod_meta':lod_meta, 'x':x, 'params':params, 'dydx':dydx, 'grad': grad, 'grad_input':grad_input, 'max_level': max_level}
).blocked_autorange())
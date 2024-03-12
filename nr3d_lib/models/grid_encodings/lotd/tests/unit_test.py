import faulthandler; faulthandler.enable()

import math
import torch
import numpy as np
from icecream import ic
from torch.utils.benchmark import Timer
import nr3d_lib.bindings._lotd as _backend

torch.cuda.manual_seed_all(42)

device = torch.device('cuda')
input_dtype = torch.float
param_dtype = torch.float16

max_level = None
# lod_meta = _backend.LoDMeta(3, [4,8,16,32,64], [4,2,4,16,4], ["Dense","Dense","VM","NPlaneMul","CPfast"])
# lod_meta = _backend.LoDMeta(
#     3, 
#     [34,      55,     90, 140, 230, 370, 600, 1000, 1600], 
#     [2,       2,      2,  2,   2,   2,   2,   2,    2], 
#     ['Dense','Dense','VM','VM','VM','VM','VM','VM','VM'], 
#     None, 
#     False # use_smooth_step
# )
lod_meta = _backend.LoDMeta(
    3, 
    [34,      55,     90, 140, 230, 370, 600, 1000, 1600], 
    [2,       2,      2,  2,   2,   2,   2,   2,    2], 
    ['Dense','Dense','Hash','Hash','Hash','Hash','Hash','Hash','Hash'], 
    2**20, # hashmap_size
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

#---- Performance configs
# lod_meta.c_hash_only = True # By default
# NOTE: In hash_only mode, if both c_prefetch and c_permute_dydx are set to False, \
#       the performance is closed to non-hashonly (original) one
# lod_meta.c_prefetch = True # By default
# lod_meta.c_permute_dydx = True # By default
# lod_meta.c_bmm_backend = 1 # By default

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
# dLddLdy, dLdparam_i, dLdx_i = _backend.lod_bwd_bwd_input(lod_meta, grad_input, grad, x, params, dydx, None, None, None, max_level, False, False, True)
# dLddLdy, dLdparam_i, dLdx_i = _backend.lod_bwd_bwd_input(lod_meta, grad_input, grad, x, params, dydx, None, None, None, max_level, False, True, False)
# dLddLdy, dLdparam_i, dLdx_i = _backend.lod_bwd_bwd_input(lod_meta, grad_input, grad, x, params, dydx, None, None, None, max_level, True, False, False)
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

print("\n\n#---- Forward: ") 
#---- Dense+VM
# 2.59 ms @ float,half + 360k rand pts
# 5.14 ms @ float,half + 3.6M real pts
#---- Dense+hash
# 0.86 ms @ float,half + 360k rand pts
# 5.83 ms @ float,half + 3.6M real pts
#---- Dense+hash (hashonly impl)
# 0.34 ms @ float,half + 360k rand pts; non-switch-case: 1.21 ms
# 1.24 ms @ float,half + 3.6M real pts; non-switch-case: 5.61 ms
print(Timer(
    stmt='_backend.lod_fwd(lod_meta, x, params, None, None, None, max_level, False)',
    globals={'_backend':_backend, 'lod_meta':lod_meta, 'x':x, 'params':params, 'max_level': max_level}
).blocked_autorange())



print("\n\n#---- Forward with dydx: ") 
#---- Dense+VM
# 10.7 ms @ float,half + 360k rand pts
# 20.2 ms @ float,half + 3.6M real pts
#---- Dense+hash
# 3.38 ms @ float,half + 360k rand pts
# 18.5 ms @ float,half + 3.6M real pts
#---- Dense+hash (hashonly impl)
# 0.62 ms @ float,half + 360k rand pts; no-prefetch: 2.39 ms; no-permute-dydx: 1.21 ms; non-switch-case: 1.56 ms
# 2.87 ms @ float,half + 3.6M real pts; no-prefetch: 12.2 ms; no-permute-dydx: 7.53 ms; non-switch-case: 7.23 ms
print(Timer(
    stmt='_backend.lod_fwd(lod_meta, x, params, None, None, None, max_level, True)',
    globals={'_backend':_backend, 'lod_meta':lod_meta, 'x':x, 'params':params, 'max_level': max_level}
).blocked_autorange())



print("\n\n#---- Backward input: ") 
#---- Dense+VM
# 0.904 ms @ float,half + 360k rand pts
# 9.15 ms @ float,half + 3.6M real pts
#---- Dense+hash
# 0.87 ms @ float,half + 360k rand pts; bmm_backend=0: 0.87 ms
# 8.71 ms @ float,half + 3.6M real pts; bmm_backend=0: 8.80 ms
#---- Dense+hash (hashonly impl)
# 0.37 ms @ float,half + 360k rand pts; bmm_backend=0: 1.06 ms
# 3.69 ms @ float,half + 3.6M real pts; bmm_backend=0: 10.5 ms
print(Timer(
    stmt='_backend.lod_bwd(lod_meta, grad, x, params, dydx, None, None, None, max_level, True, False)',
    globals={'_backend':_backend, 'lod_meta':lod_meta, 'x':x, 'params':params, 'dydx':dydx, 'grad': grad, 'max_level': max_level}
).blocked_autorange())



print("\n\n#---- Backward gradient: ") 
#---- Dense+VM
# 12.5 ms @ float,half + 360k rand pts
# 208 ms @ float,half + 3.6M real pts
#---- Dense+hash
# 1.52 ms @ float,half + 360k rand pts
# 38.0 ms @ float,half + 3.6M real pts
#---- Dense+hash (hashonly impl)
# 1.04 ms @ float,half + 360k rand pts
# 36.0 ms @ float,half + 3.6M real pts
print(Timer(
    stmt='_backend.lod_bwd(lod_meta, grad, x, params, dydx, None, None, None, max_level, False, True)',
    globals={'_backend':_backend, 'lod_meta':lod_meta, 'x':x, 'params':params, 'dydx':dydx, 'grad': grad, 'max_level': max_level}
).blocked_autorange())



print("\n\n#---- Double backward dL_dparam: ") 
#---- Dense+VM
#  ms @ float,half + 360k rand pts
#  ms @ float,half + 3.6M real pts
#---- Dense+hash
# 5.65 ms @ float,half + 360k rand pts
# 110.8 ms @ float,half + 3.6M real pts
#---- Dense+hash (hashonly impl)
# 5.59 ms @ float,half + 360k rand pts
# 110.1 ms @ float,half + 3.6M real pts
print(Timer(
    stmt='_backend.lod_bwd_bwd_input(lod_meta, grad_input, grad, x, params, dydx, None, None, None, max_level, False, True, False)', 
    globals={'_backend':_backend, 'lod_meta':lod_meta, 'x':x, 'params':params, 'dydx':dydx, 'grad': grad, 'grad_input':grad_input, 'max_level': max_level}
).blocked_autorange())



print("\n\n#---- Double backward dL_ddLdy: ") 
#---- Dense+VM
#  ms @ float,half + 360k rand pts
#  ms @ float,half + 3.6M real pts
#---- Dense+hash
# 0.42 ms @ float,half + 360k rand pts; bmm_backend=0: 5.04 ms
# 4.11 ms @ float,half + 3.6M real pts; bmm_backend=0: 50.9 ms
#---- Dense+hash (hashonly impl)
# 0.46 ms @ float,half + 360k rand pts; bmm_backend=0: 5.07 ms
# 4.70 ms @ float,half + 3.6M real pts; bmm_backend=0: 51.0 ms
print(Timer(
    stmt='_backend.lod_bwd_bwd_input(lod_meta, grad_input, grad, x, params, dydx, None, None, None, max_level, True, False, False)', 
    globals={'_backend':_backend, 'lod_meta':lod_meta, 'x':x, 'params':params, 'dydx':dydx, 'grad': grad, 'grad_input':grad_input, 'max_level': max_level}
).blocked_autorange())



print("\n\n#---- Double backward dL_ddLdy & dL_dparam: ") 
#---- Dense+VM
# 41.7 ms @ float,half + 360k rand pts
# 678 ms @ float,half + 3.6M real pts
#---- Dense+hash
# 6.04 ms @ float,half + 360k rand pts; bmm_backend=0: 10.5 ms
# 117 ms @ float,half + 3.6M real pts; bmm_backend=0: 166 ms
#---- Dense+hash (hashonly impl)
# 6.08 ms @ float,half + 360k rand pts; bmm_backend=0: 10.4 ms
# 116 ms @ float,half + 3.6M real pts; bmm_backend=0: 162 ms
print(Timer(
    stmt='_backend.lod_bwd_bwd_input(lod_meta, grad_input, grad, x, params, dydx, None, None, None, max_level, True, True, False)', 
    globals={'_backend':_backend, 'lod_meta':lod_meta, 'x':x, 'params':params, 'dydx':dydx, 'grad': grad, 'grad_input':grad_input, 'max_level': max_level}
).blocked_autorange())



print("\n\n#---- Double backward dL_ddLdy & dL_dparam & dL_dinput: ") 
#---- Dense+VM
# 56.1 ms @ float,half + 360k rand pts
# 681 ms @ float,half + 3.6M real pts
#---- Dense+hash
# 8.73 ms @ float,half + 360k rand pts; bmm_backend=0: 13.3 ms
# 126 ms @ float,half + 3.6M real pts; bmm_backend=0: 171 ms
#---- Dense+hash (hashonly impl)
# 8.78 ms @ float,half + 360k rand pts; bmm_backend=0: 13.2 ms
# 124 ms @ float,half + 3.6M real pts; bmm_backend=0: 173 ms
print(Timer(
    stmt='_backend.lod_bwd_bwd_input(lod_meta, grad_input, grad, x, params, dydx, None, None, None, max_level, True, True, True)', 
    globals={'_backend':_backend, 'lod_meta':lod_meta, 'x':x, 'params':params, 'dydx':dydx, 'grad': grad, 'grad_input':grad_input, 'max_level': max_level}
).blocked_autorange())
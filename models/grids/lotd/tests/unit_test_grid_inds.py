import faulthandler; faulthandler.enable()

import math
import torch
import numpy as np

import nr3d_lib_bindings._lotd as _backend

torch.cuda.manual_seed_all(42)

device = torch.device("cuda")
input_dtype = torch.float
param_dtype = torch.float16

max_level = None
lod_meta = _backend.LoDMeta(
    3, 
    [4,      8], 
    [4,       4], 
    ['Dense','Dense'], 
    None, 
    False # use_smooth_step
)

params = torch.randn([lod_meta.n_params], device=device, dtype=param_dtype) / 1.0e+2

x = torch.tensor([[0.7, 0.5, 0.2], [-0.9, 0, 0.3]], device=device, dtype=input_dtype)
# x = torch.rand([365365, 3], device=device, dtype=input_dtype)
# x = torch.tensor(np.load('./dev_test/test_lotd/input.npz')['x'], device=device, dtype=input_dtype)/2+0.5 # [3.6M]

grid_inds = _backend.lod_get_grid_index(lod_meta, x, None, None, None, max_level)
y, dydx = _backend.lod_fwd(lod_meta, x, params, None, None, None, max_level, True)
print(y)

params[grid_inds] = 0
y, dydx = _backend.lod_fwd(lod_meta, x, params, None, None, None, max_level, True)
print(y)

y, dydx = _backend.lod_fwd(lod_meta, x+0.2, params, None, None, None, max_level, True)
print(y)

import faulthandler; faulthandler.enable()
from icecream import ic
import numpy as np

import torch
from torch.utils.benchmark import Timer

import nr3d_lib.bindings._permuto as _backend
# import nr3d_lib.bindings._permuto_thrust as _backend
# import nr3d_lib.bindings._permuto_quicksort as _backend

input_dtype = torch.float32
param_dtype = torch.float32
device = torch.device('cuda')

supported_n_input_dims = _backend.supported_n_input_dims
ic(supported_n_input_dims)

meta = _backend.PermutoEncMeta(
    64, # n_input_dim
    2**16, # hashmap_size
    [16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0], # res_list
    [4, 4, 2, 2, 2, 2, 2, 2], # n_feats_list
)

# meta = _backend.PermutoEncMeta(
#     7, # n_input_dim
#     2**18, # hashmap_size
#     1./np.geomspace(1.0, 0.0001, num=24), # res_list
#     [2]*24, # n_feats_list
# )

# batch_size = 1
# batch_size = 4
# batch_size = 1024
batch_size = 3653653
positions = torch.rand([batch_size, meta.n_dims_to_encode], dtype=input_dtype, device=device)
lattice_values = torch.randn([meta.n_params], dtype=param_dtype, device=device)
level_random_shifts = 10.0 * torch.randn([meta.n_levels, meta.n_dims_to_encode], dtype=input_dtype, device=device)

encoded = _backend.permuto_enc_fwd(
    meta, positions, lattice_values, level_random_shifts, None, None, None, None)

ic(encoded.shape)

# The trailing two bools: need_input_grad, need_param_grad
dL_dy = torch.randn_like(encoded)
dL_dx, dL_dlattice_val = _backend.permuto_enc_bwd(
    meta, dL_dy, positions, lattice_values, level_random_shifts, None, None, None, None, None, True, True
)

dL_dx_3, _ = _backend.permuto_enc_bwd(
    meta, dL_dy, positions, lattice_values, level_random_shifts, None, None, None, None, 3, True, True
)

# The trailing two bools: need_dL_ddLdy, need_dL_dparams
dL_ddLdx = torch.randn_like(dL_dx)
dL_ddLdy, dL_dlattice_val2 = _backend.permuto_enc_bwd_bwd_input(
    meta, dL_ddLdx, dL_dy, positions, lattice_values, level_random_shifts, None, None, None, None, True, True
)


"""
Temporal benchmark evaluated @ n_feats_list = [4, 4, 2, 2, 2, 2, 2, 2], n_input_dim=7, batch_size=3653653, 
@param_dtype=half: n_input_dim=7 | 32 | 64
@param_dtype=float: n_input_dim=7 | 32 | 64

Note that the computation costs about O(d^2)
"""
print("\n\n#---- Forward: ") 
#---- N_THREADS = 64
# half: 12.22 ms | 79.4 ms | 419 ms
# float: 6.97 ms | 80.3 ms | 464 ms
#---- N_THREADS = 128
# half: 12.31 ms | 80.8 ms | 434 ms
# float: 7.26 ms | 81.3 ms | 451 ms
#---- N_THREADS = 256
# half: 12.39 ms | 92.6 ms | 434 ms
# float: 6.98 ms | 93.8 ms | 480 ms
#---- N_THREADS = 512
# half: 12.52 ms | 122 ms | 820~900 ms
# float: 7.39 ms | 122 ms | 766 ms
print(Timer(
    stmt="_backend.permuto_enc_fwd(meta, positions, lattice_values, level_random_shifts, None, None, None, None)", 
    globals={'_backend': _backend, 'meta':meta, 'positions':positions, 'lattice_values':lattice_values, 'level_random_shifts':level_random_shifts}
).blocked_autorange())

print("\n\n#---- Backward input: ") 
#---- N_THREADS_BACK = 64
# half:  17.03 ms | 118 ms | 413 ms
# float: 16.96 ms | 123 ms | 420 ms
#---- N_THREADS_BACK = 128
# half:  19.77 ms | 116 ms | 423 ms
# float: 20.01 ms | 120 ms | 432 ms
#---- N_THREADS_BACK = 256
# half:  18.07 ms | 114 ms | 452 ms
# float: 19.50 ms | 116 ms | 454 ms
#---- N_THREADS_BACK = 512
# half: 18.52 ms | 114 ms | 682 ms
# float: 18.27 ms | 116 ms | 696 ms
print(Timer(
    stmt="_backend.permuto_enc_bwd(meta, dL_dy, positions, lattice_values, level_random_shifts, None, None, None, None, None, True, False)", 
    globals={'_backend': _backend, 'meta':meta, 'positions':positions, 'lattice_values':lattice_values, 'dL_dy':dL_dy, 'level_random_shifts':level_random_shifts}
).blocked_autorange())

print("\n\n#---- Backward input 3D: ") 
#---- N_THREADS_BACK = 64
# half:  14.31 ms | 89 ms | 299 ms
# float: 13.77 ms | 96 ms | 311 ms
#---- N_THREADS_BACK = 128
# half:  15.48 ms | 87 ms | 308 ms
# float: 15.53 ms | 91 ms | 320 ms
#---- N_THREADS_BACK = 256
# half:  15.32 ms | 83 ms | 329 ms
# float: 15.26 ms | 85 ms | 334 ms
#---- N_THREADS_BACK = 512
# half: 15.31 ms | 84 ms | 571 ms
# float: 15.23 ms | 86 ms | 618 ms
print(Timer(
    stmt="_backend.permuto_enc_bwd(meta, dL_dy, positions, lattice_values, level_random_shifts, None, None, None, None, 3, True, False)", 
    globals={'_backend': _backend, 'meta':meta, 'positions':positions, 'lattice_values':lattice_values, 'dL_dy':dL_dy, 'level_random_shifts':level_random_shifts}
).blocked_autorange())

print("\n\n#---- Backward gradient: ")  
#---- N_THREADS_BACK = 64
# half:  12.30 ms |  89 ms | 427 ms
# float: 10.66 ms | 103 ms | 476 ms
#---- N_THREADS_BACK = 128
# half:  12.28 ms |  90 ms | 436 ms
# float: 10.66 ms | 108 ms | 491 ms
#---- N_THREADS_BACK = 256
# half:  12.38 ms |  99 ms | 450 ms
# float: 10.53 ms | 106 ms | 511 ms
#---- N_THREADS_BACK = 512
# half:  12.43 ms | 130 ms | 886 ms
# float: 10.70 ms | 161 ms | 933 ms
print(Timer(
    stmt="_backend.permuto_enc_bwd(meta, dL_dy, positions, lattice_values, level_random_shifts, None, None, None, None, None, False, True)", 
    globals={'_backend': _backend, 'meta':meta, 'positions':positions, 'lattice_values':lattice_values, 'dL_dy':dL_dy, 'level_random_shifts':level_random_shifts}
).blocked_autorange())


print("\n\n#---- Double backward dL_ddLdy & dL_dparam: ") 
#---- N_THREADS_BACK = 64
# half:  22.27 ms | 164 ms | 526 ms
# float: 27.48 ms | 187 ms | 561 ms
#---- N_THREADS_BACK = 128
# half:  24.52 ms | 164 ms | 544 ms
# float: 29.43 ms | 187 ms | 564 ms
#---- N_THREADS_BACK = 256
# half:  24.22 ms | 164 ms | 570 ms
# float: 29.28 ms | 183 ms | 601 ms
#---- N_THREADS_BACK = 512
# half:  23.41 ms | 171 ms | 808 ms
# float: 28.41 ms | 191 ms | 800 ms
print(Timer(
    stmt="_backend.permuto_enc_bwd_bwd_input(meta, dL_ddLdx, dL_dy, positions, lattice_values, level_random_shifts, None, None, None, None, True, True)", 
    globals={'_backend': _backend, 'meta':meta, 'positions':positions, 'lattice_values':lattice_values, 'dL_ddLdx':dL_ddLdx, 'dL_dy':dL_dy, 'level_random_shifts':level_random_shifts}
).blocked_autorange())

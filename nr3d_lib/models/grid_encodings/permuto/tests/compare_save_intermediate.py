import faulthandler; faulthandler.enable()
from icecream import ic

import torch
from torch.utils.benchmark import Timer

import nr3d_lib.bindings._permuto_intermediate as _backend_1
import nr3d_lib.bindings._permuto as _backend_2

input_dtype = torch.float32
param_dtype = torch.float16
device = torch.device('cuda')

meta = _backend_1.PermutoEncMeta(
    7, # n_input_dim
    2**16, # hashmap_size
    [16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0], # res_list
    # [4, 4, 2, 2, 2, 2, 2, 2], # n_feats_list
    [2, 2, 2, 2, 2, 2, 2, 2], # n_feats_list
)

# meta = _backend_1.PermutoEncMeta(
#     2, # n_input_dim
#     1024, # hashmap_size
#     [4.0], # res_list
#     [2], # n_feats_list
# )

# batch_size = 1
# batch_size = 4
# batch_size = 1024
batch_size = 3653653
positions = torch.rand([batch_size, meta.n_dims_to_encode], dtype=input_dtype, device=device)
lattice_values = torch.randn([meta.n_params], dtype=param_dtype, device=device)


# The trailing two bools: need_encoded, need_intermediate
encoded, rank, rem0 = _backend_1.permuto_enc_fwd(
    meta, positions, lattice_values, None, None, None, None, True, True)

ic(encoded.shape)

encoded_2 = _backend_2.permuto_enc_fwd(meta, positions, lattice_values, None, None, None, None)

# The trailing two bools: need_input_grad, need_param_grad
dL_dy = torch.randn_like(encoded)
dL_dx, dL_dlattice_val = _backend_1.permuto_enc_bwd(
    meta, dL_dy, positions, lattice_values, rank, rem0, None, None, None, None, True, True
)

dL_dx_2, dL_dlattice_val_2 = _backend_2.permuto_enc_bwd(
    meta, dL_dy, positions, lattice_values, None, None, None, None, None, True, True
)

# The trailing two bools: need_dL_ddLdy, need_dL_dparams
dL_ddLdx = torch.randn_like(dL_dx)
dL_ddLdy, dL_dlattice_val2 = _backend_1.permuto_enc_bwd_bwd_input(
    meta, dL_ddLdx, dL_dy, positions, lattice_values, rank, rem0, None, None, None, None, True, True
)

dL_ddLdy_2, dL_dlattice_val2_2 = _backend_2.permuto_enc_bwd_bwd_input(
    meta, dL_ddLdx, dL_dy, positions, lattice_values, None, None, None, None, True, True
)

print(torch.allclose(encoded, encoded_2))
print(torch.allclose(dL_dx, dL_dx_2))
print(torch.allclose(dL_dlattice_val, dL_dlattice_val_2)) # False, the differences are bigger at higher resolutions; there are significant errors at a few points due to misalignment.
print(torch.allclose(dL_ddLdy, dL_ddLdy_2))
print(torch.allclose(dL_dlattice_val2, dL_dlattice_val2_2)) # False, the differences are bigger at higher resolutions; there are significant errors at a few points due to misalignment.

"""
As it turns out: the time spent on additional storage and retrieval of rank and rem0 here actually exceeds the time spent on recalculating rank and rem0 (calculation is faster than memory read)
Especially for GPU memory storage, even when all level n_feat are equal (no memory write conflict), writing rank and rem0 into the GPU memory is very slow.
"""

print("\n\n#---- Forward: ") # 36.52 ms
print(Timer(
    stmt="_backend_1.permuto_enc_fwd(meta, positions, lattice_values, None, None, None, None, True, True)", 
    globals={'_backend_1': _backend_1, 'meta':meta, 'positions':positions, 'lattice_values':lattice_values}
).blocked_autorange())

print("\n\n#---- Forward2: ") # 4.77 ms
print(Timer(
    stmt="_backend_2.permuto_enc_fwd(meta, positions, lattice_values, None, None, None, None)", 
    globals={'_backend_2': _backend_2, 'meta':meta, 'positions':positions, 'lattice_values':lattice_values}
).blocked_autorange())

print("\n\n#---- Backward input: ") # 17.04 ms
print(Timer(
    stmt="_backend_1.permuto_enc_bwd(meta, dL_dy, positions, lattice_values, rank, rem0, None, None, None, None, True, False)", 
    globals={'_backend_1': _backend_1, 'meta':meta, 'positions':positions, 'lattice_values':lattice_values, 'dL_dy':dL_dy, 'rank':rank, 'rem0':rem0}
).blocked_autorange())


print("\n\n#---- Backward2 input: ") # 14.48 ms
print(Timer(
    stmt="_backend_2.permuto_enc_bwd(meta, dL_dy, positions, lattice_values, None, None, None, None, None, True, False)", 
    globals={'_backend_2': _backend_2, 'meta':meta, 'positions':positions, 'lattice_values':lattice_values, 'dL_dy':dL_dy}
).blocked_autorange())

print("\n\n#---- Backward gradient: ")  # 28.81 ms
print(Timer(
    stmt="_backend_1.permuto_enc_bwd(meta, dL_dy, positions, lattice_values, rank, rem0, None, None, None, None, False, True)", 
    globals={'_backend_1': _backend_1, 'meta':meta, 'positions':positions, 'lattice_values':lattice_values, 'dL_dy':dL_dy, 'rank':rank, 'rem0':rem0}
).blocked_autorange())


print("\n\n#---- Backward2 gradient: ") # 9.96 ms
print(Timer(
    stmt="_backend_2.permuto_enc_bwd(meta, dL_dy, positions, lattice_values, None, None, None, None, None, False, True)", 
    globals={'_backend_2': _backend_2, 'meta':meta, 'positions':positions, 'lattice_values':lattice_values, 'dL_dy':dL_dy}
).blocked_autorange())

print("\n\n#---- Double backward dL_ddLdy & dL_dparam: ") # 21.00 ms
print(Timer(
    stmt="_backend_1.permuto_enc_bwd_bwd_input(meta, dL_ddLdx, dL_dy, positions, lattice_values, rank, rem0, None, None, None, None, True, True)", 
    globals={'_backend_1': _backend_1, 'meta':meta, 'positions':positions, 'lattice_values':lattice_values, 'dL_ddLdx':dL_ddLdx, 'dL_dy':dL_dy, 'rank':rank, 'rem0':rem0}
).blocked_autorange())

print("\n\n#---- Double backward2 dL_ddLdy & dL_dparam: ") # 18.90 ms
print(Timer(
    stmt="_backend_2.permuto_enc_bwd_bwd_input(meta, dL_ddLdx, dL_dy, positions, lattice_values, None, None, None, None, True, True)", 
    globals={'_backend_2': _backend_2, 'meta':meta, 'positions':positions, 'lattice_values':lattice_values, 'dL_ddLdx':dL_ddLdx, 'dL_dy':dL_dy}
).blocked_autorange())
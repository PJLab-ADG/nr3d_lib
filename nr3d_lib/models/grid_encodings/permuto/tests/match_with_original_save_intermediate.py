import faulthandler; faulthandler.enable()

import numpy as np
from icecream import ic

import torch
from torch import autograd
from torch.utils.benchmark import Timer

import nr3d_lib.bindings._permuto_intermediate as _backend

import permutohedral_encoding as permuto_enc # Original built

input_dtype = torch.float32
param_dtype = torch.float32
device = torch.device('cuda')

pos_dim=7
capacity=2**18
nr_levels = 24 
nr_feat_per_level = 2 
coarsest_scale = 1.0 
finest_scale = 0.0001 
scale_list = np.geomspace(coarsest_scale, finest_scale, num=nr_levels)
res_list = 1./ scale_list

lattice_values0 = torch.randn([nr_levels, capacity, nr_feat_per_level], dtype=param_dtype, device=device)
lattice_values = lattice_values0.flatten()

#---- Original
enc0 = permuto_enc.PermutoEncoding(pos_dim, capacity, nr_levels, nr_feat_per_level, scale_list)
enc0.lattice_values.data = lattice_values0

#---- Ours
meta = _backend.PermutoEncMeta(pos_dim, capacity, res_list.tolist(), [nr_feat_per_level] * len(res_list), enc0.random_shift_per_level.tolist())

# batch_size = 1
# batch_size = 4
# batch_size = 1024
batch_size = 3653653
positions = torch.rand([batch_size, meta.n_dims_to_encode], dtype=input_dtype, device=device)

#---- Original Forward
encoded0 = enc0(positions)
@torch.no_grad()
def fn_original_fwd():
    enc0(positions)

#---- Ours Forward
encoded, rank, rem0 = _backend.permuto_enc_fwd(meta, positions, lattice_values, None, None, None, None, True, True)
@torch.no_grad()
def fn_ours_fwd(need_intermediate=False):
    # need_intermediate can significantly impact the execution time, potentially turning a 17.59 millisecond process into a 123.38 millisecond one.
    _backend.permuto_enc_fwd(meta, positions, lattice_values, None, None, None, None, True, need_intermediate)

# The error is slightly larger for higher resolutions.
print(torch.allclose(encoded0.data, encoded.data, atol=3.0e-2, rtol=3.0e-2))

# print(Timer(
#     stmt="fn_original_fwd()", 
#     globals={'fn_original_fwd': fn_original_fwd}
# ).blocked_autorange())

print(Timer(
    stmt="fn_ours_fwd()", 
    globals={'fn_ours_fwd': fn_ours_fwd}
).blocked_autorange())

print(Timer(
    stmt="fn_ours_fwd(need_intermediate=True)", 
    globals={'fn_ours_fwd': fn_ours_fwd}
).blocked_autorange())

print(Timer(
    stmt=f"a=torch.zeros([batch_size, nr_levels, pos_dim+1], dtype=torch.int32, device=device)", 
    globals={'device': device, 'batch_size': batch_size, 'nr_levels': nr_levels, 'pos_dim': pos_dim}
).blocked_autorange())
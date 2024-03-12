import faulthandler; faulthandler.enable()

import time
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

#---- Ours
meta = _backend.PermutoEncMeta(pos_dim, capacity, res_list.tolist(), [nr_feat_per_level] * len(res_list))

# batch_size = 1
# batch_size = 4
# batch_size = 1024
batch_size = 3653653
positions = torch.rand([batch_size, meta.n_dims_to_encode], dtype=input_dtype, device=device)

#---- Ours Forward
# encoded, rank, rem0 = _backend.permuto_enc_fwd(meta, positions, lattice_values, None, None, None, None, True, True)
@torch.no_grad()
def fn_ours_fwd(need_intermediate=False):
    # need_intermediate significantly affects the runtime, a process originally taking 17.59 milliseconds might become 123.38 ms.
    _backend.permuto_enc_fwd(meta, positions, lattice_values, None, None, None, None, True, need_intermediate)

time.sleep(3)

print(Timer(
    stmt="fn_ours_fwd(need_intermediate=False)", 
    globals={'fn_ours_fwd': fn_ours_fwd}
).blocked_autorange())

print(Timer(
    stmt="fn_ours_fwd(need_intermediate=True)", 
    globals={'fn_ours_fwd': fn_ours_fwd}
).blocked_autorange())
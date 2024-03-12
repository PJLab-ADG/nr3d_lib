import faulthandler; faulthandler.enable()
from icecream import ic
import numpy as np

import torch
from torch.utils.benchmark import Timer

import nr3d_lib.bindings._permuto_intermediate as _backend

input_dtype = torch.float32
param_dtype = torch.float16
device = torch.device('cuda')

meta = _backend.PermutoEncMeta(
    7, # n_input_dim
    2**16, # hashmap_size
    [32.0], # res_list
    [4], # n_feats_list
)

batch_size = 1
# batch_size = 4
# batch_size = 1024
# batch_size = 3653653
positions = torch.rand([batch_size, meta.n_dims_to_encode], dtype=input_dtype, device=device)
lattice_values = torch.randn([meta.n_params], dtype=param_dtype, device=device)
level_random_shifts = 10.0 * torch.randn([meta.n_levels, meta.n_dims_to_encode], dtype=input_dtype, device=device)

_, rank, rank2, rank3, rem0, elevated = _backend.permuto_enc_fwd(meta, positions, lattice_values, None, None, None, None, False, True)
delta = elevated - rem0
_rank = torch.argsort(torch.argsort(delta, descending=True))

print(torch.equal(rank, _rank))
print(torch.equal(rank, rank2))
print(torch.equal(rank, rank3))

delta_sorted = delta.sort(descending=True, dim=-1).values
_delta = torch.gather(delta_sorted, -1, rank.long())
print(torch.equal(delta, _delta))

_ = 1
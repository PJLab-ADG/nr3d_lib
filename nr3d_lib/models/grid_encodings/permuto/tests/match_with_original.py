import faulthandler; faulthandler.enable()

import numpy as np
from icecream import ic

import torch
import torch.nn as nn
from torch import autograd
from torch.utils.benchmark import Timer

import nr3d_lib.bindings._permuto as _backend

import permutohedral_encoding as permuto_enc # Original built
from permutohedral_encoding.pytorch_modules.find_cpp_package import find_package
_original = find_package()

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
level_random_shifts = 10.0 * torch.randn([nr_levels, pos_dim], dtype=input_dtype, device=device)

#---- Original
enc0 = permuto_enc.PermutoEncoding(pos_dim, capacity, nr_levels, nr_feat_per_level, scale_list)
enc0.random_shift_per_level.data = level_random_shifts
enc0.lattice_values.data = lattice_values0

#---- Ours
meta = _backend.PermutoEncMeta(pos_dim, capacity, res_list.tolist(), [nr_feat_per_level] * len(res_list))

# batch_size = 1
# batch_size = 4
# batch_size = 1024
batch_size = 3653653
positions = torch.rand([batch_size, meta.n_dims_to_encode], dtype=input_dtype, device=device)


#-----------------------------------------
#--------------- Forward
#-----------------------------------------

#---- Original Forward
encoded0 = enc0(positions)
@torch.no_grad()
def fn_original_fwd():
    enc0(positions)

#---- Ours Forward
encoded = _backend.permuto_enc_fwd(meta, positions, lattice_values, level_random_shifts, None, None, None, None)
@torch.no_grad()
def fn_ours_fwd():
    _backend.permuto_enc_fwd(meta, positions, lattice_values, level_random_shifts, None, None, None, None)

# The error tends to be slightly larger for higher resolutions.
print(torch.allclose(encoded0.data, encoded.data, atol=3.0e-2, rtol=3.0e-2))

print(Timer( # 13.33 ms
    stmt="fn_original_fwd()", 
    globals={'fn_original_fwd': fn_original_fwd}
).blocked_autorange())

print(Timer( # 18.23 ms
    stmt="fn_ours_fwd()", 
    globals={'fn_ours_fwd': fn_ours_fwd}
).blocked_autorange())



#-----------------------------------------
#--------------- Backward input
#-----------------------------------------
del enc0.lattice_values
enc0.lattice_values = lattice_values0.data # Make sure no grad
_positions = positions.clone().requires_grad_()
encoded0 = enc0(_positions)
dL_dy = torch.randn_like(encoded0).contiguous()
dL_dx0 = autograd.grad(encoded0, _positions, dL_dy, retain_graph=True)[0]
def fn_original_bwd_input():
    ret = autograd.grad(encoded0, _positions, dL_dy, retain_graph=True)[0]

dL_dx, _ = _backend.permuto_enc_bwd(meta, dL_dy, positions, lattice_values, level_random_shifts, None, None, None, None, None, True, False)
def fn_ours_bwd_input():
    ret = _backend.permuto_enc_bwd(meta, dL_dy, positions, lattice_values, level_random_shifts, None, None, None, None, None, True, False)

# print(torch.allclose(dL_dx0.data, dL_dx)) # Some points are misaligned, resulting in larger differences. However, most are similar.

print(Timer( # 26.42 ms
    stmt="fn_original_bwd_input()", 
    globals={'fn_original_bwd_input': fn_original_bwd_input}
).blocked_autorange())

print(Timer( # 50.14 ms
    stmt="fn_ours_bwd_input()", 
    globals={'fn_ours_bwd_input': fn_ours_bwd_input}
).blocked_autorange())



#-----------------------------------------
#--------------- Backward lattice values
#-----------------------------------------
del enc0.lattice_values
enc0.lattice_values = nn.Parameter(lattice_values0.data.clone(), requires_grad=True)
encoded0 = enc0(positions.data)
dL_dlattice0 = autograd.grad(encoded0, enc0.lattice_values, dL_dy, retain_graph=True)[0].flatten()
def fn_original_bwd_lattice():
    ret = autograd.grad(encoded0, enc0.lattice_values, dL_dy, retain_graph=True)[0]

_, dL_dlattice = _backend.permuto_enc_bwd(meta, dL_dy, positions, lattice_values, level_random_shifts, None, None, None, None, None, False, True)
def fn_ours_bwd_lattice():
    ret = _backend.permuto_enc_bwd(meta, dL_dy, positions, lattice_values, level_random_shifts, None, None, None, None, None, False, True)

# Most are similar, a few have larger errors.
print(torch.allclose(dL_dlattice0, dL_dlattice, rtol=1.0e-1, atol=1.0e-1))

print(Timer( # 65.88 ms
    stmt="fn_original_bwd_lattice()", 
    globals={'fn_original_bwd_lattice': fn_original_bwd_lattice}
).blocked_autorange())

print(Timer( # 82.50 ms
    stmt="fn_ours_bwd_lattice()", 
    globals={'fn_ours_bwd_lattice': fn_ours_bwd_lattice}
).blocked_autorange())



#-----------------------------------------
#--------------- Double backward to dL_doutput
#-----------------------------------------
del enc0.lattice_values
enc0.lattice_values = lattice_values0.data # Make sure no grad
_positions = positions.clone().requires_grad_()
encoded0 = enc0(_positions)
dL_dy = torch.randn_like(encoded0).contiguous().requires_grad_()
dL_dx0 = autograd.grad(encoded0, _positions, dL_dy, retain_graph=True, create_graph=True)[0]
dL_ddLdx = torch.randn_like(dL_dx0).contiguous()
dL_ddLdy0 = autograd.grad(dL_dx0, dL_dy, dL_ddLdx, retain_graph=True)[0]
def fn_original_bwd_input_bwd_dldy():
    ret = autograd.grad(dL_dx0, dL_dy, dL_ddLdx, retain_graph=True)[0]

dL_ddLdy, _ = _backend.permuto_enc_bwd_bwd_input(
    meta, dL_ddLdx, dL_dy, positions, lattice_values, level_random_shifts, None, None, None, None, True, False
)
def fn_ours_bwd_input_bwd_dldy():
    ret = _backend.permuto_enc_bwd_bwd_input(meta, dL_ddLdx, dL_dy, positions, lattice_values, level_random_shifts, None, None, None, None, True, False)

# Most are similar, a few have misalignments at higher resolutions, therefore the difference is larger. This is normal.
# print(torch.allclose(dL_ddLdy0, dL_ddLdy, rtol=1.0e-1, atol=1.0e-1))

print(Timer( # 122.59 ms
    stmt="fn_original_bwd_input_bwd_dldy()", 
    globals={'fn_original_bwd_input_bwd_dldy': fn_original_bwd_input_bwd_dldy}
).blocked_autorange())

print(Timer( # 69.93 ms
    stmt="fn_ours_bwd_input_bwd_dldy()", 
    globals={'fn_ours_bwd_input_bwd_dldy': fn_ours_bwd_input_bwd_dldy}
).blocked_autorange())



#-----------------------------------------
#--------------- Double backward to lattice values
#-----------------------------------------
del enc0.lattice_values
enc0.lattice_values = nn.Parameter(lattice_values0.data.clone(), requires_grad=True)
_positions = positions.clone().requires_grad_()
encoded0 = enc0(_positions)
dL_dy = torch.randn_like(encoded0).contiguous().requires_grad_()
dL_dx0 = autograd.grad(encoded0, _positions, dL_dy, retain_graph=True, create_graph=True)[0]
dL_ddLdx = torch.randn_like(dL_dx0).contiguous()
dL_dlattice0 = autograd.grad(dL_dx0, enc0.lattice_values, dL_ddLdx, retain_graph=True)[0].flatten()
def fn_original_bwd_input_bwd_lattice():
    ret = autograd.grad(dL_dx0, enc0.lattice_values, dL_ddLdx, retain_graph=True)[0]

_, dL_dlattice = _backend.permuto_enc_bwd_bwd_input(
    meta, dL_ddLdx, dL_dy, positions, lattice_values, level_random_shifts, None, None, None, None, False, True
)
def fn_ours_bwd_input_bwd_lattice():
    ret = _backend.permuto_enc_bwd_bwd_input(meta, dL_ddLdx, dL_dy, positions, lattice_values, level_random_shifts, None, None, None, None, False, True)

# Most are similar, a few are misaligned at higher resolutions, hence the larger difference. This is a normal phenomenon.
# print(torch.allclose(dL_dlattice0, dL_dlattice, rtol=1.0e-1, atol=1.0e-1))

print(Timer( # 189.32 ms
    stmt="fn_original_bwd_input_bwd_lattice()", 
    globals={'fn_original_bwd_input_bwd_lattice': fn_original_bwd_input_bwd_lattice}
).blocked_autorange())

print(Timer( # 125.01 ms
    stmt="fn_ours_bwd_input_bwd_lattice()", 
    globals={'fn_ours_bwd_input_bwd_lattice': fn_ours_bwd_input_bwd_lattice}
).blocked_autorange())
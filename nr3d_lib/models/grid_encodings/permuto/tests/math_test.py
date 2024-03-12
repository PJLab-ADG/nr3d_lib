
import faulthandler; faulthandler.enable()

import numpy as np
from icecream import ic

import torch
import torch.nn as nn
from torch import autograd
from torch.utils.benchmark import Timer

import nr3d_lib.bindings._permuto as _backend

from nr3d_lib.models.grid_encodings.permuto import PermutoEncImpl

device = torch.device('cuda')
input_dtype = torch.float32
param_dtype = torch.float32

pos_scale = 10.0

meta = PermutoEncImpl(
    4, 
    [8.0, 16.0], 
    [2, 2], 
    log2_hashmap_size=4, 
    apply_random_shifts_per_level=True, 
    pos_scale=pos_scale, 
    dtype=param_dtype, device=device
)

lattice_values = torch.randn(meta.n_params, dtype=torch.float, device=device, requires_grad=True)
x_eg = torch.randn([7, 4], dtype=input_dtype, device=device)
y_eg = meta.forward(x_eg, lattice_values)
dLdy_eg = torch.randn_like(y_eg)


def apply_on_x(x):
    return meta.forward(x, lattice_values, need_dL_dinput=True)

# NOTE:
#       y   w.r.t. x     i.e. dy_dx
autograd.gradcheck(
    apply_on_x, 
    x_eg.data.clone().requires_grad_(True), 
    eps=1.0e-5, rtol=0.2, atol=0.5 * pos_scale
)






# NOTE: 
#       y   w.r.t. param     i.e. dy_dparam (passed)

def apply_on_param(param):
    return meta.forward(x_eg, param, need_dL_dinput=False)

autograd.gradcheck(
    apply_on_param, 
    lattice_values, 
    eps=1.0e-5, rtol=0.2, atol=0.05 * pos_scale
)





# NOTE: 
#       dLdx   w.r.t. dLdy     i.e. d(dLdx)_d(dLdy) (passed)

def bwdinput_apply_on_dldy(dldy):
    return meta.backward_dydx(dldy, x_eg, lattice_values)

autograd.gradcheck(
    bwdinput_apply_on_dldy, 
    dLdy_eg.data.clone().requires_grad_(True), 
    eps=1.0e-5, rtol=0.2, atol=0.1 * pos_scale
)




# NOTE: Most are correct, a very few have larger errors due to misalignment.
#       dLdx   w.r.t. param     i.e. d(dLdx)_d(param)

def bwdinput_apply_on_param(param):
    return meta.backward_dydx(dLdy_eg, x_eg, param)

autograd.gradcheck(
    bwdinput_apply_on_param, 
    lattice_values, 
    eps=1.0e-5, rtol=0.2, atol=0.1 * pos_scale
)
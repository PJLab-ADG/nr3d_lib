import math
import torch
import numpy as np
from tqdm import tqdm
from icecream import ic

from torch import autograd
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

import nr3d_lib_bindings._lotd as _backend
from nr3d_lib.models.grids.lotd import LoTDFunction, LoTDFunctionBwdDydx

""" NOTE:
- If the resolution is relatively high, larger errors can easily occur at some staggered grid points (after all, torch calculates gradients using the eps discretization method)
- In many cases, if it doesn't pass at once, it can pass next time (the random input of the next time avoids the case of staggered grid points)
- All current tests have passed: "Dense", "VM", "NPlaneSum", "CPfast", "Hash", "NPlaneMul", "CP"
"""


device = torch.device('cuda')
dtype = torch.float
use_smooth_step = False
# lod_meta = _backend.LoDMeta(3, [3], [2], ["Dense"])
# lod_meta = _backend.LoDMeta(3, [4, 8], [2, 4], ["Dense", "VM"], None, use_smooth_step)
# lod_meta = _backend.LoDMeta(3, [4, 8, 4, 8], [2, 4, 2, 4], ["Dense", "VM", "NPlaneMul", "CP"], None, use_smooth_step)
# lod_meta = _backend.LoDMeta(3, [4, 8, 4, 8, 4], [2, 4, 2, 4, 2], ["Dense", "VM", "NPlaneSum", "CPfast", "Hash"], 2**10, use_smooth_step)
lod_meta = _backend.LoDMeta(3, [4,8], [4, 4], ["VecZMatXoY", "VecZMatXoY"])

n_params = lod_meta.n_params

#----------------------------------
# # batched inference
#----------------------------------
batch_size = 4
# x_eg = torch.tensor([[0.3,0.4,0.5]], dtype=torch.float, device=device).repeat([batch_size,1])
x_eg = torch.rand([batch_size,3], dtype=torch.float, device=device, requires_grad=True)
batch_data_size = int(math.prod(x_eg.shape[1:-1]))
params = torch.randn([n_params * batch_size], dtype=dtype, device=device, requires_grad=True)

y_eg, dydx_eg = _backend.lod_fwd(lod_meta, x_eg, params, None, None, batch_data_size, None, True)
grad_eg = torch.randn_like(y_eg, device=device, dtype=dtype)
dLdx_eg, dLdgrid_eg = _backend.lod_bwd(lod_meta, grad_eg, x_eg, params, dydx_eg, None, None, batch_data_size, None, True, True)
grad_input_eg = torch.randn_like(x_eg, dtype=torch.float, device=device)
dL_ddLdoutput_eg, dL_dparams_eg, dL_dinput_eg = _backend.lod_bwd_bwd_input(lod_meta, grad_input_eg, grad_eg, x_eg, params, dydx_eg, None, None, batch_data_size, None, True, True, True)

# class LoTDFunctionOnlyDydx(torch.autograd.Function):
#     @staticmethod
#     def forward(
#         ctx,
#         # 0       1  2     3                4                   5                     6
#         lod_meta, x, grid, batch_inds=None, batch_offsets=None, batch_data_size=None, loss_scale=1.0, max_level=None):
        
#         ctx.set_materialize_grads(False)
#         prefix = x.shape[:-1]
        
#         # x = x.clamp(1.0e-6, 1-1.0e-6)
#         y, dy_dx = _backend.lod_fwd(lod_meta, x.flatten(0, -2), grid, None if batch_inds is None else batch_inds.flatten(0, -1), batch_offsets, batch_data_size, max_level)
        
#         if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
#             ctx.save_for_backward(x, grid, dy_dx, batch_inds, batch_offsets)
#             ctx.lod_meta = lod_meta
#             ctx.batch_data_size = batch_data_size
#             ctx.loss_scale = loss_scale
#             ctx.max_level = max_level
        
#         return y.unflatten(0, prefix)
    
#     @staticmethod
#     @once_differentiable
#     def backward(ctx, dL_dy):
#         """
#         from:   dL_dy
#         to:     dL_dx, ~~dL_dgrid~~
#         """
#         if dL_dy is None:
#             dL_dx = None
#         else:
#             x, grid, dy_dx, batch_inds, batch_offsets = ctx.saved_tensors
#             loss_scale = ctx.loss_scale
#             prefix = x.shape[:-1]
#             dL_dx, _ = _backend.lod_bwd(
#                 ctx.lod_meta, dL_dy.flatten(0, -2) * loss_scale, x.flatten(0, -2), grid, dy_dx, batch_inds, batch_offsets, ctx.batch_data_size, ctx.max_level, 
#                 # need_input_grad,       need_params_grad
#                 ctx.needs_input_grad[1], False
#             )
            
#             dL_dx = None if dL_dx is None else dL_dx.unflatten(0, prefix) / loss_scale
#             ctx.mark_non_differentiable(dL_dx)
#         # 0:lod_meta, 1:x,   2:grid, 3:batch_inds, 4:batch_offsets, 5:batch_data_size, 6:loss_scale, 7:max_level
#         return None,  dL_dx, None,   None,         None,            None,              None,         None

def apply_on_x(x):
    return LoTDFunction.apply(lod_meta, x, params, None, None, batch_data_size, 1.0, None)

# NOTE: passed
#       
#       y   w.r.t. x     i.e. dy_dx (passed)
autograd.gradcheck(
    apply_on_x, 
    x_eg.data.clone().requires_grad_(True),
    eps=1.0e-4, rtol=1.0e-2, atol=1.0e-3,)















def apply_on_grid(grid):
    return LoTDFunction.apply(lod_meta, x_eg, grid, None, None, batch_data_size, 1.0, None)

# NOTE: passed
#       
#       y   w.r.t. grid     i.e. dy_dgrid (passed)
autograd.gradcheck(
    apply_on_grid, 
    params,
    eps=1.0e-4, rtol=1.0e-2, atol=1.0e-3,)


def dydx_apply_on_dldy(grad):
    y, dydx = _backend.lod_fwd(lod_meta, x_eg, params, None, None, batch_data_size, None,  True)
    dldx = LoTDFunctionBwdDydx.apply(lod_meta, grad, x_eg, params, dydx,  None, None, batch_data_size, 1.0, None, None)
    return dldx

def dydx_apply_on_x(x):
    bds = int(math.prod(x.shape[1:-1]))
    y, dydx = _backend.lod_fwd(lod_meta, x, params, None, None, bds, None,  True)
    dldx = LoTDFunctionBwdDydx.apply(lod_meta, grad_eg, x, params, dydx,  None, None, bds, 1.0, None, None)
    return dldx

def dydx_apply_on_grid(grid):
    y, dydx = _backend.lod_fwd(lod_meta, x_eg, grid, None, None, batch_data_size, None, True)
    dldx = LoTDFunctionBwdDydx.apply(lod_meta, grad_eg, x_eg, grid, dydx,  None, None, batch_data_size, 1.0, None, None)
    return dldx

# NOTE: passed
#       dL_dx       w.r.t. dLdy     i.e. ddLdx_ddLdy (passed)
autograd.gradcheck(
    dydx_apply_on_dldy,
    grad_eg.clone().requires_grad_(True),
    eps=1.0e-4, rtol=1.0e-2, atol=1.0e-2,
    # nondet_tol=0.001 # due to non-determinism of atomicAdd
)

# # NOTE: disabled
# #       dL_dx       w.r.t. x     i.e. ddLdx_dx (passed)
# autograd.gradcheck(
#     dydx_apply_on_x,
#     x_eg.data..clone().requires_grad_(True),
#     eps=1.0e-3, atol=1.0e-3,
#     nondet_tol=0.001 # due to non-determinism of atomicAdd
# )

# NOTE: passed
#       dL_dx       w.r.t. grid     i.e. ddLdx_dgrid (passed)
autograd.gradcheck(
    dydx_apply_on_grid,
    params,
    eps=1.0e-4, rtol=1.0e-2, atol=1.0e-2,
    # nondet_tol=0.001 # due to non-determinism of atomicAdd
)

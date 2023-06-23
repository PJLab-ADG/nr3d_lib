import random
import numpy as np
from icecream import ic

import torch
import torch.nn as nn
from torch import autograd

from nr3d_lib.config import ConfigDict
from nr3d_lib.models.blocks import get_blocks
from nr3d_lib.models.embedders import get_embedder
from nr3d_lib.models.grids.lotd import LoTDEncoding

from torch.autograd.anomaly_mode import set_detect_anomaly
set_detect_anomaly(True)

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device('cuda')
dtype = torch.float16
# enc = LoTDEncoding(3, lotd_cfg=ConfigDict(lod_res=[4, 8], lod_n_feats=[2, 2], lod_types=[''Dense'', ''Dense'']), device=device, dtype=dtype)
enc = LoTDEncoding(
    3, lotd_cfg=ConfigDict(
        lod_res=[16,    23,    31,    43,    59,    81,   112,  154,  213,  295,  407,  562,  777,  1073, 1483, 2048], 
        lod_n_feats=[2,     2,     2,     2,     2,     2,    2,    2,    2,    2,    2,    2,    2,    2,    2,    2], 
        lod_types=['Dense', 'Dense', 'Dense', 'Dense', 'Dense', 'Hash', 'Hash', 'Hash', 'Hash', 'Hash', 'Hash', 'Hash', 'Hash', 'Hash', 'Hash', 'Hash'], 
        hashmap_size=524288, 
    ), 
    device=device, dtype=dtype)

with torch.no_grad():
    nn.init.uniform_(enc.flattened_params, -1.0e-4, 1.0e-4)

extra_pos_embed_fn, n_extra_embed = get_embedder(embed_cfg=ConfigDict(type='identity'))

dec = get_blocks(enc.out_features + n_extra_embed, 1, D=1, W=64, device=device, dtype=dtype)

x = torch.rand([4096 * 1024, 3], device=device, dtype=torch.float) * 2 - 1

def fun_sdf(x):
    h = enc.forward(x)
    h_embed = extra_pos_embed_fn(x)
    h_full = torch.cat([h, h_embed.to(h.dtype)], dim=-1)
    y = dec(h_full)
    return y[..., 0]

def numeric_nablas(x, eps=1.0e-3):
    x1 = x.clone(); x1[..., 0] += eps
    x2 = x.clone(); x2[..., 1] += eps
    x3 = x.clone(); x3[..., 2] += eps
    
    x1_ = x.clone(); x1_[..., 0] -= eps
    x2_ = x.clone(); x2_[..., 1] -= eps
    x3_ = x.clone(); x3_[..., 2] -= eps
    
    y1 = fun_sdf(x1)
    y2 = fun_sdf(x2)
    y3 = fun_sdf(x3)

    y1_ = fun_sdf(x1_)
    y2_ = fun_sdf(x2_)
    y3_ = fun_sdf(x3_)

    dy_dx1 = (y1-y1_) / (2.*eps)
    dy_dx2 = (y2-y2_) / (2.*eps)
    dy_dx3 = (y3-y3_) / (2.*eps)
    nablas = torch.stack([dy_dx1, dy_dx2, dy_dx3], dim=-1)
    return nablas

def fun1(x, opt=1, need_loss_backward_input = True):
    """
    only 1 autograd call
    """
    for n, p in enc.named_parameters():
        p.grad = None

    for n, p in dec.named_parameters():
        p.grad = None
    
    x = x.clone().requires_grad_(True)

    h, dy_dx = enc.forward_dydx(x, need_loss_backward_input=need_loss_backward_input)
    # NOTE: All is correct
    if opt == 1:
        h_embed = extra_pos_embed_fn(x).to(h.dtype)
        h_full = torch.cat([h, h_embed], dim=-1)
    elif opt == 2:
        h_embed = extra_pos_embed_fn(x)
        h_full = torch.cat([h, h_embed.to(h.dtype)], dim=-1)
    elif opt == 3:
        h_embed = extra_pos_embed_fn(x) * 1
        h_full = torch.cat([h, h_embed.to(h.dtype)], dim=-1)
    else:
        raise RuntimeError(f"Invalid opt={opt}")
    
    y = dec(h_full)

    # Decoder bwd_input; 
    dL_dh_full = autograd.grad(y, h_full, y.new_ones(y.shape), retain_graph=True, create_graph=True, only_inputs=True)[0]
    # Encoding bwd_input
    nablas = enc.backward_dydx(dL_dh_full[..., :h.shape[-1]], dy_dx, x)

    # Extra nablas from extra_embed stream.
    dL_dh_embed = dL_dh_full[..., h.shape[-1]:]
    if extra_pos_embed_fn._embedder_type == 'identity':
        nablas_extra = dL_dh_embed
    else:
        nablas_extra = autograd.grad(h_embed, x, dL_dh_embed, retain_graph=True, create_graph=True, only_inputs=True)[0]

    return nablas.float(), nablas_extra.float()

def fun2(x, opt: int = 1, need_loss_backward_input = True):
    """
    2 autograd call
    """
    for n, p in enc.named_parameters():
        p.grad = None

    for n, p in dec.named_parameters():
        p.grad = None
    
    x = x.clone().requires_grad_(True)

    h, dy_dx = enc.forward_dydx(x, need_loss_backward_input=need_loss_backward_input)
    # NOTE: If embed_fn is identity & the output has the same dtype as h, then h_embed and x are the same tensor handle address
    #       This will result in actually calculating autograd.grad(y, x, ...) when calling autograd.grad(y, h_embed, ...) later!
    #       This is also the reason for the extra invocation of LoTDFunctionFwdDydx's backward function
    # BUG:  opt==2 is proved to be wrong !
    if opt == 1:
        #---- Option1 # (Sometimes) Wrong nablas + wrong dL_grid !
        h_embed = extra_pos_embed_fn(x).to(h.dtype)
        h_full = torch.cat([h, h_embed], dim=-1)
    elif opt == 2:
        #---- Option2; # Wrong nablas (HUGE ERROR) ! + wrong dL_dgrid + Raise error because accidentally calls another 2nd-backward !
        h_embed = extra_pos_embed_fn(x)
        h_full = torch.cat([h, h_embed.to(h.dtype)], dim=-1)
    elif opt == 3:
        #---- Option3; # Correct
        h_embed = extra_pos_embed_fn(x) * 1
        h_full = torch.cat([h, h_embed.to(h.dtype)], dim=-1)
    else:
        raise RuntimeError(f"Invalid opt={opt}")

    y = dec(h_full)

    # Decoder bwd_input; 
    dL_dh = autograd.grad(y, h, y.new_ones(y.shape), retain_graph=True, create_graph=True, only_inputs=True, allow_unused=True)[0]
    # Encoding bwd_input
    nablas = enc.backward_dydx(dL_dh, dy_dx, x) if dL_dh is not None else 0

    # Extra nablas from extra_embed stream.
    dL_dh_embed = autograd.grad(y, h_embed, y.new_ones(y.shape), retain_graph=True, create_graph=True, only_inputs=True)[0]
    if extra_pos_embed_fn._embedder_type == 'identity':
        nablas_extra = dL_dh_embed
    else:
        nablas_extra = autograd.grad(h_embed, x, dL_dh_embed, retain_graph=True, create_graph=True, only_inputs=True)[0]
    
    return nablas.float(), nablas_extra.float()

nablas_fake = numeric_nablas(x)

print("Only one autograd opt1".center(40, '='))
nablas1_1, nablas_extra1_1 = fun1(x, opt=1, need_loss_backward_input=True)
(nablas1_1 + nablas_extra1_1).norm(dim=-1).mean().backward()
for n, p in enc.named_parameters():
    if p is not None and p.grad is not None:
        print(n, p.grad.data.abs().sum().item())
for n, p in dec.named_parameters():
    if p is not None and p.grad is not None:
        print(n, p.grad.data.abs().sum().item())
nablas1_1, nablas_extra1_1 = nablas1_1.data, nablas_extra1_1.data
nablas1_1_final = nablas1_1 + nablas_extra1_1

print("Only one autograd opt1 + no need loss input".center(40, '='))
nablas1_1_false, nablas_extra1_1_false = fun1(x, opt=1, need_loss_backward_input=False)
(nablas1_1_false + nablas_extra1_1_false).norm(dim=-1).mean().backward()
for n, p in enc.named_parameters():
    if p is not None and p.grad is not None:
        print(n, p.grad.data.abs().sum().item())
for n, p in dec.named_parameters():
    if p is not None and p.grad is not None:
        print(n, p.grad.data.abs().sum().item())
nablas1_1_false, nablas_extra1_1_false = nablas1_1_false.data, nablas_extra1_1_false.data
nablas1_1_false_final = nablas1_1_false + nablas_extra1_1_false
ic(torch.allclose(nablas1_1, nablas1_1_false))
ic(torch.allclose(nablas_extra1_1, nablas_extra1_1_false))

print("Only one autograd opt2".center(40, '='))
nablas1_2, nablas_extra1_2 = fun1(x, opt=2)
(nablas1_2 + nablas_extra1_2).norm(dim=-1).mean().backward()
for n, p in enc.named_parameters():
    if p is not None and p.grad is not None:
        print(n, p.grad.data.abs().sum().item())
for n, p in dec.named_parameters():
    if p is not None and p.grad is not None:
        print(n, p.grad.data.abs().sum().item())
nablas1_2, nablas_extra1_2 = nablas1_2.data, nablas_extra1_2.data
ic(torch.allclose(nablas1_1, nablas1_2))
ic(torch.allclose(nablas_extra1_1, nablas_extra1_2))

print("Only one autograd opt3".center(40, '='))
nablas1_3, nablas_extra1_3 = fun1(x, opt=3)
(nablas1_3 + nablas_extra1_3).norm(dim=-1).mean().backward()
for n, p in enc.named_parameters():
    if p is not None and p.grad is not None:
        print(n, p.grad.data.abs().sum().item())
for n, p in dec.named_parameters():
    if p is not None and p.grad is not None:
        print(n, p.grad.data.abs().sum().item())
nablas1_3, nablas_extra1_3 = nablas1_3.data, nablas_extra1_3.data
ic(torch.allclose(nablas1_1, nablas1_3))
ic(torch.allclose(nablas_extra1_1, nablas_extra1_3))

print("Two autograd opt1".center(40, '='))
nablas2_1, nablas_extra2_1 = fun2(x, opt=1)
(nablas2_1 + nablas_extra2_1).norm(dim=-1).mean().backward()
for n, p in enc.named_parameters():
    if p is not None and p.grad is not None:
        print(n, p.grad.data.abs().sum().item())
for n, p in dec.named_parameters():
    if p is not None and p.grad is not None:
        print(n, p.grad.data.abs().sum().item())
nablas2_1, nablas_extra2_1 = nablas2_1.data, nablas_extra2_1.data
ic(torch.allclose(nablas1_1, nablas2_1))
ic(torch.allclose(nablas1_2, nablas2_1))
ic(torch.allclose(nablas1_3, nablas2_1))
ic(torch.allclose(nablas_extra1_1, nablas_extra2_1))
ic(torch.allclose(nablas_extra1_2, nablas_extra2_1))
ic(torch.allclose(nablas_extra1_3, nablas_extra2_1))

print("Two autograd opt2".center(40, '='))
nablas2_2, nablas_extra2_2 = fun2(x, opt=2)
(nablas2_2 + nablas_extra2_2).norm(dim=-1).mean().backward()
for n, p in enc.named_parameters():
    if p is not None and p.grad is not None:
        print(n, p.grad.data.abs().sum().item())
for n, p in dec.named_parameters():
    if p is not None and p.grad is not None:
        print(n, p.grad.data.abs().sum().item())
nablas2_2, nablas_extra2_2 = nablas2_2.data, nablas_extra2_2.data
ic(torch.allclose(nablas1_1, nablas2_2))
ic(torch.allclose(nablas1_2, nablas2_2))
ic(torch.allclose(nablas1_3, nablas2_2))
ic(torch.allclose(nablas_extra1_1, nablas_extra2_2))
ic(torch.allclose(nablas_extra1_2, nablas_extra2_2))
ic(torch.allclose(nablas_extra1_3, nablas_extra2_2))

print("Two autograd opt3".center(40, '='))
nablas2_3, nablas_extra2_3 = fun2(x, opt=3)
(nablas2_3 + nablas_extra2_3).norm(dim=-1).mean().backward()
for n, p in enc.named_parameters():
    if p is not None and p.grad is not None:
        print(n, p.grad.data.abs().sum().item())
for n, p in dec.named_parameters():
    if p is not None and p.grad is not None:
        print(n, p.grad.data.abs().sum().item())
nablas2_3, nablas_extra2_3 = nablas2_3.data, nablas_extra2_3.data
ic(torch.allclose(nablas1_1, nablas2_3))
ic(torch.allclose(nablas1_2, nablas2_3))
ic(torch.allclose(nablas1_3, nablas2_3))
ic(torch.allclose(nablas_extra1_1, nablas_extra2_3))
ic(torch.allclose(nablas_extra1_2, nablas_extra2_3))
ic(torch.allclose(nablas_extra1_3, nablas_extra2_3))

from torch.utils.benchmark import Timer
# half: 80.44 ms;     float: 115.97 ms
print(Timer(stmt="fun1(x, opt=1)", globals={'fun1':fun1, 'x':x}).blocked_autorange())
# half: 81.15 ms;     float: 115.64 ms
print(Timer(stmt="fun1(x, opt=2)", globals={'fun1':fun1, 'x':x}).blocked_autorange())
# half: 81.26 ms;     float: 116.10 ms
print(Timer(stmt="fun1(x, opt=3)", globals={'fun1':fun1, 'x':x}).blocked_autorange())
# half: 96.65 ms;   float: 183.24 ms
print(Timer(stmt="fun2(x, opt=1)", globals={'fun2':fun2, 'x':x}).blocked_autorange())
# half: 122.7 ms & easily OOM;  float: 183.00 ms
print(Timer(stmt="fun2(x, opt=2)", globals={'fun2':fun2, 'x':x}).blocked_autorange())
# half: 97.88 ms    float: 138.80 ms
print(Timer(stmt="fun2(x, opt=3)", globals={'fun2':fun2, 'x':x}).blocked_autorange())
"""
@file   unit_test.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Unit test of pack_ops
"""

import torch

from nr3d_lib.render.raysample import *
from nr3d_lib.render.pack_ops import *

if __name__ == "__main__":
    #---- Pack ops
    def test_packed_sum(device=torch.device('cuda')):
        from kaolin.render.spc import sum_reduce
        from torch.utils.benchmark import Timer
        n_per_pack = torch.randint(32, 96, [4096], device=device)
        pack_infos = get_pack_infos_from_n(n_per_pack)
        num_feats = n_per_pack.sum().item()
        boundary = torch.zeros([num_feats], dtype=torch.bool, device=device)
        boundary[pack_infos[...,0]] = 1
        feats = torch.randn([num_feats,1], dtype=torch.float, device=device)
        
        y1 = packed_sum(feats, pack_infos)
        y2 = sum_reduce(feats, boundary)
        print(torch.allclose(y1, y2, atol=1.0e-5))

        y1 = packed_cumsum(feats, pack_infos, reverse=True)

        # 47.72 us
        print(Timer(
            stmt="sum_reduce(feats, boundary)",
            globals={'sum_reduce': sum_reduce, 'feats': feats, 'boundary':boundary}
        ).blocked_autorange())

        # 25.15 us; twice faster.
        print(Timer(
            stmt="packed_sum(feats, pack_infos)",
            globals={'packed_sum': packed_sum, 'feats': feats, 'pack_infos':pack_infos}
        ).blocked_autorange())

        # 23 us
        print(Timer(
            stmt="packed_cumsum(feats, pack_infos)",
            globals={'packed_cumsum': packed_cumsum, 'feats': feats, 'pack_infos':pack_infos}
        ).blocked_autorange())

        # 14 us
        print(Timer(
            stmt="packed_cumprod(feats, pack_infos)",
            globals={'packed_cumprod': packed_cumprod, 'feats': feats, 'pack_infos':pack_infos}
        ).blocked_autorange())

    def test_packed_diff(device=torch.device('cuda')):
        from kaolin import _C as _kaolin_backend
        from torch.utils.benchmark import Timer
        from icecream import ic
        def packed_diff_v1(feats: torch.Tensor, pack_infos: torch.LongTensor):
            delta = _kaolin_backend.render.spc.diff_cuda(feats, pack_infos[...,0].contiguous())
            return delta
        
        n_per_pack = torch.tensor([3,1,2,1], device=device, dtype=torch.long)
        pack_infos = get_pack_infos_from_n(n_per_pack)
        num_feats = n_per_pack.sum().item()
        feats = torch.randn([num_feats,1], dtype=torch.float, device=device)
        out = packed_diff(feats, pack_infos) # v2
        ic(out)
        
        n_per_pack = torch.randint(32, 96, [4096], device=device, dtype=torch.long)
        pack_infos = get_pack_infos_from_n(n_per_pack)
        num_feats = n_per_pack.sum().item()
        boundary = torch.zeros([num_feats], dtype=torch.bool, device=device); boundary[pack_infos[...,0]] = 1
        feats = torch.randn([num_feats,1], dtype=torch.float, device=device)
        
        y1 = packed_diff_v1(feats, pack_infos) 
        y2 = packed_diff(feats, pack_infos) # v2
        print(torch.allclose(y1, y2))

        # 28.02 us
        print(Timer(
            stmt="packed_diff_v1(feats, pack_infos)",
            globals={'packed_diff_v1': packed_diff_v1, 'feats': feats, 'pack_infos':pack_infos}
        ).blocked_autorange())

        # 128 threads: 12.56 us
        print(Timer(
            stmt="packed_diff(feats, pack_infos)",
            globals={'packed_diff': packed_diff, 'feats': feats, 'pack_infos':pack_infos}
        ).blocked_autorange())
        
        #------------------------- Test grad: all pass.
        feats = torch.rand([3,7], device=device, dtype=torch.float, requires_grad=True)
        first_inds = torch.arange(3, device=device, dtype=torch.long) * 7
        pack_infos = get_pack_infos_from_first(first_inds, feats.numel())
        diff1 = torch.cat([feats.diff(dim=-1), feats.new_zeros([3,1])], dim=-1)
        diff2 = packed_diff(feats.view(-1,1), pack_infos).view(3,7)
        grad = torch.rand_like(diff1, dtype=torch.float, device=device)
        grad1 = torch.autograd.grad(diff1, feats, grad, retain_graph=True, create_graph=False)[0]
        grad2 = torch.autograd.grad(diff2, feats, grad, retain_graph=True, create_graph=False)[0]
        print(torch.allclose(grad1, grad2))
        del feats, diff1, diff2
        
        feats = torch.rand([3,7], device=device, dtype=torch.float, requires_grad=True)
        first_inds = torch.arange(3, device=device, dtype=torch.long) * 7
        pack_infos = get_pack_infos_from_first(first_inds, feats.numel())
        append = torch.rand([3, 1], dtype=torch.float, device=device, requires_grad=True)
        diff1 = feats.diff(dim=-1, append=append)
        diff2 = packed_diff(feats.view(-1,1), pack_infos, pack_appends=append).view(3,7)
        print(torch.allclose(diff1.data, diff2.data))
        grad = torch.rand_like(diff1, dtype=torch.float, device=device)
        grad1 = torch.autograd.grad(diff1, feats, grad, retain_graph=True, create_graph=False)[0]
        grad1_1 = torch.autograd.grad(diff1, append, grad, retain_graph=True, create_graph=False)[0]
        grad2 = torch.autograd.grad(diff2, feats, grad, retain_graph=True, create_graph=False)[0]
        grad2_1 = torch.autograd.grad(diff2, append, grad, retain_graph=True, create_graph=False)[0]
        print(torch.allclose(grad1, grad2))
        print(torch.allclose(grad1_1, grad2_1))
        del feats, diff1, diff2
        
        feats = torch.rand([3,7], device=device, dtype=torch.float, requires_grad=True)
        first_inds = torch.arange(3, device=device, dtype=torch.long) * 7
        pack_infos = get_pack_infos_from_first(first_inds, feats.numel())
        last_fills = torch.rand([3, 1], dtype=torch.float, device=device, requires_grad=True)
        diff1 = torch.cat([feats.diff(dim=-1), last_fills], dim=-1)
        diff2 = packed_diff(feats.view(-1,1), pack_infos, pack_last_fill=last_fills).view(3,7)
        print(torch.allclose(diff1.data, diff2.data))
        grad = torch.rand_like(diff1, dtype=torch.float, device=device)
        grad1 = torch.autograd.grad(diff1, feats, grad, retain_graph=True, create_graph=False)[0]
        grad1_1 = torch.autograd.grad(diff1, last_fills, grad, retain_graph=True, create_graph=False)[0]
        grad2 = torch.autograd.grad(diff2, feats, grad, retain_graph=True, create_graph=False)[0]
        grad2_1 = torch.autograd.grad(diff2, last_fills, grad, retain_graph=True, create_graph=False)[0]
        print(torch.allclose(grad1, grad2))
        print(torch.allclose(grad1_1, grad2_1))

    def test_backward_diff(device=torch.device('cuda')):
        from kaolin import _C as _kaolin_backend
        from torch.utils.benchmark import Timer
        from icecream import ic
        n_per_pack = torch.tensor([3,1,2,1], device=device, dtype=torch.long)
        pack_infos = get_pack_infos_from_n(n_per_pack)
        num_feats = n_per_pack.sum().item()
        feats = torch.randn([num_feats,1], dtype=torch.float, device=device)
        out = packed_backward_diff(feats, pack_infos) # v2
        ic(out)
        
        #------------------------- Test grad: all pass.
        feats = torch.rand([3,7], device=device, dtype=torch.float, requires_grad=True)
        first_inds = torch.arange(3, device=device, dtype=torch.long) * 7
        pack_infos = get_pack_infos_from_first(first_inds, feats.numel())
        diff1 = torch.cat([feats.new_zeros([3,1]), feats.diff(dim=-1)], dim=-1)
        diff2 = packed_backward_diff(feats.view(-1,1), pack_infos).view(3,7)
        grad = torch.rand_like(diff1, dtype=torch.float, device=device)
        grad1 = torch.autograd.grad(diff1, feats, grad, retain_graph=True, create_graph=False)[0]
        grad2 = torch.autograd.grad(diff2, feats, grad, retain_graph=True, create_graph=False)[0]
        print(torch.allclose(grad1, grad2))
        del feats, diff1, diff2
        
        feats = torch.rand([3,7], device=device, dtype=torch.float, requires_grad=True)
        first_inds = torch.arange(3, device=device, dtype=torch.long) * 7
        pack_infos = get_pack_infos_from_first(first_inds, feats.numel())
        prepend = torch.rand([3, 1], dtype=torch.float, device=device, requires_grad=True)
        diff1 = feats.diff(dim=-1, prepend=prepend)
        diff2 = packed_backward_diff(feats.view(-1,1), pack_infos, pack_prepends=prepend).view(3,7)
        print(torch.allclose(diff1.data, diff2.data))
        grad = torch.rand_like(diff1, dtype=torch.float, device=device)
        grad1 = torch.autograd.grad(diff1, feats, grad, retain_graph=True, create_graph=False)[0]
        grad1_1 = torch.autograd.grad(diff1, prepend, grad, retain_graph=True, create_graph=False)[0]
        grad2 = torch.autograd.grad(diff2, feats, grad, retain_graph=True, create_graph=False)[0]
        grad2_1 = torch.autograd.grad(diff2, prepend, grad, retain_graph=True, create_graph=False)[0]
        print(torch.allclose(grad1, grad2))
        print(torch.allclose(grad1_1, grad2_1))
        del feats, diff1, diff2
        
        feats = torch.rand([3,7], device=device, dtype=torch.float, requires_grad=True)
        first_inds = torch.arange(3, device=device, dtype=torch.long) * 7
        pack_infos = get_pack_infos_from_first(first_inds, feats.numel())
        first_fills = torch.rand([3, 1], dtype=torch.float, device=device, requires_grad=True)
        diff1 = torch.cat([first_fills, feats.diff(dim=-1)], dim=-1)
        diff2 = packed_backward_diff(feats.view(-1,1), pack_infos, pack_first_fill=first_fills).view(3,7)
        print(torch.allclose(diff1.data, diff2.data))
        grad = torch.rand_like(diff1, dtype=torch.float, device=device)
        grad1 = torch.autograd.grad(diff1, feats, grad, retain_graph=True, create_graph=False)[0]
        grad1_1 = torch.autograd.grad(diff1, first_fills, grad, retain_graph=True, create_graph=False)[0]
        grad2 = torch.autograd.grad(diff2, feats, grad, retain_graph=True, create_graph=False)[0]
        grad2_1 = torch.autograd.grad(diff2, first_fills, grad, retain_graph=True, create_graph=False)[0]
        print(torch.allclose(grad1, grad2))
        print(torch.allclose(grad1_1, grad2_1))

    def test_packed_binary_ops_arithmetic(device=torch.device('cuda')):
        from torch.utils.benchmark import Timer
        n_per_pack = torch.randint(32, 96, [4096], device=device)
        pack_infos = get_pack_infos_from_n(n_per_pack)
        feats = torch.randn([pack_infos[-1].sum().item()], device=device, requires_grad=True)
        other = torch.randn([4096], device=device)
        other[other.data.abs() < 1.0e-2] = 1.0
        other.requires_grad_(True)
        
        y_add_1 = packed_add(feats, other, pack_infos)
        y_sub_1 = packed_sub(feats, other, pack_infos)
        y_mul_1 = packed_mul(feats, other, pack_infos)
        y_div_1 = packed_div(feats, other, pack_infos)
        
        y_add_2 = feats + torch.repeat_interleave(other, pack_infos[:,1], dim=0)
        y_sub_2 = feats - torch.repeat_interleave(other, pack_infos[:,1], dim=0)
        y_mul_2 = feats * torch.repeat_interleave(other, pack_infos[:,1], dim=0)
        y_div_2 = feats / torch.repeat_interleave(other, pack_infos[:,1], dim=0)
        
        print(torch.allclose(y_add_1, y_add_2))
        print(torch.allclose(y_sub_1, y_sub_2))
        print(torch.allclose(y_mul_1, y_mul_2))
        print(torch.allclose(y_div_1, y_div_2))
        
        grad = torch.randn(feats.shape, device=device)
        
        y_grad_add_11 = torch.autograd.grad(y_add_1, feats, grad, retain_graph=True, create_graph=False, only_inputs=True)[0]
        y_grad_add_12 = torch.autograd.grad(y_add_2, feats, grad, retain_graph=True, create_graph=False, only_inputs=True)[0]
        y_grad_add_21 = torch.autograd.grad(y_add_1, other, grad, retain_graph=True, create_graph=False, only_inputs=True)[0]
        y_grad_add_22 = torch.autograd.grad(y_add_2, other, grad, retain_graph=True, create_graph=False, only_inputs=True)[0]
        
        y_grad_sub_11 = torch.autograd.grad(y_sub_1, feats, grad, retain_graph=True, create_graph=False, only_inputs=True)[0]
        y_grad_sub_12 = torch.autograd.grad(y_sub_2, feats, grad, retain_graph=True, create_graph=False, only_inputs=True)[0]
        y_grad_sub_21 = torch.autograd.grad(y_sub_1, other, grad, retain_graph=True, create_graph=False, only_inputs=True)[0]
        y_grad_sub_22 = torch.autograd.grad(y_sub_2, other, grad, retain_graph=True, create_graph=False, only_inputs=True)[0]

        y_grad_mul_11 = torch.autograd.grad(y_mul_1, feats, grad, retain_graph=True, create_graph=False, only_inputs=True)[0]
        y_grad_mul_12 = torch.autograd.grad(y_mul_2, feats, grad, retain_graph=True, create_graph=False, only_inputs=True)[0]
        y_grad_mul_21 = torch.autograd.grad(y_mul_1, other, grad, retain_graph=True, create_graph=False, only_inputs=True)[0]
        y_grad_mul_22 = torch.autograd.grad(y_mul_2, other, grad, retain_graph=True, create_graph=False, only_inputs=True)[0]

        y_grad_div_11 = torch.autograd.grad(y_div_1, feats, grad, retain_graph=True, create_graph=False, only_inputs=True)[0]
        y_grad_div_12 = torch.autograd.grad(y_div_2, feats, grad, retain_graph=True, create_graph=False, only_inputs=True)[0]
        y_grad_div_21 = torch.autograd.grad(y_div_1, other, grad, retain_graph=True, create_graph=False, only_inputs=True)[0]
        y_grad_div_22 = torch.autograd.grad(y_div_2, other, grad, retain_graph=True, create_graph=False, only_inputs=True)[0]
        
        print(torch.allclose(y_grad_add_11, y_grad_add_12, atol=1e-5))
        print(torch.allclose(y_grad_add_21, y_grad_add_22, atol=1e-5))
        print(torch.allclose(y_grad_sub_11, y_grad_sub_12, atol=1e-5))
        print(torch.allclose(y_grad_sub_21, y_grad_sub_22, atol=1e-5))
        print(torch.allclose(y_grad_mul_11, y_grad_mul_12, atol=1e-5))
        print(torch.allclose(y_grad_mul_21, y_grad_mul_22, atol=1e-5))
        print(torch.allclose(y_grad_div_11, y_grad_div_12, atol=1e-5))
        print(torch.allclose(y_grad_div_21, y_grad_div_22, atol=0, rtol=1.0e-2)) # ∣input−other∣ ≤ atol+rtol×∣other∣

        #---- Addition
        # 22.69 us
        print(Timer(
            stmt="packed_add(feats, other, pack_infos)", 
            globals={'packed_add':packed_add, 'feats':feats, 'other':other, 'pack_infos':pack_infos}
        ).blocked_autorange())

        # 75.34 us # NOTE: repeat_interleave is slow
        print(Timer(
            stmt="feats + torch.repeat_interleave(other, pack_infos[:,1], dim=0)", 
            globals={'feats':feats, 'other':other, 'pack_infos':pack_infos}
        ).blocked_autorange())

        #---- Subtraction
        # 22.72 us
        print(Timer(
            stmt="packed_sub(feats, other, pack_infos)", 
            globals={'packed_sub':packed_sub, 'feats':feats, 'other':other, 'pack_infos':pack_infos}
        ).blocked_autorange())

        # 74.69 us # NOTE: repeat_interleave is slow
        print(Timer(
            stmt="feats - torch.repeat_interleave(other, pack_infos[:,1], dim=0)", 
            globals={'feats':feats, 'other':other, 'pack_infos':pack_infos}
        ).blocked_autorange())

        #---- Multiplication
        # 22.70 us
        print(Timer(
            stmt="packed_mul(feats, other, pack_infos)", 
            globals={'packed_mul':packed_mul, 'feats':feats, 'other':other, 'pack_infos':pack_infos}
        ).blocked_autorange())

        # 77.28 us # NOTE: repeat_interleave is slow
        print(Timer(
            stmt="feats * torch.repeat_interleave(other, pack_infos[:,1], dim=0)", 
            globals={'feats':feats, 'other':other, 'pack_infos':pack_infos}
        ).blocked_autorange())

        #---- Division
        # 25.09 us
        print(Timer(
            stmt="packed_div(feats, other, pack_infos)", 
            globals={'packed_div':packed_div, 'feats':feats, 'other':other, 'pack_infos':pack_infos}
        ).blocked_autorange())

        # 74.91 us # NOTE: repeat_interleave is slow
        print(Timer(
            stmt="feats / torch.repeat_interleave(other, pack_infos[:,1], dim=0)", 
            globals={'feats':feats, 'other':other, 'pack_infos':pack_infos}
        ).blocked_autorange())

    def test_packed_binary_ops_compare(device=torch.device('cuda')):
        from torch.utils.benchmark import Timer
        n_per_pack = torch.randint(32, 96, [4096], device=device)
        pack_infos = get_pack_infos_from_n(n_per_pack)
        feats = torch.randn([pack_infos[-1].sum().item()], device=device, requires_grad=True)
        other = torch.randn([4096], device=device, requires_grad=True)
        
        y_gt_1 = packed_gt(feats, other, pack_infos)
        y_geq_1 = packed_geq(feats, other, pack_infos)
        y_lt_1 = packed_lt(feats, other, pack_infos)
        y_leq_1 = packed_leq(feats, other, pack_infos)
        y_eq_1 = packed_eq(feats, other, pack_infos)
        y_neq_1 = packed_neq(feats, other, pack_infos)
        
        y_gt_2 = feats > torch.repeat_interleave(other, pack_infos[:,1], dim=0)
        y_geq_2 = feats >= torch.repeat_interleave(other, pack_infos[:,1], dim=0)
        y_lt_2 = feats < torch.repeat_interleave(other, pack_infos[:,1], dim=0)
        y_leq_2 = feats <= torch.repeat_interleave(other, pack_infos[:,1], dim=0)
        y_eq_2 = feats == torch.repeat_interleave(other, pack_infos[:,1], dim=0)
        y_neq_2 = feats != torch.repeat_interleave(other, pack_infos[:,1], dim=0)
        
        print(torch.equal(y_gt_1, y_gt_2))
        print(torch.equal(y_geq_1, y_geq_2))
        print(torch.equal(y_lt_1, y_lt_2))
        print(torch.equal(y_leq_1, y_leq_2))
        print(torch.equal(y_eq_1, y_eq_2))
        print(torch.equal(y_neq_1, y_neq_2))
        
        #---- Gt
        print(Timer(
            stmt="packed_gt(feats, other, pack_infos)", 
            globals={'packed_gt':packed_gt, 'feats':feats, 'other':other, 'pack_infos':pack_infos}
        ).blocked_autorange())

        print(Timer(
            stmt="feats > torch.repeat_interleave(other, pack_infos[:,1], dim=0)", 
            globals={'feats':feats, 'other':other, 'pack_infos':pack_infos}
        ).blocked_autorange())

        #---- Geq
        print(Timer(
            stmt="packed_geq(feats, other, pack_infos)", 
            globals={'packed_geq':packed_geq, 'feats':feats, 'other':other, 'pack_infos':pack_infos}
        ).blocked_autorange())

        print(Timer(
            stmt="feats >= torch.repeat_interleave(other, pack_infos[:,1], dim=0)", 
            globals={'feats':feats, 'other':other, 'pack_infos':pack_infos}
        ).blocked_autorange())

        #---- Lt
        print(Timer(
            stmt="packed_lt(feats, other, pack_infos)", 
            globals={'packed_lt':packed_lt, 'feats':feats, 'other':other, 'pack_infos':pack_infos}
        ).blocked_autorange())

        print(Timer(
            stmt="feats < torch.repeat_interleave(other, pack_infos[:,1], dim=0)", 
            globals={'feats':feats, 'other':other, 'pack_infos':pack_infos}
        ).blocked_autorange())

        #---- Leq
        print(Timer(
            stmt="packed_leq(feats, other, pack_infos)", 
            globals={'packed_leq':packed_leq, 'feats':feats, 'other':other, 'pack_infos':pack_infos}
        ).blocked_autorange())

        print(Timer(
            stmt="feats <= torch.repeat_interleave(other, pack_infos[:,1], dim=0)", 
            globals={'feats':feats, 'other':other, 'pack_infos':pack_infos}
        ).blocked_autorange())

        #---- Eq
        print(Timer(
            stmt="packed_eq(feats, other, pack_infos)", 
            globals={'packed_eq':packed_eq, 'feats':feats, 'other':other, 'pack_infos':pack_infos}
        ).blocked_autorange())

        print(Timer(
            stmt="feats == torch.repeat_interleave(other, pack_infos[:,1], dim=0)", 
            globals={'feats':feats, 'other':other, 'pack_infos':pack_infos}
        ).blocked_autorange())

        #---- Neq
        print(Timer(
            stmt="packed_neq(feats, other, pack_infos)", 
            globals={'packed_neq':packed_neq, 'feats':feats, 'other':other, 'pack_infos':pack_infos}
        ).blocked_autorange())

        print(Timer(
            stmt="feats != torch.repeat_interleave(other, pack_infos[:,1], dim=0)", 
            globals={'feats':feats, 'other':other, 'pack_infos':pack_infos}
        ).blocked_autorange())

    def test_packed_matmul(device=torch.device('cuda')):
        from torch.utils.benchmark import Timer
        import nr3d_lib_bindings._pack_ops as _backend
        
        feats = torch.tensor([[2., 4.]], device=device, dtype=torch.float) # [1, 2]
        other = torch.tensor([[[1.5, 1.5]]], device=device, dtype=torch.float) # [1, 1, 2]
        pack_infos = torch.tensor([[0, 1]], device=device, dtype=torch.long)
        y_prod = _backend.packed_matmul(feats, other, pack_infos) # [1, 1]
        
        num_out_dim = 7
        num_packs = 4096
        n_per_pack = torch.randint(32, 96, [num_packs], device=device)
        pack_infos = get_pack_infos_from_n(n_per_pack)
        feats = torch.randn([pack_infos[-1].sum().item(), 3], device=device, requires_grad=True) # [num_feats, feat_dim]
        other = torch.randn([num_packs, num_out_dim, 3], device=device, requires_grad=True) # [num_packs, out_feat_dim, feat_dim]
        y_prod_1 = packed_matmul(feats, other, pack_infos)
        y_prod_2 = (feats.unsqueeze(-2) * torch.repeat_interleave(other, pack_infos[:,1], dim=0)).sum(-1)
        
        print(torch.allclose(y_prod_1, y_prod_2))
        
        # 512.4 us
        print(Timer(
            stmt="_backend.packed_matmul(feats, other, pack_infos)", 
            globals={'_backend':_backend, 'feats':feats, 'other':other, 'pack_infos':pack_infos}
        ).blocked_autorange())

        # 216.9 us # repeat interleave is faster
        print(Timer(
            stmt="(feats.unsqueeze(-2) * torch.repeat_interleave(other, pack_infos[:,1], dim=0)).sum(-1)", 
            globals={'feats':feats, 'other':other, 'pack_infos':pack_infos}
        ).blocked_autorange())

    def test_n_per_pack_t(device=torch.device('cuda')):
        from kaolin import _C as _kaolin_backend
        from torch.utils.benchmark import Timer
        ridx = torch.randint(0, 4096, [4096], device=device).repeat_interleave(torch.randint(32,96, [4096], device=device))
        boundary = mark_pack_boundaries(ridx)
        
        n1 = torch.diff(torch.nonzero(boundary)[..., 0], 1, append=torch.tensor([boundary.shape[0]],device=boundary.device, dtype=torch.long))
        n2 = torch.unique_consecutive(_kaolin_backend.render.spc.inclusive_sum_cuda(boundary.int()), return_counts=True)[1]
        n3 = torch.unique_consecutive(ridx, return_counts=True)[1]
        print(torch.allclose(n1,n2))
        print(torch.allclose(n1,n3))
        
        # 66.61 us
        print(Timer(
            stmt="torch.diff(torch.nonzero(boundary)[..., 0], 1, append=torch.tensor([boundary.shape[0]],device=boundary.device, dtype=torch.long))",
            globals={'_kaolin_backend': _kaolin_backend, 'boundary': boundary}
        ).blocked_autorange())

        # 58.28 us
        print(Timer(
            stmt="torch.unique_consecutive(_kaolin_backend.render.spc.inclusive_sum_cuda(boundary.int()), return_counts=True)[1]",
            globals={'_kaolin_backend': _kaolin_backend, 'boundary': boundary}
        ).blocked_autorange())
        
        # 32.48 us
        print(Timer(
            stmt="torch.unique_consecutive(ridx, return_counts=True)[1]",
            globals={'ridx': ridx}
        ).blocked_autorange())

    def test_pack_infos_t(device=torch.device('cuda')):
        from kaolin import _C as _kaolin_backend
        from torch.utils.benchmark import Timer
        ridx = torch.randint(0, 4096, [4096], device=device).repeat_interleave(torch.randint(32,96, [4096], device=device))
        boundary = mark_pack_boundaries(ridx)
        # 79.25 us for `boundray` sizeof 256k with 4096 trues
        print(Timer(
            stmt="torch.cumsum(torch.unique_consecutive(_kaolin_backend.render.spc.inclusive_sum_cuda(boundary.int()), return_counts=True)[1],0)-1",
            globals={'_kaolin_backend': _kaolin_backend, 'boundary': boundary}
        ).blocked_autorange())

        # 28.36 us for `boundray` sizeof 256k with 4096 trues
        print(Timer(
            stmt="torch.nonzero(boundary)[..., 0]",
            globals={'boundary': boundary}
        ).blocked_autorange())

        # 19.22 us, `boundray` sizeof 256k with 4096 trues
        print(Timer(
            stmt="_kaolin_backend.render.spc.inclusive_sum_cuda(boundary.int())",
            globals={'_kaolin_backend': _kaolin_backend, 'boundary': boundary}
        ).blocked_autorange())

        # 32.53 us, `inclusive_sum` sizeof 256k of int
        print(Timer(
            stmt="torch.unique_consecutive(inclusive_sum, return_counts=True)[1]",
            globals={'inclusive_sum': _kaolin_backend.render.spc.inclusive_sum_cuda(boundary.int())}
        ).blocked_autorange())

        #------------------- get last inds
        # 13.88 us, `cnt` sizeof 4096 of long
        cnt = torch.unique_consecutive(_kaolin_backend.render.spc.inclusive_sum_cuda(boundary.int()), return_counts=True)[1]
        print(Timer(
            stmt="torch.cumsum(cnt,0)-1",
            globals={'cnt': cnt}
        ).blocked_autorange())

        #------------------- get pack indices (starting inds)
        cnt = torch.unique_consecutive(_kaolin_backend.render.spc.inclusive_sum_cuda(boundary.int()), return_counts=True)[1]
        first_inds1 = cnt.cumsum(0)-cnt
        first_inds2 = cnt.cumsum(0).roll(1); first_inds2[0]=0
        print(torch.allclose(first_inds1, first_inds2))
        # 13.36 us
        print(Timer(
            stmt="torch.cumsum(cnt,0)-cnt",
            globals={'cnt': cnt}
        ).blocked_autorange())
        # 25.05 us
        print(Timer(
            stmt="y=torch.cumsum(cnt,0).roll(1); y[0]=0",
            globals={'cnt': cnt}
        ).blocked_autorange())
        # def inclusive_sum(boundary: torch.Tensor):
        #     return _kaolin_backend.render.spc.inclusive_sum_cuda(boundary.int()).contiguous()

    def test_expand_pack_boundary(device=torch.device('cuda')):
        from kaolin import _C as _kaolin_backend
        from torch.utils.benchmark import Timer
        ridx = torch.randint(0, 4096, [4096], device=device).repeat_interleave(torch.randint(8,24, [4096], device=device))
        boundary = mark_pack_boundaries(ridx)
        # def expand_pack_boundary(pack_boundary: torch.Tensor, num_samples: int):
        #     """Expands the pack boundaries according to the number of samples.

        #     Args:
        #         pack_boundary (torch.BoolTensor): pack boundaries [N]
        #         num_samples (int): Number of samples

        #     Returns:
        #         (torch.BoolTensor): pack boundaries of shape [N*num_samples]
        #     """
            
        #     bigpack_boundary = torch.zeros(pack_boundary.shape[0]*num_samples, device=pack_boundary.device, dtype=torch.bool)
        #     bigpack_boundary[pack_boundary.nonzero().long() * num_samples] = True
        #     bigpack_boundary = bigpack_boundary.int()
        #     return bigpack_boundary
        
        # 70.39 us
        print(Timer(
            stmt="expand_pack_boundary(boundary, 4)",
            globals={'expand_pack_boundary': expand_pack_boundary, 'boundary': boundary}
        ).blocked_autorange())

    def test_exclusive(device=torch.device('cuda')):
        from kaolin.render.spc import cumsum, cumprod
        from icecream import ic
        feat = torch.rand([13,1], device=device)
        boundary = torch.tensor([1,0,0,0,1,0,0,0,0,0,0,1,0], dtype=torch.bool, device=device)
        y = cumsum(feat, boundary)
        y_exclusive = cumsum(feat, boundary, exclusive=True)
        ic(feat.squeeze())
        ic(y.squeeze())
        ic(y_exclusive.squeeze())

        """
        `cumsum` + `exclusive=True` = cumsum that is right-shifted by one position, with the first position filled with 0
        The `exclusive` variant of `cumsum` is actually just `cumsum` shifted by one position, using 0 for padding.
        
        tensor([0.8750, 0.0581, 0.9378, 0.9638,    0.9859, 0.4652, 0.9105, 0.5071, 0.0173, 0.6071, 0.7123,    0.7371, 0.8094], device='cuda:0')
        tensor([0.8750, 0.9331, 1.8709, 2.8347,    0.9859, 1.4512, 2.3617, 2.8688, 2.8860, 3.4931, 4.2054,    0.7371, 1.5465], device='cuda:0')
        tensor([0.0000, 0.8750, 0.9331, 1.8709,    0.0000, 0.9859, 1.4512, 2.3617, 2.8688, 2.8860, 3.4931,    0.0000, 0.7371], device='cuda:0')
        """
        
        feat = torch.randn([13,1], device=device)
        y = cumprod(feat, boundary)
        y_exclusive = cumprod(feat, boundary, exclusive=True)
        ic(feat.squeeze())
        ic(y.squeeze())
        ic(y_exclusive.squeeze())
        """
        `cumprod` + `exclusive=True` = cumprod that is right-shifted by one position, with the first position filled with 1
        tensor([-0.8033,  1.2413,  0.1971,  1.2183,     1.3434,  1.7485,  0.0624, -0.3419, -0.1997, -1.4790,  1.1720,     0.1686, -0.0704], device='cuda:0')
        tensor([-0.8033, -0.9972, -0.1965, -0.2394,     1.3434,  2.3489,  0.1465, -0.0501,  0.0100, -0.0148, -0.0173,     0.1686, -0.0119], device='cuda:0')
        tensor([ 1.0000, -0.8033, -0.9972, -0.1965,     1.0000,  1.3434,  2.3489,  0.1465, -0.0501,  0.0100, -0.0148,     1.0000,  0.1686], device='cuda:0')
        """

    def test_ops_t(device=torch.device('cuda')):
        from kaolin import _C as _kaolin_backend
        ridx = torch.randint(0, 4096, [4096], device=device).repeat_interleave(torch.randint(32,96, [4096], device=device))
        boundary = mark_pack_boundaries(ridx)
        _kaolin_backend.render.spc.inclusive_sum_cuda(boundary.int())

    def test_mask_ridx(device=torch.device('cuda')):
        from torch.utils.benchmark import Timer
        from nr3d_lib.render.utils import create_ray_id_cuda
        
        # [256k input]
        mask = torch.rand([4096, 64], device=device) > 0.5
        ridx0 = create_ray_id_cuda(4096).unsqueeze(-1).tile(1,64)[mask]
        ridx1 = torch.arange(4096, device=device).unsqueeze(-1).tile(1,64)[mask]
        ridx2 = torch.nonzero(mask).long()[..., 0]
        
        print(torch.allclose(ridx0, ridx1))
        print(torch.allclose(ridx1, ridx2))
        
        # 57.74 us  
        print(Timer(
            stmt="torch.arange(4096, device=device).unsqueeze(-1).tile(1,64)[mask]",
            globals={'mask': mask, 'device':device}
        ).blocked_autorange())
        
        # 29.13 us
        print(Timer(
            stmt="torch.nonzero(mask).long()[..., 0]",
            globals={'mask': mask}
        ).blocked_autorange())

        # 46.41 us
        print(Timer(
            stmt="create_ray_id_cuda(4096).unsqueeze(-1).tile(1,64)[mask]",
            globals={'mask': mask, 'device':device, 'create_ray_id_cuda': create_ray_id_cuda}
        ).blocked_autorange())

    def test_ridx_hit_t(device=torch.device('cuda')):
        from torch.utils.benchmark import Timer
        ridx_hit = torch.arange(4096, device=device)
        num_per_hit = torch.randint(32,96, [4096], device=device)
        ridx = ridx_hit.repeat_interleave(num_per_hit)
        
        feats = torch.randn([4096, 3], device=device)
        
        y1 = feats[ridx]
        y2 = feats[ridx_hit].repeat_interleave(num_per_hit, dim=0)
        print(torch.allclose(y1,y2))
        
        # 7.8 us
        print(Timer(
            stmt="feats[ridx]",
            globals={'feats': feats, 'ridx':ridx}
        ).blocked_autorange())

        # 60.39 us
        print(Timer(
            stmt="feats[ridx_hit].repeat_interleave(num_per_hit, dim=0)",
            globals={'feats': feats, 'ridx_hit':ridx_hit, 'num_per_hit':num_per_hit}
        ).blocked_autorange())

    def test_int_repeat(device=torch.device('cuda')):
        from torch.utils.benchmark import Timer
        ridx = torch.randint(0, 4096, [4096], device=device).repeat_interleave(torch.randint(8,24, [4096], device=device))
        
        y1 = ridx.view(-1,1).tile(1,4).view(-1).contiguous()
        y2 = ridx.repeat_interleave(4)
        print(torch.allclose(y1,y2))
        
        # 11 us
        print(Timer(
            stmt="ridx.view(-1,1).tile(1,4).view(-1).contiguous()",
            globals={'ridx':ridx}
        ).blocked_autorange())

        # 80 us
        print(Timer(
            stmt="ridx.repeat_interleave(4)",
            globals={'ridx':ridx}
        ).blocked_autorange())

    def test_comma(device=torch.device('cuda')):
        from torch.utils.benchmark import Timer
        ridx = torch.randint(0, 4096, [4096], device=device).repeat_interleave(torch.randint(32,96, [4096], device=device))
        feat1 = torch.randn([4096, 3], device=device, dtype=torch.float)
        feat2 = torch.randn([4096, 1], device=device, dtype=torch.float)
        feat3 = torch.randn([4096, 2], device=device, dtype=torch.float)
        feat4 = torch.cat([feat1, feat2, feat3], dim=-1)
        
        # 20.73 us
        print(Timer(
            stmt="y1=feat1[ridx]\ny2=feat2[ridx]\ny3=feat3[ridx]",
            globals={'ridx':ridx, "feat1":feat1, "feat2":feat2, "feat3":feat3}
        ).blocked_autorange())

        # only 0.5 us faster
        # 20.27 us
        print(Timer(
            stmt="y1,y2,y3=feat1[ridx],feat2[ridx],feat3[ridx]",
            globals={'ridx':ridx, "feat1":feat1, "feat2":feat2, "feat3":feat3}
        ).blocked_autorange())
        
        # 16.93 us
        print(Timer(
            stmt="y4=feat4[ridx]",
            globals={'ridx':ridx, "feat4":feat4}
        ).blocked_autorange())

    def test_interleave_arange(device=torch.device('cuda')):
        from torch.utils.benchmark import Timer
        from icecream import ic
        # @torch.jit.script
        def interleave_arange_pt(n_per_pack: torch.Tensor):
            assert n_per_pack.dim() == 1, 'Only works for 1D Tensors.'
            cumsum = n_per_pack.cumsum(0)
            # pack_inds = cumsum.roll(1); pack_inds[0] = 0
            pack_inds = cumsum-n_per_pack
            # BUG in pytorch-1.11. Fixed in pytorch-1.12
            # https://github.com/pytorch/pytorch/issues/78787
            # return torch.arange(cumsum[-1], device=n_per_pack.device)-pack_inds.repeat_interleave(n_per_pack)
            return torch.arange(cumsum[-1], device=n_per_pack.device)-torch.repeat_interleave(pack_inds, n_per_pack)
        
        n_per_pack = torch.randint(32, 96, [4096], device=device)
        y1 = interleave_arange_simple(n_per_pack, False)
        y2 = interleave_arange_pt(n_per_pack)
        y3 = interleave_arange(n_per_pack.new_full(n_per_pack.shape, 0), n_per_pack, n_per_pack.new_full(n_per_pack.shape, 1), False)
        print(torch.allclose(y1, y2))
        print(torch.allclose(y1, y3))

        y1 = interleave_arange(
            torch.tensor([1.1, 2.2, 3.3], device=device), 
            torch.tensor([9.9, 5.5, 7.7], device=device),
            torch.tensor([0.5, 1.3, 0.9], device=device), False)
        ic(y1)

        # 120 us
        print(Timer(
            stmt="interleave_arange_pt(n_per_pack)",
            globals={'interleave_arange_pt': interleave_arange_pt, 'n_per_pack': n_per_pack}
        ).blocked_autorange())

        # 40 us
        print(Timer(
            stmt="interleave_arange_simple(n_per_pack)",
            globals={'interleave_arange_simple': interleave_arange_simple, 'n_per_pack': n_per_pack}
        ).blocked_autorange())
        
        # 60 us
        print(Timer(
            stmt="interleave_arange(n_per_pack.new_full(n_per_pack.shape, 0), n_per_pack, n_per_pack.new_full(n_per_pack.shape, 1))",
            globals={'interleave_arange': interleave_arange, 'n_per_pack': n_per_pack}
        ).blocked_autorange())

    def test_interleave_linstep(device=torch.device('cuda')):
        from torch.utils.benchmark import Timer
        # Integer Tensor
        near = -torch.randint(16, 48, [4096,], device=device)
        far = torch.randint(16, 48, [4096,], device=device)
        step_size = torch.ones([4096,], device=device, dtype=torch.long)
        num_steps = far-near
        y1 = interleave_linstep(near, num_steps, step_size, False)[0].float()
        # Floating Tensor
        near_ = near.float()
        step_size_ = step_size.float()
        y2 = interleave_linstep(near_, num_steps, step_size_, False)[0]
        # Floating Tensor -> scalar step
        y3 = interleave_linstep(near_, num_steps, 1.0, False)[0]
        
        print(torch.allclose(y1, y2))
        print(torch.allclose(y1, y3))
        
        # Old impl: cur_val += step: 40 us
        # New impl: mul
        print(Timer(
            stmt="interleave_linstep(near, num_steps, step_size)[0]",
            globals={'interleave_linstep':interleave_linstep, 'near':near, 'num_steps':num_steps, 'step_size':step_size}
        ).blocked_autorange())
        
        # Even faster than old impl: 40 us
        print(Timer(
            stmt="interleave_linstep(near_, num_steps, step_size_)[0]",
            globals={'interleave_linstep':interleave_linstep, 'near_':near_, 'num_steps':num_steps, 'step_size_':step_size_}
        ).blocked_autorange())

    def test_mark_consecutive_segments(device=torch.device('cuda')):
        from kaolin.ops.mesh import sample_points
        from kaolin.ops.spc import unbatched_points_to_octree
        from kaolin.ops.conversions import unbatched_pointcloud_to_spc
        from kaolin.rep.spc import Spc
        from nr3d_lib.render.raytest import unbatched_raytrace
        from nr3d_lib.geometry import look_at_opencv
        from nr3d_lib.utils import check_to_torch
        from nr3d_lib.render.cameras import pinhole_get_rays, camera_mat_from_hwf
        from torch.utils.benchmark import Timer
        coords = torch.tensor([[0,3,3],[1,3,3],[2,3,3],[5,3,3],[6,3,3]], device=device, dtype=torch.short)
        octree = unbatched_points_to_octree(coords, level=3)
        lengths = torch.tensor([len(octree)], dtype=torch.int32)
        spc = Spc(octree, lengths)
        spc._apply_scan_octrees()
        spc._apply_generate_points()
        
        rays_o = torch.tensor([[-2,0,0]], dtype=torch.float, device=device)
        rays_d = torch.tensor([[1., 1e-10, 1e-10]], dtype=torch.float, device=device)
        
        ridx, pidx, depth = unbatched_raytrace(octree, spc.point_hierarchies, spc.pyramids[0], spc.exsum, rays_o, rays_d, level=3, with_exit=True)
        
        boundary = mark_pack_boundaries(ridx)
        pack_infos = get_pack_infos_from_boundary(boundary)
        
        mark_start, mark_end = octree_mark_consecutive_segments(pidx, pack_infos, spc.point_hierarchies)
        
        seg_ridx = ridx[mark_start]
        seg_near, seg_far = depth[mark_start,0], depth[mark_end,1]
        seg_boundary = boundary[mark_start]
        seg_pack_infos = get_pack_infos_from_boundary(seg_boundary)
        
        #-----------------------------------
        # Performance test
        vertices = torch.tensor([[1, -1, -1], [-1, -1, 1], [0, 1, 0]], dtype=torch.float, device=device)
        faces = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)
        sampled_verts, sampled_faces = sample_points(vertices.unsqueeze(0), faces, 1000000)
        spc = unbatched_pointcloud_to_spc(sampled_verts.squeeze(0), level=7)
        spc._apply_generate_points()
        spc._apply_scan_octrees()
        
        c2w = check_to_torch(look_at_opencv([-2.5,2.5,-2.5], [0,0,0]), dtype=torch.float, device=device)
        intr = camera_mat_from_hwf(200,200,200.)
        rays_o, rays_d = pinhole_get_rays(c2w, intr, 200, 200)
        
        ridx, pidx, depths = unbatched_raytrace(spc.octrees, spc.point_hierarchies, spc.pyramids[0], spc.exsum, rays_o, rays_d, level=7, return_depth=True, with_exit=True)
        boundary = mark_pack_boundaries(ridx)
        
        #-------------------- benchmark calculation of pack indices
        first_inds = boundary.nonzero().long()[..., 0]
        cnt = torch.unique_consecutive(ridx, return_counts=True)[1]
        first_inds1 = cnt.cumsum(0)-cnt
        first_inds2 = torch.unique_consecutive(ridx, return_counts=True)[1].cumsum(0).roll(1)
        first_inds2[0] = 0
        print(torch.allclose(first_inds, first_inds1))
        print(torch.allclose(first_inds, first_inds2))
        # 28.81 us <- 6717 nuggets with 4480 packs (trues)
        print(Timer(
            stmt="boundary.nonzero().long()[..., 0]",
            globals={'boundary': boundary}
        ).blocked_autorange())
        # 45.11 us <- 6717 nuggets 
        print(Timer(
            stmt="cnt = torch.unique_consecutive(ridx, return_counts=True)[1]; first_inds1 = cnt.cumsum(0)-cnt",
            globals={'ridx': ridx}
        ).blocked_autorange())
        # 69.76 us <- 6717 nuggets
        print(Timer(
            stmt="ind=torch.unique_consecutive(ridx, return_counts=True)[1].cumsum(0).roll(1);ind[0]=0",
            globals={'ridx': ridx}
        ).blocked_autorange())
        
        #-------------------- benchmark octree_mark_consecutive_segments
        pack_infos = get_pack_infos_from_boundary(boundary)
        mark_start, mark_end = octree_mark_consecutive_segments(pidx, pack_infos, spc.point_hierarchies)
        # 17.99 us <- 6717 nuggets
        print(Timer(
            stmt="octree_mark_consecutive_segments(pidx, pack_infos, spc.point_hierarchies)",
            globals={'pidx': pidx, 'pack_infos':pack_infos, 'spc':spc, 'octree_mark_consecutive_segments':octree_mark_consecutive_segments}
        ).blocked_autorange())

    def test_cumsum_t(device=torch.device('cuda')):
        from torch.utils.benchmark import Timer
        # [4096] -> 8.12 us
        n_per = torch.randint(32, 64, [4096], device=device, dtype=torch.long)
        print(Timer(
            stmt="n_per.cumsum(0)",
            globals={'n_per': n_per}
        ).blocked_autorange())
        # [40960] -> 8.13 us
        n_per = torch.randint(4, 8, [40960], device=device, dtype=torch.long)
        print(Timer(
            stmt="n_per.cumsum(0)",
            globals={'n_per': n_per}
        ).blocked_autorange())
        # [409600] -> 9.22 us
        n_per = torch.randint(4, 8, [409600], device=device, dtype=torch.long)
        print(Timer(
            stmt="n_per.cumsum(0)",
            globals={'n_per': n_per}
        ).blocked_autorange())

    def test_merge_two_batch(device=torch.device('cuda')):
        from torch.utils.benchmark import Timer
        
        pack_infos = get_pack_infos_from_batch(4096, 32, device=device)
        # 27 us
        print(Timer(
            stmt="get_pack_infos_from_batch(4096, 32, device=device)", 
            globals={'get_pack_infos_from_batch': get_pack_infos_from_batch, 'device': device}
        ).blocked_autorange())
        
        vals_a = torch.randn([4, 3], device=device)
        nidx_a = torch.tensor([10, 11, 12, 13], device=device)
        vals_b = torch.randn([2, 5], device=device)
        nidx_b = torch.tensor([10, 12], device=device)
        pidx_a, pidx_b, pack_infos = merge_two_batch_a_includes_b(vals_a, nidx_a, vals_b, nidx_b, a_sorted=False)

        vals = vals_a.new_empty([vals_a.numel() + vals_b.numel()])
        vals[pidx_a], vals[pidx_b] = vals_a, vals_b
        
        vals_a = torch.randn([2, 3], device=device)
        nidx_a = torch.tensor([10, 12], device=device)
        vals_b = torch.randn([2, 5], device=device)
        nidx_b = torch.tensor([10, 12], device=device)
        pidx_a, pidx_b, pack_infos = merge_two_batch_a_includes_b(vals_a, nidx_a, vals_b, nidx_b, a_sorted=False)
        vals = vals_a.new_empty([vals_a.numel() + vals_b.numel()])
        vals[pidx_a], vals[pidx_b] = vals_a, vals_b
        
        vals_a = torch.randn([4096, 32], device=device)
        nidx_a = torch.arange(4096, device=device)
        vals_b = torch.randn([2048, 64], device=device)
        nidx_b = torch.randint(1, 3, [2048], device=device).cumsum(0)
        pidx_a, pidx_b, pack_infos = merge_two_batch_a_includes_b(vals_a, nidx_a, vals_b, nidx_b, a_sorted=False)
        vals = vals_a.new_empty([vals_a.numel() + vals_b.numel()])
        vals[pidx_a], vals[pidx_b] = vals_a, vals_b
        
        # 438 us @ merging total of 256k pts
        print(Timer(
            stmt="merge_two_batch_a_includes_b(vals_a, nidx_a, vals_b, nidx_b, a_sorted=False)", 
            globals={'merge_two_batch_a_includes_b':merge_two_batch_a_includes_b, 'vals_a':vals_a, 'nidx_a':nidx_a, 'vals_b':vals_b, 'nidx_b':nidx_b}
        ).blocked_autorange())

    def test_intersect1d(device=torch.device('cuda')):
        from torch.utils.benchmark import Timer
        # a = torch.unique(torch.randint(4096, [4096], device=device))
        # b = torch.unique(torch.randint(4096, [8192], device=device))

        a = torch.unique(torch.randint(40960, [40960], device=device))
        b = torch.unique(torch.randint(40960, [81920], device=device))
        
        # @profile
        def test_torch_intersect1d(a: torch.Tensor, b: torch.Tensor):
            # NOTE: requires a, b to be unique 1D Tensor in advance.
            # Method 2: unique
            num_a, num_b = a.numel(), b.numel()
            u, inv, cnt = torch.unique(torch.cat([a,b]), return_counts=True, return_inverse=True)
            
            cnt_ab = cnt[inv]
            cnt_a, cnt_b = cnt_ab[:num_a], cnt_ab[num_a:]
            m_a = (cnt_a == 2)
            inds_a = m_a.nonzero()[..., 0]
            inds_a_exclusive = (~m_a).nonzero()[..., 0]
            inds_b_exclusive = (cnt_b == 1).nonzero()[..., 0]
        
            intersection = a[inds_a]
            a_exclusive = a[inds_a_exclusive]
            b_exclusive = b[inds_b_exclusive]
            return intersection, a_exclusive, b_exclusive

        # @profile
        def test_torch_intersect1d_dense_pair(a: torch.Tensor, b: torch.Tensor):
            # NOTE: requires a, b to be unique 1D Tensor in advance.
            # Method 1: expands to dense 2D pair matrix
            match = (a.view(1,-1) == b.view(-1,1))
            m_a, m_b = match.any(dim=0), match.any(dim=1)
            inds_a = m_a.nonzero()[..., 0]
            inds_a_exclusive = (~m_a).nonzero()[..., 0]
            inds_b_exclusive = (~m_b).nonzero()[..., 0]
            
            intersection = a[inds_a]
            a_exclusive = a[inds_a_exclusive]
            b_exclusive = b[inds_b_exclusive]
            return intersection, a_exclusive, b_exclusive

        i1, a1, b1 = test_torch_intersect1d(a, b)
        i2, a2, b2 = test_torch_intersect1d_dense_pair(a, b)
        print(torch.equal(i1, i2))
        print(torch.equal(a1, a2))
        print(torch.equal(b1, b2))
        
        print(Timer(
            stmt="test_torch_intersect1d(a, b)", 
            globals={'test_torch_intersect1d':test_torch_intersect1d, 'a': a, 'b': b}
        ).blocked_autorange())
        
        print(Timer(
            stmt="test_torch_intersect1d_dense_pair(a, b)", 
            globals={'test_torch_intersect1d_dense_pair':test_torch_intersect1d_dense_pair, 'a': a, 'b': b}
        ).blocked_autorange())

    def test_packed_merge_sorted_two_aligned(device=torch.device('cuda')):
        from icecream import ic
        from torch.utils.benchmark import Timer
        
        
        #-------------- Case 1
        vals_a = torch.tensor([0.1,0.2,0.3,0.4,0.5,   0.2,0.8], device=device, dtype=torch.float)
        pack_infos_a = get_pack_infos_from_n(torch.tensor([5, 2], device=device, dtype=torch.long))
        vals_b = torch.tensor([0.0,0.25,0.26,0.6,   0.1,0.15,0.3,0.4], device=device, dtype=torch.float)
        pack_infos_b = get_pack_infos_from_n(torch.tensor([4, 4], device=device, dtype=torch.long))
        # Should be [1,2,5,6,7, 11,14] and [0,3,4,8,  10,9,12,13]
        pidx_a, pidx_b, pack_infos = merge_two_packs_sorted_aligned(vals_a, pack_infos_a, vals_b, pack_infos_b)
        ic(pidx_a, pidx_b)
        print(torch.equal(pidx_a, pidx_a.new_tensor([1,2,5,6,7,   11,14])))
        print(torch.equal(pidx_b, pidx_b.new_tensor([0,3,4,8,   9,10,12,13])))

        #-------------- Case 2 with duplicated values in a/b
        vals_a = torch.tensor([0.15,0.2,0.3,0.4,0.5,   0.2,0.2], device=device, dtype=torch.float)
        pack_infos_a = get_pack_infos_from_n(torch.tensor([5, 2], device=device, dtype=torch.long))
        vals_b = torch.tensor([0.1,0.1,0.1,0.1,   0.1,0.15,0.3,0.4], device=device, dtype=torch.float)
        pack_infos_b = get_pack_infos_from_n(torch.tensor([4, 4], device=device, dtype=torch.long))
        pidx_a, pidx_b, pack_infos = merge_two_packs_sorted_aligned(vals_a, pack_infos_a, vals_b, pack_infos_b)
        ic(pidx_a, pidx_b)

        n_per_pack_1 = torch.randint(32, 96, [4096], device=device)
        pack_infos_1 = get_pack_infos_from_n(n_per_pack_1)
        n_per_pack_2 = torch.randint(32, 96, [4096], device=device)
        pack_infos_2 = get_pack_infos_from_n(n_per_pack_2)
        
        t1 = interleave_linspace(-torch.randn([4096], device=device).abs(), torch.randn([4096], device=device).abs(), n_per_pack_1, return_idx=False)
        t2 = interleave_linspace(-torch.randn([4096], device=device).abs(), torch.randn([4096], device=device).abs(), n_per_pack_2, return_idx=False)
        p1, p2, pack_infos_total = merge_two_packs_sorted_aligned(t1, pack_infos_1, t2, pack_infos_2)
        
        # 160 us @ merging 2 x 256k vs. 172 us wheh `b_sorted`=False
        print(Timer(
            stmt='merge_two_packs_sorted_aligned(t1, pack_infos_1, t2, pack_infos_2)', 
            globals={'merge_two_packs_sorted_aligned':merge_two_packs_sorted_aligned, 't1':t1, 't2':t2, 'pack_infos_1':pack_infos_1, 'pack_infos_2':pack_infos_2}
        ).blocked_autorange())        

        # debug_data = torch.load("./dev_test/dbg_merge/data.pt", map_location=device)
        # vals_a, pack_infos_a, nidx_a = debug_data['vals_a'], debug_data['pack_infos_a'], debug_data['nidx_a']
        # vals_b, pack_infos_b, nidx_b = debug_data['vals_b'], debug_data['pack_infos_b'], debug_data['nidx_b']
        # pidx_a, pidx_b, pack_infos = merge_two_packs_sorted_aligned(vals_a, pack_infos_a, vals_b, pack_infos_b, b_sorted=False, return_val=False)

    def test_merge_two_packs_sorted_a_includes_b(device=torch.device('cuda')):
        from icecream import ic
        from torch.utils.benchmark import Timer

        vals_a = torch.tensor([0.1,0.2,0.3,0.4,0.5,   11.1,11.2,   0.2,0.8], device=device, dtype=torch.float)
        n_a = torch.tensor([5, 2, 2], device=device, dtype=torch.long)
        pack_infos_a = get_pack_infos_from_n(n_a)
        nidx_a = torch.tensor([11, 12, 13], device=device, dtype=torch.long)
        
        vals_b = torch.tensor([0.0,0.25,0.26,0.6,   0.1,0.2,0.3,0.4], device=device, dtype=torch.float)
        n_b = torch.tensor([4, 4], device=device, dtype=torch.long)
        pack_infos_b = get_pack_infos_from_n(n_b)
        nidx_b = torch.tensor([11, 13], device=device, dtype=torch.long)
        
        pidx_a, pidx_b, pack_infos = merge_two_packs_sorted_a_includes_b(vals_a, pack_infos_a, nidx_a, vals_b, pack_infos_b, nidx_b, return_val=False)
        ic(pidx_a, pidx_b)
        
        # Should be [ 1,  2,  5,  6,  7,  9, 10, 13, 16] and [ 0,  3,  4,  8, 11, 12, 14, 15]
        print(torch.equal(pidx_a, pidx_a.new_tensor([ 1,2,5,6,7,  9,10,  13,16])))
        print(torch.equal(pidx_b, pidx_b.new_tensor([ 0,3,4,8,   11,12,14,15])))
        print(torch.equal(pack_infos, pack_infos.new_tensor([[ 0,  9],  [ 9,  2],  [11,  6]])))
        
        nidx_a = torch.unique(torch.randint(4096, [8192], device=device)) # about 3.5k
        nidx_b = nidx_a[torch.unique(torch.randint(nidx_a.numel(), [4096], device=device))] # about 2.5k
        
        n_per_pack_a = torch.randint(32, 96, [nidx_a.numel()], device=device)
        n_per_pack_b = torch.randint(32, 96, [nidx_b.numel()], device=device)
        pack_infos_a = get_pack_infos_from_n(n_per_pack_a)
        pack_infos_b = get_pack_infos_from_n(n_per_pack_b)
        
        ta = interleave_linspace(-torch.randn([nidx_a.numel()], device=device).abs(), torch.randn([nidx_a.numel()], device=device).abs(), n_per_pack_a, return_idx=False)
        tb = interleave_linspace(-torch.randn([nidx_b.numel()], device=device).abs(), torch.randn([nidx_b.numel()], device=device).abs(), n_per_pack_b, return_idx=False)
        
        pa, pb, pack_infos_total = merge_two_packs_sorted_a_includes_b(ta, pack_infos_a, nidx_a, tb, pack_infos_b, nidx_b, b_sorted=True, return_val=False)
        # # 929 us
        print(Timer(
            stmt='merge_two_packs_sorted(ta, pack_infos_a, nidx_a, tb, pack_infos_b, nidx_b, b_sorted=True, return_val=False)', 
            globals={'merge_two_packs_sorted':merge_two_packs_sorted, 'nidx_a': nidx_a, 'nidx_b': nidx_b, 'ta':ta, 'tb':tb, 'pack_infos_a':pack_infos_a, 'pack_infos_b':pack_infos_b}
        ).blocked_autorange())
        # 575 us
        print(Timer(
            stmt='merge_two_packs_sorted_a_includes_b(ta, pack_infos_a, nidx_a, tb, pack_infos_b, nidx_b, b_sorted=True, return_val=False)', 
            globals={'merge_two_packs_sorted_a_includes_b':merge_two_packs_sorted_a_includes_b, 'nidx_a': nidx_a, 'nidx_b': nidx_b, 'ta':ta, 'tb':tb, 'pack_infos_a':pack_infos_a, 'pack_infos_b':pack_infos_b}
        ).blocked_autorange())

        # debug_data = torch.load("./dev_test/dbg_merge/data.pt", map_location=device)
        # vals_a, pack_infos_a, nidx_a = debug_data['vals_a'], debug_data['pack_infos_a'], debug_data['nidx_a']
        # vals_b, pack_infos_b, nidx_b = debug_data['vals_b'], debug_data['pack_infos_b'], debug_data['nidx_b']
        # pidx_a, pidx_b, pack_infos = merge_two_packs_sorted_a_includes_b(**debug_data, b_sorted=False, return_val=False)

    def test_merge_two_packs_sorted(device=torch.device('cuda')):
        from torch.utils.benchmark import Timer
        from icecream import ic
        vals_a = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 11.1, 11.2, 0.2, 0.8], device=device, dtype=torch.float)
        n_a = torch.tensor([5, 2, 2], device=device, dtype=torch.long)
        pack_infos_a = get_pack_infos_from_n(n_a)
        nidx_a = torch.tensor([11, 12, 13], device=device, dtype=torch.long)
        
        vals_b = torch.tensor([21.1, 0.0, 0.25, 0.26, 0.6, 0.1, 0.2, 0.3, 0.4, 31.1, 31.2, 31.3], device=device, dtype=torch.float)
        n_b = torch.tensor([1, 4, 4, 3], device=device, dtype=torch.long)
        pack_infos_b = get_pack_infos_from_n(n_b)
        nidx_b = torch.tensor([10, 11, 13, 14], device=device, dtype=torch.long)
        
        pidx_a, pidx_b, pack_infos = merge_two_packs_sorted(vals_a, pack_infos_a, nidx_a, vals_b, pack_infos_b, nidx_b, return_val=False)
        ic(pidx_a, pidx_b)
        # Should be [ 2,  3,  6,  7,  8, 10, 11, 14, 17] and [ 0,  1,  4,  5,  9, 12, 13, 15, 16, 18, 19, 20]
        print(torch.equal(pidx_a, pidx_a.new_tensor([ 2,  3,  6,  7,  8, 10, 11, 14, 17])))
        print(torch.equal(pidx_b, pidx_b.new_tensor([ 0,  1,  4,  5,  9, 12, 13, 15, 16, 18, 19, 20])))
        print(torch.equal(pack_infos, pack_infos.new_tensor([[0,1], [1,9], [10,2], [12,6], [18,3]])))
        
        nidx_1 = torch.unique(torch.randint(4096, [4096], device=device)) # about 2.5k
        nidx_2 = torch.unique(torch.randint(4096, [8192], device=device)) # about 3.5k
        
        n_per_pack_1 = torch.randint(32, 96, [nidx_1.numel()], device=device)
        n_per_pack_2 = torch.randint(32, 96, [nidx_2.numel()], device=device)
        pack_infos_1 = get_pack_infos_from_n(n_per_pack_1)
        pack_infos_2 = get_pack_infos_from_n(n_per_pack_2)
        
        t1 = interleave_linspace(-torch.randn([nidx_1.numel()], device=device).abs(), torch.randn([nidx_1.numel()], device=device).abs(), n_per_pack_1, return_idx=False)
        t2 = interleave_linspace(-torch.randn([nidx_2.numel()], device=device).abs(), torch.randn([nidx_2.numel()], device=device).abs(), n_per_pack_2, return_idx=False)
        
        # p1, p2, pack_infos_total = merge_two_packs_sorted(t1, pack_infos_1, nidx_1, t2, pack_infos_2, nidx_2, b_sorted=True, return_val=False)
        # 1.1 ms
        print(Timer(
            stmt='merge_two_packs_sorted(t1, pack_infos_1, nidx_1, t2, pack_infos_2, nidx_2, b_sorted=True, return_val=False)', 
            globals={'merge_two_packs_sorted':merge_two_packs_sorted, 'nidx_1': nidx_1, 'nidx_2': nidx_2, 't1':t1, 't2':t2, 'pack_infos_1':pack_infos_1, 'pack_infos_2':pack_infos_2}
        ).blocked_autorange())

    def test_packed_sort(device=torch.device('cuda')):
        from torch.utils.benchmark import Timer
        from icecream import ic
        vals = torch.tensor([0.2,0.1,0.3,  2.9,2.3,2.5,2.4,2.1,  1.0,1.1], device=device, dtype=torch.float)
        vals_copy = vals.clone()
        n_per_pack = torch.tensor([3, 5, 2], device=device, dtype=torch.long)
        pack_infos = get_pack_infos_from_n(n_per_pack)
        idx = packed_sort_inplace(vals_copy, pack_infos, return_idx=True)
        ic(vals[idx])
        print(torch.equal(vals[idx], vals_copy))
        
        n_per_pack = torch.randint(32, 64, [4096], device=device, dtype=torch.long)
        pack_infos = get_pack_infos_from_n(n_per_pack)
        vals = torch.randn(n_per_pack.sum(), device=device, dtype=torch.float)
        vals_sorted, indices = packed_sort(vals, pack_infos)
        # [thrust::sort] 6.39 ms @ 200k pts
        # [quicksort] 334 us @ 200k pts
        print(Timer(
            stmt="packed_sort(vals, pack_infos)", 
            globals={'packed_sort':packed_sort, 'vals':vals, 'pack_infos':pack_infos}
        ).blocked_autorange())
        
        # 90 us @ 200k pts (NOTE: Not the same thing, but can demonstrate speed difference)
        print(Timer(
            stmt="vals.sort()", 
            globals={'vals':vals}
        ).blocked_autorange())

        n_per_pack = torch.randint(320, 640, [4096], device=device, dtype=torch.long)
        pack_infos = get_pack_infos_from_n(n_per_pack)
        vals = torch.randn(n_per_pack.sum(), device=device, dtype=torch.float)
        vals_sorted, indices = packed_sort(vals, pack_infos)
        # [thrust::sort] 20 ms @ 2 Mi pts
        # [quicksort] 12 ms @ 2 Mi pts
        print(Timer(
            stmt="packed_sort(vals, pack_infos)", 
            globals={'packed_sort':packed_sort, 'vals':vals, 'pack_infos':pack_infos}
        ).blocked_autorange())

    def test_packed_search_sorted(device=torch.device('cuda')):
        from torch.utils.benchmark import Timer
        from icecream import ic
        bins = torch.tensor([2,3,7,  1,  0,4,  8,  3,6], device=device, dtype=torch.float)
        u = torch.tensor([4,        0,   2,    9,   7], device=device, dtype=torch.float)
        n_per_pack = torch.tensor([3,1,2,1,2], device=device, dtype=torch.long)
        pack_infos = get_pack_infos_from_n(n_per_pack)
        idx = packed_searchsorted(bins, u.unsqueeze(-1), pack_infos)
        ic(idx)

        u_packed = torch.tensor([3.5,1.2,      2.2,-1,3.2,   6.0,7.0,    5.0], device=device, dtype=torch.float)
        u_n_per_pack = torch.tensor([2,0,3,2,1], device=device, dtype=torch.long)
        u_pack_infos = get_pack_infos_from_n(u_n_per_pack)
        idx = packed_searchsorted_packed_vals(bins, pack_infos, u_packed, u_pack_infos)
        ic(idx)
        
        bins = torch.tensor([2,3,7,          1,2,      3,4,6,8], dtype=torch.float, device=device)
        cdfs = torch.tensor([0.0,0.4,1.0,    0.0,1.0,  0.0,0.1,0.8,1.0], dtype=torch.float, device=device)
        n_per_pack = torch.tensor([3, 2, 4], dtype=torch.long, device=device)
        pack_infos = get_pack_infos_from_n(n_per_pack)
        u = torch.tensor([0.5, 0.9], dtype=torch.float, device=device).tile([3, 1])
        samples, bin_idx = packed_invert_cdf(bins, cdfs, u, pack_infos)
        ic(samples, bin_idx)

    def test_interleave_sample_step_wrt_depth_clamped(device=torch.device('cuda')):
        import numpy as np
        import matplotlib.pyplot as plt
        from torch.utils.benchmark import Timer
        import nr3d_lib_bindings._pack_ops as _backend
        near = torch.tensor([0.5], dtype=torch.float, device=device)
        far = torch.tensor([250.], dtype=torch.float, device=device)
        t_samples, deltas, ridx, pack_infos = interleave_sample_step_wrt_depth_clamped(
            near, far, max_steps=512, dt_gamma=0.02, min_step_size=0.05, max_step_size=4.0)
        fig = plt.figure()
        plt.hist(t_samples.flatten().data.cpu().numpy(), bins=np.arange(250), label='sample_density')
        ax = plt.gca()
        ax2 = ax.twinx()
        ax2.plot(t_samples.data.cpu().numpy(), deltas.data.cpu().numpy(), 'brown', label='dt')
        ax2.plot(t_samples.data.cpu().numpy(), (deltas.flatten() / t_samples.flatten()).data.cpu().numpy(), 'green', label='dt_r')
        fig.legend(loc='center right', bbox_to_anchor=(1, 0.5), bbox_transform=ax.transAxes)
        plt.show()

        # t_samples0, deltas0, ridx0, pack_infos0 = interleave_sample_step_wrt_depth_clamped(
        #     near, far, max_steps=512, dt_gamma=0.1, min_step_size=0.01, max_step_size=1.0, perturb=True)

        near = 1.0 * torch.rand([4096,], device=device, dtype=torch.float)
        far = (100.0 * torch.rand([4096,], device=device, dtype=torch.float)).clamp_min_(2.0)
        t_samples, deltas, ridx, pack_infos = _backend.interleave_sample_step_wrt_depth_clamp_deprecated(near, far, 512, 0.01, 0.01, 10.0)
        # t_samples2, deltas2, ridx2, pack_infos2 = _backend.sample_step_wrt_gamma_depth_clamp_v2(near, far, 512, 0.01, 0.01, 10.0)
        t_samples3, deltas3, ridx3, pack_infos3 = _backend.interleave_sample_step_wrt_depth_clamped(near, far, 512, 0.01, 0.01, 10.0)
        print(torch.allclose(t_samples, t_samples3))
        print(torch.allclose(deltas, deltas3))
        print(torch.allclose(ridx, ridx3))
        print(torch.allclose(pack_infos, pack_infos3))
        
        """ About 1.7M samples generated (4096 packs with ~480 points average each)
        v1: slice:
            128 threads -> 454 us
            256 threads -> 929 us
            512 threads -> 1.70 ms
            1024 threads -> 3.36 ms
        v2: atomic counter (orders not preserverd, causing trouble)
            128 threads -> 220 us
            256 threads -> 411 us
            512 threads -> 726 us
            1024 threads -> 1.39 ms
        v3: first pass, second pass, two kernels (still preserves order)
            128 threads -> 219 us
            256 threads -> 
            512 threads -> 
            1024 threads -> 
        """
        print(Timer(
            stmt="_backend.interleave_sample_step_wrt_depth_clamp_deprecated(near, far, 512, 0.01, 0.01, 10.0)",
            globals={'near':near, 'far':far, '_backend':_backend}
        ).blocked_autorange())
        # print(Timer(
        #     stmt="_backend.sample_step_wrt_gamma_depth_clamp_v2(near, far, 512, 0.01, 0.01, 10.0)",
        #     globals={'near':near, 'far':far, '_backend':_backend}
        # ).blocked_autorange())
        print(Timer(
            stmt="_backend.interleave_sample_step_wrt_depth_clamped(near, far, 512, 0.01, 0.01, 10.0)",
            globals={'near':near, 'far':far, '_backend':_backend}
        ).blocked_autorange())

        #---------------------------------
        #------------ test perturb
        #---------------------------------
        near = 1.0 * torch.rand([4096,], device=device, dtype=torch.float)
        far = (100.0 * torch.rand([4096,], device=device, dtype=torch.float)).clamp_min_(2.0)
        
        # t_samples, deltas, ridx, pack_infos = interleave_sample_step_wrt_depth_clamped(
        #     near, far, max_steps=512, dt_gamma=0.1, min_step_size=0.01, max_step_size=1.0, perturb=True)
        
        """ about 1.7M samples generated (4096 packs with ~480 points average each)
        not perturb: 231 us
        perturb:     334 us (re-calculate delta takes 100us)
        """
        print(Timer(
            stmt="fn(near, far, perturb=False, max_steps=512, dt_gamma=0.01, min_step_size=0.01, max_step_size=10.0)",
            globals={'near':near, 'far':far, 'fn':interleave_sample_step_wrt_depth_clamped}
        ).blocked_autorange())

        print(Timer(
            stmt="fn(near, far, perturb=True, max_steps=512, dt_gamma=0.01, min_step_size=0.01, max_step_size=10.0)",
            globals={'near':near, 'far':far, 'fn':interleave_sample_step_wrt_depth_clamped}
        ).blocked_autorange())

    def test_interleave_sample_step_wrt_depth_in_packed_segments(device=torch.device('cuda')):
        import numpy as np
        import matplotlib.pyplot as plt        
        from torch.utils.benchmark import Timer
        from nr3d_lib.render.pack_ops import get_pack_infos_from_first
        near = torch.tensor([0.5], dtype=torch.float, device=device)
        far = torch.tensor([250.], dtype=torch.float, device=device)
        seg_entry = torch.tensor([1.0, 10.0, 90.0,  200.0], dtype=torch.float, device=device)
        seg_exit =  torch.tensor([2.0, 11.0, 100.0, 260.0], dtype=torch.float, device=device)
        seg_pack_indices = torch.tensor([0], dtype=torch.long, device=device)
        seg_pack_infos = get_pack_infos_from_first(seg_pack_indices, seg_entry.numel())
        t_samples, deltas, ridx, pack_infos, sidx, _ = interleave_sample_step_wrt_depth_in_packed_segments(
            near, far, seg_entry, seg_exit, seg_pack_infos, max_steps=512, dt_gamma=0.02, min_step_size=0.05, max_step_size=4.0)
        fig = plt.figure()
        plt.hist(t_samples.flatten().data.cpu().numpy(), bins=np.arange(250), label='sample_density')
        ax = plt.gca()
        ax2 = ax.twinx()
        ax2.plot(t_samples.data.cpu().numpy(), deltas.data.cpu().numpy(), 'rx-', label='dt')
        ax2.plot(t_samples.data.cpu().numpy(), (deltas.flatten() / t_samples.flatten()).data.cpu().numpy(), 'gx--', label='dt_r')
        fig.legend(loc='center right', bbox_to_anchor=(1, 0.5), bbox_transform=ax.transAxes)
        plt.show()

    # test_packed_sum()
    # test_packed_diff()
    # test_backward_diff()
    # test_packed_search_sorted()
    # test_packed_binary_ops_arithmetic()
    # test_packed_binary_ops_compare()
    # test_packed_matmul()
    # test_n_per_pack_t()
    # test_pack_infos_t()
    # test_expand_pack_boundary()
    test_exclusive()
    # test_mask_ridx()
    # test_ridx_hit_t()
    # test_int_repeat()
    # test_comma()
    # test_interleave_arange()
    # test_interleave_linstep()
    # test_mark_consecutive_segments()
    # test_cumsum_t()
    # test_merge_two_batch()
    # test_intersect1d()
    # test_packed_merge_sorted_two_aligned()
    # test_merge_two_packs_sorted_a_includes_b()
    # test_merge_two_packs_sorted()
    # test_packed_sort()
    # test_interleave_sample_step_wrt_depth_clamped()
    # test_interleave_sample_step_wrt_depth_in_packed_segments()

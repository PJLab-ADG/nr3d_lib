"""
@file   neus_ray_query.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Utility funcitons for NeuS model and rendering. Implemented on both batched-rays and packed-rays.

Below we provide two implementations of alpha calculation.
- `diff impl` stands for our implementation
- `nablas + estimate impl` stands for neus original implmentation

For more details on differences of these two implementations, please refer to nr3d_lib/models/fields/neus/renderer_mixin.py
"""

__all__ = [
    'neus_ray_query_sphere_trace', 
    'neus_ray_query_coarse_multi_upsample', 
    'neus_ray_query_march_occ_multi_upsample', 
    'neus_ray_query_march_occ_multi_upsample_compressed'
]

from operator import itemgetter
from typing import Dict, List, Literal, Tuple, Union

import torch
import torch.nn.functional as F

from nr3d_lib.profile import profile

from nr3d_lib.models.utils import batchify_query
from nr3d_lib.models.accelerations import \
    accel_types_single, accel_types_dynamic, accel_types_batched, accel_types_batched_dynamic

from nr3d_lib.graphics.raysample import batch_sample_pdf, batch_sample_step_wrt_depth, \
    batch_sample_step_linear, batch_sample_step_wrt_sqrt_depth, packed_sample_cdf
from nr3d_lib.graphics.pack_ops import get_pack_infos_from_batch, merge_two_batch_a_includes_b, \
    packed_cumsum, merge_two_packs_sorted_aligned, packed_diff, packed_div, packed_sum
from nr3d_lib.graphics.nerf import packed_alpha_to_vw, ray_alpha_to_vw, \
    packed_volume_render_compression
from nr3d_lib.graphics.neus.neus_utils import neus_packed_sdf_to_alpha, neus_packed_sdf_to_upsample_alpha, \
    neus_ray_sdf_to_alpha, neus_ray_sdf_to_upsample_alpha

def neus_ray_query_sphere_trace(
    model, ray_tested: Dict[str, torch.Tensor], *, 
    # Common params
    with_rgb: bool = True, with_normal: bool = True, 
    perturb: bool = False, nablas_has_grad: bool = False,
    # Distinct params
    debug: bool = False, **sphere_trace_cfg, 
    ) -> Tuple[dict, dict]:
    
    assert hasattr(model, 'forward'), "model.forward() is requried"
    assert hasattr(model, 'forward_sdf'), "model.forward_sdf() is requried"
    assert hasattr(model, 'forward_inv_s'), "model.forward_inv_s() is requried"
    assert getattr(model, 'accel', None) is not None, "model.accel is required"

    use_ts = getattr(model, 'use_ts', False)
    use_fidx = getattr(model, 'use_fidx', False)
    use_bidx = getattr(model, 'use_bidx', False)
    fwd_sdf_use_pix = getattr(model, 'fwd_sdf_use_pix', False)
    fwd_sdf_use_h_appear = getattr(model, 'fwd_sdf_use_h_appear', False)
    fwd_sdf_use_view_dirs = getattr(model, 'fwd_sdf_use_view_dirs', False)
    use_pix = (getattr(model, 'use_pix', False) and with_rgb) or fwd_sdf_use_pix
    use_h_appear = (getattr(model, 'use_h_appear', False) and with_rgb) or fwd_sdf_use_h_appear
    use_view_dirs = (getattr(model, 'use_view_dirs', False) and with_rgb) or fwd_sdf_use_view_dirs

    empty_buffer = dict(type="empty", rays_inds_hit=[])
    if ray_tested["num_rays"] == 0:
        return empty_buffer
    assert ray_tested["rays_o"].dim() == 2

    from nr3d_lib.graphics.sphere_trace import SphereTracer, DenseGrid
    model.tracer = SphereTracer(
        DenseGrid(*model.accel.occ.resolution, model.accel.occ.occ_grid),
        **sphere_trace_cfg)
    
    rays_to_trace = {
        "rays_o": ray_tested["rays_o"],
        "rays_d": ray_tested["rays_d"],
        "near": ray_tested["near"],
        "far": ray_tested["far"]
    }
    
    ret_st = model.tracer.trace(rays_to_trace, model.forward_sdf, print_debug_log=debug)
    if ret_st["idx"].numel() == 0:
        return empty_buffer, {}
    
    with profile("Acquire volume buffer"):
        volume_buffer = dict(
            type="batched", 
            rays_inds_hit=ray_tested["rays_inds"],
            num_per_hit=1,
            t=torch.zeros_like(ray_tested["near"])[:, None],
            opacity_alpha=torch.zeros_like(ray_tested["near"])[:, None]
        )
        volume_buffer["opacity_alpha"][ret_st["idx"]] = 1.
        if use_bidx: 
            volume_buffer['rays_bidx_hit'] = torch.full_like(ray_tested["near"], -1)[:, None]
            volume_buffer['rays_bidx_hit'][ret_st["idx"]] = ray_tested['rays_bidx'][ret_st["idx"]]
        
        if with_rgb or with_normal:
            fwd_kwargs = dict(
                x=ret_st["pos"], 
                nablas_has_grad=nablas_has_grad, with_rgb=with_rgb, with_normal=with_normal
            )
            # Extra infos attached to each ray
            if use_ts: fwd_kwargs['ts'] = ray_tested['rays_ts'][ret_st["idx"]]
            if use_fidx: fwd_kwargs['fidx'] = ray_tested['rays_fidx'][ret_st["idx"]]
            if use_bidx: fwd_kwargs['bidx'] = ray_tested['rays_bidx'][ret_st["idx"]]
            if use_pix: fwd_kwargs['pix'] = ray_tested['rays_pix'][ret_st["idx"]]
            if use_h_appear: fwd_kwargs['h_appear'] = ray_tested['rays_h_appear'][ret_st["idx"]]
            if use_view_dirs: fwd_kwargs['v'] = F.normalize(ret_st["dir"], 2, -1)
            # Net forward
            net_out = model.forward(**fwd_kwargs)
            volume_buffer["net_x"] = torch.zeros_like(ray_tested["rays_o"])[:, None]
            volume_buffer["net_x"][ret_st["idx"]] = ret_st["pos"][:, None]
            if "nablas" in net_out:
                volume_buffer["nablas"] = torch.zeros_like(ray_tested["rays_o"])[:, None]
                volume_buffer["nablas"][ret_st["idx"]] = \
                    net_out["nablas"][:, None].to(volume_buffer["nablas"].dtype)
            if "rgb" in net_out:
                volume_buffer["rgb"] = ray_tested["rays_o"].new_zeros(ray_tested["rays_o"].shape[0], 1, net_out["rgb"].shape[-1])
                volume_buffer["rgb"][ret_st["idx"]] = \
                    net_out["rgb"][:, None].to(volume_buffer["rgb"].dtype)

            # import nr3d_lib.bindings._sphere_trace as _backend
            # rays_alive = self.tracer.backend.get_rays(_backend.ALIVE)
            # volume_buffer["opacity_alpha"][rays_alive["idx"]] = 1.
            # volume_buffer["rgb"][rays_alive["idx"]] = torch.tensor([1., 0., 0.], device=rays_alive["idx"].device)

        details = {'render.num_per_ray': 1}
    return volume_buffer, details

def neus_ray_query_coarse_multi_upsample(
    model, ray_tested: Dict[str, torch.Tensor], 
    # Common params
    with_rgb: bool = True, with_normal: bool = True, 
    perturb: bool = False, nablas_has_grad: bool = False, forward_inv_s: float = None, 
    # Distinct params
    compression = True, # Whether to early stop when already accumulated enough weights
    num_coarse: int = 64, coarse_step_cfg = dict(step_mode='linear'), 
    upsample_mode: str = 'multistep_estimate', num_fine: int = 64, 
    upsample_inv_s: float = 64., upsample_s_divisor: float = 1.0, 
    upsample_inv_s_factors: List[int] = [1, 2, 4, 8], upsample_use_estimate_alpha=False,  # For upsample_mode = multistep_estimate
    num_nograd: int = 1024, chunksize_query: int = 2**24 # For upsample_mode = direct_more
    ) -> Tuple[dict, dict]:
    """
    Vanilla NeuS ray query mode
    """
    assert hasattr(model, 'forward'), "model.forward() is requried"
    assert hasattr(model, 'forward_sdf'), "model.forward_sdf() is requried"
    assert hasattr(model, 'forward_inv_s'), "model.forward_inv_s() is requried"

    use_ts = getattr(model, 'use_ts', False)
    use_fidx = getattr(model, 'use_fidx', False)
    use_bidx = getattr(model, 'use_bidx', False)
    fwd_sdf_use_pix = getattr(model, 'fwd_sdf_use_pix', False)
    fwd_sdf_use_h_appear = getattr(model, 'fwd_sdf_use_h_appear', False)
    fwd_sdf_use_view_dirs = getattr(model, 'fwd_sdf_use_view_dirs', False)
    use_pix = (getattr(model, 'use_pix', False) and with_rgb) or fwd_sdf_use_pix
    use_h_appear = (getattr(model, 'use_h_appear', False) and with_rgb) or fwd_sdf_use_h_appear
    use_view_dirs = (getattr(model, 'use_view_dirs', False) and with_rgb) or fwd_sdf_use_view_dirs

    empty_volume_buffer = dict(type='empty', rays_inds_hit=[])
    if (num_rays:=ray_tested['num_rays']) == 0:
        return empty_volume_buffer, {}

    upsample_inv_s /= upsample_s_divisor
    forward_inv_s = model.forward_inv_s() if forward_inv_s is None else forward_inv_s

    # NOTE: Normalized rays in network's space
    rays_o, rays_d, near, far, rays_inds = itemgetter('rays_o', 'rays_d', 'near', 'far', 'rays_inds')(ray_tested)
    assert (rays_o.dim() == 2) and (rays_d.dim()==2)
    
    # NOTE: Extra infos attached to each ray
    if use_pix: rays_pix = ray_tested['rays_pix']
    if use_ts: rays_ts = ray_tested['rays_ts']
    if use_fidx: rays_fidx = ray_tested['rays_fidx']
    if use_bidx: rays_bidx = ray_tested['rays_bidx']
    if use_h_appear: rays_h_appear = ray_tested['rays_h_appear']
    
    # NOTE: The device & dtype of output
    device, dtype = rays_o.device, rays_o.dtype
    # NOTE: The spatial length scale on each ray caused by scaling rays_d 
    dir_scale = rays_d.data.norm(dim=-1)  # [num_rays]
    # NOTE: The normalized ray direction vector in network's space
    view_dirs = rays_d / dir_scale.clamp_min_(1.0e-10).unsqueeze(-1) # [num_rays, 3]
    
    #----------------
    # Coarse sampling
    #----------------
    coarse_step_cfg = coarse_step_cfg.copy()
    step_mode = coarse_step_cfg.pop('step_mode')
    if step_mode == 'linear':
        depths_coarse_1, deltas_coarse_1 = batch_sample_step_linear(near, far, num_coarse+1, perturb=perturb, return_dt=True, **coarse_step_cfg)
    elif step_mode == 'depth':
        depths_coarse_1, deltas_coarse_1 = batch_sample_step_wrt_depth(near, far, num_coarse+1, perturb=perturb, return_dt=True, **coarse_step_cfg)
    elif step_mode == 'sqrt_depth':
        depths_coarse_1, deltas_coarse_1 = batch_sample_step_wrt_sqrt_depth(near, far, num_coarse+1, perturb=perturb, return_dt=True, **coarse_step_cfg)
    else:
        raise RuntimeError(f"Invalid step_mode={step_mode}")
    
    @torch.no_grad()
    def upsample():
        if upsample_mode == 'direct_use':
            fwd_kwargs = dict(x=torch.addcmul(rays_o.unsqueeze(-2), rays_d.unsqueeze(-2), depths_coarse_1.unsqueeze(-1)))
            if use_ts: fwd_kwargs['ts'] = rays_ts.unsqueeze(-1).expand(*fwd_kwargs['x'].shape[:-1]).contiguous()
            if use_fidx: fwd_kwargs['fidx'] = rays_fidx.unsqueeze(-1).expand(*fwd_kwargs['x'].shape[:-1]).contiguous()
            if use_bidx: fwd_kwargs['bidx'] = rays_bidx.unsqueeze(-1).expand(*fwd_kwargs['x'].shape[:-1]).contiguous()
            if fwd_sdf_use_pix: fwd_kwargs['pix'] = rays_pix.unsqueeze(-2).expand(*fwd_kwargs['x'].shape[:-1], -1).contiguous()
            if fwd_sdf_use_h_appear: fwd_kwargs['h_appear'] = rays_h_appear.unsqueeze(-2).expand(*fwd_kwargs['x'].shape[:-1], -1).contiguous()
            if fwd_sdf_use_view_dirs: fwd_kwargs['v'] = view_dirs.unsqueeze(-2).expand(*fwd_kwargs['x'].shape[:-1], -1).contiguous()
            sdf_coarse = model.forward_sdf(**fwd_kwargs)['sdf']
            if upsample_use_estimate_alpha:
                alpha_coarse = neus_ray_sdf_to_upsample_alpha(sdf_coarse, depths_coarse_1, upsample_inv_s)
            else:
                alpha_coarse = neus_ray_sdf_to_alpha(sdf_coarse, upsample_inv_s)
            vw_coarse = ray_alpha_to_vw(alpha_coarse)
            # Fine points
            d_fine = batch_sample_pdf(depths_coarse_1, vw_coarse, num_fine, perturb=perturb)
            # Gather points
            d_all = torch.cat([depths_coarse_1, d_fine], dim=-1).sort(d_all, dim=-1).values
        elif upsample_mode == 'direct_more':
            _d = near.lerp(far, torch.linspace(0, 1, num_nograd, device=device, dtype=torch.float))
            fwd_kwargs = dict(x=torch.addcmul(rays_o.unsqueeze(-2), rays_d.unsqueeze(-2), _d.unsqueeze(-1)))
            if use_ts: fwd_kwargs['ts'] = rays_ts.unsqueeze(-1).expand(*fwd_kwargs['x'].shape[:-1]).contiguous()
            if use_fidx: fwd_kwargs['fidx'] = rays_fidx.unsqueeze(-1).expand(*fwd_kwargs['x'].shape[:-1]).contiguous()
            if use_bidx: fwd_kwargs['bidx'] = rays_bidx.unsqueeze(-1).expand(*fwd_kwargs['x'].shape[:-1]).contiguous()
            if fwd_sdf_use_pix: fwd_kwargs['pix'] = rays_pix.unsqueeze(-2).expand(*fwd_kwargs['x'].shape[:-1], -1).contiguous()
            if fwd_sdf_use_h_appear: fwd_kwargs['h_appear'] = rays_h_appear.unsqueeze(-2).expand(*fwd_kwargs['x'].shape[:-1], -1).contiguous()
            if fwd_sdf_use_view_dirs: fwd_kwargs['v'] = view_dirs.unsqueeze(-2).expand(*fwd_kwargs['x'].shape[:-1], -1).contiguous()
            _sdf = batchify_query(model.forward_sdf, **fwd_kwargs, chunk=chunksize_query)['sdf']
            if upsample_use_estimate_alpha:
                _alpha = neus_ray_sdf_to_upsample_alpha(_sdf, _d, upsample_inv_s)
            else:
                _alpha = neus_ray_sdf_to_alpha(_sdf, upsample_inv_s)
            _vw = ray_alpha_to_vw(_alpha)
            # Fine points
            d_fine = batch_sample_pdf(_d, _vw, num_fine, perturb=perturb)
            # Gather points
            d_all = torch.cat([_d, d_fine], dim=-1).sort(d_all, dim=-1).values
        elif upsample_mode == 'multistep_estimate':
            d_all = depths_coarse_1
            fwd_kwargs = dict(x=torch.addcmul(rays_o.unsqueeze(-2), rays_d.unsqueeze(-2), depths_coarse_1.unsqueeze(-1)))
            if use_ts: fwd_kwargs['ts'] = rays_ts.unsqueeze(-1).expand(*fwd_kwargs['x'].shape[:-1]).contiguous()
            if use_fidx: fwd_kwargs['fidx'] = rays_fidx.unsqueeze(-1).expand(*fwd_kwargs['x'].shape[:-1]).contiguous()
            if use_bidx: fwd_kwargs['bidx'] = rays_bidx.unsqueeze(-1).expand(*fwd_kwargs['x'].shape[:-1]).contiguous()
            if fwd_sdf_use_pix: fwd_kwargs['pix'] = rays_pix.unsqueeze(-2).expand(*fwd_kwargs['x'].shape[:-1], -1).contiguous()
            if fwd_sdf_use_h_appear: fwd_kwargs['h_appear'] = rays_h_appear.unsqueeze(-2).expand(*fwd_kwargs['x'].shape[:-1], -1).contiguous()
            if fwd_sdf_use_view_dirs: fwd_kwargs['v'] = view_dirs.unsqueeze(-2).expand(*fwd_kwargs['x'].shape[:-1], -1).contiguous()
            sdf_all = model.forward_sdf(**fwd_kwargs)['sdf']
            num_fine_per_iter = num_fine//2*2+1 # Had better always be odd
            extra_fwd_kwargs_per_iter = {}
            if use_ts: extra_fwd_kwargs_per_iter['ts'] = rays_ts.unsqueeze(-1).expand(num_rays, num_fine_per_iter).contiguous()
            if use_fidx: extra_fwd_kwargs_per_iter['fidx'] = rays_fidx.unsqueeze(-1).expand(num_rays, num_fine_per_iter).contiguous()
            if use_bidx: extra_fwd_kwargs_per_iter['bidx'] = rays_bidx.unsqueeze(-1).expand(num_rays, num_fine_per_iter).contiguous()
            if fwd_sdf_use_pix: extra_fwd_kwargs_per_iter['pix'] = rays_pix.unsqueeze(-2).expand(num_rays, num_fine_per_iter, -1).contiguous()
            if fwd_sdf_use_h_appear: extra_fwd_kwargs_per_iter['h_appear'] = rays_h_appear.unsqueeze(-2).expand(num_rays, num_fine_per_iter, -1).contiguous()
            if fwd_sdf_use_view_dirs: extra_fwd_kwargs_per_iter['v'] = view_dirs.unsqueeze(-2).expand(num_rays, num_fine_per_iter, -1).contiguous()
            for i, factor in enumerate(upsample_inv_s_factors):
                if upsample_use_estimate_alpha:
                    alpha = neus_ray_sdf_to_upsample_alpha(sdf_all, d_all, upsample_inv_s * factor)
                else:
                    alpha = neus_ray_sdf_to_alpha(sdf_all, upsample_inv_s * factor)
                vw = ray_alpha_to_vw(alpha)
                d_fine_ter = batch_sample_pdf(d_all, vw, num_fine_per_iter, perturb=perturb)
                d_all, d_sort_indices = torch.sort(torch.cat([d_all, d_fine_ter], dim=-1), dim=-1)
                if i < len(upsample_inv_s_factors)-1:
                    sdf_fine_iter = model.forward_sdf(
                        x=torch.addcmul(rays_o.unsqueeze(-2), rays_d.unsqueeze(-2), d_fine_ter.unsqueeze(-1)), 
                        **extra_fwd_kwargs_per_iter)['sdf']
                    sdf_all = torch.gather(torch.cat([sdf_all, sdf_fine_iter], dim=-1), -1, d_sort_indices) 
        else:
            raise RuntimeError(f"Invalid upsample_mode={upsample_mode}")
        return d_all
    
    d_all = upsample()
    d_mid = 0.5 * (d_all[..., 1:] + d_all[..., :-1])
    fwd_kwargs = dict(x=torch.addcmul(rays_o.unsqueeze(-2), rays_d.unsqueeze(-2), d_all.unsqueeze(-1)))
    if use_ts: fwd_kwargs['ts'] = rays_ts.unsqueeze(-1).expand(*fwd_kwargs['x'].shape[:-1]).contiguous()
    if use_fidx: fwd_kwargs['fidx'] = rays_fidx.unsqueeze(-1).expand(*fwd_kwargs['x'].shape[:-1]).contiguous()
    if use_bidx: fwd_kwargs['bidx'] = rays_bidx.unsqueeze(-1).expand(*fwd_kwargs['x'].shape[:-1]).contiguous()
    if fwd_sdf_use_pix: fwd_kwargs['pix'] = rays_pix.unsqueeze(-2).expand(*fwd_kwargs['x'].shape[:-1], -1).contiguous()
    if fwd_sdf_use_h_appear: fwd_kwargs['h_appear'] = rays_h_appear.unsqueeze(-2).expand(*fwd_kwargs['x'].shape[:-1], -1).contiguous()
    if fwd_sdf_use_view_dirs: fwd_kwargs['v'] = view_dirs.unsqueeze(-2).expand(*fwd_kwargs['x'].shape[:-1], -1).contiguous()
    alpha = neus_ray_sdf_to_alpha(model.forward_sdf(**fwd_kwargs)['sdf'], forward_inv_s) # The same shape with d_mid
    
    if compression:
        # NOTE: `pack_infos` is for all `ray_tested` rays / `rays_inds` / `num_rays`
        pack_infos = get_pack_infos_from_batch(alpha.shape[0], alpha.shape[1], device=device)
        nidx_useful, pack_infos_useful, pidx_useful = packed_volume_render_compression(alpha.flatten(), pack_infos)
        
        if nidx_useful.numel() == 0:
            return empty_volume_buffer, {}
        else:
            depths_packed, alpha_packed = d_mid.flatten()[pidx_useful], alpha.flatten()[pidx_useful]
            volume_buffer = dict(
                type='packed', 
                rays_inds_hit=rays_inds[nidx_useful], 
                pack_infos_hit=pack_infos_useful, 
                t=depths_packed.to(dtype), 
                opacity_alpha=alpha_packed.to(dtype))
            if use_bidx: volume_buffer['rays_bidx_hit'] = rays_bidx[nidx_useful]

            if with_rgb or with_normal:
                ridx_all = torch.arange(alpha.shape[0], device=device, dtype=torch.long).unsqueeze(-1).expand_as(alpha)
                ridx_all = ridx_all.flatten()[pidx_useful]
                pts_mid = torch.addcmul(rays_o[ridx_all], rays_d[ridx_all], depths_packed.unsqueeze(-1))
                # Basic inputs: [num_rays_hit, num_fine_all, ...]
                fwd_kwargs = dict(
                    x=pts_mid, 
                    nablas_has_grad=nablas_has_grad, with_rgb=with_rgb, with_normal=with_normal
                )
                # Extra infos attached to each ray
                if use_ts: fwd_kwargs['ts'] = rays_ts[ridx_all]
                if use_fidx: fwd_kwargs['fidx'] = rays_fidx[ridx_all]
                if use_bidx: fwd_kwargs['bidx'] = rays_bidx[ridx_all]
                if use_pix: fwd_kwargs['pix'] = rays_pix[ridx_all]
                if use_h_appear: fwd_kwargs['h_appear'] = rays_h_appear[ridx_all]
                if use_view_dirs: fwd_kwargs['v'] = view_dirs[ridx_all]
                #----------- Net forward
                net_out = model.forward(**fwd_kwargs)
                volume_buffer['net_x'] = pts_mid
                if "nablas" in net_out: volume_buffer["nablas"] = net_out["nablas"].to(dtype)
                if "rgb" in net_out: volume_buffer["rgb"] = net_out["rgb"].to(dtype)
            details = {'render.num_per_ray0': d_mid.size(-1), 
                       'render.num_per_ray': pack_infos_useful[:, 1]}
            return volume_buffer, details
    else: # not compression
        volume_buffer = dict(
            type='batched', 
            rays_inds_hit=rays_inds, 
            num_per_hit=d_mid.size(-1), 
            t=d_mid.to(dtype), 
            opacity_alpha=alpha.to(dtype))
        if use_bidx: volume_buffer['rays_bidx_hit'] = rays_bidx

        if with_rgb or with_normal:
            pts_mid = torch.addcmul(rays_o.unsqueeze(-2), rays_d.unsqueeze(-2), d_mid.unsqueeze(-1))
            # Basic inputs: 
            fwd_kwargs = dict(
                x=pts_mid, 
                nablas_has_grad=nablas_has_grad, with_rgb=with_rgb, with_normal=with_normal
            )
            # Extra infos attached to each ray
            if use_ts: fwd_kwargs['ts'] = rays_ts.unsqueeze(-1).expand(*d_mid.shape).contiguous()
            if use_fidx: fwd_kwargs['fidx'] = rays_fidx.unsqueeze(-1).expand(*d_mid.shape).contiguous()
            if use_bidx: fwd_kwargs['bidx'] = rays_bidx.unsqueeze(-1).expand(*d_mid.shape).contiguous()
            if use_pix: fwd_kwargs['pix'] = rays_pix.unsqueeze(-2).expand(*d_mid.shape, -1).contiguous()
            if use_h_appear: fwd_kwargs['h_appear'] = rays_h_appear.unsqueeze(-2).expand(*d_mid.shape,-1).contiguous()
            if use_view_dirs: fwd_kwargs['v'] = view_dirs.unsqueeze(-2).expand(*d_mid.shape,-1).contiguous()
            #----------- Net forward
            net_out = model.forward(**fwd_kwargs)
            volume_buffer['net_x'] = pts_mid
            if "nablas" in net_out: volume_buffer["nablas"] = net_out["nablas"].to(dtype)
            if "rgb" in net_out: volume_buffer["rgb"] = net_out["rgb"].to(dtype)
        details = {'render.num_per_ray': d_mid.size(-1)}
        return volume_buffer, details

def neus_ray_query_march_occ_multi_upsample(
    model, ray_tested: Dict[str, torch.Tensor], 
    # Common params
    with_rgb: bool = True, with_normal: bool = True, 
    perturb: bool = False, nablas_has_grad: bool = False, forward_inv_s: float = None, 
    # Distinct params
    num_coarse: int = 0, coarse_step_cfg = dict(step_mode='linear'), 
    chunksize_query: int = 2**24,  march_cfg = dict(), num_fine: int = 8,    
    upsample_inv_s: float = 64., upsample_s_divisor: float = 1.0, 
    upsample_inv_s_factors: List[int] = [1, 4, 16], upsample_use_estimate_alpha=False, 
    debug_query_data: dict = None, 
    ) -> Tuple[dict, dict]:
    """
    Multi-stage upsampling on marched samples of occupancy grids (without reduction/compression of meaningless samples)
    Introduced in StreetSurf Section 4.1
    https://arxiv.org/abs/2306.04988
    """
    assert hasattr(model, 'forward'), "model.forward() is requried"
    assert hasattr(model, 'forward_sdf'), "model.forward_sdf() is requried"
    assert hasattr(model, 'forward_inv_s'), "model.forward_inv_s() is requried"
    assert getattr(model, 'accel', None) is not None, "model.accel is required"

    use_ts = getattr(model, 'use_ts', False)
    use_fidx = getattr(model, 'use_fidx', False)
    use_bidx = getattr(model, 'use_bidx', False)
    fwd_sdf_use_pix = getattr(model, 'fwd_sdf_use_pix', False)
    fwd_sdf_use_h_appear = getattr(model, 'fwd_sdf_use_h_appear', False)
    fwd_sdf_use_view_dirs = getattr(model, 'fwd_sdf_use_view_dirs', False)
    use_pix = (getattr(model, 'use_pix', False) and with_rgb) or fwd_sdf_use_pix
    use_h_appear = (getattr(model, 'use_h_appear', False) and with_rgb) or fwd_sdf_use_h_appear
    use_view_dirs = (getattr(model, 'use_view_dirs', False) and with_rgb) or fwd_sdf_use_view_dirs

    empty_volume_buffer = dict(type='empty', rays_inds_hit=[])
    if (num_rays:=ray_tested['num_rays']) == 0:
        return empty_volume_buffer, {}
    
    if isinstance(num_fine, int):
        num_fine = [num_fine] * len(upsample_inv_s_factors)
    assert len(num_fine) == len(upsample_inv_s_factors), \
        f"num_fine should be of the same length={len(upsample_inv_s_factors)} with upsample"
    num_fine = [n // 2 * 2 + 1 for n in num_fine] # Had better always be odd

    upsample_inv_s /= upsample_s_divisor
    forward_inv_s = model.forward_inv_s() if forward_inv_s is None else forward_inv_s

    # NOTE: Normalized rays in network's space
    rays_o, rays_d, near, far, rays_inds = itemgetter('rays_o', 'rays_d', 'near', 'far', 'rays_inds')(ray_tested)
    assert (rays_o.dim() == 2) and (rays_d.dim()==2)
    
    # NOTE: Extra infos attached to each ray
    if use_pix: rays_pix = ray_tested['rays_pix']
    if use_ts: rays_ts = ray_tested['rays_ts']
    if use_fidx: rays_fidx = ray_tested['rays_fidx']
    if use_bidx: rays_bidx = ray_tested['rays_bidx']
    if use_h_appear: rays_h_appear = ray_tested['rays_h_appear']
    
    # NOTE: The device & dtype of output
    device, dtype = rays_o.device, rays_o.dtype
    
    # NOTE: The spatial length scale on each ray caused by scaling rays_d 
    dir_scale = rays_d.data.norm(dim=-1)  # [num_rays]
    # NOTE: The normalized ray direction vector in network's space
    view_dirs = rays_d / dir_scale.clamp_min_(1.0e-10).unsqueeze(-1) # [num_rays, 3]
    
    #----------------
    # Coarse sampling
    #----------------
    if num_coarse > 0:
        coarse_step_cfg = coarse_step_cfg.copy()
        step_mode = coarse_step_cfg.pop('step_mode')
        if step_mode == 'linear':
            depths_coarse_1, deltas_coarse_1 = batch_sample_step_linear(near, far, num_coarse+1, perturb=perturb, return_dt=True, **coarse_step_cfg)
        elif step_mode == 'depth':
            depths_coarse_1, deltas_coarse_1 = batch_sample_step_wrt_depth(near, far, num_coarse+1, perturb=perturb, return_dt=True, **coarse_step_cfg)
        elif step_mode == 'sqrt_depth':
            depths_coarse_1, deltas_coarse_1 = batch_sample_step_wrt_sqrt_depth(near, far, num_coarse+1, perturb=perturb, return_dt=True, **coarse_step_cfg)
        else:
            raise RuntimeError(f"Invalid step_mode={step_mode}")
        deltas_coarse = deltas_coarse_1[..., :num_coarse]
        depths_coarse = depths_coarse_1[..., :num_coarse] + deltas_coarse / 2.

    #----------------
    # Ray marching
    #----------------
    with profile("Ray marching"):
        accel = model.accel
        if isinstance(accel, accel_types_single):
            marched = accel.ray_march(
                rays_o, rays_d, 
                near=near, far=far, perturb=perturb, **march_cfg)
        elif isinstance(accel, accel_types_dynamic):
            marched = accel.ray_march(
                rays_o, rays_d, ray_tested['rays_ts'], 
                near=near, far=far, perturb=perturb, **march_cfg)
        elif isinstance(accel, accel_types_batched):
            marched = accel.cur_batch__ray_march(
                rays_o, rays_d, ray_tested['rays_bidx'], 
                near=near, far=far, perturb=perturb, **march_cfg)
        elif isinstance(accel, accel_types_batched_dynamic):
            marched = accel.cur_batch__ray_march(
                rays_o, rays_d, ray_tested['rays_bidx'], ray_tested['rays_ts'], 
                near=near, far=far, perturb=perturb, **march_cfg)
        else:
            raise RuntimeError(f"Unsupported accel of type={type(accel)}")
    
    #----------------
    # Upsampling & compute volume_buffer
    #----------------
    if marched.ridx_hit is not None:
        with profile("Upsampling"):
            pack_infos = marched.pack_infos.clone()
            depth_samples = marched.depth_samples
            rays_inds_hit = rays_inds[marched.ridx_hitx]
            # [num_rays_hit, 1, 3]
            rays_o_hit = rays_o[marched.ridx_hit].unsqueeze(-2)
            rays_d_hit = rays_d[marched.ridx_hit].unsqueeze(-2)
            if use_ts: rays_ts_hit = rays_ts[marched.ridx_hit]
            if use_fidx: rays_fidx_hit = rays_fidx[marched.ridx_hit]
            if use_bidx: rays_bidx_hit = rays_bidx[marched.ridx_hit]
            if use_pix: rays_pix_hit = rays_pix[marched.ridx_hit]
            if use_h_appear: rays_h_appear_hit = rays_h_appear[marched.ridx_hit]
            if use_view_dirs: view_dirs_hit = view_dirs[marched.ridx_hit]
            
            #----------------
            # Upsample on marched samples
            with torch.no_grad():
                fwd_kwargs = dict(x=marched.samples)
                if use_ts: fwd_kwargs['ts'] = rays_ts[marched.ridx]
                if use_fidx: fwd_kwargs['fidx'] = rays_fidx[marched.ridx]
                if use_bidx: fwd_kwargs['bidx'] = rays_bidx[marched.ridx]
                if fwd_sdf_use_pix: fwd_kwargs['pix'] = rays_pix[marched.ridx]
                if fwd_sdf_use_h_appear: fwd_kwargs['h_appear'] = rays_h_appear[marched.ridx]
                if fwd_sdf_use_view_dirs: fwd_kwargs['v'] = view_dirs[marched.ridx]
                sdf = batchify_query(lambda **kw: model.forward_sdf(**kw)['sdf'], **fwd_kwargs, chunk=chunksize_query)
                
                depths_1 = []
                
                for i, factor in enumerate(upsample_inv_s_factors):
                    pinfo_fine_per_iter = get_pack_infos_from_batch(marched.num_hit_rays, num_fine[i], device=device)
                    fwd_extra_kwargs_pter_iter = dict()
                    if use_ts: fwd_extra_kwargs_pter_iter['ts'] = rays_ts_hit.unsqueeze(-1).expand(marched.num_hit_rays, num_fine[i]).flatten(0,-1).contiguous()
                    if use_fidx: fwd_extra_kwargs_pter_iter['fidx'] = rays_fidx_hit.unsqueeze(-1).expand(marched.num_hit_rays, num_fine[i]).flatten(0,-1).contiguous()
                    if use_bidx: fwd_extra_kwargs_pter_iter['bidx'] = rays_bidx_hit.unsqueeze(-1).expand(marched.num_hit_rays, num_fine[i]).flatten(0,-1).contiguous()
                    if fwd_sdf_use_pix: fwd_extra_kwargs_pter_iter['pix'] = rays_pix_hit.unsqueeze(-2).expand(marched.num_hit_rays, num_fine[i], -1).flatten(0,-2).contiguous()
                    if fwd_sdf_use_h_appear: fwd_extra_kwargs_pter_iter['h_appear'] = rays_h_appear_hit.unsqueeze(-2).expand(marched.num_hit_rays, num_fine[i], -1).flatten(0,-2).contiguous()
                    if fwd_sdf_use_view_dirs: fwd_extra_kwargs_pter_iter['v'] = view_dirs_hit.unsqueeze(-2).expand(marched.num_hit_rays, num_fine[i], -1).flatten(0,-2).contiguous()
                    
                    if upsample_use_estimate_alpha:
                        alpha = neus_packed_sdf_to_upsample_alpha(sdf, depth_samples, upsample_inv_s * factor, pack_infos) # This could leads to artifacts
                    else:
                        alpha = neus_packed_sdf_to_alpha(sdf, upsample_inv_s * factor, pack_infos)
                    
                    vw = packed_alpha_to_vw(alpha, pack_infos)
                    
                    neus_cdf = packed_cumsum(vw, pack_infos, exclusive=True)
                    last_cdf = neus_cdf[pack_infos[...,0] + pack_infos[...,1] - 1]
                    neus_cdf = packed_div(neus_cdf, last_cdf.clamp_min(1e-5), pack_infos)
                    depths_fine_iter = packed_sample_cdf(depth_samples, neus_cdf.to(depth_samples.dtype), pack_infos, num_fine[i], perturb=perturb)[0]
                    depths_1.append(depths_fine_iter)
                    
                    if len(upsample_inv_s_factors) > 1:
                        # 273 us @ 930k + 25k
                        # Merge fine samples of current upsample iter to previous packed buffer.
                        # NOTE: The new `pack_infos` is calculated here.
                        pidx0, pidx1, pack_infos = merge_two_packs_sorted_aligned(
                            depth_samples, pack_infos, depths_fine_iter.flatten(), pinfo_fine_per_iter, 
                            b_sorted=True, return_val=False)
                        num_samples_iter = depth_samples.numel()
                        depth_samples_iter = depth_samples.new_empty([num_samples_iter + depths_fine_iter.numel()])
                        depth_samples_iter[pidx0], depth_samples_iter[pidx1] = depth_samples, depths_fine_iter.flatten()
                        depth_samples = depth_samples_iter

                        if i < len(upsample_inv_s_factors)-1:
                            x_fine = torch.addcmul(rays_o_hit, rays_d_hit, depths_fine_iter.unsqueeze(-1))
                            sdf_iter = sdf.new_empty([num_samples_iter + depths_fine_iter.numel()])
                            sdf_iter[pidx0], sdf_iter[pidx1] = sdf, model.forward_sdf(x_fine.flatten(0, -2), **fwd_extra_kwargs_pter_iter)['sdf']
                            sdf = sdf_iter
                
                if len(upsample_inv_s_factors) > 1:
                    depths_sort_ret = torch.cat(depths_1, dim=-1).sort(dim=-1)
                    depths_1 = depths_sort_ret.values
                    # NOTE: Upsample stage marker for each upsampled point
                    upsample_stages = torch.repeat_interleave(1+torch.arange(len(upsample_inv_s_factors), dtype=torch.long, device=device), 
                                                            torch.tensor(num_fine, dtype=torch.long, device=device))
                    upsample_stages = torch.take_along_dim(upsample_stages.unsqueeze(0), depths_sort_ret.indices, dim=-1)
                else:
                    depths_1 = depths_1[0]
                    upsample_stages = torch.ones_like(depths_1, dtype=torch.long, device=device)

        #----------------
        # Acquire volume_buffer via quering network and gather results
        with profile("Acquire volume buffer"):
            if num_coarse == 0: # num_coarse == 0 and ridx_hit is not None
                # [num_rays_hit, num_fine_all]
                fwd_kwargs = dict(x=torch.addcmul(rays_o_hit, rays_d_hit, depths_1.unsqueeze(-1)))
                if use_ts: fwd_kwargs['ts'] = rays_ts_hit.unsqueeze(-1).expand(marched.num_hit_rays, depths_1.size(-1)).contiguous()
                if use_fidx: fwd_kwargs['fidx'] = rays_fidx_hit.unsqueeze(-1).expand(marched.num_hit_rays, depths_1.size(-1)).contiguous()
                if use_bidx: fwd_kwargs['bidx'] = rays_bidx_hit.unsqueeze(-1).expand(marched.num_hit_rays, depths_1.size(-1)).contiguous()
                if fwd_sdf_use_pix: fwd_kwargs['pix'] = rays_pix_hit.unsqueeze(-2).expand(marched.num_hit_rays, depths_1.size(-1), -1).contiguous()
                if fwd_sdf_use_h_appear: fwd_kwargs['h_appear'] = rays_h_appear_hit.unsqueeze(-2).expand(marched.num_hit_rays, depths_1.size(-1), -1).contiguous()
                if fwd_sdf_use_view_dirs: fwd_kwargs['v'] = view_dirs_hit.unsqueeze(-2).expand(marched.num_hit_rays, depths_1.size(-1), -1).contiguous()

                if model.training:
                    sdf = model.forward_sdf(**fwd_kwargs)['sdf']
                else:
                    sdf = batchify_query(lambda **kw: model.forward_sdf(**kw)['sdf'], **fwd_kwargs, chunk=chunksize_query)
                alpha = neus_ray_sdf_to_alpha(sdf, forward_inv_s)
                depths = (depths_1[..., :-1] + depths_1.diff(dim=-1)/2.)
                
                if debug_query_data is not None:
                    debug_query_data["fine"] = {
                        "ridx": rays_inds_hit[..., None].expand_as(depths).flatten(),
                        "depth": depths_1.data.to(dtype).flatten(),
                        "sdf": sdf.data.to(dtype).flatten(),
                        "upsample_stages": upsample_stages.flatten()
                    }
                
                volume_buffer = dict(
                    type='batched', 
                    rays_inds_hit=rays_inds_hit, 
                    num_per_hit=depths.size(-1), 
                    t=depths.to(dtype), opacity_alpha=alpha.to(dtype))
                if use_bidx: volume_buffer['rays_bidx_hit'] = rays_bidx_hit

                if with_rgb or with_normal:
                    # [num_rays_hit, num_fine_all, ...]
                    # Basic inputs: 
                    fwd_kwargs = dict(
                        x=torch.addcmul(rays_o_hit, rays_d_hit, depths[..., None]),
                        nablas_has_grad=nablas_has_grad, with_rgb=with_rgb, with_normal=with_normal
                    )
                    # Extra infos attached to each ray
                    if use_ts: fwd_kwargs['ts'] = rays_ts_hit.unsqueeze(-1).expand(*depths.shape).contiguous()
                    if use_fidx: fwd_kwargs['fidx'] = rays_fidx_hit.unsqueeze(-1).expand(*depths.shape).contiguous()
                    if use_bidx: fwd_kwargs['bidx'] = rays_bidx_hit.unsqueeze(-1).expand(*depths.shape).contiguous()
                    if use_pix: fwd_kwargs['pix'] = rays_pix_hit.unsqueeze(-2).expand(*depths.shape, -1).contiguous()
                    if use_h_appear: fwd_kwargs['h_appear'] = rays_h_appear_hit.unsqueeze(-2).expand(*depths.shape,-1).contiguous()
                    if use_view_dirs: fwd_kwargs['v'] = view_dirs_hit.unsqueeze(-2).expand(*depths.shape,-1).contiguous()
                    #----------- Net forward
                    net_out = model.forward(**fwd_kwargs)
                    if "nablas" in net_out: volume_buffer["nablas"] = net_out["nablas"].to(dtype)
                    if "rgb" in net_out: volume_buffer["rgb"] = net_out["rgb"].to(dtype)
                    volume_buffer['net_x'] = fwd_kwargs['x']
                details = {'march.num_per_ray': marched.pack_infos[:, 1], 
                        'render.num_per_ray': depths.size(-1)}
                return volume_buffer, details

            else: # num_coarse != 0 and ridx_hit is not None
                def merge():
                    ridx_hit_coarse = torch.arange(rays_inds.numel(), device=device)
                    pidx0, pidx1, pack_infos = merge_two_batch_a_includes_b(depths_coarse_1, ridx_hit_coarse, depths_1, marched.ridx_hit, a_sorted=True)
                    num_samples = depths_1.numel() + depths_coarse_1.numel()
                    depths_1_packed = depths_1.new_zeros([num_samples])
                    sample_stages_packed = upsample_stages.new_zeros([num_samples])
                    ridx_all = marched.ridx_hit.new_zeros([num_samples])
                    ridx_all[pidx0], ridx_all[pidx1] = ridx_hit_coarse.unsqueeze(-1), marched.ridx_hit.unsqueeze(-1)
                    depths_1_packed[pidx0], depths_1_packed[pidx1] = depths_coarse_1, depths_1
                    sample_stages_packed[pidx0], sample_stages_packed[pidx1] = 0, upsample_stages
                    return ridx_all, depths_1_packed, sample_stages_packed, pack_infos

                # NOTE: `pack_infos` here is for all `ray_tested` rays / `ray_inds` / `num_rays`
                ridx_all, depths_1_packed, sample_stages_packed, pack_infos = merge() 
                rays_o_packed, rays_d_packed = rays_o[ridx_all], rays_d[ridx_all]
                # Extra infos attached to each ray
                extra_fwd_kwargs = dict()
                if use_ts: extra_fwd_kwargs['ts'] = rays_ts[ridx_all]
                if use_fidx: extra_fwd_kwargs['fidx'] = rays_fidx[ridx_all]
                if use_bidx: extra_fwd_kwargs['bidx'] = rays_bidx[ridx_all]
                if fwd_sdf_use_pix: extra_fwd_kwargs['pix'] = rays_pix[ridx_all]
                if fwd_sdf_use_h_appear: extra_fwd_kwargs['h_appear'] = rays_h_appear[ridx_all]
                if fwd_sdf_use_view_dirs: extra_fwd_kwargs['v'] = view_dirs[ridx_all]
                sdf_packed = model.forward_sdf(x=torch.addcmul(rays_o_packed, rays_d_packed, depths_1_packed.unsqueeze(-1)), 
                                                **extra_fwd_kwargs)['sdf']
                alpha_packed = neus_packed_sdf_to_alpha(sdf_packed, forward_inv_s, pack_infos)
                depths_packed = (depths_1_packed + packed_diff(depths_1_packed, pack_infos) / 2.)
                
                if debug_query_data is not None:
                    coarse_sample_indices = (sample_stages_packed == 0).nonzero(as_tuple=True)[0]
                    fine_sample_indices = (sample_stages_packed > 0).nonzero(as_tuple=True)[0]
                    debug_query_data["coarse"] = {
                        "ridx": ridx_all[coarse_sample_indices],
                        "depth": depths_1_packed.data.to(dtype)[coarse_sample_indices],
                        "sdf": sdf_packed.data.to(dtype)[coarse_sample_indices]
                    }
                    debug_query_data["fine"] = {
                        "ridx": ridx_all[fine_sample_indices],
                        "depth": depths_1_packed.data.to(dtype)[fine_sample_indices],
                        "sdf": sdf_packed.data.to(dtype)[fine_sample_indices],
                        "upsample_stages": sample_stages_packed[fine_sample_indices]
                    }
                
                volume_buffer = dict(
                    type='packed', 
                    rays_inds_hit=rays_inds, 
                    pack_infos_hit=pack_infos, 
                    t=depths_packed.to(dtype), 
                    opacity_alpha=alpha_packed.to(dtype))
                if use_bidx: volume_buffer['rays_bidx_hit'] = rays_bidx

                if with_rgb or with_normal:
                    # Basic inputs: 
                    fwd_kwargs = dict(
                        x=torch.addcmul(rays_o_packed, rays_d_packed, depths_packed.unsqueeze(-1)), 
                        nablas_has_grad=nablas_has_grad, with_rgb=with_rgb, with_normal=with_normal
                    )
                    # Extra infos attached to each ray
                    if use_pix: extra_fwd_kwargs['pix'] = rays_pix[ridx_all]
                    if use_h_appear: extra_fwd_kwargs['h_appear'] = rays_h_appear[ridx_all]
                    if use_view_dirs: extra_fwd_kwargs['v'] = view_dirs[ridx_all]
                    #----------- Net forward
                    net_out = model.forward(**fwd_kwargs, **extra_fwd_kwargs)
                    volume_buffer['net_x'] = fwd_kwargs['x']
                    if "nablas" in net_out: volume_buffer["nablas"] = net_out["nablas"].to(dtype)
                    if "rgb" in net_out: volume_buffer["rgb"] = net_out["rgb"].to(dtype)
                details = {'march.num_per_ray': marched.pack_infos[:, 1], 
                        'render.num_per_ray': pack_infos[:, 1]}
                return volume_buffer, details
    
    else: # ridx_hit is None
        
        if num_coarse == 0: # ridx_hit is None and num_coarse == 0
            return empty_volume_buffer, {}
        
        else: # ridx_hit is None and num_coarse > 0
            with profile("Acquire volume buffer"):
                fwd_kwargs = dict(x=torch.addcmul(rays_o[..., None, :], rays_d[..., None, :], depths_coarse_1[..., None]))
                if use_ts: fwd_kwargs['ts'] = rays_ts.unsqueeze(-1).expand(*depths_coarse_1.shape).contiguous()
                if use_fidx: fwd_kwargs['fidx'] = rays_fidx.unsqueeze(-1).expand(*depths_coarse_1.shape).contiguous()
                if use_bidx: fwd_kwargs['bidx'] = rays_bidx.unsqueeze(-1).expand(*depths_coarse_1.shape).contiguous()
                if fwd_sdf_use_pix: fwd_kwargs['pix'] = rays_pix.unsqueeze(-2).expand(*depths_coarse_1.shape, -1).contiguous()
                if fwd_sdf_use_h_appear: fwd_kwargs['h_appear'] = rays_h_appear.unsqueeze(-2).expand(*depths_coarse_1.shape, -1).contiguous()
                if fwd_sdf_use_view_dirs: fwd_kwargs['v'] = view_dirs.unsqueeze(-2).expand(*depths_coarse_1.shape, -1).contiguous()
                sdf_coarse = model.forward_sdf(**fwd_kwargs)['sdf']
                alpha_coarse = neus_ray_sdf_to_alpha(sdf_coarse, forward_inv_s)
                
                deltas_coarse = deltas_coarse_1[..., :num_coarse]
                depths_coarse = (depths_coarse_1[..., :num_coarse] + deltas_coarse / 2.)
                
                if debug_query_data is not None:
                    debug_query_data["coarse"] = {
                        "ridx": rays_inds[..., None].expand(-1, num_coarse).flatten(),
                        "depth": deltas_coarse_1.data.flatten().to(dtype),
                        "sdf": sdf_coarse.data.flatten().to(dtype)
                    }
                
                volume_buffer = dict(
                    type='batched', 
                    rays_inds_hit=rays_inds, 
                    num_per_hit=num_coarse, 
                    t=depths_coarse.to(dtype), 
                    opacity_alpha=alpha_coarse.to(dtype))
                if use_bidx: volume_buffer['rays_bidx_hit'] = rays_bidx

                if with_rgb or with_normal:
                    # Basic inputs: 
                    fwd_kwargs = dict(
                        x=torch.addcmul(rays_o[..., None, :], rays_d[..., None, :], depths_coarse[..., None]), 
                        nablas_has_grad=nablas_has_grad, with_rgb=with_rgb, with_normal=with_normal
                    )
                    # Extra infos attached to each ray
                    if use_ts: fwd_kwargs['ts'] = rays_ts.unsqueeze(-1).expand(*depths_coarse.shape).contiguous()
                    if use_fidx: fwd_kwargs['fidx'] = rays_fidx.unsqueeze(-1).expand(*depths_coarse.shape).contiguous()
                    if use_bidx: fwd_kwargs['bidx'] = rays_bidx.unsqueeze(-1).expand(*depths_coarse.shape).contiguous()
                    if use_pix: fwd_kwargs['pix'] = rays_pix.unsqueeze(-2).expand(*depths_coarse.shape, -1).contiguous()
                    if use_h_appear: fwd_kwargs['h_appear'] = rays_h_appear.unsqueeze(-2).expand(*depths_coarse.shape, -1).contiguous()
                    if use_view_dirs: fwd_kwargs['v'] = view_dirs.unsqueeze(-2).expand(*depths_coarse.shape, -1).contiguous()
                    #----------- Net forward
                    net_out = model.forward(**fwd_kwargs)
                    volume_buffer['net_x'] = fwd_kwargs['x']
                    if "nablas" in net_out: volume_buffer["nablas"] = net_out["nablas"].to(dtype)
                    if "rgb" in net_out: volume_buffer["rgb"] = net_out["rgb"].to(dtype)
                details = {'render.num_per_ray': num_coarse}
                return volume_buffer, details

def neus_ray_query_march_occ_multi_upsample_compressed(
    model, ray_tested: Dict[str, torch.Tensor], 
    # Common params
    with_rgb: bool = True, with_normal: bool = True, 
    perturb: bool = False, nablas_has_grad: bool = False, forward_inv_s: float = None, 
    # Distinct params
    num_coarse: int = 0, coarse_step_cfg = dict(step_mode='linear'), 
    chunksize_query: int = 2**24,  march_cfg = dict(), num_fine: int = 8,    
    upsample_inv_s: float = 64., upsample_s_divisor: float = 1.0, 
    upsample_inv_s_factors: List[int] = [1, 4, 16], upsample_use_estimate_alpha=False
    ) -> Tuple[dict, dict]:
    """
    Multi-stage upsampling on marched samples of occupancy grids (with reduction/compression of meaningless samples)
    Introduced in StreetSurf Section 4.1
    https://arxiv.org/abs/2306.04988
    """
    assert hasattr(model, 'forward'), "model.forward() is requried"
    assert hasattr(model, 'forward_sdf'), "model.forward_sdf() is requried"
    assert hasattr(model, 'forward_inv_s'), "model.forward_inv_s() is requried"
    assert getattr(model, 'accel', None) is not None, "model.accel is required"

    use_ts = getattr(model, 'use_ts', False)
    use_fidx = getattr(model, 'use_fidx', False)
    use_bidx = getattr(model, 'use_bidx', False)
    fwd_sdf_use_pix = getattr(model, 'fwd_sdf_use_pix', False)
    fwd_sdf_use_h_appear = getattr(model, 'fwd_sdf_use_h_appear', False)
    fwd_sdf_use_view_dirs = getattr(model, 'fwd_sdf_use_view_dirs', False)
    use_pix = (getattr(model, 'use_pix', False) and with_rgb) or fwd_sdf_use_pix
    use_h_appear = (getattr(model, 'use_h_appear', False) and with_rgb) or fwd_sdf_use_h_appear
    use_view_dirs = (getattr(model, 'use_view_dirs', False) and with_rgb) or fwd_sdf_use_view_dirs

    empty_volume_buffer = dict(type='empty', rays_inds_hit=[])
    if (num_rays:=ray_tested['num_rays']) == 0:
        return empty_volume_buffer, {}
    
    if isinstance(num_fine, int):
        num_fine = [num_fine] * len(upsample_inv_s_factors)
    assert len(num_fine) == len(upsample_inv_s_factors), \
        f"num_fine should be of the same length={len(upsample_inv_s_factors)} with upsample"
    num_fine = [n // 2 * 2 + 1 for n in num_fine] # Had better always be odd

    upsample_inv_s /= upsample_s_divisor
    forward_inv_s = model.forward_inv_s() if forward_inv_s is None else forward_inv_s


    # NOTE: Normalized rays in network's space
    rays_o, rays_d, near, far, rays_inds = itemgetter('rays_o', 'rays_d', 'near', 'far', 'rays_inds')(ray_tested)
    assert (rays_o.dim() == 2) and (rays_d.dim()==2)
    
    # NOTE: Extra infos attached to each ray
    if use_pix: rays_pix = ray_tested['rays_pix']
    if use_ts: rays_ts = ray_tested['rays_ts']
    if use_fidx: rays_fidx = ray_tested['rays_fidx']
    if use_bidx: rays_bidx = ray_tested['rays_bidx']
    if use_h_appear: rays_h_appear = ray_tested['rays_h_appear']
    
    # NOTE: The device & dtype of output
    device, dtype = rays_o.device, rays_o.dtype
    
    # NOTE: The spatial length scale on each ray caused by scaling rays_d 
    dir_scale = rays_d.data.norm(dim=-1)  # [num_rays]
    # NOTE: The normalized ray direction vector in network's space
    view_dirs = rays_d / dir_scale.clamp_min_(1.0e-10).unsqueeze(-1) # [num_rays, 3]
    
    #----------------
    # Coarse sampling
    #----------------
    if num_coarse > 0:
        coarse_step_cfg = coarse_step_cfg.copy()
        step_mode = coarse_step_cfg.pop('step_mode')
        if step_mode == 'linear':
            depths_coarse_1, deltas_coarse_1 = batch_sample_step_linear(near, far, num_coarse+1, perturb=perturb, return_dt=True, **coarse_step_cfg)
        elif step_mode == 'depth':
            depths_coarse_1, deltas_coarse_1 = batch_sample_step_wrt_depth(near, far, num_coarse+1, perturb=perturb, return_dt=True, **coarse_step_cfg)
        elif step_mode == 'sqrt_depth':
            depths_coarse_1, deltas_coarse_1 = batch_sample_step_wrt_sqrt_depth(near, far, num_coarse+1, perturb=perturb, return_dt=True, **coarse_step_cfg)
        else:
            raise RuntimeError(f"Invalid step_mode={step_mode}")
        deltas_coarse = deltas_coarse_1[..., :num_coarse]
        depths_coarse = depths_coarse_1[..., :num_coarse] + deltas_coarse / 2.
    
    #----------------
    # Ray marching
    #----------------
    with profile("Ray marching"):
        accel = model.accel
        if isinstance(accel, accel_types_single):
            marched = accel.ray_march(
                rays_o, rays_d, 
                near=near, far=far, perturb=perturb, **march_cfg)
        elif isinstance(accel, accel_types_dynamic):
            marched = accel.ray_march(
                rays_o, rays_d, ray_tested['rays_ts'], 
                near=near, far=far, perturb=perturb, **march_cfg)
        elif isinstance(accel, accel_types_batched):
            marched = accel.cur_batch__ray_march(
                rays_o, rays_d, ray_tested['rays_bidx'], 
                near=near, far=far, perturb=perturb, **march_cfg)
        elif isinstance(accel, accel_types_batched_dynamic):
            marched = accel.cur_batch__ray_march(
                rays_o, rays_d, ray_tested['rays_bidx'], ray_tested['rays_ts'], 
                near=near, far=far, perturb=perturb, **march_cfg)
        else:
            raise RuntimeError(f"Unsupported accel of type={type(accel)}")

    #----------------
    # Upsampling & compute volume_buffer
    #----------------
    if marched.ridx_hit is not None:
        with profile("Upsampling"):
            pack_infos = marched.pack_infos.clone()
            depth_samples = marched.depth_samples
            # [num_rays_hit, 1, 3]
            rays_inds_hit = rays_inds[marched.ridx_hit]
            rays_o_hit = rays_o[marched.ridx_hit].unsqueeze(-2)
            rays_d_hit = rays_d[marched.ridx_hit].unsqueeze(-2)
            if use_ts: rays_ts_hit = rays_ts[marched.ridx_hit]
            if use_fidx: rays_fidx_hit = rays_fidx[marched.ridx_hit]
            if use_bidx: rays_bidx_hit = rays_bidx[marched.ridx_hit]
            if use_pix: rays_pix_hit = rays_pix[marched.ridx_hit]
            if use_h_appear: rays_h_appear_hit = rays_h_appear[marched.ridx_hit]
            if use_view_dirs: view_dirs_hit = view_dirs[marched.ridx_hit]
            
            #----------------
            # Upsample on marched samples
            with torch.no_grad():
                fwd_kwargs = dict(x=marched.samples)
                if use_ts: fwd_kwargs['ts'] = rays_ts[marched.ridx]
                if use_fidx: fwd_kwargs['fidx'] = rays_fidx[marched.ridx]
                if use_bidx: fwd_kwargs['bidx'] = rays_bidx[marched.ridx]
                if fwd_sdf_use_pix: fwd_kwargs['pix'] = rays_pix[marched.ridx]
                if fwd_sdf_use_h_appear: fwd_kwargs['h_appear'] = rays_h_appear[marched.ridx]
                if fwd_sdf_use_view_dirs: fwd_kwargs['v'] = view_dirs[marched.ridx]
                sdf = batchify_query(lambda **kw: model.forward_sdf(**kw)['sdf'], **fwd_kwargs, chunk=chunksize_query)
                
                depths_1 = []
                
                for i, factor in enumerate(upsample_inv_s_factors):
                    pinfo_fine_per_iter = get_pack_infos_from_batch(marched.num_hit_rays, num_fine[i], device=device)
                    fwd_extra_kwargs_pter_iter = dict()
                    if use_ts: fwd_extra_kwargs_pter_iter['ts'] = rays_ts_hit.unsqueeze(-1).expand(marched.num_hit_rays, num_fine[i]).flatten(0,-1).contiguous()
                    if use_fidx: fwd_extra_kwargs_pter_iter['fidx'] = rays_fidx_hit.unsqueeze(-1).expand(marched.num_hit_rays, num_fine[i]).flatten(0,-1).contiguous()
                    if use_bidx: fwd_extra_kwargs_pter_iter['bidx'] = rays_bidx_hit.unsqueeze(-1).expand(marched.num_hit_rays, num_fine[i]).flatten(0,-1).contiguous()
                    if fwd_sdf_use_pix: fwd_extra_kwargs_pter_iter['pix'] = rays_pix_hit.unsqueeze(-2).expand(marched.num_hit_rays, num_fine[i], -1).flatten(0,-2).contiguous()
                    if fwd_sdf_use_h_appear: fwd_extra_kwargs_pter_iter['h_appear'] = rays_h_appear_hit.unsqueeze(-2).expand(marched.num_hit_rays, num_fine[i], -1).flatten(0,-2).contiguous()
                    if fwd_sdf_use_view_dirs: fwd_extra_kwargs_pter_iter['v'] = view_dirs_hit.unsqueeze(-2).expand(marched.num_hit_rays, num_fine[i], -1).flatten(0,-2).contiguous()
                    
                    if upsample_use_estimate_alpha:
                        alpha = neus_packed_sdf_to_upsample_alpha(sdf, depth_samples, upsample_inv_s * factor, pack_infos) # This could leads to artifacts
                    else:
                        alpha = neus_packed_sdf_to_alpha(sdf, upsample_inv_s * factor, pack_infos)
                    
                    vw = packed_alpha_to_vw(alpha, pack_infos)
                    
                    neus_cdf = packed_cumsum(vw, pack_infos, exclusive=True)
                    last_cdf = neus_cdf[pack_infos[...,0] + pack_infos[...,1] - 1]
                    neus_cdf = packed_div(neus_cdf, last_cdf.clamp_min(1e-5), pack_infos)
                    depths_fine_iter = packed_sample_cdf(depth_samples, neus_cdf.to(depth_samples.dtype), pack_infos, num_fine[i], perturb=perturb)[0]
                    depths_1.append(depths_fine_iter)
                    
                    if len(upsample_inv_s_factors) > 1:
                        # 273 us @ 930k + 25k
                        # Merge fine samples of current upsample iter to previous packed buffer.
                        # NOTE: The new `pack_infos` is calculated here.
                        pidx0, pidx1, pack_infos = merge_two_packs_sorted_aligned(
                            depth_samples, pack_infos, depths_fine_iter.flatten(), pinfo_fine_per_iter, 
                            b_sorted=True, return_val=False)
                        num_samples_iter = depth_samples.numel()
                        depth_samples_iter = depth_samples.new_empty([num_samples_iter + depths_fine_iter.numel()])
                        depth_samples_iter[pidx0], depth_samples_iter[pidx1] = depth_samples, depths_fine_iter.flatten()
                        depth_samples = depth_samples_iter

                        if i < len(upsample_inv_s_factors)-1:
                            x_fine = torch.addcmul(rays_o_hit, rays_d_hit, depths_fine_iter.unsqueeze(-1))
                            sdf_iter = sdf.new_empty([num_samples_iter + depths_fine_iter.numel()])
                            sdf_iter[pidx0], sdf_iter[pidx1] = sdf, model.forward_sdf(x_fine.flatten(0, -2), **fwd_extra_kwargs_pter_iter)['sdf']
                            sdf = sdf_iter
                
                if len(upsample_inv_s_factors) > 1:
                    depths_1 = torch.cat(depths_1, dim=-1).sort(dim=-1).values
                else:
                    depths_1 = depths_1[0]

        #----------------
        # Acquire volume_buffer via quering network and gather results
        with profile("Acquire volume buffer"):
            if num_coarse == 0: # num_coarse == 0 and ridx_hit is not None
                # [num_rays_hit, num_fine_all]
                fwd_kwargs = dict(x=torch.addcmul(rays_o_hit, rays_d_hit, depths_1.unsqueeze(-1)))
                if use_ts: fwd_kwargs['ts'] = rays_ts_hit.unsqueeze(-1).expand(marched.num_hit_rays, depths_1.size(-1)).contiguous()
                if use_fidx: fwd_kwargs['fidx'] = rays_fidx_hit.unsqueeze(-1).expand(marched.num_hit_rays, depths_1.size(-1)).contiguous()
                if use_bidx: fwd_kwargs['bidx'] = rays_bidx_hit.unsqueeze(-1).expand(marched.num_hit_rays, depths_1.size(-1)).contiguous()
                if fwd_sdf_use_pix: fwd_kwargs['pix'] = rays_pix_hit.unsqueeze(-2).expand(marched.num_hit_rays, depths_1.size(-1), -1).contiguous()
                if fwd_sdf_use_h_appear: fwd_kwargs['h_appear'] = rays_h_appear_hit.unsqueeze(-2).expand(marched.num_hit_rays, depths_1.size(-1), -1).contiguous()
                if fwd_sdf_use_view_dirs: fwd_kwargs['v'] = view_dirs_hit.unsqueeze(-2).expand(marched.num_hit_rays, depths_1.size(-1), -1).contiguous()
                
                if model.training:
                    alpha = neus_ray_sdf_to_alpha(model.forward_sdf(**fwd_kwargs)['sdf'], forward_inv_s)
                else:
                    alpha = neus_ray_sdf_to_alpha(
                        batchify_query(lambda **kw: model.forward_sdf(**kw)['sdf'], **fwd_kwargs, chunk=chunksize_query), 
                        forward_inv_s)
                
                depths = (depths_1[..., :-1] + depths_1.diff(dim=-1)/2.)
                # NOTE: `pack_infos` is for all `marched` rays / `rays_inds_hit` / `rays_inds[marched.ridx_hit]` / `marched.num_hit_rays`
                pack_infos = get_pack_infos_from_batch(marched.num_hit_rays, depths.size(-1), device=device)
                nidx_useful, pack_infos_useful, pidx_useful = packed_volume_render_compression(alpha.flatten(), pack_infos)
                
                if nidx_useful.numel() == 0:
                    return empty_volume_buffer, {}
                else:
                    depths_packed, alpha_packed = depths.flatten()[pidx_useful], alpha.flatten()[pidx_useful]
                    
                    volume_buffer = dict(
                        type='packed', 
                        rays_inds_hit=rays_inds_hit[nidx_useful], 
                        pack_infos_hit=pack_infos_useful, 
                        t=depths_packed.to(dtype), 
                        opacity_alpha=alpha_packed.to(dtype))
                    if use_bidx: volume_buffer['rays_bidx_hit'] = rays_bidx_hit[nidx_useful]

                    if with_rgb or with_normal:
                        ridx_all = marched.ridx_hit.unsqueeze(-1).expand(marched.num_hit_rays, depths.size(-1)).flatten()[pidx_useful]
                        # Basic inputs: 
                        fwd_kwargs = dict(
                            x=torch.addcmul(rays_o[ridx_all], rays_d[ridx_all], depths_packed.unsqueeze(-1)), 
                            nablas_has_grad=nablas_has_grad, with_rgb=with_rgb, with_normal=with_normal
                        )
                        # Extra infos attached to each pts
                        if use_ts: fwd_kwargs['ts'] = rays_ts[ridx_all]
                        if use_fidx: fwd_kwargs['fidx'] = rays_fidx[ridx_all]
                        if use_bidx: fwd_kwargs['bidx'] = rays_bidx[ridx_all]
                        if use_pix: fwd_kwargs['pix'] = rays_pix[ridx_all]
                        if use_h_appear: fwd_kwargs['h_appear'] = rays_h_appear[ridx_all]
                        if use_view_dirs: fwd_kwargs['v'] = view_dirs[ridx_all]
                        #----------- Net forward
                        net_out = model.forward(**fwd_kwargs)
                        volume_buffer['net_x'] = fwd_kwargs['x']
                        if "nablas" in net_out: volume_buffer["nablas"] = net_out["nablas"].to(dtype)
                        if "rgb" in net_out: volume_buffer["rgb"] = net_out["rgb"].to(dtype)
                    
                    details = {'march.num_per_ray': marched.pack_infos[:, 1], 
                                'render.num_per_ray0': depths.size(-1), 
                                'render.num_per_ray': pack_infos_useful[:, 1]}
                    return volume_buffer, details

            else: # num_coarse != 0 and ridx_hit is not None
                def merge():
                    ridx_hit_coarse = torch.arange(rays_inds.numel(), device=device)
                    pidx0, pidx1, pack_infos = merge_two_batch_a_includes_b(depths_coarse_1, ridx_hit_coarse, depths_1, marched.ridx_hit, a_sorted=True)
                    num_samples = depths_1.numel() + depths_coarse_1.numel()
                    depths_1_packed = depths_1.new_zeros([num_samples])
                    ridx_all = marched.ridx_hit.new_zeros([num_samples])
                    ridx_all[pidx0], ridx_all[pidx1] = ridx_hit_coarse.unsqueeze(-1), marched.ridx_hit.unsqueeze(-1)
                    depths_1_packed[pidx0], depths_1_packed[pidx1] = depths_coarse_1, depths_1
                    return ridx_all, depths_1_packed, pack_infos
                
                # NOTE: `pack_infos` here is for all `ray_tested` rays / `ray_inds` / `num_rays`
                ridx_all, depths_1_packed, pack_infos = merge()
                depths_packed = depths_1_packed + packed_diff(depths_1_packed, pack_infos) / 2.
                
                fwd_kwargs = dict(x=torch.addcmul(rays_o[ridx_all], rays_d[ridx_all], depths_1_packed.unsqueeze(-1)))
                if use_ts: fwd_kwargs['ts'] = rays_ts[ridx_all]
                if use_fidx: fwd_kwargs['fidx'] = rays_fidx[ridx_all]
                if use_bidx: fwd_kwargs['bidx'] = rays_bidx[ridx_all]
                if fwd_sdf_use_pix: fwd_kwargs['pix'] = rays_pix_hit[ridx_all]
                if fwd_sdf_use_h_appear: fwd_kwargs['h_appear'] = rays_h_appear_hit[ridx_all]
                if fwd_sdf_use_view_dirs: fwd_kwargs['v'] = view_dirs_hit[ridx_all]
                
                if model.training:
                    alpha_packed = neus_packed_sdf_to_alpha(model.forward_sdf(**fwd_kwargs)['sdf'], forward_inv_s, pack_infos)
                else:
                    alpha_packed = neus_packed_sdf_to_alpha(
                        batchify_query(lambda **kw: model.forward_sdf(**kw)['sdf'], **fwd_kwargs, chunk=chunksize_query), 
                        forward_inv_s, pack_infos)
                nidx_useful, pack_infos_useful, pidx_useful = packed_volume_render_compression(alpha_packed, pack_infos)
                
                if nidx_useful.numel() == 0:
                    return empty_volume_buffer, {}
                else:
                    # Update
                    ridx_all, depths_packed, alpha_packed  = ridx_all[pidx_useful], depths_packed[pidx_useful], alpha_packed[pidx_useful]
                    
                    volume_buffer = dict(
                        type='packed', 
                        rays_inds_hit=rays_inds[nidx_useful], 
                        pack_infos_hit=pack_infos_useful, 
                        t=depths_packed.to(dtype), 
                        opacity_alpha=alpha_packed.to(dtype))
                    if use_bidx: volume_buffer['rays_bidx_hit'] = rays_bidx[nidx_useful]
                    
                    if with_rgb or with_normal:
                        # Basic inputs: 
                        fwd_kwargs = dict(
                            x=torch.addcmul(rays_o[ridx_all], rays_d[ridx_all], depths_packed.unsqueeze(-1)), 
                            nablas_has_grad=nablas_has_grad, with_rgb=with_rgb, with_normal=with_normal
                        )
                        # Extra infos attached to each ray
                        if use_ts: fwd_kwargs['ts'] = rays_ts[ridx_all]
                        if use_fidx: fwd_kwargs['fidx'] = rays_fidx[ridx_all]
                        if use_bidx: fwd_kwargs['bidx'] = rays_bidx[ridx_all]
                        if use_pix: fwd_kwargs['pix'] = rays_pix[ridx_all]
                        if use_h_appear: fwd_kwargs['h_appear'] = rays_h_appear[ridx_all]
                        if use_view_dirs: fwd_kwargs['v'] = view_dirs[ridx_all]
                        #----------- Net forward
                        net_out = model.forward(**fwd_kwargs)
                        volume_buffer['net_x'] = fwd_kwargs['x']
                        if "nablas" in net_out: volume_buffer["nablas"] = net_out["nablas"].to(dtype)
                        if "rgb" in net_out: volume_buffer["rgb"] = net_out["rgb"].to(dtype)
                    details = {'march.num_per_ray': marched.pack_infos[:, 1], 
                                'render.num_per_ray0': pack_infos[:, 1], 
                                'render.num_per_ray': pack_infos_useful[:, 1]}
                    return volume_buffer, details
    
    else: # ridx_hit is None
        if num_coarse == 0: # ridx_hit is None and num_coarse == 0
            return empty_volume_buffer, {}
        
        else: # ridx_hit is None and num_coarse > 0
            with profile("Acquire volume buffer"):
                fwd_kwargs = dict(x=torch.addcmul(rays_o.unsqueeze(-2), rays_d.unsqueeze(-2), depths_coarse_1.unsqueeze(-1)))
                if use_ts: fwd_kwargs['ts'] = rays_ts.unsqueeze(-1).expand(*depths_coarse_1.shape).contiguous()
                if use_bidx: fwd_kwargs['bidx'] = rays_bidx.unsqueeze(-1).expand(*depths_coarse_1.shape).contiguous()
                if use_fidx: fwd_kwargs['fidx'] = rays_fidx.unsqueeze(-1).expand(*depths_coarse_1.shape).contiguous()
                if fwd_sdf_use_pix: fwd_kwargs['pix'] = rays_pix.unsqueeze(-2).expand(*depths_coarse_1.shape, -1).contiguous()
                if fwd_sdf_use_h_appear: fwd_kwargs['h_appear'] = rays_h_appear.unsqueeze(-2).expand(*depths_coarse_1.shape, -1).contiguous()
                if fwd_sdf_use_view_dirs: fwd_kwargs['v'] = view_dirs.unsqueeze(-2).expand(*depths_coarse_1.shape, -1).contiguous()
                if model.training:
                    alpha_coarse = neus_ray_sdf_to_alpha(model.forward_sdf(**fwd_kwargs)['sdf'], forward_inv_s)
                else:
                    alpha_coarse = neus_ray_sdf_to_alpha(
                        batchify_query(lambda **kw: model.forward_sdf(**kw)['sdf'], **fwd_kwargs, chunk=chunksize_query), 
                        forward_inv_s)
                depths_coarse = depths_coarse_1[..., :num_coarse] + deltas_coarse_1[..., :num_coarse] / 2.
                
                pack_infos_coarse = get_pack_infos_from_batch(num_rays, depths_coarse.size(-1), device=device)
                nidx_useful, pack_infos_useful, pidx_useful = packed_volume_render_compression(alpha_coarse.flatten(), pack_infos_coarse)
                
                if nidx_useful.numel() == 0:
                    return empty_volume_buffer, {}
                else:
                    depths_packed, alpha_packed = depths_coarse.flatten()[pidx_useful], alpha_coarse.flatten()[pidx_useful]
                    
                    volume_buffer = dict(
                        type='packed', 
                        rays_inds_hit=rays_inds[nidx_useful], 
                        pack_infos_hit=pack_infos_useful, 
                        t=depths_packed.to(dtype), 
                        opacity_alpha=alpha_packed.to(dtype))
                    if use_bidx: volume_buffer['rays_bidx_hit'] = rays_bidx[nidx_useful]
                    
                    if with_rgb or with_normal:
                        ridx_all = torch.arange(rays_inds.numel(), device=device).unsqueeze_(-1).expand_as(depths_coarse).flatten()[pidx_useful]
                        # Basic inputs: 
                        fwd_kwargs = dict(
                            x=torch.addcmul(rays_o[ridx_all], rays_d[ridx_all], depths_packed.unsqueeze(-1)), 
                            nablas_has_grad=nablas_has_grad, with_rgb=with_rgb, with_normal=with_normal
                        )
                        # Extra infos attached to each ray
                        if use_ts: fwd_kwargs['ts'] = rays_ts[ridx_all]
                        if use_fidx: fwd_kwargs['fidx'] = rays_fidx[ridx_all]
                        if use_bidx: fwd_kwargs['bidx'] = rays_bidx[ridx_all]
                        if use_pix: fwd_kwargs['pix'] = rays_pix[ridx_all]
                        if use_h_appear: fwd_kwargs['h_appear'] = rays_h_appear[ridx_all]
                        if use_view_dirs: fwd_kwargs['v'] = view_dirs[ridx_all]
                        #----------- Net forward
                        net_out = model.forward(**fwd_kwargs)
                        volume_buffer['net_x'] = fwd_kwargs['x']
                        if "nablas" in net_out: volume_buffer["nablas"] = net_out["nablas"].to(dtype)
                        if "rgb" in net_out: volume_buffer["rgb"] = net_out["rgb"].to(dtype)
                    details = {'render.num_per_ray0': depths_coarse.size(-1), 
                                'render.num_per_ray': pack_infos_useful[:, 1]}
                    return volume_buffer, details

def neus_ray_query_march_occ_multi_upsample_compressed_strategy(
    model, ray_tested: Dict[str, torch.Tensor], 
    # Common params
    with_rgb: bool = True, with_normal: bool = True, 
    perturb: bool = False, nablas_has_grad: bool = False, forward_inv_s: float = None, 
    # Distinct params
    num_coarse_max: int = 32, num_coarse_min: int = 0, num_coarse_anneal_type: str = 'linear', 
    coarse_step_cfg = dict(step_mode='linear'), 
    chunksize_query: int = 2**24,  march_cfg = dict(), 
    upsample_inv_s: float = 64., upsample_s_factors_full: List[int] = [1, 4, 16], upsample_use_estimate_alpha=False
    ) -> Tuple[dict, dict]:
    # TODO: Using learned `forward_inv_s` directly for control causes oscillations
    forward_inv_s = model.forward_inv_s() if forward_inv_s is None else forward_inv_s
    raise NotImplementedError
    # if forward_inv_s <= 200:
    #     num_coarse = 128
    #     num_fine = 16
    #     upsample_inv_s_factors = [1.]
    # elif forward_inv_s <= 400:
    #     num_coarse = 64
    #     num_fine = 16
    #     upsample_inv_s_factors = [1., 4.]
    # elif forward_inv_s <= 800:
    #     num_coarse = 32
    #     num_fine = 16
    #     upsample_inv_s_factors = [1., 4., 16.]
    # else:
    #     num_coarse = 16
    #     num_fine = 8
    #     upsample_inv_s_factors = [1., 4., 16.]
    # return self._ray_query_march_occ_multi_upsample_compressed(
    #     ray_tested, with_rgb=with_rgb, with_normal=with_normal,
    #     perturb=perturb, nablas_has_grad=nablas_has_grad, forward_inv_s=forward_inv_s, 
    #     num_coarse=num_coarse, coarse_step_cfg=coarse_step_cfg, 
    #     chunksize_query=chunksize_query, march_cfg=march_cfg, 
    #     num_fine=num_fine, upsample_inv_s=upsample_inv_s, upsample_inv_s_factors=upsample_inv_s_factors, 
    #     upsample_use_estimate_alpha=upsample_use_estimate_alpha
    # )

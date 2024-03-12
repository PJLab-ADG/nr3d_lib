
__all__ = [
    'nerf_ray_query_march_occ', 
    'nerf_ray_query_march_occ_multi_upsample_compressed', 
]

import os
from operator import itemgetter
from typing import Dict, List, Tuple

import torch

from nr3d_lib.profile import profile

from nr3d_lib.models.utils import batchify_query
from nr3d_lib.graphics.raysample import batch_sample_pdf, batch_sample_step_wrt_depth, \
    batch_sample_step_linear, batch_sample_step_wrt_sqrt_depth, packed_sample_cdf
from nr3d_lib.graphics.pack_ops import get_pack_infos_from_batch, merge_two_batch_a_includes_b, \
    packed_cumsum, merge_two_packs_sorted_a_includes_b, packed_diff, packed_div, packed_sum
from nr3d_lib.models.accelerations import \
    accel_types_single, accel_types_dynamic, accel_types_batched, accel_types_batched_dynamic
from nr3d_lib.graphics.nerf import packed_alpha_to_vw, ray_alpha_to_vw, \
    packed_volume_render_compression
from nr3d_lib.graphics.raysample import batch_sample_pdf, batch_sample_step_wrt_depth, \
    batch_sample_step_linear, batch_sample_step_wrt_sqrt_depth, packed_sample_cdf

from nr3d_lib.graphics.nerf import packed_volume_render_compression, tau_to_alpha

def nerf_ray_query_march_occ(
    model, ray_tested: Dict[str, torch.Tensor], 
    with_rgb: bool = True, 
    perturb: bool = False, march_cfg=dict(), 
    compression=True, 
    bypass_sigma_fn=None, 
    bypass_alpha_fn=None, 
    forward_params: dict = {}, 
    ) -> Tuple[dict, dict]:

    assert hasattr(model, 'forward'), "model.forward() is requried"
    assert getattr(model, 'accel', None) is not None, "model.accel is required"
    if not with_rgb:
        assert hasattr(model, 'forward_density'), "model.forward_density() is requried"
    if compression:
        assert hasattr(model, 'query_density'), "model.query_density() is requried"

    use_ts = getattr(model, 'use_ts', False)
    use_fidx = getattr(model, 'use_fidx', False)
    use_bidx = getattr(model, 'use_bidx', False)
    fwd_density_use_pix = getattr(model, 'fwd_density_use_pix', False)
    fwd_density_use_h_appear = getattr(model, 'fwd_density_use_h_appear', False)
    fwd_density_use_view_dirs = getattr(model, 'fwd_density_use_view_dirs', False)
    use_pix = (getattr(model, 'use_pix', False) and with_rgb) or fwd_density_use_pix
    use_h_appear = (getattr(model, 'use_h_appear', False) and with_rgb) or fwd_density_use_h_appear
    use_view_dirs = (getattr(model, 'use_view_dirs', False) and with_rgb) or fwd_density_use_view_dirs
    
    empty_volume_buffer = dict(type='empty', rays_inds_hit=[])
    if (num_rays:=ray_tested['num_rays']) == 0:
        return empty_volume_buffer, {}
    
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

    if marched.ridx_hit is None:
        return empty_volume_buffer, {}

    if compression:
        #----------------
        # Calc compact samples based on rendered visibility
        with torch.no_grad():
            fwd_kwargs = dict(x=marched.samples)
            if use_ts: fwd_kwargs['ts'] = rays_ts[marched.ridx]
            if use_fidx: fwd_kwargs['fidx'] = rays_fidx[marched.ridx]
            if use_bidx: fwd_kwargs['bidx'] = rays_bidx[marched.ridx]
            if fwd_density_use_pix: fwd_kwargs['pix'] = rays_pix[marched.ridx]
            if fwd_density_use_h_appear: fwd_kwargs['h_appear'] = rays_h_appear[marched.ridx]
            if fwd_density_use_view_dirs: fwd_kwargs['v'] = view_dirs[marched.ridx]
            assert int(bypass_sigma_fn is not None) + int(bypass_alpha_fn is not None) <= 1, "Please pass at most one of bypass_sigma_fn or bypass_alpha_fn"
            if bypass_sigma_fn is not None:
                sigmas = bypass_sigma_fn(**fwd_kwargs)
                alphas = tau_to_alpha(sigmas * marched.deltas)
            elif bypass_alpha_fn is not None:
                alphas = bypass_alpha_fn(**fwd_kwargs)
            else:
                sigmas = model.query_density(**fwd_kwargs)
                alphas = tau_to_alpha(sigmas * marched.deltas)
        
            old_pack_infos = marched.pack_infos.clone()
            nidx_useful, pack_infos_hit_useful, pidx_useful = packed_volume_render_compression(alphas, marched.pack_infos)
        
        if nidx_useful.numel() == 0:
            return empty_volume_buffer, {}
        
        pack_infos = pack_infos_hit_useful
        ridx_hit = marched.ridx_hit[nidx_useful]
        ridx_all = marched.ridx[pidx_useful]
        depth_samples = marched.depth_samples[pidx_useful]
        deltas = marched.deltas[pidx_useful]
        samples = marched.samples[pidx_useful]
    else:
        pack_infos = marched.pack_infos
        ridx_hit = marched.ridx_hit
        ridx_all = marched.ridx
        depth_samples = marched.depth_samples
        deltas = marched.deltas
        samples = marched.samples
    
    #----------------
    # Gather volume_buffer
    #----------------
    volume_buffer = dict(
        type='packed', 
        rays_inds_hit=rays_inds[ridx_hit], 
        pack_infos_hit=pack_infos, 
        t=depth_samples.to(dtype))
    if use_bidx: volume_buffer['rays_bidx_hit'] = rays_bidx[nidx_useful]
    if compression:
        details = {
            'march.num_per_ray': old_pack_infos[:,1], 
            'render.num_per_ray0': old_pack_infos[:,1], 
            'render.num_per_ray': pack_infos[:,1]
        }
    else:
        details = {
            'march.num_per_ray': pack_infos[:,1], 
            'render.num_per_ray': pack_infos[:,1], 
        }

    fwd_kwargs = dict(x=samples)
    if use_ts: fwd_kwargs['ts'] = rays_ts[ridx_all]
    if use_fidx: fwd_kwargs['fidx'] = rays_fidx[ridx_all]
    if use_bidx: fwd_kwargs['bidx'] = rays_bidx[ridx_all]
    if use_pix: fwd_kwargs['pix'] = rays_pix[ridx_all]
    if use_h_appear: fwd_kwargs['h_appear'] = rays_h_appear[ridx_all]
    if use_view_dirs: fwd_kwargs['v'] = view_dirs[ridx_all]
    if with_rgb:
        net_out = model.forward(**fwd_kwargs, **forward_params)
        volume_buffer["rgb"] = net_out["rgb"].to(dtype)
    else:
        net_out = model.forward_density(**fwd_kwargs, **forward_params)
    
    volume_buffer["deltas"] = deltas.to(dtype)
    volume_buffer["sigma"] = net_out["sigma"].to(dtype)
    volume_buffer["opacity_alpha"] = tau_to_alpha(volume_buffer["sigma"] * volume_buffer["deltas"])
    for k in ['flow_fwd', 'flow_fwd_pred_bwd', 'flow_bwd', 'flow_bwd_pred_fwd']\
        + ['sigma_static', 'rgb_static', 'sigma_dynamic', 'rgb_dynamic']:
        if k in net_out:
            volume_buffer[k] = net_out[k].to(dtype)
    
    return volume_buffer, details

def nerf_ray_query_march_occ_multi_upsample_compressed(
    model, ray_tested: Dict[str, torch.Tensor], 
    with_rgb: bool = True, 
    perturb: bool = False, 
    num_coarse: int = 32, coarse_step_cfg = dict(step_mode='linear'), 
    num_fine: int = 32, march_cfg=dict(), chunksize_query: int = 2**24,
    combine_marched_and_coarse: bool = True, 
    bypass_sigma_fn=None, 
    bypass_alpha_fn=None, 
    forward_params: dict = {}, 
    ):

    assert hasattr(model, 'forward'), "model.forward() is requried"
    assert getattr(model, 'accel', None) is not None, "model.accel is required"
    if not with_rgb:
        assert hasattr(model, 'forward_density'), "model.forward_density() is requried"
    if True: # Compression
        assert hasattr(model, 'query_density'), "model.query_density() is requried"

    use_ts = getattr(model, 'use_ts', False)
    use_fidx = getattr(model, 'use_fidx', False)
    use_bidx = getattr(model, 'use_bidx', False)
    fwd_density_use_pix = getattr(model, 'fwd_density_use_pix', False)
    fwd_density_use_h_appear = getattr(model, 'fwd_density_use_h_appear', False)
    fwd_density_use_view_dirs = getattr(model, 'fwd_density_use_view_dirs', False)
    use_pix = (getattr(model, 'use_pix', False) and with_rgb) or fwd_density_use_pix
    use_h_appear = (getattr(model, 'use_h_appear', False) and with_rgb) or fwd_density_use_h_appear
    use_view_dirs = (getattr(model, 'use_view_dirs', False) and with_rgb) or fwd_density_use_view_dirs
    
    empty_volume_buffer = dict(type='empty', rays_inds_hit=[])
    if (num_rays:=ray_tested['num_rays']) == 0:
        return empty_volume_buffer, {}
    
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
        ridx_hit_coarse = torch.arange(rays_inds.numel(), device=device)
        pack_infos_coarse = get_pack_infos_from_batch(rays_inds.numel(), num_coarse, device=device)

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
            marched_pack_infos = marched.pack_infos.clone()
            if num_coarse and combine_marched_and_coarse > 0:
                #---- Merge coarse samples and marched samples
                pidx0, pidx1, pack_infos = merge_two_packs_sorted_a_includes_b(
                    depths_coarse.flatten(), pack_infos_coarse, ridx_hit_coarse, 
                    marched.depth_samples, marched.pack_infos, marched.ridx_hit, b_sorted=True)
                num_samples = marched.depth_samples.numel() + depths_coarse.numel()
                depths_all = marched.depth_samples.new_zeros([num_samples])
                ridx_all = marched.ridx_hit.new_zeros([num_samples])
                ridx_all[pidx0], ridx_all[pidx1] = ridx_hit_coarse.unsqueeze(-1).expand(-1, num_coarse).flatten(), marched.ridx
                depths_all[pidx0], depths_all[pidx1] = depths_coarse.flatten(), marched.depth_samples
                samples_all = torch.addcmul(rays_o[ridx_all], rays_d[ridx_all], depths_all.unsqueeze(-1))
                deltas_all = packed_diff(depths_all, pack_infos)
                ridx_hit = ridx_hit_coarse
            else:
                samples_all = marched.samples
                depths_all = marched.depth_samples
                ridx_all = marched.ridx
                deltas_all = marched.deltas
                
                pack_infos = marched.pack_infos
                ridx_hit = marched.ridx_hit
            
            # [num_rays_hit, 1, 3]
            rays_inds_hit = rays_inds[ridx_hit]
            if use_ts: rays_ts_hit = rays_ts[ridx_hit]
            if use_fidx: rays_fidx_hit = rays_fidx[ridx_hit]
            if use_bidx: rays_bidx_hit = rays_bidx[ridx_hit]
            if use_pix: rays_pix_hit = rays_pix[ridx_hit]
            if use_h_appear: rays_h_appear_hit = rays_h_appear[ridx_hit]
            if use_view_dirs: view_dirs_hit = view_dirs[ridx_hit]
            
            #----------------
            # Upsample on marched samples
            with torch.no_grad():
                fwd_kwargs = dict(x=samples_all)
                if use_ts: fwd_kwargs['ts'] = rays_ts[ridx_all]
                if use_fidx: fwd_kwargs['fidx'] = rays_fidx[ridx_all]
                if use_bidx: fwd_kwargs['bidx'] = rays_bidx[ridx_all]
                if fwd_density_use_pix: fwd_kwargs['pix'] = rays_pix[ridx_all]
                if fwd_density_use_h_appear: fwd_kwargs['h_appear'] = rays_h_appear[ridx_all]
                if fwd_density_use_view_dirs: fwd_kwargs['v'] = view_dirs[ridx_all]
                sigma = batchify_query(lambda **kw: model.query_density(**kw), **fwd_kwargs, chunk=chunksize_query)
                alpha = tau_to_alpha(sigma * deltas_all)
                
                nerf_cdf = packed_cumsum(alpha, pack_infos)
                last_cdf = nerf_cdf[pack_infos[...,0] + pack_infos[...,1] - 1]
                nerf_cdf = packed_div(nerf_cdf, last_cdf.clamp_min(1e-5), pack_infos)
                depths_fine_1 = packed_sample_cdf(depths_all, nerf_cdf.to(depths_all.dtype), pack_infos, num_fine + 1, perturb=perturb)[0]
                
                #---- Merge coarse samples and upsampled-samples
                if num_coarse > 0:
                    ridx_fine_1 = ridx_hit.unsqueeze(-1).expand(depths_fine_1.size(0), depths_fine_1.size(1)).flatten()
                    pack_infos_fine_1 = get_pack_infos_from_batch(
                        depths_fine_1.size(0), depths_fine_1.size(1), device=device)
                    pidx0, pidx1, pack_infos = merge_two_packs_sorted_a_includes_b(
                        depths_coarse.flatten(), pack_infos_coarse, ridx_hit_coarse,
                        depths_fine_1.flatten(), pack_infos_fine_1, ridx_hit, b_sorted=False)
                    num_samples = depths_fine_1.numel() + depths_coarse.numel()
                    depths_all = depths_fine_1.new_zeros([num_samples])
                    _ridx_all = ridx_hit.new_zeros([num_samples])
                    _ridx_all[pidx0], _ridx_all[pidx1] = ridx_hit_coarse.unsqueeze(-1).expand(-1, num_coarse).flatten(), ridx_fine_1
                    ridx_all = _ridx_all
                    # ridx_hit = ridx_hit
                    depths_all[pidx0], depths_all[pidx1] = depths_coarse.flatten(), depths_fine_1.flatten()
                    deltas_all = packed_diff(depths_all, pack_infos)
                else:
                    deltas_all = depths_fine_1.diff(dim=-1)
                    depths_all = depths_fine_1[..., :num_fine] + deltas_all
                    ridx_all = ridx_hit.unsqueeze(-1).expand(depths_fine_1.size(0), num_fine).flatten()
                    # ridx_hit = ridx_hit
                    pack_infos = get_pack_infos_from_batch(depths_fine_1.size(0), num_fine, device=device)
    
    else: # No marched samples, only coarse samples
        deltas_all = deltas_coarse.flatten()
        depths_all = depths_coarse.flatten()
        ridx_hit = ridx_hit_coarse
        ridx_all = ridx_hit_coarse.unsqueeze(-1).expand(num_rays, num_coarse).flatten()
        pack_infos = get_pack_infos_from_batch(num_rays, num_coarse, device=device)

    details = {'render.num_per_ray0': pack_infos[:,1]}
    if marched.ridx_hit is not None:
        details['march.num_per_ray'] = marched_pack_infos[:,1]

    #----------------
    # Acquire volume_buffer via quering network and gather results
    with profile("Acquire volume buffer"):
        fwd_kwargs = dict(x=torch.addcmul(rays_o[ridx_all], rays_d[ridx_all], depths_all.unsqueeze(-1)))
        if use_ts: fwd_kwargs['ts'] = rays_ts[ridx_all]
        if use_fidx: fwd_kwargs['fidx'] = rays_fidx[ridx_all]
        if use_bidx: fwd_kwargs['bidx'] = rays_bidx[ridx_all]
        if fwd_density_use_pix: fwd_kwargs['pix'] = rays_pix_hit[ridx_all]
        if fwd_density_use_h_appear: fwd_kwargs['h_appear'] = rays_h_appear_hit[ridx_all]
        if fwd_density_use_view_dirs: fwd_kwargs['v'] = view_dirs_hit[ridx_all]
        
        assert int(bypass_sigma_fn is not None) + int(bypass_alpha_fn is not None) <= 1, "Please pass at most one of bypass_sigma_fn or bypass_alpha_fn"
        if bypass_sigma_fn is not None:
            sigmas = bypass_sigma_fn(**fwd_kwargs)
            alphas = tau_to_alpha(sigmas * deltas_all)
        elif bypass_alpha_fn is not None:
            alphas = bypass_alpha_fn(**fwd_kwargs)
        else:
            sigmas = model.query_density(**fwd_kwargs)
            alphas = tau_to_alpha(sigmas * deltas_all)
        
        #----------------
        # Calc compact samples based on rendered visibility
        nidx_useful, pack_infos_hit_useful, pidx_useful = packed_volume_render_compression(alphas, pack_infos)

        if nidx_useful.numel() == 0:
            return empty_volume_buffer, {}
        
        pack_infos = pack_infos_hit_useful
        ridx_hit = ridx_hit[nidx_useful]
        ridx_all = ridx_all[pidx_useful]
        depths_all = depths_all[pidx_useful]
        deltas_all = deltas_all[pidx_useful]

    details['render.num_per_ray'] = pack_infos[:,1]

    #----------------
    # Gather volume_buffer
    #----------------
    volume_buffer = dict(
        type='packed', 
        rays_inds_hit=rays_inds[ridx_hit], 
        pack_infos_hit=pack_infos, 
        t=depths_all.to(dtype))
    if use_bidx: volume_buffer['rays_bidx_hit'] = rays_bidx[nidx_useful]

    fwd_kwargs = dict(x=torch.addcmul(rays_o[ridx_all], rays_d[ridx_all], depths_all.unsqueeze(-1)))
    if use_ts: fwd_kwargs['ts'] = rays_ts[ridx_all]
    if use_fidx: fwd_kwargs['fidx'] = rays_fidx[ridx_all]
    if use_bidx: fwd_kwargs['bidx'] = rays_bidx[ridx_all]
    if use_pix: fwd_kwargs['pix'] = rays_pix[ridx_all]
    if use_h_appear: fwd_kwargs['h_appear'] = rays_h_appear[ridx_all]
    if use_view_dirs: fwd_kwargs['v'] = view_dirs[ridx_all]
    if with_rgb:
        net_out = model.forward(**fwd_kwargs, **forward_params)
        volume_buffer["rgb"] = net_out["rgb"].to(dtype)
    else:
        net_out = model.forward_density(**fwd_kwargs, **forward_params)
    
    volume_buffer["deltas"] = deltas_all.to(dtype)
    volume_buffer["sigma"] = net_out["sigma"].to(dtype)
    volume_buffer["opacity_alpha"] = tau_to_alpha(volume_buffer["sigma"] * volume_buffer["deltas"])
    
    for k in ['flow_fwd', 'flow_fwd_pred_bwd', 'flow_bwd', 'flow_bwd_pred_fwd']\
        + ['sigma_static', 'rgb_static', 'sigma_dynamic', 'rgb_dynamic']:
        if k in net_out:
            volume_buffer[k] = net_out[k].to(dtype)
    
    return volume_buffer, details

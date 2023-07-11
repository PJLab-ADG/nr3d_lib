"""
@file   renderer_mixin.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Rendering mixin for NeuS with acceleration.

NOTE: Different NeuS implementations:

In the original NeuS implementation, for each network query point, \
    its alpha-value is calculated based on differences between CDF values of two points \
    extending half an interval forward and extending half an interval backward respectively. \
These two points' SDF values are estimated using the current point's SDF and nablas values.

>>> NeuS original implmentaion

                Interval's alpha, using estimated SDFs of two middle points
                        {  - - - - - - - - - - - - - - - - - }
                        
    | ----------------- o ----------------- | -------------- o -------------- |
    ^                   ^                   ^                ^                ^
Previous point    Point extending    Current point    Point extending     Next point
                  half an interval                    half an interval
                  backward                            forward
                        V                   V                V
                  Estimated SDF      Queried SDF,     Estimated SDF
                                     nablas, colors



However, in practice, nablas of multires-grid-based representations are often much wilder than original MLPs, \
    which brings a lot of noise to the estimation and unstablizes training.

Hence, in our implementation, we first query interval boundaries' SDF to calculate alphas, \
    then query middle points' color and nablas for rendering.

>>> Our implmentation in nr3d_lib / neuralsim / neuralgen / neurecon

               Interval's alpha                       Interval's alpha      
          using "real" queried SDFs              using "real" queried SDFs  
    { - - - - - - - - - - - - - - - - - - -} { - - - - - - - - - - - - - - - -}
    
    | ----------------- x ----------------- | -------------- x -------------- |
    ^                   ^                   ^                ^                ^
Boundary pts       Middle pts          Boundary pts      Middle pts      Boundary pts
    V                   V                   V                V                V
 Queried SDF       Queried             Queried SDF       Queried         Quereid SDF
                   nablas, colors                        nablas colors

"""

__all__ = [
    'neus_renderer_mixin'
]

import numpy as np
from typing import Dict, List
from operator import itemgetter

import torch
import torch.nn.functional as F
from torch.utils.benchmark import Timer

from nr3d_lib.fmt import log
from nr3d_lib.logger import Logger
from nr3d_lib.config import ConfigDict

from nr3d_lib.models.utils import batchify_query
from nr3d_lib.models.annealers import get_annealer
from nr3d_lib.models.spatial import AABBSpace
from nr3d_lib.models.spatial_accel import get_accel
from nr3d_lib.models.fields.neus.utils import neus_packed_sdf_to_alpha, neus_packed_sdf_to_upsample_alpha, neus_ray_sdf_to_alpha, neus_ray_sdf_to_upsample_alpha

from nr3d_lib.render.volume_graphics import packed_alpha_to_vw, packed_volume_render_compression, ray_alpha_to_vw
from nr3d_lib.render.pack_ops import packed_div, packed_sum, packed_cumsum, packed_diff, merge_two_packs_sorted_aligned, get_pack_infos_from_batch, merge_two_batch_a_includes_b
from nr3d_lib.render.raysample import batch_sample_cdf, batch_sample_pdf, batch_sample_step_wrt_depth, batch_sample_step_linear, batch_sample_step_wrt_sqrt_depth, packed_sample_cdf

class neus_renderer_mixin:
    def __init__(
        self, 
        accel_cfg: ConfigDict = None, 
        sample_anneal_cfg: ConfigDict = None, 
        shrink_milestones: List[int]=[]) -> None:
        self.accel_cfg = accel_cfg
        self.sample_anneal_cfg = sample_anneal_cfg
        
        # Shrink
        self.shrink_milestones = shrink_milestones

    def populate(self, *args, **kwargs):
        super().populate(*args, **kwargs)
        
        # Acceleration struct
        self.accel = get_accel(space=self.space, device=self.device, **self.accel_cfg) if self.accel_cfg is not None else None
        self.upsample_s_divisor = 1.0
        
        # Sample annealing
        self.sample_ctrl = get_annealer(**self.sample_anneal_cfg) if self.sample_anneal_cfg is not None else None

    @property
    def space(self) -> AABBSpace:
        return super().space

    def uniform_sample(self, num_samples: int):
        x = self.space.uniform_sample_points(num_samples)
        ret = self.forward_sdf_nablas(x, ignore_update=True) # Do not upsate_samples here (usally there are too less samples here.)
        for k, v in ret.items():
            ret[k] = v.to(x.dtype)
        ret['net_x'] = x # NOTE: in network's uniformed space; not in world space.
        if 'nablas' in ret:
            ret['nablas_norm'] = ret['nablas'].norm(dim=-1)
        # NOTE: This is changed to be called every time `forward` is called, which is much more often.
        # if self.accel is not None:
        #     self.accel.gather_samples(ret['net_x'], val=ret['sdf'].data)
        return ret

    def forward_sdf(self, x: torch.Tensor, ignore_update=False, **kwargs):
        ret = super().forward_sdf(x, **kwargs)
        if not ignore_update and self.accel is not None:
            self.accel.gather_samples(x, val=ret['sdf'].data)
        return ret
    def forward_sdf_nablas(self, x: torch.Tensor, ignore_update=False, **kwargs):
        ret = super().forward_sdf_nablas(x, **kwargs)
        if not ignore_update and self.accel is not None:
            self.accel.gather_samples(x, val=ret['sdf'].data)
        return ret

    def preprocess_per_train_step(self, cur_it: int, logger: Logger = None):
        self.it = cur_it
        super().preprocess_per_train_step(cur_it, logger=logger)
        if self.accel is not None:
            # NOTE: Important to ignore update when query!
            self.accel.preprocess_per_train_step(cur_it, query_fn=lambda x: self.forward_sdf(x, ignore_update=True)['sdf'], logger=logger)
            self.upsample_s_divisor = 2 ** self.accel.training_granularity

    def postprocess_per_train_step(self, cur_it: int, logger: Logger = None):
        if hasattr(super(), 'postprocess_per_train_step'):
            super().postprocess_per_train_step(cur_it, logger=logger)
        if self.accel is not None:
            # NOTE: Important to ignore update when query!
            self.accel.postprocess_per_train_step(cur_it, lambda x: self.forward_sdf(x, ignore_update=True)['sdf'], logger=logger)

            #------------ Shrink according to actual occupied space.
            if cur_it in self.shrink_milestones:
                self.shrink()

    @torch.no_grad()
    def shrink(self):
        old_aabb = self.space.aabb
        new_aabb = self.accel.try_shrink()
        # Rescale network
        super().rescale_volume(new_aabb)
        # Rescale acceleration struct
        self.accel.rescale_volume(new_aabb)
        # Rescale space
        # NOTE: Always rescale space at the last step, since the old space is required by prev steps
        self.space.rescale_volume(new_aabb)
        old_aabb_str = '[' + ', '.join([str(np.round(i, decimals=2)) for i in old_aabb.tolist()]) + ']'
        new_aabb_str = '[' + ', '.join([str(np.round(i, decimals=2)) for i in new_aabb.tolist()]) + ']'
        log.info(f"=> Shrink from {old_aabb_str} to {new_aabb_str}")

    def ray_test(self, rays_o: torch.Tensor, rays_d: torch.Tensor, near=None, far=None, **extra_ray_shaped):
        assert (rays_o.dim() == 2) and (rays_d.dim() == 2), "Expect rays_o and rays_d to be of shape [N, 3]"
        return self.space.ray_test(rays_o, rays_d, near=near, far=far, return_rays=True, **extra_ray_shaped)

    def _ray_query_march_occ(
        ):
        """
        NGP's ray query mode
        """
        pass
    
    def _ray_query_coarse_multi_upsample(
        self, ray_tested: Dict[str, torch.Tensor], *, 
        # Common params
        with_rgb: bool = True, with_normal: bool = True, 
        perturb: bool = False, nablas_has_grad: bool = False, forward_inv_s: float = None, 
        # Distinct params
        compression = True, # Whether to early stop when already accumulated enough weights
        num_coarse: int = 64, coarse_step_cfg = ConfigDict(step_mode='linear'), 
        upsample_inv_s: float = 64., upsample_mode: str = 'multistep_estimate', num_fine: int = 64, 
        upsample_inv_s_factors: List[int] = [1, 2, 4, 8], upsample_use_estimate_alpha=False,  # For upsample_mode = multistep_estimate
        num_nograd: int = 1024, chunksize_query: int = 2**24 # For upsample_mode = direct_more
        ):
        """
        Vanilla NeuS ray query mode
        """

        upsample_inv_s /= self.upsample_s_divisor
        use_view_dirs = self.radiance_use_view_dirs
        forward_inv_s = self.forward_inv_s() if forward_inv_s is None else forward_inv_s

        empty_volume_buffer = dict(buffer_type='empty', ray_inds_hit=[])
        if (num_rays:=ray_tested['num_rays']) == 0:
            return empty_volume_buffer
        
        # NOTE: Normalized rays in network's space
        rays_o, rays_d, near, far, ray_inds = itemgetter('rays_o', 'rays_d', 'near', 'far', 'ray_inds')(ray_tested)
        assert (rays_o.dim() == 2) and (rays_d.dim()==2)
        
        # NOTE: The device & dtype of output
        device, dtype = rays_o.device, rays_o.dtype
        # NOTE: The spatial length scale on each ray caused by scaling rays_d 
        dir_scale = rays_d.norm(dim=-1)  # [num_rays]
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
        
        #----------------
        # Upsampling & gather volume_buffer
        #----------------
        @torch.no_grad()
        def upsample():
            if upsample_mode == 'direct_use':
                sdf_coarse = self.implicit_surface.forward(torch.addcmul(rays_o.unsqueeze(-2), rays_d.unsqueeze(-2), depths_coarse_1.unsqueeze(-1)))['sdf']
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
                _d = near.lerp(far, torch.linspace(0, 1, num_nograd).float().to(device))
                _sdf = batchify_query(self.implicit_surface.forward, torch.addcmul(rays_o.unsqueeze(-2), rays_d.unsqueeze(-2), _d.unsqueeze(-1)), chunk=chunksize_query)['sdf']
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
                sdf_all = self.implicit_surface.forward(torch.addcmul(rays_o.unsqueeze(-2), rays_d.unsqueeze(-2), depths_coarse_1.unsqueeze(-1)))['sdf']
                num_fine_per_iter = num_fine//2*2+1 # Had better always be odd
                for i, factor in enumerate(upsample_inv_s_factors):
                    if upsample_use_estimate_alpha:
                        alpha = neus_ray_sdf_to_upsample_alpha(sdf_all, d_all, upsample_inv_s * factor)
                    else:
                        alpha = neus_ray_sdf_to_alpha(sdf_all, upsample_inv_s * factor)
                    vw = ray_alpha_to_vw(alpha)
                    d_fine_ter = batch_sample_pdf(d_all, vw, num_fine_per_iter, perturb=perturb)
                    d_all, d_sort_indices = torch.sort(torch.cat([d_all, d_fine_ter], dim=-1), dim=-1)
                    if i < len(upsample_inv_s_factors)-1:
                        sdf_fine_iter = self.implicit_surface.forward(torch.addcmul(rays_o.unsqueeze(-2), rays_d.unsqueeze(-2), d_fine_ter.unsqueeze(-1)))['sdf']
                        sdf_all = torch.gather(torch.cat([sdf_all, sdf_fine_iter], dim=-1), -1, d_sort_indices) 
            else:
                raise RuntimeError(f"Invalid upsample_mode={upsample_mode}")
            return d_all

        d_all = upsample()
        d_mid = 0.5 * (d_all[..., 1:] + d_all[..., :-1])
        alpha = neus_ray_sdf_to_alpha(
            self.forward_sdf(
                torch.addcmul(rays_o.unsqueeze(-2), rays_d.unsqueeze(-2), d_all.unsqueeze(-1))
            )['sdf'], forward_inv_s) # The same shape with d_mid
        
        if compression:
            pack_infos = get_pack_infos_from_batch(alpha.shape[0], alpha.shape[1], device=device)
            nidx_useful, pack_infos_hit_useful, pidx_useful = packed_volume_render_compression(alpha.flatten(), pack_infos)
            
            if nidx_useful.numel() == 0:
                return empty_volume_buffer
            else:
                depths_packed, alpha_packed = d_mid.flatten()[pidx_useful], alpha.flatten()[pidx_useful]
                volume_buffer = dict(
                    buffer_type='packed', 
                    ray_inds_hit=ray_inds[nidx_useful], pack_infos_hit=pack_infos_hit_useful, 
                    t=depths_packed.to(dtype), opacity_alpha=alpha_packed.to(dtype))
                
                if with_rgb or with_normal:
                    ridx_all = torch.arange(alpha.shape[0], device=device, dtype=torch.long).unsqueeze(-1).expand_as(alpha)
                    ridx_all = ridx_all.flatten()[pidx_useful]
                    pts_mid = torch.addcmul(rays_o[ridx_all], rays_d[ridx_all], depths_packed.unsqueeze(-1))
                    # Get embedding code
                    h_appear_embed = ray_tested["rays_h_appear_embed"][ridx_all] \
                        if ray_tested.get('rays_h_appear_embed', None) is not None else None
                    #----------- Net query
                    # [num_rays_hit, num_fine_all, ...]
                    net_out = self.forward(
                        x=pts_mid, 
                        v=view_dirs[ridx_all] if use_view_dirs else None,
                        h_appear_embed=h_appear_embed,
                        nablas_has_grad=nablas_has_grad, with_rgb=with_rgb, with_normal=with_normal)
                    volume_buffer['net_x'] = pts_mid
                    if "nablas" in net_out: volume_buffer["nablas"] = net_out["nablas"].to(dtype)
                    if "radiances" in net_out: volume_buffer["rgb"] = net_out["radiances"].to(dtype)
                volume_buffer['details'] = {'render.num_per_ray0': d_mid.shape[-1], 'render.num_per_ray': pack_infos_hit_useful[:, 1]}
                return volume_buffer
        else:
            volume_buffer = dict(
                buffer_type='batched', 
                ray_inds_hit=ray_inds, num_per_hit=d_mid.shape[-1], 
                t=d_mid.to(dtype), opacity_alpha=alpha.to(dtype))
            
            if with_rgb or with_normal:
                pts_mid = torch.addcmul(rays_o.unsqueeze(-2), rays_d.unsqueeze(-2), d_mid.unsqueeze(-1))
                # Get embedding code
                h_appear_embed = ray_tested["rays_h_appear_embed"].unsqueeze(-2).expand(*d_mid.shape,-1)\
                    if ray_tested.get('rays_h_appear_embed', None) is not None else None

                net_out = self.forward(
                    x=pts_mid, 
                    v=view_dirs.unsqueeze(-2).expand_as(pts_mid) if use_view_dirs else None,
                    h_appear_embed=h_appear_embed,
                    nablas_has_grad=nablas_has_grad, with_rgb=with_rgb, with_normal=with_normal
                )
                volume_buffer['net_x'] = pts_mid
                if "nablas" in net_out: volume_buffer["nablas"] = net_out["nablas"].to(dtype)
                if "radiances" in net_out: volume_buffer["rgb"] = net_out["radiances"].to(dtype)
            volume_buffer['details'] = {'render.num_per_ray': d_mid.shape[-1]}
            return volume_buffer

    def _ray_query_march_occ_multi_upsample(
        self, ray_tested: Dict[str, torch.Tensor], 
        # Common params
        with_rgb: bool = True, with_normal: bool = True, 
        perturb: bool = False, nablas_has_grad: bool = False, forward_inv_s: float = None, 
        # Distinct params
        num_coarse: int = 0, coarse_step_cfg = ConfigDict(step_mode='linear'), 
        chunksize_query: int = 2**24, march_cfg = ConfigDict(), num_fine: int = 8, 
        upsample_inv_s: float = 64., upsample_inv_s_factors: List[int] = [1, 4, 16], upsample_use_estimate_alpha=False
        # with_near_sdf = ..., depth_use_normalized_vw = ..., query_mode = ..., near = ..., far = ..., with_env = ..., rayschunk = ...
        ):
        """
        Multi-stage upsampling on marched samples of occupancy grids (without reduction/compression of meaningless samples)
        Introduced in StreetSurf Section 4.1
        https://arxiv.org/abs/2306.04988
        """
        assert self.accel is not None, "Need a non-empty AccelStruct"
        
        upsample_inv_s /= self.upsample_s_divisor
        use_view_dirs = self.radiance_use_view_dirs
        forward_inv_s = self.forward_inv_s() if forward_inv_s is None else forward_inv_s

        empty_volume_buffer = dict(buffer_type='empty', ray_inds_hit=[])
        if (num_rays:=ray_tested['num_rays']) == 0:
            return empty_volume_buffer
        
        # NOTE: Normalized rays in network's space
        rays_o, rays_d, near, far, ray_inds = itemgetter('rays_o', 'rays_d', 'near', 'far', 'ray_inds')(ray_tested)
        assert (rays_o.dim() == 2) and (rays_d.dim()==2)
        
        # NOTE: The device & dtype of output
        device, dtype = rays_o.device, rays_o.dtype
        
        # NOTE: The spatial length scale on each ray caused by scaling rays_d 
        dir_scale = rays_d.norm(dim=-1)  # [num_rays]
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
            # alpha_coarse = neus_ray_sdf_to_alpha(self.forward_sdf(torch.addcmul(rays_o.unsqueeze(-2), rays_d.unsqueeze(-2), depths_coarse_1.unsqueeze(-1)))['sdf'], forward_inv_s)
            deltas_coarse = deltas_coarse_1[..., :num_coarse]
            depths_coarse = depths_coarse_1[..., :num_coarse] + deltas_coarse / 2.
            # samples_coarse = torch.addcmul(rays_o.unsqueeze(-2), rays_d.unsqueeze(-2), depths_coarse.unsqueeze(-1))
            # sample_dirs, sample_dir_scales = view_dirs.unsqueeze(-2), dir_scale.unsqueeze(-1)
            # net_out = self.forward(samples_coarse, sample_dirs if use_view_dirs else None, nablas_has_grad=nablas_has_grad, with_rgb=with_rgb, with_normal=with_normal)
            # sdf_coarse, nablas_coarse, radiances_coarse = net_out['sdf'], net_out['nablas'], net_out['radiances']

        #----------------
        # Ray marching
        #----------------
        # raymarch_fix_continuity = config.get('raymarch_fix_continuity', True)
        ridx_hit, samples, depth_samples, _, _, pack_infos, _, _ = \
            self.accel.ray_march(rays_o, rays_d, near, far, perturb=perturb, **march_cfg)

        #----------------
        # Upsampling & gather volume_buffer
        #----------------
        if ridx_hit is not None:
            pack_infos_marched = pack_infos.clone()
            # [num_rays_hit, 1, 3], [num_rays_hit, 1]
            rays_o_hit, rays_d_hit = rays_o[ridx_hit].unsqueeze(-2), rays_d[ridx_hit].unsqueeze(-2) # 27 us @ 9.6k rays; 80 us @ 500k rays
            
            #----------------
            # Upsample on marched samples
            with torch.no_grad():
                sdf = batchify_query(lambda x: self.forward_sdf(x)['sdf'], samples, chunk=chunksize_query)
                # sdf = self.forward_sdf(samples)['sdf']
                
                depths_1 = []
                num_fine_per_iter = num_fine//2*2+1 # Had better always be odd
                pinfo_fine_per_iter = get_pack_infos_from_batch(ridx_hit.numel(), num_fine_per_iter, device=device)
                for i, factor in enumerate(upsample_inv_s_factors):
                    if upsample_use_estimate_alpha:
                        alpha = neus_packed_sdf_to_upsample_alpha(sdf, depth_samples, upsample_inv_s * factor, pack_infos) # This could leads to artifacts
                    else:
                        alpha = neus_packed_sdf_to_alpha(sdf, upsample_inv_s * factor, pack_infos)
                    
                    # if raymarch_fix_continuity and (con_pack_infos is not None):
                    #     con_last_inds = con_pack_infos[...,0] + con_pack_infos[...,1] - 1
                    #     alpha[con_last_inds] = 0.  # To fix wrong alpha results between consecutive segments
                    vw = packed_alpha_to_vw(alpha, pack_infos)
                    
                    neus_cdf = packed_cumsum(vw, pack_infos, exclusive=True)
                    last_cdf = neus_cdf[pack_infos[...,0] + pack_infos[...,1] - 1]
                    neus_cdf = packed_div(neus_cdf, last_cdf.clamp_min(1e-5), pack_infos)
                    depths_fine_iter = packed_sample_cdf(depth_samples, neus_cdf.to(depth_samples.dtype), pack_infos, num_fine_per_iter, perturb=perturb)[0]
                    depths_1.append(depths_fine_iter)
                    
                    if len(upsample_inv_s_factors) > 1:
                        # 273 us @ 930k + 25k
                        # Merge fine samples of current upsample iter to previous packed buffer.
                        # NOTE: The new `pack_infos` is calculated here.
                        pidx0, pidx1, pack_infos = merge_two_packs_sorted_aligned(depth_samples, pack_infos, depths_fine_iter.flatten(), pinfo_fine_per_iter, b_sorted=True, return_val=False)
                        num_samples_iter = depth_samples.numel()
                        depth_samples_iter = depth_samples.new_empty([num_samples_iter + depths_fine_iter.numel()])
                        depth_samples_iter[pidx0], depth_samples_iter[pidx1] = depth_samples, depths_fine_iter.flatten()
                        depth_samples = depth_samples_iter

                        if i < len(upsample_inv_s_factors)-1:
                            x_fine = torch.addcmul(rays_o_hit, rays_d_hit, depths_fine_iter.unsqueeze(-1))
                            sdf_iter = sdf.new_empty([num_samples_iter + depths_fine_iter.numel()])
                            sdf_iter[pidx0], sdf_iter[pidx1] = sdf, self.forward_sdf(x_fine.flatten(0, -2))['sdf']
                            sdf = sdf_iter

                    # import matplotlib.pyplot as plt
                    # i = 400
                    # _0, _1 = pack_infos[i, 0].item(), pack_infos[i+1, 0].item()
                    # _t = depth_samples[_0:_1].data.cpu().numpy()
                    # _vw = vw[_0:_1].data.cpu().numpy()
                    # _sdf = sdf[_0:_1].data.cpu().numpy()
                    # _t_fine = depths_fine_iter[i].data.cpu().numpy()
                    # fig = plt.figure()
                    # # n_0, n_1 = nugget_pack_infos[i, 0].item(), nugget_pack_infos[i+1, 0].item()
                    # # for ni in range(n_0, n_1, 1):
                    # #     plt.axvspan(nugget_depths[ni,0].item(), nugget_depths[ni,1].item(), alpha=0.2, color='brown')
                    # plt.vlines(_t_fine, 0, _vw.max(), 'r', label='t_fine')
                    # plt.plot(_t, _vw, 'x-', label='vw')
                    # plt.ylabel('vw')
                    # ax2 = (ax:=plt.gca()).twinx()
                    # ax2.plot(_t, _sdf, color='orange', marker='x', ls='-', label='sdf')
                    # ax2.axhline(y=0, color='g', label='sdf=0')
                    # ax2.set_ylabel('sdf')
                    # fig.legend(loc='center right', bbox_to_anchor=(1, 0.5), bbox_transform=ax.transAxes)
                    # fig.suptitle(f"i={i}, upsample_inv_s={upsample_inv_s * factor}, num_fine={num_fine}, vw_sum={_vw.sum()}", fontsize=14)
                    # plt.show()

                    # # NOTE: Do not sample on those rays with zero accumulated cdfs
                    # actual_hit = (last_cdf > 0).nonzero().long()[..., 0]
                    # ridx_hit, depths_fine_iter = ridx_hit[actual_hit], depths_fine_iter[actual_hit]
                
                if len(upsample_inv_s_factors) > 1:
                    depths_1 = torch.cat(depths_1, dim=-1).sort(dim=-1).values
                else:
                    depths_1 = depths_1[0]
            
            #----------------
            # Acquire volume_buffer via quering network and gather results
            if num_coarse == 0:                    
                # [num_rays_hit, num_fine_all]
                alpha = neus_ray_sdf_to_alpha(self.forward_sdf(torch.addcmul(rays_o_hit, rays_d_hit, depths_1.unsqueeze(-1)))['sdf'], forward_inv_s)
                depths = (depths_1[..., :-1] + depths_1.diff(dim=-1)/2.)

                volume_buffer = dict(
                    buffer_type='batched', 
                    ray_inds_hit=ray_inds[ridx_hit], num_per_hit=depths.shape[-1], 
                    t=depths.to(dtype), opacity_alpha=alpha.to(dtype))
                
                if with_rgb or with_normal:
                    # Get embedding code
                    h_appear_embed = ray_tested["rays_h_appear_embed"][ridx_hit].unsqueeze(-2).expand(*depths.shape,-1)\
                        if ray_tested.get('rays_h_appear_embed', None) is not None else None
                    pts_mid = torch.addcmul(rays_o_hit, rays_d_hit, depths[..., None])
                    #----------- Net query
                    # [num_rays_hit, num_fine_all, ...]
                    net_out = self.forward(
                        x=pts_mid,
                        v=view_dirs[ridx_hit][..., None, :] if use_view_dirs else None, 
                        h_appear_embed=h_appear_embed,
                        nablas_has_grad=nablas_has_grad, with_rgb=with_rgb, with_normal=with_normal)
                    if "nablas" in net_out: volume_buffer["nablas"] = net_out["nablas"].to(dtype)
                    if "radiances" in net_out: volume_buffer["rgb"] = net_out["radiances"].to(dtype)
                    volume_buffer['net_x'] = pts_mid
                volume_buffer['details'] = {'march.num_per_ray': pack_infos_marched[:, 1], 'render.num_per_ray': depths.shape[-1]}
                return volume_buffer
                
            else:
                def merge():
                    ridx_coarse = torch.arange(ray_inds.numel(), device=device)
                    pidx0, pidx1, pack_infos = merge_two_batch_a_includes_b(depths_coarse_1, ridx_coarse, depths_1, ridx_hit, a_sorted=True)
                    num_samples = depths_1.numel() + depths_coarse_1.numel()
                    depths_1_packed = depths_1.new_zeros([num_samples])
                    ridx_all = ridx_hit.new_zeros([num_samples])
                    ridx_all[pidx0], ridx_all[pidx1] = ridx_coarse.unsqueeze(-1), ridx_hit.unsqueeze(-1)
                    depths_1_packed[pidx0], depths_1_packed[pidx1] = depths_coarse_1, depths_1
                    return ridx_all, depths_1_packed, pack_infos
                
                ridx_all, depths_1_packed, pack_infos_hit = merge()
                rays_o_packed, rays_d_packed = rays_o[ridx_all], rays_d[ridx_all]
                alpha_packed = neus_packed_sdf_to_alpha(self.forward_sdf(torch.addcmul(rays_o_packed, rays_d_packed, depths_1_packed.unsqueeze(-1)))['sdf'], forward_inv_s, pack_infos_hit)
                depths_packed = depths_1_packed + packed_diff(depths_1_packed, pack_infos_hit) / 2.
                
                volume_buffer = dict(
                    buffer_type='packed', 
                    ray_inds_hit=ray_inds, pack_infos_hit=pack_infos_hit, 
                    t=depths_packed.to(dtype), opacity_alpha=alpha_packed.to(dtype))
                
                if with_rgb or with_normal:
                    # Get embedding code
                    h_appear_embed = ray_tested["rays_h_appear_embed"][ridx_all]\
                        if ray_tested.get('rays_h_appear_embed', None) is not None else None
                    pts_mid = torch.addcmul(rays_o_packed, rays_d_packed, depths_packed.unsqueeze(-1))
                    net_out = self.forward(
                        x=pts_mid, 
                        v=view_dirs[ridx_all] if use_view_dirs else None, 
                        h_appear_embed=h_appear_embed,
                        nablas_has_grad=nablas_has_grad, with_rgb=with_rgb, with_normal=with_normal)
                    volume_buffer['net_x'] = pts_mid
                    if "nablas" in net_out: volume_buffer["nablas"] = net_out["nablas"].to(dtype)
                    if "radiances" in net_out: volume_buffer["rgb"] = net_out["radiances"].to(dtype)
                volume_buffer['details'] = {'march.num_per_ray': pack_infos_marched[:, 1], 'render.num_per_ray': pack_infos_hit[:, 1]}
                return volume_buffer
        
        else: # ridx_hit is None
            
            if num_coarse == 0: # ridx_hit is None and num_coarse == 0
                return empty_volume_buffer

            else: # ridx_hit is None and num_coarse > 0
                alpha_coarse = neus_ray_sdf_to_alpha(self.forward_sdf(torch.addcmul(rays_o.unsqueeze(-2), rays_d.unsqueeze(-2), depths_coarse_1.unsqueeze(-1)))['sdf'], forward_inv_s)

                deltas_coarse = deltas_coarse_1[..., :num_coarse]
                depths_coarse = depths_coarse_1[..., :num_coarse] + deltas_coarse / 2.

                volume_buffer = dict(
                    buffer_type='batched', 
                    ray_inds_hit=ray_inds, num_per_hit=num_coarse, 
                    t=depths_coarse.to(dtype), opacity_alpha=alpha_coarse.to(dtype))

                if with_rgb or with_normal:
                    # Get embedding code
                    h_appear_embed = ray_tested["rays_h_appear_embed"].unsqueeze(-2).expand(*depths_coarse.shape,-1)\
                        if ray_tested.get('rays_h_appear_embed', None) is not None else None
                    pts_mid = torch.addcmul(rays_o[..., None, :], rays_d[..., None, :], depths_coarse[..., None])
                    net_out = self.forward(
                        x=pts_mid, 
                        v=view_dirs[..., None, :] if use_view_dirs else None,
                        h_appear_embed=h_appear_embed,
                        nablas_has_grad=nablas_has_grad, with_rgb=with_rgb, with_normal=with_normal)
                    volume_buffer['net_x'] = pts_mid
                    if "nablas" in net_out: volume_buffer["nablas"] = net_out["nablas"].to(dtype)
                    if "radiances" in net_out: volume_buffer["rgb"] = net_out["radiances"].to(dtype)
                volume_buffer['details'] = {'render.num_per_ray': num_coarse}
                return volume_buffer

    # @profile
    def _ray_query_march_occ_multi_upsample_compressed(
        self, ray_tested: Dict[str, torch.Tensor], *, 
        # Common params
        with_rgb: bool = True, with_normal: bool = True, 
        perturb: bool = False, nablas_has_grad: bool = False, forward_inv_s: float = None, 
        # Distinct params
        num_coarse: int = 0, coarse_step_cfg = ConfigDict(step_mode='linear'), 
        chunksize_query: int = 2**24,  march_cfg = ConfigDict(), num_fine: int = 8, 
        upsample_inv_s: float = 64., upsample_inv_s_factors: List[int] = [1, 4, 16], upsample_use_estimate_alpha=False
        ):
        """
        Multi-stage upsampling on marched samples of occupancy grids (with reduction/compression of meaningless samples)
        Introduced in StreetSurf Section 4.1
        https://arxiv.org/abs/2306.04988
        """
        
        assert self.accel is not None, "Need a non-empty AccelStruct"
        
        if isinstance(num_fine, int):
            num_fine = [num_fine] * len(upsample_inv_s_factors)
        assert len(num_fine) == len(upsample_inv_s_factors), f"num_fine should be of the same length={len(upsample_inv_s_factors)} with upsample"
        
        upsample_inv_s /= self.upsample_s_divisor
        use_view_dirs = self.radiance_use_view_dirs
        forward_inv_s = self.forward_inv_s() if forward_inv_s is None else forward_inv_s

        empty_volume_buffer = dict(buffer_type='empty', ray_inds_hit=[])
        if (num_rays:=ray_tested['num_rays']) == 0:
            return empty_volume_buffer
        
        # NOTE: Normalized rays in network's space
        rays_o, rays_d, near, far, ray_inds = itemgetter('rays_o', 'rays_d', 'near', 'far', 'ray_inds')(ray_tested)
        assert (rays_o.dim() == 2) and (rays_d.dim()==2)
        
        # NOTE: The device & dtype of output
        device, dtype = rays_o.device, rays_o.dtype
        # NOTE: The spatial length scale on each ray caused by scaling rays_d 
        dir_scale = rays_d.norm(dim=-1)  # [num_rays]
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
            # alpha_coarse = neus_ray_sdf_to_alpha(self.forward_sdf(torch.addcmul(rays_o.unsqueeze(-2), rays_d.unsqueeze(-2), depths_coarse_1.unsqueeze(-1)))['sdf'], forward_inv_s)
            deltas_coarse = deltas_coarse_1[..., :num_coarse]
            depths_coarse = depths_coarse_1[..., :num_coarse] + deltas_coarse / 2.
            # samples_coarse = torch.addcmul(rays_o.unsqueeze(-2), rays_d.unsqueeze(-2), depths_coarse.unsqueeze(-1))
            # sample_dirs, sample_dir_scales = view_dirs.unsqueeze(-2), dir_scale.unsqueeze(-1)
            # net_out = self.forward(samples_coarse, sample_dirs if use_view_dirs else None, nablas_has_grad=nablas_has_grad, with_rgb=with_rgb, with_normal=with_normal)
            # sdf_coarse, nablas_coarse, radiances_coarse = net_out['sdf'], net_out['nablas'], net_out['radiances']

        #----------------
        # Ray marching
        #----------------
        # raymarch_fix_continuity = config.get('raymarch_fix_continuity', True)
        ridx_hit, samples, depth_samples, _, _, pack_infos, _, _ = \
            self.accel.ray_march(rays_o, rays_d, near, far, perturb=perturb, **march_cfg)

        #----------------
        # Upsampling & gather volume_buffer
        #----------------
        if ridx_hit is not None:
            pack_infos_marched = pack_infos.clone()
            # [num_rays_hit, 1, 3], [num_rays_hit, 1]
            rays_o_hit, rays_d_hit = rays_o[ridx_hit].unsqueeze(-2), rays_d[ridx_hit].unsqueeze(-2) # 27 us @ 9.6k rays; 80 us @ 500k rays
            
            #----------------
            # Upsample on marched samples
            with torch.no_grad():
                sdf = batchify_query(lambda x: self.forward_sdf(x)['sdf'], samples, chunk=chunksize_query)
                # sdf = self.forward_sdf(samples)['sdf']
                
                depths_1 = []
                for i, factor in enumerate(upsample_inv_s_factors):
                    num_fine_per_iter = num_fine[i]//2*2+1 # Had better always be odd
                    pinfo_fine_per_iter = get_pack_infos_from_batch(ridx_hit.numel(), num_fine_per_iter, device=device)
                    
                    if upsample_use_estimate_alpha:
                        alpha = neus_packed_sdf_to_upsample_alpha(sdf, depth_samples, upsample_inv_s * factor, pack_infos) # This could leads to artifacts
                    else:
                        alpha = neus_packed_sdf_to_alpha(sdf, upsample_inv_s * factor, pack_infos)
                    
                    # if raymarch_fix_continuity and (con_pack_infos is not None):
                    #     con_last_inds = con_pack_infos[...,0] + con_pack_infos[...,1] - 1
                    #     alpha[con_last_inds] = 0.  # To fix wrong alpha results between consecutive segments
                    vw = packed_alpha_to_vw(alpha, pack_infos)
                    
                    neus_cdf = packed_cumsum(vw, pack_infos, exclusive=True)
                    last_cdf = neus_cdf[pack_infos[...,0] + pack_infos[...,1] - 1]
                    neus_cdf = packed_div(neus_cdf, last_cdf.clamp_min(1e-5), pack_infos)
                    depths_fine_iter = packed_sample_cdf(depth_samples, neus_cdf.to(depth_samples.dtype), pack_infos, num_fine_per_iter, perturb=perturb)[0]
                    depths_1.append(depths_fine_iter)
                    
                    if len(upsample_inv_s_factors) > 1:
                        # 273 us @ 930k + 25k
                        # Merge fine samples of current upsample iter to previous packed buffer.
                        # NOTE: The new `pack_infos` is calculated here.
                        pidx0, pidx1, pack_infos = merge_two_packs_sorted_aligned(depth_samples, pack_infos, depths_fine_iter.flatten(), pinfo_fine_per_iter, b_sorted=True, return_val=False)
                        num_samples_iter = depth_samples.numel()
                        depth_samples_iter = depth_samples.new_empty([num_samples_iter + depths_fine_iter.numel()])
                        depth_samples_iter[pidx0], depth_samples_iter[pidx1] = depth_samples, depths_fine_iter.flatten()
                        depth_samples = depth_samples_iter

                        if i < len(upsample_inv_s_factors)-1:
                            x_fine = torch.addcmul(rays_o_hit, rays_d_hit, depths_fine_iter.unsqueeze(-1))
                            sdf_iter = sdf.new_empty([num_samples_iter + depths_fine_iter.numel()])
                            sdf_iter[pidx0], sdf_iter[pidx1] = sdf, self.forward_sdf(x_fine.flatten(0, -2))['sdf']
                            sdf = sdf_iter
                
                if len(upsample_inv_s_factors) > 1:
                    depths_1 = torch.cat(depths_1, dim=-1).sort(dim=-1).values
                else:
                    depths_1 = depths_1[0]
            
            #----------------
            # Acquire volume_buffer via quering network and gather results
            if num_coarse == 0:
                # [num_rays_hit, num_fine_all]
                if self.training:
                    alpha = neus_ray_sdf_to_alpha(
                        self.forward_sdf(torch.addcmul(rays_o_hit, rays_d_hit, depths_1.unsqueeze(-1)))['sdf'], 
                        forward_inv_s)
                else:
                    alpha = neus_ray_sdf_to_alpha(
                        batchify_query(
                            lambda x: self.forward_sdf(x)['sdf'], 
                            torch.addcmul(rays_o_hit, rays_d_hit, depths_1.unsqueeze(-1)), 
                            chunk=chunksize_query), 
                        forward_inv_s)
                depths = (depths_1[..., :-1] + depths_1.diff(dim=-1)/2.)
                
                pack_infos_hit = get_pack_infos_from_batch(ridx_hit.numel(), depths.shape[-1], device=device)
                nidx_useful, pack_infos_hit_useful, pidx_useful = packed_volume_render_compression(alpha.flatten(), pack_infos_hit)
                
                if nidx_useful.numel() == 0:
                    return empty_volume_buffer
                else:
                    depths_packed, alpha_packed = depths.flatten()[pidx_useful], alpha.flatten()[pidx_useful]

                    volume_buffer = dict(
                        buffer_type='packed', 
                        ray_inds_hit=ray_inds[ridx_hit][nidx_useful], pack_infos_hit=pack_infos_hit_useful, 
                        t=depths_packed.to(dtype), opacity_alpha=alpha_packed.to(dtype))
                    
                    if with_rgb or with_normal:
                        ridx_all = ridx_hit.unsqueeze(-1).expand(-1,depths.shape[-1]).flatten()[pidx_useful]
                        # Get embedding code
                        h_appear_embed = ray_tested["rays_h_appear_embed"][ridx_all]\
                            if ray_tested.get('rays_h_appear_embed', None) is not None else None
                        pts_mid = torch.addcmul(rays_o[ridx_all], rays_d[ridx_all], depths_packed.unsqueeze(-1))
                        #----------- Net query
                        # [num_rays_hit, num_fine_all, ...]
                        net_out = self.forward(
                            x=pts_mid, 
                            v=view_dirs[ridx_all] if use_view_dirs else None,
                            h_appear_embed=h_appear_embed,
                            nablas_has_grad=nablas_has_grad, with_rgb=with_rgb, with_normal=with_normal)
                        volume_buffer['net_x'] = pts_mid
                        if "nablas" in net_out: volume_buffer["nablas"] = net_out["nablas"].to(dtype)
                        if "radiances" in net_out: volume_buffer["rgb"] = net_out["radiances"].to(dtype)
                    volume_buffer['details'] = {'march.num_per_ray': pack_infos_marched[:, 1], 'render.num_per_ray0': depths.shape[-1], 'render.num_per_ray': pack_infos_hit_useful[:, 1]}
                    return volume_buffer

            else:
                def merge():
                    ridx_coarse = torch.arange(ray_inds.numel(), device=device)
                    pidx0, pidx1, pack_infos = merge_two_batch_a_includes_b(depths_coarse_1, ridx_coarse, depths_1, ridx_hit, a_sorted=True)
                    num_samples = depths_1.numel() + depths_coarse_1.numel()
                    depths_1_packed = depths_1.new_zeros([num_samples])
                    ridx_all = ridx_hit.new_zeros([num_samples])
                    ridx_all[pidx0], ridx_all[pidx1] = ridx_coarse.unsqueeze(-1), ridx_hit.unsqueeze(-1)
                    depths_1_packed[pidx0], depths_1_packed[pidx1] = depths_coarse_1, depths_1
                    return ridx_all, depths_1_packed, pack_infos
                
                ridx_all, depths_1_packed, pack_infos_hit = merge()
                depths_packed = depths_1_packed + packed_diff(depths_1_packed, pack_infos_hit) / 2.
                
                if self.training:
                    alpha_packed = neus_packed_sdf_to_alpha(
                        self.forward_sdf(torch.addcmul(rays_o[ridx_all], rays_d[ridx_all], depths_1_packed.unsqueeze(-1)))['sdf'], 
                        forward_inv_s, pack_infos_hit)
                else:
                    alpha_packed = neus_packed_sdf_to_alpha(
                        batchify_query(
                            lambda x: self.forward_sdf(x)['sdf'], 
                            torch.addcmul(rays_o[ridx_all], rays_d[ridx_all], depths_1_packed.unsqueeze(-1)), 
                            chunk=chunksize_query), 
                        forward_inv_s, pack_infos_hit)
                nidx_useful, pack_infos_hit_useful, pidx_useful = packed_volume_render_compression(alpha_packed, pack_infos_hit)
                
                if nidx_useful.numel() == 0:
                    return empty_volume_buffer
                else:
                    # Update
                    ridx_all, depths_packed, alpha_packed  = ridx_all[pidx_useful], depths_packed[pidx_useful], alpha_packed[pidx_useful]
                    
                    volume_buffer = dict(
                        buffer_type='packed', 
                        ray_inds_hit=ray_inds[nidx_useful], pack_infos_hit=pack_infos_hit_useful, 
                        t=depths_packed.to(dtype), opacity_alpha=alpha_packed.to(dtype))
                    
                    if with_rgb or with_normal:
                        # Get embedding code
                        h_appear_embed = ray_tested["rays_h_appear_embed"][ridx_all]\
                            if ray_tested.get('rays_h_appear_embed', None) is not None else None
                        pts_mid = torch.addcmul(rays_o[ridx_all], rays_d[ridx_all], depths_packed.unsqueeze(-1))
                        net_out = self.forward(
                            x=pts_mid, 
                            v=view_dirs[ridx_all] if use_view_dirs else None, 
                            h_appear_embed=h_appear_embed,
                            nablas_has_grad=nablas_has_grad, with_rgb=with_rgb, with_normal=with_normal)
                        volume_buffer['net_x'] = pts_mid
                        if "nablas" in net_out: volume_buffer["nablas"] = net_out["nablas"].to(dtype)
                        if "radiances" in net_out: volume_buffer["rgb"] = net_out["radiances"].to(dtype)
                    volume_buffer['details'] = {'march.num_per_ray': pack_infos_marched[:, 1], 'render.num_per_ray0': pack_infos_hit[:, 1], 'render.num_per_ray': pack_infos_hit_useful[:, 1]}
                    return volume_buffer

        else: # ridx_hit is None
            if num_coarse == 0: # ridx_hit is None and num_coarse == 0
                return empty_volume_buffer
            
            else: # ridx_hit is None and num_coarse > 0
                if self.training:
                    alpha_coarse = neus_ray_sdf_to_alpha(
                        self.forward_sdf(torch.addcmul(rays_o.unsqueeze(-2), rays_d.unsqueeze(-2), depths_coarse_1.unsqueeze(-1)))['sdf'], 
                        forward_inv_s)
                else:
                    alpha_coarse = neus_ray_sdf_to_alpha(
                        batchify_query(
                            lambda x: self.forward_sdf(x)['sdf'], 
                            torch.addcmul(rays_o.unsqueeze(-2), rays_d.unsqueeze(-2), depths_coarse_1.unsqueeze(-1)), 
                            chunk=chunksize_query), 
                        forward_inv_s)
                depths_coarse = depths_coarse_1[..., :num_coarse] + deltas_coarse_1[..., :num_coarse] / 2.
                
                pack_infos_coarse = get_pack_infos_from_batch(num_rays, depths_coarse.shape[-1], device=device)
                nidx_useful, pack_infos_hit_useful, pidx_useful = packed_volume_render_compression(alpha_coarse.flatten(), pack_infos_coarse)
                
                if nidx_useful.numel() == 0:
                    return empty_volume_buffer
                else:
                    depths_packed, alpha_packed = depths_coarse.flatten()[pidx_useful], alpha_coarse.flatten()[pidx_useful]

                    volume_buffer = dict(
                        buffer_type='packed', 
                        ray_inds_hit=ray_inds[nidx_useful], pack_infos_hit=pack_infos_hit_useful, 
                        t=depths_packed.to(dtype), opacity_alpha=alpha_packed.to(dtype))

                    if with_rgb or with_normal:
                        ridx_all = torch.arange(ray_inds.numel(), device=device).unsqueeze_(-1).expand_as(alpha_coarse).flatten()[pidx_useful]
                        # Get embedding code
                        h_appear_embed = ray_tested["rays_h_appear_embed"][ridx_all]\
                            if ray_tested.get('rays_h_appear_embed', None) is not None else None
                        pts_mid = torch.addcmul(rays_o[ridx_all], rays_d[ridx_all], depths_packed.unsqueeze(-1))
                        net_out = self.forward(
                            x=pts_mid, 
                            v=view_dirs[ridx_all] if use_view_dirs else None, 
                            h_appear_embed=h_appear_embed,
                            nablas_has_grad=nablas_has_grad, with_rgb=with_rgb, with_normal=with_normal)
                        volume_buffer['net_x'] = pts_mid
                        if "nablas" in net_out: volume_buffer["nablas"] = net_out["nablas"].to(dtype)
                        if "radiances" in net_out: volume_buffer["rgb"] = net_out["radiances"].to(dtype)
                    volume_buffer['details'] = {'render.num_per_ray0': num_coarse, 'render.num_per_ray': pack_infos_hit_useful[:, 1]}
                    return volume_buffer

    def _ray_query_march_occ_multi_upsample_compressed_strategy(
        self, ray_tested: Dict[str, torch.Tensor], 
        # Common params
        with_rgb: bool = True, with_normal: bool = True, 
        perturb: bool = False, nablas_has_grad: bool = False, forward_inv_s: float = None, 
        # Distinct params
        num_coarse_max: int = 32, num_coarse_min: int = 0, num_coarse_anneal_type: str = 'linear', 
        coarse_step_cfg = ConfigDict(step_mode='linear'), 
        chunksize_query: int = 2**24,  march_cfg = ConfigDict(), 
        upsample_inv_s: float = 64., upsample_s_factors_full: List[int] = [1, 4, 16], upsample_use_estimate_alpha=False):
        # TODO: Using learned `forward_inv_s` directly for control causes oscillations
        forward_inv_s = self.forward_inv_s() if forward_inv_s is None else forward_inv_s
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

    # @profile
    def ray_query(
        self, 
        # ray query inputs
        ray_input: Dict[str, torch.Tensor]=None, ray_tested: Dict[str, torch.Tensor]=None, 
        # ray query function config
        config=ConfigDict(), 
        # function config
        return_buffer=False, return_details=False, render_per_obj=False) -> dict:
        """ Query the model with input rays. 
            Conduct the core logic of ray sampling, ray marching and network query procedures.

        Args:
            ray_input (Dict[str, torch.Tensor], optional): All input rays. A dict composed of:
                rays_o: [num_total_rays, 3]
                rays_d: [num_total_rays, 3]
                near:   [num_total_rays] tensor or float or None
                far:    [num_total_rays] tensor or float or None
            ray_tested (Dict[str, torch.Tensor], optional): Tested rays (Typicallly those that intersect with objects). A dict composed of:
                num_rays:   int, Number of tested rays
                ray_inds:   [num_rays] tensor, ray indices in `num_total_rays` of the tested rays
                rays_o:     [num_rays, 3]
                rays_d:     [num_rays, 3]
                near:       [num_rays] tensor or float or None
                far:        [num_rays] tensor or float or None
            config (ConfigDict, optional): Config of ray_query. Defaults to ConfigDict().
            return_buffer (bool, optional): If return the queried volume buffer. Defaults to False.
            return_details (bool, optional): If return training / debugging related details. Defaults to False.
            render_per_obj (bool, optional): If return single object / seperate volume rendering results. Defaults to False.

        Returns:
            nested dict, The queried results, including 'volume_buffer', 'details', 'rendered'.
            
            ['volume_buffer'] dict, The queried volume buffer. Available if `return_buffer` is set True.
                For now, two types of buffers might be queried depending on the ray sampling algorithms, 
                    namely `batched` buffers and `packed` buffers.
                
                If there are no tested rays or no hit rays, the returned buffer is empty:
                    'volume_buffer" {'type': 'empty'}
                
                An example `batched` buffer:
                    'volume_buffer': {
                        'type': 'batched',
                        'ray_inds_hit':     [num_rays_hit] tensor, ray indices in `num_total_rays` of the hit & queried rays
                        't':                [num_rays_hit, num_samples_per_ray] batched tensor, real depth of the queried samples
                        'opacity_alpha':    [num_rays_hit, num_samples_per_ray] batched tensor, the queried alpha-values
                        'rgb':              [num_rays_hit, num_samples_per_ray] batched tensor, optional, the queried rgb values
                        'nablas':           [num_rays_hit, num_samples_per_ray] batched tensor, optional, the queried nablas values
                        'feature':          [num_rays_hit, num_samples_per_ray] batched tensor, optional, the queried features
                    }
                
                An example `packed` buffer:
                    'volume_buffer': {
                        'type': 'packed',
                        'ray_inds_hit':     [num_rays_hit] tensor, ray indices in `num_total_rays` of the hit & queried rays
                        'pack_infos_hit'    [num_rays_hit, 2] tensor, pack infos of the queried packed tensors
                        't':                [num_packed_samples] packed tensor, real depth of the queried samples
                        'opacity_alpha':    [num_packed_samples] packed tensor, the queried alpha-values
                        'rgb':              [num_packed_samples] packed tensor, optional, the queried rgb values
                        'nablas':           [num_packed_samples] packed tensor, optional, the queried nablas values
                        'feature':          [num_packed_samples] packed tensor, optional, the queried features
                    }
            
            ['details'] nested dict, Details for training. Available if `return_details` is set True.
            
            ['rendered'] dict, stand-alone rendered results. Available if `render_per_obj` is set True.
                An example rendered dict:
                    'rendered' {
                        'mask_volume':      [num_total_rays,] The rendered opacity / occupancy, in range [0,1]
                        'depth_volume':     [num_total_rays,] The rendered real depth
                        'rgb_volume':       [num_total_rays, 3] The rendered rgb, in range [0,1]
                        'normals_volume':   [num_total_rays, 3] The rendered normals, in range [-1,1]
                    }
        """
        
        #----------------
        # Inputs
        #----------------
        if ray_tested is None:
            assert ray_input is not None
            ray_tested = self.ray_test(**ray_input)
        
        #----------------
        # Shortcuts
        #----------------
        # NOTE: The device & dtype of output
        device, dtype = self.device, torch.float
        query_mode, with_rgb, with_normal = config.query_mode, config.with_rgb, config.with_normal
        forward_inv_s = config.get('forward_inv_s', self.forward_inv_s())
        # forward_inv_s = upsample_inv_s / 4.
        
        #----------------
        # Prepare outputs
        #----------------
        raw_ret = dict()
        if return_buffer:
            raw_ret['volume_buffer'] = dict(buffer_type='empty', ray_inds_hit=[])
        if return_details:
            details = raw_ret['details'] = {}
            if self.accel is not None:
                details['accel'] = self.accel.debug_stats()
            details['inv_s'] = forward_inv_s.item() if isinstance(forward_inv_s, torch.Tensor) else forward_inv_s
            details['s'] = 1./ details['inv_s']
            if hasattr(self, 'radiance_net') and hasattr(self.radiance_net, 'blocks') \
                and hasattr(self.radiance_net.blocks, 'lipshitz_bound_full'):
                details['radiance.lipshitz_bound'] = self.radiance_net.blocks.lipshitz_bound_full().item()
                
        if render_per_obj:
            prefix_rays = ray_input['rays_o'].shape[:-1]
            rendered = raw_ret['rendered'] = dict(
                depth_volume = torch.zeros([*prefix_rays], dtype=dtype, device=device),
                mask_volume = torch.zeros([*prefix_rays], dtype=dtype, device=device),
            )
            if with_rgb:
                rendered['rgb_volume'] = torch.zeros([*prefix_rays, 3], dtype=dtype, device=device)
            if with_normal:
                rendered['normals_volume'] = torch.zeros([*prefix_rays, 3], dtype=dtype, device=device)

        if ray_tested['num_rays'] == 0:
            return raw_ret
        
        #----------------
        # Ray query
        #----------------
        if query_mode == 'raymarch+batchup+jianfei':
            volume_buffer = self._ray_query_march_occ_multi_upsample(
                ray_tested, with_rgb=with_rgb, with_normal=with_normal, 
                perturb=config.perturb, forward_inv_s=forward_inv_s, **config.query_param)
        elif query_mode == 'raymarch+batchup+compression+jianfei':
            volume_buffer = self._ray_query_march_occ_multi_upsample_compressed(
                ray_tested, with_rgb=with_rgb, with_normal=with_normal, 
                perturb=config.perturb, forward_inv_s=forward_inv_s, **config.query_param)
        elif query_mode == 'raymarch+batchup+compression+timevar+jianfei':
            volume_buffer = self._ray_query_march_occ_multi_upsample_compressed_strategy(
                ray_tested, with_rgb=with_rgb, with_normal=with_normal, 
                perturb=config.perturb, forward_inv_s=forward_inv_s, **config.query_param)
        elif query_mode == 'coarse+batchup':
            volume_buffer = self._ray_query_coarse_multi_upsample(
                ray_tested, with_rgb=with_rgb, with_normal=with_normal, 
                perturb=config.perturb, forward_inv_s=forward_inv_s, **config.query_param)
        elif query_mode == 'only_raymarch':
            volume_buffer = self._ray_query_march_occ(
                ray_tested, with_rgb=with_rgb, with_normal=with_normal, 
                perturb=config.perturb, forward_inv_s=forward_inv_s, **config.query_param)
        else:
            raise RuntimeError(f"Invalid query_mode={query_mode}")

        #----------------
        # Calc outputs
        #----------------
        if return_buffer:
            raw_ret['volume_buffer'] = volume_buffer
        
        if return_details:
            if config.get('with_near_sdf', False):
                x_near = torch.addcmul(ray_tested['rays_o'], ray_tested['rays_d'], ray_tested['near'].unsqueeze(-1))
                details['near_sdf'] = self.forward_sdf(x_near)['sdf']
        
        if render_per_obj:
            if (buffer_type:=volume_buffer['buffer_type']) != 'empty':
                ray_inds_hit = volume_buffer['ray_inds_hit']
                depth_use_normalized_vw = config.get('depth_use_normalized_vw', True)
                
                if buffer_type == 'batched':
                    volume_buffer['vw'] = vw = ray_alpha_to_vw(volume_buffer['opacity_alpha'])
                    vw_sum = rendered['mask_volume'][ray_inds_hit] = vw.sum(dim=-1)
                    if depth_use_normalized_vw:
                        # TODO: This can also be differed by training / non-training
                        vw_normalized = vw / (vw_sum.unsqueeze(-1)+1e-10)
                        rendered['depth_volume'][ray_inds_hit] = (vw_normalized * volume_buffer['t']).sum(dim=-1)
                    else:
                        rendered['depth_volume'][ray_inds_hit] = (vw * volume_buffer['t']).sum(dim=-1)
                        
                    if with_rgb:
                        rendered['rgb_volume'][ray_inds_hit] = (vw.unsqueeze(-1) * volume_buffer['rgb']).sum(dim=-2)
                    if with_normal:
                        # rendered['normals_volume'][ray_inds_hit] = (vw.unsqueeze(-1) * volume_buffer['nablas']).sum(dim=-2)
                        if self.training:
                            rendered['normals_volume'][ray_inds_hit] = (vw.unsqueeze(-1) * volume_buffer['nablas']).sum(dim=-2)
                        else:
                            rendered['normals_volume'][ray_inds_hit] = (vw.unsqueeze(-1) * F.normalize(volume_buffer['nablas'].clamp_(-1,1), dim=-1)).sum(dim=-2)
                elif buffer_type == 'packed':
                    pack_infos_hit = volume_buffer['pack_infos_hit']
                    # [num_sampels]
                    volume_buffer['vw'] = vw = packed_alpha_to_vw(volume_buffer['opacity_alpha'], pack_infos_hit)
                    # [num_rays_hit]
                    vw_sum = rendered['mask_volume'][ray_inds_hit] = packed_sum(vw.view(-1), pack_infos_hit)
                    # [num_samples]
                    if depth_use_normalized_vw:
                        vw_normalized = packed_div(vw, vw_sum + 1e-10, pack_infos_hit)
                        rendered['depth_volume'][ray_inds_hit] = packed_sum(vw_normalized * volume_buffer['t'].view(-1), pack_infos_hit)
                    else:
                        rendered['depth_volume'][ray_inds_hit] = packed_sum(vw.view(-1) * volume_buffer['t'].view(-1), pack_infos_hit)
                    if with_rgb:
                        rendered['rgb_volume'][ray_inds_hit] = packed_sum(vw.view(-1,1) * volume_buffer['rgb'].view(-1,3), pack_infos_hit)
                    if with_normal:
                        # rendered['normals_volume'][ray_inds_hit] = packed_sum(vw.view(-1,1) * volume_buffer['nablas'].view(-1,3), pack_infos_hit)
                        if self.training:
                            rendered['normals_volume'][ray_inds_hit] = packed_sum(vw.view(-1,1) * volume_buffer['nablas'].view(-1,3), pack_infos_hit)
                        else:
                            rendered['normals_volume'][ray_inds_hit] = packed_sum(vw.view(-1,1) * F.normalize(volume_buffer['nablas'].clamp_(-1,1), dim=-1).view(-1,3), pack_infos_hit)
        return raw_ret
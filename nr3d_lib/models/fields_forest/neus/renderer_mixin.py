"""
@file   renderer_mixin.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Renderer mixin for forest neus with acceleration.
"""

__all__ = [
    'NeuSRendererMixinForest'
]

import functools
from operator import itemgetter
from typing import Dict, List, Literal, Tuple

import torch
import torch.nn.functional as F

from nr3d_lib.logger import Logger

from nr3d_lib.models.model_base import ModelMixin
from nr3d_lib.models.utils import batchify_query
from nr3d_lib.models.spatial import ForestBlockSpace
from nr3d_lib.models.accelerations import get_accel, accel_types_forest_t

from nr3d_lib.graphics.raymarch import RaymarchRetForest
from nr3d_lib.graphics.raysample import packed_sample_cdf, packed_sample_pdf
from nr3d_lib.graphics.pack_ops import get_pack_infos_from_batch, merge_two_packs_sorted_a_includes_b, merge_two_packs_sorted_aligned, packed_cumsum, packed_diff, packed_sum, packed_div
from nr3d_lib.graphics.nerf import packed_volume_render_compression, ray_alpha_to_vw, packed_alpha_to_vw
from nr3d_lib.graphics.neus import *

class NeuSRendererMixinForest(ModelMixin):
    def __init__(
        self, 
        # Renderer mixin kwargs
        ray_query_cfg: dict = dict(), 
        accel_cfg: dict = None, 
        # Network kwargs
        **net_kwargs
        ) -> None:
        
        mro = type(self).mro()
        super_class = mro[mro.index(NeuSRendererMixinForest)+1]
        assert super_class is not ModelMixin, "Incorrect class inheritance. Three possible misuse scenarios:\n"\
            "Case 1: The Net class for mixin should also inherit from `ModelMixin`.\n"\
            "Case 2: RendererMixin should come before the Net class when inheriting.\n"\
            "Case 3: You should not directly instantiate this mixin class."
        
        super().__init__(**net_kwargs) # Will call network's __init__() (e.g. LoTDForestNeuS ...)
        
        self.ray_query_cfg = ray_query_cfg
        self.accel: accel_types_forest_t = None if accel_cfg is None else \
            get_accel(space=self.space, device=self.device, **accel_cfg)
        self.upsample_s_divisor = 1.0
    
    def populate(self, **kwargs):
        super().populate(**kwargs)
        self.accel.populate()
    
    # @property
    # def space(self) -> ForestBlockSpace:
    #     return super().space
    
    def sample_pts_uniform(self, num_samples: int):
        # NOTE: Returns normalized `block_x`
        block_x, blidx = self.space.sample_pts_uniform(num_samples)
        ret = self.forward_sdf_nablas(block_x, blidx, input_normalized=True, skip_accel=True) # Do not upsate_samples here (usally there are too less samples here.)
        ret = {k: v.to(block_x.dtype) for k, v in ret.items()}
        ret['net_x'] = block_x # NOTE: in network's uniformed space; not in world space.
        if 'nablas' in ret:
            ret['nablas_norm'] = ret['nablas'].norm(dim=-1)
        return ret
    
    def sample_pts_in_occupied(self, num_samples: int):
        assert self.accel is not None, "Requires self.accel not to be None"
        block_x, blidx = self.accel.sample_pts_in_occupied(num_samples)
        ret = self.forward_sdf_nablas(block_x, blidx, input_normalized=True, skip_accel=True) # Do not upsate_samples here (usally there are too less samples here.)
        ret = {k: v.to(block_x.dtype) for k, v in ret.items()}
        ret['net_x'] = block_x # NOTE: in network's uniformed space; not in world space.
        if 'nablas' in ret:
            ret['nablas_norm'] = ret['nablas'].norm(dim=-1)
        return ret
    
    def forward_sdf(
        self, 
        x: torch.Tensor, block_inds: torch.Tensor=None, block_offsets: torch.Tensor=None, 
        skip_accel=False, input_normalized=False, **kwargs):
        ret = super().forward_sdf(x, block_inds, block_offsets, input_normalized=input_normalized, **kwargs)
        if (not skip_accel) and (self.accel is not None):
            self.accel.collect_samples(x, block_inds, ret['sdf'].data, normalized=input_normalized)
        return ret
    
    def forward_sdf_nablas(
        self, 
        x: torch.Tensor, block_inds: torch.Tensor=None, block_offsets: torch.Tensor=None, 
        skip_accel=False, input_normalized=False, **kwargs):
        ret = super().forward_sdf_nablas(x, block_inds, block_offsets, input_normalized=input_normalized, **kwargs)
        if (not skip_accel) and (self.accel is not None):
            self.accel.collect_samples(x, block_inds, ret['sdf'].data, normalized=input_normalized)
        return ret
    
    def query_sdf(self, x: torch.Tensor, block_inds: torch.Tensor=None, block_offsets: torch.Tensor=None, 
                  input_normalized=False, **kwargs):
        return super().query_sdf(x, block_inds=block_inds, block_offsets=block_offsets, 
                                 input_normalized=input_normalized, **kwargs)
    
    def training_initialize(self, config=dict(), logger=None, log_prefix: str = None, skip_accel=False) -> bool:
        updated = super().training_initialize(config, logger=logger, log_prefix=log_prefix)
        if (not skip_accel) and (self.accel is not None):
            # NOTE: Important to set input_normalized!
            self.accel.init(functools.partial(self.query_sdf, input_normalized=True), logger=logger)
        return updated
    
    def training_before_per_step(self, cur_it: int, logger: Logger = None, skip_accel=False):
        super().training_before_per_step(cur_it, logger=logger)
        if (not skip_accel) and (self.accel is not None):
            if self.training:
                # NOTE: Important to set input_normalized!
                val_query_fn = functools.partial(self.query_sdf, input_normalized=True)
                self.accel.step(cur_it, val_query_fn, logger)
                # if cur_it == 0:
                #     self.accel.init(val_query_fn, logger)
                # else:
                #     self.accel.step(cur_it, val_query_fn, logger)
            self.upsample_s_divisor = 2 ** self.accel.training_granularity
    
    def training_after_per_step(self, cur_it: int, logger: Logger = None):
        super().training_after_per_step(cur_it, logger=logger)
    
    def ray_test(self, rays_o: torch.Tensor, rays_d: torch.Tensor, near :float=None, far: float=None, return_rays=True, **extra_ray_data):
        assert (rays_o.dim() == 2) and (rays_d.dim() == 2), "Expect rays_o and rays_d to be of shape [N, 3]"
        return self.space.ray_test(rays_o, rays_d, near=near, far=far, return_rays=return_rays, **extra_ray_data)
    
    def _forest_ray_query_inblock_coarse_multi_upsample(
        self, ray_tested: Dict[str, torch.Tensor], 
        # Common params
        with_rgb: bool = True, with_normal: bool = True, 
        perturb: bool = False, nablas_has_grad: bool = False, forward_inv_s: float = None, 
        # Distinct params
        coarse_step_cfg = dict(), step_size_nograd: float = 0.01, 
        chunksize_query: int = 2**24, 
        upsample_mode: Literal['direct_use',  'direct_more', 'multistep_estimate'] = 'direct_more', 
        num_fine: int = 64, upsample_inv_s: float = 64., upsample_inv_s_factors: List[int] = [1, 4, 16], upsample_use_estimate_alpha=False
        ) -> Tuple[dict, dict]:
        upsample_inv_s /= self.upsample_s_divisor
        use_view_dirs = self.use_view_dirs
        forward_inv_s = self.forward_inv_s() if forward_inv_s is None else forward_inv_s
        # forward_inv_s = config.upsample_inv_s / self.upsample_s_divisor / 4.

        empty_volume_buffer = dict(type='empty')
        if (num_rays:=ray_tested['num_rays']) == 0:
            return empty_volume_buffer, {}
        
        rays_o, rays_d, near, far, rays_inds, seg_pack_infos, seg_block_inds, seg_entries, seg_exits = \
            itemgetter('rays_o', 'rays_d', 'near', 'far', 'rays_inds', 'seg_pack_infos', 'seg_block_inds', 'seg_entries', 'seg_exits')(ray_tested)
        assert (rays_o.dim() == 2) and (rays_d.dim()==2)
        
        # NOTE: The device & dtype of output
        device, dtype = rays_o.device, rays_o.dtype
        # NOTE: The spatial length scale on each ray caused by scaling rays_d 
        dir_scale = rays_d.data.norm(dim=-1)  # [num_rays]
        # NOTE: The normalized ray direction vector in network's space
        view_dirs = rays_d / dir_scale.clamp_min_(1.0e-10).unsqueeze(-1) # [num_rays, 3]
        
        #----------------
        # Coarse sampling
        #----------------
        coarse_ret = self.space.ray_step_coarse(rays_o, rays_d, near, far, seg_block_inds, seg_entries, seg_exits, seg_pack_infos, **coarse_step_cfg)

        #----------------
        # Up Sampling
        #----------------
        @torch.no_grad()
        def upsample():
            def merge(ret1: RaymarchRetForest, d2: torch.Tensor, blidx2: torch.Tensor, return_more=False):
                num_samples = ret1.depth_samples.numel() + d2.numel()
                ridx = ret1.ridx.new_empty([num_samples])
                blidx = ret1.blidx.new_empty([num_samples])
                depths_packed = ret1.depth_samples.new_empty([num_samples])
                pinfo_fine = get_pack_infos_from_batch(ret1.ridx_hit.numel(), d2.shape[-1], device=device)
                pidx0, pidx1, pack_infos = merge_two_packs_sorted_aligned(ret1.depth_samples, ret1.pack_infos, d2.flatten(), pinfo_fine, b_sorted=True, return_val=False)
                depths_packed[pidx0], depths_packed[pidx1] = ret1.depth_samples, d2.flatten()
                ridx[pidx0], ridx[pidx1] = ret1.ridx, ret1.ridx_hit.unsqueeze(-1).expand(-1,d2.shape[-1]).flatten()
                blidx[pidx0], blidx[pidx1] = ret1.blidx, blidx2.flatten()
                if return_more:
                    return depths_packed, ridx, blidx, pack_infos, num_samples, pidx0, pidx1
                else:
                    return depths_packed, ridx, blidx, pack_infos
            
            if upsample_mode == 'direct_use':
                sdf_coarse = self.implicit_surface.forward(coarse_ret.samples, coarse_ret.blidx)['sdf']
                if upsample_use_estimate_alpha:
                    alpha_coarse = neus_packed_sdf_to_upsample_alpha(sdf_coarse, coarse_ret.depth_samples, upsample_inv_s, coarse_ret.pack_infos)
                else:
                    alpha_coarse = neus_packed_sdf_to_alpha(sdf_coarse, upsample_inv_s, coarse_ret.pack_infos)
                vw_coarse = packed_alpha_to_vw(alpha_coarse)
                # Fine points
                d_fine, pidx_fine = packed_sample_pdf(coarse_ret.depth_samples, vw_coarse, coarse_ret.pack_infos, num_fine, perturb=perturb)
                # Gather points
                d_all, ridx_all, blidx_all, pack_infos = merge(coarse_ret, d_fine, coarse_ret.blidx[pidx_fine])
                return d_all, ridx_all, blidx_all, pack_infos
            
            elif upsample_mode == 'direct_more':
                _ret = self.space.ray_step_coarse(
                    rays_o, rays_d, near, far, seg_block_inds, seg_entries, seg_exits, seg_pack_infos, 
                    step_mode='linear', step_size=step_size_nograd, max_steps=4096, perturb=perturb)
                _sdf = batchify_query(self.implicit_surface.forward, _ret.samples, _ret.blidx, chunk=chunksize_query)['sdf']
                if upsample_use_estimate_alpha:
                    _alpha = neus_packed_sdf_to_upsample_alpha(_sdf, _ret.depth_samples, upsample_inv_s, _ret.pack_infos)
                else:
                    _alpha = neus_packed_sdf_to_alpha(_sdf, upsample_inv_s, _ret.pack_infos)
                _vw = packed_alpha_to_vw(_alpha)
                # Fine points
                d_fine, pidx_fine = packed_sample_pdf(_ret.depth_samples, _vw, _ret.pack_infos, num_fine, perturb=perturb)
                # Gather points
                d_all, ridx_all, blidx_all, pack_infos = merge(_ret, d_fine, _ret.blidx[pidx_fine])
                return d_all, ridx_all, blidx_all, pack_infos
            
            elif upsample_mode == 'multistep_estimate':
                depths_1 = []
                collect_all = RaymarchRetForest(coarse_ret.ridx_hit.numel(), coarse_ret.ridx_hit, None, coarse_ret.depth_samples, None, 
                                                coarse_ret.ridx, coarse_ret.pack_infos, coarse_ret.blidx, None, None, None)
                sdf_all = self.implicit_surface.forward(coarse_ret.samples, coarse_ret.blidx)['sdf']
                num_fine_per_iter = num_fine//2*2+1 # Had better always be odd
                for i, factor in enumerate(upsample_inv_s_factors):
                    if upsample_use_estimate_alpha:
                        alpha = neus_packed_sdf_to_upsample_alpha(sdf_all, collect_all.depth_samples, upsample_inv_s * factor, collect_all.pack_infos)
                    else:
                        alpha = neus_packed_sdf_to_alpha(sdf_all, upsample_inv_s * factor, collect_all.pack_infos)
                    vw = packed_alpha_to_vw(alpha, collect_all.pack_infos)
                    d_fine_iter, pidx_fine_iter = packed_sample_pdf(collect_all.depth_samples, vw, collect_all.pack_infos, num_fine_per_iter, perturb=perturb)
                    blidx_fine_iter = collect_all.blidx[pidx_fine_iter]
                    depths_1.append(d_fine_iter)
                    if len(upsample_inv_s_factors) > 1:
                        collect_all.depth_samples, collect_all.ridx, collect_all.blidx, collect_all.pack_infos, num_samples, pidx0, pidx1 = \
                            merge(collect_all, d_fine_iter, blidx_fine_iter, return_more=True)
                        if i < len(upsample_inv_s_factors)-1:
                            x_fine = torch.addcmul(rays_o.unsqueeze(-2), rays_d.unsqueeze(-2), d_fine_iter.unsqueeze(-1))
                            sdf_iter = sdf_all.new_empty([num_samples])
                            sdf_iter[pidx0], sdf_iter[pidx1] = sdf_all, self.forward_sdf(x_fine.flatten(0,-2), blidx_fine_iter.flatten())['sdf']
                            sdf_all = sdf_iter
                return collect_all.depth_samples, collect_all.ridx, collect_all.blidx, collect_all.pack_infos
            
            else:
                raise RuntimeError(f"Invalid upsample_mode={upsample_mode}")
        
        #----------------
        # Acquire volume_buffer via quering network and gather results
        #----------------
        d_all, ridx_all, blidx_all, pack_infos = upsample()
        d_mid = d_all + packed_diff(d_all, pack_infos)/2.
        rays_o_packed, rays_d_packed = rays_o[ridx_all], rays_d[ridx_all]
        pts = torch.addcmul(rays_o_packed, rays_d_packed, d_all.unsqueeze(-1))
        alpha = neus_packed_sdf_to_alpha(self.forward_sdf(pts, blidx_all)['sdf'], forward_inv_s, pack_infos)
        
        volume_buffer = dict(
            type='packed', 
            rays_inds_hit=rays_inds, pack_infos_hit=pack_infos, 
            t=d_mid.to(dtype), opacity_alpha=alpha.to(dtype))
        
        if with_rgb or with_normal:
            h_appear = ray_tested['rays_h_appear'][ridx_all] \
                if ray_tested.get('rays_h_appear', None) is not None else None
            net_out = self.forward(
                torch.addcmul(rays_o_packed, rays_d_packed, d_mid.unsqueeze(-1)), 
                view_dirs[ridx_all] if use_view_dirs else None, blidx_all,
                h_appear=h_appear, 
                nablas_has_grad=nablas_has_grad, with_rgb=with_rgb, with_normal=with_normal)
            if "nablas" in net_out: volume_buffer["nablas"] = net_out["nablas"].to(dtype)
            if "rgb" in net_out: volume_buffer["rgb"] = net_out["rgb"].to(dtype)
        details = {'render.num_per_ray': pack_infos[:, 1]}

        return volume_buffer, details

    def _forest_ray_query_inblock_march_occ_multi_upsample(
        self, ray_tested: Dict[str, torch.Tensor], 
        # Common params
        with_rgb: bool = True, with_normal: bool = True, 
        perturb: bool = False, nablas_has_grad: bool = False, forward_inv_s: float = None, 
        # Distinct params
        coarse_step_cfg: dict = None, # Optional
        chunksize_query: int = 2**24, march_cfg = dict(), num_fine: int = 8, 
        upsample_inv_s: float = 64., upsample_inv_s_factors: List[int] = [1, 4, 16], upsample_use_estimate_alpha=False
        ) -> Tuple[dict, dict]:
        assert self.accel is not None, "Need a non-empty AccelStruct"
                
        upsample_inv_s /= self.upsample_s_divisor
        use_view_dirs = self.use_view_dirs
        forward_inv_s = self.forward_inv_s() if forward_inv_s is None else forward_inv_s
        
        empty_volume_buffer = dict(type='empty', rays_inds_hit=[])
        
        if (num_rays:=ray_tested['num_rays']) == 0:
            return empty_volume_buffer, {}
        
        rays_o, rays_d, near, far, rays_inds, seg_pack_infos, seg_block_inds, seg_entries, seg_exits = \
            itemgetter('rays_o', 'rays_d', 'near', 'far', 'rays_inds', 'seg_pack_infos', 'seg_block_inds', 'seg_entries', 'seg_exits')(ray_tested)
        assert (rays_o.dim() == 2) and (rays_d.dim()==2)
        
        # NOTE: The device & dtype of output
        device, dtype = rays_o.device, rays_o.dtype
        
        # NOTE: The spatial length scale on each ray caused by scaling rays_d 
        dir_scale = rays_d.data.norm(dim=-1)  # [num_rays]
        # NOTE: The normalized ray direction vector in network's space
        view_dirs = rays_d / dir_scale.clamp_min_(1.0e-10).unsqueeze(-1) # [num_rays, 3]

        #----------------
        # Coarse sampling
        #----------------
        if coarse_step_cfg is not None:
            coarse_ret = self.space.ray_step_coarse(rays_o, rays_d, near, far, seg_block_inds, seg_entries, seg_exits, seg_pack_infos, **coarse_step_cfg)
        
        #----------------
        # Ray marching
        #----------------
        ridx_hit, samples, depth_samples, _, _, ray_pack_infos, blidx, blidx_pack_infos, _, _ = self.accel.ray_march(
            rays_o, rays_d, near, far, seg_block_inds, seg_entries, seg_exits, seg_pack_infos, perturb=perturb, **march_cfg)

        #----------------
        # Upsampling & gather volume_buffer
        #----------------
        if ridx_hit is not None:
            pack_infos_marched = ray_pack_infos.clone()
            rays_o_hit, rays_d_hit = rays_o[ridx_hit].unsqueeze(-2), rays_d[ridx_hit].unsqueeze(-2)

            #----------------
            # Upsample on marched samples
            with torch.no_grad():
                sdf = batchify_query(lambda x,b: self.forward_sdf(x, b)['sdf'], samples, blidx, chunk=chunksize_query)
                depths_1 = []
                blidxs_1 = []
                
                num_fine_per_iter = num_fine//2*2+1 # Had better always be odd
                pinfo_fine_per_iter = get_pack_infos_from_batch(ridx_hit.numel(), num_fine_per_iter, device=device)
                for i, factor in enumerate(upsample_inv_s_factors):
                    if upsample_use_estimate_alpha:
                        alpha = neus_packed_sdf_to_upsample_alpha(sdf, depth_samples, upsample_inv_s * factor, ray_pack_infos)
                    else:
                        alpha = neus_packed_sdf_to_alpha(sdf, upsample_inv_s * factor, ray_pack_infos)
                    vw = packed_alpha_to_vw(alpha, ray_pack_infos)
                    
                    neus_cdf = packed_cumsum(vw, ray_pack_infos, exclusive=True)
                    last_cdf = neus_cdf[ray_pack_infos.sum(-1).sub_(1)]
                    neus_cdf = packed_div(neus_cdf, last_cdf.clamp_min(1e-5), ray_pack_infos)
                    
                    depths_fine_iter, pidx_fine_iter = packed_sample_cdf(depth_samples, neus_cdf.to(depth_samples.dtype), ray_pack_infos, num_fine_per_iter, perturb=perturb)
                    blidx_fine_iter = blidx[pidx_fine_iter]
                    depths_1.append(depths_fine_iter)
                    blidxs_1.append(blidx_fine_iter)
                    
                    if len(upsample_inv_s_factors) > 1:
                        # Merge fine samples of current upsample iter to previous packed buffer.
                        # NOTE: `ray_pack_infos` is updated.
                        pidx0, pidx1, ray_pack_infos = merge_two_packs_sorted_aligned(depth_samples, ray_pack_infos, depths_fine_iter.flatten(), pinfo_fine_per_iter, b_sorted=True, return_val=False)
                        num_samples_iter = depth_samples.numel()
                        depth_samples_iter = depth_samples.new_empty([num_samples_iter + depths_fine_iter.numel()])
                        depth_samples_iter[pidx0], depth_samples_iter[pidx1] = depth_samples, depths_fine_iter.flatten()
                        # NOTE: `depth_samples` is updated.
                        depth_samples = depth_samples_iter

                        if i < len(upsample_inv_s_factors)-1:
                            x_fine = torch.addcmul(rays_o_hit, rays_d_hit, depths_fine_iter.unsqueeze(-1))
                            sdf_iter = sdf.new_empty([num_samples_iter + depths_fine_iter.numel()])
                            blidx_iter = blidx.new_empty([num_samples_iter + depths_fine_iter.numel()])
                            sdf_iter[pidx0], sdf_iter[pidx1] = sdf, self.forward_sdf(x_fine, blidx_fine_iter)['sdf'].flatten()
                            blidx_iter[pidx0], blidx_iter[pidx1] = blidx, blidx_fine_iter.flatten()
                            
                            # NOTE: `sdf` and `blidx` is updated.
                            sdf = sdf_iter
                            blidx = blidx_iter
                
                if len(upsample_inv_s_factors) > 1:
                    depths_1, sort_indices = torch.cat(depths_1, dim=-1).sort(dim=-1)
                    blidxs_1 = torch.cat(blidxs_1, -1).gather(-1, sort_indices)
                else:
                    depths_1 = depths_1[0]
                    blidxs_1 = depths_1[0]
            
            #----------------
            # Acquire volume_buffer via quering network and gather results
            if coarse_step_cfg is None:
                alpha = neus_ray_sdf_to_alpha(
                    self.forward_sdf(torch.addcmul(rays_o_hit, rays_d_hit, depths_1.unsqueeze(-1)), blidxs_1)["sdf"],
                    forward_inv_s)
                depths = (depths_1[..., :-1] + depths_1.diff(dim=-1)/2.)
                
                volume_buffer = dict(
                    type='batched', 
                    rays_inds_hit=rays_inds[ridx_hit], num_per_hit=depths.shape[-1], 
                    t=depths.to(dtype), opacity_alpha=alpha.to(dtype))
                
                if with_rgb or with_normal:
                    h_appear = ray_tested['rays_h_appear'][ridx_hit].unsqueeze(-2) \
                        if ray_tested.get('rays_h_appear', None) is not None else None
                    #----------- Net forward
                    # [num_rays_hit, num_fine_all, ...]
                    net_out = self.forward(
                        torch.addcmul(rays_o_hit, rays_d_hit, depths.unsqueeze(-1)), 
                        view_dirs[ridx_hit].unsqueeze(-2) if use_view_dirs else None, 
                        None, # NOTE: Not using `blidxs_1`. Let the forest to do the normalization because `depths` is calculated using non-reliable diff's delta.
                        h_appear=h_appear, 
                        nablas_has_grad=nablas_has_grad, with_rgb=with_rgb, with_normal=with_normal)
                    if "nablas" in net_out: volume_buffer["nablas"] = net_out["nablas"].to(dtype)
                    if "rgb" in net_out: volume_buffer["rgb"] = net_out["rgb"].to(dtype)
                details = {'march.num_per_ray': pack_infos_marched[:, 1], 'render.num_per_ray': depths.shape[-1]}
                return volume_buffer, details
            
            else:
                def merge():
                    pinfo_fine = get_pack_infos_from_batch(ridx_hit.numel(), depths_1.shape[-1], device=device)
                    pidx0, pidx1, pack_infos = merge_two_packs_sorted_a_includes_b(
                        coarse_ret.depth_samples, coarse_ret.pack_infos, coarse_ret.ridx_hit, 
                        depths_1.flatten(), pinfo_fine, ridx_hit, b_sorted=True, return_val=False)
                    num_samples = depths_1.numel() + coarse_ret.depth_samples.numel()
                    depths_1_packed = depths_1.new_zeros([num_samples])
                    ridx_all = ridx_hit.new_zeros([num_samples])
                    blidx_all = blidx.new_zeros([num_samples])
                    ridx_all[pidx0], ridx_all[pidx1] = coarse_ret.ridx, ridx_hit.unsqueeze(-1).expand(-1,depths_1.shape[-1]).flatten()
                    blidx_all[pidx0], blidx_all[pidx1] = coarse_ret.blidx, blidxs_1.flatten()
                    depths_1_packed[pidx0], depths_1_packed[pidx1] = coarse_ret.depth_samples, depths_1.flatten()
                    return ridx_all, blidx_all, depths_1_packed, pack_infos
                
                ridx_all, blidx_all, depths_1_packed, pack_infos_hit = merge()
                rays_o_packed, rays_d_packed = rays_o[ridx_all], rays_d[ridx_all]
                alpha_packed = neus_packed_sdf_to_alpha(self.forward_sdf(torch.addcmul(rays_o_packed, rays_d_packed, depths_1_packed.unsqueeze(-1)), blidx_all)['sdf'], forward_inv_s, pack_infos_hit)
                depths_packed = depths_1_packed + packed_diff(depths_1_packed, pack_infos_hit) / 2.
                
                volume_buffer = dict(
                    type='packed', 
                    rays_inds_hit=rays_inds, pack_infos_hit=pack_infos_hit, 
                    t=depths_packed.to(dtype), opacity_alpha=alpha_packed.to(dtype))
                
                if with_rgb or with_normal:
                    h_appear = ray_tested['rays_h_appear'][ridx_all] \
                        if ray_tested.get('rays_h_appear', None) is not None else None
                    net_out = self.forward(
                        torch.addcmul(rays_o_packed, rays_d_packed, depths_packed.unsqueeze(-1)), 
                        view_dirs[ridx_all] if use_view_dirs else None, 
                        None, # NOTE: Not using `blidx_all`. Let the forest to do the normalization because `depths` is calculated using non-reliable diff's delta.
                        h_appear=h_appear, 
                        nablas_has_grad=nablas_has_grad, with_rgb=with_rgb, with_normal=with_normal)
                    if "nablas" in net_out: volume_buffer["nablas"] = net_out['nablas'].to(dtype)
                    if "rgb" in net_out: volume_buffer["rgb"] = net_out['rgb'].to(dtype)
                details = {'march.num_per_ray': pack_infos_marched[:, 1], 'render.num_per_ray': pack_infos_hit[:, 1]}
                return volume_buffer, details
                
        else:
            if coarse_step_cfg is None:
                return empty_volume_buffer, {}
            else:
                alpha_coarse = neus_packed_sdf_to_alpha(self.forward_sdf(coarse_ret.samples, coarse_ret.blidx)['sdf'], forward_inv_s, coarse_ret.pack_infos)
                depths_coarse = coarse_ret.depth_samples + coarse_ret.deltas/2.
                
                volume_buffer = dict(
                    type='packed', 
                    rays_inds_hit=rays_inds[coarse_ret.ridx_hit], pack_infos_hit=coarse_ret.pack_infos, 
                    t=depths_coarse.to(dtype), opacity_alpha=alpha_coarse.to(dtype))
                
                if with_rgb or with_normal:
                    h_appear = ray_tested['rays_h_appear'][coarse_ret.ridx] \
                        if ray_tested.get('rays_h_appear', None) is not None else None
                    net_out = self.forward(
                        torch.addcmul(rays_o[coarse_ret.ridx], rays_d[coarse_ret.ridx], depths_coarse.unsqueeze(-1)), 
                        view_dirs[coarse_ret.ridx] if use_view_dirs else None, 
                        # TODO: According to experiments, using the original blidx here still generates a lot of "out-of-bounds points", i.e., points that have exceeded the original block range significantly;
                        #       For these points, the forward process will be forcibly clipped to ensure no runtime bugs, but in terms of computation logic, the coordinates of these points are clamped quite a bit, deviating from the light ray's range significantly
                        #       (because clamping is equivalent to moving the point along the coordinate system's diagonal, not along the direction of the light ray).
                        coarse_ret.blidx, 
                        h_appear=h_appear, 
                        nablas_has_grad=nablas_has_grad, with_rgb=with_rgb, with_normal=with_normal)
                    if "nablas" in net_out: volume_buffer["nablas"] = net_out['nablas'].to(dtype)
                    if "rgb" in net_out: volume_buffer["rgb"] = net_out['rgb'].to(dtype)
                details = {'render.num_per_ray': coarse_ret.pack_infos[:, 1]}
                return volume_buffer, details

    def _forest_ray_query_inblock_march_occ_multi_upsample_compressed(
        self, ray_tested: Dict[str, torch.Tensor], 
        # Common params
        with_rgb: bool = True, with_normal: bool = True, 
        perturb: bool = False, nablas_has_grad: bool = False, forward_inv_s: float = None, 
        # Distinct params
        coarse_step_cfg: dict = None, # Optional coarse sampling
        chunksize_query: int = 2**24, march_cfg = dict(), num_fine: int = 8, 
        upsample_inv_s: float = 64., upsample_inv_s_factors: List[int] = [1, 4, 16], upsample_use_estimate_alpha=False
        ) -> Tuple[dict, dict]:
        assert self.accel is not None, "Need a non-empty AccelStruct"
                
        upsample_inv_s /= self.upsample_s_divisor
        use_view_dirs = self.use_view_dirs
        forward_inv_s = self.forward_inv_s() if forward_inv_s is None else forward_inv_s
        
        empty_volume_buffer = dict(type='empty', rays_inds_hit=[])
        
        if (num_rays:=ray_tested['num_rays']) == 0:
            return empty_volume_buffer, {}
        
        rays_o, rays_d, near, far, rays_inds, seg_pack_infos, seg_block_inds, seg_entries, seg_exits = \
            itemgetter('rays_o', 'rays_d', 'near', 'far', 'rays_inds', 'seg_pack_infos', 'seg_block_inds', 'seg_entries', 'seg_exits')(ray_tested)
        assert (rays_o.dim() == 2) and (rays_d.dim()==2)
        
        # NOTE: The device & dtype of output
        device, dtype = rays_o.device, rays_o.dtype
        
        # NOTE: The spatial length scale on each ray caused by scaling rays_d 
        dir_scale = rays_d.data.norm(dim=-1)  # [num_rays]
        # NOTE: The normalized ray direction vector in network's space
        view_dirs = rays_d / dir_scale.clamp_min_(1.0e-10).unsqueeze(-1) # [num_rays, 3]

        #----------------
        # Coarse sampling
        #----------------
        if coarse_step_cfg is not None:
            coarse_ret = self.space.ray_step_coarse(rays_o, rays_d, near, far, seg_block_inds, seg_entries, seg_exits, seg_pack_infos, **coarse_step_cfg)
        
        #----------------
        # Ray marching
        #----------------
        ridx_hit, samples, depth_samples, _, _, ray_pack_infos, blidx, blidx_pack_infos = self.accel.ray_march(
            rays_o, rays_d, near, far, seg_block_inds, seg_entries, seg_exits, seg_pack_infos, perturb=perturb, **march_cfg)

        #----------------
        # Upsampling & gather volume_buffer
        #----------------
        if ridx_hit is not None:
            pack_infos_marched = ray_pack_infos.clone()
            rays_o_hit, rays_d_hit = rays_o[ridx_hit].unsqueeze(-2), rays_d[ridx_hit].unsqueeze(-2)

            #----------------
            # Upsample on marched samples
            with torch.no_grad():
                sdf = batchify_query(lambda x,b: self.forward_sdf(x, b)['sdf'], samples, blidx, chunk=chunksize_query)
                depths_1 = []
                blidxs_1 = []
                
                num_fine_per_iter = num_fine//2*2+1 # Had better always be odd
                pinfo_fine_per_iter = get_pack_infos_from_batch(ridx_hit.numel(), num_fine_per_iter, device=device)
                for i, factor in enumerate(upsample_inv_s_factors):
                    if upsample_use_estimate_alpha:
                        alpha = neus_packed_sdf_to_upsample_alpha(sdf, depth_samples, upsample_inv_s * factor, ray_pack_infos)
                    else:
                        alpha = neus_packed_sdf_to_alpha(sdf, upsample_inv_s * factor, ray_pack_infos)
                    vw = packed_alpha_to_vw(alpha, ray_pack_infos)
                    
                    neus_cdf = packed_cumsum(vw, ray_pack_infos, exclusive=True)
                    last_cdf = neus_cdf[ray_pack_infos.sum(-1).sub_(1)]
                    neus_cdf = packed_div(neus_cdf, last_cdf.clamp_min(1e-5), ray_pack_infos)
                    
                    depths_fine_iter, pidx_fine_iter = packed_sample_cdf(depth_samples, neus_cdf.to(depth_samples.dtype), ray_pack_infos, num_fine_per_iter, perturb=perturb)
                    blidx_fine_iter = blidx[pidx_fine_iter]
                    depths_1.append(depths_fine_iter)
                    blidxs_1.append(blidx_fine_iter)
                    
                    if len(upsample_inv_s_factors) > 1:
                        # Merge fine samples of current upsample iter to previous packed buffer.
                        # NOTE: `ray_pack_infos` is updated.
                        pidx0, pidx1, ray_pack_infos = merge_two_packs_sorted_aligned(depth_samples, ray_pack_infos, depths_fine_iter.flatten(), pinfo_fine_per_iter, b_sorted=True, return_val=False)
                        num_samples_iter = depth_samples.numel()
                        depth_samples_iter = depth_samples.new_empty([num_samples_iter + depths_fine_iter.numel()])
                        depth_samples_iter[pidx0], depth_samples_iter[pidx1] = depth_samples, depths_fine_iter.flatten()
                        # NOTE: `depth_samples` is updated.
                        depth_samples = depth_samples_iter

                        if i < len(upsample_inv_s_factors)-1:
                            x_fine = torch.addcmul(rays_o_hit, rays_d_hit, depths_fine_iter.unsqueeze(-1))
                            sdf_iter = sdf.new_empty([num_samples_iter + depths_fine_iter.numel()])
                            blidx_iter = blidx.new_empty([num_samples_iter + depths_fine_iter.numel()])
                            sdf_iter[pidx0], sdf_iter[pidx1] = sdf, self.forward_sdf(x_fine, blidx_fine_iter)['sdf'].flatten()
                            blidx_iter[pidx0], blidx_iter[pidx1] = blidx, blidx_fine_iter.flatten()
                            
                            # NOTE: `sdf` and `blidx` is updated.
                            sdf = sdf_iter
                            blidx = blidx_iter
                
                if len(upsample_inv_s_factors) > 1:
                    depths_1, sort_indices = torch.cat(depths_1, dim=-1).sort(dim=-1)
                    blidxs_1 = torch.cat(blidxs_1, -1).gather(-1, sort_indices)
                else:
                    depths_1 = depths_1[0]
                    blidxs_1 = depths_1[0]
            
            #----------------
            # Acquire volume_buffer via quering network and gather results
            if coarse_step_cfg is None:
                if self.training:
                    alpha = neus_ray_sdf_to_alpha(
                        self.forward_sdf(torch.addcmul(rays_o_hit, rays_d_hit, depths_1.unsqueeze(-1)), blidxs_1)['sdf'], 
                        forward_inv_s)
                else:
                    alpha = neus_ray_sdf_to_alpha(
                        batchify_query(
                            lambda x,b: self.forward_sdf(x, b)['sdf'], 
                            torch.addcmul(rays_o_hit, rays_d_hit, depths_1.unsqueeze(-1)), 
                            blidxs_1, chunk=chunksize_query), 
                        forward_inv_s)
                depths = (depths_1[..., :-1] + depths_1.diff(dim=-1)/2.)
                
                pack_infos_hit = get_pack_infos_from_batch(ridx_hit.numel(), depths.shape[-1], device=device)
                nidx_useful, pack_infos_hit_useful, pidx_useful = packed_volume_render_compression(alpha.flatten(), pack_infos_hit)
                
                if nidx_useful.numel() == 0:
                    return empty_volume_buffer, {}
                else:
                    depths_packed, alpha_packed = depths.flatten()[pidx_useful], alpha.flatten()[pidx_useful]
                    
                    volume_buffer = dict(
                        type='packed', 
                        rays_inds_hit=rays_inds[ridx_hit][nidx_useful], pack_infos_hit=pack_infos_hit_useful, 
                        t=depths_packed.to(dtype), opacity_alpha=alpha_packed.to(dtype))
                    
                    if with_rgb or with_normal:
                        ridx_all = ridx_hit.unsqueeze(-1).expand(-1,depths.shape[-1]).flatten()[pidx_useful]
                        h_appear = ray_tested['rays_h_appear'][ridx_all] \
                            if ray_tested.get('rays_h_appear', None) is not None else None
                        #----------- Net forward
                        # [num_rays_hit, num_fine_all, ...]
                        net_out = self.forward(
                            torch.addcmul(rays_o[ridx_all], rays_d[ridx_all], depths_packed.unsqueeze(-1)), 
                            view_dirs[ridx_all] if use_view_dirs else None, 
                            None, # NOTE: Not using `blidxs_1`. Let the forest to do the normalization because `depths` is calculated using non-reliable diff's delta.
                            h_appear=h_appear, 
                            nablas_has_grad=nablas_has_grad, with_rgb=with_rgb, with_normal=with_normal)
                        if "nablas" in net_out: volume_buffer["nablas"] = net_out["nablas"].to(dtype)
                        if "rgb" in net_out: volume_buffer["rgb"] = net_out["rgb"].to(dtype)
                    details = {'march.num_per_ray': pack_infos_marched[:, 1], 'render.num_per_ray0': depths.shape[-1], 'render.num_per_ray': pack_infos_hit_useful[:, 1]}
                    return volume_buffer, details
            
            else:
                def merge():
                    pinfo_fine = get_pack_infos_from_batch(ridx_hit.numel(), depths_1.shape[-1], device=device)
                    pidx0, pidx1, pack_infos = merge_two_packs_sorted_a_includes_b(
                        coarse_ret.depth_samples, coarse_ret.pack_infos, coarse_ret.ridx_hit, 
                        depths_1.flatten(), pinfo_fine, ridx_hit, b_sorted=True, return_val=False)
                    num_samples = depths_1.numel() + coarse_ret.depth_samples.numel()
                    depths_1_packed = depths_1.new_zeros([num_samples])
                    ridx_all = ridx_hit.new_zeros([num_samples])
                    blidx_all = blidx.new_zeros([num_samples])
                    ridx_all[pidx0], ridx_all[pidx1] = coarse_ret.ridx, ridx_hit.unsqueeze(-1).expand(-1,depths_1.shape[-1]).flatten()
                    blidx_all[pidx0], blidx_all[pidx1] = coarse_ret.blidx, blidxs_1.flatten()
                    depths_1_packed[pidx0], depths_1_packed[pidx1] = coarse_ret.depth_samples, depths_1.flatten()
                    return ridx_all, blidx_all, depths_1_packed, pack_infos
                
                ridx_all, blidx_all, depths_1_packed, pack_infos_hit = merge()
                depths_packed = depths_1_packed + packed_diff(depths_1_packed, pack_infos_hit) / 2.
                
                if self.training:
                    alpha_packed = neus_packed_sdf_to_alpha(
                        self.forward_sdf(
                            torch.addcmul(rays_o[ridx_all], rays_d[ridx_all], depths_1_packed.unsqueeze(-1)), 
                            blidx_all)['sdf'], 
                        forward_inv_s, pack_infos_hit)
                else:
                    alpha_packed = neus_packed_sdf_to_alpha(
                        batchify_query(
                            lambda x,b: self.forward_sdf(x,b)['sdf'], 
                            torch.addcmul(rays_o[ridx_all], rays_d[ridx_all], depths_1_packed.unsqueeze(-1)), 
                            blidx_all, chunk=chunksize_query), 
                        forward_inv_s, pack_infos_hit)
                
                nidx_useful, pack_infos_hit_useful, pidx_useful = packed_volume_render_compression(alpha_packed, pack_infos_hit)
                
                if nidx_useful.numel() == 0:
                    return empty_volume_buffer, {}
                else:
                    # Update
                    ridx_all, depths_packed, alpha_packed = ridx_all[pidx_useful], depths_packed[pidx_useful], alpha_packed[pidx_useful]
                    
                    volume_buffer = dict(
                        type='packed', 
                        rays_inds_hit=rays_inds[nidx_useful], pack_infos_hit=pack_infos_hit_useful, 
                        t=depths_packed.to(dtype), opacity_alpha=alpha_packed.to(dtype))
                    
                    if with_rgb or with_normal:
                        h_appear = ray_tested['rays_h_appear'][ridx_all] \
                            if ray_tested.get('rays_h_appear', None) is not None else None
                        net_out = self.forward(
                            torch.addcmul(rays_o[ridx_all], rays_d[ridx_all], depths_packed.unsqueeze(-1)), 
                            view_dirs[ridx_all] if use_view_dirs else None, 
                            None, # NOTE: Not using `blidx_all`. Let the forest to do the normalization because `depths` is calculated using non-reliable diff's delta.
                            h_appear=h_appear,
                            nablas_has_grad=nablas_has_grad, with_rgb=with_rgb, with_normal=with_normal)
                        if "nablas" in net_out: volume_buffer["nablas"] = net_out["nablas"].to(dtype)
                        if "rgb" in net_out: volume_buffer["rgb"] = net_out["rgb"].to(dtype)
                    details = {'march.num_per_ray': pack_infos_marched[:, 1], 'render.num_per_ray0': pack_infos_hit[:, 1], 'render.num_per_ray': pack_infos_hit_useful[:, 1]}
                    return volume_buffer, details

        else: # ridx_hit is None:
            if coarse_step_cfg is None:
                return empty_volume_buffer, {}
            else:
                if self.training:
                    alpha_coarse = neus_packed_sdf_to_alpha(
                        self.forward_sdf(coarse_ret.samples, coarse_ret.blidx)['sdf'], 
                        forward_inv_s, coarse_ret.pack_infos)
                else:
                    alpha_coarse = neus_packed_sdf_to_alpha(
                        batchify_query(
                            lambda x,b: self.forward_sdf(x, b)['sdf'], 
                            coarse_ret.samples, 
                            coarse_ret.blidx, chunk=chunksize_query), 
                        forward_inv_s, coarse_ret.pack_infos)
                depths_coarse = coarse_ret.depth_samples + coarse_ret.deltas/2.
                
                nidx_useful, pack_infos_hit_useful, pidx_useful = packed_volume_render_compression(alpha_coarse, coarse_ret.pack_infos)
                
                if nidx_useful.numel() == 0:
                    return empty_volume_buffer, {}
                else:
                    depths_packed, alpha_packed = depths_coarse[pidx_useful], alpha_coarse[pidx_useful]
                    
                    volume_buffer = dict(
                        type='packed', 
                        rays_inds_hit=rays_inds[coarse_ret.ridx_hit][nidx_useful], pack_infos_hit=pack_infos_hit_useful, 
                        t=depths_packed.to(dtype), opacity_alpha=alpha_packed.to(dtype))
                    
                    if with_rgb or with_normal:
                        ridx_all = coarse_ret.ridx[pidx_useful]
                        blidx_all = coarse_ret.blidx[pidx_useful]
                        h_appear = ray_tested['rays_h_appear'][ridx_all] \
                            if ray_tested.get('rays_h_appear', None) is not None else None
                        net_out = self.forward(
                            torch.addcmul(rays_o[ridx_all], rays_d[ridx_all], depths_packed.unsqueeze(-1)), 
                            view_dirs[ridx_all] if use_view_dirs else None, 
                            # TODO: Experimental verification needed, using the original blidx here still produces many "out-of-bound points", i.e., points that have significantly exceeded the original block range;
                            #       For these points, the forward process will have a forced clip to ensure no runtime bugs, but in computational logic, the coordinates of these points are clamped quite a lot, deviating significantly from the light ray range
                            #       (Because clamping is equivalent to moving the point along the coordinate system's diagonal, not along the direction of the light ray)
                            blidx_all,  
                            h_appear=h_appear, 
                            nablas_has_grad=nablas_has_grad, with_rgb=with_rgb, with_normal=with_normal)
                        if "nablas" in net_out: volume_buffer["nablas"] = net_out["nablas"].to(dtype)
                        if "rgb" in net_out: volume_buffer["rgb"] = net_out["rgb"].to(dtype)
                    details = {'render.num_per_ray0': coarse_ret.pack_infos[:, 1], 'render.num_per_ray': pack_infos_hit_useful[:, 1]}
                    return volume_buffer, details

    def ray_query(
        self, 
        # ray query inputs
        ray_input: Dict[str, torch.Tensor]=None, ray_tested: Dict[str, torch.Tensor]=None, 
        # ray query function config
        config=dict(),
        # function config
        return_buffer=False, return_details=False, render_per_obj_individual=False):
        
        """
        Params:
            ray_input:
                rays_o: [num_total_rays, 3]
                rays_d: [num_total_rays, 3]
                near:   [num_total_rays] or float or None
                far:    [num_total_rays] or float or None
            
            ray_tested:
                num_rays:   int
                rays_o:     [num_rays, 3]
                rays_d:     [num_rays, 3]
                rays_inds:   [num_rays]
                seg_pack_infos:     [num_rays, 2]
                seg_block_inds:     [num_segments]
                seg_entries:  [num_segments]
                seg_exits:    [num_segments]
                
            return_buffer:  If return the queried volume buffer.
            return_details: If return training / debugging related details.
            render_per_obj_individual: If return single object volume rendering results.
        Return:
            Dict
        """
        
        #----------------
        # Inputs
        #----------------
        if ray_tested is None:
            assert ray_input is not None
            ray_tested = self.ray_test(**ray_input, return_rays=True)
        
        #----------------
        # Shortcuts
        #----------------
        # NOTE: The device & dtype of output
        device, dtype = self.device, torch.float
        query_mode, with_rgb, with_normal = config.query_mode, config.with_rgb, config.with_normal
        forward_inv_s = config.get('forward_inv_s', self.forward_inv_s())
        
        #----------------
        # Prepare outputs & compute outputs that are needed even when (num_rays==0)
        #----------------
        raw_ret = dict()
        if return_buffer:
            raw_ret['volume_buffer'] = dict(type='empty', rays_inds_hit=[])
        if return_details:
            details = raw_ret['details'] = {}
            details['inv_s'] = forward_inv_s.item() if isinstance(forward_inv_s, torch.Tensor) else forward_inv_s
            details['s'] = 1./ details['inv_s']
            if (self.accel is not None) and hasattr(self.accel, 'debug_stats'):
                details['accel'] = self.accel.debug_stats()
        if render_per_obj_individual:
            prefix_rays = ray_input['rays_o'].shape[:-1]
            raw_ret['rendered'] = rendered = dict(
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
        if query_mode == 'inblock_coarse_multi_upsample':
            volume_buffer, query_details = self._forest_ray_query_inblock_coarse_multi_upsample(
                ray_tested, with_rgb=with_rgb, with_normal=with_normal, 
                perturb=config.perturb, forward_inv_s=forward_inv_s, 
                **config.query_param)
        elif query_mode == 'inblock_march_occ_multi_upsample':
            volume_buffer, query_details = self._forest_ray_query_inblock_march_occ_multi_upsample(
                ray_tested, with_rgb=with_rgb, with_normal=with_normal, 
                perturb=config.perturb, forward_inv_s=forward_inv_s, 
                **config.query_param)
        elif query_mode == 'inblock_march_occ_multi_upsample_compressed':
            volume_buffer, query_details = self._forest_ray_query_inblock_march_occ_multi_upsample_compressed(
                ray_tested, with_rgb=with_rgb, with_normal=with_normal, 
                perturb=config.perturb, forward_inv_s=forward_inv_s, 
                **config.query_param)
        else:
            raise RuntimeError(f"Invalid query_mode={query_mode}")
        
        #----------------
        # Calc outputs
        #----------------
        if return_buffer:
            raw_ret['volume_buffer'] = volume_buffer
        
        if return_details:
            details.update(query_details)
            if config.get('with_near_sdf', False):
                fwd_kwargs = dict(x=torch.addcmul(ray_tested['rays_o'], ray_tested['rays_d'], ray_tested['near'].unsqueeze(-1)))
                if self.use_ts: fwd_kwargs['ts'] = ray_tested['rays_ts']
                if self.use_fidx: fwd_kwargs['fidx'] = ray_tested['rays_fidx']
                details['near_sdf'] = self.forward_sdf(**fwd_kwargs, input_normalized=False)['sdf']
        
        if render_per_obj_individual:
            if (buffer_type := volume_buffer['type']) != 'empty':
                rays_inds_hit = volume_buffer['rays_inds_hit']
                depth_use_normalized_vw = config.get('depth_use_normalized_vw', True)
                
                if buffer_type == 'batched':
                    volume_buffer['vw'] = vw = ray_alpha_to_vw(volume_buffer['opacity_alpha'])
                    rendered['mask_volume'][rays_inds_hit] = vw_sum = vw.sum(dim=-1)
                    if depth_use_normalized_vw:
                        # TODO: This can also be differed by training / non-training
                        vw_normalized = vw / (vw_sum.unsqueeze(-1)+1e-10)
                        rendered['depth_volume'][rays_inds_hit] = (vw_normalized * volume_buffer['t']).sum(dim=-1)
                    else:
                        rendered['depth_volume'][rays_inds_hit] = (vw * volume_buffer['t']).sum(dim=-1)
                        
                    if with_rgb:
                        rendered['rgb_volume'][rays_inds_hit] = (vw.unsqueeze(-1) * volume_buffer['rgb']).sum(dim=-2)
                    if with_normal:
                        # rendered['normals_volume'][rays_inds_hit] = (vw.unsqueeze(-1) * volume_buffer['nablas']).sum(dim=-2)
                        if self.training:
                            rendered['normals_volume'][rays_inds_hit] = (vw.unsqueeze(-1) * volume_buffer['nablas']).sum(dim=-2)
                        else:
                            rendered['normals_volume'][rays_inds_hit] = (vw.unsqueeze(-1) * F.normalize(volume_buffer['nablas'].clamp_(-1,1), dim=-1)).sum(dim=-2)
                elif buffer_type == 'packed':
                    pack_infos_hit = volume_buffer['pack_infos_hit']
                    # [num_sampels]
                    volume_buffer['vw'] = vw = packed_alpha_to_vw(volume_buffer['opacity_alpha'], pack_infos_hit)
                    # [num_rays_hit]
                    rendered['mask_volume'][rays_inds_hit] = vw_sum = packed_sum(vw.view(-1), pack_infos_hit)
                    # [num_samples]
                    if depth_use_normalized_vw:
                        vw_normalized = packed_div(vw, vw_sum + 1e-10, pack_infos_hit)
                        rendered['depth_volume'][rays_inds_hit] = packed_sum(vw_normalized * volume_buffer['t'].view(-1), pack_infos_hit)
                    else:
                        rendered['depth_volume'][rays_inds_hit] = packed_sum(vw.view(-1) * volume_buffer['t'].view(-1), pack_infos_hit)
                    if with_rgb:
                        rendered['rgb_volume'][rays_inds_hit] = packed_sum(vw.view(-1,1) * volume_buffer['rgb'].view(-1,3), pack_infos_hit)
                    if with_normal:
                        rendered['normals_volume'][rays_inds_hit] = packed_sum(vw.view(-1,1) * volume_buffer['nablas'].view(-1,3), pack_infos_hit)
                        if self.training:
                            rendered['normals_volume'][rays_inds_hit] = packed_sum(vw.view(-1,1) * volume_buffer['nablas'].view(-1,3), pack_infos_hit)
                        else:
                            rendered['normals_volume'][rays_inds_hit] = packed_sum(vw.view(-1,1) * F.normalize(volume_buffer['nablas'].clamp_(-1,1), dim=-1).view(-1,3), pack_infos_hit)
        return raw_ret
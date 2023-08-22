"""
@file   renderer_mixin.py
@author Nianchen Deng, Shanghai AI Lab & Jianfei Guo, Shanghai AI Lab
@brief  Rendering mixin for NeRF++
"""

import math
import numpy as np
from operator import itemgetter
from collections import namedtuple
from typing import Dict, Literal, Optional, Union

import torch
import torch.nn.functional as F
from torch.utils.benchmark import Timer

from nr3d_lib.logger import Logger
from nr3d_lib.config import ConfigDict

from nr3d_lib.models.base import ModelMixin
from nr3d_lib.models.spatial import AABBSpace
from nr3d_lib.models.spatial_accel import get_accel

from nr3d_lib.render.pack_ops import get_pack_infos_from_n, packed_div, packed_sum
from nr3d_lib.render.volume_graphics import packed_alpha_to_vw, ray_alpha_to_vw, tau_to_alpha, packed_volume_render_compression


__all__ = [
    'NeRFDistantRendererMixin'
]

def ray_sphere_intersect(rays_o: torch.Tensor, rays_d: torch.Tensor, r: Union[float, torch.Tensor]) -> torch.Tensor:
    """ Calculate intersections of each rays and each spheres

    Args:
        rays_o (torch.Tensor): [B,3]
        rays_d (torch.Tensor): [B,3]
        r (Union[float, torch.Tensor]): [(B,)P] radius of spheres

    Returns:
        torch.Tensor: [B,P] depths of intersections along rays
    """
    # p, v: Expand to (B, 1, 3)
    p = rays_o.unsqueeze(1)
    v = rays_d.unsqueeze(1)
    # pp, vv, pv: (B, 1)
    pp = (p * p).sum(dim=2)
    vv = (v * v).sum(dim=2)
    pv = (p * v).sum(dim=2)
    z = (((pv * pv - vv * (pp - r * r)).sqrt() - pv) / vv)  # (B, P)
    return z


def ray_box_intersect(rays_o: torch.Tensor, rays_d: torch.Tensor, r: Union[float, torch.Tensor]) -> torch.Tensor:
    """ Calculate intersections of each rays and each boxes

    Args:
        rays_o (torch.Tensor): [B,3]
        rays_d (torch.Tensor): [B,3]
        r (Union[float, torch.Tensor]): [(B,)P] half side-lengths of boxes

    Returns:
        torch.Tensor: [B,P] depths of intersections along rays
    """
    # Expand to (B, 1, 3)
    rays_o = rays_o.unsqueeze(1)
    rays_d = rays_d.unsqueeze(1)

    # Expand to ([B, ]P, 1)
    r = r[..., None]

    # t_min, t_max: (B, P, 3)
    t_min = (-r - rays_o) / rays_d
    t_max = (r - rays_o) / rays_d

    # t_near, t_far: (B, P)
    t_near = torch.minimum(t_min, t_max).max(dim=-1).values
    t_far = torch.maximum(t_min, t_max).min(dim=-1).values

    # Check if rays are inside boxes and boxes are in front of the ray origin
    mask_intersect = (t_far > t_near) & (t_far > 0)
    t_far[~mask_intersect] = math.nan
    return t_far


named_tuple_nerfpp_raymarch_ret = namedtuple("nerfpp_raymarch_ret", "ridx_hit samples depth_samples deltas ridx pack_infos")

class NeRFDistantRendererMixin(ModelMixin):
    def __init__(
        self, 
        # Renderer mixin kwargs
        ray_query_cfg: ConfigDict = ConfigDict(), 
        accel_cfg: ConfigDict = None, 
        radius_scale_min: float = 1.0, 
        radius_scale_max: float = 100.0, 
        include_inf_distance: bool = None, 
        # Network kwargs
        **net_kwargs
        ) -> None:
        
        mro = type(self).mro()
        super_class = mro[mro.index(NeRFDistantRendererMixin)+1]
        assert super_class is not ModelMixin, "Incorrect class inheritance. Three possible misuse scenarios:\n"\
            "Case 1: The Net class for mixin should also inherit from `ModelMixin`.\n"\
            "Case 2: RendererMixin should come before the Net class when inheriting.\n"\
            "Case 3: You should not directly instantiate this mixin class."
        
        super().__init__(**net_kwargs) # Will call network's __init__() (e.g. LoTDNeRF ...)  
        
        self.ray_query_cfg = ray_query_cfg
        self.accel_cfg = accel_cfg
        self.radius_scale_min = radius_scale_min
        self.radius_scale_max = radius_scale_max
        self.include_inf_distance = include_inf_distance

    def populate(self, *args, **kwargs):
        super().populate(*args, **kwargs)
        self.accel = None if self.accel_cfg is None else \
            get_accel(space=self.space, device=self.device, **self.accel_cfg)
        # self.cr_obj = cr_obj

    @property
    def space(self) -> AABBSpace:
        # NOTE: The inner box's AABB
        return super().space

    # def uniform_sample(self, num_samples: int):
    #     x = self.space.uniform_sample_points(num_samples)
    #     # Do not upsate_samples here (usally there are too less samples here.)
    #     ret = self.forward_sigma(x, ignore_update=True)
    #     ret['net_x'] = x  # NOTE: in network's uniformed space; not in world space.
    #     if 'nablas' in ret:
    #         ret['nablas_norm'] = ret['nablas'].norm(dim=-1)
    #     # NOTE: This is changed to be called every time `forward` is called, which is much more often.
    #     # if self.accel is not None:
    #     #     self.accel.gather_samples(ret['net_x'], val=ret['sdf'].data)
    #     return ret

    def forward_sigma(self, x: torch.Tensor, ignore_update=False, **kwargs):
        ret = super().forward_sigma(x, **kwargs)
        if self.accel and not ignore_update:
            self.accel.gather_samples(x, ret['sigma'])
        return ret

    def preprocess_per_train_step(self, cur_it: int, logger: Logger = None):
        super().preprocess_per_train_step(cur_it, logger=logger)
        if self.accel:
            # NOTE: Important to ignore update when query!
            self.accel.preprocess_per_train_step(
                cur_it, lambda x: self.forward_sigma(x, ignore_update=True)['sigma'], logger)

    def postprocess_per_train_step(self, cur_it: int, logger: Logger = None):
        super().postprocess_per_train_step(cur_it, logger=logger)
        if self.accel:
            # NOTE: Important to ignore update when query!
            self.accel.postprocess_per_train_step(
                cur_it, lambda x: self.forward_sigma(x, ignore_update=True)['sigma'], logger)

    def ray_test(self, rays_o: torch.Tensor, rays_d: torch.Tensor, near=None, far=None, **extra_ray_data):
        raise NotImplementedError
        assert (rays_o.dim() == 2) and (rays_d.dim() == 2), \
            "Expect rays_o and rays_d to be of shape [N, 3]"
        return self.space.ray_test(rays_o, rays_d, near=near, far=far, return_rays=True)

    def _ray_marching(
        self,
        rays_o: torch.Tensor, rays_d: torch.Tensor,
        t_min: Optional[torch.Tensor] = None, t_max: Optional[torch.Tensor] = None, 
        perturb: bool = False, max_steps: int = 256,
        sample_mode: str = "box", interval_type: str = "inverse_proportional"
        ):
        if not rays_o.is_cuda:
            raise NotImplementedError("Only support cuda inputs.")
        num_rays = rays_o.shape[0]
        device, dtype = rays_o.device, rays_o.dtype
        
        if interval_type == 'inverse_proportional':
            r_scale_min_reci, r_scale_max_reci = 1. / self.radius_scale_min, 1. / self.radius_scale_max
            step_size = (r_scale_max_reci - r_scale_min_reci) / max_steps
            # [num_rays, max_steps]
            r_reci = torch.arange(r_scale_min_reci, r_scale_max_reci, step_size, device=device).expand(num_rays, -1)
            if perturb:
                r_reci = (r_reci + torch.rand_like(r_reci) * step_size).clamp(1e-5)
            r = r_reci.reciprocal()
        elif interval_type == 'logarithm' or interval_type == 'logarithm_with_inverse_prop_input':
            r_scale_min_log, r_scale_max_log = np.log10(self.radius_scale_min), np.log10(self.radius_scale_max)
            step_size = (r_scale_max_log - r_scale_min_log) / max_steps
            # [num_rays, max_steps]
            r_log = torch.arange(r_scale_min_log, r_scale_max_log, step_size, device=device).expand(num_rays, -1)
            if perturb:
                r_log = r_log + torch.rand_like(r_log) * step_size
            r = 10 ** r_log
            r_reci = r.reciprocal()
        else:
            raise RuntimeError(f"Invalid interval_type={interval_type}")

        # [num_rays, max_steps+1] Extend the last step accordingly
        r_extended = torch.cat([
            r,
            torch.full([num_rays, 1], 1.0e10 if self.include_inf_distance else self.radius_scale_max, device=device, dtype=dtype)
        ], dim=-1)

        if isinstance(self.space, AABBSpace):
            aabb = self.space.aabb
            stretch = aabb[1] - aabb[0]
            # Scaled rays in normalized space (from cuboid aabb to [-1,1]^3 cubic)
            rays_o_normed, rays_d_normed = self.space.normalize_rays(rays_o, rays_d)
        else:
            raise NotImplementedError(f"Not supported inner boundary space type={type(self.space)}")
        
        if sample_mode == 'fixed_cuboid_shells' or sample_mode == 'box':
            t_extended = ray_box_intersect(rays_o_normed, rays_d_normed, r_extended)
            # [num_rays, max_steps]
            deltas = t_extended.diff(dim=-1)
            t = t_extended[:, :-1]
            # [num_rays, max_steps, 3]
            x = torch.addcmul(rays_o_normed.unsqueeze(-2), rays_d_normed.unsqueeze(-2), t.unsqueeze(-1))
            norm_is_radius = True
        elif sample_mode == 'fixed_ellipsoid_shells' or sample_mode == 'ellipsoid':
            t_extended = ray_sphere_intersect(rays_o_normed, rays_d_normed, r_extended)
            deltas = t_extended.diff(dim=-1) # [num_rays, max_steps]
            t = t_extended[:, :-1]
            # [num_rays, max_steps, 3]
            x = torch.addcmul(rays_o_normed.unsqueeze(-2), rays_d_normed.unsqueeze(-2), t.unsqueeze(-1))
            norm_is_radius = True
        elif sample_mode == 'fixed_cubic_shells' or sample_mode == 'cube':
            # NOTE: Use the shortest length as the cube's sidelength
            t_extended = ray_box_intersect(rays_o, rays_d, r_extended * stretch.min())
            # [num_rays, max_steps]
            deltas = t_extended.diff(dim=-1) 
            t = t_extended[:, :-1]
            # [num_rays, max_steps, 3]
            x_in_obj = torch.addcmul(rays_o.unsqueeze(-2), rays_d.unsqueeze(-2), t.unsqueeze(-1))
            x = self.space.normalize_coords(x_in_obj)
            norm_is_radius = False
        elif sample_mode == 'fixed_spherical_shells' or sample_mode == 'spherical':
            # NOTE: Use the shortest length as the sphere's radius
            t_extended = ray_sphere_intersect(rays_o, rays_d, r_extended * stretch.min())
            # [num_rays, max_steps]
            deltas = t_extended.diff(dim=-1) 
            t = t_extended[:, :-1]
            # [num_rays, max_steps, 3]
            x_in_obj = torch.addcmul(rays_o.unsqueeze(-2), rays_d.unsqueeze(-2), t.unsqueeze(-1))
            x = self.space.normalize_coords(x_in_obj)
            norm_is_radius = False
        elif sample_mode == 'moving_spherical_shells' or sample_mode == 'lindisp':
            t_extended = r_extended
            t = t_extended[:, :-1]
            # [num_rays, max_steps, 3]
            x_in_obj = torch.addcmul(rays_o.unsqueeze(-2), rays_d.unsqueeze(-2), t.unsqueeze(-1))
            x = self.space.normalize_coords(x_in_obj)
            norm_is_radius = False
        elif sample_mode == 'moving_far_spherical' or sample_mode == 'neurecon01' or sample_mode == 'neurecon02':
            raise RuntimeError(f"sample_mode={sample_mode} no longer supported.")
        else:
            raise RuntimeError(f"Invalid sample_mode={sample_mode}")

        x_dir = x * r_reci.unsqueeze(-1) if norm_is_radius else F.normalize(x,dim=-1)
        # Length definition
        if interval_type == 'inverse_proportional' or interval_type == 'logarithm_with_inverse_prop_input':
            x_norm_reci = r_reci if norm_is_radius else x.norm(dim=-1).reciprocal()
            r_in_net = x_norm_reci.unsqueeze(-1) * 2. - 1
        elif interval_type == 'logarithm':
            x_norm_log = r_log if norm_is_radius else torch.log10(x.norm(dim=-1))
            # NOTE: Major fallback: can not handle infinite far
            r_in_net = (x_norm_log.unsqueeze(-1) - r_scale_min_log) / (r_scale_max_log - r_scale_min_log) * 2 - 1.
        else:
            raise RuntimeError(f"Invalid interval_type={interval_type}")
        # [num_rays, max_steps, 4], 4D NeRF++ input in grid coordinate [-1,1]
        x = torch.cat([x_dir, r_in_net], dim=-1)

        # Pack valid samples
        valid_samples = ~(torch.isnan(t) | (t < t_min[:, None]))
        ridx, pidx = valid_samples.nonzero().long().t()
        if ridx.numel() == 0:
            return named_tuple_nerfpp_raymarch_ret(None, None, None, None, None, None)
        else:
            num_samples_per_ray = valid_samples.sum(-1)
            pack_infos = get_pack_infos_from_n(num_samples_per_ray)
            ridx_hit = pack_infos[...,1].nonzero().long()[..., 0].contiguous()
            pack_infos = pack_infos[ridx_hit].contiguous().long()
            return named_tuple_nerfpp_raymarch_ret(
                ridx_hit, x[ridx, pidx], t[ridx, pidx], deltas[ridx, pidx], ridx, pack_infos)

    def _ray_query_march_occ(
        self, ray_tested: Dict[str, torch.Tensor], *, 
        with_rgb=True, with_normal=False, perturb=False, march_cfg=ConfigDict(), 
        compression=True, bypass_sigma_fn=None, bypass_alpha_fn=None):
        
        empty_volume_buffer = dict(buffer_type='empty', ray_inds_hit=[])
        if ray_tested['num_rays'] == 0:
            return empty_volume_buffer

        # NOTE: Normalized rays in network's space
        rays_o, rays_d, near, far, ray_inds = itemgetter("rays_o", "rays_d", "near", "far", "ray_inds")(ray_tested)
        assert (rays_o.dim() == 2) and (rays_d.dim() == 2)

        # NOTE: The device & dtype of output
        device, dtype = rays_o.device, rays_o.dtype
        
        #----------------
        # Ray marching
        #----------------
        raymarch_ret = self._ray_marching(rays_o, rays_d, near, far, perturb=perturb, **march_cfg)
        if raymarch_ret.ridx_hit is None:
            return empty_volume_buffer

        if compression:
            #----------------
            # Calc compact samples using rendered visibility
            with torch.no_grad():
                assert int(bypass_sigma_fn is not None) + int(bypass_alpha_fn is not None) <= 1, "Please pass at most one of bypass_sigma_fn or bypass_alpha_fn"
                if bypass_sigma_fn is not None:
                    sigmas = bypass_sigma_fn(raymarch_ret)
                    alphas = tau_to_alpha(sigmas * raymarch_ret.deltas)
                elif bypass_alpha_fn is not None:
                    alphas = bypass_alpha_fn(raymarch_ret)
                else:
                    sigmas = self.forward_sigma(raymarch_ret.samples)['sigma']
                    alphas = tau_to_alpha(sigmas * raymarch_ret.deltas)
            old_pack_infos = raymarch_ret.pack_infos.clone()
            nidx_useful, pack_infos_hit_useful, pidx_useful = packed_volume_render_compression(alphas, raymarch_ret.pack_infos)
            
            if nidx_useful.numel() == 0:
                return empty_volume_buffer
            
            # Update compact raymarch_ret
            raymarch_ret = named_tuple_nerfpp_raymarch_ret(
                raymarch_ret.ridx_hit[nidx_useful], 
                raymarch_ret.samples[pidx_useful], raymarch_ret.depth_samples[pidx_useful], raymarch_ret.deltas[pidx_useful], 
                raymarch_ret.ridx[pidx_useful], pack_infos_hit_useful)
        else:
            pass

        #----------------
        # Gather volume_buffer
        #----------------
        volume_buffer = dict(
            buffer_type='packed', 
            ray_inds_hit=ray_inds[raymarch_ret.ridx_hit], 
            pack_infos_hit=raymarch_ret.pack_infos, 
            t=raymarch_ret.depth_samples.to(dtype))
        if compression:
            volume_buffer['details'] = {
                'march.num_per_ray': old_pack_infos[:,1], 
                'render.num_per_ray0': old_pack_infos[:,1], 
                'render.num_per_ray': raymarch_ret.pack_infos[:,1]
            }
        else:
            volume_buffer['details'] = {
                'march.num_per_ray': raymarch_ret.pack_infos[:,1], 
                'render.num_per_ray': raymarch_ret.pack_infos[:,1]
            }
        
        if with_rgb:
            # NOTE: The spatial length scale on each ray caused by scaling rays_d 
            dir_scale = rays_d.norm(dim=-1)  # [num_rays]
            # NOTE: The normalized ray direction vector in network's space
            view_dirs = rays_d / dir_scale.clamp_min_(1.0e-10).unsqueeze(-1) # [num_rays, 3]
            # Get embedding code
            h_appear_embed = ray_tested["rays_h_appear_embed"][raymarch_ret.ridx]\
                if ray_tested.get('rays_h_appear_embed', None) is not None else None
            
            net_out = self.forward(x=raymarch_ret.samples, v=view_dirs[raymarch_ret.ridx], h_appear_embed=h_appear_embed)
            volume_buffer["rgb"] = net_out["radiances"].to(dtype)
        else:
            net_out = self.forward_sigma(raymarch_ret.samples)
        volume_buffer["sigma"] = net_out["sigma"].to(dtype)
        volume_buffer["opacity_alpha"] = tau_to_alpha(net_out["sigma"] * raymarch_ret.deltas).to(dtype)
        return volume_buffer

    # @profile
    def ray_query(
        self,
        # ray query inputs
        ray_input: Dict[str, torch.Tensor] = None,
        ray_tested: Dict[str, torch.Tensor] = None, 
        # ray query function config
        config=ConfigDict(), 
        # function config
        return_buffer=False, return_details=False, render_per_obj=False):
        """ Query the model with input rays. 
            Conduct the core ray sampling, ray marching, ray upsampling and network query operations.

        Args:
            ray_input (Dict[str, torch.Tensor], optional): All input rays.
                See more details in `ray_test`
            ray_tested (Dict[str, torch.Tensor], optional): Tested rays (Typicallly those that intersect with objects). A dict composed of:
                num_rays:   int, Number of tested rays
                ray_inds:   [num_rays] tensor, ray indices in `num_total_rays` of the tested rays
                rays_o:     [num_rays, 3] tensor, the indexed and scaled rays' origin
                rays_d:     [num_rays, 3] tensor, the indexed and scaled rays' direction
                near:       [num_rays] tensor, entry depth of intersection
                far:        [num_rays] tensor, exit depth of intersection
            config (ConfigDict, optional): Config of ray_query. Defaults to ConfigDict().
            return_buffer (bool, optional): If return the queried volume buffer. Defaults to False.
            return_details (bool, optional): If return training / debugging related details. Defaults to False.
            render_per_obj (bool, optional): If return single object / seperate volume rendering results. Defaults to False.

        Returns:
            nested dict: The queried results, including 'volume_buffer', 'details', 'rendered'.
            
            ['volume_buffer']: dict, The queried volume buffer. Available if `return_buffer` is set True.
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
                        'rgb':              [num_rays_hit, num_samples_per_ray, 3] packed tensor, optional, the queried rgb values (Only if `with_rgb` is True)
                        'nablas':           [num_rays_hit, num_samples_per_ray, 3] packed tensor, optional, the queried nablas values (Only if `with_normal` is True)
                        'feature':          [num_rays_hit, num_samples_per_ray, with_feature_dim] batched tensor, optional, the queried features (Only if `with_feature_dim` > 0)
                    }
                
                An example `packed` buffer:
                    'volume_buffer': {
                        'type': 'packed',
                        'ray_inds_hit':     [num_rays_hit] tensor, ray indices in `num_total_rays` of the hit & queried rays
                        'pack_infos_hit'    [num_rays_hit, 2] tensor, pack infos of the queried packed tensors
                        't':                [num_packed_samples] packed tensor, real depth of the queried samples
                        'opacity_alpha':    [num_packed_samples] packed tensor, the queried alpha-values
                        'rgb':              [num_packed_samples, 3] packed tensor, optional, the queried rgb values (Only if `with_rgb` is True)
                        'nablas':           [num_packed_samples, 3] packed tensor, optional, the queried nablas values (Only if `with_normal` is True)
                        'feature':          [num_packed_samples, with_feature_dim] packed tensor, optional, the queried features (Only if `with_feature_dim` > 0)
                    }

            ['details']: nested dict, Details for training. Available if `return_details` is set True.
            
            ['rendered']: dict, stand-alone rendered results. Available if `render_per_obj` is set True.
                An example rendered dict:
                    'rendered' {
                        'mask_volume':      [num_total_rays,] The rendered opacity / occupancy, in range [0,1]
                        'depth_volume':     [num_total_rays,] The rendered real depth
                        'rgb_volume':       [num_total_rays, 3] The rendered rgb, in range [0,1] (Only if `with_rgb` is True)
                        'normals_volume':   [num_total_rays, 3] The rendered normals, in range [-1,1] (Only if `with_normal` is True)
                        'feature_volume':   [num_total_rays, with_feature_dim] The rendered feature. (Only if `with_feature_dim` > 0)
                    }
        """

        # ----------------
        # Inputs
        # ----------------
        if ray_tested is None:
            assert ray_input is not None
            ray_tested = self.ray_test(**ray_input)

        # ----------------
        # Shortcuts
        # ----------------
        device, dtype = self.device, self.dtype
        query_mode, with_rgb, with_normal = config.query_mode, config.with_rgb, config.with_normal

        # ----------------
        # Prepare outputs
        # ----------------
        raw_ret = {}
        if return_buffer:
            raw_ret['volume_buffer'] = dict(buffer_type='empty', ray_inds_hit=[])
        if return_details:
            raw_ret['details'] = details = {}
            if self.accel:
                details['accel'] = self.accel.debug_stats()
        if render_per_obj:
            raw_ret['rendered'] = rendered = {
                "depth_volume": torch.zeros_like(ray_input["rays_o"][..., 0]),
                "mask_volume": torch.zeros_like(ray_input["rays_o"][..., 0])
            }
            if with_rgb:
                rendered['rgb_volume'] = torch.zeros_like(ray_input["rays_o"])
            if with_normal:
                rendered['normals_volume'] = torch.zeros_like(ray_input["rays_o"])

        if ray_tested['num_rays'] == 0:
            return raw_ret
            
        #----------------
        # Ray query
        #----------------
        if query_mode == 'march_occ':
            volume_buffer = self._ray_query_march_occ(
                ray_tested, with_rgb=with_rgb, perturb=config.perturb, **config.query_param)
        else:
            raise RuntimeError(f"Invalid query_mode={query_mode}")

        if return_buffer:
            raw_ret['volume_buffer'] = volume_buffer

        if return_details:
            pass
        
        if render_per_obj:
            if (buffer_type:=volume_buffer['buffer_type']) != 'empty':
                ray_inds_hit = volume_buffer['ray_inds_hit']
                depth_use_normalized_vw = config.get('depth_use_normalized_vw', True)

                if buffer_type == 'batched':
                    volume_buffer['vw'] = vw = ray_alpha_to_vw(volume_buffer['opacity_alpha'])
                    vw_sum = rendered['mask_volume'][ray_inds_hit] = vw.sum(dim=-1)
                    if depth_use_normalized_vw:
                        vw_normalized = vw / (vw_sum.unsqueeze(-1)+1e-10)
                        rendered['depth_volume'][ray_inds_hit] = (vw_normalized * volume_buffer['t']).sum(dim=-1)
                    else:
                        rendered['depth_volume'][ray_inds_hit] = (vw * volume_buffer['t']).sum(dim=-1)
                    if with_rgb:
                        rendered['rgb_volume'][ray_inds_hit] = (vw.unsqueeze(-1) * volume_buffer['rgb']).sum(dim=-2)
                    if with_normal:
                        pass
                
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

        return raw_ret

if __name__ == "__main__":
    def test_r_inveser_sampling(device=torch.device('cuda:0')):
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        r_scale_min = 1.0
        r_scale_max = 1000.0
        max_steps = 16
        num_rand_tests = 10000
        
        plt.subplot(1, 3, 1)
        r_scale_min_reci = 1. / r_scale_min
        r_scale_max_reci = 1. / r_scale_max
        step_size = (r_scale_max_reci - r_scale_min_reci) / max_steps
        # [max_steps]; Minus value
        r_scale_reci = torch.arange(r_scale_min_reci, r_scale_max_reci, step_size, device=device)
        # [num_rand_tests, max_steps]
        r_scale_reci_perturbed = (r_scale_reci + torch.rand([num_rand_tests, max_steps], device=device) * step_size).clamp(1e-5)
        r_scale = r_scale_reci.reciprocal()
        r_scale_perturbed = r_scale_reci_perturbed.reciprocal()
        r_scale_perturbed_log = torch.log10(r_scale_perturbed)
        
        plt.plot(np.arange(max_steps), np.log10(r_scale.cpu().numpy()), 'r*--', label=f'r_scale @ N={max_steps}')
        plt.errorbar(np.arange(max_steps), r_scale_perturbed_log.mean(dim=0).data.cpu().numpy(), 
                     yerr= torch.stack([
                         (r_scale_perturbed_log.mean(dim=0)-r_scale_perturbed_log.min(dim=0).values), 
                         (r_scale_perturbed_log.max(dim=0).values-r_scale_perturbed_log.mean(dim=0))
                     ], dim=0).data.cpu().numpy(), 
                     elinewidth=2, capsize=2, capthick=1, barsabove=True, 
                     label=f'r_scale_perturbed @ N={max_steps}'
                     )
        plt.ylabel("log10")
        plt.title("perturb with inverse linear step_size (need clamp)")
        plt.legend()
        
        plt.subplot(1, 3, 2)
        z = torch.linspace(1, 0, max_steps+2, device=device)[..., 0:-1]
        _t = r_scale_min / z
        r_scale, deltas = _t[..., 1:], _t.diff(dim=-1)
        mids = 0.5 * (r_scale[..., 1:] + r_scale[..., :-1])
        upper = torch.cat([mids, r_scale[..., -1:]], -1)
        lower = torch.cat([r_scale[..., :1], mids], -1)
        r_scale_perturbed = lower + (upper-lower) * torch.rand([num_rand_tests, max_steps], device=device)
        r_scale_perturbed_log = torch.log10(r_scale_perturbed)
        
        plt.plot(np.log10(r_scale.cpu().numpy()), 'r*--',label=f'r_scale @ N={max_steps}')
        plt.errorbar(np.arange(max_steps), r_scale_perturbed_log.mean(dim=0).data.cpu().numpy(), 
                     yerr= torch.stack([
                         (r_scale_perturbed_log.mean(dim=0)-r_scale_perturbed_log.min(dim=0).values), 
                         (r_scale_perturbed_log.max(dim=0).values-r_scale_perturbed_log.mean(dim=0))
                     ], dim=0).data.cpu().numpy(), 
                     elinewidth=2, capsize=2, capthick=1, barsabove=True, 
                     label=f'r_scale_perturbed @ N={max_steps}'
                     )
        plt.ylabel("log10")
        plt.title("perturb in between")
        plt.legend()
        
        plt.subplot(1, 3, 3)
        z = torch.linspace(1, 0, max_steps+2, device=device)[..., :-1]
        z_perturbed = (z - torch.rand([num_rand_tests, max_steps+1], device=device) / (max_steps+1) ).clamp_min_(1e-10)
        # [max_steps+1]
        r_scale_ex = r_scale_min / z
        r_scale, deltas = r_scale_ex[..., :-1], r_scale_ex.diff(dim=-1)
        # [num_rand_tests, max_steps+1]
        r_scale_perturbed_ex = r_scale_min / z_perturbed
        r_scale_perturbed, deltas_perturbed = r_scale_perturbed_ex[..., :-1], r_scale_perturbed_ex.diff(dim=-1)
        r_scale_perturbed_log = torch.log10(r_scale_perturbed)
        plt.plot(np.log10(r_scale.cpu().numpy()), 'r*--', label=f'r_scale @ N={max_steps}')
        plt.errorbar(np.arange(max_steps), r_scale_perturbed_log.mean(dim=0).data.cpu().numpy(), 
                     yerr= torch.stack([
                         (r_scale_perturbed_log.mean(dim=0)-r_scale_perturbed_log.min(dim=0).values), 
                         (r_scale_perturbed_log.max(dim=0).values-r_scale_perturbed_log.mean(dim=0))
                     ], dim=0).data.cpu().numpy(), 
                     elinewidth=2, capsize=2, capthick=1, barsabove=True, 
                     label=f'r_scale_perturbed @ N={max_steps}'
                     )
        plt.ylabel("log10")
        plt.title("perturb with extra points")
        plt.legend()
        
        plt.show()
    
    def test_r_log_sampling(device=torch.device('cuda:0')):
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        r_scale_min = 1.0
        r_scale_max = 1000.0
        max_steps = 64
        num_rand_tests = 10000
        
        # NOTE: log-scale linear
        log_r_scale_min = np.log10(r_scale_min)
        log_r_scale_max = np.log10(r_scale_max)
        step_size = (log_r_scale_max - log_r_scale_min) / max_steps
        # [max_steps]; Minus value
        log_r_scale = torch.arange(log_r_scale_min, log_r_scale_max, step_size, device=device)
        # [num_rand_tests, max_steps]
        log_r_scale_perturbed = log_r_scale + torch.rand([num_rand_tests, max_steps], device=device) * step_size
        r_scale = 10 ** log_r_scale
        r_scale_perturbed = 10 ** log_r_scale_perturbed
        r_scale_perturbed_log = torch.log10(r_scale_perturbed)
        
        plt.subplot(1, 2, 1)
        plt.plot(np.arange(max_steps), np.log10(r_scale.cpu().numpy()), 'r*--', label=f'r_scale @ N={max_steps}')
        plt.errorbar(np.arange(max_steps), r_scale_perturbed_log.mean(dim=0).data.cpu().numpy(), 
                     yerr= torch.stack([
                         (r_scale_perturbed_log.mean(dim=0)-r_scale_perturbed_log.min(dim=0).values), 
                         (r_scale_perturbed_log.max(dim=0).values-r_scale_perturbed_log.mean(dim=0))
                     ], dim=0).data.cpu().numpy(), 
                     elinewidth=2, capsize=2, capthick=1, barsabove=True, 
                     label=f'r_scale_perturbed @ N={max_steps}'
                     )
        plt.ylabel("log10")
        plt.title("perturb with log linear step_size [log10]")
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(np.arange(max_steps), r_scale.cpu().numpy(), 'r*--', label=f'r_scale @ N={max_steps}')
        plt.errorbar(np.arange(max_steps), r_scale_perturbed.mean(dim=0).data.cpu().numpy(), 
                     yerr= torch.stack([
                         (r_scale_perturbed.mean(dim=0)-r_scale_perturbed.min(dim=0).values), 
                         (r_scale_perturbed.max(dim=0).values-r_scale_perturbed.mean(dim=0))
                     ], dim=0).data.cpu().numpy(), 
                     elinewidth=2, capsize=2, capthick=1, barsabove=True, 
                     label=f'r_scale_perturbed @ N={max_steps}'
                     )
        plt.ylabel("linear")
        plt.title("perturb with log linear step_size [linear]")
        plt.legend()
        plt.show()
    
    # def test_r_inverse_sampling_with_k(device=torch.device('cuda:0')):
    #     import numpy as np
    #     import matplotlib.pyplot as plt
    #     import seaborn as sns
        
    #     r_scale_min = 1.0
    #     r_scale_max = 1000.0
    #     max_steps = 32
    #     num_rand_tests = 10000
    #     k = 10. # No affect on the final sampled r.
        
    #     r_scale_min_reci = k / r_scale_min
    #     r_scale_max_reci = k / r_scale_max
    #     step_size = (r_scale_max_reci - r_scale_min_reci) / max_steps
    #     # [max_steps]; Minus value
    #     r_scale_reci = torch.arange(r_scale_min_reci, r_scale_max_reci, step_size, device=device)
    #     # [num_rand_tests, max_steps]
    #     r_scale_reci_perturbed = (r_scale_reci + torch.rand([num_rand_tests, max_steps], device=device) * step_size).clamp(1e-5)
    #     r_scale = k / r_scale_reci
    #     r_scale_perturbed = k / r_scale_reci_perturbed
    #     r_scale_perturbed_log = torch.log10(r_scale_perturbed)
        
    #     plt.plot(np.arange(max_steps), np.log10(r_scale.cpu().numpy()), 'r*--', label=f'r_scale @ k={k}')
    #     plt.errorbar(np.arange(max_steps), r_scale_perturbed_log.mean(dim=0).data.cpu().numpy(), 
    #                  yerr= torch.stack([
    #                      (r_scale_perturbed_log.mean(dim=0)-r_scale_perturbed_log.min(dim=0).values), 
    #                      (r_scale_perturbed_log.max(dim=0).values-r_scale_perturbed_log.mean(dim=0))
    #                  ], dim=0).data.cpu().numpy(), 
    #                  elinewidth=2, capsize=2, capthick=1, barsabove=True, 
    #                  label=f'r_scale_perturbed @ k={k}'
    #                  )
    #     plt.ylabel("log10")
    #     plt.title("perturb with inverse linear step_size (need clamp)")
    #     plt.legend()
    #     plt.show()

    test_r_inveser_sampling()
    test_r_log_sampling()
    # test_r_inverse_sampling_with_k()
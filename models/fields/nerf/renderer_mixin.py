"""
@file   renderer_mixin.py
@author Jianfei Guo, Shanghai AI Lab & Nianchen Deng, Shanghai AI Lab
@brief  Rendering mixin for NeRF with acceleration.
"""

__all__ = [
    'NeRFRendererMixin'
]

from typing import Dict, List
from operator import itemgetter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.benchmark import Timer

from nr3d_lib.logger import Logger
from nr3d_lib.config import ConfigDict

from nr3d_lib.models.base import ModelMixin
from nr3d_lib.models.spatial import AABBSpace
from nr3d_lib.models.spatial_accel import get_accel

from nr3d_lib.render.pack_ops import packed_div, packed_sum
from nr3d_lib.render.volume_graphics import packed_alpha_to_vw, packed_volume_render_compression, ray_alpha_to_vw, tau_to_alpha


class NeRFRendererMixin(ModelMixin):
    def __init__(
        self, 
        # Renderer mixin kwargs
        ray_query_cfg: ConfigDict = ConfigDict(), 
        accel_cfg: ConfigDict = None, 
        shrink_milestones: List[int] = [], 
        # Network kwargs
        **net_kwargs
        ) -> None:
        
        mro = type(self).mro()
        super_class = mro[mro.index(NeRFRendererMixin)+1]
        assert super_class is not ModelMixin, "Incorrect class inheritance. Three possible misuse scenarios:\n"\
            "Case 1: The Net class for mixin should also inherit from `ModelMixin`.\n"\
            "Case 2: RendererMixin should come before the Net class when inheriting.\n"\
            "Case 3: You should not directly instantiate this mixin class."
        
        super().__init__(**net_kwargs) # Will call network's __init__() (e.g. LoTDNeRF ...)  
        
        self.ray_query_cfg = ray_query_cfg
        self.accel_cfg = accel_cfg
        self.shrink_milestones = shrink_milestones
    
    def populate(self, *args, **kwargs):
        super().populate(*args, **kwargs)
        self.accel = None if self.accel_cfg is None else \
            get_accel(space=self.space, device=self.device, **self.accel_cfg)
        #self.accel.occ.occ_grid.fill_(True)
        #self.accel.occ.occ_val_grid.fill_(1000.)

    @property
    def space(self) -> AABBSpace:
        return super().space

    def uniform_sample(self, num_samples: int):
        x = self.space.uniform_sample_points(num_samples)
        # Do not upsate_samples here (usally there are too less samples here.)
        ret = self.forward_sigma(x, ignore_update=True)
        ret['net_x'] = x  # NOTE: in network's uniformed space; not in world space.
        if 'nablas' in ret:
            ret['nablas_norm'] = ret['nablas'].norm(dim=-1)
        # NOTE: This is changed to be called every time `forward` is called, which is much more often.
        # if self.accel is not None:
        #     self.accel.gather_samples(ret['net_x'], val=ret['sdf'].data)
        return ret

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

            # ------------ Shrink according to actual occupied space.
            if cur_it in self.shrink_milestones:
                self.shrink()

    @torch.no_grad()
    def shrink(self):
        new_aabb = self.accel.try_shrink()
        # Rescale network
        super().rescale_volume(new_aabb)
        # Rescale acceleration struct
        self.accel.rescale_volume(new_aabb)
        # Rescale space
        # NOTE: Always rescale space at the last step, since the old space is required by prev steps
        self.space.rescale_volume(new_aabb)

    def ray_test(self, rays_o: torch.Tensor, rays_d: torch.Tensor, near=None, far=None, **extra_ray_data) -> dict:
        """ Test input rays' intersection with the model's space (AABB, Blocks, etc.)

        Args:
            rays_o (torch.Tensor): [num_total_rays,3] Input rays' origin
            rays_d (torch.Tensor): [num_total_rays,3] Input rays' direction
            near (Union[float, torch.Tensor], optional): [num_total_rays] tensor or float. Defaults to None.
            far (Union[float, torch.Tensor], optional): [num_total_rays] tensor or float. Defaults to None.
        Returns:
            dict: The ray_tested result. An example dict:
                num_rays:   int, Number of tested rays
                ray_inds:   [num_rays] tensor, ray indices in `num_total_rays` of the tested rays
                near:       [num_rays] tensor, entry depth of intersection
                far:        [num_rays] tensor, exit depth of intersection
                rays_o:     [num_rays, 3] tensor, the indexed and scaled rays' origin
                rays_d:     [num_rays, 3] tensor, the indexed and scaled rays' direction
        """
        assert (rays_o.dim() == 2) and (rays_d.dim() == 2), "Expect `rays_o` and `rays_d` to be of shape [N, 3]"
        for k, v in extra_ray_data.items():
            if isinstance(v, torch.Tensor):
                assert v.shape[0] == rays_o.shape[0], f"Expect `{k}` has the same shape prefix with rays_o {rays_o.shape[0]}, but got {v.shape[0]}"
        return self.space.ray_test(rays_o, rays_d, near=near, far=far, return_rays=True, **extra_ray_data)

    def _ray_query_march_occ(
        self, ray_tested: Dict[str, torch.Tensor], *, 
        with_rgb: bool = True, with_normal: bool = False, 
        perturb: bool = False, march_cfg=ConfigDict(), 
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
        raymarch_ret = self.accel.ray_march(rays_o, rays_d, near, far, perturb=perturb, **march_cfg)
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
            raymarch_ret = type(raymarch_ret)(
                raymarch_ret.ridx_hit[nidx_useful], 
                raymarch_ret.samples[pidx_useful], raymarch_ret.depth_samples[pidx_useful], raymarch_ret.deltas[pidx_useful], 
                raymarch_ret.ridx[pidx_useful], pack_infos_hit_useful, None, None)
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
            ray_input (Dict[str, torch.Tensor], optional): All input rays. A dict composed of:
                rays_o: [num_total_rays, 3]
                rays_d: [num_total_rays, 3]
                near:   [num_total_rays] tensor or float or None
                far:    [num_total_rays] tensor or float or None
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

"""
@file   renderer_mixin.py
@author Jianfei Guo, Shanghai AI Lab & Nianchen Deng, Shanghai AI Lab
@brief  Rendering mixin for NeRF with acceleration.
"""

__all__ = [
    'NeRFRendererMixin'
]

from typing import Dict, List, Tuple

import torch

from nr3d_lib.logger import Logger

from nr3d_lib.models.model_base import ModelMixin
from nr3d_lib.models.spatial import AABBSpace
from nr3d_lib.models.accelerations import get_accel, accel_types_single_t

from nr3d_lib.graphics.pack_ops import packed_div, packed_sum
from nr3d_lib.graphics.nerf import *


class NeRFRendererMixin(ModelMixin):

    # NOTE: Configuration for common information usage in the forward process (and their defaults)
    use_view_dirs: bool = False # Determines if view_dirs are needed in forward process
    use_nablas: bool = False # Determines if nablas are required in forward process
    use_h_appear: bool = False # Determines if global per-frame appearance embeddings are necessary in forward process
    use_ts: bool = False # Determines if global timestamps are used in forward process
    use_pix: bool = False # Determines if pixel locations (in range [0,1]^2) are required in forward process
    use_fidx: bool = False # Determines if global frame indices are used in forward process
    use_bidx: bool = False # Determines if batch indices are used in forward process
    fwd_density_use_pix: bool = False
    fwd_density_use_h_appear: bool = False
    fwd_density_use_view_dirs: bool = False
    
    def __init__(
        self, 
        # Renderer mixin kwargs
        ray_query_cfg: dict = dict(), 
        accel_cfg: dict = None, 
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
        self.accel: accel_types_single_t = None if self.accel_cfg is None \
            else get_accel(space=self.space, device=self.device, **self.accel_cfg)
        #self.accel.occ.occ_grid.fill_(True)
        #self.accel.occ.occ_val_grid.fill_(1000.)

    # @property
    # def space(self) -> AABBSpace:
    #     return super().space

    def sample_pts_uniform(self, num_samples: int):
        # NOTE: Returns normalized `x`
        x = self.space.sample_pts_uniform(num_samples)
        # Do not upsate_samples here (usally there are too less samples here.)
        ret = self.forward_density(x, skip_accel=True)
        ret['net_x'] = x  # NOTE: in network's uniformed space; not in world space.
        if 'nablas' in ret:
            ret['nablas_norm'] = ret['nablas'].norm(dim=-1)
        return ret

    def forward_density(self, x: torch.Tensor, skip_accel=False, **kwargs):
        ret = super().forward_density(x, **kwargs)
        if (not skip_accel) and (self.accel is not None):
            self.accel.collect_samples(x, ret['sigma'])
        return ret
    
    @torch.no_grad()
    def query_density(self, x: torch.Tensor, **kwargs):
        return super().query_density(x, **kwargs)

    def training_initialize(self, config=dict(), logger=None, log_prefix: str=None, skip_accel=False) -> bool:
        updated = super().training_initialize(config, logger=logger, log_prefix=log_prefix)
        if (not skip_accel) and (self.accel is not None):
            self.accel.init(self.query_density, logger=logger)
        return updated

    def training_before_per_step(self, cur_it: int, logger: Logger = None, skip_accel=False):
        super().training_before_per_step(cur_it, logger=logger)
        if self.training and (not skip_accel) and (self.accel is not None):
            self.accel.step(cur_it, self.query_density, logger)
            # if cur_it == 0:
            #     self.accel.init(self.query_density, logger)
            # else:
            #     self.accel.step(cur_it, self.query_density, logger)

    def training_after_per_step(self, cur_it: int, logger: Logger = None, skip_accel=False):
        super().training_after_per_step(cur_it, logger=logger)
        if (not skip_accel) and (self.accel is not None):
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
                rays_inds:   [num_rays] tensor, ray indices in `num_total_rays` of the tested rays
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

    # @profile
    def ray_query(
        self,
        # ray query inputs
        ray_input: Dict[str, torch.Tensor] = None,
        ray_tested: Dict[str, torch.Tensor] = None,
        # ray query function config
        config=dict(),
        # function config
        return_buffer=False, return_details=False, render_per_obj_individual=False):
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
                rays_inds:   [num_rays] tensor, ray indices in `num_total_rays` of the tested rays
                rays_o:     [num_rays, 3] tensor, the indexed and scaled rays' origin
                rays_d:     [num_rays, 3] tensor, the indexed and scaled rays' direction
                near:       [num_rays] tensor, entry depth of intersection
                far:        [num_rays] tensor, exit depth of intersection
            config (dict, optional): Config of ray_query. Defaults to dict().
            return_buffer (bool, optional): If return the queried volume buffer. Defaults to False.
            return_details (bool, optional): If return training / debugging related details. Defaults to False.
            render_per_obj_individual (bool, optional): If return single object / seperate volume rendering results. Defaults to False.

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
                        'rays_inds_hit':     [num_rays_hit] tensor, ray indices in `num_total_rays` of the hit & queried rays
                        't':                [num_rays_hit, num_samples_per_ray] batched tensor, real depth of the queried samples
                        'opacity_alpha':    [num_rays_hit, num_samples_per_ray] batched tensor, the queried alpha-values
                        'rgb':              [num_rays_hit, num_samples_per_ray, 3] packed tensor, optional, the queried rgb values (Only if `with_rgb` is True)
                        'nablas':           [num_rays_hit, num_samples_per_ray, 3] packed tensor, optional, the queried nablas values (Only if `with_normal` is True)
                        'feature':          [num_rays_hit, num_samples_per_ray, with_feature_dim] batched tensor, optional, the queried features (Only if `with_feature_dim` > 0)
                    }
                
                An example `packed` buffer:
                    'volume_buffer': {
                        'type': 'packed',
                        'rays_inds_hit':     [num_rays_hit] tensor, ray indices in `num_total_rays` of the hit & queried rays
                        'pack_infos_hit'    [num_rays_hit, 2] tensor, pack infos of the queried packed tensors
                        't':                [num_packed_samples] packed tensor, real depth of the queried samples
                        'opacity_alpha':    [num_packed_samples] packed tensor, the queried alpha-values
                        'rgb':              [num_packed_samples, 3] packed tensor, optional, the queried rgb values (Only if `with_rgb` is True)
                        'nablas':           [num_packed_samples, 3] packed tensor, optional, the queried nablas values (Only if `with_normal` is True)
                        'feature':          [num_packed_samples, with_feature_dim] packed tensor, optional, the queried features (Only if `with_feature_dim` > 0)
                    }

            ['details']: nested dict, Details for training. Available if `return_details` is set True.
            
            ['rendered']: dict, stand-alone rendered results. Available if `render_per_obj_individual` is set True.
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
        # Prepare outputs & compute outputs that are needed even when (num_rays==0)
        # ----------------
        raw_ret = {}
        if return_buffer:
            raw_ret['volume_buffer'] = dict(type='empty', rays_inds_hit=[])
        if return_details:
            raw_ret['details'] = details = {}
            if (self.accel is not None) and hasattr(self.accel, 'debug_stats'):
                details['accel'] = self.accel.debug_stats()
        if render_per_obj_individual:
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
            volume_buffer, query_details = nerf_ray_query_march_occ(
                self, ray_tested, with_rgb=with_rgb, perturb=config.perturb, **config.query_param)
        elif query_mode == 'march_occ_multi_upsample':
            volume_buffer, query_details = nerf_ray_query_march_occ_multi_upsample_compressed(
                self, ray_tested, with_rgb=with_rgb, perturb=config.perturb, **config.query_param)
        else:
            raise RuntimeError(f"Invalid query_mode={query_mode}")

        if return_buffer:
            raw_ret['volume_buffer'] = volume_buffer

        if return_details:
            details.update(query_details)
        
        if render_per_obj_individual:
            if (buffer_type:=volume_buffer['type']) != 'empty':
                rays_inds_hit = volume_buffer['rays_inds_hit']
                depth_use_normalized_vw = config.get('depth_use_normalized_vw', True)

                if buffer_type == 'batched':
                    volume_buffer['vw'] = vw = ray_alpha_to_vw(volume_buffer['opacity_alpha'])
                    rendered['mask_volume'][rays_inds_hit] = vw_sum = vw.sum(dim=-1)
                    if depth_use_normalized_vw:
                        vw_normalized = vw / (vw_sum.unsqueeze(-1)+1e-10)
                        rendered['depth_volume'][rays_inds_hit] = (vw_normalized * volume_buffer['t']).sum(dim=-1)
                    else:
                        rendered['depth_volume'][rays_inds_hit] = (vw * volume_buffer['t']).sum(dim=-1)
                    if with_rgb:
                        rendered['rgb_volume'][rays_inds_hit] = (vw.unsqueeze(-1) * volume_buffer['rgb']).sum(dim=-2)
                    if with_normal:
                        pass
                
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

        return raw_ret

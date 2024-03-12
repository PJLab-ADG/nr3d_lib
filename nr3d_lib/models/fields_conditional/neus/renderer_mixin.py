"""
@file   renderer_mixin.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Renderer mixin for batched NeuS with acceleration.
"""

__all__ = [
    'NeuSRendererMixinBatched', 
]

from logging import Logger
from typing import Dict, List, Literal, Tuple, Union

import torch
import torch.nn.functional as F

from nr3d_lib.models.model_base import ModelMixin
from nr3d_lib.models.spatial import BatchedBlockSpace
from nr3d_lib.models.accelerations import get_accel, accel_types_batched_t

from nr3d_lib.graphics.pack_ops import packed_div, packed_sum
from nr3d_lib.graphics.nerf import packed_alpha_to_vw, ray_alpha_to_vw
from nr3d_lib.graphics.neus import *

class NeuSRendererMixinBatched(ModelMixin):
    """
    NeuS Batched Renderer Mixin class
    
    NOTE: This is a mixin class!
    Refer: https://stackoverflow.com/questions/533631/what-is-a-mixin-and-why-are-they-useful
    !!!!: The target class for this mixin should also inherit from `ModelMixin`.
    """
    
    # NOTE: Configuration for common information usage in the forward process (and their defaults)
    use_view_dirs: bool = False # Determines if view_dirs are needed in forward process
    use_nablas: bool = False # Determines if nablas are required in forward process
    use_h_appear: bool = False # Determines if global per-frame appearance embeddings are necessary in forward process
    use_ts: bool = False # Determines if global timestamps are used in forward process
    use_pix: bool = False # Determines if pixel locations (in range [0,1]^2) are required in forward process
    use_fidx: bool = False # Determines if global frame indices are used in forward process
    use_bidx: bool = True # Determines if batch indices are used in forward process
    fwd_sdf_use_pix: bool = False
    fwd_sdf_use_h_appear: bool = False
    fwd_sdf_use_view_dirs: bool = False
    
    def __init__(
        self, 
        # Renderer mixin kwargs
        ray_query_cfg: dict = dict(), 
        accel_cfg: dict = None, 
        # Network kwargs
        **net_kwargs,
        ) -> None:
        
        mro = type(self).mro()
        super_class = mro[mro.index(NeuSRendererMixinBatched)+1]
        assert super_class is not ModelMixin, "Incorrect class inheritance. Three possible misuse scenarios:\n"\
            "Case 1: The Net class for mixin should also inherit from `ModelMixin`.\n"\
            "Case 2: RendererMixin should come before the Net class when inheriting.\n"\
            "Case 3: You should not directly instantiate this mixin class."
        
        super().__init__(**net_kwargs) # Will call network's __init__() (e.g. StyleLoTDNeuS ...) 
        
        self.ray_query_cfg = ray_query_cfg
        self.accel_cfg = accel_cfg

    def populate(self, *args, **kwargs):
        super().populate(*args, **kwargs)
        self.accel: accel_types_batched_t = None if self.accel_cfg is None \
            else get_accel(space=self.space, device=self.device, **self.accel_cfg)
        self.upsample_s_divisor = 1.0

    # @property
    # def space(self) -> BatchedBlockSpace:
    #     return super().space

    def sample_pts_uniform(self, num_samples: int):
        num_pts_per_batch = num_samples // self.B
        # NOTE: Returns normalized `x`
        x, bidx = self.space.cur_batch__sample_pts_uniform(self.B, num_pts_per_batch)
        ret = self.forward_sdf_nablas(x, bidx=bidx, skip_accel=True)
        for k, v in ret.items():
            ret[k] = v.to(x.dtype)
        ret['net_x'] = x # NOTE: in network's uniformed space; not in world space.
        if 'nablas' in ret:
            ret['nablas_norm'] = ret['nablas'].norm(dim=-1)
        return ret

    def sample_pts_in_occupied(self, num_samples: int):
        assert self.accel is not None, "Requires self.accel not to be None"
        x, bidx = self.accel.cur_batch__sample_pts_in_occupied(num_samples)
        ret = self.forward_sdf_nablas(x, bidx=bidx, skip_accel=True) # Do not upsate_samples here (usally there are too less samples here.)
        ret = {k: v.to(x.dtype) for k, v in ret.items()}
        ret['net_x'] = x # NOTE: in network's uniformed space; not in world space.
        return ret

    def set_condition(self, *args, ins_inds_per_batch: torch.LongTensor = None, skip_accel=False, **kwargs):
        super().set_condition(*args, **kwargs)
        if (not skip_accel) and (self.accel is not None):
            self.accel.set_condition(
                self.B, 
                ins_inds_per_batch=ins_inds_per_batch, 
                val_query_fn_normalized_x_bi=self.query_sdf)
            if self.training and (self.it > 0):
                self.accel.cur_batch__step(self.it, self.query_sdf)

    def clean_condition(self, skip_accel=False):
        super().clean_condition()
        if (not skip_accel) and (self.accel is not None):
            self.accel.clean_condition()

    def query_sdf(self, x: torch.Tensor, bidx: torch.LongTensor, **kwargs):
        return super().query_sdf(x, bidx=bidx, **kwargs)

    def forward_sdf(self, x: torch.Tensor, bidx: torch.LongTensor, skip_accel=False, **kwargs):
        ret = super().forward_sdf(x, bidx=bidx, **kwargs)
        if self.training and (not skip_accel) and (self.accel is not None):
            self.accel.cur_batch__collect_samples(x, bidx=bidx, val=ret['sdf'].data)
        return ret
    
    def forward_sdf_nablas(self, x: torch.Tensor, bidx: torch.LongTensor, skip_accel=False, **kwargs):
        ret = super().forward_sdf_nablas(x, bidx=bidx, **kwargs)
        if self.training and (not skip_accel) and (self.accel is not None):
            self.accel.cur_batch__collect_samples(x, bidx=bidx, val=ret['sdf'].data)
        return ret

    def training_initialize(self, config=dict(), logger=None, log_prefix: str=None, skip_accel=False) -> bool:
        updated = super().training_initialize(config, logger=logger, log_prefix=log_prefix)
        if (not skip_accel) and (self.accel is not None):
            self.accel.init(self.query_sdf, logger=logger)
        return updated

    def training_before_per_step(self, cur_it: int, logger: Logger = None):
        self.it = cur_it
        super().training_before_per_step(cur_it, logger=logger)
        # NOTE: Occ grid is moved to set_condition(), because potention occ grid update must happen after set_condition().
        # if self.training:
        #     self.accel.cur_batch__step(cur_it, self.query_sdf, logger=logger)

    def batched_ray_test(
        self, 
        rays_o: torch.Tensor, rays_d: torch.Tensor, 
        near=None, far=None, 
        compact_batch=False, **extra_ray_data):
        """Test batched input rays' intersection with the model's space (AABB, Blocks, etc.)

        Args:
            rays_o (torch.Tensor): [num_total_batches, num_total_rays,3] Batched input rays' origin
            rays_d (torch.Tensor): [num_total_batches, num_total_rays,3] Batched input rays' direction
            near (Union[float, torch.Tensor], optional): [num_total_batches, num_total_rays] tensor or float. Defaults to None.
            far (Union[float, torch.Tensor], optional): [num_total_batches, num_total_rays] tensor or float. Defaults to None.
            compact_batch (bool, optional): If true, will compactify the tested results' rays_bidx. Defaults to False.
                For example, if the resulting original `rays_full_bidx` are [1,1,3,3,3,4], 
                    which means there are no intersection with 0-th and 2-nd batches at all, 
                the compacted `rays_bidx` will be [0,0,1,1,1,2], 
                the `full_bidx_map` will be [1,3,4] 
                    so you can restore the original `rays_full_bidx` with `full_bidx_map[rays_bidx]`
                This mechanism is helpful for multi-object rendering to reduce waste,
                    when batches (objects) with no ray intersections will not be grown at all.

        Returns:
            dict: The ray_tested result. An example dict:
                num_rays:           int, Number of tested rays
                rays_inds:           [num_rays] tensor, indices of ray in `num_total_rays` for each tested ray
                rays_bidx:         [num_rays] tensor, indices of batch in `num_compacted_batches` for each tested ray
                rays_full_bidx:    [num_rays] tensor, indices of batch in `num_total_batches` for each tested ray
                full_bidx_map: [num_compacted_batches] tensor, the actual full batch ind corresponding to each compacted batch ind
                rays_o:     [num_rays, 3] tensor, the indexed and scaled rays' origin
                rays_d:     [num_rays, 3] tensor, the indexed and scaled rays' direction
                near:       [num_rays] tensor, entry depth of intersection
                far:        [num_rays] tensor, exit depth of intersection
        """
        assert (rays_o.dim() == 3) and (rays_d.dim() == 3), "Expect rays_o and rays_d to be of shape [B, N, 3]"
        for k, v in extra_ray_data.items():
            if isinstance(v, torch.Tensor):
                assert [*v.shape[:2]] == [*rays_o.shape[:2]], f"Expect `{k}` has the same shape prefix with rays_o {rays_o.shape[:2]}, but got {v.shape[:2]}"
        
        # B = rays_o.shape[0]
        # assert self.B is not None, "Please call set_condition first"
        # assert B == self.B, f"batch size mismatch: {B} != {self.B}"
        return self.space.cur_batch__ray_test(rays_o, rays_d, near, far, return_rays=True, compact_batch=compact_batch, **extra_ray_data)
    
    def batched_ray_query(
        self, 
        # Ray query inputs
        batched_ray_input: Dict[str, torch.Tensor]=None, batched_ray_tested: Dict[str, torch.Tensor]=None, 
        # Ray query function config
        config=dict(),
        # Function config
        return_buffer=False, return_details=False, render_per_obj_individual=False):
        """ Query the conditional model with batched input rays. 
            Conduct the core ray sampling, ray marching, ray upsampling and network query operations.

        Args:
            batched_ray_input (Dict[str, torch.Tensor], optional): All input rays.
                See more details in `batched_ray_test`. 
            batched_ray_tested (Dict[str, torch.Tensor], optional): Tested rays (Typicallly those that intersect with objects). A dict composed of:
                num_rays:           int, Number of tested rays
                rays_inds:          [num_rays] tensor, indices of ray in `num_total_rays` for each tested ray
                rays_bidx:          [num_rays] tensor, indices of batch in `num_compacted_batches` for each tested ray
                rays_full_bidx:     [num_rays] tensor, indices of batch in `num_total_batches` for each tested ray
                full_bidx_map:      [num_compacted_batches] tensor, the actual full batch ind corresponding to each compacted batch ind
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
                For now, two types of buffers might be queried depending on the ray sampling algorithms, namely `batched` buffers and `packed` buffers.
                
                If there are no tested rays or no hit rays, the returned buffer is empty:
                    'volume_buffer" {'type': 'empty'}
                
                An example `batched` buffer:
                    'volume_buffer': {
                        'type': 'batched',
                        'rays_inds_hit':     [num_rays_hit] tensor, ray indices in `num_total_rays` of the hit & queried rays
                        'rays_bidx_hit':   [num_rays_hit] tensor, batch indices in `num_compacted_batches` of the hit & queried rays
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
                        'rays_bidx_hit':   [num_rays_hit] tensor, batch indices in `num_compacted_batches` of the hit & queried rays
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
                        'mask_volume':      [num_total_batches, num_total_rays,] The rendered opacity / occupancy, in range [0,1]
                        'depth_volume':     [num_total_batches, num_total_rays,] The rendered real depth
                        'rgb_volume':       [num_total_batches, num_total_rays, 3] The rendered rgb, in range [0,1] (Only if `with_rgb` is True)
                        'normals_volume':   [num_total_batches, num_total_rays, 3] The rendered normals, in range [-1,1] (Only if `with_normal` is True)
                        'feature_volume':   [num_total_batches, num_total_rays, with_feature_dim] The rendered feature. (Only if `with_feature_dim` > 0)
                    }
        """

        #----------------
        # Inputs
        #----------------
        if batched_ray_tested is None:
            assert batched_ray_input is not None
            batched_ray_tested = self.batched_ray_test(**batched_ray_input, return_rays=True)
        
        #----------------
        # Shortcuts
        #----------------
        # NOTE: The device & dtype of output
        device, dtype = self.device, torch.float
        query_mode, with_rgb, with_normal = config.query_mode, config.with_rgb, config.with_normal
        forward_inv_s = config.get('forward_inv_s', self.forward_inv_s())
        # forward_inv_s = upsample_inv_s / 4.

        #----------------
        # Prepare outputs & compute outputs that are needed even when (num_rays==0)
        #----------------
        raw_ret = dict()
        if return_buffer:
            raw_ret['volume_buffer'] = dict(type='empty')
        if return_details:
            details = raw_ret['details'] = {}
            details['inv_s'] = forward_inv_s.item() if isinstance(forward_inv_s, torch.Tensor) else forward_inv_s
            details['s'] = 1./ details['inv_s']
            if (self.accel is not None) and hasattr(self.accel, 'debug_stats'):
                details['accel'] = self.accel.debug_stats()
        if render_per_obj_individual:
            prefix_rays = batched_ray_input['rays_o'].shape[:-1]
            raw_ret['rendered'] = rendered = dict(
                depth_volume = torch.zeros([*prefix_rays], dtype=dtype, device=device),
                mask_volume = torch.zeros([*prefix_rays], dtype=dtype, device=device),
            )
            if with_rgb:
                rendered['rgb_volume'] = torch.zeros([*prefix_rays, 3], dtype=dtype, device=device)
            if with_normal:
                rendered['normals_volume'] = torch.zeros([*prefix_rays, 3], dtype=dtype, device=device)
        
        if batched_ray_tested['num_rays'] == 0:
            return raw_ret
        
        if query_mode == 'coarse_multi_upsample':
            volume_buffer, query_details = neus_ray_query_coarse_multi_upsample(
                self, batched_ray_tested, with_rgb=with_rgb, with_normal=with_normal, 
                perturb=config.perturb, forward_inv_s=forward_inv_s,
                **config.query_param)

        elif query_mode == 'march_occ_multi_upsample':
            volume_buffer, query_details = neus_ray_query_march_occ_multi_upsample(
                self, batched_ray_tested, with_rgb=with_rgb, with_normal=with_normal, 
                perturb=config.perturb, forward_inv_s=forward_inv_s,
                **config.query_param)
        
        elif query_mode == 'march_occ_multi_upsample_compressed':
            volume_buffer, query_details = neus_ray_query_march_occ_multi_upsample_compressed(
                self, batched_ray_tested, with_rgb=with_rgb, with_normal=with_normal, 
                perturb=config.perturb, forward_inv_s=forward_inv_s,
                **config.query_param)
        
        else:
            raise RuntimeError(f"Invalid query_mode={query_mode}")
        
        if return_buffer:
            raw_ret['volume_buffer'] = volume_buffer
        
        if return_details:
            details.update(query_details)
            if config.get('with_near_sdf', False):
                fwd_kwargs = dict(
                    x=torch.addcmul(batched_ray_tested['rays_o'], batched_ray_tested['rays_d'], batched_ray_tested['near'].unsqueeze(-1)), 
                    bidx=batched_ray_tested['rays_bidx'])
                if self.use_ts: fwd_kwargs['ts'] = batched_ray_tested['rays_ts']
                if self.use_bidx: fwd_kwargs['bidx'] = batched_ray_tested['rays_bidx']
                if self.use_fidx: fwd_kwargs['fidx'] = batched_ray_tested['rays_fidx']
                if self.fwd_sdf_use_pix: fwd_kwargs['pix'] = batched_ray_tested['rays_pix']
                if self.fwd_sdf_use_h_appear: fwd_kwargs['h_appear'] = batched_ray_tested['rays_h_appear']
                if self.fwd_sdf_use_view_dirs: fwd_kwargs['v'] = F.normalize(batched_ray_tested['rays_d'], dim=-1)
                details['near_sdf'] = self.forward_sdf(**fwd_kwargs)['sdf']
        
        if render_per_obj_individual:
            if (buffer_type:=volume_buffer['type']) != 'empty':
                rays_inds_hit, rays_bidx_hit = volume_buffer['rays_inds_hit'], volume_buffer['rays_bidx_hit']
                if 'full_bidx_map' in batched_ray_tested:
                    rays_bidx_hit = batched_ray_tested['full_bidx_map'][rays_bidx_hit]
                inds = (rays_bidx_hit, rays_inds_hit)
                depth_use_normalized_vw = config.get('depth_use_normalized_vw', True)
            
                if buffer_type == 'batched':
                    volume_buffer['vw'] = vw = ray_alpha_to_vw(volume_buffer['opacity_alpha'])
                    rendered['mask_volume'][inds] = vw_sum = vw.sum(dim=-1)
                    if depth_use_normalized_vw:
                        # TODO: This can also be differed by training / non-training
                        vw_normalized = vw / (vw_sum.unsqueeze(-1)+1e-10)
                        rendered['depth_volume'][inds] = (vw_normalized * volume_buffer['t']).sum(dim=-1)
                    else:
                        rendered['depth_volume'][inds] = (vw * volume_buffer['t']).sum(dim=-1)
                        
                    if with_rgb:
                        rendered['rgb_volume'][inds] = (vw.unsqueeze(-1) * volume_buffer['rgb']).sum(dim=-2)
                    if with_normal:
                        # rendered['normals_volume'][inds] = (vw.unsqueeze(-1) * volume_buffer['nablas']).sum(dim=-2)
                        if self.training:
                            rendered['normals_volume'][inds] = (vw.unsqueeze(-1) * volume_buffer['nablas']).sum(dim=-2)
                        else:
                            rendered['normals_volume'][inds] = (vw.unsqueeze(-1) * F.normalize(volume_buffer['nablas'].clamp_(-1,1), dim=-1)).sum(dim=-2)
                elif buffer_type == 'packed':
                    pack_infos_hit = volume_buffer['pack_infos_hit']
                    # [num_sampels]
                    volume_buffer['vw'] = vw = packed_alpha_to_vw(volume_buffer['opacity_alpha'], pack_infos_hit)
                    # [num_rays_hit]
                    rendered['mask_volume'][inds] = vw_sum = packed_sum(vw.view(-1), pack_infos_hit)
                    # [num_samples]
                    if depth_use_normalized_vw:
                        vw_normalized = packed_div(vw, vw_sum + 1e-10, pack_infos_hit)
                        rendered['depth_volume'][inds] = packed_sum(vw_normalized * volume_buffer['t'].view(-1), pack_infos_hit)
                    else:
                        rendered['depth_volume'][inds] = packed_sum(vw.view(-1) * volume_buffer['t'].view(-1), pack_infos_hit)
                    if with_rgb:
                        rendered['rgb_volume'][inds] = packed_sum(vw.view(-1,1) * volume_buffer['rgb'].view(-1,3), pack_infos_hit)
                    if with_normal:
                        # rendered['normals_volume'][inds] = packed_sum(vw.view(-1,1) * volume_buffer['nablas'].view(-1,3), pack_infos_hit)
                        if self.training:
                            rendered['normals_volume'][inds] = packed_sum(vw.view(-1,1) * volume_buffer['nablas'].view(-1,3), pack_infos_hit)
                        else:
                            rendered['normals_volume'][inds] = packed_sum(vw.view(-1,1) * F.normalize(volume_buffer['nablas'].clamp_(-1,1), dim=-1).view(-1,3), pack_infos_hit)
        return raw_ret

"""
@file   base.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Common APIs for models
"""

from copy import deepcopy
from numbers import Number
from typing import Dict, List, Union

import torch
import torch.nn as nn

from nr3d_lib.logger import Logger
from nr3d_lib.models.utils import get_optimizer, get_scheduler
from nr3d_lib.models.spatial import AABBSpace, BatchedBlockSpace, ForestBlockSpace

class ModelMixin:
    """
    Defines common APIs for models
    """
    # Every field has a valid representing space.
    # BUG: Assigning to default None causes self.space to be always None 
    #       (because pytorch overwrites setattr)
    #     -> Do not set default attributes to fields with nn.Module values
    space: Union[AABBSpace, BatchedBlockSpace, ForestBlockSpace]
    ray_query_cfg: dict = dict() # Config of `ray_query` / `batched_ray_query` process

    optimizer = None
    scheduler = None
    it: int = -1 # `-1` is the undefined state (a model that has not yet initialized)

    #--------------------------------------
    #---- Preparation and finializations
    #--------------------------------------   
    def model_setup(self, ):
        """
        Operations that need to be executed only once throughout the entire rendering process, 
        as long as the network params remains untouched.
        
        TODO: What would be the difference with populate()?
        """
        pass

    def training_initialize(self, config=dict(), logger=None, log_prefix: str=None) -> bool:
        """
        Operations that needs to take place when it==0
        """
        pass

    def training_setup(self, training_cfg: Union[Number, dict], name_prefix: str = ''):
        self.optimizer = None
        self.scheduler = None
        if hasattr(self, 'parameters'):
            training_cfg = deepcopy(training_cfg)
            if isinstance(training_cfg, Number):
                training_cfg = {'lr': training_cfg}
            # Optionally, configure lr scheduler
            sched_kwargs = training_cfg.pop('scheduler', None)
            param_groups = {
                'name': 'params' if name_prefix == '' else name_prefix, 
                'params': self.parameters(), 
                'capturable': True
                # NOTE: Potential dict items for lr scheduler & grad clipping
                # 'lr_init', 
                # 'clip_grad_val', 
                # 'clip_grad_norm', 
            }
            self.optimizer = get_optimizer([param_groups], **training_cfg)
            if sched_kwargs is not None:
                self.scheduler = get_scheduler(self.optimizer, **sched_kwargs)

    def training_update_lr(self, it: int):
        if self.scheduler is not None:
            self.scheduler.step(it)
    
    @torch.no_grad()
    def training_clip_grad(self):
        if self.optimizer is not None:
            for group in self.optimizer.param_groups:
                if 'clip_grad_val' in group:
                    nn.utils.clip_grad.clip_grad_value_(group['params'], group['clip_grad_val'])
                elif 'clip_grad_norm' in group:
                    nn.utils.clip_grad.clip_grad_norm_(group['params'], group['clip_grad_norm'])

    def training_before_per_step(self, cur_it: int, logger: Logger=None):
        """"
        Operations that need to be executed before each training step (before `trainer.forward`).
        """
        pass
    
    def training_after_per_step(self, cur_it: int, logger: Logger=None):
        """
        Operations that need to be executed after each training step (after `optmizer.step`).
        """
        pass

    # def training_finalize(self, final_it: int, logger: Logger = None):
    #     self.training_before_per_step(cur_it=final_it, logger=logger)

    def rendering_before_per_view(self, renderer, observer, per_frame_info: dict={}):
        """
        Operations that need to be executed for every frame or view.
        Here it's internally called within the renderer.
        """
        pass

    def stat_param(self, with_grad=False, prefix: str = ''):
        """
        Produce statistics on the current param
        """
        pass

    #--------------------------------------
    #---- Rendering related
    #--------------------------------------
    def ray_test(
        self, 
        rays_o: torch.Tensor, rays_d: torch.Tensor, 
        near: Union[float, torch.Tensor]=None, far: Union[float, torch.Tensor]=None, 
        **extra_ray_data: Dict[str, torch.Tensor]
        ) -> dict:
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
        raise NotImplementedError
        assert (rays_o.dim() == 2) and (rays_d.dim() == 2), "Expect `rays_o` and `rays_d` to be of shape [N, 3]"
        for k, v in extra_ray_data.items():
            if isinstance(v, torch.Tensor):
                assert v.shape[0] == rays_o.shape[0], f"Expect `{k}` has the same shape prefix with rays_o {rays_o.shape[0]}, but got {v.shape[0]}"

    def ray_query(
        self, 
        ray_input: Dict[str, torch.Tensor] = None, 
        ray_tested: Dict[str, torch.Tensor] = None, 
        config=dict(), 
        return_buffer=False, return_details=False, render_per_obj_individual=False
        ) -> dict:
        """ Query the model with input rays. 
            Conduct the core ray sampling, ray marching, ray upsampling and network query operations.

        Args:
            ray_input (Dict[str, torch.Tensor], optional): All input rays.
                See more details in `ray_test`
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
        raise NotImplementedError
    
    #---- For shared / conditional models
    def set_condition(self, *args, **kwargs):
        """
        Set the condition params for shared models
        """
        raise NotImplementedError
    
    def clean_condition(self):
        """
        Clean the condition set before.
        """
        raise NotImplementedError
    
    def batched_ray_test(
        self, 
        rays_o: torch.Tensor, rays_d: torch.Tensor, 
        near: Union[float, torch.Tensor]=None, far: Union[float, torch.Tensor]=None, 
        compact_batch=False, **extra_ray_data: Dict[str, torch.Tensor]) -> dict:
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
                near:       [num_rays] tensor, entry depth of intersection
                far:        [num_rays] tensor, exit depth of intersection
                rays_o:     [num_rays, 3] tensor, the indexed and scaled rays' origin
                rays_d:     [num_rays, 3] tensor, the indexed and scaled rays' direction
        """
        raise NotImplementedError
        assert (rays_o.dim() == 3) and (rays_d.dim() == 3), "Expect rays_o and rays_d to be of shape [B, N, 3]"
        for k, v in extra_ray_data.items():
            if isinstance(v, torch.Tensor):
                assert [*v.shape[:2]] == [*rays_o.shape[:2]], f"Expect `{k}` has the same shape prefix with rays_o {rays_o.shape[:2]}, but got {v.shape[:2]}"

    def batched_ray_query(
        self, 
        batched_ray_input: Dict[str, torch.Tensor]=None, batched_ray_tested: Dict[str, torch.Tensor]=None, 
        config=dict(),
        return_buffer=False, return_details=False, render_per_obj_individual=False
        ) -> dict:
        """ Query the conditional model with batched input rays. 
            Conduct the core ray sampling, ray marching, ray upsampling and network query operations.

        Args:
            batched_ray_input (Dict[str, torch.Tensor], optional): All input rays.
                See more details in `batched_ray_test`
            batched_ray_tested (Dict[str, torch.Tensor], optional): Tested rays (Typicallly those that intersect with objects). A dict composed of:
                num_rays:           int, Number of tested rays
                rays_inds:           [num_rays] tensor, indices of ray in `num_total_rays` for each tested ray
                rays_bidx:         [num_rays] tensor, indices of batch in `num_compacted_batches` for each tested ray
                rays_full_bidx:    [num_rays] tensor, indices of batch in `num_total_batches` for each tested ray
                full_bidx_map: [num_compacted_batches] tensor, the actual full batch ind corresponding to each compacted batch ind
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
        raise NotImplementedError
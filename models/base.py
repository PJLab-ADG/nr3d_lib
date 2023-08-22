"""
@file   base.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Common APIs for models
"""

from numbers import Number
from typing import Dict, List, Union

import torch
import torch.nn as nn

from nr3d_lib.logger import Logger
from nr3d_lib.config import ConfigDict
from nr3d_lib.models.spatial import AABBSpace, BatchedBlockSpace, ForestBlockSpace
from nr3d_lib.models.utils import get_param_group

class ModelMixin:
    """
    Defines common APIs for models
    """
    # Every field has a valid representing space.
    # BUG: Assigning to default None causes self.space to be always None 
    #       (because pytorch overwrites setattr)
    #     -> Do not set default attributes to fields with nn.Module values
    space: Union[AABBSpace, BatchedBlockSpace, ForestBlockSpace]

    ray_query_cfg: ConfigDict = ConfigDict() # Config of `ray_query` / `batched_ray_query` process

    #--------------------------------------
    #---- Preparation and finializations
    #--------------------------------------   
    def preprocess_per_train_step(self, cur_it: int, logger: Logger=None):
        """"
        Operations that need to be executed before each training step (before `trainer.forward`).
        """
        pass
    
    def postprocess_per_train_step(self, cur_it: int, logger: Logger=None):
        """
        Operations that need to be executed after each training step (after `optmizer.step`).
        """
        pass

    def preprocess_model(self, ):
        """
        Operations that need to be executed only once throughout the entire rendering process, 
        as long as the network params remains untouched.
        """
        pass
    
    def preprocess_per_render_frame(self, renderer, observer, per_frame_info: dict={}):
        """
        Operations that need to be executed for every frame or view.
        """
        pass

    #--------------------------------------
    #---- Other misc
    #--------------------------------------
    def get_param_group(self, optim_cfg: Union[Number, dict], prefix: str = '') -> List[dict]:
        """
        Get parameter groups from the given optimization configs
        """
        return get_param_group(self, optim_cfg, prefix=prefix)
    
    def custom_grad_clip_step(self, ):
        """
        Custom operations to clip gradients before optimizer.step
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
                ray_inds:   [num_rays] tensor, ray indices in `num_total_rays` of the tested rays
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
        config=ConfigDict(), 
        return_buffer=False, return_details=False, render_per_obj=False
        ) -> dict:
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
        raise NotImplementedError
    
    #---- For shared / conditional models
    def set_condition(self, batched_infos: dict):
        """
        Set the condition params for shared models
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
            compact_batch (bool, optional): If true, will compactify the tested results' batch_inds. Defaults to False.
                For example, if the resulting original `full_batch_inds` are [1,1,3,3,3,4], 
                    which means there are no intersection with 0-th and 2-nd batches at all, 
                the compacted `batch_inds` will be [0,0,1,1,1,2], 
                the `full_batch_ind_map` will be [1,3,4] 
                    so you can restore the original `full_batch_inds` with `full_batch_ind_map[batch_inds]`
                This mechanism is helpful for multi-object rendering to reduce waste,
                    when batches (objects) with no ray intersections will not be grown at all.

        Returns:
            dict: The ray_tested result. An example dict:
                num_rays:           int, Number of tested rays
                ray_inds:           [num_rays] tensor, indices of ray in `num_total_rays` for each tested ray
                batch_inds:         [num_rays] tensor, indices of batch in `num_compacted_batches` for each tested ray
                full_batch_inds:    [num_rays] tensor, indices of batch in `num_total_batches` for each tested ray
                full_batch_ind_map: [num_compacted_batches] tensor, the actual full batch ind corresponding to each compacted batch ind
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
        config=ConfigDict(),
        return_buffer=False, return_details=False, render_per_obj=False
        ) -> dict:
        """ Query the conditional model with batched input rays. 
            Conduct the core ray sampling, ray marching, ray upsampling and network query operations.

        Args:
            batched_ray_input (Dict[str, torch.Tensor], optional): All input rays.
                See more details in `batched_ray_test`
            batched_ray_tested (Dict[str, torch.Tensor], optional): Tested rays (Typicallly those that intersect with objects). A dict composed of:
                num_rays:           int, Number of tested rays
                ray_inds:           [num_rays] tensor, indices of ray in `num_total_rays` for each tested ray
                batch_inds:         [num_rays] tensor, indices of batch in `num_compacted_batches` for each tested ray
                full_batch_inds:    [num_rays] tensor, indices of batch in `num_total_batches` for each tested ray
                full_batch_ind_map: [num_compacted_batches] tensor, the actual full batch ind corresponding to each compacted batch ind
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
                For now, two types of buffers might be queried depending on the ray sampling algorithms, namely `batched` buffers and `packed` buffers.
                
                If there are no tested rays or no hit rays, the returned buffer is empty:
                    'volume_buffer" {'type': 'empty'}
                
                An example `batched` buffer:
                    'volume_buffer': {
                        'type': 'batched',
                        'ray_inds_hit':     [num_rays_hit] tensor, ray indices in `num_total_rays` of the hit & queried rays
                        'batch_inds_hit':   [num_rays_hit] tensor, batch indices in `num_compacted_batches` of the hit & queried rays
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
                        'batch_inds_hit':   [num_rays_hit] tensor, batch indices in `num_compacted_batches` of the hit & queried rays
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
                        'mask_volume':      [num_total_batches, num_total_rays,] The rendered opacity / occupancy, in range [0,1]
                        'depth_volume':     [num_total_batches, num_total_rays,] The rendered real depth
                        'rgb_volume':       [num_total_batches, num_total_rays, 3] The rendered rgb, in range [0,1] (Only if `with_rgb` is True)
                        'normals_volume':   [num_total_batches, num_total_rays, 3] The rendered normals, in range [-1,1] (Only if `with_normal` is True)
                        'feature_volume':   [num_total_batches, num_total_rays, with_feature_dim] The rendered feature. (Only if `with_feature_dim` > 0)
                    }
        """
        raise NotImplementedError
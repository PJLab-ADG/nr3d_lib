"""
EmerNeRF

NOTE:
- dynamic_only is basically what remains after removing the static field and sky
- sky_net is optional and can also be handled by the renderer
- [x] We should store timestamps / time embeddings, different models may require different time inputs, \
    and the timestamp is an important basis for several dynamic fields, especially the flowfield, which is equivalent to part of the network structure

Main differences from the official model:
- [x] (WIP) Allow different cameras to be at different subdivided timestamps?
  - [x] Need to include timestamp as part of scene meta
  - [x] The camera can also be allowed to store timestamps, which can handle situations where different cameras are at different timestamps, or even different pixels are at different timestamps
  - [x] The overall definition of rays_ts needs some changes, including rays which increasingly seem to be better formed as a struct, each ray can bind more additional information, including frame index, pixel position, floating-point timestamp, etc.
  - [x] In this way, even if there are only 200 frames in total, the appearance of intermediate tween timestamps is allowed;
  - [x] ~~Consider letting the flow field predict velocity, and the specific displacement estimate can be obtained through v*t?~~
  - [x] Under the setting where different cameras are at different timestamps, how should the flow field be defined? How to aggregate multi-frame information? -> Can be expressed with dt

- [x] (WIP) 4D occupancy grid acceleration, instead of prop net
  - [x] Time can be subdivided at custom intervals (for example, a coarser scale than before, and pedestrians, vehicles, etc. can consider using different time division granularity)\
      For any continuous timestamp input, approximate it with the nearest discrete time scale
"""

__all__ = [
    'EmerNeRF', 
    'EmerNeRFModel', 
    'EmerNeRFOnlyDynamic', 
    'EmerNeRFOnlyDynamicModel'
]

import os
import functools
import numpy as np
from copy import deepcopy
from numbers import Number
from typing import Callable, Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from nr3d_lib.maths import trunc_exp
from nr3d_lib.utils import check_to_torch, torch_dtype
from nr3d_lib.profile import profile
from nr3d_lib.logger import Logger

from nr3d_lib.models.model_base import ModelMixin
from nr3d_lib.models.blocks import get_blocks
from nr3d_lib.models.embeddings import SeqEmbedding
from nr3d_lib.models.layers import get_nonlinearity
from nr3d_lib.models.spatial import AABBDynamicSpace
from nr3d_lib.models.grid_encodings.lotd import LoTDEncoding
from nr3d_lib.models.grid_encodings.permuto import PermutoEncoding
from nr3d_lib.models.grid_encodings.multires_decoder import get_multires_decoder
from nr3d_lib.models.accelerations import get_accel, accel_types_dynamic_t, OccGridAccelStaticAndDynamic
from nr3d_lib.models.fields.nerf import RadianceNet
from nr3d_lib.models.fields_dynamic.nerf.renderer_mixin import NerfRendererMixinDynamic

from nr3d_lib.graphics.pack_ops import *
from nr3d_lib.graphics.nerf import *

class EmerNeRF(ModelMixin, nn.Module):
    def __init__(
        self, 
        n_geometry_feat: int = 64, 
        n_semantic_feat: int = 0, 
        
        static_encoding_cfg: dict = dict(type='lotd', param=...), 
        static_decoder_cfg: dict = dict(type='mlp', D=1, W=64, activation='relu'), 
        
        time_embedding_cfg: dict = dict(dim=1, learnable=False, 
                                  weight_init=dict(type='linspace', start=-10., end=10.)), 
        
        dynamic_encoding_cfg: dict = dict(type='permuto', param=...), 
        dynamic_decoder_cfg: dict = dict(type='mlp', D=1, W=64), 
        
        use_flow_field: bool = True, 
        use_flow_in_obj: bool = True, 
        flow_encoding_cfg: dict = dict(type='permuto', param=...), 
        flow_decoder_cfg: dict = dict(type='mlp', D=2, W=64), 
        
        use_shadow: bool = True, 
        shadow_cfg: dict = dict(D=1, W=64), 
        
        density_activation: Union[str, Callable] = lambda x: trunc_exp(x-1), 
        radiance_cfg: dict = dict(D=2, W=64, 
                                  dir_embed_cfg=dict(type='spherical', degree=4)), 
        
        dtype=torch.half, device=None, 
        ) -> None:
        """ 
        Define the network structure of EmerNeRF.
        """
        
        super().__init__()
        
        self.dtype = torch_dtype(dtype)
        self.set_device = device
        
        self.space: AABBDynamicSpace = None
        
        self.n_geometry_feat = n_geometry_feat
        self.n_semantic_feat = n_semantic_feat
        
        self.static_encoding_cfg = static_encoding_cfg
        self.static_decoder_cfg = static_decoder_cfg
        
        self.time_embedding_cfg = time_embedding_cfg
        
        self.dynamic_encoding_cfg = dynamic_encoding_cfg
        self.dynamic_decoder_cfg = dynamic_decoder_cfg
        
        self.use_shadow = use_shadow
        self.shadow_cfg = shadow_cfg
        
        self.use_flow_field = use_flow_field
        self.use_flow_in_obj = use_flow_in_obj
        self.flow_encoding_cfg = flow_encoding_cfg
        self.flow_decoder_cfg = flow_decoder_cfg
        
        self.radiance_cfg = radiance_cfg
        self.density_activation = density_activation

    @property
    def device(self) -> torch.device:
        return self.space.device

    def populate(
        self, 
        ts_keyframes: Union[np.ndarray, torch.Tensor], 
        bounding_size=None, aabb=None, device=None):
        
        device = device or self.set_device
        self.set_device = device
        
        #----------------------------------------
        #---- Valid representing space
        #----------------------------------------
        self.space = AABBDynamicSpace(bounding_size=bounding_size, aabb=aabb, dtype=torch.float, device=device)
        
        #----------------------------------------
        #---- Static field
        #----------------------------------------
        static_encoding_cfg = deepcopy(self.static_encoding_cfg)
        if static_encoding_cfg['type'].lower() == 'lotd':
            self.static_encoding = LoTDEncoding(
                3, space=self.space, 
                **static_encoding_cfg['param'], dtype=self.dtype, device=device)
        elif static_encoding_cfg['type'].lower() == 'permuto':
            self.static_encoding = PermutoEncoding(
                3, space=self.space, 
                **static_encoding_cfg['param'], dtype=self.dtype, device=device)
        else:
            raise RuntimeError(f"Invalid static_encoding_cfg['type']={static_encoding_cfg['type']}")

        self.static_decoder, self.static_decoder_type = get_multires_decoder(
            self.static_encoding.level_n_feats, 
            self.n_geometry_feat + self.n_semantic_feat, 
            **self.static_decoder_cfg, 
            dtype=self.dtype, device=device
        )
        
        #----------------------------------------
        #---- Time embedding
        #----------------------------------------
        ts_keyframes = check_to_torch(ts_keyframes, dtype=torch.float, device=device)
        # NOTE: Use normalized `ts`
        ts_keyframes = self.space.normalize_ts(ts_keyframes)
        dt = (ts_keyframes[-1] - ts_keyframes[0]).item() / (len(ts_keyframes) - 1)
        self.dt: float = dt # Save `dt` for using forward & backward flow to aggregate features
        self.time_embedding = SeqEmbedding(ts_keyframes, **self.time_embedding_cfg, dtype=torch.float, device=device)
        x_scale = (self.space.radius3d / self.space.radius3d.min()).tolist()
        permuto_input_scale = check_to_torch([*x_scale, *[1]*self.time_embedding.embedding_dim], dtype=torch.float, device=device)

        #----------------------------------------
        #---- Dynamic field
        #----------------------------------------
        dynamic_encoding_cfg = deepcopy(self.dynamic_encoding_cfg)
        if dynamic_encoding_cfg['type'].lower() == 'lotd':
            self.dynamic_encoding = LoTDEncoding(
                3+self.time_embedding.embedding_dim, space=self.space, 
                **dynamic_encoding_cfg['param'], dtype=self.dtype, device=device)
        elif dynamic_encoding_cfg['type'].lower() == 'permuto':
            pos_scale = permuto_input_scale * dynamic_encoding_cfg['param'].pop('pos_scale', 1.0)
            self.dynamic_encoding = PermutoEncoding(
                3+self.time_embedding.embedding_dim, 
                space=self.space, pos_scale=pos_scale, 
                **dynamic_encoding_cfg['param'], dtype=self.dtype, device=device)
        else:
            raise RuntimeError(f"Invalid dynamic_encoding_cfg['type']={dynamic_encoding_cfg['type']}")
        
        self.dynamic_decoder, self.dynamic_decoder_type = get_multires_decoder(
            self.dynamic_encoding.level_n_feats, 
            self.n_geometry_feat + self.n_semantic_feat, 
            **self.dynamic_decoder_cfg, 
            dtype=self.dtype, device=device
        )

        #----------------------------------------
        #---- Flow field
        #----------------------------------------
        if self.use_flow_field:
            self.h_dynamic_3t_weights = [0.25, 0.5, 0.25]
            flow_encoding_cfg = deepcopy(self.flow_encoding_cfg)
            if flow_encoding_cfg['type'].lower() == 'lotd':
                self.flow_encoding = LoTDEncoding(
                    3+self.time_embedding.embedding_dim, space=self.space, 
                    **flow_encoding_cfg['param'], dtype=self.dtype, device=device)
            elif flow_encoding_cfg['type'].lower() == 'permuto':
                pos_scale = permuto_input_scale * flow_encoding_cfg['param'].pop('pos_scale', 1.0)
                self.flow_encoding = PermutoEncoding(
                    3+self.time_embedding.embedding_dim, 
                    space=self.space, pos_scale=pos_scale, 
                    **flow_encoding_cfg['param'], dtype=self.dtype, device=device)
            else:
                raise RuntimeError(f"Invalid flow_encoding_cfg['type']={flow_encoding_cfg['type']}")
            
            self.flow_decoder, self.flow_decoder_type = get_multires_decoder(
                self.flow_encoding.level_n_feats, 
                6, # Outputs forward displacement (3) + backward displacement (3)
                **self.flow_decoder_cfg, 
                dtype=self.dtype, device=device
            )

        #----------------------------------------
        #---- Shadow head
        #----------------------------------------
        if self.use_shadow:
            self.dynamic_shadow_net = get_blocks(
                self.n_geometry_feat, 
                1, 
                **self.shadow_cfg, 
                output_activation='sigmoid', 
                dtype=self.dtype, device=device
            )

        #----------------------------------------
        #---- Density head
        #----------------------------------------
        if isinstance(self.density_activation, (str, dict)):
            self.density_activation = get_nonlinearity(self.density_activation).nl
        
        #----------------------------------------
        #---- Color head 
        #----------------------------------------
        self.radiance_net = RadianceNet(
            # use_pos=False, 
            # use_nablas=False, 
            # use_view_dirs=True, 
            n_extra_feat=self.n_geometry_feat, 
            **self.radiance_cfg, 
            dtype=self.dtype, device=device
        )
        self.use_view_dirs = self.radiance_net.use_view_dirs
        self.use_h_appear = self.radiance_net.use_h_appear

    def forward_h_static(
        self, x: torch.Tensor, *, input_normalized=True
        ) -> Dict[str, torch.Tensor]:
        
        if not input_normalized:
            x = self.space.normalize_coords(x)
        enc_static = self.static_encoding(x)
        h_static = self.static_decoder(enc_static)
        return dict(h_static=h_static)

    def forward_h_dynamic(
        self, x: torch.Tensor, *, ts: torch.Tensor, input_normalized=True
        ) -> Dict[str, torch.Tensor]:
        
        if not input_normalized:
            x = self.space.normalize_coords(x)
            ts = self.space.normalize_ts(ts)
        
        z_time = self.time_embedding(ts, mode='interp')
        input = torch.cat((x, z_time), dim=-1)
        
        with torch.autocast(device_type='cuda', dtype=self.dtype):
            if self.use_flow_field:
                # Get prev, next flow
                flow_output = self.flow_decoder(self.flow_encoding(input))
                flow_fwd, flow_bwd = flow_output[..., :3], flow_output[..., 3:]
                
                # Get warped space-time
                if self.training:
                    noise_ratio = 1.5 * torch.rand_like(ts)
                    z_time_bwd = self.time_embedding(ts - self.dt * noise_ratio, mode='interp')
                    z_time_fwd = self.time_embedding(ts + self.dt * noise_ratio, mode='interp')
                    if self.use_flow_in_obj:
                        x_bwd = x + flow_bwd / self.space.radius3d * noise_ratio.unsqueeze(-1)
                        x_fwd = x + flow_fwd / self.space.radius3d * noise_ratio.unsqueeze(-1)
                    else:
                        x_bwd = x + flow_bwd * noise_ratio.unsqueeze(-1)
                        x_fwd = x + flow_fwd * noise_ratio.unsqueeze(-1)
                else:
                    z_time_bwd = self.time_embedding(ts - self.dt, mode='interp')
                    z_time_fwd = self.time_embedding(ts + self.dt, mode='interp')
                    if self.use_flow_in_obj:
                        x_bwd = x + flow_bwd / self.space.radius3d
                        x_fwd = x + flow_fwd / self.space.radius3d
                    else:
                        x_bwd = x + flow_bwd
                        x_fwd = x + flow_fwd

                input_fwd = torch.cat((x_fwd, z_time_fwd), dim=-1)
                input_bwd = torch.cat((x_bwd, z_time_bwd), dim=-1)
                
                # Forward dynamic field on prev,now,next
                input_3t = torch.stack((input_bwd, input, input_fwd), dim=0)
                enc_dynamic_3t = self.dynamic_encoding(input_3t)
                h_dynamic_3t = self.dynamic_decoder(enc_dynamic_3t)
                
                # Aggregate 3 frames
                w = self.h_dynamic_3t_weights
                h_dynamic = w[0] * h_dynamic_3t[0] + w[1] * h_dynamic_3t[1] + w[2] * h_dynamic_3t[2]
                
                if self.training and torch.is_grad_enabled():
                    # For consistency
                    fwd_pred_flow = self.flow_decoder(self.flow_encoding(input_fwd))
                    bwd_pred_flow = self.flow_decoder(self.flow_encoding(input_bwd))
                else:
                    fwd_pred_flow = bwd_pred_flow = None
            else:
                # Only use the features of the current `ts`
                enc_dynamic = self.dynamic_encoding(input)
                h_dynamic = self.dynamic_decoder(enc_dynamic)
        
        #---- Gather returns
        ret = dict(h_dynamic=h_dynamic)
        if self.use_flow_field:
            ret.update(flow_fwd=flow_fwd, flow_bwd=flow_bwd)
            if fwd_pred_flow is not None:
                ret.update(
                    flow_fwd_pred_bwd=fwd_pred_flow[..., 3:], 
                    flow_bwd_pred_fwd=bwd_pred_flow[..., :3], 
                )
        return ret

    def forward(
        self, 
        x: torch.Tensor, *, 
        v: torch.Tensor, 
        ts: torch.Tensor, 
        h_appear: torch.Tensor = None, 
        input_normalized: bool=True, 
        with_rgb=True, with_h_sem=False, 
        return_flow: bool=True, 
        return_details: bool=False, 
        return_seperate: bool=None, 
        ) -> Dict[str, torch.Tensor]:
        
        if not input_normalized:
            x = self.space.normalize_coords(x)
            ts = self.space.normalize_ts(ts)
        
        ret_static = self.forward_h_static(x, input_normalized=True)
        h_static = ret_static['h_static']
        
        ret_dynamic = self.forward_h_dynamic(x, ts=ts, input_normalized=True)
        h_dynamic = ret_dynamic['h_dynamic']

        with torch.autocast(device_type='cuda', dtype=self.dtype):
            sigma_static = self.density_activation(h_static[..., 0])
            sigma_dynamic = self.density_activation(h_dynamic[..., 0])
            sigma = sigma_static + sigma_dynamic
            if with_rgb or with_h_sem:
                # NOTE: If not autocast() or .float(), below division will introduce nan grads when float16.
                ratio_dynamic = sigma_dynamic / (sigma+1e-6)
                ratio_static = sigma_static / (sigma+1e-6)
            if with_rgb:
                rgb_static = self.radiance_net(x=x, v=v, n=None, h_extra=h_static[..., :self.n_geometry_feat], h_appear=h_appear)['rgb']
                rgb_dynamic = self.radiance_net(x=x, v=v, n=None, h_extra=h_dynamic[..., :self.n_geometry_feat], h_appear=h_appear)['rgb']
                if self.use_shadow:
                    ratio_shadow = self.dynamic_shadow_net(h_dynamic).squeeze(-1)
                    rgb = (ratio_static * (1 - ratio_shadow)).unsqueeze(-1) * rgb_static \
                        + ratio_dynamic.unsqueeze(-1) * rgb_dynamic
                else:
                    rgb = ratio_static.unsqueeze(-1) * rgb_static \
                        + ratio_dynamic.unsqueeze(-1) * rgb_dynamic
            if with_h_sem:
                h_sem_static = h_static[..., self.n_geometry_feat:]
                h_sem_dynamic = h_dynamic[..., self.n_geometry_feat:]
                h_sem = ratio_static.unsqueeze(-1) * h_sem_static \
                    + ratio_dynamic.unsqueeze(-1) * h_sem_dynamic
        
        #---- Gather returns
        ret = dict(sigma=sigma)
        if with_rgb:
            ret.update(rgb=rgb)
        if with_h_sem:
            ret.update(h_sem=h_sem)
        if return_flow:
            for k in ['flow_fwd', 'flow_fwd_pred_bwd', 'flow_bwd', 'flow_bwd_pred_fwd']:
                if k in ret_dynamic:
                    ret[k] = ret_dynamic[k]
        if return_details:
            ret.update(ret_static)
            ret.update(ret_dynamic)
        if return_details or return_seperate:
            ret.update(sigma_static=sigma_static, sigma_dynamic=sigma_dynamic)
            if with_rgb:
                ret.update(rgb_static=rgb_static, rgb_dynamic=rgb_dynamic)
                if self.use_shadow:
                    ret.update(ratio_shadow=ratio_shadow)
            if with_h_sem:
                ret.update(h_sem_static=h_sem_static, h_sem_dynamic=h_sem_dynamic)
        return ret

    def forward_density(
        self, 
        x: torch.Tensor, *, 
        ts: torch.Tensor, 
        input_normalized: bool=True, 
        return_flow: bool=True, 
        return_details: bool=False, 
        return_seperate: bool=None, 
        ) -> Dict[str, torch.Tensor]:
        
        if not input_normalized:
            x = self.space.normalize_coords(x)
            ts = self.space.normalize_ts(ts)
        
        ret_static = self.forward_h_static(x, input_normalized=True)
        h_static = ret_static['h_static']
        
        ret_dynamic = self.forward_h_dynamic(x, ts=ts, input_normalized=True)
        h_dynamic = ret_dynamic['h_dynamic']

        with torch.autocast(device_type='cuda', dtype=self.dtype):
            sigma_static = self.density_activation(h_static[..., 0])
            sigma_dynamic = self.density_activation(h_dynamic[..., 0])
            sigma = sigma_static + sigma_dynamic
        
        #---- Gather returns
        ret = dict(sigma=sigma)
        if return_flow:
            for k in ['flow_fwd', 'flow_bwd', 'flow_fwd_pred_bwd', 'flow_bwd_pred_fwd']:
                if k in ret_dynamic:
                    ret[k] = ret_dynamic[k]
        if return_details:
            ret.update(ret_static)
            ret.update(ret_dynamic)
        if return_details or return_seperate:
            ret.update(sigma_static=sigma_static, sigma_dynamic=sigma_dynamic)
        return ret
        
    @torch.no_grad()
    def query_density(self, x: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        h_static = self.forward_h_static(x, input_normalized=True)['h_static']
        h_dynamic = self.forward_h_dynamic(x, ts=ts, input_normalized=True)['h_dynamic']
        sigma_static = self.density_activation(h_static[..., 0])
        sigma_dynamic = self.density_activation(h_dynamic[..., 0])
        sigma = sigma_static + sigma_dynamic
        return sigma

    def get_weight_reg(self, norm_type: float = 2.0) -> torch.Tensor:
        items = [
            self.static_decoder.get_weight_reg(norm_type).to(self.device), 
            self.dynamic_decoder.get_weight_reg(norm_type).to(self.device), 
            self.radiance_net.get_weight_reg(norm_type).to(self.device), 
        ]
        if self.use_flow_field:
            items.append(self.flow_decoder.get_weight_reg(norm_type).to(self.device))
        if self.use_shadow:
            items.append(self.dynamic_shadow_net.get_weight_reg(norm_type).to(self.device))
        return torch.cat(items)

    """ Seperate static / dynamic model """
    def forward_static(
        self, x: torch.Tensor, *, v: torch.Tensor, h_appear: torch.Tensor = None, 
        input_normalized: bool=True, with_rgb: bool=False, with_h_sem: bool=False, 
        ) -> Dict[str, torch.Tensor]:
        if not input_normalized:
            x = self.space.normalize_coords(x)
        ret_static = self.forward_h_static(x, input_normalized=True)
        h_static = ret_static['h_static']
        with torch.autocast(device_type='cuda', dtype=self.dtype):
            sigma_static = self.density_activation(h_static[..., 0])
            if with_rgb:
                rgb_static = self.radiance_net(x=x, v=v, n=None, h_extra=h_static[..., :self.n_geometry_feat], h_appear=h_appear)['rgb']
            if with_h_sem:
                h_sem_static = h_static[..., self.n_geometry_feat:]
        #---- Gather returns
        ret = dict(sigma=sigma_static)
        if with_rgb:
            ret['rgb'] = rgb_static
        if with_h_sem:
            ret['h_sem'] = h_sem_static
        return ret

    def forward_dynamic(
        self, x: torch.Tensor, *, v: torch.Tensor, ts: torch.Tensor, h_appear: torch.Tensor = None, 
        input_normalized: bool=True, with_rgb: bool=False, with_h_sem: bool=False, 
        ) -> Dict[str, torch.Tensor]:
        if not input_normalized:
            x = self.space.normalize_coords(x)
            ts = self.space.normalize_ts(ts)
        ret_dynamic = self.forward_h_dynamic(x, ts=ts, input_normalized=True)
        h_dynamic = ret_dynamic['h_dynamic']
        with torch.autocast(device_type='cuda', dtype=self.dtype):
            sigma_dynamic = self.density_activation(h_dynamic[..., 0])
            if with_rgb:
                rgb_dynamic = self.radiance_net(x=x, v=v, n=None, h_extra=h_dynamic[..., :self.n_geometry_feat], h_appear=h_appear)['rgb']
            if with_h_sem:
                h_sem_dynamic = h_dynamic[..., self.n_geometry_feat:]
        #---- Gather returns
        ret = dict(sigma=sigma_dynamic)
        if with_rgb:
            ret['rgb'] = rgb_dynamic
        if with_h_sem:
            ret['h_sem'] = h_sem_dynamic
        return ret

    @torch.no_grad()
    def query_density_static(self, x: torch.Tensor) -> torch.Tensor:
        ret_static = self.forward_h_static(x, input_normalized=True)
        h_static = ret_static['h_static']
        sigma_static = self.density_activation(h_static[..., 0])
        return sigma_static
    
    @torch.no_grad()
    def query_density_dynamic(self, x: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        ret_dynamic = self.forward_h_dynamic(x, ts=ts, input_normalized=True)
        h_dynamic = ret_dynamic['h_dynamic']
        sigma_dynamic = self.density_activation(h_dynamic[..., 0])
        return sigma_dynamic

class EmerNeRFModel(EmerNeRF):
    # Define renderer mixin methods
    """
    NeuS Renderer Mixin class
    
    NOTE: This is a mixin class!
    Refer: https://stackoverflow.com/questions/533631/what-is-a-mixin-and-why-are-they-useful
    !!!!: The target class for this mixin should also inherit from `ModelMixin`.
    """
    
    # NOTE: Configuration for common information usage in the forward process (and their defaults)
    use_view_dirs: bool = True # Determines if view_dirs are needed in forward process
    use_nablas: bool = False # Determines if nablas are required in forward process
    use_h_appear: bool = False # Determines if global per-frame appearance embeddings are necessary in forward process
    use_ts: bool = True # Determines if global timestamps are used in forward process
    use_pix: bool = False # Determines if pixel locations (in range [0,1]^2) are required in forward process
    use_fidx: bool = False # Determines if global frame indices are used in forward process
    use_bidx: bool = False # Determines if batch indices are used in forward process
    fwd_density_use_pix: bool = False
    fwd_density_use_h_appear: bool = False
    fwd_density_use_view_dirs: bool = False
    
    def __init__(
        self, *, 
        # Renderer mixin kwargs
        ray_query_cfg: dict = dict(), 
        accel_cfg: dict = None, 
        # Network kwargs
        **net_kwargs) -> None:
        super().__init__(**net_kwargs) # Will call network's __init__() (e.g. LoTDNeus ...)
        
        self.ray_query_cfg = ray_query_cfg
        self.accel_cfg = accel_cfg

    def populate(self, *args, **kwargs):
        super().populate(*args, **kwargs)
        
        # Acceleration struct
        self.accel: accel_types_dynamic_t = None if self.accel_cfg is None \
            else get_accel(space=self.space, device=self.device, **self.accel_cfg)
    
    def sample_pts_uniform(self, num_samples: int):
        # NOTE: Returns normalized `x`
        x, ts = self.space.sample_pts_uniform(num_samples)
        ret = self.forward_density(x, ts=ts, skip_accel=True, return_flow=True, return_seperate=True) # Do not upsate_samples here (usally there are too less samples here.)
        ret = {k: v.to(x.dtype) for k, v in ret.items()}
        ret['net_x'] = x # NOTE: in network's uniformed space; not in world space.
        return ret

    def sample_pts_in_occupied(self, num_samples: int):
        raise NotImplementedError

    def forward_density(self, x: torch.Tensor, ts: torch.Tensor, skip_accel=False, **kwargs):
        if self.training and (not skip_accel) and isinstance(self.accel, OccGridAccelStaticAndDynamic):
            kwargs.setdefault('return_seperate', True)
        ret = super().forward_density(x, ts=ts, **kwargs)
        if self.training and (not skip_accel) and (self.accel is not None):
            if isinstance(self.accel, OccGridAccelStaticAndDynamic):
                self.accel.collect_samples(x, ts=ts, val_static=ret['sigma_static'], val_dynamic=ret['sigma_dynamic'])
            else:
                self.accel.collect_samples(x, ts=ts, val=ret['sigma'].data)
        return ret

    def forward(self, x: torch.Tensor, ts: torch.Tensor, skip_accel=False, **kwargs):
        if self.training and (not skip_accel) and isinstance(self.accel, OccGridAccelStaticAndDynamic):
            kwargs.setdefault('return_seperate', True)
        ret = super().forward(x, ts=ts, **kwargs)
        if self.training and (not skip_accel) and (self.accel is not None):
            if isinstance(self.accel, OccGridAccelStaticAndDynamic):
                self.accel.collect_samples(x, ts=ts, val_static=ret['sigma_static'], val_dynamic=ret['sigma_dynamic'])
            else:
                self.accel.collect_samples(x, ts=ts, val=ret['sigma'].data)
        return ret

    def training_initialize(self, config=dict(), logger=None, log_prefix: str=None, skip_accel=False) -> bool:
        updated = super().training_initialize(config, logger=logger, log_prefix=log_prefix)
        if (not skip_accel) and (self.accel is not None):
            if isinstance(self.accel, OccGridAccelStaticAndDynamic):
                self.accel.init(self.query_density_static, self.query_density_dynamic, logger=logger)
            else:
                self.accel.init(self.query_density, logger=logger)
        return updated
    
    def training_before_per_step(self, cur_it: int, logger: Logger = None, skip_accel=False):
        self.it = cur_it
        super().training_before_per_step(cur_it, logger=logger)
        if self.training and (not skip_accel) and (self.accel is not None):
            if isinstance(self.accel, OccGridAccelStaticAndDynamic):
                self.accel.step(cur_it, self.query_density_static, self.query_density_dynamic, logger)
            else:
                self.accel.step(cur_it, self.query_density, logger)
    
    def training_after_per_step(self, cur_it: int, logger: Logger = None, skip_accel=False):
        super().training_after_per_step(cur_it, logger=logger)
    
    def ray_test(self, rays_o: torch.Tensor, rays_d: torch.Tensor, near=None, far=None, **extra_ray_data: Dict[str, torch.Tensor]):
        """ Test input rays' intersection with the model's space (AABB, Blocks, etc.)

        Args:
            rays_o (torch.Tensor): [num_total_rays,3] Input rays' origin
            rays_d (torch.Tensor): [num_total_rays,3] Input rays' direction
            near (Union[float, torch.Tensor], optional): [num_total_rays] tensor or float. Defaults to None.
            far (Union[float, torch.Tensor], optional): [num_total_rays] tensor or float. Defaults to None.
        Returns:
            dict: The ray_tested result. An example dict:
                num_rays:   int, Number of tested rays
                rays_inds:  [num_rays] tensor, ray indices in `num_total_rays` of the tested rays
                rays_o:     [num_rays, 3] tensor, the indexed and scaled rays' origin
                rays_d:     [num_rays, 3] tensor, the indexed and scaled rays' direction
                near:       [num_rays] tensor, entry depth of intersection
                far:        [num_rays] tensor, exit depth of intersection
        """
        assert (rays_o.dim() == 2) and (rays_d.dim() == 2), "Expect `rays_o` and `rays_d` to be of shape [N, 3]"
        for k, v in extra_ray_data.items():
            if isinstance(v, torch.Tensor):
                assert v.shape[0] == rays_o.shape[0], f"Expect `{k}` has the same shape prefix with rays_o {rays_o.shape[0]}, but got {v.shape[0]}"
        return self.space.ray_test(rays_o, rays_d, near=near, far=far, return_rays=True, **extra_ray_data)
    
    @profile
    def ray_query(
        self, 
        # ray query inputs
        ray_input: Dict[str, torch.Tensor]=None, ray_tested: Dict[str, torch.Tensor]=None, 
        # ray query function config
        config=dict(), 
        # function config
        return_buffer=False, return_details=False, render_per_obj_individual=False) -> dict:
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
        query_mode = config.query_mode
        with_rgb = config.with_rgb
        with_normal = config.with_normal
        with_flow = config.get('with_flow', False)
        with_static_dynamic = config.get('with_static_dynamic', False)
        forward_params = dict()
        if with_flow:
            forward_params['return_flow'] = True
        if with_static_dynamic:
            forward_params['return_seperate'] = True
        
        #----------------
        # Prepare outputs & compute outputs that are needed even when (num_rays==0)
        #----------------
        raw_ret = dict()
        if return_buffer:
            raw_ret['volume_buffer'] = dict(type='empty', rays_inds_hit=[])
        if return_details:
            details = raw_ret['details'] = {}
            if (self.accel is not None) and hasattr(self.accel, 'debug_stats'):
                details['accel'] = self.accel.debug_stats()
            if hasattr(self, 'radiance_net') and hasattr(self.radiance_net, 'blocks') \
                and hasattr(self.radiance_net.blocks, 'lipshitz_bound_full'):
                details['radiance.lipshitz_bound'] = self.radiance_net.blocks.lipshitz_bound_full().item()
                
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
            if with_flow:
                rendered['flow_fwd'] = torch.zeros([*prefix_rays, 3], dtype=dtype, device=device)
                rendered['flow_bwd'] = torch.zeros([*prefix_rays, 3], dtype=dtype, device=device)
                # rendered['flow_bwd_pred_fwd'] = torch.zeros([*prefix_rays, 3], dtype=dtype, device=device)
                # rendered['flow_fwd_pred_bwd'] = torch.zeros([*prefix_rays, 3], dtype=dtype, device=device)
            if with_static_dynamic:
                rendered['mask_static'] = torch.zeros([*prefix_rays], dtype=dtype, device=device)
                rendered['mask_dynamic'] = torch.zeros([*prefix_rays], dtype=dtype, device=device)
                rendered['depth_static'] = torch.zeros([*prefix_rays], dtype=dtype, device=device)
                rendered['depth_dynamic'] = torch.zeros([*prefix_rays], dtype=dtype, device=device)
                if with_rgb:
                    rendered['rgb_static'] = torch.zeros([*prefix_rays, 3], dtype=dtype, device=device)
                    rendered['rgb_dynamic'] = torch.zeros([*prefix_rays, 3], dtype=dtype, device=device)
                if with_normal:
                    rendered['normals_static'] = torch.zeros([*prefix_rays, 3], dtype=dtype, device=device)
                    rendered['normals_dynamic'] = torch.zeros([*prefix_rays, 3], dtype=dtype, device=device)

        if ray_tested['num_rays'] == 0:
            return raw_ret
        
        #----------------
        # Ray query
        #----------------
        if query_mode == 'march_occ':
            volume_buffer, query_details = nerf_ray_query_march_occ(
                self, ray_tested, 
                with_rgb=with_rgb, # with_normal=with_normal, 
                perturb=config.perturb, 
                forward_params=forward_params, 
                **config.query_param)
        elif query_mode == 'march_occ_multi_upsample_compressed':
            volume_buffer, query_details = nerf_ray_query_march_occ_multi_upsample_compressed(
                self, ray_tested, 
                with_rgb=with_rgb, # with_normal=with_normal, 
                perturb=config.perturb, 
                forward_params=forward_params, 
                **config.query_param)
        else:
            raise RuntimeError(f"Invalid query_mode={query_mode}")

        if with_static_dynamic:
            if (buffer_type:=volume_buffer['type']) != 'empty':
                if 'opacity_alpha_static' not in volume_buffer:
                    volume_buffer['opacity_alpha_static'] = tau_to_alpha(volume_buffer['sigma_static'] * volume_buffer['deltas'])
                if 'opacity_alpha_dynamic' not in volume_buffer:
                    volume_buffer['opacity_alpha_dynamic'] = tau_to_alpha(volume_buffer['sigma_dynamic'] * volume_buffer['deltas'])

        #----------------
        # Calc outputs
        #----------------
        if return_buffer:
            raw_ret['volume_buffer'] = volume_buffer
        
        if return_details:
            details.update(query_details)
        
        if render_per_obj_individual:
            with profile("Render per-object"):
                if (buffer_type:=volume_buffer['type']) != 'empty':
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
                        if with_normal and ('nablas' in volume_buffer):
                            # rendered['normals_volume'][rays_inds_hit] = (vw.unsqueeze(-1) * volume_buffer['nablas']).sum(dim=-2)
                            if self.training:
                                rendered['normals_volume'][rays_inds_hit] = (vw.unsqueeze(-1) * volume_buffer['nablas']).sum(dim=-2)
                            else:
                                rendered['normals_volume'][rays_inds_hit] = (vw.unsqueeze(-1) * F.normalize(volume_buffer['nablas'].clamp_(-1,1), dim=-1)).sum(dim=-2)
                        # if with_flow:
                        #     for k in ['flow_fwd', 'flow_fwd_pred_bwd', 'flow_bwd', 'flow_bwd_pred_fwd']:
                        #         if k in volume_buffer:
                        #             rendered[k][rays_inds_hit] = (vw.unsqueeze(-1) * volume_buffer[k]).sum(dim=-2)

                        if with_static_dynamic:
                            vw_static = ray_alpha_to_vw(volume_buffer['opacity_alpha_static'])
                            vw_dynamic = ray_alpha_to_vw(volume_buffer['opacity_alpha_dynamic'])
                            rendered['mask_static'][rays_inds_hit] = vw_sum_static = vw_static.sum(dim=-1)
                            rendered['mask_dynamic'][rays_inds_hit] = vw_sum_dynamic = vw_dynamic.sum(dim=-1)
                            if depth_use_normalized_vw:
                                vw_normalized_static = vw_static / (vw_sum_static.unsqueeze(-1)+1e-10)
                                vw_normalized_dynamic = vw_dynamic / (vw_sum_dynamic.unsqueeze(-1)+1e-10)
                                rendered['depth_static'][rays_inds_hit] = (vw_normalized_static * volume_buffer['t']).sum(dim=-1)
                                rendered['depth_dynamic'][rays_inds_hit] = (vw_normalized_dynamic * volume_buffer['t']).sum(dim=-1)
                            else:
                                rendered['depth_static'][rays_inds_hit] = (vw_static * volume_buffer['t']).sum(dim=-1)
                                rendered['depth_dynamic'][rays_inds_hit] = (vw_dynamic * volume_buffer['t']).sum(dim=-1)
                            if with_rgb:
                                rendered['rgb_static'][rays_inds_hit] = (vw_static.unsqueeze(-1) * volume_buffer['rgb_static']).sum(dim=-2)
                                rendered['rgb_dynamic'][rays_inds_hit] = (vw_dynamic.unsqueeze(-1) * volume_buffer['rgb_dynamic']).sum(dim=-2)
                            if with_normal:
                                if 'nablas_static' in volume_buffer:
                                    if self.training:
                                        rendered['normals_static'][rays_inds_hit] = (vw_static.unsqueeze(-1) * volume_buffer['nablas_static']).sum(dim=-2)
                                    else:
                                        rendered['normals_static'][rays_inds_hit] = (vw_static.unsqueeze(-1) * F.normalize(volume_buffer['nablas_static'].clamp_(-1,1), dim=-1)).sum(dim=-2)
                                if 'nablas_dynamic' in volume_buffer:
                                    if self.training:
                                        rendered['normals_dynamic'][rays_inds_hit] = (vw_dynamic.unsqueeze(-1) * volume_buffer['nablas_dynamic']).sum(dim=-2)
                                    else:
                                        rendered['normals_dynamic'][rays_inds_hit] = (vw_dynamic.unsqueeze(-1) * F.normalize(volume_buffer['nablas_dynamic'].clamp_(-1,1), dim=-1)).sum(dim=-2)
                            if with_flow:
                                for k in ['flow_fwd', 'flow_fwd_pred_bwd', 'flow_bwd', 'flow_bwd_pred_fwd']:
                                    if k in volume_buffer: # NOTE: Use `vw_dynamic` to render flow instead. (since flows are only valid for dynamic)
                                        rendered[k][rays_inds_hit] = (vw_dynamic.unsqueeze(-1) * volume_buffer[k]).sum(dim=-2)

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
                        if with_normal and ('nablas' in volume_buffer):
                            # rendered['normals_volume'][rays_inds_hit] = packed_sum(vw.view(-1,1) * volume_buffer['nablas'].view(-1,3), pack_infos_hit)
                            if self.training:
                                rendered['normals_volume'][rays_inds_hit] = packed_sum(vw.view(-1,1) * volume_buffer['nablas'].view(-1,3), pack_infos_hit)
                            else:
                                rendered['normals_volume'][rays_inds_hit] = packed_sum(vw.view(-1,1) * F.normalize(volume_buffer['nablas'].clamp_(-1,1), dim=-1).view(-1,3), pack_infos_hit)
                        # if with_flow:
                        #     for k in ['flow_fwd', 'flow_fwd_pred_bwd', 'flow_bwd', 'flow_bwd_pred_fwd']:
                        #         if k in volume_buffer:
                        #             rendered[k][rays_inds_hit] = packed_sum(vw.view(-1,1) * volume_buffer[k].view(-1,3), pack_infos_hit)
                        
                        if with_static_dynamic:
                            vw_static = packed_alpha_to_vw(volume_buffer['opacity_alpha_static'], pack_infos_hit)
                            vw_dynamic = packed_alpha_to_vw(volume_buffer['opacity_alpha_dynamic'], pack_infos_hit)
                            rendered['mask_static'][rays_inds_hit] = vw_sum_static = packed_sum(vw_static.view(-1), pack_infos_hit)
                            rendered['mask_dynamic'][rays_inds_hit] = vw_sum_dynamic = packed_sum(vw_dynamic.view(-1), pack_infos_hit)
                            if depth_use_normalized_vw:
                                vw_normalized_static = packed_div(vw_static, vw_sum_static + 1e-10, pack_infos_hit)
                                vw_normalized_dynamic = packed_div(vw_dynamic, vw_sum_dynamic + 1e-10, pack_infos_hit)
                                rendered['depth_static'][rays_inds_hit] = packed_sum(vw_normalized_static * volume_buffer['t'].view(-1), pack_infos_hit)
                                rendered['depth_dynamic'][rays_inds_hit] = packed_sum(vw_normalized_dynamic * volume_buffer['t'].view(-1), pack_infos_hit)
                            else:
                                rendered['depth_static'][rays_inds_hit] = packed_sum(vw_static.view(-1) * volume_buffer['t'].view(-1), pack_infos_hit)
                                rendered['depth_dynamic'][rays_inds_hit] = packed_sum(vw_dynamic.view(-1) * volume_buffer['t'].view(-1), pack_infos_hit)
                            if with_rgb:
                                rendered['rgb_static'][rays_inds_hit] = packed_sum(vw_static.view(-1,1) * volume_buffer['rgb_static'].view(-1,3), pack_infos_hit)
                                rendered['rgb_dynamic'][rays_inds_hit] = packed_sum(vw_dynamic.view(-1,1) * volume_buffer['rgb_dynamic'].view(-1,3), pack_infos_hit)
                            if with_normal:
                                if 'nablas_static' in volume_buffer:
                                    if self.training:
                                        rendered['normals_static'][rays_inds_hit] = packed_sum(vw_static.view(-1,1) * volume_buffer['nablas_static'].view(-1,3), pack_infos_hit)
                                    else:
                                        rendered['normals_static'][rays_inds_hit] = packed_sum(vw_static.view(-1,1) * F.normalize(volume_buffer['nablas_static'].clamp_(-1,1), dim=-1).view(-1,3), pack_infos_hit)
                                if 'nablas_dynamic' in volume_buffer:
                                    if self.training:
                                        rendered['normals_dynamic'][rays_inds_hit] = packed_sum(vw_dynamic.view(-1,1) * volume_buffer['nablas_dynamic'].view(-1,3), pack_infos_hit)
                                    else:
                                        rendered['normals_dynamic'][rays_inds_hit] = packed_sum(vw_dynamic.view(-1,1) * F.normalize(volume_buffer['nablas_dynamic'].clamp_(-1,1), dim=-1).view(-1,3), pack_infos_hit)
                            if with_flow:
                                for k in ['flow_fwd', 'flow_fwd_pred_bwd', 'flow_bwd', 'flow_bwd_pred_fwd']:
                                    if k in volume_buffer: # NOTE: Use `vw_dynamic` to render flow instead. (since flows are only valid for dynamic)
                                        rendered[k][rays_inds_hit] = packed_sum(vw_dynamic.view(-1,1) * volume_buffer[k].view(-1,3), pack_infos_hit)

        return raw_ret

    def ray_query_static(self):
        # TODO: Consider combining this with the regular renderer_mixin, maybe create a subclass within the class?
        # TODO: Consider using self.static_model = build_static_model()? But be careful not to make the model a submodule within nn.Module
        static_model = object()
        static_model.use_ts = False
        static_model.use_fidx = False
        static_model.use_bidx = False
        static_model.use_pix = self.use_pix
        static_model.use_h_appear = self.use_h_appear
        static_model.use_view_dirs = self.use_view_dirs
        static_model.query_density = self.query_density_static
        static_model.forward_density = functools.partial(self.forward_static, with_rgb=False, with_h_sem=False)
        static_model.forward = self.forward_static
        if self.accel is not None:
            static_model.accel = self.accel.get_accel_static()
        # TODO
    
    def ray_query_dynamic(self):
        dynamic_model = object()
        dynamic_model.use_ts = True
        dynamic_model.use_fidx = False
        dynamic_model.use_bidx = False
        dynamic_model.use_pix = self.use_pix
        dynamic_model.use_h_appear = self.use_h_appear
        dynamic_model.use_view_dirs = self.use_view_dirs
        dynamic_model.query_density = self.query_density_dynamic
        dynamic_model.forward_density = functools.partial(self.forward_dynamic, with_rgb=False, with_h_sem=False)
        dynamic_model.forward = self.forward_dynamic
        if self.accel is not None:
            dynamic_model.accel = self.accel.get_accel_dynamic()
        # TODO

class EmerNeRFOnlyDynamic(ModelMixin, nn.Module):
    def __init__(
        self, 
        n_geometry_feat: int = 64, 
        n_semantic_feat: int = 0, 
        
        time_embedding_cfg: dict = dict(dim=1, learnable=False, 
                                  weight_init=dict(type='linspace', start=-1., end=1.)), 
        
        dynamic_encoding_cfg: dict = dict(type='permuto', param=...), 
        dynamic_decoder_cfg: dict = dict(type='mlp', D=1, W=64), 
        
        use_flow_field: bool = True, 
        use_flow_in_obj: bool = True, 
        flow_encoding_cfg: dict = dict(type='permuto', param=...), 
        flow_decoder_cfg: dict = dict(type='mlp', D=2, W=64), 
        
        use_shadow: bool = True, 
        shadow_cfg: dict = dict(D=1, W=64), 
        
        density_activation: Union[str, Callable] = lambda x: trunc_exp(x-1), 
        radiance_cfg: dict = dict(D=2, W=64, 
                                  dir_embed_cfg=dict(type='spherical', degree=4)), 
        
        dtype=torch.half, device=None, 
        ) -> None:
        super().__init__()
        
        self.dtype = torch_dtype(dtype)
        self.set_device = device
        
        self.space: AABBDynamicSpace = None
        
        self.n_geometry_feat = n_geometry_feat
        self.n_semantic_feat = n_semantic_feat
        
        self.time_embedding_cfg = time_embedding_cfg
        
        self.dynamic_encoding_cfg = dynamic_encoding_cfg
        self.dynamic_decoder_cfg = dynamic_decoder_cfg
        
        self.use_shadow = use_shadow
        self.shadow_cfg = shadow_cfg
        
        self.use_flow_field = use_flow_field
        self.use_flow_in_obj = use_flow_in_obj
        self.flow_encoding_cfg = flow_encoding_cfg
        self.flow_decoder_cfg = flow_decoder_cfg
        
        self.radiance_cfg = radiance_cfg
        self.density_activation = density_activation

    @property
    def device(self) -> torch.device:
        return self.space.device
    
    def populate(
        self, 
        ts_keyframes: Union[np.ndarray, torch.Tensor], 
        bounding_size=None, aabb=None, device=None):
        
        device = device or self.set_device
        self.set_device = device
        
        #----------------------------------------
        #---- Valid representing space
        #----------------------------------------
        self.space = AABBDynamicSpace(bounding_size=bounding_size, aabb=aabb, dtype=torch.float, device=device)

        #----------------------------------------
        #---- Time embedding
        #----------------------------------------
        ts_keyframes = check_to_torch(ts_keyframes, dtype=torch.float, device=device)
        # NOTE: Use normalized `ts`
        ts_keyframes = self.space.normalize_ts(ts_keyframes)
        dt = (ts_keyframes[-1] - ts_keyframes[0]).item() / (len(ts_keyframes) - 1)
        self.dt: float = dt # Save `dt` for using forward & backward flow to aggregate features
        self.time_embedding = SeqEmbedding(ts_keyframes, **self.time_embedding_cfg, dtype=torch.float, device=device)
        x_scale = (self.space.radius3d / self.space.radius3d.min()).tolist()
        permuto_input_scale = check_to_torch([*x_scale, *[1]*self.time_embedding.embedding_dim], dtype=torch.float, device=device)

        #----------------------------------------
        #---- Dynamic field
        #----------------------------------------
        dynamic_encoding_cfg = deepcopy(self.dynamic_encoding_cfg)
        if dynamic_encoding_cfg['type'].lower() == 'lotd':
            self.dynamic_encoding = LoTDEncoding(
                3+self.time_embedding.embedding_dim, space=self.space, 
                **dynamic_encoding_cfg['param'], dtype=self.dtype, device=device)
        elif dynamic_encoding_cfg['type'].lower() == 'permuto':
            pos_scale = permuto_input_scale * dynamic_encoding_cfg['param'].pop('pos_scale', 1.0)
            self.dynamic_encoding = PermutoEncoding(
                3+self.time_embedding.embedding_dim, 
                space=self.space, pos_scale=pos_scale, 
                **dynamic_encoding_cfg['param'], dtype=self.dtype, device=device)
        else:
            raise RuntimeError(f"Invalid dynamic_encoding_cfg['type']={dynamic_encoding_cfg['type']}")
        
        self.dynamic_decoder, self.dynamic_decoder_type = get_multires_decoder(
            self.dynamic_encoding.level_n_feats, 
            self.n_geometry_feat + self.n_semantic_feat, 
            **self.dynamic_decoder_cfg, 
            dtype=self.dtype, device=device
        )

        #----------------------------------------
        #---- Flow field
        #----------------------------------------
        if self.use_flow_field:
            self.h_dynamic_3t_weights = [0.25, 0.5, 0.25]
            flow_encoding_cfg = deepcopy(self.flow_encoding_cfg)
            if flow_encoding_cfg['type'].lower() == 'lotd':
                self.flow_encoding = LoTDEncoding(
                    3+self.time_embedding.embedding_dim, space=self.space, 
                    **flow_encoding_cfg['param'], dtype=self.dtype, device=device)
            elif flow_encoding_cfg['type'].lower() == 'permuto':
                pos_scale = permuto_input_scale * flow_encoding_cfg['param'].pop('pos_scale', 1.0)
                self.flow_encoding = PermutoEncoding(
                    3+self.time_embedding.embedding_dim, 
                    space=self.space, pos_scale=pos_scale, 
                    **flow_encoding_cfg['param'], dtype=self.dtype, device=device)
            else:
                raise RuntimeError(f"Invalid flow_encoding_cfg['type']={flow_encoding_cfg['type']}")
            
            self.flow_decoder, self.flow_decoder_type = get_multires_decoder(
                self.flow_encoding.level_n_feats, 
                6, # Outputs forward displacement (3) + backward displacement (3)
                **self.flow_decoder_cfg, 
                dtype=self.dtype, device=device
            )

        #----------------------------------------
        #---- Shadow head
        #----------------------------------------
        if self.use_shadow:
            self.dynamic_shadow_net = get_blocks(
                self.n_geometry_feat, 
                1, 
                **self.shadow_cfg, 
                output_activation='sigmoid', 
                dtype=self.dtype, device=device
            )

        #----------------------------------------
        #---- Density head
        #----------------------------------------
        if isinstance(self.density_activation, (str, dict)):
            self.density_activation = get_nonlinearity(self.density_activation).nl
        
        #----------------------------------------
        #---- Color head 
        #----------------------------------------
        self.radiance_net = RadianceNet(
            # use_pos=False, 
            # use_nablas=False, 
            # use_view_dirs=True, 
            n_extra_feat=self.n_geometry_feat, 
            **self.radiance_cfg, 
            dtype=self.dtype, device=device
        )
        self.use_view_dirs = self.radiance_net.use_view_dirs
        self.use_h_appear = self.radiance_net.use_h_appear

    def forward_h_dynamic(
        self, x: torch.Tensor, *, ts: torch.Tensor, input_normalized=True
        ) -> Dict[str, torch.Tensor]:
        
        if not input_normalized:
            x = self.space.normalize_coords(x)
            ts = self.space.normalize_ts(ts)
        
        z_time = self.time_embedding(ts, mode='interp')
        input = torch.cat((x, z_time), dim=-1)
        
        with torch.autocast(device_type='cuda', dtype=self.dtype):
            if self.use_flow_field:
                # Get prev, next flow
                flow_output = self.flow_decoder(self.flow_encoding(input))
                flow_fwd, flow_bwd = flow_output[..., :3], flow_output[..., 3:]
                
                # Get warped space-time
                if self.training:
                    noise_ratio = 1.5 * torch.rand_like(ts)
                    z_time_bwd = self.time_embedding(ts - self.dt * noise_ratio, mode='interp')
                    z_time_fwd = self.time_embedding(ts + self.dt * noise_ratio, mode='interp')
                    if self.use_flow_in_obj:
                        x_bwd = x + flow_bwd / self.space.radius3d * noise_ratio.unsqueeze(-1)
                        x_fwd = x + flow_fwd / self.space.radius3d * noise_ratio.unsqueeze(-1)
                    else:
                        x_bwd = x + flow_bwd * noise_ratio.unsqueeze(-1)
                        x_fwd = x + flow_fwd * noise_ratio.unsqueeze(-1)
                else:
                    z_time_bwd = self.time_embedding(ts - self.dt, mode='interp')
                    z_time_fwd = self.time_embedding(ts + self.dt, mode='interp')
                    if self.use_flow_in_obj:
                        x_bwd = x + flow_bwd / self.space.radius3d
                        x_fwd = x + flow_fwd / self.space.radius3d
                    else:
                        x_bwd = x + flow_bwd
                        x_fwd = x + flow_fwd

                input_fwd = torch.cat((x_fwd, z_time_fwd), dim=-1)
                input_bwd = torch.cat((x_bwd, z_time_bwd), dim=-1)
                
                # Forward dynamic field on prev,now,next
                input_3t = torch.stack((input_bwd, input, input_fwd), dim=0)
                enc_dynamic_3t = self.dynamic_encoding(input_3t)
                h_dynamic_3t = self.dynamic_decoder(enc_dynamic_3t)
                
                # Aggregate 3 frames
                w = self.h_dynamic_3t_weights
                h_dynamic = w[0] * h_dynamic_3t[0] + w[1] * h_dynamic_3t[1] + w[2] * h_dynamic_3t[2]
                
                if self.training and torch.is_grad_enabled():
                    # For consistency
                    fwd_pred_flow = self.flow_decoder(self.flow_encoding(input_fwd))
                    bwd_pred_flow = self.flow_decoder(self.flow_encoding(input_bwd))
                else:
                    fwd_pred_flow = bwd_pred_flow = None
            else:
                # Only use the features of the current `ts`
                enc_dynamic = self.dynamic_encoding(input)
                h_dynamic = self.dynamic_decoder(enc_dynamic)
        
        #---- Gather returns
        ret = dict(h_dynamic=h_dynamic)
        if self.use_flow_field:
            ret.update(flow_fwd=flow_fwd, flow_bwd=flow_bwd)
            if fwd_pred_flow is not None:
                ret.update(
                    flow_fwd_pred_bwd=fwd_pred_flow[..., 3:], 
                    flow_bwd_pred_fwd=bwd_pred_flow[..., :3], 
                )
        return ret
    
    def forward(
        self, 
        x: torch.Tensor, *, 
        v: torch.Tensor, 
        ts: torch.Tensor, 
        h_appear: torch.Tensor = None, 
        input_normalized: bool=True, 
        with_rgb=True, with_h_sem=False, 
        return_flow: bool=True, 
        return_details: bool=False
        ) -> Dict[str, torch.Tensor]:
        
        if not input_normalized:
            x = self.space.normalize_coords(x)
            ts = self.space.normalize_ts(ts)
        
        ret_dynamic = self.forward_h_dynamic(x, ts=ts, input_normalized=True)
        h_dynamic = ret_dynamic['h_dynamic']

        with torch.autocast(device_type='cuda', dtype=self.dtype):
            sigma_dynamic = self.density_activation(h_dynamic[..., 0])
            if with_rgb:
                rgb_dynamic = self.radiance_net(x=x, v=v, n=None, h_extra=h_dynamic[..., :self.n_geometry_feat], h_appear=h_appear)['rgb']
            if with_h_sem:
                h_sem_dynamic = h_dynamic[..., self.n_geometry_feat:]
            if self.use_shadow:
                ratio_shadow = self.dynamic_shadow_net(h_dynamic).squeeze(-1)

        #---- Gather returns
        ret = dict(sigma=sigma_dynamic)
        if with_rgb:
            ret.update(rgb=rgb_dynamic)
        if with_h_sem:
            ret.update(h_sem=h_sem_dynamic)
        if self.use_shadow:
            ret.update(ratio_shadow=ratio_shadow)
        if return_flow:
            for k in ['flow_fwd', 'flow_fwd_pred_bwd', 'flow_bwd', 'flow_bwd_pred_fwd']:
                if k in ret_dynamic:
                    ret[k] = ret_dynamic[k]
        if return_details:
            ret.update(ret_dynamic)
        return ret

    def forward_density(
        self, 
        x: torch.Tensor, *, 
        ts: torch.Tensor, 
        input_normalized: bool=True, 
        return_flow: bool=True, 
        return_details: bool=False, 
        ) -> Dict[str, torch.Tensor]:
        
        if not input_normalized:
            x = self.space.normalize_coords(x)
            ts = self.space.normalize_ts(ts)
        
        ret_dynamic = self.forward_h_dynamic(x, ts=ts, input_normalized=True)
        h_dynamic = ret_dynamic['h_dynamic']

        with torch.autocast(device_type='cuda', dtype=self.dtype):
            sigma_dynamic = self.density_activation(h_dynamic[..., 0])
        
        #---- Gather returns
        ret = dict(sigma=sigma_dynamic)
        if return_flow:
            for k in ['flow_fwd', 'flow_bwd', 'flow_fwd_pred_bwd', 'flow_bwd_pred_fwd']:
                if k in ret_dynamic:
                    ret[k] = ret_dynamic[k]
        if return_details:
            ret.update(ret_dynamic)
        return ret
    
    @torch.no_grad()
    def query_density(self, x: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        h_dynamic = self.forward_h_dynamic(x, ts=ts, input_normalized=True)['h_dynamic']
        sigma_dynamic = self.density_activation(h_dynamic[..., 0])
        return sigma_dynamic

    def get_weight_reg(self, norm_type: float = 2.0) -> torch.Tensor:
        items = [
            self.dynamic_decoder.get_weight_reg(norm_type).to(self.device), 
            self.radiance_net.get_weight_reg(norm_type).to(self.device), 
        ]
        if self.use_flow_field:
            items.append(self.flow_decoder.get_weight_reg(norm_type).to(self.device))
        if self.use_shadow:
            items.append(self.dynamic_shadow_net.get_weight_reg(norm_type).to(self.device))
        return torch.cat(items)

class EmerNeRFOnlyDynamicModel(NerfRendererMixinDynamic, EmerNeRFOnlyDynamic):
    pass

if __name__ == "__main__":
    def unit_test():
        device = torch.device('cuda')
        m = EmerNeRF(
            n_geometry_feat=64, 
            n_semantic_feat=0, 
            static_encoding_cfg=dict(
                type='lotd', 
                param=dict(
                    lotd_use_cuboid=True, 
                    lotd_auto_compute_cfg=dict(
                        type='ngp', 
                        target_num_params=32*(2**20), 
                        min_res=16, 
                        n_feats=2, 
                        max_num_levels=None, 
                        log2_hashmap_size=20
                    ), 
                    param_init_cfg=dict(type='uniform_to_type', bound=1.0e-4), 
                    anneal_cfg=dict(type='hardmask', start_it=0, stop_it=4000, start_level=2)
                )
            ), 
            static_decoder_cfg=dict(type='mlp', D=1, W=64, activation='relu'), 
            time_embedding_cfg=dict(
                dim=1, learnable=False, 
                weight_init=dict(type='linspace', start=-10., end=10.)), 
            dynamic_encoding_cfg=dict(
                type='permuto', 
                param=dict(
                    permuto_auto_compute_cfg=dict(
                        type='multi_res', 
                        coarsest_res=16.0, 
                        finest_res=2000.0, 
                        n_levels=16, 
                        n_feats=2, 
                        log2_hashmap_size=18, 
                        apply_random_shifts_per_level=True
                    )
                )
            ), 
            dynamic_decoder_cfg=dict(type='mlp', D=1, W=64), 
            use_flow_field=True, 
            flow_encoding_cfg=dict(
                type='permuto', 
                param=dict(
                    permuto_auto_compute_cfg=dict(
                        type='multi_res', 
                        coarsest_res=16.0, 
                        finest_res=2000.0, 
                        n_levels=16, 
                        n_feats=2, 
                        log2_hashmap_size=18, 
                        apply_random_shifts_per_level=True
                    )
                )
            ), 
            flow_decoder_cfg=dict(type='mlp', D=2, W=64), 
            use_shadow=True, 
            shadow_cfg=dict(D=1, W=64), 
            radiance_cfg=dict(D=2, W=64, dir_embed_cfg=dict(type='spherical', degree=4)), 
            dtype=torch.half, 
            device=device
        )
        m.populate(
            ts_keyframes=torch.linspace(-1,1,20, dtype=torch.float,device=device), 
            aabb=[[-1,-10,-1],[1,10,1]])
        
        x = torch.randn((123,3), dtype=torch.float, device=device)
        v = F.normalize(torch.randn((123,3), dtype=torch.float, device=device), dim=-1)
        ts = torch.rand((123,), dtype=torch.float, device=device)*2-1
        
        m.forward(x=x, v=v, ts=ts, with_rgb=True, return_details=True, return_seperate=True)
    
    unit_test()

"""
@file   permuto_sdf.py
@author Jianfei Guo, Shanghai AI Lab
@brief  SDF network characterized using the Permutohedral-encoding model.
"""

__all__ = [
    'PermutoSDF'
]

import numpy as np
from numbers import Number
from typing import Dict, Literal, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd

from nr3d_lib.logger import Logger
from nr3d_lib.profile import profile
from nr3d_lib.utils import tensor_statistics, torch_dtype

from nr3d_lib.models.utils import clip_norm_
from nr3d_lib.models.model_base import ModelMixin
from nr3d_lib.models.embedders import get_embedder
from nr3d_lib.models.grid_encodings.permuto import PermutoEncoding
from nr3d_lib.models.grid_encodings.multires_decoder import get_multires_decoder
from nr3d_lib.models.fields.sdf.utils import idr_geometric_init, pretrain_sdf_sphere

class PermutoSDF(ModelMixin, nn.Module):
    def __init__(
        self,
        n_input_dim: int = 3, 
        encoding_cfg: dict = ...,
        decoder_cfg: dict = ...,
        n_extra_feat_from_output: int = 0,
        sdf_scale: float = 1.0,
        
        extra_pos_embed_cfg: dict = None,
        bounding_size=2.0, aabb=None, radius_init=0.5, 
        geo_init_method: str = None, inside_out=False, 
        
        clip_level_grad_ema_factor: float=0, 

        dtype=torch.half, device=None, use_tcnn_backend=False, 
        ) -> None:
        """_summary_

        Args:
            n_input_dim (int, optional): _description_. Defaults to 3.
            encoding_cfg (dict, optional): _description_. Defaults to ....
            decoder_cfg (dict, optional): _description_. Defaults to ....
            n_extra_feat_from_output (int, optional): \
                Determines whether downstream networks (e.g., RGB decoders or semantic decoders) \
                use the direct encoding output or extra output channels from the decoder. \
                If set to 0: downstream networks use the encoded features directly. \
                    The SDF decoder outputs 1-channel (SDF).
                If set to >0: downstream networks use the additional output from the SDF decoder (output[.., 1:]). \
                    The SDF decoder outputs (1+`n_extra_feat_from_output`)-channel (SDF and feature). \
                Default is 0.
            sdf_scale (float, optional): _description_. Defaults to 1.0.
            extra_pos_embed_cfg (dict, optional): _description_. Defaults to None.
            bounding_size (float, optional): _description_. Defaults to 2.0.
            aabb (_type_, optional): _description_. Defaults to None.
            radius_init (float, optional): _description_. Defaults to 0.5.
            geo_init_method (str, optional): _description_. Defaults to None.
            inside_out (bool, optional): _description_. Defaults to False.
            clip_level_grad_ema_factor (float, optional): _description_. Defaults to 0.
            dtype (_type_, optional): _description_. Defaults to torch.half.
            device (_type_, optional): _description_. Defaults to None.
            use_tcnn_backend (bool, optional): _description_. Defaults to False.
        """
        
        super().__init__()
        self.dtype = torch_dtype(dtype)
        
        self.n_input_dim = n_input_dim
        self.encoding_cfg = encoding_cfg
        self.decoder_cfg = decoder_cfg
        self.use_tcnn_backend = use_tcnn_backend
        
        self.sdf_scale = sdf_scale
        self.use_extra_embed = extra_pos_embed_cfg is not None
        self.radius_init = radius_init
        self.inside_out = inside_out
        self.geo_init_method = geo_init_method
        self.n_extra_feat_from_output = n_extra_feat_from_output
        self.is_extra_feat_from_output = self.n_extra_feat_from_output > 0
        self.clip_level_grad_ema_factor = clip_level_grad_ema_factor
        
        #------- Permutohedral encoding
        if self.clip_level_grad_ema_factor > 0:
            self.encoding_cfg.setdefault('clip_level_grad_ema_factor', self.clip_level_grad_ema_factor)
        space_cfg = self.encoding_cfg.setdefault('space_cfg', {'type': 'aabb'})
        if aabb is not None:
            space_cfg.update(aabb=aabb)
        if bounding_size is not None:
            space_cfg.update(bounding_size=bounding_size)
        self.encoding_cfg.update(space_cfg=space_cfg)
        self.encoding = PermutoEncoding(self.n_input_dim, **self.encoding_cfg, dtype=self.dtype, device=device)

        #------- (Optional) extra coords embedding
        if self.use_extra_embed:
            extra_pos_embed_cfg.setdefault('use_tcnn_backend', use_tcnn_backend)
            self.extra_embed_fn, self.n_extra_embed = get_embedder(extra_pos_embed_cfg, self.n_input_dim)
        else:
            self.n_extra_embed = 0
        
        #------- SDF decoder
        self.decoder_cfg.setdefault('use_tcnn_backend', self.use_tcnn_backend)
        if self.is_extra_feat_from_output:
            self.n_extra_feat = self.n_extra_feat_from_output
            self.decoder, self.decoder_type = get_multires_decoder(self.encoding.meta.level_n_feats, (1+self.n_extra_feat_from_output), n_extra_embed_ch=self.n_extra_embed, **self.decoder_cfg, dtype=self.dtype, device=device)
        else:
            self.n_extra_feat = self.encoding.out_features
            self.decoder, self.decoder_type = get_multires_decoder(self.encoding.meta.level_n_feats, 1, n_extra_embed_ch=self.n_extra_embed ,**self.decoder_cfg, dtype=self.dtype, device=device)
        if self.clip_level_grad_ema_factor > 0 and (length:=len(list(self.decoder.parameters()))) > 0:
            self.register_buffer(f'decoder_grad_ema', torch.ones([length], device=device, dtype=torch.float))
        
        if self.geo_init_method == 'geometric' or self.geo_init_method == 'pretrain_after_geometric':
            # NOTE: Only work for MLP decoders & at least an identity extra pos embed.
            assert self.use_extra_embed, "Geometric init only works with sinusoidal/identity extra embedder"
            emb_tp = self.extra_embed_fn._embedder_type
            assert ('sinusoidal' in emb_tp) or (emb_tp == 'identity'), "Geometric init only works with sinusoidal/identity extra embedder"
            # Finally get to work!
            idr_geometric_init(self.decoder, n_embed=self.n_extra_embed, radius_init=self.radius_init, inside_out=self.inside_out)
        
        elif self.geo_init_method == 'zero_out' or self.geo_init_method == 'pretrain_after_zero_out':
            # NOTE: For multires-annealing, set zero to non-active part of decoder input at start
            if self.encoding.annealer is not None:
                start_level = self.encoding.annealer.start_level
                start_n_feats = sum(self.encoding.meta.level_n_feats[:start_level+1])
            else:
                start_level = 0
                start_n_feats = 0
            with torch.no_grad():
                # for l in range(start_level, self.encoding.meta.n_levels, 1):
                #     self.encoding.get_level_param(l).zero_()
                if isinstance(self.decoder, nn.Sequential):
                    # [MultiresSelect, Decoder]
                    nn.init.zeros_(self.decoder[1].layers[0].weight[:, start_n_feats:])
                else:
                    nn.init.zeros_(self.decoder.layers[0].weight[:, start_n_feats:])
        
        self.register_buffer('is_pretrained', torch.tensor([False], dtype=torch.bool), persistent=True)

    @property
    def device(self) -> torch.device:
        return self.encoding.device

    @property
    def space(self):
        return self.encoding.space
    
    def training_before_per_step(self, cur_it: int, logger: Logger = None):
        self.encoding.set_anneal_iter(cur_it)
    
    @profile
    def forward(
        self, x: torch.Tensor, *, input_normalized=True, 
        return_h=False, max_level: int = None) -> Dict[str, torch.Tensor]:
        if not input_normalized:
            x = self.space.normalize_coords(x)
        # NOTE: x must be in range [-1,1]
        h = self.encoding(x, max_level=max_level)
        if not self.use_extra_embed:
            output = self.decoder(h)
        else:
            h_embed = self.extra_embed_fn(x)
            output = self.decoder(torch.cat([h, h_embed.to(h.dtype)], dim=-1))
        sdf = output[..., 0]
        if return_h:
            if self.is_extra_feat_from_output:
                h = output[..., 1:]
            return dict(sdf=sdf, h=h)
        else:
            return dict(sdf=sdf)

    def forward_sdf(
        self, x: torch.Tensor, *, input_normalized=True, max_level: int = None):
        return self.forward(x, input_normalized=input_normalized, return_h=False, max_level=max_level)

    @profile
    def forward_sdf_nablas(
        self, x: torch.Tensor, *, input_normalized=True, 
        has_grad:bool=None, nablas_has_grad:bool=None, 
        max_level: int=None) -> Dict[str, torch.Tensor]:
        
        if not input_normalized:
            x = self.space.normalize_coords(x)
        # NOTE: x must be in range [-1,1]
        has_grad = torch.is_grad_enabled() if has_grad is None else has_grad
        nablas_has_grad = has_grad if nablas_has_grad is None else (nablas_has_grad and has_grad)
        need_dL_dinput = has_grad and x.requires_grad
        x = x.requires_grad_(True)
        with torch.enable_grad():
            h = self.encoding.forward(x, max_level=max_level, need_dL_dinput=need_dL_dinput)
            if not self.use_extra_embed:
                h_full = h
            else:
                h_embed = self.extra_embed_fn(x)
                h_full = torch.cat([h, h_embed.to(h.dtype)], dim=-1)
            output = self.decoder(h_full)
            sdf = output[..., 0]
        
        #---- Decoder bwd_input
        dL_dh_full = autograd.grad(sdf, h_full, sdf.new_ones(sdf.shape), retain_graph=has_grad, create_graph=nablas_has_grad, only_inputs=True)[0]

        if not self.use_extra_embed:
            #---- Encoding bwd_input
            nablas = self.encoding.backward_dydx(dL_dh_full, x, max_level=max_level)
        else:
            #---- Encoding bwd_input
            nablas = self.encoding.backward_dydx(dL_dh_full[..., :h.size(-1)], x, max_level=max_level)
            
            #---- Extra nablas from extra_embed stream.
            # NOTE: Calculates dl_dhxxx only once.
            dL_dh_embed = dL_dh_full[..., h.size(-1):]
            if self.extra_embed_fn._embedder_type == 'identity':
                nablas_extra = dL_dh_embed
            else:
                nablas_extra = autograd.grad(h_embed, x, dL_dh_embed, retain_graph=has_grad, create_graph=nablas_has_grad, only_inputs=True)[0]
            nablas = nablas + nablas_extra
        
        if self.is_extra_feat_from_output: h = output[..., 1:]
        if not nablas_has_grad:
            nablas = nablas.detach()
        if not has_grad:
            sdf, h = sdf.detach(), h.detach()
        
        # NOTE: The 'x' used to compute 'dydx' here is already in the normalized space of [-1,1]^n_input_dim. 
        #       Hence, the computed 'nablas' need to be divided by 'self.space.radius3d_original' to obtain 'nablas' under the original input space.
        # NOTE: Returned nablas are already in obj's coords & scale, not in network's coords & scale
        return dict(sdf=sdf, h=h, nablas=(nablas * self.sdf_scale / self.space.radius3d_original))

    def forward_in_obj(self, x: torch.Tensor, invalid_sdf: float = 1.1, return_h=False, with_normal=False) -> Dict[str, torch.Tensor]:
        """
        Forward using coords from object's coordinates
        """
        net_x = self.space.normalize_coords(x)
        # NOTE: 1e-6 is LoTD's clamp eps. TODO: Make it correspondent.
        valid_i = ((net_x > -1+1e-6) & (net_x < 1-1e-6)).all(dim=-1).nonzero(as_tuple=True)
        
        if valid_i[0].numel() > 0:
            if with_normal:
                raw_ret = self.forward_sdf_nablas(net_x[valid_i], input_normalized=True, return_h=return_h)
            else:
                raw_ret = self.forward(net_x[valid_i], input_normalized=True, return_h=return_h)
        
        sdf = torch.full([*x.shape[:-1]], invalid_sdf, dtype=self.dtype, device=self.device)
        if valid_i[0].numel() > 0:
            sdf = sdf.to(raw_ret['sdf'])
            sdf.index_put_(valid_i, raw_ret['sdf'])
        ret = dict(sdf=sdf)
        
        if with_normal:
            nablas = torch.full([*x.shape[:-1], self.n_input_dim], 0., dtype=torch.float, device=self.device)
            if valid_i[0].numel() > 0:
                nablas = nablas.to(raw_ret['nablas'])
                nablas.index_put_(valid_i, raw_ret['nablas'])
            ret.update(nablas=nablas)

        if return_h:
            h = torch.full([*x.shape[:-1], self.n_extra_feat], dtype=self.dtype, device=self.device)
            if valid_i[0].numel() > 0:
                h = h.to(raw_ret['h'])
                h.index_put_(valid_i, raw_ret['h'])
            ret.update(h=h)

        return ret

    def training_initialize(self, config=dict(), logger=None, log_prefix: str=None):
        geo_init_method = self.geo_init_method
        if geo_init_method == 'pretrain' or ('pretrain_after' in geo_init_method):
            # NOTE: if pretrain_after_geometric, geometric init is already done in __init__() or populate().
            if not self.is_pretrained:
                pretrain_sdf_sphere(
                    self, inside_out=self.inside_out, 
                    target_radius=self.radius_init, 
                    logger=logger, log_prefix=log_prefix, **config)
                self.is_pretrained = ~self.is_pretrained
                return True
        elif geo_init_method == 'pretrain_after_zero_out':
            raise NotImplementedError
        return False

    def get_weight_reg(self, norm_type: float = 2.0):
        return self.decoder.get_weight_reg(norm_type)

    def get_sdf_curvature_1d(self, x: torch.Tensor, normals: torch.Tensor = None, eps: float = 1.0e-4, input_normalized=True):
        x = x.data # Detach potential pose gradients.
        if not input_normalized:
            x = self.space.normalize_coords(x)
        normal_query_fn = lambda x: self.forward_sdf_nablas(x, input_normalized=True, nablas_has_grad=True, has_grad=True)['nablas']
        if normals is None:
            normals = normal_query_fn(x)
        rand_dirs = F.normalize(torch.randn_like(x), dim=-1)
        normals = F.normalize(normals, dim=-1)
        # A random vector on the tangent plane of `x`
        tangent = torch.cross(normals, rand_dirs)
        # Shift along the tangent vector
        x_shifted = x + tangent * eps
        # Shifted query
        nablas_shifted = normal_query_fn(x_shifted)
        normals_shifted = F.normalize(nablas_shifted, dim=-1)
        # Dot product of two normals
        dot = (normals * normals_shifted).sum(dim=-1, keepdim=True)
        # (Original comment in permuto-SDF) 
        #   The dot would assign low weight importance to normals that are almost the same, \
        #   and increasing error the more they deviate. So it's something like and L2 loss. 
        #   But we want a L1 loss so we get the angle, and then we map it to range [0,1]
        angle = torch.acos(dot.clamp_(-1.0+1e-6, 1.0-1e-6))
        curvature = angle / (2*np.pi) # Map to [0,1]
        return curvature

    def training_setup(self, training_cfg: Union[Number, dict], name_prefix: str = ''):
        return super().training_setup(training_cfg, name_prefix)

    def training_update_lr(self, it: int):
        return super().training_update_lr(it)

    @torch.no_grad()
    def training_clip_grad(self):
        # Encodings' sepcial clip grad
        self.encoding.clip_grad_and_update_ema()
        # Decoder's sepcial clip grad 
        if self.clip_level_grad_ema_factor > 0:
            # Clip decoder's grad 
            # gnorm = torch.stack([p.grad.abs().max() for n,p in self.decoder.named_parameters() if p.grad is not None])
            gnorm = torch.stack([p.grad.norm() for n,p in self.decoder.named_parameters() if p.grad is not None])
            
            ema = self.decoder_grad_ema.copy_(gnorm.lerp(self.decoder_grad_ema, 0.99))
            for i, (n, p) in enumerate(self.decoder.named_parameters()):
                val = ema[i].item() * self.clip_level_grad_ema_factor
                # p.grad.clip_(-val, val)
                clip_norm_(p.grad, val)
        # General `clip_grad_val` or `clip_grad_norm`
        super().training_clip_grad()
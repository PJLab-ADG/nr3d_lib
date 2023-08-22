"""
@file   lotd_sdf.py
@author Jianfei Guo, Shanghai AI Lab
@brief  lotd_encoding + lotd_decoder
"""

__all__ = [
    'LoTDSDF'
]

import numpy as np
from typing import Dict, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd

from nr3d_lib.logger import Logger
from nr3d_lib.config import ConfigDict
from nr3d_lib.utils import tensor_statistics, torch_dtype

from nr3d_lib.models.utils import clip_norm_
from nr3d_lib.models.base import ModelMixin
from nr3d_lib.models.blocks import get_blocks
from nr3d_lib.models.embedders import get_embedder
from nr3d_lib.models.grids.lotd import LoTDEncoding, get_lotd_decoder
from nr3d_lib.models.fields.sdf.utils import idr_geometric_init, pretrain_sdf_sphere

class LoTDSDF(ModelMixin, nn.Module):
    def __init__(
        self,
        
        encoding_cfg: ConfigDict,
        decoder_cfg: ConfigDict,
        n_rgb_used_output: int = 0,
        sdf_scale: float = 1.0, # Length in real-world object's coord represented by one unit of SDF
        
        extra_pos_embed_cfg: dict = None,
        bounding_size=2.0, aabb=None, radius_init=0.5, 
        geo_init_method: str = None, inside_out=False, 
        
        clip_level_grad_ema_factor: float=0, 

        dtype=torch.float, device=torch.device("cuda"), use_tcnn_backend=False, 
        ) -> None:
        super().__init__()
        """
        n_rgb_used_output: 
            1. sdf decoder output width = 1 + n_rgb_used_output
            2. the extra_feat that the radiance net used:
                set to == 0: comes from encoding()
                set to  > 0: comes from output[.., 1:]
        """
        self.dtype = torch_dtype(dtype)
        self.device = device
        
        self.encoding_cfg = encoding_cfg
        self.decoder_cfg = decoder_cfg
        self.use_tcnn_backend = use_tcnn_backend
        
        self.sdf_scale = sdf_scale
        self.use_extra_embed = extra_pos_embed_cfg is not None
        self.radius_init = radius_init
        self.inside_out = inside_out
        self.geo_init_method = geo_init_method
        self.n_rgb_used_output = n_rgb_used_output
        self.is_extrafeat_from_output = self.n_rgb_used_output > 0
        self.clip_level_grad_ema_factor = clip_level_grad_ema_factor
        self.should_clip_level_grad_with_ema = clip_level_grad_ema_factor > 0

        #------- LoTD encoding
        if self.should_clip_level_grad_with_ema:
            self.encoding_cfg.setdefault('clip_level_grad_ema_factor', self.clip_level_grad_ema_factor)
        if aabb is not None:
            self.encoding_cfg.update(aabb=aabb)
        if bounding_size is not None:
            self.encoding_cfg.update(bounding_size=bounding_size)
        self.encoding = LoTDEncoding(3, **self.encoding_cfg, dtype=self.dtype, device=self.device)

        #------- (Optional) extra coords embedding
        if self.use_extra_embed:
            extra_pos_embed_cfg.setdefault('use_tcnn_backend', use_tcnn_backend)
            self.extra_embed_fn, self.n_extra_embed = get_embedder(extra_pos_embed_cfg, 3)
        else:
            self.n_extra_embed = 0

        #------- SDF decoder
        self.decoder_cfg.setdefault('use_tcnn_backend', self.use_tcnn_backend)
        if self.is_extrafeat_from_output:
            self.n_rgb_used_extrafeat = self.n_rgb_used_output
            self.decoder, self.decoder_type = get_lotd_decoder(self.encoding.lod_meta, (1+self.n_rgb_used_output), n_extra_embed_ch=self.n_extra_embed, **self.decoder_cfg, dtype=self.dtype, device=self.device)
        else:
            self.n_rgb_used_extrafeat = self.encoding.out_features
            self.decoder, self.decoder_type = get_lotd_decoder(self.encoding.lod_meta, 1, n_extra_embed_ch=self.n_extra_embed ,**self.decoder_cfg, dtype=self.dtype, device=self.device)
        if self.should_clip_level_grad_with_ema and (length:=len(list(self.decoder.parameters()))) > 0:
            self.register_buffer(f'decoder_grad_ema', torch.ones([length], device=self.device, dtype=torch.float))
        
        if self.geo_init_method == 'geometric' or self.geo_init_method == 'pretrain_after_geometric':
            # NOTE: Only work for MLP decoders & at least an identity extra pos embed.
            assert self.use_extra_embed, "Geometric init only works with sinusoidal/identity extra embedder"
            emb_tp = self.extra_embed_fn._embedder_type
            assert ('sinusoidal' in emb_tp) or (emb_tp == 'identity'), "Geometric init only works with sinusoidal/identity extra embedder"
            # Finally get to work!
            idr_geometric_init(self.decoder, n_embed=self.n_extra_embed, radius_init=self.radius_init, inside_out=self.inside_out)
        
        elif self.geo_init_method == 'zero_out' or self.geo_init_method == 'pretrain_after_zero_out':
            # NOTE: For lotd-annealing, set zero to non-active part of decoder input at start
            if self.encoding.annealer is not None:
                start_level = self.encoding.annealer.start_level
                start_n_feats = sum(self.encoding.lotd.level_n_feats[:start_level+1])
            else:
                start_level = 0
                start_n_feats = 0
            with torch.no_grad():
                # for l in range(start_level, self.encoding.lotd.n_levels, 1):
                #     self.encoding.get_level_param(l).zero_()
                if isinstance(self.decoder, nn.Sequential):
                    # [LoTDSelect, Decoder]
                    nn.init.zeros_(self.decoder[1].layers[0].weight[:, start_n_feats:])
                else:
                    nn.init.zeros_(self.decoder.layers[0].weight[:, start_n_feats:])
        
        self.register_buffer('is_pretrained', torch.tensor([False], dtype=torch.bool), persistent=True)

    @property
    def space(self):
        return self.encoding.space

    def preprocess_per_train_step(self, cur_it: int, logger: Logger = None):
        self.encoding.set_anneal_iter(cur_it)

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
            if self.is_extrafeat_from_output:
                h = output[..., 1:]
            return dict(sdf=sdf, h=h)
        else:
            return dict(sdf=sdf)

    def forward_sdf(
        self, x: torch.Tensor, *, input_normalized=True, max_level: int = None):
        return self.forward(x, input_normalized=input_normalized, return_h=False, max_level=max_level)

    # @profile
    def forward_sdf_nablas(
        self, x: torch.Tensor, *, input_normalized=True, 
        has_grad:bool=None, nablas_has_grad:bool=None, 
        max_level: int=None, grad_guard=None) -> Dict[str, torch.Tensor]:
        
        if not input_normalized:
            x = self.space.normalize_coords(x)
        # NOTE: x must be in range [-1,1]
        has_grad = torch.is_grad_enabled() if has_grad is None else has_grad
        nablas_has_grad = has_grad if nablas_has_grad is None else (nablas_has_grad and has_grad)
        need_loss_backward_input = x.requires_grad
        x = x.requires_grad_(True)
        with torch.enable_grad():
            h, dy_dx = self.encoding.forward_dydx(x, max_level=max_level, need_loss_backward_input=need_loss_backward_input)
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
            nablas = self.encoding.backward_dydx(dL_dh_full, dy_dx, x, max_level=max_level, grad_guard=grad_guard)
        else:
            #---- Encoding bwd_input
            nablas = self.encoding.backward_dydx(dL_dh_full[..., :h.shape[-1]], dy_dx, x, max_level=max_level, grad_guard=grad_guard)
            
            #---- Extra nablas from extra_embed stream.
            # NOTE: Calculates dl_dhxxx only once.
            dL_dh_embed = dL_dh_full[..., h.shape[-1]:]
            if self.extra_embed_fn._embedder_type == 'identity':
                nablas_extra = dL_dh_embed
            else:
                nablas_extra = autograd.grad(h_embed, x, dL_dh_embed, retain_graph=has_grad, create_graph=nablas_has_grad, only_inputs=True)[0]
            nablas = nablas + nablas_extra
        
        if self.is_extrafeat_from_output: h = output[..., 1:]
        if not nablas_has_grad:
            nablas = nablas.detach()
        if not has_grad:
            sdf, h = sdf.detach(), h.detach()
        
        # NOTE: The 'x' used to compute 'dydx' here is already in the normalized space of [-1,1]^3. 
        #       Hence, the computed 'nablas' need to be divided by 'self.space.scale0' to obtain 'nablas' under the original input space.
        # NOTE: Returned nablas are already in obj's coords & scale, not in network's coords & scale
        return dict(sdf=sdf, h=h, nablas=(nablas * self.sdf_scale / self.space.scale0))

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
            nablas = torch.full([*x.shape[:-1], 3], 0., dtype=torch.float, device=self.device)
            if valid_i[0].numel() > 0:
                nablas = nablas.to(raw_ret['nablas'])
                nablas.index_put_(valid_i, raw_ret['nablas'])
            ret.update(nablas=nablas)

        if return_h:
            h = torch.full([*x.shape[:-1], self.n_rgb_used_extrafeat], dtype=self.dtype, device=self.device)
            if valid_i[0].numel() > 0:
                h = h.to(raw_ret['h'])
                h.index_put_(valid_i, raw_ret['h'])
            ret.update(h=h)

        return ret

    """
    # NOTE: torch.autograd is unreliable !!! `only_inputs` not working properly !!! 
    # Due to torch.autograd issue https://github.com/pytorch/pytorch/issues/56500,
    #       additional dy_dparam is calculated when we just want dydx (and optionally its 2nd order gradients),
    #       which is terribly time consuming.
    # This bug will be fixed sometime after pytorch-1.13.  https://github.com/pytorch/pytorch/pull/82544
    # For compatibility consideration, we use a hacky solution instead (see the other `forward_sdf_nablas` below).
    # For more details, check lod_encodings.py: LoTDFunction, LoTDFunctionFwdDydx, LoTDFunctionBwdDydx
    
    def forward_sdf_nablas(self, x: torch.Tensor, has_grad:bool=None, nablas_has_grad:bool=None) -> Dict[str, torch.Tensor]:
        has_grad = torch.is_grad_enabled() if has_grad is None else has_grad
        nablas_has_grad = has_grad if nablas_has_grad is None else (nablas_has_grad and has_grad)
        # Force enabling grad for normal calculation
        with torch.enable_grad():
            x = x.requires_grad_(True)
            h = self.encoding(x)
            output = self.decoder(h)
            sdf = output[..., 0]
            if self.is_extrafeat_from_output: h = output[..., 1:]
            # NOTE: autograd.grad often introduces unecessary calculations that consume a lot of time 
            nablas = autograd.grad(sdf, x, torch.ones_like(sdf, device=x.device), create_graph=nablas_has_grad, retain_graph=has_grad, only_inputs=True)[0] # 450 ms; 15 ms after v1 fix
        if not has_grad:
            ret = dict(sdf=sdf.detach(), h=h.detach(), nablas=nablas.detach())
        else:
            ret = dict(sdf=sdf, h=h, nablas=nablas)
        return ret
    """

    def initialize(self, config=ConfigDict(), logger=None, log_prefix=None):
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

    @torch.no_grad()
    def rescale_volume(self, new_aabb: torch.Tensor):
        return self.encoding.rescale_volume(new_aabb)

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

    @torch.no_grad()
    def custom_grad_clip_step(self):
        self.encoding.custom_grad_clip_step()
        if self.should_clip_level_grad_with_ema:
            # Clip decoder's grad 
            # gnorm = torch.stack([p.grad.abs().max() for n,p in self.decoder.named_parameters() if p.grad is not None])
            gnorm = torch.stack([p.grad.norm() for n,p in self.decoder.named_parameters() if p.grad is not None])
            
            ema = self.decoder_grad_ema.copy_(gnorm.lerp(self.decoder_grad_ema, 0.99))
            for i, (n, p) in enumerate(self.decoder.named_parameters()):
                val = ema[i].item() * self.clip_level_grad_ema_factor
                # p.grad.clip_(-val, val)
                clip_norm_(p.grad, val)

    @torch.no_grad()
    def stat_param(self, with_grad=False, prefix: str='') -> Dict[str, float]:
        prefix_ = prefix + ('.' if prefix and not prefix.endswith('.') else '')
        ret = {}
        ret.update(self.encoding.stat_param(with_grad=with_grad, prefix=prefix_ + "encoding"))
        ret.update({prefix_ + f'decoder.total.{k}' : v for k,v in tensor_statistics(torch.cat([p.data.flatten() for p in self.decoder.parameters()])).items()})
        ret.update({prefix_ + f"decoder.{n}.{k}": v for n, p in self.decoder.named_parameters() for k, v in tensor_statistics(p.data).items()})
        if with_grad:
            ret.update({prefix_ + f'decoder.grad.total.{k}': v for k, v in tensor_statistics(torch.cat([p.grad.data.flatten() for p in self.decoder.parameters() if p.grad is not None])).items()})
            ret.update({prefix_ + f"decoder.grad.{n}.{k}": v for n, p in self.decoder.named_parameters() if p.grad is not None for k, v in  tensor_statistics(p.grad.data).items() })
        if self.should_clip_level_grad_with_ema:
            ret.update({prefix_ + f"decoder.grad.{n}.ema" : self.decoder_grad_ema[i].item() for i,(n,_) in enumerate(self.decoder.named_parameters())})
        return ret
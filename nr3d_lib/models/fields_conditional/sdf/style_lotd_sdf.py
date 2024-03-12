"""
@file   style_lotd_sdf.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Style-LoTD encodings + SDF decoder.
"""

__all__ = [
    'StyleLoTDSDF'
]

from numbers import Number
from typing import Dict, Union

import torch
import torch.nn as nn
from torch import autograd
import torch.nn.functional as F

from nr3d_lib.logger import Logger
from nr3d_lib.config import ConfigDict
from nr3d_lib.utils import tensor_statistics, torch_dtype

from nr3d_lib.models.model_base import ModelMixin
from nr3d_lib.models.embedders import get_embedder
from nr3d_lib.models.grid_encodings.lotd import LoTDBatched
from nr3d_lib.models.grid_encodings.multires_decoder import get_multires_decoder
from nr3d_lib.models.fields.sdf import idr_geometric_init
from nr3d_lib.models.fields_conditional.sdf.utils import pretrain_conditional_sdf_sphere

class StyleLoTDSDF(ModelMixin, nn.Module):
    def __init__(
        self, 
        
        lotd_grower_cfg: dict, 
        lotd_anneal_cfg: dict = None, 
        decoder_cfg: dict = dict(), 
        n_extra_feat_from_output: int = 0,
        
        extra_pos_embed_cfg: dict = None,
        bounding_size=2.0, radius_init=0.5,
        geo_init_method: str = None, inside_out=False, 
        
        dtype=torch.half, device=None, use_tcnn_backend=False, 
        ) -> None:
        """_summary_

        Args:
            lotd_grower_cfg (dict): _description_
            lotd_anneal_cfg (dict, optional): _description_. Defaults to None.
            decoder_cfg (dict, optional): _description_. Defaults to dict().
            n_extra_feat_from_output (int, optional): \
                Determines whether downstream networks (e.g., RGB decoders or semantic decoders) \
                use the direct encoding output or extra output channels from the decoder. \
                If set to 0: downstream networks use the encoded features directly. \
                    The SDF decoder outputs 1-channel (SDF).
                If set to >0: downstream networks use the additional output from the SDF decoder (output[.., 1:]). \
                    The SDF decoder outputs (1+`n_extra_feat_from_output`)-channel (SDF and feature). \
                Default is 0.
            extra_pos_embed_cfg (dict, optional): _description_. Defaults to None.
            bounding_size (float, optional): _description_. Defaults to 2.0.
            radius_init (float, optional): _description_. Defaults to 0.5.
            geo_init_method (str, optional): _description_. Defaults to None.
            inside_out (bool, optional): _description_. Defaults to False.
            dtype (_type_, optional): _description_. Defaults to torch.half.
            device (_type_, optional): _description_. Defaults to None.
            use_tcnn_backend (bool, optional): _description_. Defaults to False.
        """
        
        super().__init__()
        self.dtype = torch_dtype(dtype)

        self.radius_init = radius_init
        self.inside_out = inside_out
        self.use_extra_embed = extra_pos_embed_cfg is not None
        self.geo_init_method = geo_init_method
        self.is_extra_feat_from_output = n_extra_feat_from_output > 0
        self.clip_level_grad_ema_factor = 0

        #------- LoTD encoding
        self.encoding = LoTDBatched(
            3, grower_cfg=lotd_grower_cfg, anneal_cfg=lotd_anneal_cfg,
            bounding_size=bounding_size, dtype=self.dtype, device=device)
        
        #------- (Optional) extra coords embedding
        if self.use_extra_embed:
            extra_pos_embed_cfg.setdefault('use_tcnn_backend', use_tcnn_backend)
            self.extra_embed_fn, self.n_extra_embed = get_embedder(extra_pos_embed_cfg, 3)
        else:
            self.n_extra_embed = 0
        
        #------- SDF decoder
        decoder_cfg.setdefault('use_tcnn_backend', use_tcnn_backend)
        if self.is_extra_feat_from_output:
            self.n_extra_feat = n_extra_feat_from_output
            self.decoder, self.decoder_type = get_multires_decoder(
                self.encoding.lod_meta.level_n_feats, (1+n_extra_feat_from_output), n_extra_embed_ch=self.n_extra_embed, 
                **decoder_cfg, dtype=self.dtype, device=device)
        else:
            self.n_extra_feat = self.encoding.out_features
            self.decoder, self.decoder_type = get_multires_decoder(
                self.encoding.lod_meta.level_n_feats, 1, n_extra_embed_ch=self.n_extra_embed, 
                **decoder_cfg, dtype=self.dtype, device=device)

        if self.geo_init_method == 'geometric' or self.geo_init_method == 'pretrain_after_geometric':
            # NOTE: Only work for MLP decoders & at least an identity extra pos embed.
            assert self.use_extra_embed, "Geometric init only works with sinusoidal/identity extra embedder"
            emb_tp = self.extra_embed_fn._embedder_type
            assert ('sinusoidal' in emb_tp) or (emb_tp == 'identity'), "Geometric init only works with sinusoidal/identity extra embedder"
            # Finally get to work!
            idr_geometric_init(self.decoder, n_embed=self.n_extra_embed, radius_init=self.radius_init, inside_out=self.inside_out)

        self.register_buffer('is_pretrained', torch.tensor([False], dtype=torch.bool), persistent=True)

    @property
    def device(self) -> torch.device:
        return self.encoding.device

    @property
    def space(self):
        return self.encoding.space

    def training_before_per_step(self, cur_it: int, logger: Logger = None):
        self.encoding.set_anneal_iter(cur_it)

    def set_condition(self, z: torch.Tensor):
        assert z.shape[-1] == self.encoding.z_dim, "latent dim does not match"
        self.encoding.grow(z)
        self.space.set_condition()

    def clean_condition(self):
        self.encoding.clear()
        self.space.clean_condition()

    def forward(self, x: torch.Tensor, bidx: torch.Tensor = None, *, return_h=False, max_level: int = None) -> Dict[str, torch.Tensor]:
        # NOTE: x must be in range [-1,1]
        h = self.encoding(x, bidx, max_level=max_level)
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

    def forward_sdf_nablas(
        self, x: torch.Tensor, bidx: torch.Tensor = None, *, 
        has_grad:bool=None, nablas_has_grad:bool=None, max_level: int = None) -> Dict[str, torch.Tensor]:
        
        has_grad = torch.is_grad_enabled() if has_grad is None else has_grad
        nablas_has_grad = has_grad if nablas_has_grad is None else (nablas_has_grad and has_grad)
        need_dL_dinput = has_grad and x.requires_grad
        x = x.requires_grad_(True)
        with torch.enable_grad():
            h, dy_dx = self.encoding.forward_dydx(x, bidx, max_level=max_level, need_dL_dinput=need_dL_dinput)
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
            nablas = self.encoding.backward_dydx(dL_dh_full, dy_dx, x, bidx, max_level=max_level)
        else:
            #---- Encoding bwd_input
            nablas = self.encoding.backward_dydx(dL_dh_full[..., :h.size(-1)], dy_dx, x, bidx, max_level=max_level)
            
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
        return dict(sdf=sdf, h=h, nablas=nablas)

    def training_setup(self, training_cfg: Union[Number, dict], name_prefix: str = ''):
        return super().training_setup(training_cfg, name_prefix)

    def training_update_lr(self, it: int):
        return super().training_update_lr(it)

    @torch.no_grad()
    def training_clip_grad(self):
        super().training_clip_grad()

    @torch.no_grad()
    def stat_param(self, with_grad=False, prefix: str='') -> Dict[str, float]:
        prefix_ = prefix + ('.' if prefix and not prefix.endswith('.') else '')
        ret = {}
        ret.update(self.encoding.stat_param(with_grad=with_grad, prefix=prefix_+"encoding"))
        ret.update({prefix_ + f'decoder.total.{k}' : v for k,v in tensor_statistics(torch.cat([p.data.flatten() for p in self.decoder.parameters()])).items()})
        ret.update({prefix_ + f"decoder.{n}.{k}": v for n, p in self.decoder.named_parameters() for k, v in tensor_statistics(p.data).items()})
        if with_grad and len(pl:=[p.grad.data.flatten() for p in self.decoder.parameters() if p.grad is not None]) > 0:
            ret.update({prefix_ + f'decoder.grad.total.{k}': v for k, v in tensor_statistics(torch.cat(pl)).items()})
            ret.update({prefix_ + f"decoder.grad.{n}.{k}": v for n, p in self.decoder.named_parameters() if p.grad is not None for k, v in tensor_statistics(p.grad.data).items() })
        if self.clip_level_grad_ema_factor > 0:
            ret.update({prefix_ + f"decoder.grad.{n}.ema" : self.decoder_grad_ema[i].item() for i,(n,_) in enumerate(self.decoder.named_parameters())})
        return ret

    def training_initialize(self, config=dict(), logger=None, log_prefix: str=None):
        geo_init_method = self.geo_init_method
        bs = config.setdefault('batch_size', 1)
        z = config.get('z', torch.zeros([bs, self.encoding.z_dim], dtype=self.dtype, device=self.device))
        if geo_init_method == 'pretrain' or ('pretrain_after' in geo_init_method):
            if not self.is_pretrained:
                pretrain_conditional_sdf_sphere(
                    self, inside_out=self.inside_out, 
                    target_radius=self.radius_init, z=z, 
                    logger=logger, log_prefix=log_prefix, **config)
                self.is_pretrained = ~self.is_pretrained
                return True
        return False
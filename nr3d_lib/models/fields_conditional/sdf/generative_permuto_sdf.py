"""
@file   generative_permuto_sdf.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Conditional / generative SDF model based on high-dimensional permutohedral-lattice-based encoding.
"""

__all__ = [
    'GenerativePermutoConcatSDF', 
    'GenerativeMLLConcatSDF', 
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
from nr3d_lib.models.grid_encodings.permuto import PermutoEncoding, MLLNet
from nr3d_lib.models.grid_encodings.multires_decoder import get_multires_decoder
from nr3d_lib.models.fields.sdf.utils import idr_geometric_init
from nr3d_lib.models.fields_conditional.sdf.utils import pretrain_conditional_sdf_sphere

class GenerativePermutoConcatSDF(ModelMixin, nn.Module):
    def __init__(
        self,
        n_latent_dim: int, 
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
            n_latent_dim (int): _description_
            n_input_dim (int, optional): 2D/3D. Defaults to 3.
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
            sdf_scale (float, optional): \
                Length in real-world object's coord represented by one unit of SDF. \
                Defaults to 1.0.
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
        
        assert n_input_dim in [2,3], f"{self.__class__.__name__} expects only 2D/3D input, but got n_input_dim={n_input_dim}."
        self.n_input_dim = n_input_dim
        self.n_latent_dim = n_latent_dim
        self.encoding_cfg = encoding_cfg
        self.decoder_cfg = decoder_cfg
        self.use_tcnn_backend = use_tcnn_backend
        self.use_extra_embed = extra_pos_embed_cfg is not None
        
        self.sdf_scale = sdf_scale
        self.radius_init = radius_init
        self.inside_out = inside_out
        self.geo_init_method = geo_init_method
        self.n_extra_feat_from_output = n_extra_feat_from_output
        self.is_extra_feat_from_output = self.n_extra_feat_from_output > 0
        self.clip_level_grad_ema_factor = clip_level_grad_ema_factor
        
        #------- Permutohedral encoding
        if self.clip_level_grad_ema_factor > 0:
            self.encoding_cfg.setdefault('clip_level_grad_ema_factor', self.clip_level_grad_ema_factor)
        space_cfg = self.encoding_cfg.setdefault('space_cfg', {'type': 'batched'})
        if aabb is not None:
            space_cfg.update(aabb=aabb)
        if bounding_size is not None:
            space_cfg.update(bounding_size=bounding_size)
        self.encoding_cfg.update(space_cfg=space_cfg)
        self.encoding = PermutoEncoding(self.n_input_dim + self.n_latent_dim, **self.encoding_cfg, dtype=self.dtype, device=device)

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
    
    def set_condition(self, z: torch.Tensor):
        assert z.size(-1) == self.n_latent_dim, f"The input latents should have data_dim={self.n_latent_dim}, but got {z.size(-1)}"
        self.z_per_batch = z
        self.B = z.size(0)
        self.space.set_condition()
    
    def clean_condition(self):
        self.z_per_batch = None
        self.B = None
        self.space.clean_condition()
    
    def _check_or_get_z_per_x(
        self, 
        x: torch.Tensor, *, z: torch.Tensor = None, bidx: torch.Tensor = None, # per pts
        z_per_batch: torch.Tensor = None, # per batch
        ) -> torch.Tensor:
        if z is not None:
            assert [*z.shape[:-1]] == [*x.shape[:-1]], f"`z` should be of shape [{','.join([str(_s) for _s in x.shape[:-1]])},...]"
        else:
            z_per_batch = self.z_per_batch if z_per_batch is None else z_per_batch
            if bidx is not None:
                assert [*bidx.shape] == [*x.shape[:-1]], f"`bidx` should be of shape [{','.join([str(_s) for _s in x.shape[:-1]])}]"
                z = z_per_batch[bidx]
            else:
                assert x.size(0) == self.B, f"The input should have batch_size={self.B} when `bidx` and `z` are not given."
                z = z_per_batch.view(self.B,*[1]*(x.dim()-2),self.n_latent_dim).expand(*x.shape[:-1], -1)
        return z

    @profile
    def forward(
        self, 
        x: torch.Tensor, *, z: torch.Tensor = None, bidx: torch.Tensor = None, # per pts
        z_per_batch: torch.Tensor = None, # per batch
        input_normalized=True, return_h=False, max_level: int = None) -> Dict[str, torch.Tensor]:
        
        #---- Prepare x input
        assert x.size(-1) == self.n_input_dim, f"`x` should be of shape [...,{self.n_input_dim}], but got {[*x.shape]}"
        if not input_normalized:
            x = self.space.cur_batch__normalize_coords(x, bidx)
        
        #---- Prepare per-x z input
        z = self._check_or_get_z_per_x(x=x, z=z, bidx=bidx, z_per_batch=z_per_batch)
        
        h = self.encoding(torch.cat((x, z), dim=-1), max_level=max_level)
        if not self.use_extra_embed:
            output = self.decoder(h)
        else:
            output = self.decoder(torch.cat([h, self.extra_embed_fn(x).to(h.dtype)], dim=-1))
        
        sdf = output[..., 0]
        if return_h:
            if self.is_extra_feat_from_output:
                h = output[..., 1:]
            return dict(sdf=sdf, h=h)
        else:
            return dict(sdf=sdf)

    @profile
    def forward_sdf_nablas(
        self, x: torch.Tensor, *, z: torch.Tensor = None, 
        z_per_batch: torch.Tensor = None, bidx: torch.Tensor = None, 
        input_normalized=True, has_grad:bool=None, nablas_has_grad:bool=None, max_level: int=None
        ) -> Dict[str, torch.Tensor]:
        
        #---- Prepare x input
        assert x.size(-1) == self.n_input_dim, f"`x` should be of shape [...,{self.n_input_dim}], but got {[*x.shape]}"
        if not input_normalized:
            x = self.space.cur_batch__normalize_coords(x, bidx)
        
        #---- Prepare per-x z input
        z = self._check_or_get_z_per_x(x=x, z=z, bidx=bidx, z_per_batch=z_per_batch)
        
        has_grad = torch.is_grad_enabled() if has_grad is None else has_grad
        nablas_has_grad = has_grad if nablas_has_grad is None else (nablas_has_grad and has_grad)
        need_dL_dinput = has_grad and (x.requires_grad or z.requires_grad)
        x = x.requires_grad_(True)
        with torch.enable_grad():
            x_cat = torch.cat((x, z), dim=-1)
            h = self.encoding.forward(x_cat, max_level=max_level, need_dL_dinput=need_dL_dinput)
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
            nablas = self.encoding.backward_dydx(dL_dh_full, x_cat, max_level=max_level, max_pos_dims=self.n_input_dim)[..., :self.n_input_dim]
        else:
            #---- Encoding bwd_input
            nablas = self.encoding.backward_dydx(dL_dh_full[..., :h.size(-1)], x_cat, max_level=max_level, max_pos_dims=self.n_input_dim)[..., :self.n_input_dim]
            
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
        
        # NOTE: The 'x' used to compute 'dydx' here is already in the normalized space of [-1,1]^3. 
        #       Hence, the computed 'nablas' need to be divided by 'self.space.radius3d_original' to obtain 'nablas' under the original input space.
        # NOTE: Returned nablas are already in obj's coords & scale, not in network's coords & scale
        return dict(sdf=sdf, h=h, nablas=(nablas * self.sdf_scale / self.space.radius3d_original))

    def get_weight_reg(self, norm_type: float = 2.0):
        return self.decoder.get_weight_reg(norm_type)

    def training_initialize(self, config=dict(), logger=None, log_prefix: str=None):
        geo_init_method = self.geo_init_method
        bs = config.setdefault('batch_size', 1)
        z = config.get('z', torch.zeros([bs, self.n_latent_dim], dtype=self.dtype, device=self.device))
        if geo_init_method == 'pretrain' or ('pretrain_after' in geo_init_method):
            if not self.is_pretrained:
                pretrain_conditional_sdf_sphere(
                    self, inside_out=self.inside_out, 
                    target_radius=self.radius_init, z=z, 
                    logger=logger, log_prefix=log_prefix, **config)
                self.is_pretrained = ~self.is_pretrained
                return True
        return False

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

class GenerativeMLLConcatSDF(ModelMixin, nn.Module):
    def __init__(
        self,
        n_latent_dim: int, 
        n_input_dim: int = 3,
        mll_cfg: dict = ...,
        n_extra_feat_from_output: int = 0,
        sdf_scale: float = 1.0,
        
        bounding_size=2.0, aabb=None, radius_init=0.5, 
        geo_init_method: str = None, inside_out=False, 
        
        clip_level_grad_ema_factor: float=0, 

        dtype=torch.half, device=None, use_tcnn_backend=False, 
        ) -> None:
        """_summary_

        Args:
            n_latent_dim (int): _description_
            n_input_dim (int, optional): 2D/3D. Defaults to 3.
            mll_cfg (dict, optional): _description_. Defaults to ....
            n_extra_feat_from_output (int, optional): \
                Determines whether downstream networks (e.g., RGB decoders or semantic decoders) \
                use the direct encoding output or extra output channels from the decoder. \
                If set to 0: downstream networks use the encoded features directly. \
                    The SDF decoder outputs 1-channel (SDF).
                If set to >0: downstream networks use the additional output from the SDF decoder (output[.., 1:]). \
                    The SDF decoder outputs (1+`n_extra_feat_from_output`)-channel (SDF and feature). \
                Default is 0.
            sdf_scale (float, optional): \
                Length in real-world object's coord represented by one unit of SDF. \
                Defaults to 1.0.
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
        
        assert n_input_dim in [2,3], f"{self.__class__.__name__} expects only 2D/3D input, but got n_input_dim={n_input_dim}."
        self.n_input_dim = n_input_dim
        self.n_latent_dim = n_latent_dim
        self.mll_cfg = mll_cfg
        self.use_tcnn_backend = use_tcnn_backend
        
        self.sdf_scale = sdf_scale
        self.radius_init = radius_init
        self.inside_out = inside_out
        self.geo_init_method = geo_init_method
        self.n_extra_feat_from_output = n_extra_feat_from_output
        self.is_extra_feat_from_output = self.n_extra_feat_from_output > 0
        self.clip_level_grad_ema_factor = clip_level_grad_ema_factor
        
        #------- MLL Network
        if self.clip_level_grad_ema_factor > 0:
            self.mll_cfg.setdefault('clip_level_grad_ema_factor', self.clip_level_grad_ema_factor)
        if aabb is not None:
            self.mll_cfg.update(aabb=aabb)
        if bounding_size is not None:
            self.mll_cfg.update(bounding_size=bounding_size)
        
        self.mll = MLLNet(
            self.n_input_dim + self.n_latent_dim, 
            (1 + self.n_extra_feat_from_output) if self.is_extra_feat_from_output else 1, 
            **self.mll_cfg, space_cfg=dict(type='batched'), dtype=self.dtype, device=device)
        
        self.n_extra_feat = self.n_extra_feat_from_output if self.is_extra_feat_from_output else self.mll.last_encoded_features
        
        self.register_buffer('is_pretrained', torch.tensor([False], dtype=torch.bool), persistent=True)

    @property
    def device(self) -> torch.device:
        return self.mll.device
    
    @property
    def space(self):
        return self.mll.space
    
    def training_before_per_step(self, cur_it: int, logger: Logger = None):
        self.mll.set_anneal_iter(cur_it)
    
    def set_condition(self, z: torch.Tensor):
        assert z.size(-1) == self.n_latent_dim, f"The input latents should have data_dim={self.n_latent_dim}, but got {z.size(-1)}"
        self.z_per_batch = z
        self.B = z.size(0)
    
    def clean_condition(self):
        self.z_per_batch = None
        self.B = None

    def _check_or_get_z_per_x(
        self, 
        x: torch.Tensor, *, z: torch.Tensor = None, bidx: torch.Tensor = None, # per pts
        z_per_batch: torch.Tensor = None, # per batch
        ) -> torch.Tensor:
        if z is not None:
            assert [*z.shape[:-1]] == [*x.shape[:-1]], f"`z` should be of shape [{','.join([str(_s) for _s in x.shape[:-1]])},...]"
        else:
            z_per_batch = self.z_per_batch if z_per_batch is None else z_per_batch
            if bidx is not None:
                assert [*bidx.shape] == [*x.shape[:-1]], f"`bidx` should be of shape [{','.join([str(_s) for _s in x.shape[:-1]])}]"
                z = z_per_batch[bidx]
            else:
                assert x.size(0) == self.B, f"The input should have batch_size={self.B} when `bidx` and `z` are not given."
                z = z_per_batch.view(self.B,*[1]*(x.dim()-2),self.n_latent_dim).expand(*x.shape[:-1], -1)
        return z

    @profile
    def forward(
        self, x: torch.Tensor, *, z: torch.Tensor = None, 
        z_per_batch: torch.Tensor = None, bidx: torch.Tensor = None, 
        input_normalized=True, return_h=False, max_level: int = None) -> Dict[str, torch.Tensor]:
        
        #---- Prepare x input
        assert x.size(-1) == self.n_input_dim, f"`x` should be of shape [...,{self.n_input_dim}], but got {[*x.shape]}"
        if not input_normalized:
            x = self.space.cur_batch__normalize_coords(x, bidx)
        
        #---- Prepare per-x z input
        z = self._check_or_get_z_per_x(x=x, z=z, bidx=bidx, z_per_batch=z_per_batch)
        
        raw_ret = self.mll(torch.cat((x, z), dim=-1), max_level=max_level, return_h=True)
        sdf = raw_ret['output'][..., 0]
        if return_h:
            if self.is_extra_feat_from_output:
                h = raw_ret['output'][..., 1:]
            else:
                h = raw_ret['h']
            return dict(sdf=sdf, h=h)
        else:
            return dict(sdf=sdf)
    
    @profile
    def forward_sdf_nablas(
        self, x: torch.Tensor, *, z: torch.Tensor = None, 
        z_per_batch: torch.Tensor = None, bidx: torch.Tensor = None, 
        input_normalized=True, has_grad:bool=None, nablas_has_grad:bool=None, max_level: int=None
        ) -> Dict[str, torch.Tensor]:

        #---- Prepare x input
        assert x.size(-1) == self.n_input_dim, f"`x` should be of shape [...,{self.n_input_dim}], but got {[*x.shape]}"
        if not input_normalized:
            x = self.space.cur_batch__normalize_coords(x, bidx)
        
        #---- Prepare per-x z input
        z = self._check_or_get_z_per_x(x=x, z=z, bidx=bidx, z_per_batch=z_per_batch)
        
        has_grad = torch.is_grad_enabled() if has_grad is None else has_grad
        nablas_has_grad = has_grad if nablas_has_grad is None else (nablas_has_grad and has_grad)
        need_dL_dinput = has_grad and (x.requires_grad or z.requires_grad)
        
        raw_ret = self.mll.forward_with_nablas(
            torch.cat((x, z), dim=-1), 
            need_dL_dinput=need_dL_dinput, 
            has_grad=has_grad, nablas_has_grad=nablas_has_grad, 
            max_level=max_level, max_pos_dims=self.n_input_dim, max_out_dims=1)
        
        sdf = raw_ret['output'][..., 0]
        if self.is_extra_feat_from_output:
            h = raw_ret['output'][..., 1:]
        else:
            h = raw_ret['h']
            
        # NOTE: The 'x' used to compute 'dydx' here is already in the normalized space of [-1,1]^3. 
        #       Hence, the computed 'nablas' need to be divided by 'self.space.radius3d_original' to obtain 'nablas' under the original input space.
        # NOTE: Returned nablas are already in obj's coords & scale, not in network's coords & scale
        nablas = raw_ret['nablas'] * self.sdf_scale / self.space.radius3d_original

        return dict(sdf=sdf, h=h, nablas=nablas)

    def training_initialize(self, config=dict(), logger=None, log_prefix: str=None):
        geo_init_method = self.geo_init_method
        bs = config.setdefault('batch_size', 1)
        z = config.get('z', torch.zeros([bs, self.n_latent_dim], dtype=self.dtype, device=self.device))
        if geo_init_method == 'pretrain' or ('pretrain_after' in geo_init_method):
            if not self.is_pretrained:
                pretrain_conditional_sdf_sphere(
                    self, inside_out=self.inside_out, 
                    target_radius=self.radius_init, z=z, 
                    logger=logger, log_prefix=log_prefix, **config)
                self.is_pretrained = ~self.is_pretrained
                return True
        return False
    
    def get_weight_reg(self, norm_type: float = 2.0):
        return self.mll.get_weight_reg(norm_type)
    
    def training_setup(self, training_cfg: Union[Number, dict], name_prefix: str = ''):
        return super().training_setup(training_cfg, name_prefix)

    def training_update_lr(self, it: int):
        return super().training_update_lr(it)
    
    @torch.no_grad()
    def training_clip_grad(self):
        self.mll.clip_grad_and_update_ema()
        # General `clip_grad_val` or `clip_grad_norm`
        super().training_clip_grad()

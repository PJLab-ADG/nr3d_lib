"""
@file   lotd_forest_sdf.py
@author Jianfei Guo, Shanghai AI Lab
@brief  LoTD-Forest Encoding + SDF decoder.
"""

__all__ = [
    'LoTDForestSDF'
]

from numbers import Number
from typing import Dict, Literal, Union

import torch
import torch.nn as nn
from torch import autograd

from nr3d_lib.logger import Logger
from nr3d_lib.utils import tensor_statistics, torch_dtype

from nr3d_lib.models.utils import clip_norm_
from nr3d_lib.models.model_base import ModelMixin
from nr3d_lib.models.grid_encodings.lotd import LoTDForestEncoding
from nr3d_lib.models.grid_encodings.multires_decoder import get_multires_decoder

class LoTDForestSDF(ModelMixin, nn.Module):
    def __init__(
        self,
        
        encoding_cfg: dict,
        decoder_cfg: dict,
        n_extra_feat_from_output: int = 0,
        sdf_scale: float = 1.0,
        
        geo_init_method: str = None, radius_init=0.5, inside_out=False, 
        
        clip_level_grad_ema_factor: float=0, 

        dtype=torch.half, device=None, use_tcnn_backend=False, 
        ) -> None:
        """_summary_

        Args:
            encoding_cfg (dict): _description_
            decoder_cfg (dict): _description_
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
            geo_init_method (str, optional): _description_. Defaults to None.
            radius_init (float, optional): _description_. Defaults to 0.5.
            inside_out (bool, optional): _description_. Defaults to False.
            clip_level_grad_ema_factor (float, optional): _description_. Defaults to 0.
            dtype (_type_, optional): _description_. Defaults to torch.half.
            device (_type_, optional): _description_. Defaults to None.
            use_tcnn_backend (bool, optional): _description_. Defaults to False.
        """
        
        super().__init__()
        self.dtype = torch_dtype(dtype)
        
        self.sdf_scale = sdf_scale
        self.radius_init = radius_init
        self.inside_out = inside_out
        self.geo_init_method = geo_init_method
        self.n_extra_feat_from_output = n_extra_feat_from_output
        self.is_extra_feat_from_output = n_extra_feat_from_output > 0
        self.clip_level_grad_ema_factor = clip_level_grad_ema_factor
        
        #------- LoTD encoding
        if self.clip_level_grad_ema_factor > 0:
            self.encoding_cfg.setdefault('clip_level_grad_ema_factor', self.clip_level_grad_ema_factor)
        self.encoding = LoTDForestEncoding(3, **encoding_cfg, dtype=self.dtype, device=device)
        
        #------- SDF decoder
        decoder_cfg.setdefault('use_tcnn_backend', use_tcnn_backend)
        if self.is_extra_feat_from_output:
            self.n_extra_feat = n_extra_feat_from_output
            self.decoder, self.decoder_type = get_multires_decoder(self.encoding.lod_meta.level_n_feats, (1+n_extra_feat_from_output), **decoder_cfg, dtype=self.dtype, device=device)
        else:
            self.n_extra_feat = self.encoding.out_features
            self.decoder, self.decoder_type = get_multires_decoder(self.encoding.lod_meta.level_n_feats, 1, **decoder_cfg, dtype=self.dtype, device=device)
        if self.clip_level_grad_ema_factor > 0 and (length:=len(list(self.decoder.parameters()))) > 0:
            self.register_buffer(f'decoder_grad_ema', torch.ones([length], device=device, dtype=torch.float))
        
        self.register_buffer('is_pretrained', torch.tensor([False], dtype=torch.bool), persistent=True)

    @property
    def device(self) -> torch.device:
        return self.encoding.device

    @property
    def space(self):
        return self.encoding.space

    def populate(self, **kwargs):
        # Construct the forest representation
        self.encoding.populate(**kwargs)

    def training_before_per_step(self, cur_it: int, logger: Logger = None):
        self.encoding.set_anneal_iter(cur_it)

    def forward(
        self, 
        x: torch.Tensor, block_inds: torch.Tensor=None, block_offsets: torch.Tensor=None, 
        *, return_h=True, input_normalized=False) -> Dict[str, torch.Tensor]:
        
        if not input_normalized:
            block_x, block_inds = self.space.normalize_coords(x, block_inds)
        else:
            block_x = x
            
        # NOTE: block_x must be in range [-1,1]
        h = self.encoding(block_x, block_inds, block_offsets)
        output = self.decoder(h)
        ret = dict(sdf=output[..., 0])
        if return_h:
            if self.is_extra_feat_from_output:
                h = output[..., 1:]
            ret.update(h=h)
        return ret

    def forward_sdf(
        self, 
        x: torch.Tensor, block_inds: torch.Tensor=None, block_offsets: torch.Tensor=None, 
        *, return_h=True, input_normalized=False) -> Dict[str, torch.Tensor]:
        return self.forward(x, block_inds, block_offsets, return_h=return_h, input_normalized=input_normalized)
        
    def forward_sdf_nablas(
        self, 
        x: torch.Tensor, block_inds: torch.Tensor=None, block_offsets: torch.Tensor=None, 
        *, return_h=True, has_grad:bool=None, nablas_has_grad:bool=None, 
        input_normalized=False, max_level: int = None
        ) -> Dict[str, torch.Tensor]:
        
        if not input_normalized:
            block_x, block_inds = self.space.normalize_coords(x, block_inds)
        else:
            block_x = x
        
        # NOTE: `block_x` must be in range [-1,1]
        #       Always use the nablas in the normalized space in net (i.e. in block).
        has_grad = torch.is_grad_enabled() if has_grad is None else has_grad
        nablas_has_grad = has_grad if nablas_has_grad is None else (nablas_has_grad and has_grad)
        need_dL_dinput = has_grad and block_x.requires_grad
        block_x = block_x.requires_grad_(True)
        with torch.enable_grad():
            h, dy_dx = self.encoding.forward_dydx(block_x, block_inds, block_offsets, max_level=max_level, need_dL_dinput=need_dL_dinput)
            output = self.decoder(h)
            sdf = output[..., 0]
        
        #---- Decoder bwd_input
        dL_dy = autograd.grad(sdf, h, sdf.new_ones(sdf.shape), retain_graph=has_grad, create_graph=nablas_has_grad, only_inputs=True)[0]
        #---- Encoding bwd_input
        nablas = self.encoding.backward_dydx(dL_dy, dy_dx, block_x, block_inds, max_level=max_level)

        if self.is_extra_feat_from_output: h = output[..., 1:]
        if not nablas_has_grad:
            nablas = nablas.detach()
        if not has_grad:
            sdf, h = sdf.detach(), h.detach()

        ret = dict(sdf=sdf, nablas=nablas * (self.sdf_scale / self.space.world_block_size0 * 2.))        
        if return_h:
            ret.update(h=h)
        return ret

    def forward_in_obj(
        self, x: torch.Tensor, invalid_sdf: float = 1.1, 
        return_h=False, with_normal=False) -> Dict[str, torch.Tensor]:
        """
        Forward using x from object's coordinates
        """
        block_x, blidx = self.space.normalize_coords(x)
        valid_i = (blidx!=-1).nonzero(as_tuple=True)
        
        if valid_i[0].numel() > 0:
            if with_normal:
                raw_ret = self.forward_sdf_nablas(block_x[valid_i], blidx[valid_i], return_h=return_h)
            else:
                raw_ret = self.forward(block_x[valid_i], blidx[valid_i], return_h=return_h)
        
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
            h = torch.full([*x.shape[:-1], self.n_extra_feat], dtype=self.dtype, device=self.device)
            if valid_i[0].numel() > 0:
                h = h.to(raw_ret['h'])
                h.index_put_(valid_i, raw_ret['h'])
            ret.update(h=h)
        
        return ret
        
        # h_valid = self.encoding(block_x[valid_i], blidx[valid_i])
        # output_valid = self.decoder(h_valid)
        # sdf = output_valid.new_full([*block_x.shape[:-1]], invalid_sdf)
        # sdf.index_put_((valid_i,), output_valid[..., 0])
        
        # if return_h:
        #     h = torch.full([*block_x.shape[:-1]], 0)
        #     if self.is_extra_feat_from_output:
        #         h.index_put_((valid_i,), output_valid[..., 1:])
        #     else:
        #         h.index_put_((valid_i,), h_valid)
        #     return dict(sdf=sdf, h=h)
        # else:
        #     return dict(sdf=sdf)
    
    def training_initialize(self, config=dict(), logger=None, log_prefix: str=None):
        pass

    @torch.no_grad()
    def rescale_volume(self, new_aabb: torch.Tensor):
        # TODO: Do block pruning here
        pass

    def get_weight_reg(self, norm_type: float = 2.0):
        return self.decoder.get_weight_reg(norm_type)

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
        if self.clip_level_grad_ema_factor > 0:
            ret.update({prefix_ + f"decoder.grad.{n}.ema" : self.decoder_grad_ema[i].item() for i,(n,_) in enumerate(self.decoder.named_parameters())})
        return ret
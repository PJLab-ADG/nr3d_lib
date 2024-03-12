"""
@file   lotd_forest_neus.py
@author Jianfei Guo, Shanghai AI Lab
@brief  LoTD-Forest SDF + radiance net + neus control
"""

__all__ = [
    'LoTDForestNeuS', 
    'LoTDForestNeuSModel'
]


import numpy as np
from copy import deepcopy
from numbers import Number
from typing import Dict, List, Union

import torch
import torch.nn as nn

from nr3d_lib.fmt import log
from nr3d_lib.logger import Logger
from nr3d_lib.utils import tensor_statistics

from nr3d_lib.models.annealers import get_annealer
from nr3d_lib.models.utils import get_optimizer, get_scheduler
from nr3d_lib.models.model_base import ModelMixin
from nr3d_lib.models.fields.nerf import RadianceNet
from nr3d_lib.models.fields.neus import get_neus_var_ctrl
from nr3d_lib.models.fields_forest.sdf import LoTDForestSDF

from nr3d_lib.models.fields_forest.neus.renderer_mixin import NeuSRendererMixinForest

class LoTDForestNeuS(ModelMixin, nn.Module):
    def __init__(
        self, 
        var_ctrl_cfg=dict(ln_inv_s_init=0.3, ln_inv_s_factor=10.0), 
        cos_anneal_cfg: dict=None,
        
        surface_cfg=dict(), 
        radiance_cfg=dict(),
        
        dtype=torch.half, device=None, use_tcnn_backend=False, 
        ) -> None:
        super().__init__()
        
        self.dtype = dtype

        #------- Surface network
        surface_cfg.setdefault('use_tcnn_backend', use_tcnn_backend)
        self.implicit_surface = LoTDForestSDF(device=device, dtype=self.dtype, **surface_cfg)
        
        #------- Radiance network
        if radiance_cfg is not None:
            radiance_cfg['n_extra_feat'] = self.implicit_surface.n_extra_feat
            if use_tcnn_backend:
                from nr3d_lib.models.fields.nerf import TcnnRadianceNet
                self.radiance_net = TcnnRadianceNet(**radiance_cfg, device=device, dtype=self.dtype)
            else:
                self.radiance_net = RadianceNet(**radiance_cfg, device=device, dtype=self.dtype)
            self.use_view_dirs = self.radiance_net.use_view_dirs
            self.use_nablas = self.radiance_net.use_nablas
            self.use_h_appear = self.radiance_net.use_h_appear
        else:
            self.radiance_net = None
            self.use_view_dirs = False
            self.use_nablas = False
            self.use_h_appear = False

        #------- Variance control (manual or learnable)
        self.ctrl_var = get_neus_var_ctrl(**var_ctrl_cfg, device=device)
        
        #------- (Optional) Cosine value control as in original NeuS repo (for estimating prev/next sdfs)
        self.ctrl_cos_anneal = get_annealer(**cos_anneal_cfg) if cos_anneal_cfg is not None else None

    @property
    def device(self) -> torch.device:
        return self.implicit_surface.device
    
    @property
    def space(self):
        return self.implicit_surface.space

    def model_setup(self):
        self.implicit_surface.model_setup()

    def training_before_per_step(self, cur_it: int, logger: Logger = None):
        self.ctrl_var.set_iter(cur_it)
        if self.ctrl_cos_anneal is not None:
            self.ctrl_cos_anneal.set_iter(cur_it)
        self.implicit_surface.training_before_per_step(cur_it, logger=logger)

    def forward_inv_s(self):
        return self.ctrl_var()
    
    # @profile
    def forward_sdf(
        self, 
        x: torch.Tensor, block_inds: torch.Tensor=None, block_offsets: torch.Tensor=None, 
        *, input_normalized=False):
        return self.implicit_surface(x, block_inds, block_offsets, input_normalized=input_normalized)
    # @profile
    def forward_sdf_nablas(
        self,  
        x: torch.Tensor, block_inds: torch.Tensor=None, block_offsets: torch.Tensor=None, 
        *, has_grad:bool=None, nablas_has_grad:bool=None, input_normalized=False):
        return self.implicit_surface.forward_sdf_nablas(x, block_inds, block_offsets, has_grad=has_grad, nablas_has_grad=nablas_has_grad, input_normalized=input_normalized)
    # @profile
    def forward(
        self, x: torch.Tensor, *, 
        v: torch.Tensor, block_inds: torch.Tensor=None, block_offsets: torch.Tensor=None, h_appear: torch.Tensor = None,
        has_grad:bool=None, nablas_has_grad:bool=None, with_rgb=True, with_normal=True, input_normalized=False):
        prefix = x.shape[:-1]
        if with_normal or (with_rgb and self.use_nablas):
            raw_ret = self.forward_sdf_nablas(x, block_inds, block_offsets, has_grad=has_grad, nablas_has_grad=nablas_has_grad, input_normalized=input_normalized)
        else:
            raw_ret = self.forward_sdf(x, block_inds, block_offsets, input_normalized=input_normalized)
        if with_rgb:
            raw_ret.update(self.radiance_net(
                x, 
                v=v.expand(*prefix, 3) if self.use_view_dirs else None, 
                n=raw_ret['nablas'].detach().clamp(-1,1) if self.use_nablas else None, # NOTE: `.clamp(-1,1)` is to avoid salt & pepper noise on RGB
                h_extra=raw_ret['h'], 
                h_appear=h_appear.expand(*prefix,-1)) if h_appear is not None else None)
        return raw_ret

    def populate(self, **kwargs):
        # Construct the forest representation
        self.implicit_surface.populate(**kwargs)

    def training_initialize(self, config=dict(), logger=None, log_prefix: str=None):
        """
        #--- Initialize the implicit surface ---
        """
        pass

    def get_weight_reg(self, norm_type: float = 2.0, alpha_sdf: float = 1.0, alpha_rgb: float = 1.0) -> torch.Tensor:
        sdf_weight_reg = self.implicit_surface.decoder.get_weight_reg(norm_type).to(self.device)
        if self.radiance_net is not None:
            return torch.cat([alpha_sdf * sdf_weight_reg, alpha_rgb * self.radiance_net.get_weight_reg(norm_type).to(self.device)])
        else:
            return alpha_sdf * sdf_weight_reg

    def get_color_lipshitz_bound(self) -> torch.Tensor:
        return self.radiance_net.blocks.lipshitz_bound_full()

    @torch.no_grad()
    def stat_param(self, with_grad=False, prefix: str='') -> Dict[str, float]:
        prefix_ = prefix + ('.' if prefix and not prefix.endswith('.') else '')
        ret = {}
        ret.update(self.implicit_surface.stat_param(with_grad=with_grad, prefix=prefix_ + 'surface.'))
        ret.update({prefix_ + f'radiance_net.total.{k}': v for k, v in tensor_statistics(torch.cat([p.data.flatten() for p in self.radiance_net.parameters()])).items()})
        ret.update({prefix_ + f"radiance_net.{n}.{k}": v for n, p in self.radiance_net.named_parameters() for k, v in tensor_statistics(p.data).items()})
        if with_grad:
            ret.update({prefix_ + f'radiance_net.grad.total.{k}': v for k, v in tensor_statistics(torch.cat([p.grad.data.flatten() for p in self.radiance_net.parameters() if p.grad is not None])).items()})
            ret.update({prefix_ + f"radiance_net.grad.{n}.{k}": v for n, p in self.radiance_net.named_parameters() if p.grad is not None for k, v in tensor_statistics(p.grad.data).items()})
        return ret

    def training_setup(self, training_cfg: Union[Number, dict], name_prefix: str = ''):
        """
        NOTE: This function is just for seperately specifying `betas` for learnable inv_s ctrl
              Stable SDF (eikonal) training prefers betas=(0.9,0.99), \
                  but the original inv_s self-adaptiveness works better with betas=(0.9,0.999)
        """
        if (self.ctrl_var is None) \
            or (len(list(self.ctrl_var.parameters())) == 0) \
            or (training_cfg.get('invs_betas', None) is None):
            return super().training_setup(training_cfg, name_prefix=name_prefix)
        else:
            training_cfg = deepcopy(training_cfg)
            prefix_ = name_prefix + ('.' if name_prefix and not name_prefix.endswith('.') else '')
            sched_kwargs = training_cfg.pop('scheduler', None)
            betas = training_cfg.pop('betas', [0.9, 0.999])
            invs_betas = training_cfg.pop('invs_betas', betas)
            
            pg_invs = {
                'name': prefix_ + "invs", 
                'params': [], 
                'betas': invs_betas
            }
            
            pg_network = {
                'name': prefix_ + "network", 
                'params': [], 
                'betas': betas
            }
            
            for n, p in self.named_parameters():
                if 'ctrl_var' in n:
                    pg_invs['params'].append(p)
                else:
                    pg_network['params'].append(p)
            
            param_groups = [pg_invs, pg_network]
            self.optimizer = get_optimizer(param_groups, **training_cfg)
            self.scheduler = get_scheduler(self.optimizer, **sched_kwargs)

    def training_update_lr(self, it: int):
        return super().training_update_lr(it)

    @torch.no_grad()
    def training_clip_grad(self):
        # SDF model's sepcial clip grad
        self.implicit_surface.training_clip_grad()
        # General `clip_grad_val` or `clip_grad_norm`
        super().training_clip_grad()



class LoTDForestNeuSModel(NeuSRendererMixinForest, LoTDForestNeuS):
    """
    MRO:
    -> NeuSRendererMixinForest
    -> LoTDForestNeuS
    -> ModelMixin
    -> nn.Module
    """
    pass

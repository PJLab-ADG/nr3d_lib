"""
@file   dynamic_permuto_neus.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Dynamic NeuS network characterized using the Permutohedral-encoding model.
"""

__all__ = [
    'DynamicPermutoConcatNeuS', 
    'DynamicPermutoConcatNeuSModel'
]

import numpy as np
from copy import deepcopy
from numbers import Number
from typing import Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from nr3d_lib.fmt import log
from nr3d_lib.logger import Logger
from nr3d_lib.config import ConfigDict
from nr3d_lib.profile import profile
from nr3d_lib.utils import tensor_statistics, torch_dtype

from nr3d_lib.models.model_base import ModelMixin
from nr3d_lib.models.utils import get_optimizer, get_scheduler
from nr3d_lib.models.annealers import get_annealer
from nr3d_lib.models.fields.nerf import RadianceNet
from nr3d_lib.models.fields.neus.variance import get_neus_var_ctrl
from nr3d_lib.models.fields_dynamic.neus.renderer_mixin import NeusRendererMixinDynamic
from nr3d_lib.models.fields_dynamic.sdf import DynamicPermutoConcatSDF, DynamicMLLConcatSDF

class DynamicPermutoConcatNeuS(ModelMixin, nn.Module):
    def __init__(
        self,
        surface_cfg: dict,
        var_ctrl_cfg: dict=dict(ln_inv_s_init=0.3, ln_inv_s_factor=10.0), 
        radiance_cfg: dict = None,
        cos_anneal_cfg: dict = None,
        dtype=torch.half, device=None, use_tcnn_backend=False
        ) -> None:
        super().__init__()
        self.dtype = torch_dtype(dtype)
        self.set_device = device
        
        self.surface_cfg = deepcopy(surface_cfg)
        self.radiance_cfg = deepcopy(radiance_cfg) if radiance_cfg is not None else None
        self.use_tcnn_backend = use_tcnn_backend

        self.cos_anneal_cfg = cos_anneal_cfg
        
        self.var_ctrl_cfg = var_ctrl_cfg

        self.max_level = None
        
    def populate(self, bounding_size=None, aabb=None, device=None, n_latent_dim: int = ...):
        device = device or self.set_device
        self.set_device = device
        
        #------- Surface network
        self.surface_cfg.update(n_latent_dim=n_latent_dim, use_tcnn_backend=self.use_tcnn_backend)
        surface_net_use_mll = self.surface_cfg.pop('use_mll', False)
        if aabb is not None:
            self.surface_cfg.update(aabb=aabb)
        if bounding_size is not None:
            self.surface_cfg.update(bounding_size=bounding_size)
        if surface_net_use_mll:
            self.implicit_surface = DynamicMLLConcatSDF(**self.surface_cfg, n_input_dim=3, dtype=self.dtype, device=device)
        else:
            self.implicit_surface = DynamicPermutoConcatSDF(**self.surface_cfg, n_input_dim=3, dtype=self.dtype, device=device)
        self.implicit_surface.populate(device=device)
        
        #------- Radiance network
        if self.radiance_cfg is not None:
            self.radiance_cfg['n_extra_feat'] = self.implicit_surface.n_extra_feat
            if self.use_tcnn_backend:
                from nr3d_lib.models.fields.nerf import TcnnRadianceNet
                self.radiance_net = TcnnRadianceNet(**self.radiance_cfg, device=device, dtype=self.dtype)
            else:
                self.radiance_net = RadianceNet(**self.radiance_cfg, device=device, dtype=self.dtype)
            self.use_view_dirs = self.radiance_net.use_view_dirs
            self.use_nablas = self.radiance_net.use_nablas
            self.use_h_appear = self.radiance_net.use_h_appear
        else:
            self.radiance_net = None
            self.use_view_dirs = False
            self.use_nablas = False
            self.use_h_appear = False

        #------- Variance control (manual or learnable)
        self.ctrl_var = get_neus_var_ctrl(**self.var_ctrl_cfg, device=device)

        #------- (Optional) Cosine value control as in original NeuS repo (for estimating prev/next sdfs)
        self.ctrl_cos_anneal = get_annealer(**self.cos_anneal_cfg) if self.cos_anneal_cfg is not None else None

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

    def forward_sdf(self, x: torch.Tensor, *, z_time: torch.Tensor = None, z_time_single: torch.Tensor = None, return_h=False, input_normalized=True):
        return self.implicit_surface(x, z_time=z_time, z_time_single=z_time_single, return_h=return_h, max_level=self.max_level, input_normalized=input_normalized)
    
    def query_sdf(self, x: torch.Tensor, z_time: torch.Tensor = None, z_time_single: torch.Tensor = None, input_normalized=True) -> torch.Tensor:
        return self.forward_sdf(x, z_time=z_time, z_time_single=z_time_single, input_normalized=input_normalized)['sdf']
    
    def forward_sdf_nablas(
        self, x: torch.Tensor, *, z_time: torch.Tensor = None, z_time_single: torch.Tensor = None, 
        input_normalized=True, has_grad: bool=None, nablas_has_grad: bool=None):
        return self.implicit_surface.forward_sdf_nablas(
            x, z_time=z_time, z_time_single=z_time_single, 
            input_normalized=input_normalized, has_grad=has_grad, nablas_has_grad=nablas_has_grad, 
            max_level=self.max_level)
    
    @profile
    def forward(
        self, x: torch.Tensor, *, v: torch.Tensor = None, 
        z_time: torch.Tensor = None, z_time_single: torch.Tensor = None, h_appear: torch.Tensor = None, 
        input_normalized=True, has_grad:bool=None, nablas_has_grad: bool=None, with_rgb=True, with_normal=True):
        prefix = x.shape[:-1]
        if not input_normalized:
            x = self.space.normalize_coords(x)

        if with_normal or (with_rgb and self.use_nablas):
            raw_ret = self.forward_sdf_nablas(
                x, z_time=z_time, z_time_single=z_time_single,  
                has_grad=has_grad, nablas_has_grad=nablas_has_grad, input_normalized=True)
        else:
            raw_ret = self.forward_sdf(x, input_normalized=True)
        if with_rgb:
            raw_ret.update(self.radiance_net(
                x, 
                v=v.expand(*prefix, 3) if self.use_view_dirs else None, 
                # NOTE: `.clamp(-1,1)` is to avoid salt & pepper noise on RGB caused by large dydx
                n=raw_ret['nablas'].detach().clamp(-1,1) if self.use_nablas else None, # opt1
                # raw_ret['nablas'] if self.use_nablas else None, # opt2
                # F.normalize(raw_ret['nablas'], dim=-1) if self.use_nablas else None, # opt3
                # F.normalize(raw_ret['nablas'].detach(), dim=-1) if self.use_nablas else None,  # opt4
                h_extra=raw_ret['h'], 
                h_appear=h_appear.expand(*prefix,-1) if h_appear is not None else None)
            )
        return raw_ret

    @torch.no_grad()
    def rescale_volume(self, new_aabb: torch.Tensor):
        self.implicit_surface.rescale_volume(new_aabb)

    def get_weight_reg(self, norm_type: float = 2.0, alpha_sdf: float = 1.0, alpha_rgb: float = 1.0) -> torch.Tensor:
        sdf_weight_reg = self.implicit_surface.decoder.get_weight_reg(norm_type).to(self.device)
        if self.radiance_net is not None:
            return torch.cat([alpha_sdf * sdf_weight_reg, alpha_rgb * self.radiance_net.get_weight_reg(norm_type).to(self.device)])
        else:
            return alpha_sdf * sdf_weight_reg

    def get_sdf_curvature_1d(self, *args, **kwargs):
        return self.implicit_surface.get_sdf_curvature_1d(*args, **kwargs)

    def get_color_lipshitz_bound(self, *args, **kwargs):
        return self.radiance_net.blocks.lipshitz_bound_full(*args, **kwargs)

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
        # SDF model's special clip grad
        self.implicit_surface.training_clip_grad()
        # General `clip_grad_val` or `clip_grad_norm`
        super().training_clip_grad()

class DynamicPermutoConcatNeuSModel(NeusRendererMixinDynamic, DynamicPermutoConcatNeuS):
    """
    MRO:
    -> DynamicPermutoConcatNeuSModel
    -> NeusRendererMixinDynamic
    -> DynamicPermutoConcatNeuS
    -> ModelMixin
    -> nn.Module
    """
    pass

"""
@file   neus.py
@author Jianfei Guo, Shanghai AI Lab
@brief  sdf + radiance_net + neus control

TODO: What's the difference between `lotd_neus.py` and `neus.py`? Maybe merge them in the future?
Recorded differences between the two up to now:
1. `lotd_neus` has a `self.max_level` variable in `neus`, while `mlp_neus` does not;
    used in `forward_sdf` and `forward_sdf_nablas`.
    This is because if we later consider the radiance network of `lotd_neus` as a separate `lotd` as well, 
    combined with the LOD paradigm, it becomes necessary.
2. `lotd_neus`' `forward_sdf_nablas` has an optional `grad_guard` variable.
3. `mlp_neus` does not have `rescale_volume` and `custom_grad_clip_step`.
"""

__all__ = [
    'MlpPENeuS', 
    'MlpPENeuSModel'
]

import numpy as np
from typing import Dict

import torch
import torch.nn as nn

from nr3d_lib.fmt import log
from nr3d_lib.logger import Logger
from nr3d_lib.config import ConfigDict
from nr3d_lib.utils import tensor_statistics, torch_dtype

from nr3d_lib.models.base import ModelMixin
from nr3d_lib.models.annealers import get_annealer
from nr3d_lib.models.fields.sdf import MlpPESDF
from nr3d_lib.models.fields.nerf import RadianceNet
from nr3d_lib.models.fields.neus.variance import get_neus_var_ctrl
from nr3d_lib.models.fields.neus.renderer_mixin import NeusRendererMixin

class MlpPENeuS(ModelMixin, nn.Module):
    def __init__(
        self,
        surface_cfg: ConfigDict,
        var_ctrl_cfg: ConfigDict=ConfigDict(ln_inv_s_init=0.3, ln_inv_s_factor=10.0), 
        radiance_cfg: ConfigDict=None,
        cos_anneal_cfg: ConfigDict=None,
        dtype=torch.float, device=torch.device("cuda"), use_tcnn_backend=False, 
        ) -> None:
        super().__init__()
        self.dtype = torch_dtype(dtype)
        self.device = device
        
        self.surface_cfg = surface_cfg
        self.radiance_cfg = radiance_cfg
        self.use_tcnn_backend = use_tcnn_backend

        self.cos_anneal_cfg = cos_anneal_cfg
        self.var_ctrl_cfg = var_ctrl_cfg
        
    def populate(self, bounding_size=None, aabb=None, dtype=None, device=None):
        if dtype is not None:
            self.dtype = dtype
        if device is not None:
            self.device = device
        
        #------- Surface network
        self.surface_cfg.update(use_tcnn_backend=self.use_tcnn_backend)
        if aabb is not None:
            self.surface_cfg.update(aabb=aabb)
        if bounding_size is not None:
            self.surface_cfg.update(bounding_size=bounding_size)
        self.implicit_surface = MlpPESDF(**self.surface_cfg, dtype=self.dtype, device=self.device)
        
        #------- Radiance network
        if self.radiance_cfg is not None:
            self.radiance_cfg['n_rgb_used_extrafeat'] = self.implicit_surface.n_rgb_used_extrafeat
            if self.use_tcnn_backend:
                from nr3d_lib.models.fields.nerf import TcnnRadianceNet
                self.radiance_net = TcnnRadianceNet(**self.radiance_cfg, device=self.device, dtype=self.dtype)
            else:
                self.radiance_net = RadianceNet(**self.radiance_cfg, device=self.device, dtype=self.dtype)
            self.radiance_use_view_dirs = self.radiance_net.use_view_dirs
            self.radiance_use_nablas = self.radiance_net.use_nablas
        else:
            self.radiance_net = None
            self.radiance_use_view_dirs = False
            self.radiance_use_nablas = False

        #------- Variance control (manual or learnable)
        self.ctrl_var = get_neus_var_ctrl(**self.var_ctrl_cfg, device=self.device)

        #------- (Optional) Cosine value control as in original NeuS repo (for estimating prev/next sdfs)
        self.ctrl_cos_anneal = get_annealer(**self.cos_anneal_cfg) if self.cos_anneal_cfg is not None else None

        # TODO: To be removed later.
        self._register_load_state_dict_pre_hook(self.load_state_dict_hook)

    def load_state_dict_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        if prefix + 'ln_s' in state_dict:
            log.warning("=> ln_s is deprecated. Already changed to ctrl_var.ln_inv_s for clarification.")
            state_dict[prefix + 'ctrl_var.ln_inv_s'] = state_dict.pop(prefix + 'ln_s')

    @property
    def space(self):
        return self.implicit_surface.space

    def preprocess_model(self):
        self.implicit_surface.preprocess_model()

    def preprocess_per_train_step(self, cur_it: int, logger: Logger = None):
        self.ctrl_var.set_iter(cur_it)
        if self.ctrl_cos_anneal is not None:
            self.ctrl_cos_anneal.set_iter(cur_it)
        self.implicit_surface.preprocess_per_train_step(cur_it, logger=logger)

    def forward_inv_s(self):
        return self.ctrl_var()

    # @profile
    def forward_sdf(self, x: torch.Tensor, input_normalized=True):
        return self.implicit_surface(x, input_normalized=input_normalized)
    def query_sdf(self, x: torch.Tensor, input_normalized=True) -> torch.Tensor:
        return self.forward_sdf(x, input_normalized=input_normalized)['sdf']
    # @profile
    def forward_sdf_nablas(self,  x: torch.Tensor, *, has_grad:bool=None, nablas_has_grad:bool=None, input_normalized=True):
        return self.implicit_surface.forward_sdf_nablas(x, has_grad=has_grad, nablas_has_grad=nablas_has_grad, input_normalized=input_normalized)
    def forward(
        self, x: torch.Tensor, v: torch.Tensor, *, h_appear_embed: torch.Tensor = None, 
        input_normalized=True, has_grad:bool=None, nablas_has_grad:bool=None, 
        with_rgb=True, with_normal=True):
        prefix = x.shape[:-1]
        if not input_normalized:
            x = self.space.normalize_coords(x)
        if with_normal or (with_rgb and self.radiance_use_nablas):
            raw_ret = self.forward_sdf_nablas(
                x, input_normalized=True, has_grad=has_grad, nablas_has_grad=nablas_has_grad)
        else:
            raw_ret = self.forward_sdf(x, input_normalized=True)
        if with_rgb:
            raw_ret.update(
                self.radiance_net(
                    x, 
                    v.expand(*prefix, 3) if self.radiance_use_view_dirs else None, 
                    # NOTE: `.clamp(-1,1)` is to avoid salt & pepper noise on RGB caused by large dydx
                    raw_ret['nablas'].detach().clamp(-1,1) if self.radiance_use_nablas else None, 
                    # raw_ret['nablas'] if self.radiance_use_nablas else None
                    # F.normalize(raw_ret['nablas'], dim=-1) if self.radiance_use_nablas else None
                    # F.normalize(raw_ret['nablas'].detach(), dim=-1) if self.radiance_use_nablas else None
                    h_extra=raw_ret['h'], h_appear_embed=h_appear_embed)
            )
        return raw_ret

    def get_weight_reg(self, norm_type: float = 2.0, alpha_sdf: float = 1.0, alpha_rgb: float = 1.0) -> torch.Tensor:
        sdf_weight_reg = self.implicit_surface.decoder.get_weight_reg(norm_type).to(self.device)
        if self.radiance_net is not None:
            return torch.cat([alpha_sdf * sdf_weight_reg, alpha_rgb * self.radiance_net.get_weight_reg(norm_type).to(self.device)])
        else:
            return alpha_sdf * sdf_weight_reg

    @torch.no_grad()
    def stat_param(self, with_grad=False, prefix: str='') -> Dict[str, float]:
        prefix_ = prefix + ('.' if prefix and not prefix.endswith('.') else '')
        ret = {}
        ret.update(self.implicit_surface.stat_param(with_grad=with_grad, prefix=prefix_ + "surface"))
        ret.update({prefix_ + f'radiance_net.total.{k}': v for k, v in tensor_statistics(torch.cat([p.data.flatten() for p in self.radiance_net.parameters()])).items()})
        ret.update({prefix_ + f"radiance_net.{n}.{k}": v for n, p in self.radiance_net.named_parameters() for k, v in tensor_statistics(p.data).items()})
        if with_grad:
            ret.update({prefix_ + f'radiance_net.grad.total.{k}': v for k, v in tensor_statistics(torch.cat([p.grad.data.flatten() for p in self.radiance_net.parameters() if p.grad is not None])).items()})
            ret.update({prefix_ + f"radiance_net.grad.{n}.{k}": v for n, p in self.radiance_net.named_parameters() if p.grad is not None for k, v in tensor_statistics(p.grad.data).items()})
        return ret

    # @torch.no_grad()
    # def rescale_volume(self, new_aabb: torch.Tensor):
    #     self.implicit_surface.rescale_volume(new_aabb)

    # @torch.no_grad()
    # def custom_grad_clip_step(self):
    #     self.implicit_surface.custom_grad_clip_step()

class MlpPENeuSModel(NeusRendererMixin, MlpPENeuS):
    """
    MRO: NeusRendererMixin -> MlpPENeuS -> ModelMixin -> nn.Module
    """
    pass

if __name__ == "__main__":
    def unit_test(device=torch.device('cuda')):
        pass
        
    unit_test()
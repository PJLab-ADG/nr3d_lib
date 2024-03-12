"""
@file   style_lotd_neus.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Style-LoTD SDF + radiance net + neus control.
"""

__all__ = [
    'StyleLoTDNeuS', 
    'StyleLoTDNeuSModel', 
    'StyleNeuSLXY', 
    'StyleNeuSLXYModel', 
]

import numpy as np
from copy import deepcopy
from numbers import Number
from typing import Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from nr3d_lib.fmt import log
from nr3d_lib.logger import Logger
from nr3d_lib.utils import tensor_statistics, torch_dtype

from nr3d_lib.models.model_base import ModelMixin
from nr3d_lib.models.annealers import get_annealer
from nr3d_lib.models.fields.nerf import RadianceNet
from nr3d_lib.models.fields.neus import get_neus_var_ctrl
from nr3d_lib.models.fields_conditional.sdf import StyleLoTDSDF
from nr3d_lib.models.fields_conditional.neus.renderer_mixin import NeuSRendererMixinBatched
from nr3d_lib.models.spatial import BatchedBlockSpace

class StyleLoTDNeuS(ModelMixin, nn.Module):
    def __init__(
        self,
        surface_cfg: dict,
        radiance_cfg: dict=None,
        var_ctrl_cfg: dict=dict(ln_inv_s_init=0.3, ln_inv_s_factor=10.0), 
        cos_anneal_cfg: dict=None,
        dtype=torch.half, device=None, use_tcnn_backend=False, 
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
        
        #------- Variance control (manual or learnable)
        self.ctrl_var = get_neus_var_ctrl(**self.var_ctrl_cfg, device=device)

        #------- Surface network
        self.surface_cfg.update(n_latent_dim=n_latent_dim, use_tcnn_backend=self.use_tcnn_backend)
        if aabb is not None:
            self.surface_cfg.update(aabb=aabb)
        if bounding_size is not None:
            self.surface_cfg.update(bounding_size=bounding_size)
        self.implicit_surface = StyleLoTDSDF(**self.surface_cfg, dtype=self.dtype, device=device)
        
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
        else:
            self.radiance_net = None
            self.use_view_dirs = False
            self.use_nablas = False

        #------- (Optional) Cosine value control as in original NeuS repo (for estimating prev/next sdfs)
        self.ctrl_cos_anneal = get_annealer(**self.cos_anneal_cfg) if self.cos_anneal_cfg is not None else None

        # TODO: Just for compatibility. To be removed later.
        self._register_load_state_dict_pre_hook(self.before_load_state_dict)

    def before_load_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        if prefix + 'ln_s' in state_dict:
            log.warning("=> ln_s is deprecated. Already changed to ctrl_var.ln_inv_s for clarification.")
            state_dict[prefix + 'ctrl_var.ln_inv_s'] = state_dict.pop(prefix + 'ln_s')

    @property
    def device(self) -> torch.device:
        return self.implicit_surface.device

    @property
    def space(self):
        return self.implicit_surface.space

    def set_condition(self, z: torch.Tensor):
        assert z.dim()==2
        self.B = z.size(0)
        self.implicit_surface.set_condition(z)

    def clean_condition(self):
        self.implicit_surface.clean_condition()
        self.B = None

    def training_before_per_step(self, cur_it: int, logger: Logger = None):
        self.ctrl_var.set_iter(cur_it)
        if self.ctrl_cos_anneal is not None:
            self.ctrl_cos_anneal.set_iter(cur_it)
        self.implicit_surface.training_before_per_step(cur_it, logger=logger)
    
    def forward_inv_s(self):
        return self.ctrl_var()
    
    def forward_sdf(self, x: torch.Tensor, bidx: torch.Tensor = None):
        return self.implicit_surface.forward(x, bidx)
    def query_sdf(self, x: torch.Tensor, bidx: torch.Tensor = None) -> torch.Tensor:
        return self.implicit_surface.forward(x, bidx)['sdf']
    def forward_sdf_nablas(self, x: torch.Tensor, bidx: torch.Tensor = None, *, has_grad: bool=None, nablas_has_grad: bool=None):
        return self.implicit_surface.forward_sdf_nablas(x, bidx, has_grad=has_grad, nablas_has_grad=nablas_has_grad, max_level=self.max_level)
    def forward(
        self, x: torch.Tensor, *, 
        v: torch.Tensor, bidx: torch.Tensor = None, 
        has_grad:bool=None, nablas_has_grad:bool=None, 
        with_rgb=True, with_normal=True):
        prefix = x.shape[:-1]
        if with_normal or (with_rgb and self.use_nablas):
            raw_ret = self.forward_sdf_nablas(x, bidx, has_grad=has_grad, nablas_has_grad=nablas_has_grad)
        else:
            raw_ret = self.forward_sdf(x, bidx)
        if with_rgb:
            raw_ret.update(
                self.radiance_net(
                    x, 
                    v=v.expand(*prefix, 3) if self.use_view_dirs else None, 
                    # NOTE: `.clamp(-1,1)` is to avoid salt & pepper noise on RGB caused by large dydx
                    n=raw_ret['nablas'].detach().clamp(-1,1) if self.use_nablas else None, 
                    # raw_ret['nablas'] if self.use_nablas else None
                    # F.normalize(raw_ret['nablas'], dim=-1) if self.use_nablas else None
                    # F.normalize(raw_ret['nablas'].detach(), dim=-1) if self.use_nablas else None
                    h_extra=raw_ret['h'])
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
        if with_grad and len(pl:=[p.grad.data.flatten() for p in self.radiance_net.parameters() if p.grad is not None]) > 0:
            ret.update({prefix_ + f'radiance_net.grad.total.{k}': v for k, v in tensor_statistics(torch.cat(pl)).items()})
            ret.update({prefix_ + f"radiance_net.grad.{n}.{k}": v for n, p in self.radiance_net.named_parameters() if p.grad is not None for k, v in tensor_statistics(p.grad.data).items()})
        return ret

    def training_setup(self, training_cfg: Union[Number, dict], name_prefix: str = ''):
        return super().training_setup(training_cfg, name_prefix)

    def training_update_lr(self, it: int):
        return super().training_update_lr(it)

    @torch.no_grad()
    def training_clip_grad(self):
        # SDF model's sepcial clip grad
        self.implicit_surface.training_clip_grad()
        # General `clip_grad_val` or `clip_grad_norm`
        super().training_clip_grad()

    def training_initialize(self, config=dict(), logger=None, log_prefix: str=None):
        return self.implicit_surface.training_initialize(config=config, logger=logger, log_prefix=log_prefix)

from nr3d_lib.models.layers import DenseLayer
from nr3d_lib.models.grid_encodings.lotd import LoTD

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            DenseLayer(32, 32, activation=nn.LeakyReLU(0.2, inplace=True), equal_lr=True),
            DenseLayer(32, 1+3, equal_lr=True),
        )

    def forward(self, x):
        return self.layers(x.to(torch.float16)).to(torch.float32)

class StyleNeuSLXY(ModelMixin, nn.Module):
    def __init__(
        self, path="/mnt/petrelfs/huangqiusheng/exp/multi_waymo/obj.pt", K=None, device=None):
        super().__init__()
        self.encode = LoTD(3, [512], [32], ['NPlaneSum'], dtype=torch.float32, device = device)
        self.decode = Decoder().to('cuda')

        params =  torch.load(path, map_location=device)

        self.encoding = params['encoding']
        self.decode.load_state_dict(params['decoder'])

        # Valid representing space
        self.space = BatchedBlockSpace(bounding_size=2, device = device)
        # 4 x 4
        self.K = torch.Tensor(K, device=device).T if K is not None else None
        self.radiance_net = self.decode
        self.use_view_dirs = False
        self.use_nablas = True
        self.B = 1

    @property
    def device(self) -> torch.device:
        return self.space.device

    def set_condition(self):
        pass

    def clean_condition(self):
        pass

    def training_before_per_step(self, cur_it: int, logger: Logger = None):
        pass

    # @property
    def forward_inv_s(self):
        return 1000

    def transform_points(self, x):
        if self.K is None:
            return x
        x_shape = x.shape
        x = torch.cat([x, torch.ones_like(x[..., :1])], -1) 
        x = x.view(-1, 4)
        x = x.mm(self.K)
        return x[..., :3].view(*x_shape)
    
    def forward_sdf(self, x: torch.Tensor, bidx: torch.Tensor = None):
        x = (self.transform_points(x) + 1) / 2
        h = self.encode(x, self.encoding, bidx=bidx)
        output = self.decode(h)
        sdf = output[..., 0]
        radiance = (output[..., 1:].tanh() + 1) / 2
        return dict(sdf=sdf, h=h, radiance=radiance)

    def query_sdf(self, x: torch.Tensor, bidx: torch.Tensor = None) -> torch.Tensor:
        x = (self.transform_points(x) + 1) / 2
        h = self.encode(x, self.encoding, bidx=bidx)
        output = self.decode(h)
        sdf = output[..., 0]
        return sdf

    def forward_sdf_nablas(self, x: torch.Tensor, bidx: torch.Tensor = None, *, has_grad: bool=None, nablas_has_grad: bool=None):
        # x = x
        has_grad = torch.is_grad_enabled() if has_grad is None else has_grad
        nablas_has_grad = has_grad if nablas_has_grad is None else (nablas_has_grad and has_grad)
        need_dL_dinput = has_grad and x.requires_grad
        x = ((self.transform_points(x) + 1) / 2).requires_grad_(True)
        with torch.enable_grad():
            h, dy_dx = self.encode.forward_dydx(x, self.encoding, bidx=bidx, need_dL_dinput=need_dL_dinput)
            output = self.decode(h)
            sdf = output[..., 0]

        dL_dy = torch.autograd.grad(sdf, h, sdf.new_ones(sdf.shape), retain_graph=has_grad, create_graph=nablas_has_grad, only_inputs=True)[0]

        nablas = self.encode.backward_dydx(dL_dy, dy_dx, x, self.encoding, bidx=bidx)

        radiance = (output[..., 1:].tanh() + 1) / 2

        if not nablas_has_grad:
            nablas = nablas.detach()
        if not has_grad:
            sdf, h = sdf.detach(), h.detach()

        return dict(sdf=sdf, h=h, nablas=nablas, rgb=radiance)

    def forward(
        self, x: torch.Tensor, *, 
        v: torch.Tensor, bidx: torch.Tensor = None, 
        has_grad:bool=None, nablas_has_grad:bool=None, 
        with_rgb=True, with_normal=True):

        if with_normal:
            raw_ret = self.forward_sdf_nablas(x, bidx, has_grad=has_grad, nablas_has_grad=nablas_has_grad)
        else:
            raw_ret = self.forward_sdf(x, bidx)

        return raw_ret

class StyleNeuSLXYModel(NeuSRendererMixinBatched, StyleNeuSLXY):
    """
    MRO:
    -> NeuSRendererMixinBatched
    -> StyleNeuSLXY
    -> ModelMixin
    -> nn.Module
    """
    pass

class StyleLoTDNeuSModel(NeuSRendererMixinBatched, StyleLoTDNeuS):
    """
    MRO:
    -> NeuSRendererMixinBatched
    -> StyleLoTDNeuS
    -> ModelMixin
    -> nn.Module
    """
    pass


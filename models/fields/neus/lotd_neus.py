"""
@file   lotd_neus.py
@author Jianfei Guo, Shanghai AI Lab
@brief  lotd_sdf + radiance_net + neus control
"""

__all__ = [
    'LoTDNeuS', 
    'LoTDNeuSModel'
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

from nr3d_lib.models.base import ModelMixin
from nr3d_lib.models.annealers import get_annealer
from nr3d_lib.models.fields.sdf import LoTDSDF
from nr3d_lib.models.fields.nerf import RadianceNet
from nr3d_lib.models.fields.neus.variance import get_neus_var_ctrl
from nr3d_lib.models.fields.neus.renderer_mixin import NeusRendererMixin

class LoTDNeuS(ModelMixin, nn.Module):
    def __init__(
        self,
        surface_cfg: ConfigDict,
        var_ctrl_cfg: ConfigDict=ConfigDict(ln_inv_s_init=0.3, ln_inv_s_factor=10.0), 
        radiance_cfg: ConfigDict = None,
        cos_anneal_cfg: ConfigDict = None,
        dtype=torch.float, device=torch.device("cuda"), use_tcnn_backend=False
        ) -> None:
        super().__init__()
        self.dtype = torch_dtype(dtype)
        self.device = device
        
        self.surface_cfg = surface_cfg
        self.radiance_cfg = radiance_cfg
        self.use_tcnn_backend = use_tcnn_backend

        self.cos_anneal_cfg = cos_anneal_cfg
        
        self.var_ctrl_cfg = var_ctrl_cfg

        self.max_level = None
        
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
        self.implicit_surface = LoTDSDF(**self.surface_cfg, dtype=self.dtype, device=self.device)
        
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
        # if prefix + 'implicit_surface.encoding.space.aabb' in state_dict:
        #     aabb = state_dict[prefix + 'implicit_surface.encoding.space.aabb']
        #     # Re-populate on loaded aabb -> BUG: Breaks training! "No inf checks were recorded for this optimizer."
        #     LoTDNeuS.populate(self, aabb=aabb)

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

    @profile
    def forward_sdf(self, x: torch.Tensor, *, return_h=False, input_normalized=True):
        return self.implicit_surface(x, return_h=return_h, max_level=self.max_level, input_normalized=input_normalized)
    
    def query_sdf(self, x: torch.Tensor, input_normalized=True) -> torch.Tensor:
        return self.forward_sdf(x, input_normalized=input_normalized)['sdf']
    
    @profile
    def forward_sdf_nablas(
        self,  x: torch.Tensor, *, input_normalized=True, 
        has_grad: bool=None, nablas_has_grad: bool=None, grad_guard=None):
        return self.implicit_surface.forward_sdf_nablas(
            x, input_normalized=input_normalized, 
            has_grad=has_grad, nablas_has_grad=nablas_has_grad, 
            max_level=self.max_level, grad_guard=grad_guard)
    
    @profile
    def forward(
        self, x: torch.Tensor, v: torch.Tensor, *, h_appear_embed: torch.Tensor = None, 
        input_normalized=True, has_grad:bool=None, nablas_has_grad: bool=None, 
        with_rgb=True, with_normal=True):
        prefix = x.shape[:-1]
        if not input_normalized:
            x = self.space.normalize_coords(x)

        if with_normal or (with_rgb and self.radiance_use_nablas):
            raw_ret = self.forward_sdf_nablas(
                x, has_grad=has_grad, nablas_has_grad=nablas_has_grad, input_normalized=True)
        else:
            raw_ret = self.forward_sdf(x, input_normalized=True)
        if with_rgb:
            raw_ret.update(self.radiance_net(
                x, 
                v.expand(*prefix, 3) if self.radiance_use_view_dirs else None, 
                # NOTE: `.clamp(-1,1)` is to avoid salt & pepper noise on RGB caused by large dydx
                raw_ret['nablas'].detach().clamp(-1,1) if self.radiance_use_nablas else None, # opt1
                # raw_ret['nablas'] if self.radiance_use_nablas else None, # opt2
                # F.normalize(raw_ret['nablas'], dim=-1) if self.radiance_use_nablas else None, # opt3
                # F.normalize(raw_ret['nablas'].detach(), dim=-1) if self.radiance_use_nablas else None,  # opt4
                h_extra=raw_ret['h'], 
                h_appear_embed=h_appear_embed.expand(*prefix,-1) if h_appear_embed is not None else None)
            )
        return raw_ret

    def forward_in_obj(
        self, x: torch.Tensor, v: torch.Tensor, *, h_appear_embed: torch.Tensor = None, 
        invalid_sdf: float = 1.1, invalid_color: float = 0, 
        has_grad:bool=None, nablas_has_grad: bool=None, with_rgb=True, with_normal=True):
        """
        Forward using coords from object's coordinates
        """
        net_x = self.space.normalize_coords(x)
        # NOTE: 1e-6 is LoTD's clamp eps. TODO: Make it correspondent.
        valid_i = ((net_x > -1+1e-6) & (net_x < 1-1e-6)).all(dim=-1).nonzero(as_tuple=True)
        
        prefix = x.shape[:-1]
        if valid_i[0].numel() > 0:
            if with_normal or (with_rgb and self.radiance_use_nablas):
                raw_ret: Dict[str, torch.Tensor] = self.forward_sdf_nablas(
                    net_x[valid_i], input_normalized=True, return_h=False, 
                    has_grad=has_grad, nablas_has_grad=nablas_has_grad)
            else:
                raw_ret: Dict[str, torch.Tensor] = self.forward(
                    net_x[valid_i], input_normalized=True, return_h=False)

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
        
        if with_rgb:
            radiances = torch.full([*x.shape[:-1], 3], invalid_color, dtype=torch.float, device=self.device)
            if valid_i[0].numel() > 0:
                raw_ret.update(self.radiance_net(
                    x[valid_i], 
                    v.expand(*prefix, 3)[valid_i] if self.radiance_use_view_dirs else None, 
                    raw_ret['nablas'].detach().clamp(-1,1) if self.radiance_use_nablas else None, 
                    h_extra=raw_ret['h'], h_appear_embed=h_appear_embed))
                radiances.index_put_(valid_i, raw_ret['radiances'])
            ret.update(radiances=radiances)
        
        return ret

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

    @torch.no_grad()
    def custom_grad_clip_step(self):
        self.implicit_surface.custom_grad_clip_step()

    def get_param_group(self, optim_cfg: Union[Number,dict], prefix: str = '') -> List[dict]:
        prefix_ = prefix + ('.' if prefix and not prefix.endswith('.') else '')
        
        # NOTE: This function is just for seperately specifying `betas` for learnable inv_s ctrl
        #       Stable SDF (eikonal) training prefers betas=(0.9,0.99), \
        #           but the original inv_s self-adaptiveness works better with betas=(0.9,0.999)
        
        if (self.ctrl_var is None) \
            or (len(list(self.ctrl_var.parameters())) == 0) \
            or (optim_cfg.get('invs_betas', None) is None):
            return super().get_param_group(optim_cfg, prefix)
        else:
            optim_cfg = deepcopy(optim_cfg)
            invs_betas = optim_cfg.pop('invs_betas', [0.9,0.999])
            invs_optim_cfg = deepcopy(optim_cfg)
            invs_optim_cfg['betas'] = invs_betas
            
            pg_invs = {
                'name': prefix_ + "invs_group", 
                'params': [], 
                **invs_optim_cfg
            }
            
            pg_others = {
                'name': prefix_ + "network_group", 
                'params': [], 
                **optim_cfg
            }
            
            for n, p in self.named_parameters():
                if 'ctrl_var' in n:
                    pg_invs['params'].append(p)
                else:
                    pg_others['params'].append(p)
            
            all_pg = [pg_invs, pg_others]
            return all_pg

class LoTDNeuSModel(NeusRendererMixin, LoTDNeuS):
    """
    MRO: LoTDNeuSModel -> NeusRendererMixin -> LoTDNeuS -> ModelMixin -> nn.Module
    """
    pass

if __name__ == "__main__":
    def unit_test(device=torch.device('cuda')):
        # class TestNet(nn.Module): # NOTE: Not inheriting ModelMixin --- troublesome
        #     def __init__(self) -> None:
        #         super().__init__()
        #         self.encoder = nn.ModuleDict()
        #         self.decoder = nn.ModuleDict()
        #     def populate(self):
        #         print("I have useful populate process !")
        # class TestModel1(NeusRendererMixin, TestNet):
        #     """
        #     MRO: TestModel1 -> NeusRendererMixin -> ModelMixin -> TestNet -> nn.Module
        #     NOTE: This is wrong ! TestNet should inherit from ModelMixin !
        #           This will cause the default empty ModelMixin.populate() overwrites the meaningful TestNet.populate()!
        #     """
        #     def __init__(self) -> None:
        #         super().__init__()
        # class TestModel2(LoTDNeuS, NeusRendererMixin):
        #     """
        #     MRO: TestModel2 -> LoTDNeuS -> NeusRendererMixin -> ModelMixin -> nn.Module
        #     NOTE: This is wrong ! NeusRendererMixin should comes before LoTDNeuS !
        #           This will cause mixin'__init__(),  populate(), etc. never being used (being overwritten by LoTDNeuS).
        #     """
        #     def __init__(self) -> None:
        #         super().__init__()
        # m1 = TestModel1()
        # m1.populate()
        # m2 = TestModel2()
        # m2.populate()
        
        cfg = ConfigDict(
            dtype='half', 
            var_ctrl_cfg=ConfigDict(ctrl_type='learned', ln_inv_s_init=0.1, ln_inv_s_factor=10.0), 
            surface_cfg=ConfigDict(
                bounding_size=2., 
                encoding_cfg=ConfigDict(
                    lotd_auto_compute_cfg=ConfigDict(
                        type='ngp', target_num_params=12*2**20, min_res=16, 
                    ), 
                ), 
                decoder_cfg=ConfigDict(type='mlp', D=1, W=64), 
                n_rgb_used_output=0, 
                geo_init_method='pretrain_after_zero_out'
            ), 
            radiance_cfg=ConfigDict(
                use_pos=True, use_nablas=True, use_view_dirs=True, dir_embed_cfg=ConfigDict(
                    type='spherical', degree=4, 
                ), D=2, W=64
            ), 
            mixin_cfg=ConfigDict(
                accel_cfg=ConfigDict(
                    type='occ_grid', 
                    occ_cfg=ConfigDict(
                        resolution=[64, 64, 64], 
                        occ_val_fn_cfg=ConfigDict(
                            type='sdf', 
                            inv_s=256.0
                        )
                    )
                )
            )
        )
        m = LoTDNeuSModel(**cfg)
        m.populate()
        print(m)
        x = torch.randn([4096, 3], dtype=torch.float, device=device)
        v = torch.randn([4096, 3], dtype=torch.float, device=device)
        out = m.forward(x, v)
        print(out['sdf'].shape, out['radiances'].shape)
                
    unit_test()
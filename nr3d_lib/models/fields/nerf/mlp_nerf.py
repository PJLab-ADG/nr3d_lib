"""
@file   mlp_nerf.py
@author Jianfei Guo, Shanghai AI Lab
@brief  MLP-based vanilla NeRF.
"""

__all__ = [
    'NeRF', 
    'EmbededNeRF', 
    'RadianceNet', 
    'NeRFModel'
]

import math
import numpy as np
from typing import Union

import torch
import torch.nn as nn
from torch import autograd
import torch.nn.functional as F

from nr3d_lib.profile import profile
from nr3d_lib.utils import torch_dtype
from nr3d_lib.config import ConfigDict

from nr3d_lib.models.model_base import ModelMixin
from nr3d_lib.models.layers import get_nonlinearity, DenseLayer
from nr3d_lib.models.blocks import get_blocks
from nr3d_lib.models.embedders import get_embedder
from nr3d_lib.models.layers import DenseLayer, get_nonlinearity
from nr3d_lib.models.spatial import AABBSpace, BatchedBlockSpace
from nr3d_lib.models.fields.nerf.renderer_mixin import NeRFRendererMixin
from nr3d_lib.maths import eval_sh


# Modified from https://github.com/yenchenlin/nerf-pytorch
class NeRF(ModelMixin, nn.Module):
    def __init__(
        self, 
        input_ch_pts=3, input_ch_view=3, use_view_dirs=True, 
        D=8, W=256, skips=[4], activation: str='relu', sigma_activation: str='none', rgb_acitvation: str='sigmoid', 
        space: nn.Module = None, 
        space_cfg: dict = None, 
        dtype=torch.float, device=None, use_tcnn_backend=False):
        super().__init__()
        self.dtype = torch_dtype(dtype)
        self.use_view_dirs = use_view_dirs
        self.pts_blocks = get_blocks(input_ch_pts, W, D=D, W=W, skips=skips, activation=activation, output_activation='none', use_tcnn_backend=use_tcnn_backend, dtype=self.dtype, device=device)
        if self.use_view_dirs:
            # W -> 1
            self.sigma_layer = DenseLayer(W, 1, activation=sigma_activation, dtype=self.dtype, device=device)
            # W + input_ch_view -> W//2 -> W//2 -> 3
            self.rgb_blocks = get_blocks(input_ch_view + W, 3, D=2, W=W//2, activation=activation, output_activation=rgb_acitvation, use_tcnn_backend=use_tcnn_backend, dtype=self.dtype, device=device)
        else:
            self.output_linear = DenseLayer(W, 4, dtype=self.dtype, device=device)
            self.sigma_activation = get_nonlinearity(sigma_activation).nl
            self.rgb_activation = get_nonlinearity(rgb_acitvation).nl

        #------- Valid representing space
        if space is not None:
            # Directly use the externally provided space object definition
            self.space = space
        elif space_cfg is not None:
            space_cfg = space_cfg.copy()
            space_type = space_cfg.pop('type').lower()
            if space_type == 'aabb':
                space = AABBSpace(**space_cfg)
            elif space_type == 'batched': # Batched AABB
                space = BatchedBlockSpace(**space_cfg)
            elif space_type == 'unbounded' or space_type == 'none':
                space = None
            else:
                raise RuntimeError(f"Invalid space_type={space_type}")
            self.space = space
        else:
            # Do not need input space definition and conversion
            self.space = None
    
    def populate(self, bounding_size=None, aabb=None, device=None):
        pass

    @property
    def device(self) -> torch.device:
        return self.pts_blocks.device

    def forward(self, x: torch.Tensor, *, v: torch.Tensor=None, input_normalized=True):
        if not input_normalized:
            x = self.space.normalize_coords(x)
        h = self.pts_blocks(x)
        if self.use_view_dirs:
            sigma = self.sigma_layer(h)
            rgb = self.rgb_blocks(torch.cat([h, v], dim=-1))
        else:
            outputs = self.output_linear(h)
            rgb, sigma = self.rgb_activation(outputs[..., :3]), self.sigma_activation(outputs[..., 3:])
        return dict(rgb=rgb, sigma=sigma.squeeze(-1))
    
    def forward_density(self, x: torch.Tensor, *, input_normalized=True):
        if not input_normalized:
            x = self.space.normalize_coords(x)
        h = self.pts_blocks(x)
        if self.use_view_dirs:
            sigma = self.sigma_layer(h)
        else:
            outputs = self.output_linear(h)
            sigma = self.sigma_activation(outputs[..., 3:])
        return dict(sigma=sigma[..., 0])

class EmbededNeRF(ModelMixin, nn.Module):
    def __init__(
        self,
        input_ch_pts=3, input_ch_view=3, 
        pos_embed_cfg:dict={'type':'sinusoidal', 'n_frequencies': 10},
        use_view_dirs=True, dir_embed_cfg:dict={'type':'spherical', 'degree': 4}, 
        n_geo_embedding=0, geo_embed_cfg:dict={'type':'identity'}, 
        n_appear_embedding=0, appear_embed_cfg:dict={'type':'identity'}, 
        D=8, W=256, skips=[4], activation: str='relu', sigma_activation: str='none', rgb_acitvation: str='sigmoid',
        dtype=torch.float, device=None, use_tcnn_backend=False):
        super().__init__()
        self.dtype = torch_dtype(dtype)
        
        self.use_view_dirs = use_view_dirs
        self.use_geo_embedding = n_geo_embedding > 0
        self.use_h_appear = n_appear_embedding > 0
        
        # x
        self.embed_fn, input_ch_pts = get_embedder(pos_embed_cfg, input_ch_pts)
        # h_geo_embed
        if self.use_geo_embedding:
            self.embed_fn_h_geo, input_ch_h_geo = get_embedder(geo_embed_cfg, n_geo_embedding)
        else:
            input_ch_h_geo = 0
        
        # v
        if self.use_view_dirs:
            self.embed_fn_view, input_ch_views = get_embedder(dir_embed_cfg, input_ch_view)
        else:
            input_ch_views = 0

        # h_appear
        if self.use_h_appear:
            self.embed_fn_appear, input_ch_h_appear = get_embedder(appear_embed_cfg, n_appear_embedding)
        else:
            input_ch_h_appear = 0

        self.nerf_base = NeRF(
            input_ch_pts=(input_ch_pts + input_ch_h_geo), 
            use_view_dirs=(self.use_view_dirs or self.use_geo_embedding), input_ch_view=(input_ch_views + input_ch_h_appear), 
            D=D, W=W, skips=skips, activation=activation, sigma_activation=sigma_activation, rgb_acitvation=rgb_acitvation, 
            dtype=self.dtype, device=device, use_tcnn_backend=use_tcnn_backend)
    
    @property
    def device(self) -> torch.device:
        return self.nerf_base.device
    
    @property
    def space(self):
        return self.nerf_base.space
    
    def populate(self, bounding_size=None, aabb=None, device=None):
        pass

    def forward(
        self, x, *, v: torch.Tensor = None, input_normalized=True, 
        h_geo_embed: torch.Tensor=None, h_appear: torch.Tensor=None):
        if not input_normalized:
            x = self.space.normalize_coords(x)
        # Calculate radiance field
        input_pts = self.embed_fn(x)
        if self.use_geo_embedding:
            input_pts = torch.cat([input_pts, self.embed_fn_h_geo(h_geo_embed)], dim=-1)
        input_views = []        
        if self.use_view_dirs:
            input_views.append(self.embed_fn_view(v))
        if self.use_h_appear:
            input_views.append(self.embed_fn_appear(h_appear))
        input_views = None if len(input_views) == 0 else torch.cat(input_views, dim=-1)
        return self.nerf_base.forward(input_pts, input_views)
    def forward_density(self, x, h_geo_embed: torch.Tensor=None, input_normalized=True):
        if not input_normalized:
            x = self.space.normalize_coords(x)
        input_pts = self.embed_fn(x)
        if self.use_geo_embedding:
            input_pts = torch.cat([input_pts, self.embed_fn_h_geo(h_geo_embed)], dim=-1)
        return self.nerf_base.forward_density(input_pts)

class RadianceNet(ModelMixin, nn.Module):
    def __init__(
        self,
        use_pos=True, pos_dim=3, pos_embed_cfg:dict={'type':'identity'}, 
        use_view_dirs=True, dir_embed_cfg:dict={'type':'identity'}, 
        use_nablas=None, nablas_embed_cfg:dict={'type':'identity'}, 
        n_extra_feat=0, extra_feat_embed_cfg:dict={'type':'identity'},
        n_appear_embedding=0, appear_embed_cfg:dict={'type':'identity'}, 
        D=4, W=256, skips=[], activation='relu', output_activation='sigmoid', 
        
        sh_degs: int = None, sh_degs_anneal_iters: int = 2000,
        dtype=torch.float, device=None, 
        weight_norm=False, use_tcnn_mlp=False, **other_net_cfg):
            
        super().__init__()
        self.dtype = torch_dtype(dtype)
        
        self.skips = skips
        self.D = D
        self.W = W
        
        self.use_pos = use_pos
        self.use_view_dirs = use_view_dirs
        self.use_nablas = self.use_view_dirs if use_nablas is None else use_nablas
        self.use_extra_feat = n_extra_feat > 0
        self.use_h_appear = n_appear_embedding > 0
        self.sh_degs = sh_degs
        self.sh_degs_anneal_iters = sh_degs_anneal_iters
        self.current_sh_degs = sh_degs

        # x
        if self.use_pos:
            self.embed_fn, input_ch_pts = get_embedder(pos_embed_cfg, pos_dim)
        else:
            input_ch_pts = 0
        # v
        if self.use_view_dirs and sh_degs is None:
            self.embed_fn_view, input_ch_views = get_embedder(dir_embed_cfg, 3)
        else:
            input_ch_views = 0
        # n
        if self.use_nablas:
            self.embed_fn_nablas, input_ch_nablas = get_embedder(nablas_embed_cfg, 3)
        else:
            input_ch_nablas = 0
        # h_extra
        if self.use_extra_feat:
            self.embed_fn_h_extra, input_ch_h_extra = get_embedder(extra_feat_embed_cfg, n_extra_feat)
        else:
            input_ch_h_extra = 0
        # h_appear
        if self.use_h_appear:
            self.embed_fn_appear, input_ch_h_appear = get_embedder(appear_embed_cfg, n_appear_embedding)
        else:
            input_ch_h_appear = 0
        
        # [x, v, n, h_extra, h_appear]
        self._uses = (self.use_pos, self.use_view_dirs and sh_degs is None, self.use_nablas, self.use_extra_feat, self.use_h_appear)
        in_dim_0 = input_ch_pts + input_ch_views + input_ch_nablas + input_ch_h_extra + input_ch_h_appear
        out_dim = 3 if sh_degs is None else (sh_degs + 1) ** 2 * 3
        mlp_output_activation = output_activation if sh_degs is None else 'none'
        
        self.blocks = get_blocks(
            in_dim_0, out_dim, 
            D=D,  W=W, skips=skips, activation=activation, output_activation=mlp_output_activation, 
            dtype=self.dtype, device=device, weight_norm=weight_norm, use_tcnn_backend=use_tcnn_mlp, 
            **other_net_cfg)
        if sh_degs is not None:
            self.output_activation = get_nonlinearity(output_activation).nl
        
    
    def get_weight_reg(self, norm_type: float = 2.0):
        return self.blocks.get_weight_reg(norm_type)
    
    def training_before_per_step(self, cur_it: int, logger):
        if self.sh_degs is not None:
            self.current_sh_degs = min(self.sh_degs, cur_it // self.sh_degs_anneal_iters)

    @profile
    def forward(self, x: torch.Tensor, *, v: torch.Tensor=None, n: torch.Tensor=None, 
                h_extra: torch.Tensor=None, h_appear: torch.Tensor=None):
        # Calculate radiance field
        if self.use_pos:
            x = self.embed_fn(x)
        if hasattr(self, "embed_fn_view"):
            v = self.embed_fn_view(v)
        if self.use_nablas:
            n = self.embed_fn_nablas(n)
        if self.use_extra_feat:
            h_extra = self.embed_fn_h_extra(h_extra)
        if self.use_h_appear:
            h_appear = self.embed_fn_appear(h_appear)
        uses = self._uses
        radiance_input = torch.cat( [i for idx, i in enumerate((x, v, n, h_extra, h_appear)) if uses[idx]] , dim=-1)
        rgb = self.blocks(radiance_input)
        if self.sh_degs is not None:
            rgb = eval_sh(
                self.current_sh_degs, 
                rgb.reshape(-1, 3, (self.sh_degs + 1) ** 2), 
                v)
            rgb = self.output_activation(rgb)
        return dict(rgb=rgb)

class NeRFModel(NeRFRendererMixin, EmbededNeRF):
    """
    MRO:
    -> NeRFRendererMixin
    -> EmbededNeRF
    -> ModelMixin
    -> nn.Module
    """
    pass

if __name__ == "__main__":
    def unit_test(device=torch.device('cuda'), batch_size=365365):
        from icecream import ic
        from torch.utils.benchmark import Timer
        nerf = EmbededNeRF(device=device)
        ic(nerf)

        x = torch.randn([batch_size, 3], dtype=torch.float, device=device)
        v = F.normalize(torch.randn([batch_size, 3], dtype=torch.float, device=device), dim=-1)
        out = nerf.forward(x, v)
        sigma, rgb = out['sigma'], out['rgb']
        sigma2 = nerf.forward_density(x)
        
        ic(sigma.shape, sigma.dtype)
        ic(sigma2.shape, sigma2.dtype)
        ic(rgb.shape, rgb.dtype)
        
        with torch.no_grad(): 
            # 18 ms
            print(Timer(
                stmt='nerf.forward(x, v)', 
                globals={'x':x, 'v':v, 'nerf':nerf}
            ).blocked_autorange())
        
    unit_test()
        
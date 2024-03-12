"""
@file   lotd_nerf.py
@author Jianfei Guo, Shanghai AI Lab
@brief  lotd_encoding + simga_decoder + radiance_decoder
"""

__all__ = [
    'LoTDNeRF', 
    'LoTDNeRFModel'
]
from copy import deepcopy
from typing import Callable, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from nr3d_lib.logger import Logger
from nr3d_lib.utils import torch_dtype
from nr3d_lib.profile import profile
from nr3d_lib.maths import trunc_exp

from nr3d_lib.models.model_base import ModelMixin
from nr3d_lib.models.spatial import AABBSpace
from nr3d_lib.models.embedders import get_embedder
from nr3d_lib.models.layers import get_nonlinearity
from nr3d_lib.models.grid_encodings.lotd import LoTDEncoding
from nr3d_lib.models.grid_encodings.multires_decoder import get_multires_decoder
from nr3d_lib.models.fields.nerf.mlp_nerf import RadianceNet
from nr3d_lib.models.fields.nerf.renderer_mixin import NeRFRendererMixin

class LoTDNeRF(ModelMixin, nn.Module):
    def __init__(
        self,

        # Geometry representations
        encoding_cfg=dict(),
        density_decoder_cfg=dict(),
        density_activation: Union[str, Callable] = None, 
        
        extra_pos_embed_cfg: dict = None,

        # Appearance representations
        radiance_decoder_cfg=dict(),
        n_extra_feat_from_output: int = 0,

        dtype=torch.float16, device=None, use_tcnn_backend=False
        ) -> None:
        super().__init__()
        """
        n_extra_feat_from_output: 
            1. sigma decoder output width = 1 + n_extra_feat_from_output
            2. the extra_feat that the radiance net used:
                set to == 0: comes from encoding()
                set to  > 0: comes from output[.., 1:]
        """
        self.dtype = torch_dtype(dtype)
        self.set_device = device
        self.n_extra_feat_from_output = n_extra_feat_from_output
        self.is_extra_feat_from_output = n_extra_feat_from_output > 0
        self.use_extra_embed = extra_pos_embed_cfg is not None
        self.extra_pos_embed_cfg = deepcopy(extra_pos_embed_cfg)
        self.encoding_cfg = deepcopy(encoding_cfg)
        self.density_decoder_cfg = deepcopy(density_decoder_cfg)
        self.density_activation = density_activation
        self.radiance_cfg = deepcopy(radiance_decoder_cfg)
        self.use_tcnn_backend = deepcopy(use_tcnn_backend)

    def populate(self, bounding_size=None, aabb=None, device=None):
        device = device or self.set_device
        self.set_device = device
        
        #------- LoTD encoding
        input_ch = self.encoding_cfg.setdefault("input_ch", 3)
        space_cfg = self.encoding_cfg.setdefault('space_cfg', {'type': 'aabb'})
        if aabb is not None:
            space_cfg.update(aabb=aabb)
        if bounding_size is not None:
            space_cfg.update(bounding_size=bounding_size)
        self.encoding_cfg.update(space_cfg=space_cfg)
        self.encoding = LoTDEncoding(**self.encoding_cfg, dtype=self.dtype, device=device)

        #------- (Optional) extra coords embedding
        if self.use_extra_embed:
            self.extra_pos_embed_cfg.setdefault('use_tcnn_backend', self.use_tcnn_backend)
            self.extra_embed_fn, self.n_extra_embed = get_embedder(self.extra_pos_embed_cfg, input_ch)
        else:
            self.n_extra_embed = 0

        #------- Sigma decoder
        self.density_decoder_cfg.setdefault('use_tcnn_backend', self.use_tcnn_backend)
        if self.is_extra_feat_from_output:
            self.n_extra_feat = self.n_extra_feat_from_output
            self.density_decoder, self.density_decoder_type = get_multires_decoder(
                self.encoding.lod_meta.level_n_feats, (1 + self.n_extra_feat_from_output), n_extra_embed_ch=self.n_extra_embed, 
                **self.density_decoder_cfg, dtype=self.dtype, device=device)
        else:
            self.n_extra_feat = self.encoding.out_features
            self.density_decoder, self.density_decoder_type = get_multires_decoder(
                self.encoding.lod_meta.level_n_feats, 1, n_extra_embed_ch=self.n_extra_embed, 
                **self.density_decoder_cfg, dtype=self.dtype, device=device)

        if isinstance(self.density_activation, (str, dict)):
            self.density_activation = get_nonlinearity(self.density_activation).nl
        elif self.density_activation is None:
            self.density_activation = nn.Identity()

        #------- Radiance decoder
        self.radiance_cfg['n_extra_feat'] = self.n_extra_feat
        if self.use_tcnn_backend:
            from nr3d_lib.models.fields.nerf import TcnnRadianceNet
            self.rgb_decoder = TcnnRadianceNet(**self.radiance_cfg, device=device, dtype=self.dtype)
        else:
            self.rgb_decoder = RadianceNet(pos_dim=input_ch, **self.radiance_cfg, device=device, dtype=self.dtype)

    @property
    def device(self) -> torch.device:
        return self.encoding.device

    @property
    def space(self) -> AABBSpace:
        return self.encoding.space

    # def before_load_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    #     if prefix + 'encoding.space.aabb' in state_dict:
    #         aabb = state_dict[prefix + 'encoding.space.aabb']
    #         # Re-populate on loaded aabb
    #         # BUG: Breaks training! "No inf checks were recorded for this optimizer."
    #         #      Re-populate => re-register parameters => breaks parameter groups gathered right after module initialization
    #         LoTDNeRF.populate(self, aabb=aabb)

    def training_before_per_step(self, cur_it: int, logger: Logger = None):
        self.encoding.set_anneal_iter(cur_it)

    @profile
    def forward(
        self, x: torch.Tensor, *, v: torch.Tensor = None, 
        h_appear: torch.Tensor = None, input_normalized=True, ):
        if not input_normalized:
            x = self.space.normalize_coords(x)
        # NOTE: x must be in range [-1,1]
        h = self.encoding(x)
        if self.use_extra_embed:
            h_embed = self.extra_embed_fn(x)
            output = self.density_decoder(torch.cat([h, h_embed.to(h.dtype)], dim=-1))
        else:
            output = self.density_decoder(h)
        sigma = self.density_activation(output[..., 0])
        if self.is_extra_feat_from_output:
            rgb = self.rgb_decoder.forward(x, v=v, n=None, h_extra=output[..., 1:], h_appear=h_appear)['rgb']
        else:
            rgb = self.rgb_decoder(x, v=v, n=None, h_extra=h, h_appear=h_appear)['rgb']
        return dict(sigma=sigma, rgb=rgb)

    @profile
    def forward_density(self, x: torch.Tensor, input_normalized=True):
        if not input_normalized:
            x = self.space.normalize_coords(x)
        # NOTE: x must be in range [-1,1]
        h = self.encoding(x)
        if self.use_extra_embed:
            h_embed = self.extra_embed_fn(x)
            output = self.density_decoder(torch.cat([h, h_embed.to(h.dtype)], dim=-1))
        else:
            output = self.density_decoder(h)
        sigma = self.density_activation(output[..., 0])
        return dict(sigma=sigma)
    
    @torch.no_grad
    def query_density(self, x: torch.Tensor):
        # NOTE: x must be in range [-1,1]
        h = self.encoding(x)
        if self.use_extra_embed:
            h_embed = self.extra_embed_fn(x)
            output = self.density_decoder(torch.cat([h, h_embed.to(h.dtype)], dim=-1))
        else:
            output = self.density_decoder(h)
        sigma = self.density_activation(output[..., 0])
        return sigma
    
    def get_weight_reg(self, norm_type: float = 2.0, alpha_sigma: float = 1.0, alpha_rgb: float = 1.0) -> torch.Tensor:
        return torch.cat([alpha_sigma * self.density_decoder.get_weight_reg(norm_type).to(self.device), 
                          alpha_rgb * self.rgb_decoder.get_weight_reg(norm_type).to(self.device)])

    def get_color_lipshitz_bound(self) -> torch.Tensor:
        return self.rgb_decoder.blocks.lipshitz_bound_full()

    @torch.no_grad()
    def rescale_volume(self, new_aabb: torch.Tensor):
        return self.encoding.rescale_volume(new_aabb)

class LoTDNeRFModel(NeRFRendererMixin, LoTDNeRF):
    """
    MRO:
    -> NeRFRendererMixin
    -> LoTDNeRF
    -> ModelMixin
    -> nn.Module
    """
    use_view_dirs = True
    fwd_density_use_view_dirs = False
    pass

if __name__ == "__main__":
    def unit_test(device=torch.device('cuda'), batch_size=365365):
        from icecream import ic
        from torch.utils.benchmark import Timer
        encoding_cfg = {
            'lod_res': [34, 55, 90, 140, 230, 370, 600, 1000, 1600, 2600, 4200],
            'lod_n_feats': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            'lod_types': ['Dense', 'Dense', 'VM', 'VM', 'VM', 'VM', 'VM', 'VM', 'VM', 'CPfast', 'CPfast']
        }
        density_decoder_cfg = dict(
            target='nr3d_lib.models.blocks.get_blocks',
            param=dict(
                D=1, W=64,
            )
        )
        radiance_decoder_cfg = dict(
            use_pos=False,
            use_view_dirs=True, dir_embed_cfg={'type': 'spherical', 'degree': 4},
            use_nablas=False,
            D=2, W=64, activation='relu', output_activation='sigmoid'
        )
        nerf = LoTDNeRF(
            encoding_cfg, density_decoder_cfg, radiance_decoder_cfg,
            n_extra_feat_from_output=16, use_tcnn_backend=True)
        ic(nerf)

        x = torch.rand([batch_size, 3], device=device, dtype=torch.float) * 2 - 1
        v = F.normalize(torch.randn([batch_size, 3], device=device, dtype=torch.float), dim=-1)
        out = nerf.forward(x, v)

        # 4.89 ms
        print(Timer(
            stmt='nerf.forward(x, v)',
            globals={'nerf': nerf, 'x': x, 'v': v}
        ).blocked_autorange())

    unit_test()

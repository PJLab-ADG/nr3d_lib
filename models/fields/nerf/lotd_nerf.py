"""
@file   lotd_nerf.py
@author Jianfei Guo, Shanghai AI Lab
@brief  lotd_encoding + simga_decoder + radiance_decoder
"""

__all__ = [
    'LoTDNeRF', 
    'LoTDNeRFModel'
]

import torch
import torch.nn as nn
import torch.nn.functional as F

from nr3d_lib.logger import Logger
from nr3d_lib.utils import torch_dtype
from nr3d_lib.config import ConfigDict
from nr3d_lib.profile import profile

from nr3d_lib.models.base import ModelMixin
from nr3d_lib.models.spatial import AABBSpace
from nr3d_lib.models.embedders import get_embedder
from nr3d_lib.models.fields.nerf.nerf import RadianceNet
from nr3d_lib.models.grids.lotd import LoTDEncoding, get_lotd_decoder
from nr3d_lib.models.fields.nerf.renderer_mixin import NeRFRendererMixin

class LoTDNeRF(ModelMixin, nn.Module):
    def __init__(
        self,

        # Geometry representations
        encoding_cfg=ConfigDict(),
        sigma_decoder_cfg=ConfigDict(),
        extra_pos_embed_cfg: dict = None,

        # Appearance representations
        radiance_decoder_cfg=ConfigDict(),
        n_rgb_used_output: int = 0,

        dtype=torch.float16, device=torch.device("cuda"), use_tcnn_backend=False
        ) -> None:
        super().__init__()
        """
        n_rgb_used_output: 
            1. sigma decoder output width = 1 + n_rgb_used_output
            2. the extra_feat that the radiance net used:
                set to == 0: comes from encoding()
                set to  > 0: comes from output[.., 1:]
        """
        self.dtype = torch_dtype(dtype)
        self.device = device
        self.n_rgb_used_output = n_rgb_used_output
        self.is_extrafeat_from_output = n_rgb_used_output > 0
        self.use_extra_embed = extra_pos_embed_cfg is not None
        self.extra_pos_embed_cfg = extra_pos_embed_cfg
        self.encoding_cfg = encoding_cfg
        self.sigma_decoder_cfg = sigma_decoder_cfg
        self.radiance_cfg = radiance_decoder_cfg
        self.use_tcnn_backend = use_tcnn_backend

    def populate(self, bounding_size=None, aabb=None, dtype=None, device=None):
        if dtype is not None:
            self.dtype = dtype
        if device is not None:
            self.device = device
        
        #------- LoTD encoding
        input_ch = self.encoding_cfg.setdefault("input_ch", 3)
        if aabb is not None:
            self.encoding_cfg.update(aabb=aabb)
        if bounding_size is not None:
            self.encoding_cfg.update(bounding_size=bounding_size)
        self.encoding = LoTDEncoding(**self.encoding_cfg, dtype=self.dtype, device=self.device)

        #------- (Optional) extra coords embedding
        if self.use_extra_embed:
            self.extra_pos_embed_cfg.setdefault('use_tcnn_backend', self.use_tcnn_backend)
            self.extra_embed_fn, self.n_extra_embed = get_embedder(self.extra_pos_embed_cfg, input_ch)
        else:
            self.n_extra_embed = 0

        #------- Sigma decoder
        self.sigma_decoder_cfg.setdefault('use_tcnn_backend', self.use_tcnn_backend)
        if self.is_extrafeat_from_output:
            self.n_rgb_used_extrafeat = self.n_rgb_used_output
            self.sigma_decoder, self.sigma_decoder_type = get_lotd_decoder(
                self.encoding.lod_meta, (1 + self.n_rgb_used_output), n_extra_embed_ch=self.n_extra_embed, 
                **self.sigma_decoder_cfg, dtype=self.dtype, device=self.device)
        else:
            self.n_rgb_used_extrafeat = self.encoding.out_features
            self.sigma_decoder, self.sigma_decoder_type = get_lotd_decoder(
                self.encoding.lod_meta, 1, n_extra_embed_ch=self.n_extra_embed, 
                **self.sigma_decoder_cfg, dtype=self.dtype, device=self.device)

        #------- Radiance decoder
        self.radiance_cfg['n_rgb_used_extrafeat'] = self.n_rgb_used_extrafeat
        if self.use_tcnn_backend:
            from nr3d_lib.models.fields.nerf import TcnnRadianceNet
            self.rgb_decoder = TcnnRadianceNet(**self.radiance_cfg, device=self.device, dtype=self.dtype)
        else:
            self.rgb_decoder = RadianceNet(pos_dim=input_ch, **self.radiance_cfg, device=self.device, dtype=self.dtype)

    @property
    def space(self) -> AABBSpace:
        return self.encoding.space

    # def load_state_dict_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    #     if prefix + 'encoding.space.aabb' in state_dict:
    #         aabb = state_dict[prefix + 'encoding.space.aabb']
    #         # Re-populate on loaded aabb
    #         # BUG: Breaks training! "No inf checks were recorded for this optimizer."
    #         #      Re-populate => re-register parameters => breaks parameter groups gathered right after module initialization
    #         LoTDNeRF.populate(self, aabb=aabb)

    def preprocess_per_train_step(self, cur_it: int, logger: Logger = None):
        self.encoding.set_anneal_iter(cur_it)

    @profile
    def forward(
        self, x: torch.Tensor, v: torch.Tensor = None, *, input_normalized=True, 
        h_appear_embed: torch.Tensor = None):
        if not input_normalized:
            x = self.space.normalize_coords(x)
        # NOTE: x must be in range [-1,1]
        h = self.encoding(x)
        if self.use_extra_embed:
            h_embed = self.extra_embed_fn(x)
            output = self.sigma_decoder(torch.cat([h, h_embed.to(h.dtype)], dim=-1))
        else:
            output = self.sigma_decoder(h)
        sigma = output[..., 0]
        if self.is_extrafeat_from_output:
            rgb = self.rgb_decoder(x, v, None, h_extra=output[..., 1:], h_appear_embed=h_appear_embed)['radiances']
        else:
            rgb = self.rgb_decoder(x, v, None, h_extra=h, h_appear_embed=h_appear_embed)['radiances']
        return dict(sigma=sigma, radiances=rgb)

    @profile
    def forward_sigma(self, x: torch.Tensor, input_normalized=True):
        if not input_normalized:
            x = self.space.normalize_coords(x)
        # NOTE: x must be in range [-1,1]
        h = self.encoding(x)
        if self.use_extra_embed:
            h_embed = self.extra_embed_fn(x)
            output = self.sigma_decoder(torch.cat([h, h_embed.to(h.dtype)], dim=-1))
        else:
            output = self.sigma_decoder(h)
        return dict(sigma=output[..., 0])
    
    def get_weight_reg(self, norm_type: float = 2.0, alpha_sigma: float = 1.0, alpha_rgb: float = 1.0) -> torch.Tensor:
        return torch.cat([alpha_sigma * self.sigma_decoder.get_weight_reg(norm_type).to(self.device), 
                          alpha_rgb * self.rgb_decoder.get_weight_reg(norm_type).to(self.device)])

    def get_color_lipshitz_bound(self) -> torch.Tensor:
        return self.rgb_decoder.blocks.lipshitz_bound_full()

    @torch.no_grad()
    def rescale_volume(self, new_aabb: torch.Tensor):
        return self.encoding.rescale_volume(new_aabb)

class LoTDNeRFModel(NeRFRendererMixin, LoTDNeRF):
    """
    MRO: NeRFRendererMixin -> LoTDNeRF -> ModelMixin -> nn.Module
    """
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
        sigma_decoder_cfg = ConfigDict(
            target='nr3d_lib.models.blocks.get_blocks',
            param=ConfigDict(
                D=1, W=64,
            )
        )
        radiance_decoder_cfg = ConfigDict(
            use_pos=False,
            use_view_dirs=True, dir_embed_cfg={'type': 'spherical', 'degree': 4},
            use_nablas=False,
            D=2, W=64, activation='relu', output_activation='sigmoid'
        )
        nerf = LoTDNeRF(
            encoding_cfg, sigma_decoder_cfg, radiance_decoder_cfg,
            n_rgb_used_output=16, use_tcnn_backend=True)
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

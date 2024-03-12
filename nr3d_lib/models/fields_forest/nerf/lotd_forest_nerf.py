"""
@file   lotd_forest_nerf.py
@author Jianfei Guo, Shanghai AI Lab
@brief  LoTD-Forest Encoidng + sigma decoder + radiance decoder
"""

__all__ = [
    'LoTDForestNeRF'
]

import torch
import torch.nn as nn

from nr3d_lib.logger import Logger
from nr3d_lib.utils import torch_dtype
from nr3d_lib.config import ConfigDict

from nr3d_lib.models.model_base import ModelMixin
from nr3d_lib.models.grid_encodings.lotd import LoTDForestEncoding
from nr3d_lib.models.grid_encodings.multires_decoder import get_multires_decoder
from nr3d_lib.models.fields.nerf import RadianceNet

class LoTDForestNeRF(ModelMixin, nn.Module):
    def __init__(
        self,
        
        encoding_cfg: dict, 
        density_decoder_cfg: dict, 
        
        radiance_decoder_cfg: dict, 
        n_extra_feat_from_output: int = 0,
        
        dtype=torch.half, device=None, use_tcnn_backend=False, 
        ) -> None:
        """_summary_

        Args:
            encoding_cfg (dict): _description_
            density_decoder_cfg (dict): _description_
            radiance_decoder_cfg (dict): _description_
            n_extra_feat_from_output (int, optional): \
                Determines whether downstream networks (e.g., RGB decoders or semantic decoders) \
                use the direct encoding output or extra output channels from the decoder. \
                If set to 0: downstream networks use the encoded features directly. \
                    The SDF decoder outputs 1-channel (SDF).
                If set to >0: downstream networks use the additional output from the SDF decoder (output[.., 1:]). \
                    The SDF decoder outputs (1+`n_extra_feat_from_output`)-channel (SDF and feature). \
                Default is 0.
            dtype (_type_, optional): _description_. Defaults to torch.half.
            device (_type_, optional): _description_. Defaults to None.
            use_tcnn_backend (bool, optional): _description_. Defaults to False.
        """
        
        super().__init__()
        
        self.dtype = torch_dtype(dtype)

        # LoTD encoding
        self.encoding = LoTDForestEncoding(**encoding_cfg)

        # Sigma decoder
        density_decoder_cfg.param['use_tcnn_backend'] = use_tcnn_backend
        if self.is_extra_feat_from_output:
            self.n_extra_feat = n_extra_feat_from_output
            self.density_decoder, self.density_decoder_type = get_multires_decoder(self.encoding.lod_meta.level_n_feats, (1+n_extra_feat_from_output), **density_decoder_cfg, dtype=self.dtype, device=device)
        else:
            self.n_extra_feat = self.encoding.out_features
            self.density_decoder, self.density_decoder_type = get_multires_decoder(self.encoding.lod_meta.level_n_feats, 1, **density_decoder_cfg, dtype=self.dtype, device=device)
        
        # Radiance decoder
        radiance_decoder_cfg['n_extra_feat'] = self.n_extra_feat
        if use_tcnn_backend:
            from nr3d_lib.models.fields.nerf import TcnnRadianceNet
            self.rgb_decoder = TcnnRadianceNet(**radiance_decoder_cfg, device=device, dtype=self.dtype)
        else:
            self.rgb_decoder = RadianceNet(**radiance_decoder_cfg, device=device, dtype=self.dtype)

    @property
    def device(self) -> torch.device:
        return self.encoding.device

    @property
    def space(self):
        return self.encoding.space

    def training_before_per_step(self, cur_it: int, logger: Logger = None):
        self.encoding.set_anneal_iter(cur_it)

    def forward(
        self, x: torch.Tensor, *, 
        v: torch.Tensor = None, h_appear: torch.Tensor = None, block_inds: torch.Tensor = None, block_offsets: torch.Tensor = None, 
        input_normalized=False):
        
        if not input_normalized:
            block_x, block_inds = self.space.normalize_coords(x, block_inds)
        else:
            block_x = x
        
        # NOTE: block_x must be in range [-1,1]
        h = self.encoding(block_x, block_inds, block_offsets)
        
        output = self.density_decoder(h)
        sigma = output[..., 0]
        if self.is_extra_feat_from_output:
            rgb = self.rgb_decoder(x, v=v, n=None, h_extra=output[..., 1:], h_appear=h_appear)['rgb']
        else:
            rgb = self.rgb_decoder(x, v=v, n=None, h_extra=h, h_appear=h_appear)['rgb']
        return dict(sigma=sigma, rgb=rgb)

    def forward_density(
        self, x: torch.Tensor, block_inds: torch.Tensor = None, block_offsets: torch.Tensor = None, 
        *, input_normalized=False):
        
        if not input_normalized:
            block_x, block_inds = self.space.normalize_coords(x, block_inds)
        else:
            block_x = x
        
        # NOTE: x must be in range [-1,1]
        h = self.encoding(block_x, block_inds, block_offsets)
        
        output = self.density_decoder(h)
        return dict(sigma=output[..., 0])
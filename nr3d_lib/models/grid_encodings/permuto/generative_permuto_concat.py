"""
@file   generative_permuto_concat.py
@author Jianfei Guo, Shanghai AI Lab
@brief  A minimum generative model characterized by one layer of high-dimensional \
    permutohedral-lattice-based encodings and a basic multi-res feature decoder.
"""

__all__ = [
    'GenerativePermutoConcat', 
    'GenerativeMLLConcat', 
]

import numpy as np
from typing import Dict, Literal

import torch
import torch.nn as nn

from nr3d_lib.logger import Logger
from nr3d_lib.profile import profile
from nr3d_lib.config import ConfigDict
from nr3d_lib.utils import  torch_dtype

from nr3d_lib.models.embedders import get_embedder
from nr3d_lib.models.grid_encodings.multires_decoder import get_multires_decoder
from nr3d_lib.models.grid_encodings.permuto.permuto_encoding import PermutoEncoding
from nr3d_lib.models.grid_encodings.permuto.mll import MLLNet

class GenerativePermutoConcat(nn.Module):
    def __init__(
        self, 
        n_latent_dim: int, 
        n_input_dim: int,
        n_output_dim: int, 
        encoding_cfg: dict = ...,
        decoder_cfg: dict = ...,
        extra_pos_embed_cfg: dict = None,
        dtype=torch.half, device=None, use_tcnn_backend=False, 
        ) -> None:
        super().__init__()
        
        self.dtype = torch_dtype(dtype)
        
        assert n_input_dim in [2,3], f"{self.__class__.__name__} expects only 2D/3D input, but got n_input_dim={n_input_dim}."
        self.n_input_dim = n_input_dim
        self.n_latent_dim = n_latent_dim
        self.n_output_dim = n_output_dim
        self.encoding_cfg = encoding_cfg
        self.decoder_cfg = decoder_cfg
        self.use_tcnn_backend = use_tcnn_backend
        self.use_extra_embed = extra_pos_embed_cfg is not None
        
        #------- Permutohedral encoding
        self.encoding = PermutoEncoding(
            self.n_input_dim + self.n_latent_dim, 
            **self.encoding_cfg, dtype=self.dtype, device=device, space_type='unbounded')
        
        #------- (Optional) extra coords embedding concated to the encoding output before feeding into decoder
        if self.use_extra_embed:
            extra_pos_embed_cfg.setdefault('use_tcnn_backend', use_tcnn_backend)
            self.extra_embed_fn, self.n_extra_embed = get_embedder(extra_pos_embed_cfg, self.n_input_dim)
        else:
            self.n_extra_embed = 0
        
        #------- Decoder
        self.decoder_cfg.setdefault('use_tcnn_backend', self.use_tcnn_backend)
        self.decoder, self.decoder_type = get_multires_decoder(
            self.encoding.meta.level_n_feats, n_output_dim, 
            n_extra_embed_ch=self.n_extra_embed ,**self.decoder_cfg, dtype=self.dtype, device=device)
    
    @property
    def space(self):
        return self.encoding.space
    
    @property
    def device(self) -> torch.device:
        return self.encoding.device

    def training_before_per_step(self, cur_it: int, logger: Logger = None):
        self.encoding.set_anneal_iter(cur_it)

    def set_condition(self, z: torch.Tensor):
        assert z.size(-1) == self.n_latent_dim, f"The input latents should have data_dim={self.n_latent_dim}, but got {z.size(-1)}"
        self.z_per_batch = z
        self.B = z.size(0)
    
    def clean_condition(self):
        self.z_per_batch = None
        self.B = None
    
    @profile
    def forward(
        self, x: torch.Tensor, *, z: torch.Tensor = None, 
        z_per_batch: torch.Tensor = None, bidx: torch.Tensor = None, 
        return_h=False, max_level: int = None) -> Dict[str, torch.Tensor]:
        """
        Supported combinations of inputs:
        - Given [x (,z_per_batch)]:
            Batched `x` of shape [B, ..., n_input_dim], where B is the batch_size of `z_per_batch` (if given) or `self.z_per_batch`
        - Given [x, bidx (,z_per_batch)]
            `bidx` specifies the batch ind from `z_per_batch` or `self.z_per_batch` each `x` corresponds to.
        - Given [x, z]:
            `z` specifies each `z_per_batch` that each `x` corresponds to.
        """
        #---- Prepare x input
        assert x.size(-1) == self.n_input_dim, f"`x` should be of shape [...,{self.n_input_dim}], but got {[*x.shape]}"
        
        #---- Prepare per-x z input
        if z is not None:
            assert [*z.shape[:-1]] == [*x.shape[:-1]], f"`z` should be of shape [{','.join([str(s) for s in x.shape[:-1]])},...]"
        else:
            z_per_batch = self.z_per_batch if z_per_batch is None else z_per_batch
            if bidx is not None:
                assert [*bidx.shape] == [*x.shape[:-1]], f"`bidx` should be of shape [{','.join([str(s) for s in x.shape[:-1]])}]"
                z = z_per_batch[bidx]
            else:
                assert x.size(0) == self.B, f"The input should have batch_size={self.B} when `bidx` and `z` are not given."
                z = z_per_batch.view(self.B,*[1]*(x.dim()-2),self.n_latent_dim).expand(*x.shape[:-1], -1)

        h = self.encoding(torch.cat((x, z), dim=-1), max_level=max_level)
        if not self.use_extra_embed:
            output = self.decoder(h)
        else:
            output = self.decoder(torch.cat([h, self.extra_embed_fn(x).to(h.dtype)], dim=-1))
        
        if return_h:
            return dict(output=output, h=h)
        else:
            return dict(output=output)

    def get_weight_reg(self, norm_type: float = 2.0):
        return self.decoder.get_weight_reg(norm_type)

class GenerativeMLLConcat(nn.Module):
    def __init__(
        self, 
        n_latent_dim: int, 
        n_input_dim: int,
        n_output_dim: int, 
        mll_cfg: dict = ...,
        dtype=torch.half, device=None, use_tcnn_backend=False, 
        ) -> None:
        super().__init__()
        
        self.dtype = torch_dtype(dtype)
        
        assert n_input_dim in [2,3], f"{self.__class__.__name__} expects only 2D/3D input, but got n_input_dim={n_input_dim}."
        self.n_input_dim = n_input_dim
        self.n_latent_dim = n_latent_dim
        self.n_output_dim = n_output_dim
        self.mll_cfg = mll_cfg
        self.use_tcnn_backend = use_tcnn_backend
        
        #------- MLL
        self.mll = MLLNet(
            self.n_input_dim + self.n_latent_dim, self.n_output_dim, 
            **self.mll_cfg, dtype=self.dtype, device=device)

    @property
    def space(self):
        return self.mll.space

    @property
    def device(self) -> torch.device:
        return self.mll.device

    def training_before_per_step(self, cur_it: int, logger: Logger = None):
        self.mll.set_anneal_iter(cur_it)

    def set_condition(self, z: torch.Tensor):
        assert z.size(-1) == self.n_latent_dim, f"The input latents should have data_dim={self.n_latent_dim}, but got {z.size(-1)}"
        self.z_per_batch = z
        self.B = z.size(0)
    
    def clean_condition(self):
        self.z_per_batch = None
        self.B = None
    
    @profile
    def forward(
        self, x: torch.Tensor, *, z: torch.Tensor = None, 
        z_per_batch: torch.Tensor = None, bidx: torch.Tensor = None, 
        return_h=False, max_level: int = None) -> Dict[str, torch.Tensor]:
        """
        Supported combinations of inputs:
        - Given [x (,z_per_batch)]:
            Batched `x` of shape [B, ..., n_input_dim], where B is the batch_size of `z_per_batch` (if given) or `self.z_per_batch`
        - Given [x, bidx (,z_per_batch)]
            `bidx` specifies the batch ind from `z_per_batch` or `self.z_per_batch` each `x` corresponds to.
        - Given [x, z]:
            `z` specifies each `z_per_batch` that each `x` corresponds to.
        """
        #---- Prepare x input
        assert x.size(-1) == self.n_input_dim, f"`x` should be of shape [...,{self.n_input_dim}], but got {[*x.shape]}"

        #---- Prepare per-x z input
        if z is not None:
            assert [*z.shape[:-1]] == [*x.shape[:-1]], f"`z` should be of shape [{','.join([str(s) for s in x.shape[:-1]])},...]"
        else:
            z_per_batch = self.z_per_batch if z_per_batch is None else z_per_batch
            if bidx is not None:
                assert [*bidx.shape] == [*x.shape[:-1]], f"`bidx` should be of shape [{','.join([str(s) for s in x.shape[:-1]])}]"
                z = z_per_batch[bidx]
            else:
                assert x.size(0) == self.B, f"The input should have batch_size={self.B} when `bidx` and `z` are not given."
                z = z_per_batch.view(self.B,*[1]*(x.dim()-2),self.n_latent_dim).expand(*x.shape[:-1], -1)

        return self.mll(torch.cat((x, z), dim=-1), max_level=max_level, return_h=return_h)

    def get_weight_reg(self, norm_type: float = 2.0):
        return self.mll.get_weight_reg(norm_type)
"""
@file   multires_decoder.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Common decoders for multi-resolution encoded features
"""

__all__ = [
    'get_multires_decoder'
]

import numpy as np
from typing import List, Literal

import torch
import torch.nn as nn

from nr3d_lib.utils import torch_dtype
from nr3d_lib.models.layers import DenseLayer
from nr3d_lib.models.blocks import get_blocks

"""
Supported decoder types
- direct reduce (sum, mean)
- selected direct reduce (sum, mean)
- linear
- selected linear reduce
- mlp
- selected mlp reduce
TODO:
- lod-residual direct / linear / mlp
- selected lod-residual direct / linear / mlp
"""

def get_multires_decoder(
    level_n_feats: List[int], out_features: int, *, 
    type: str = 'mlp', n_extra_embed_ch: int=0, 
    level_select_n_feats: int = None, select_n_levels: int = None, 
    dtype=torch.half, device=None, **params):
    """_summary_

    Args:
        level_n_feats (List[int]): Number of input feature channel per resolution level.
        out_features (int): Output width of decoder.
        type (str, optional): The type of the decoder network. Defaults to 'mlp'.
        n_extra_embed_ch (int, optional): Number of channels of extra embeddings feeded into the decoder. Defaults to 0.
        level_select_n_feats (int, optional): Optional, should give at most one of [`level_select_n_feats`, `select_n_levels`]. 
            Number of selected features per resolution level. Defaults to None.
        select_n_levels (int, optional): Optional, should give at most one of [`level_select_n_feats`, `select_n_levels`]. 
            Number of selected levels. Defaults to None.
        dtype (Any, optional): The network parameters' dtype. Defaults to torch.float16.
        device (Any, optional): The network parameters' device. Defaults to torch.device('cuda').

    Raises:
        NotImplementedError: _description_
        RuntimeError: _description_

    Returns:
        _type_: _description_
    """
    
    dtype = torch_dtype(dtype)
    if (level_select_n_feats is None) and (select_n_levels is None):
        selector = None
        in_features = sum(level_n_feats) + n_extra_embed_ch
    else:
        selector = MultiresSelect(
            level_n_feats, device=device, 
            level_select_n_feats=level_select_n_feats, 
            select_n_levels=select_n_levels, 
            n_extra_embed_ch=n_extra_embed_ch)
        in_features = selector.out_features
    
    if type == 'sum' or type == 'mean':
        assert out_features == 1, f'type={type} only supports out_feautres==1'
        dec = get_direct_reduction(type)
        dec.dtype = dtype
        dec.device = device
    elif type == 'linear':
        params.pop('use_tcnn_backend', None)
        dec = DenseLayer(in_features, out_features, dtype=dtype, device=device, **params)
    elif type == 'mlp':
        dec = get_blocks(in_features, out_features, dtype=dtype, device=device, **params)
    elif type == 'residual':
        # TODO: Need to re-implement select for residual
        raise NotImplementedError
    else:
        raise RuntimeError(f"Invalid type={type}")

    if selector is not None:
        comp = nn.Sequential(selector, dec)
        comp.dtype = dtype
        comp.device = device
        comp.get_weight_reg = dec.get_weight_reg
        dec = comp
    return dec, type

class MultiresSelect(nn.Module):
    def __init__(
        self, 
        level_n_feats: List[int], 
        level_select_n_feats: int = None, 
        select_n_levels: int = None, 
        n_extra_embed_ch: int = 0, 
        device=None) -> None:
        super().__init__()
        
        assert bool(level_select_n_feats is not None) != bool(select_n_levels is not None), \
            "Expect one of `level_select_n_feats` or `select_n_levels`"
        
        self.n_extra_embed_ch = n_extra_embed_ch
        
        exclusive_sum = np.cumsum([0,*level_n_feats])
        if level_select_n_feats is not None:
            feat_inds = []
            feat_n = []
            for l, n in enumerate(level_n_feats):
                level_n = min(level_select_n_feats, n)
                inds = exclusive_sum[l] + torch.arange(level_n)
                feat_n.append(feat_n)
                feat_inds.append(inds)
            feat_inds = torch.cat(feat_inds).to(device=device, dtype=torch.long)
        else:
            feat_inds = torch.arange(exclusive_sum[select_n_levels], dtype=torch.long, device=device)
        
        self.register_buffer('feat_inds', feat_inds, persistent=True)
        # self.selected_level_n_feats = feat_n
        self.in_fearues = sum(level_n_feats) + self.n_extra_embed_ch
        self.out_features = len(feat_inds) + self.n_extra_embed_ch
    
    @property
    def device(self) -> torch.device:
        return self.feat_inds.device
    
    def forward(self, h: torch.Tensor):
        assert h.size(-1) == self.in_fearues, f"Input should have size of [..., {self.in_fearues}], current is {[*h.shape]}"
        if (n_extra:=self.n_extra_embed_ch) > 0:
            # The input order should be [encoding, extra_embed]
            return torch.cat([h[..., self.feat_inds], h[..., -n_extra:]], dim=-1)
        else:
            return h[..., self.feat_inds]

def get_direct_reduction(reduce: Literal['sum', 'mean']):
    class DirectReduction(nn.Module):
        def sum(self, h: torch.Tensor):
            return h.sum(dim=-1, keepdim=True)
        def mean(self, h: torch.Tensor):
            return h.mean(dim=-1, keepdim=True)
        def get_weight_reg(self, norm_type: float = 2.0):
            return torch.empty([0])
    if reduce == 'sum':
        DirectReduction.__name__ += "Sum"
        DirectReduction.forward = DirectReduction.sum
    elif reduce == 'mean':
        DirectReduction.__name__ += "Mean"
        DirectReduction.forward = DirectReduction.mean
    else:
        raise RuntimeError(f"Invalid reduce={reduce}")
    return DirectReduction()
"""
@file   blocks.py
@author Jianfei Guo, Shanghai AI Lab
@brief  MLP blocks.
"""

__all__ = [
    'FCBlock', 
    'MLPNet', 
    'LipshitzMLP'
]

from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from nr3d_lib.fmt import log
from nr3d_lib.utils import check_to_torch, torch_dtype
from nr3d_lib.models.layers import get_nonlinearity, DenseLayer
from nr3d_lib.models.embedders import get_embedder


class FCBlock(nn.Module):
    def __init__(
        self,
        in_features: int, out_features: int, *,
        D: int=4, W: Union[int, List[int]]=128, skips: List[int]=[], 
        activation: Union[str, dict]='relu', output_activation: Union[str, dict]=None, 
        bias=True, last_bias: bool=None, equal_lr=False, weight_norm=False, 
        dtype: Union[str, torch.dtype]=None, device: torch.device=None):
        """ Create a MLP (Multi-Layer-Perceptron) of D hidden layers with W width(s)

        Args:
            in_features (int): Input feature width.
            out_features (int): Output feature width.
            D (int, optional): Number of hidden layers. Defaults to 4.
            W (Union[int, List[int]], optional): Width(s) of the hidden layers. Defaults to 128.
                Can be a single integer or a list of integers.
                A list length of D will specify different widths of D different hidden layers. 
            skips (List[int], optional): Optional skip connection on the given depth of layers. Defaults to [].
            activation (Union[str, dict], optional): Hidden layer's activation. Can be a name or a config dict. Defaults to 'relu'.
            output_activation (Union[str, dict], optional): Output layer's activation. Can be a name or a config dict. Defaults to None.
            bias (bool, optional): Whether hidden layers have bias. Defaults to True.
            last_bias (bool, optional): Whether the last layer has bias. Defaults to None.
            equal_lr (bool, optional): Whether use `equal_lr`. Defaults to False.
            weight_norm (bool, optional): Whether use weight normalization. Defaults to False.
            dtype (Union[str, torch.dtype], optional): Network param's dtype. Defaults to None.
                Can be a string (e.g. "float", "half") or a torch.dtype. 
            device (torch.device, optional): Network param's device. Defaults to None.
        """
        
        super().__init__()
        
        self.device = device
        self.dtype = dtype = torch_dtype(dtype)
        nl, gain, init_fn, first_init_fn = get_nonlinearity(activation)
        last_nl, last_gain, last_init_fn, _ = get_nonlinearity(output_activation)

        if last_bias is None:
            last_bias = bias

        self.D = D
        self.Ws = [W] * D if isinstance(W, int) else W
        if self.D >= 1:
            assert len(self.Ws) == D, f"The length of list W={self.Ws} should be D={D}."
        self.in_features = in_features
        self.skips = skips
        self.activation = activation
        layers = []
        for l in range(self.D+1):
            if l == self.D:
                out_dim = out_features
            else:
                out_dim = self.Ws[l]
            
            if l == 0:
                in_dim = in_features
            elif l in self.skips:
                in_dim = in_features + self.Ws[l-1]
            else:
                in_dim = self.Ws[l-1]
            
            if l == self.D:
                layer = DenseLayer(
                    in_dim, out_dim, activation=last_nl, bias=last_bias, 
                    dtype=self.dtype, device=device, equal_lr=equal_lr)
                layer.apply(last_init_fn)
            else:
                layer = DenseLayer(
                    in_dim, out_dim, activation=nl, bias=bias, 
                    dtype=self.dtype, device=device, equal_lr=equal_lr)
                layer.apply(first_init_fn if (l==0) else init_fn)
            if weight_norm:
                layer = nn.utils.weight_norm(layer)
            
            layers.append(layer)
        
        self.layers = nn.ModuleList(layers)
    
    def get_weight_reg(self, norm_type: float = 2.0):
        return torch.stack([p.norm(p=norm_type) for n,p in self.layers.named_parameters()])
    
    def forward(self, x: torch.Tensor, return_last: bool=False, input_max_channel: int = None):
        for i, layer in enumerate(self.layers):
            if i == 0:
                h = layer(x, max_channel=input_max_channel)
            elif i in self.skips:
                h = layer(torch.cat([h, x], dim=-1))
            elif i == self.D:
                last_h = h
                h = layer(h)
            else:
                h = layer(h)
        if return_last:
            return h, last_h
        else:
            return h

class MLPNet(FCBlock):
    def __init__(
        self, 
        in_features: int, out_features: int, *,
        embed_cfg: dict={'type':'identity'},
        D: int=4, W: Union[int, List[int]]=128, skips: List[int]=[], 
        activation: Union[str, dict]='relu', output_activation: Union[str, dict]=None, 
        weight_norm=False, dtype: Union[str, torch.dtype]=None, device: torch.device=None):
        """ Create a simple MLP network with embedded input

        Args:
            in_features (int): Input feature width.
            out_features (int): Output feature width.
            embed_cfg (dict, optional): Input embedding config dict. Defaults to {'type':'identity'}.
            D (int, optional): Number of hidden layers. Defaults to 4.
            W (Union[int, List[int]], optional): Width(s) of the hidden layers. Defaults to 128.
                Can be a single integer or a list of integers.
                A list length of D will specify different widths of D different hidden layers. 
            skips (List[int], optional): Optional skip connection on the given depth of layers. Defaults to [].
            activation (Union[str, dict], optional): Hidden layer's activation. Can be a name or a config dict. Defaults to 'relu'.
            output_activation (Union[str, dict], optional): Output layer's activation. Can be a name or a config dict. Defaults to None.
            weight_norm (bool, optional): Whether use weight normalization. Defaults to False.
            dtype (Union[str, torch.dtype], optional): Network param's dtype. Defaults to None.
                Can be a string (e.g. "float", "half") or a torch.dtype. 
            device (torch.device, optional): Network param's device. Defaults to None.
        """
        
        embedder, embedded_in_ch = get_embedder(embed_cfg, in_features)
        super().__init__(
            embedded_in_ch, out_features, 
            D=D, W=W, skips=skips, activation=activation, output_activation=output_activation, 
            weight_norm=weight_norm, dtype=dtype, device=device)
        self.embedder = embedder
    def forward(self, x: torch.Tensor, return_last: bool=False):
        x = self.embedder(x)
        return super().forward(x, return_last=return_last)

# 
class LipshitzMLP(FCBlock):
    def __init__(
        self,
        in_features: int, out_features: int, *, c_init_factor: float = 1.0, 
        D: int=4, W: Union[int, List[int]]=128, skips: List[int]=[], 
        activation: Union[str, dict]='relu', output_activation: Union[str, dict]=None, 
        bias=True, last_bias: bool=None, weight_norm=False, equal_lr=False,
        dtype: Union[str, torch.dtype]=None, device: torch.device=None):
        """ Create a MLP (Multi-Layer-Perceptron) with normalization of lipshitz bound
            Borrowed and modified from https://arxiv.org/abs/2202.08345
                and https://github.com/RaduAlexandru/permuto_sdf

        Args:
            in_features (int): Input feature width.
            out_features (int): Output feature width.
            c_init_factor (float, optional): Constant c_init multiplication factor. Defaults to 1.0.
            D (int, optional): Number of hidden layers. Defaults to 4.
            W (Union[int, List[int]], optional): Width(s) of the hidden layers. Defaults to 128.
                Can be a single integer or a list of integers.
                A list length of D will specify different widths of D different hidden layers. 
            skips (List[int], optional): Optional skip connection on the given depth of layers. Defaults to [].
            activation (Union[str, dict], optional): Hidden layer's activation. Can be a name or a config dict. Defaults to 'relu'.
            output_activation (Union[str, dict], optional): Output layer's activation. Can be a name or a config dict. Defaults to None.
            bias (bool, optional): Whether hidden layers have bias. Defaults to True.
            last_bias (bool, optional): Whether the last layer has bias. Defaults to None.
            weight_norm (bool, optional): Whether use `equal_lr`. Defaults to False.
            equal_lr (bool, optional): Whether use weight normalization. Defaults to False.
            dtype (Union[str, torch.dtype], optional): Network param's dtype. Defaults to None.
                Can be a string (e.g. "float", "half") or a torch.dtype. 
            device (torch.device, optional): Network param's device. Defaults to None.
        """
        assert len(skips) == 0 and not weight_norm and not equal_lr
        super().__init__(in_features, out_features, 
                         D=D, W=W, skips=[], activation=activation, output_activation=output_activation,
                         bias=bias, last_bias=last_bias, equal_lr=False, weight_norm=False, 
                         dtype=dtype, device=device)
        
        lipshitz_bound_per_layer = []
        with torch.no_grad():
            for layer in self.layers:
                max_w: float = layer.weight.abs().sum(dim=1).max().item() # inf-norm of matrix
                c = max_w * c_init_factor # 2.0 in permuto-SDF; 1.0 in original Lipschitz Regularization
                lipshitz_bound_per_layer.append(c)
        lipshitz_bound_per_layer = check_to_torch(lipshitz_bound_per_layer, device=device, dtype=torch.float)
        self.lipshitz_bound_per_layer = nn.Parameter(lipshitz_bound_per_layer, requires_grad=True)
    
    def normalization(self, w: torch.Tensor, softplus_ci: torch.Tensor):
        absrowsum = w.abs().sum(dim=1) # inf-norm of matrix
        # NOTE: Not numerically safe when absrowsum is extremely small. 
        # scale = (softplus_ci / absrowsum).clamp_max_(1.0) # First divide then clamp is bad.
        # NOTE: Numerically safe: first clamp then divide
        scale = softplus_ci / absrowsum.clamp_min_(softplus_ci.data)
        return w * scale[:, None]
    
    def lipshitz_bound_full(self):
        lipshitz_full = torch.prod(F.softplus(self.lipshitz_bound_per_layer))
        return lipshitz_full

    def get_weight_reg(self, norm_type: float = 2.0):
        # NOTE: Not include `lipshitz_bound_per_layer``
        return torch.stack([p.norm(p=norm_type) for n,p in self.layers.named_parameters()])

    def forward(self, x: torch.Tensor, return_last: bool=False):
        h = x
        for i, layer in enumerate(self.layers):
            weight = self.normalization(layer.weight, F.softplus(self.lipshitz_bound_per_layer[i]))
            bias = layer.bias
            with torch.autocast(device_type='cuda', dtype=self.dtype):
                if i < self.D: # Layers before the last layer
                    h = F.linear(h, weight, bias)
                    if layer.activation is not None:
                        h = layer.activation(h)
                else:
                    last_h = h # The last layer
                    out = F.linear(h, weight, bias)
                    if layer.activation is not None:
                        out = layer.activation(out)
        if return_last:
            return out, last_h
        else:
            return out

if __name__ == "__main__":
    def unit_test(device=torch.device('cuda'), dtype=torch.float16):
        from icecream import ic
        #-------------- Simple
        m1 = FCBlock(
            in_features=1, out_features=1, D=4, W=64, skips=[],
            activation={'type': 'softplus', 'beta': 100.}, output_activation='sigmoid',
            weight_norm=False, dtype=dtype).to(device=device)
        ic(m1)
        x = torch.ones([1,1], dtype=dtype, device=device)
        y = m1(x)
        ic(y.dtype, y.shape)
        
        #-------------- Single layer
        m2 = FCBlock(
            32, 1, D=0, W=None, activation='tanh', bias=True, dtype=dtype
        ).to(device=device)
        ic(m2)
        for n,p in m2.named_parameters():
            print(n, p.shape)
        x = torch.ones([1,32], dtype=dtype, device=device)
        y = m2(x)
        ic(y.dtype, y.shape)
        
        #-------------- Test variable width
        m3 = FCBlock(3, 1, D=4, W=[64,32,16,8], activation='selu', dtype=dtype, device=device)
        ic(m3)
        for n,p in m3.named_parameters():
            print(n, p.shape)
        x = torch.randn([1,3], dtype=dtype, device=device)
        ic(y.dtype, y.shape)
        
        #-------------- Test siren
        m4 = FCBlock(
            in_features=1, out_features=1, D=4, W=64, skips=[],
            activation={'type': 'siren', 'w0': 30.}, output_activation='none',
            weight_norm=False, dtype=dtype).to(device=device)
        ic(m4)
        x = torch.ones([1,1], dtype=dtype, device=device)
        y = m4(x)
        ic(y.dtype, y.shape)
        
        #-------------- Test lipshitz
        m5 = LipshitzMLP(3, 3, D=4, W=128, dtype=dtype, device=device)
        ic(m5)
        x = torch.ones([1,3], dtype=dtype, device=device)
        y = m5(x)
        ic(y.dtype, y.shape)
        
    unit_test()
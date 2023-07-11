"""
@file   tcnn_adapter.py
@author Jianfei Guo, Shanghai AI Lab
@brief  A python adapter from tiny-cuda-nn's framework to nr3d_lib APIs.
"""

import torch
import torch.nn as nn
from typing import List, Literal

from nr3d_lib.fmt import log
from nr3d_lib.config import ConfigDict

try:
    import tinycudann as tcnn
    nonlinearity_map = {
        'none':         "None",
        'relu':         "ReLU",
        'exponential':  "Exponential",
        'sine':         "Sine",
        'sigmoid':      "Sigmoid",
        'square_plus':  "Squareplus",
        'softplus':     "Softplus",
    }

    encoding_map = {
        'composite':    "Composite",
        'none':         "Identity",
        'identity':     "Identity",
        'sinusoidal':   "Frequency",
        'spherical':    "SphericalHarmonics",
        'tiangle_wave': "TriangleWave",
        'oneblob':      "OneBlob"
    }

    def tcnn_network_config(D, W, activation: str, output_activation: str, use_cutlass=False):
        if not use_cutlass:
            assert W in [16, 32, 64, 128], f"W can only be 16, 32, 64, or 128. If current W={W} is necessary, set `use_cutlass=True` instead."
        activation = nonlinearity_map[activation.lower()]
        output_activation = nonlinearity_map[output_activation.lower()]
        network_config = {
            "otype": "FullyFusedMLP" if not use_cutlass else "CutlassMLP",               # Component type.
            "activation": activation,               # Activation of hidden layers.
            "output_activation": output_activation,   # Activation of the output layer.
            "n_neurons": W,           # Neurons in each hidden layer. # May only be 16, 32, 64, or 128.
            "n_hidden_layers": D,   # Number of hidden layers.
        }
        if not use_cutlass:
            network_config["feedback_alignment"] = False  # Use feedback alignment # [Lillicrap et al. 2016].
        return network_config

    def tcnn_encoding_config(input_dim: int, type: str, **params):
        otype = encoding_map[type.lower()]
        encoding_config = {
            'otype': otype,
            **params
        }
        if otype != 'Composite':
            encoding_config['n_dims_to_encode'] = input_dim
        
        # NOTE: Tcnn requires input to be also doubled.
        # if otype != 'Identity' and include_input:
        #     encoding_config = {
        #         'otype': 'Composite',
        #         'nested': [
        #             {'otype': 'Identity', 'n_dims_to_encode': input_dim}, 
        #             encoding_config
        #         ]
        #     }
        return encoding_config


    def get_tcnn_blocks(in_features, out_features, *, 
        D=8, W=256, skips: List[int]=[], activation: str='relu', output_activation: str='none', seed=42, use_cutlass=False, dtype=torch.float16, device=torch.device('cuda')):
        if len(skips) == 0:
            return TcnnFCBlock(in_features, out_features, D=D, W=W, activation=activation, output_activation=output_activation, seed=seed, use_cutlass=use_cutlass).to(device=device)
        else:
            return TcnnFCBlockWSkips(in_features, out_features, D=D, W=W, skips=skips, activation=activation, output_activation=output_activation, seed=seed, use_cutlass=use_cutlass).to(device=device)

    class TcnnFCBlock(tcnn.Network):
        def __init__(
            self, in_features, out_features, *, 
            D=4, W=128, activation: str='relu', output_activation: str='none', seed=42, 
            use_cutlass=False, dtype=torch.float16, device=torch.device('cuda'), weight_norm=False):
            assert not weight_norm, f'{self.__class__.__name__} does not support weight normalization'
            network_config = tcnn_network_config(D=D, W=W, activation=activation, output_activation=output_activation, use_cutlass=use_cutlass)
            super().__init__(in_features, out_features, network_config=network_config, seed=seed)
            self.in_features = self.n_input_dims
            self.out_features = self.n_output_dims
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            prefix = x.shape[:-1]
            return super().forward(x.to(self.dtype).flatten(0, -2)).unflatten(0, prefix)

    class TcnnFCBlockWSkips(nn.Module):
        def __init__(
            self, in_features, out_features, *, 
            D=8, W=256, skips: List[int]=[4], activation: str='relu', output_activation: str='none', seed=42, use_cutlass=False):
            super().__init__()
            if isinstance(skips, int): skips = [skips]
            self.skips = skips
            
            D_cur = 0
            blocks = []
            for i, sk in enumerate(self.skips):
                D_ = sk - D_cur
                bl = tcnn.Network(
                    in_features if i==0 else W, W, 
                    tcnn_network_config(D=D_, W=W, activation=activation, output_activation='none', use_cutlass=use_cutlass), 
                    seed=seed) 
                blocks.append(bl)
                D_cur += D_
            
            bl = tcnn.Network(
                in_features if D_cur==0 else (W + in_features), out_features, 
                tcnn_network_config(D=(D-D_cur), W=W, activation=activation, output_activation=output_activation, use_cutlass=use_cutlass),
                seed=seed)
            blocks.append(bl)
                
            self.blocks = nn.ModuleList(blocks)
        def forward(self, x: torch.Tensor):
            prefix = x.shape[:-1]
            x = x.flatten(0, -2)
            h = x
            for bl in self.blocks[:-1]:
                h = bl(h)
                h = torch.cat([h, x], dim=-1)
            h = self.blocks[-1](h)
            return h.unflatten(0, prefix)

    class TcnnEncoding(tcnn.Encoding):
        def __init__(self, input_dim: int, embed_cfg:dict={'type':'identity'}, encoding_bypass=None, include_input=False, seed=42):
            embed_cfg = embed_cfg.copy()
            include_input = embed_cfg.pop('include_input', False) or include_input
            
            self.in_features = input_dim
            
            c = tcnn_encoding_config(input_dim, **embed_cfg) if encoding_bypass is None else encoding_bypass
            self.include_input = (c['otype'] != 'Identity') and include_input
            if self.include_input:
                c = {
                    'otype': 'Composite',
                    'nested': [
                        {'otype': 'Identity', 'n_dims_to_encode': input_dim}, 
                        c
                    ]
                }
                super().__init__(input_dim * 2, c, seed=seed)
            else:
                super().__init__(input_dim, c, seed=seed)
            self.out_features = self.n_output_dims
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            prefix = x.shape[:-1]
            if self.include_input:
                return super().forward(x.flatten(0, -2).tile(2)).unflatten(0, prefix)
            else:
                return super().forward(x.flatten(0, -2)).unflatten(0, prefix)

    class TcnnNet(tcnn.NetworkWithInputEncoding):
        def __init__(
            self, in_features, out_features, *, 
            embed_cfg:dict={'type':'identity'}, encoding_bypass=None, embed_include_input=False,
            D=8, W=128, activation: str='relu', output_activation: str='none', seed=42, use_cutlass=False):
            
            self.in_features = in_features
            self.out_features = out_features
            
            network_cfg = tcnn_network_config(D=D, W=W, activation=activation, output_activation=output_activation, use_cutlass=use_cutlass)
            if encoding_bypass is None:
                encoding_cfg = tcnn_encoding_config(in_features, **embed_cfg)
            else:
                encoding_cfg = encoding_bypass
            self.embed_include_input = (encoding_cfg['otype'] != 'Identity') and embed_include_input
            if self.embed_include_input:
                c = {
                    'otype': 'Composite',
                    'nested': [
                        {'otype': 'Identity', 'n_dims_to_encode': in_features}, 
                        c
                    ]
                }
                super().__init__(in_features * 2, out_features, encoding_cfg, network_cfg, seed=seed)
            else:
                super().__init__(in_features, out_features, encoding_cfg, network_cfg, seed=seed)
            
        def forward(self, x: torch.Tensor):
            prefix = x.shape[:-1]
            if self.embed_include_input:
                return super().forward(x.flatten(0, -2).tile(2)).unflatten(0, prefix)
            else:
                return super().forward(x.flatten(0, -2)).unflatten(0, prefix)

    class TcnnNetWSkips(nn.Module):
        def __init__(
            self, in_features, out_features, *, 
            embed_cfg:dict={'type':'identity'}, encoding_bypass=None,
            D=8, W=256, skips: List[int]=[4], activation: str='relu', output_activation: str='none', seed=42, use_cutlass=False):
            super().__init__()
            if isinstance(skips, int): skips = [skips]
            self.skips = skips
            self.encoding = TcnnEncoding(in_features, embed_cfg, encoding_bypass=encoding_bypass, seed=seed)
            self.blocks = TcnnFCBlockWSkips(self.encoding.out_features, out_features, D=D, W=W, skips=skips, activation=activation, output_activation=output_activation, seed=seed, use_cutlass=use_cutlass)
        def forward(self, x: torch.Tensor):
            prefix = x.shape[:-1]
            h = self.encoding(x.flatten(0, -2))
            h = self.blocks(h)
            return h.unflatten(0, prefix)
except ImportError:
    log.info("tinycudann is not installed")

if __name__ == "__main__":
    def test_fwd(device=torch.device('cuda')):
        from icecream import ic
        dtype = torch.float16
        net = TcnnFCBlock(22, 1, D=1, W=64)
        x = torch.randn([365365, 22], dtype=dtype, device=device, requires_grad=True)
        h = net.forward(x)
        ic(h)

    def test_network_with_encoding(device=torch.device('cuda'), batch_size=365365):
        from icecream import ic
        enc_cfg = {
            "otype": "Composite",
            "nested": [
                {
                    "n_dims_to_encode": 3, # Spatial dims
                    "otype": "TriangleWave",
                    "n_frequencies": 12
                },
                {
                    "n_dims_to_encode": 5, # Non-linear appearance dims.
                    "otype": "OneBlob",
                    "n_bins": 4
                },
                {
                    # Number of remaining linear dims is automatically derived
                    "otype": "Identity"
                }
            ]
        }
        net_cfg = {
            "otype": "FullyFusedMLP",    # Component type.
            "activation": "ReLU",        # Activation of hidden layers.
            "output_activation": "None", # Activation of the output layer.
            "n_neurons": 64,            #/ Neurons in each hidden layer.
                                        # May only be 16, 32, 64, or 128.
            "n_hidden_layers": 2,        # Number of hidden layers.
            "feedback_alignment": False  # Use feedback alignment
                                        # [Lillicrap et al. 2016].
        }
        net = tcnn.NetworkWithInputEncoding(12, 1, enc_cfg, net_cfg)
        input = torch.randn([batch_size, 12], dtype=torch.float, device=device)
        y = net(input)
        
        ic(y.shape, y.dtype, y)

    def test_blocks_with_skips(device=torch.device('cuda'), batch_size=365365):
        from icecream import ic
        block = TcnnFCBlockWSkips(24, 4, D=8, W=128, skips=[])
        print(block)
        x = torch.randn([batch_size, 24], dtype=torch.float16, device=device)
        y = block(x)
        ic(y.shape, y.dtype, y)
        
        block = TcnnFCBlockWSkips(24, 4, D=8, W=128, skips=[4])
        print(block)
        x = torch.randn([batch_size, 24], dtype=torch.float16, device=device)
        y = block(x)
        ic(y.shape, y.dtype, y)

    def test_encoding_scale(device=torch.device('cuda')):
        from nr3d_lib.models.embedders import get_embedder
        from icecream import ic
        spherical_nr3d = get_embedder({'type': 'spherical', 'degree': 6})[0]
        spherical_tcnn = get_embedder({'type': 'spherical', 'degree': 6, 'use_tcnn_backend': True})[0]
        
        rays_d = torch.rand([1, 3], device=device, dtype=torch.float) * 2 - 1
        y1 = spherical_nr3d(rays_d) # Expects [-1,1]^3 input
        y2 = spherical_tcnn(rays_d)
        y3 = spherical_tcnn(rays_d/2+0.5) # This is correct, because tcnn expects [0,1]^3 input
        ic(y1)
        ic(y2)
        ic(y3)
    
    # test_fwd()
    # test_encdec()
    # test_network_with_encoding()
    # test_blocks_with_skips()
    test_encoding_scale()
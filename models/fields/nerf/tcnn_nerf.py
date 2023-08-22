"""
@file   tcnn_nerf.py
@author Jianfei Guo, Shanghai AI Lab
@brief  MLP-based vanilla NeRF, using tiny-cuda-nn modules.
"""

__all__ = [
    'TcnnNeRF', 
    'TcnnEmbeddedNeRF', 
    'TcnnRadianceNet'
]

import torch
import torch.nn as nn
import torch.nn.functional as F

from nr3d_lib.fmt import log
from nr3d_lib.utils import torch_dtype
from nr3d_lib.profile import profile
from nr3d_lib.models.base import ModelMixin
from nr3d_lib.models.layers import DenseLayer, get_nonlinearity

try:
    import tinycudann as tcnn
    from nr3d_lib.models.tcnn_adapter import TcnnFCBlock, TcnnFCBlockWSkips, TcnnNet, TcnnNetWSkips, tcnn_encoding_config, tcnn_network_config
    class TcnnNeRF(ModelMixin, nn.Module):
        def __init__(
            self, input_ch_pts=3, input_ch_view=3, use_view_dirs=True, 
            D=8, W=128, skips=[4], activation: str='relu', sigma_activation: str='none', rgb_acitvation: str='sigmoid', 
            dtype=torch.float16, device=torch.device('cuda'), seed=42):
            super().__init__()
            self.device = device
            self.dtype = torch_dtype(dtype)
            
            self.use_view_dirs = use_view_dirs
            
            self.pts_blocks = TcnnFCBlockWSkips(input_ch_pts, W, D=D, W=W, skips=skips, activation=activation, output_activation='none', seed=seed).to(self.device)
            if self.use_view_dirs:
                # W -> 1
                self.sigma_layer = DenseLayer(W, 1, activation=sigma_activation, dtype=self.dtype, device=self.device)
                # W + input_ch_view -> W//2 -> W//2 -> 3
                self.rgb_blocks = TcnnFCBlock(input_ch_view + W, 3, D=2, W=W//2, activation=activation, output_activation=rgb_acitvation, seed=seed).to(self.device)
            else:
                self.output_linear = DenseLayer(W, 4, dtype=self.dtype, device=self.device)
                self.sigma_activation = get_nonlinearity(sigma_activation).nl
                self.rgb_activation = get_nonlinearity(rgb_acitvation).nl
        def forward(self, x: torch.Tensor, v: torch.Tensor=None):
            h = self.pts_blocks(x)
            if self.use_view_dirs:
                sigma = self.sigma_layer(h)
                rgb = self.rgb_blocks(torch.cat([h, v], dim=-1))
            else:
                outputs = self.output_linear(h)
                rgb, sigma = self.rgb_activation(outputs[..., :3]), self.sigma_activation(outputs[..., 3:])
            return dict(radiances=rgb, sigma=sigma.squeeze(-1))
        def forward_sigma(self, x):
            h = self.pts_blocks(x)
            if self.use_view_dirs:
                sigma = self.sigma_layer(h)
            else:
                outputs = self.output_linear(h)
                sigma = self.sigma_activation(outputs[..., 3:])
            return sigma.squeeze(-1)

    class TcnnEmbeddedNeRF(ModelMixin, nn.Module):
        def __init__(
            self,
            n_pos_dims=3, pos_embed_cfg:dict={'type':'sinusoidal', 'n_frequencies': 10}, pos_embed_include_input=False,
            use_view_dirs=True, dir_embed_cfg:dict={'type':'spherical', 'degree': 4}, dir_embed_include_input=False,
            n_geo_embedding=0, geo_embed_cfg:dict={'type':'identity'}, 
            n_appear_embedding=0, appear_embed_cfg:dict={'type':'identity'}, 
            D=8, W=128, skips=[4], activation: str='relu', sigma_activation: str='none', rgb_acitvation: str='sigmoid',
            dtype=torch.float16, device=torch.device('cuda'), seed=42
            ) -> None:
            super().__init__()
            
            self.dtype = torch_dtype(dtype)
            self.device = device

            self.use_view_dirs = use_view_dirs
            self.use_appear_embedding = n_appear_embedding > 0
            self.use_geo_embedding = n_geo_embedding > 0
            
            #------------------ Base pts block
            pts_cfgs = []
            # x
            pos_embed_cfg = pos_embed_cfg.copy()
            pos_embed_include_input = pos_embed_cfg.pop('include_input', False) or pos_embed_include_input
            c = tcnn_encoding_config(n_pos_dims, **pos_embed_cfg)
            self.pos_embed_include_input = (c['otype'] != 'Identity') and pos_embed_include_input
            n_pos_in = n_pos_dims
            if self.pos_embed_include_input:
                # A hack for include_input
                n_pos_in = n_pos_dims * 2
                c = {
                    'otype': 'Composite',
                    'nested': [
                        {'otype': 'Identity', 'n_dims_to_encode': n_pos_dims}, 
                        c
                    ]
                }
            pts_cfgs.append(c)
            # h_geo_embed
            if self.use_geo_embedding: pts_cfgs.append(tcnn_encoding_config(n_geo_embedding, **geo_embed_cfg))
            
            pts_encoding = {'otype':'Composite', 'nested': pts_cfgs} if len(pts_cfgs) > 1 else pts_cfgs[0]
            self.pts_blocks = TcnnNetWSkips(
                # [x, h_geo_embed]
                n_pos_in + n_geo_embedding, W, encoding_bypass=pts_encoding,
                D=D, W=W, skips=skips, activation=activation, output_activation='none').to(self.device)
            self.sigma_layer = DenseLayer(W, 1, activation=sigma_activation, dtype=self.dtype, device=self.device)
            
            #------------------ Appearance blocks
            # W
            view_cfgs = [tcnn_encoding_config(W, 'identity')]
            # v
            if self.use_view_dirs:
                n_view_dirs = 3
                dir_embed_cfg = dir_embed_cfg.copy()
                dir_embed_include_input = dir_embed_cfg.pop('include_input', False) or dir_embed_include_input
                c = tcnn_encoding_config(3, **dir_embed_cfg)
                self.dir_embed_include_input = (c['otype'] != 'Identity') and dir_embed_include_input
                if self.dir_embed_include_input:
                    # A hack for include_input
                    n_view_dirs = 3 * 2
                    c = {
                        'otype': 'Composite',
                        'nested': [
                            {'otype': 'Identity', 'n_dims_to_encode': 3}, 
                            c
                        ]
                    }
                view_cfgs.append(c)
            else:
                n_view_dirs = 0
            # h_appear_embed
            if self.use_appear_embedding: view_cfgs.append(tcnn_encoding_config(n_appear_embedding, **appear_embed_cfg))
                
            view_encoding = {'otype':'Composite', 'nested': view_cfgs} if len(view_cfgs) > 1 else view_cfgs[1]
            self.rgb_blocks = TcnnNet(
                # [W, v, h_appear_embed]
                W + n_view_dirs + n_appear_embedding, 3, encoding_bypass=view_encoding,
                D=2, W=W//2, activation=activation, output_activation=rgb_acitvation, seed=seed).to(self.device)
        
        def forward(self, x: torch.Tensor, v: torch.Tensor=None, *, h_geo_embed: torch.Tensor=None, h_appear_embed: torch.Tensor=None):
            if self.pos_embed_include_input:
                x = x.tile(2)
            if self.use_view_dirs and self.dir_embed_include_input:
                v = v.tile(2)
            h = self.pts_blocks(x if not self.use_geo_embedding else torch.cat([x, h_geo_embed], dim=-1))
            sigma = self.sigma_layer(h)
            uses = (True, self.use_view_dirs, self.use_appear_embedding)
            rgb = self.rgb_blocks(torch.cat([i for idx,i in enumerate((h, v, h_appear_embed)) if uses[idx]], dim=-1))
            return dict(radiances=rgb, sigma=sigma.squeeze(-1))   
        def forward_sigma(self, x: torch.Tensor, *, h_geo_embed: torch.Tensor=None):
            if self.pos_embed_include_input:
                x = x.tile(2)
            h = self.pts_blocks(x if not self.use_geo_embedding else torch.cat([x, h_geo_embed], dim=-1))
            sigma = self.sigma_layer(h)
            return sigma.squeeze(-1)

    class TcnnRadianceNet(ModelMixin, tcnn.NetworkWithInputEncoding):
        # NOTE: Direct use tcnn's NetworkWithInputEncoding holistic model.
        def __init__(
            self, 
            use_pos=True, pos_embed_cfg:dict={'type':'identity'}, pos_embed_include_input=False,
            use_view_dirs=True, dir_embed_cfg:dict={'type':'identity'}, dir_embed_include_input=False,
            use_nablas=None, nablas_embed_cfg:dict={'type':'identity'}, 
            n_rgb_used_extrafeat=0, extrafeat_embed_cfg:dict={'type':'identity'},
            n_appear_embedding=0, appear_embed_cfg:dict={'type':'identity'},
            D=2, W=64, activation='relu', output_activation='sigmoid', 
            seed=42, dtype=torch.float16, device=torch.device('cuda')) -> None:
            
            self.use_pos = use_pos
            self.use_view_dirs = use_view_dirs
            self.use_nablas = self.use_view_dirs if use_nablas is None else use_nablas
            self.use_extrafeat = n_rgb_used_extrafeat > 0
            self.use_appear_embed = n_appear_embedding > 0
            
            # [x, v, n, h_extra, h_appear_embed]
            in_features = (3 if self.use_pos else 0) + (3 if self.use_view_dirs else 0) + (3 if self.use_nablas else 0) + n_rgb_used_extrafeat + n_appear_embedding
            assert in_features > 0, "Invalid configuration leads to zero-dim input"
            
            cfgs = []
            # x
            if self.use_pos:
                pos_embed_cfg = pos_embed_cfg.copy()
                pos_embed_include_input = pos_embed_cfg.pop('include_input', False) or pos_embed_include_input
                c = tcnn_encoding_config(3, **pos_embed_cfg)
                self.pos_embed_include_input = (c['otype'] != 'Identity') and pos_embed_include_input
                if self.pos_embed_include_input:
                    # A hack for include_input
                    in_features += 3
                    c = {
                        'otype': 'Composite',
                        'nested': [
                            {'otype': 'Identity', 'n_dims_to_encode': 3}, 
                            c
                        ]
                    }
                cfgs.append(c)
            
            # v
            if self.use_view_dirs: 
                dir_embed_cfg = dir_embed_cfg.copy()
                c = tcnn_encoding_config(3, **dir_embed_cfg)
                dir_embed_include_input = dir_embed_cfg.pop('include_input', False) or dir_embed_include_input
                self.dir_embed_include_input = (c['otype'] != 'Identity') and dir_embed_include_input
                if self.dir_embed_include_input:
                    # A hack for include_input
                    in_features += 3
                    c = {
                        'otype': 'Composite',
                        'nested': [
                            {'otype': 'Identity', 'n_dims_to_encode': 3}, 
                            c
                        ]
                    }
                cfgs.append(c)
            
            # n
            if self.use_nablas: cfgs.append(tcnn_encoding_config(3, **nablas_embed_cfg))
            
            # h_extra
            if self.use_extrafeat: cfgs.append(tcnn_encoding_config(n_rgb_used_extrafeat, **extrafeat_embed_cfg))
            
            # h_appear_embed
            if self.use_appear_embed: cfgs.append(tcnn_encoding_config(n_appear_embedding, **appear_embed_cfg))
            
            encoding_config = {"otype": "Composite", "nested": cfgs} if len(cfgs) > 1 else cfgs[0]
            # [x, v, n, h_extra, h_appear_embed]
            self._uses = (self.use_pos, self.use_view_dirs, self.use_nablas, self.use_extrafeat, self.use_appear_embed)
            
            network_config = tcnn_network_config(D=D, W=W, activation=activation, output_activation=output_activation)

            super().__init__(in_features, 3, encoding_config, network_config, seed)
            self.in_features = in_features
            self.out_features = 3
            self.to(device)
        
        @profile
        def forward(self, x: torch.Tensor, v: torch.Tensor=None, n: torch.Tensor=None, *, h_extra: torch.Tensor=None, h_appear_embed: torch.Tensor=None) -> torch.Tensor:
            prefix = x.shape[:-1]
            uses = self._uses
            if self.use_pos and self.pos_embed_include_input:
                x = x.tile(2)
            if self.use_view_dirs and self.dir_embed_include_input:
                v = v.tile(2)
            h = torch.cat([i for idx, i in enumerate((x, v, n, h_extra, h_appear_embed)) if uses[idx]], dim=-1)
            return dict(radiances=super().forward(h.flatten(0, -2)).unflatten(0, prefix))

except ImportError:
    log.info("tinycudann is not installed")
    
    class TcnnNeRF(ModelMixin, nn.Module):
        pass
    class TcnnEmbeddedNeRF(ModelMixin, nn.Module):
        pass
    class TcnnRadianceNet(ModelMixin, nn.Module):
        pass
    

if __name__ == "__main__":
    def test_radiance_net(device=torch.device('cuda'), batch_size=365365):
        from icecream import ic
        from torch.utils.benchmark import Timer
        net = TcnnRadianceNet(
            pos_embed_cfg={'type':'identity'}, 
            dir_embed_cfg={'type':'spherical', 'degree':4},
            n_rgb_used_extrafeat=22, 
            n_appear_embedding=0
        )
        ic(net)
        x = torch.randn([batch_size, 3], dtype=torch.float, device=device)
        v = F.normalize(torch.randn([batch_size, 3], dtype=torch.float, device=device), dim=-1)
        n = F.normalize(torch.randn([batch_size, 3], dtype=torch.float, device=device), dim=-1)
        h = torch.randn([batch_size, 22], dtype=torch.float, device=device)
        y = net.forward(x,v,n,h)['radiances']
        ic(y.shape, y.dtype, y)
        
        with torch.no_grad():
            # 0.907 ms
            print(Timer(
                stmt='net.forward(x,v,n,h)', 
                globals={'net':net, 'x':x, 'v':v, 'n':n, 'h':h}
            ).blocked_autorange())

    def test_nerf(device=torch.device('cuda'), batch_size=365365):
        from icecream import ic
        from torch.utils.benchmark import Timer
        nerf = TcnnEmbeddedNeRF(
            pos_embed_cfg={'type':'sinusoidal', 'n_frequencies': 10, 'include_input': True}, 
            dir_embed_cfg={'type':'spherical', 'degree':4, 'include_input': True}, 
            D=8, W=128, skips=[4])
        ic(nerf)
        
        x = torch.randn([batch_size, 3], device=device, dtype=torch.float)
        v = F.normalize(torch.randn([batch_size, 3], device=device, dtype=torch.float), dim=-1)
        out = nerf.forward(x, v)
        sigma, rgb = out['sigma'], out['radiances']
        ic(sigma.dtype, sigma.shape, sigma)
        ic(rgb.dtype, rgb.shape, rgb)
        
        with torch.no_grad():
            # 9.02 ms    
            print(Timer(
                stmt='nerf.forward(x, v)', 
                globals={'x':x, 'v':v, 'nerf':nerf}
            ).blocked_autorange())
        
    test_radiance_net()
    test_nerf()
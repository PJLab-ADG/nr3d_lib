"""
Modified from pi-gan
"""

import torch
import torch.nn as nn

from nr3d_lib.models.blocks import get_blocks
from nr3d_lib.models.layers import DenseLayer, get_nonlinearity

class FilmSirenLayer(DenseLayer):
    def __init__(self, input_dim, out_dim, bias=True, is_first=False, w0=1, dtype=torch.float, device=None):
        nl, gain, init_fn, first_init_fn = get_nonlinearity({'tp':'sine', 'param':{'w0':w0}})
        DenseLayer.__init__(self, input_dim, out_dim, bias=bias, activation=nl, device=device, dtype=dtype)
        self.apply(first_init_fn if is_first else init_fn)

    def forward(self, x, gamma=None, beta=None):
        """
            x: [..., N, dim_in]
            out: [..., N, dim_out]
            gamma: [..., dim_out]
            beta: [..., dim_out]
        """
        out = nn.Linear.forward(self, x)
        # FiLM modulation
        if gamma is not None:
            out = out * gamma[(Ellipsis,) + (None,)*(x.dim()-2) + (slice(None),)]
        if beta is not None:
            out = out + beta[(Ellipsis,) + (None,)*(x.dim()-2) + (slice(None),)]
        out = self.activation(out)
        return out

def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if (classname.find('Linear') != -1) or isinstance(m, DenseLayer):
        torch.nn.init.kaiming_normal_(
            m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')

class MappingNetwork(nn.Module):
    def __init__(self, z_dim, map_hidden_dim, map_output_dim, dtype=torch.float, device=None):
        super().__init__()
        
        self.dtype = dtype
        layers = [
            DenseLayer(z_dim, map_hidden_dim, activation={'type': 'leaky_relu', 'negative_slope': 0.2, 'inplace': True}, device=device, dtype=dtype),
            DenseLayer(map_hidden_dim, map_hidden_dim, activation={'type': 'leaky_relu', 'negative_slope': 0.2, 'inplace': True}, device=device, dtype=dtype),
            DenseLayer(map_hidden_dim, map_hidden_dim, activation={'type': 'leaky_relu', 'negative_slope': 0.2, 'inplace': True}, device=device, dtype=dtype),
        ]
        self.net = nn.Sequential(*layers)
        self.net.apply(kaiming_leaky_init)  # TODO: why, and how this value
        self.to_gamma = DenseLayer(map_hidden_dim, map_output_dim, device=device, dtype=dtype)
        self.to_beta = DenseLayer(map_hidden_dim, map_output_dim, device=device, dtype=dtype)
        with torch.no_grad():
            self.to_gamma.weight *= 0.25    # TODO: why, and how this value
            # self.to_beta.weight *= 0.25     # TODO: why, and how this value

    def forward(self, x):
        x = self.net(x)
        return self.to_gamma(x), self.to_beta(x)

class MultiLayerMapping(nn.Module):
    def __init__(self, *, num_output_layers, duplicate=True, dtype=torch.float, device=None, **mapping_layer_kwargs):
        super().__init__()
        self.dtype = dtype
        self.duplicate = duplicate
        self.num_output_layers = num_output_layers
        if self.duplicate:
            self.mapping = MappingNetwork(**mapping_layer_kwargs, dtype=dtype, device=device)
        else:
            self.mapping = nn.ModuleList([MappingNetwork(**mapping_layer_kwargs, dtype=dtype, device=device) for _ in range(num_output_layers)])

    def forward(self, input: torch.Tensor):
        # input: [..., S]
        # gammas: [..., num_layers, out_features]
        prefix_shape = input.shape[:-1]
        prefix_dim = len(prefix_shape)
        if self.duplicate:
            gamma, beta = self.mapping(input)
            gammas = gamma[..., None, :].expand([*prefix_shape, self.num_output_layers, -1])
            betas = beta[..., None, :].expand([*prefix_shape, self.num_output_layers, -1])
        else:
            gammas, betas = [], []
            for i in range(self.num_output_layers):
                gamma, beta = self.mapping[i](input)
                gammas.append(gamma)
                betas.append(beta)
            gammas = torch.stack(gammas, dim=prefix_dim)
            betas = torch.stack(betas, dim=prefix_dim)
        # NOTE: this is not correct! will lead to grad with norm > 1e+10
        # return gammas * 15 + 30, betas
        return gammas * 0.5 + 1, betas

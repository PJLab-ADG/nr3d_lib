"""
@file   sinsoidal_pytorch.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Pytorch-implemented sinsoidal positional embeddings.
"""

__all__ = [
    'SinusoidalEmbedder',
    'AnnealedSinusoidalEmbedder',
    'get_sinusoidal_embedder'
]

import numpy as np

import torch
import torch.nn as nn

class SinusoidalEmbedder(nn.Module):
    def __init__(
        self, 
        input_dim: int, N_freqs: int,
        max_freq_log2: int, min_freq_log2: int=0.,
        log_sampling=True, include_input=True):
        """ Construct a sinusoidal embedder

        Args:
            input_dim (int): dimension of input to be embedded
            N_freqs (int): number of frequency bands
            max_freq_log2 (int): log2 of max freq
            min_freq_log2 (int, optional): log2 of min freq. Defaults to 0..
            log_sampling (bool, optional): if True, frequency bands are linerly sampled in log-space. Defaults to True.
            include_input (bool, optional): if True, raw input is included in the embedding. Defaults to True.
        """
        super().__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.min_freq_log2 = min_freq_log2
        self.max_freq_log2 = max_freq_log2

        self.out_features = 0
        if self.include_input:
            self.out_features += self.input_dim

        self.out_features += self.input_dim * N_freqs * 2

        if log_sampling:
            freq_bands = 2. ** torch.linspace(min_freq_log2, max_freq_log2, N_freqs)
        else:
            freq_bands = torch.linspace(2. ** min_freq_log2, 2. ** max_freq_log2, N_freqs)
        self.register_buffer('freq_bands', freq_bands, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): [..., self.input_dim]

        Returns:
            torch.Tensor: [..., self.out_features]
        """
        assert (x.shape[-1] == self.input_dim)
        
        out = []
        if self.include_input:
            out.append(x)
        
        if len(freq_bands:=self.freq_bands) > 0:
            # [..., 1, input_dim] * [N_freqs, 1] -> [..., N_freqs, input_dim]
            angles = x.unsqueeze(-2) * freq_bands.to(x).unsqueeze(-1)
            # [..., N_freqs, 2, input_dim] -> [..., N_freqs2*input_dim]
            angles = torch.stack([angles, angles + np.pi/2.], dim=-2).flatten(-3,-1)
            features = torch.sin(angles)
            out.append(features)

        out = torch.cat(out, dim=-1)
        assert (out.shape[-1] == self.out_features)
        return out

    def extra_repr(self) -> str:
        return f"in_dim={self.input_dim}, out_dim={self.out_features}, freq_bands=({len(self.freq_bands)}){self.freq_bands}"

class AnnealedSinusoidalEmbedder(SinusoidalEmbedder):
    # Modified from https://github.com/google/nerfies/nerfies/modules.py
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('alpha', torch.tensor([1.0]), persistent=True)
        self.set_cosine_easing_window(self.alpha)
    def forward(self, x: torch.Tensor):
        assert hasattr(self, 'window'), "Must call set_cosine_easing_window(alpha)"
        assert (x.shape[-1] == self.input_dim)
        window = self.window.to(device=x.device, dtype=x.dtype)
        freq_inds = self.freq_inds.to(x.device)
        
        out = []
        if self.include_input:
            out.append(x)

        if len(self.freq_bands) > 0:
            angles = x.unsqueeze(-2) * self.freq_bands.unsqueeze(-1)
            angles = torch.stack([angles, angles + np.pi/2.], dim=-2).flatten(-2,-1)
            # NOTE: only calculate features on those activated frequencies to accelerate
            if len(freq_inds) != len(self.freq_bands):
                features = angles.new_zeros(angles.shape)
                features[..., freq_inds, :] = torch.sin(angles[..., freq_inds, :]) * window[freq_inds].view(-1,1)
            elif (window!=1).any():
                features = torch.sin(angles) * window.view(-1,1)
            else:
                features = torch.sin(angles)
            out.append(features.flatten(-2,-1))
        
        out = torch.cat(out, dim=-1)
        assert (out.shape[-1] == self.out_features)
        return out

    def set_cosine_easing_window(self, alpha: float):
        """
        alpha: from 0.0 to 1.0; increase more frequencies
        
        An example of alpha - window mapping
            alpha                    window
            -------- | ------------------------------------------
            0.           [0. 0. 0. 0. 0. 0.]
            0.0526       [0.2265 0.     0.     0.     0.     0.    ]
            0.1053       [0.7008 0.     0.     0.     0.     0.    ]
            0.1579       [0.9932 0.     0.     0.     0.     0.    ]
            0.2105       [1.     0.1614 0.     0.     0.     0.    ]
            0.2632       [1.     0.6227 0.     0.     0.     0.    ]
            0.3158       [1.     0.9729 0.     0.     0.     0.    ]
            0.3684       [1.     1.     0.1054 0.     0.     0.    ]
            0.4211       [1.     1.     0.5413 0.     0.     0.    ]
            0.4737       [1.     1.     0.9397 0.     0.     0.    ]
            0.5263       [1.     1.     1.     0.0603 0.     0.    ]
            0.5789       [1.     1.     1.     0.4587 0.     0.    ]
            0.6316       [1.     1.     1.     0.8946 0.     0.    ]
            0.6842       [1.     1.     1.     1.     0.0271 0.    ]
            0.7368       [1.     1.     1.     1.     0.3773 0.    ]
            0.7895       [1.     1.     1.     1.     0.8386 0.    ]
            0.8421       [1.     1.     1.     1.     1.     0.0068]
            0.8947       [1.     1.     1.     1.     1.     0.2992]
            0.9474       [1.     1.     1.     1.     1.     0.7735]
            1.           [1. 1. 1. 1. 1. 1.]
        
        """
        self.alpha[:] = alpha
        
        num_bands = len(self.freq_bands)
        raw = alpha * num_bands - torch.arange(num_bands)
        self.window = 0.5 * (1 + torch.cos(np.pi * torch.clip(raw, 0.0, 1.0) + np.pi))
        self.freq_inds = (raw > 0).nonzero(as_tuple=True)[0]

def get_sinusoidal_embedder(n_frequencies, input_dim=3, annealed=False):
    if n_frequencies < 0:
        return nn.Identity(), input_dim

    embed_kwargs = {
        "include_input": True,
        "input_dim": input_dim,
        "max_freq_log2": n_frequencies - 1,
        "N_freqs": n_frequencies,
        "log_sampling": True
    }

    if annealed:
        embedder_obj = AnnealedSinusoidalEmbedder(**embed_kwargs)
    else:
        embedder_obj = SinusoidalEmbedder(**embed_kwargs)
    return embedder_obj, embedder_obj.out_features

if __name__ == "__main__":
    def unit_test():
        from icecream import ic
        import numpy as np
        np.set_printoptions(precision=4,suppress=True)
        embed_fn, input_ch = get_sinusoidal_embedder(-1)
        ic(embed_fn, input_ch) # Identity, 3

        embed_fn, input_ch = get_sinusoidal_embedder(0)
        ic(embed_fn, input_ch) # equivalent to Identity, 3
        
        embed_fn, input_ch = get_sinusoidal_embedder(1)
        ic(embed_fn, input_ch) # meaningful, 9

        embed_fn, input_ch = get_sinusoidal_embedder(6)
        embed_fn = embed_fn.cuda()
        x = torch.randn([100,3]).cuda()
        ic(embed_fn, input_ch) # meaningful, 39
        y = embed_fn(x)
        ic(y.shape)

        embed_fn, input_ch = get_sinusoidal_embedder(8, annealed=True)
        x = torch.randn([100,3]).cuda()
        embed_fn = embed_fn.cuda()
        embed_fn.set_cosine_easing_window(alpha=0.7777)
        ic(embed_fn, input_ch) # meaningful, 51
        y = embed_fn(x)
        ic(y.shape)

        #----------- Generate some example table
        # for alpha in np.linspace(0.0, 1.0, 20):
        #     embed_fn.set_cosine_easing_window(alpha=alpha)
        #     print(np.array([alpha]), embed_fn.window.data.cpu().numpy())

        #----------- Generate `window-alpha` plot of AnnealedSinusoidalEmbedder
        import matplotlib.pyplot as plt
        from tqdm import tqdm
        windows = []
        alphas = np.linspace(0.0, 1.0, 1000)
        for alpha in tqdm(alphas):
            embed_fn.set_cosine_easing_window(alpha=alpha)
            windows.append(embed_fn.window.data.cpu().numpy())
        windows = np.stack(windows, axis=1)
        fig = plt.figure()
        for i, w in enumerate(windows):
            plt.plot(alphas, w, label=f"freq=[{embed_fn.freq_bands[i].item()}]")
        plt.legend()
        plt.show()
    unit_test()
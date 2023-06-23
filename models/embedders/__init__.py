from .sinsoidal_pytorch import *
from .sinusoidal_cuda import *
from .spherical_harmonics import *

import torch.nn as nn
from typing import Tuple

def get_embedder(embed_cfg:dict, input_dim=3, use_tcnn_backend=None) -> Tuple[nn.Module, int]:
    embed_cfg = embed_cfg.copy()
    use_tcnn_backend = embed_cfg.pop('use_tcnn_backend', False if use_tcnn_backend is None else use_tcnn_backend)
    if (tp:=embed_cfg['type']) == 'none' or tp == 'identity':
        enc, n_encoded_dims = nn.Identity(), input_dim
    else:
        if use_tcnn_backend:
            from nr3d_lib.models.tcnn_adapter import TcnnEncoding, encoding_map
            """
            supported types: [sinusoidal, spherical, tiangle_wave, oneblob]
            """
            assert tp in encoding_map.keys(), f"[tcnn backend] Unsupported embeder type={tp}"
            enc = TcnnEncoding(input_dim, embed_cfg)
            n_encoded_dims = enc.out_features
        else:
            """
            supported types: [sinusoidal, spherical]
            """
            tp = embed_cfg.pop('type')
            if tp == 'spherical':
                enc = SHEncoder(input_dim=input_dim, **embed_cfg)
                n_encoded_dims = enc.out_features
            elif tp == 'sinusoidal':
                # Sinusoidal CUDA
                enc = FreqEncoder(input_dim=input_dim, **embed_cfg)
                n_encoded_dims = enc.out_features
            elif tp == 'sinusoidal_legacy':
                # Sinusoidal pytorch
                enc, n_encoded_dims = get_sinusoidal_embedder(input_dim=input_dim, **embed_cfg)
            else:
                raise RuntimeError(f"[pytorch backend] Unsupported embeder type={tp}")
    enc._embedder_type = tp
    return enc, n_encoded_dims

from .blocks import *

def get_blocks(in_features: int, out_features: int, use_tcnn_backend=None, use_lipshitz=False, **params):
    use_tcnn_backend = use_tcnn_backend or False
    if use_lipshitz:
        assert not use_tcnn_backend, "LipshitzMLP does not support tcnn backend"
        return LipshitzMLP(in_features, out_features, **params)
    
    if use_tcnn_backend:
        params.pop('weight_norm', None)
        from nr3d_lib.models.tcnn_adapter import get_tcnn_blocks
        return get_tcnn_blocks(in_features, out_features, **params)
    else:
        return FCBlock(in_features, out_features, **params)
"""
@file   permuto_nerf.py
@author Jianfei Guo, Shanghai AI Lab
@brief  NeRF++ network characterized using the Permutohedral-encoding model.
"""

__all__ = [
    'PermutoNeRFDistantModel'
]

from nr3d_lib.config import ConfigDict
from nr3d_lib.models.fields.nerf import PermutoNeRF
from nr3d_lib.models.fields_distant.nerf.renderer_mixin import NeRFRendererMixinDistant

class PermutoNeRFDistantModel(NeRFRendererMixinDistant, PermutoNeRF):
    """
    MRO:
    -> NeRFRendererMixinDistant
    -> PermutoNeRF
    -> ModelMixin
    -> nn.Module
    """
    pass
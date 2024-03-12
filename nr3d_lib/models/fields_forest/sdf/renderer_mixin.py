"""
@file   renderer_mixin.py
@author Jianfei Guo, Shanghai AI Lab
@brief  SDF renderer mixin based on Forest space & OctforestAS
"""

from typing import Any
from nr3d_lib.models.model_base import ModelMixin


class SDFRendererMixinForest(ModelMixin):
    def __init__(
        self, 
        # Renderer mixin kwargs
        ray_query_cfg: dict = dict(), 
        # Network kwargs
        **net_kwargs) -> None:
        
        mro = type(self).mro()
        super_class = mro[mro.index(SDFRendererMixinForest)+1]
        assert super_class is not ModelMixin, "Incorrect class inheritance. Three possible misuse scenarios:\n"\
            "Case 1: The Net class for mixin should also inherit from `ModelMixin`.\n"\
            "Case 2: RendererMixin should come before the Net class when inheriting.\n"\
            "Case 3: You should not directly instantiate this mixin class."
        
        raise NotImplementedError
        
        super().__init__(**net_kwargs)
        
        self.ray_query_cfg = ray_query_cfg
        
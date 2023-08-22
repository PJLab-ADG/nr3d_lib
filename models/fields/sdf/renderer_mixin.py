"""
@file   renderer_mixin.py
@author Jianfei Guo, Shanghai AI Lab
@brief  SDF renderer mixin based on AABB space & OctreeAS
"""

from nr3d_lib.config import ConfigDict
from nr3d_lib.models.base import ModelMixin

class SDFRendererMixin(ModelMixin):
    """
    SDF Renderer Mixin class
    
    NOTE: This is a mixin class!
    Refer: https://stackoverflow.com/questions/533631/what-is-a-mixin-and-why-are-they-useful
    !!!!: The target class for this mixin should also inherit from `ModelMixin`.
    """
    def __init__(
        self, 
        # Renderer mixin kwargs
        ray_query_cfg: ConfigDict = ConfigDict(), 
        # Network kwargs
        **net_kwargs) -> None:
        
        mro = type(self).mro()
        super_class = mro[mro.index(SDFRendererMixin)+1]
        assert super_class is not ModelMixin, "Incorrect class inheritance. Three possible misuse scenarios:\n"\
            "Case 1: The Net class for mixin should also inherit from `ModelMixin`.\n"\
            "Case 2: RendererMixin should come before the Net class when inheriting.\n"\
            "Case 3: You should not directly instantiate this mixin class."
        
        raise NotImplementedError
    
        super().__init__(**net_kwargs)

        self.ray_query_cfg = ray_query_cfg

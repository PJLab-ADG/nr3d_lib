"""
@file   nerf.py
@author Jianfei Guo, Shanghai AI Lab
@brief  MLP-based NeRF++
"""

__all__ = [
    'NeRFDistantFramework'
]

from nr3d_lib.config import ConfigDict
from nr3d_lib.models.fields.nerf import EmbededNeRF
from nr3d_lib.models.fields_distant.nerf.renderer_mixin import nerf_distant_renderer_mixin

class NeRFDistantFramework(nerf_distant_renderer_mixin, EmbededNeRF):
    def __init__(self, *args, mixin_cfg=ConfigDict(), **kwargs) -> None:
        EmbededNeRF.__init__(self, *args, **kwargs)
        nerf_distant_renderer_mixin.__init__(self, **mixin_cfg) 

    # def ray_test(self, *args, **kwargs):
    #     if self.cr_obj is not None:
    #         # NOTE: nerf++ background should always directly use foreground's ray_test results.
    #         return self.cr_obj.model.ray_test(*args, **kwargs)
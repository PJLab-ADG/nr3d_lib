"""
@file   lotd_nerf.py
@author Jianfei Guo, Shanghai AI Lab
@brief  LoTD-encoding-based NeRF++
"""

__all__ = [
    'LoTDNeRFDistantFramework'
]

from nr3d_lib.config import ConfigDict
from nr3d_lib.models.fields.nerf import LoTDNeRF
from nr3d_lib.models.fields_distant.nerf.renderer_mixin import nerf_distant_renderer_mixin

class LoTDNeRFDistantFramework(nerf_distant_renderer_mixin, LoTDNeRF):
    def __init__(self, *args, mixin_cfg=ConfigDict(), **kwargs) -> None:
        LoTDNeRF.__init__(self, *args, **kwargs)
        nerf_distant_renderer_mixin.__init__(self, **mixin_cfg) 

    # def ray_test(self, *args, **kwargs):
    #     if self.cr_obj is not None:
    #         # NOTE: nerf++ background should always directly use foreground's ray_test results.
    #         return self.cr_obj.model.ray_test(*args, **kwargs)
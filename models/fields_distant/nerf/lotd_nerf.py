"""
@file   lotd_nerf.py
@author Jianfei Guo, Shanghai AI Lab
@brief  LoTD-encoding-based NeRF++
"""

__all__ = [
    'LoTDNeRFDistantModel'
]

from nr3d_lib.config import ConfigDict
from nr3d_lib.models.fields.nerf import LoTDNeRF
from nr3d_lib.models.fields_distant.nerf.renderer_mixin import NeRFDistantRendererMixin

class LoTDNeRFDistantModel(NeRFDistantRendererMixin, LoTDNeRF):
    """
    MRO: NeRFDistantRendererMixin -> LoTDNeRF -> ModelMixin -> nn.Module
    """
    pass
    # def ray_test(self, *args, **kwargs):
    #     if self.cr_obj is not None:
    #         # NOTE: nerf++ background should always directly use foreground's ray_test results.
    #         return self.cr_obj.model.ray_test(*args, **kwargs)
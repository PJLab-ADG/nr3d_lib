"""
@file   dynamic_generative_permuto_neus.py
@author Jianfei Guo, Shanghai AI Lab
@brief  _description_
"""

__all__ = [
    'DynamicGenerativePermutoConcatNeuS', 
    'DynamicGenerativePermutoConcatNeuSModel', 
]

from nr3d_lib.models.fields_conditional.neus import GenerativePermutoConcatNeuS
from nr3d_lib.models.fields_conditional_dynamic.neus.renderer_mixin import NeusRendererMixinBatchedDynamic

class DynamicGenerativePermutoConcatNeuS(GenerativePermutoConcatNeuS):
    # def __init__(self, *args, **kwargs) -> None:
    #     super().__init__(*args, **kwargs)
    def populate(self, *args, **kwargs):
        self.surface_cfg['encoding_cfg'].setdefault(
            'space_cfg', {'type': 'batched_dynamic'})
        return super().populate(*args, **kwargs)
    def query_sdf(self, *args, **kwargs):
        kwargs.pop('ts', None) # `ts` is already taken care of when computing `z`
        return super().query_sdf(*args, **kwargs)
    def forward_sdf(self, *args, **kwargs):
        kwargs.pop('ts', None) # `ts` is already taken care of when computing `z`
        return super().forward_sdf(*args, **kwargs)
    def forward_sdf_nablas(self, *args, **kwargs):
        kwargs.pop('ts', None) # `ts` is already taken care of when computing `z`
        return super().forward_sdf_nablas(*args, **kwargs)
    def forward(self, *args, **kwargs):
        kwargs.pop('ts', None) # `ts` is already taken care of when computing `z`
        return super().forward(*args, **kwargs)

class DynamicGenerativePermutoConcatNeuSModel(NeusRendererMixinBatchedDynamic, DynamicGenerativePermutoConcatNeuS):
    """
    MRO:
    -> NeusRendererMixinBatchedDynamic
    -> DynamicGenerativePermutoConcatNeuS
    -> GenerativePermutoConcatNeuS
    -> ModelMixin
    -> nn.Module
    """
    pass

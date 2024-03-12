"""
@file   autodecoder.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Generic Auto-Decoder Mixin modules.
"""

from typing import Dict
from copy import deepcopy

import torch
import torch.nn as nn
from nr3d_lib.models.embeddings import Embedding

from nr3d_lib.utils import check_to_torch
from nr3d_lib.models.model_base import ModelMixin

class AutoDecoderMixin(ModelMixin):
    """
    The resulting MRO:
        AD_xxxModel -> AutoDecoderMixin -> xxxModel -> xxxRendererMixin -> xxxNet -> ModelMixin -> nn.Module
    """
    def __init__(self, latents_cfg: dict = None, **model_params):
        mro = type(self).mro()
        super_class = mro[mro.index(AutoDecoderMixin)+1]
        assert super_class is not ModelMixin, "Incorrect class inheritance. Three possible misuse scenarios:\n"\
            "Case 1: The Net class for mixin should also inherit from `ModelMixin`.\n"\
            "Case 2: AutoDecoderMixin should come before the Net class when inheriting.\n"\
            "Case 3: You should not directly instantiate this mixin class."
        super().__init__(**model_params)
        # assert latents_cfg is not None, f"`latents_cfg` is required for {self.__class__.__name__}"
        self.latents_cfg = deepcopy(latents_cfg) if latents_cfg is not None else None

    def autodecoder_populate(self, key_maps: Dict[str, list], latent_maps: Dict[str, Embedding]):
        self._keys = key_maps
        self._index_maps = {kk: {v:i for i,v in enumerate(vv)} for kk, vv in self._keys.items()}
        # self._latents = nn.ParameterDict(latent_maps)
        self._latents = nn.ModuleDict(latent_maps)

    # override
    def state_dict(self, destination=None, prefix: str='', keep_vars=False):
        # Re-organize state_dict with _latent and _models
        if destination is None:
            destination = dict()
        model_dict = super().state_dict(destination=None, prefix='', keep_vars=keep_vars)
        destination[prefix + '_latents'] = dict()
        for k, _ in self._latents.named_parameters():
            destination[prefix + '_latents'][k] = model_dict.pop('_latents.' + k)
        destination[prefix + '_models'] = model_dict

        # Other stuff
        destination[prefix + '_keys'] = self._keys
        return destination

    # override
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, *args, **kwargs):
        # Re-organize state_dict in pytorch's favor
        if prefix + '_latents' in state_dict:
            latent_pnames = [k for k, _ in self._latents.named_parameters()]
            latent_dict = state_dict.pop(prefix + '_latents')
            for k in latent_pnames:
                if k in latent_dict:
                    state_dict[prefix + '_latents' + '.' + k] = latent_dict[k]
        if prefix + '_models' in state_dict:
            model_dict = state_dict.pop(prefix + '_models')
            for k in model_dict:
                state_dict[prefix + k] = model_dict[k]
        
        # Other stuff. TODO: make below more auto-matic
        if prefix + '_keys' in state_dict:
            self._keys = state_dict.pop(prefix + '_keys')
            self._index_maps = {kk: {v:i for i,v in enumerate(vv)} for kk, vv in self._keys.items()}
        elif strict:
            missing_keys.append(prefix + '_keys')
        
        # Call original pytorch's load_state_dict
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, *args, **kwargs)

if __name__ == "__main__":
    def unit_test():
        pass
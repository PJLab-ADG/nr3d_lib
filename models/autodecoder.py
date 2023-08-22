"""
@file   autodecoder.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Generic Auto-Decoder Mixin modules.

        Every categorical auto-decoder model should contain three components:
        - the model itself
        - a key list, which could be converted into a key-indices pair dict for mapping string names to integer indices
        - a latent pool, with each corresponds to a unique key in the key list.
"""

from typing import Iterable, List, Union, Dict

import torch
import torch.nn as nn

from nr3d_lib.utils import import_str
from nr3d_lib.config import ConfigDict
from nr3d_lib.models.base import ModelMixin

class AutoDecoderMixin(ModelMixin):
    """
    The resulting MRO:
        AD_xxxModel -> AutoDecoderMixin -> xxxModel -> xxxRendererMixin -> xxxNet -> ModelMixin -> nn.Module
    """
    def __init__(self, key_list: List[str], latents: ConfigDict, **model_params):
        
        mro = type(self).mro()
        super_class = mro[mro.index(AutoDecoderMixin)+1]
        assert super_class is not ModelMixin, "Incorrect class inheritance. Three possible misuse scenarios:\n"\
            "Case 1: The Net class for mixin should also inherit from `ModelMixin`.\n"\
            "Case 2: AutoDecoderMixin should come before the Net class when inheriting.\n"\
            "Case 3: You should not directly instantiate this mixin class."
        
        super().__init__(**model_params)

        self._latents_cfg = latents
        self._latents = nn.ModuleDict()
        for latent_name, latent_cfg in latents.items():
            self._latents[latent_name] = nn.Embedding(len(key_list), latent_cfg['dim'])
        
        self._keys = key_list
        self._keys_prob = [0.] * len(self._keys)
        self._ind_map = {key: i for i,key in enumerate(self._keys)}

    def set_condition(self, batched_info={}):
        if 'latents' not in batched_info:
            assert 'keys' in batched_info
            batched_info['latents'] = self.get_latents(batched_info['keys'])
        else:
            assert set(self._latents.keys()) == set(batched_info['latents'].keys())
        super().set_condition(batched_info)

    def get_inds(self, keys: Union[str, List[str], dict]):
        if isinstance(keys, dict):
            keys = keys['keys']
        if isinstance(keys, str):
            keys = [keys]
        inds = [self._ind_map[key] for key in keys]
        inds = torch.tensor(inds).long().to(self.device)
        return inds

    def get_latents(self, keys: Union[str, List[str]]):
        inds = self.get_inds(keys)
        return {lname: embeding(inds) for lname, embeding in self._latents.items()}

    def get_obj_infos(self, keys: Union[str, List[str]]):
        obj_infos = dict()
        obj_infos['inds'] = self.get_inds(keys)
        obj_infos['latents'] = self.get_latents(keys)
        return obj_infos

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
        destination[prefix + '_keys_prob'] = self._keys_prob
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
            self._ind_map = {key: i for i,key in enumerate(self._keys)}
        elif strict:
            missing_keys.append(prefix + '_keys')
        if prefix + '_keys_prob' in state_dict:
            self._keys_prob = state_dict.pop(prefix + '_keys_prob')
        elif strict:
            missing_keys.append(prefix + '_keys_prob')
        
        # Call original pytorch's load_state_dict
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, *args, **kwargs)

    # def latent_sampler(self, batch_size=None):
    #     if batch_size is None:
    #         prefix = []
    #     else:
    #         prefix = [batch_size]
    #     return {lname: torch.randn([*prefix, lcfg['dim']], device=self.device, dtype=self.dtype) \
    #                 for lname, lcfg in self._latents_cfg.items()}

def create_autodecoder(key_list: List[str], framework: ConfigDict, latents: ConfigDict = None):
    model_cls = import_str(framework.model_class)
    mixined_cls_name = "AD_" + model_cls.__name__
    mixined_model_class = type(mixined_cls_name, (AutoDecoderMixin, model_cls), {'_model_cls': model_cls})
    return mixined_model_class(key_list, latents, **framework.model_params)

# NOTE: Temporary backward-compatibility.
AutoDecoderModule = create_autodecoder

if __name__ == "__main__":
    def test():
        from addict import Dict
        from icecream import ic
        dummy_cfg = Dict()
        dummy_cfg.framework.target = 'models.categorical.PIGAN'
        dummy_cfg.framework.param = {}
        dummy_cfg.latents.shape.dim = 128
        dummy_cfg.latents.appearance.dim = 128
        # dummy_cfg.freeze()
        model = create_autodecoder(['obj1', 'obj2', ], dummy_cfg).to('cuda:0')
        model.set_condition({'keys':'obj2'})
        ret = model.forward_sigma_rgb(x=torch.randn([7, 3]).cuda(), v=torch.randn([7,3]).cuda())
        ic(ret)
    test()
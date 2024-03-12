"""
Supports keyframes and different "timestamp" indexing and tweening (in a more general sense, arbitrary floating position at a general 1D axis)
"""

__all__ = [
    'SeqEmbedding', 
    'MultiSeqEmbeddingShared', 
    'MultiSeqEmbeddingIndividual', 
]

import numpy as np
from numbers import Number
from typing import List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn

from nr3d_lib.utils import check_to_torch, torch_consecutive_nearest1d, torch_consecutive_interp1d

from .embedding import Embedding

class SeqEmbedding(Embedding):
    def __init__(
        self, 
        ts_keyframes: Union[List[float], np.ndarray], 
        v_keyframes: Union[np.ndarray, torch.Tensor] = None, 
        *, 
        dim: int, 
        learnable: bool = True, 
        weight_init: Union[str, dict] = 'normal', 
        device=None, dtype=torch.float, **kwargs
        ) -> None:
        
        ts_keyframes = check_to_torch(ts_keyframes, device=device)
        if v_keyframes is not None:
            v_keyframes = check_to_torch(v_keyframes, device=device)
        
        num_embeddings = len(ts_keyframes)
        super().__init__(num_embeddings, dim, weight=v_keyframes, weight_init=weight_init, learnable=learnable, device=device, dtype=dtype, **kwargs)
        
        self.register_buffer('ts_keyframes', ts_keyframes, persistent=True)
    
    def nearest(self, ts: Union[Number, torch.Tensor], mode: Literal['nearest', 'ceil', 'floor'] = 'nearest'):
        ts = check_to_torch(ts, device=self.device)
        inds = torch_consecutive_nearest1d(self.ts_keyframes, ts.to(self.ts_keyframes), mode=mode)[0]
        return self.forward(inds)

    def interp(self, ts: Union[Number, torch.Tensor]):
        ts = check_to_torch(ts, device=self.device, dtype=self.ts_keyframes.dtype)
        v = torch_consecutive_interp1d(self.ts_keyframes, self.weight, ts)
        return v

    def forward(self, ts: Union[Number, torch.Tensor], mode: Literal['interp', 'nearest', 'ceil', 'floor'] = 'interp'):
        if mode == 'interp':
            return self.interp(ts)
        else:
            return self.nearest(ts, mode=mode)

    def sample(self, num: int, ts_single: Number = None):
        raise NotImplementedError
    
    def get_z_per_input(
        self, 
        input_prefix: Union[Tuple, List], 
        ts_single: Number = None, 
        ts_per_input: torch.Tensor = None, 
        z_single: torch.Tensor = None, 
        z_per_input: torch.Tensor = None, 
        **unused_kwargs
        ):
        """
        A helper function to get the embedding of each input point from various input combinations

        # input_prefix: Tuple, # Dimensions of the first few dimensions excluding the last one
        # Priority order:`z_per_input` > `z_single` > `ts_per_input` > `ts_single`

        Args:
            input_prefix (Tuple): _description_
            ts_single (Number, optional): _description_. Defaults to None.
            ts_per_input (torch.Tensor, optional): _description_. Defaults to None.
            z_single (torch.Tensor, optional): _description_. Defaults to None.
            z_per_input (torch.Tensor, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        
        input_prefix = [*input_prefix] # Convert to list of int
        
        if z_per_input is not None:
            assert [*z_per_input.shape[:-1]] == input_prefix, f"`z_per_input` should be of shape [{','.join([str(i) for i in input_prefix])},{self.embedding_dim}]"
        elif z_single is not None:
            z_per_input = z_single.view(*[1]*len(input_prefix),-1).expand(*input_prefix, -1)
        elif ts_per_input is not None:
            assert [*ts_per_input.shape] == input_prefix, f"`ts_per_input` should be of shape [{','.join([str(i) for i in input_prefix])}]"
            z_per_input = self.interp(ts_per_input)
        elif ts_single is not None:
            z_single = self.interp(ts_single)
            return self.get_z_per_input(input_prefix, z_single=z_single)
        else:
            raise RuntimeError("You should specify one of [ts_single, ts_per_input, z_single, z_per_input]")
        return z_per_input

class MultiSeqEmbeddingShared(Embedding):
    def __init__(
        self, 
        num_instances: int, 
        ts_keyframes: Union[List[float], np.ndarray], 
        v_keyframes: Union[np.ndarray, torch.Tensor] = None, 
        *, 
        dim: int, 
        learnable: bool = True, 
        weight_init: Union[str, dict] = 'normal', 
        device=None, dtype=torch.float, **kwargs
        ) -> None:
        """
        Multi-instance sequential embeddings with shared time axis

        Args:
            num_instances (int): _description_
            ts_keyframes (Union[List[float], np.ndarray]): _description_
            dim (int): _description_
            v_keyframes (Union[np.ndarray, torch.Tensor], optional): _description_. Defaults to None.
            weight_init (str, optional): _description_. Defaults to 'normal'.
            learnable (bool, optional): _description_. Defaults to True.
            device (_type_, optional): _description_. Defaults to None.
            dtype (_type_, optional): _description_. Defaults to torch.float.
        """
        
        ts_keyframes = check_to_torch(ts_keyframes, device=device)
        if v_keyframes is not None:
            v_keyframes = check_to_torch(v_keyframes, device=device)
        
        num_embeddings = len(ts_keyframes)
        super().__init__(num_embeddings, dim, weight=v_keyframes, weight_init=weight_init, learnable=learnable, device=device, dtype=dtype, **kwargs)
        
        self.register_buffer('ts_keyframes', ts_keyframes, persistent=True)

    def nearest(self, ts: Union[Number, torch.Tensor], ins_inds=None, mode: Literal['nearest', 'ceil', 'floor'] = 'nearest'):
        ts = check_to_torch(ts, device=self.device)
        inds = torch_consecutive_nearest1d(self.ts_keyframes, ts.to(self.ts_keyframes), mode=mode)[0]
        return Embedding.forward(self, inds)

    def interp(self, ts: Union[Number, torch.Tensor], ins_inds=None):
        ts = check_to_torch(ts, device=self.device, dtype=self.ts_keyframes.dtype)
        v = torch_consecutive_interp1d(self.ts_keyframes, self.weight, ts)
        return v

    def forward(self, ts: Union[Number, torch.Tensor], ins_inds=None, mode: Literal['interp', 'nearest', 'ceil', 'floor'] = 'interp'):
        if mode == 'interp':
            return self.interp(ts, ins_inds=ins_inds)
        else:
            return self.nearest(ts, ins_inds=ins_inds, mode=mode)

    def sample_batched(self, batch_size: int, num_per_batch: int, ts_per_batch: torch.Tensor = None):
        raise NotImplementedError

    def check_or_get_z_per_input_batched(
        self, 
        input_prefix: Union[Tuple, List], 
        bidx: torch.Tensor = None,
        ts_per_batch: torch.Tensor = None, 
        ts_per_input: torch.Tensor = None, 
        z_per_batch: torch.Tensor = None, 
        z_per_input: torch.Tensor = None, 
        **unused_kwargs
        ):
        """

        Args:
            input_prefix (Tuple): _description_
            bidx (torch.Tensor, optional): batch indices per input. Defaults to None.
            ts_per_batch (torch.Tensor, optional): _description_. Defaults to None.
            ts_per_input (torch.Tensor, optional): _description_. Defaults to None.
            z_per_batch (torch.Tensor, optional): _description_. Defaults to None.
            z_per_input (torch.Tensor, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        
        input_prefix = [*input_prefix] # Convert to list of int
        if bidx is not None: 
            # The given `bidx` should correspond to each of the input 
            assert [*bidx.shape] == input_prefix, f"`bidx` should be of shape [{','.join([str(i) for i in input_prefix])}]"
        else:
            # If not given `bidx`, then the first dim of x should be batch_size
            batch_size = input_prefix[0]
        
        if z_per_input is not None:
            assert [*z_per_input.shape[:-1]] == input_prefix, f"`z_per_input` should be of shape [{','.join([str(i) for i in input_prefix])},{self.embedding_dim}]"
            pass
        elif z_per_batch is not None:
            if bidx is not None:
                z_per_input = z_per_batch[bidx]
            else:
                z_per_input = z_per_batch.view(batch_size,*[1]*(len(input_prefix)-1),-1).expand(*input_prefix, -1)
        elif ts_per_input is not None:
            assert [*ts_per_input.shape] == input_prefix, f"`ts_per_input` should be of shape [{','.join([str(i) for i in input_prefix])}]"
            z_per_input = self.interp(ts_per_input)
        elif ts_per_batch is not None:
            if bidx is not None:
                ts_per_input = ts_per_batch[bidx]
            else:
                ts_per_input = ts_per_batch.view(batch_size, *[1]*(len(input_prefix)-1)).expand(*input_prefix)
            return self.check_or_get_z_per_input_batched(input_prefix, bidx=bidx, ts_per_input=ts_per_input)
        # elif ts_single is not None:
        #     if not isinstance(ts_single, Number): ts_single = ts_single.item() 
        #     ts_per_input = torch.full(input_prefix, ts_single, device=self.device)
        #     return self.check_or_get_z_per_input_batched(input_prefix, bidx=bidx, ts_per_input=ts_per_input)
        else:
            raise RuntimeError("You should specify one of [ts_per_batch, ts_per_input, z_per_batch, z_per_input]")
        
        return z_per_input

class MultiSeqEmbeddingIndividual(Embedding):
    def __init__(
        self, 
        num_instances: int, 
        ts_keyframes: Union[List[float], np.ndarray], 
        v_keyframes: Union[np.ndarray, torch.Tensor] = None, 
        *, 
        dim: int, 
        learnable: bool = True, 
        weight_init: Union[str, dict] = 'normal', 
        device=None, dtype=torch.float, **kwargs
        ) -> None:
        """
        Multi-instance sequential embeddings with unique individual time axis
        
        Applicable for obtaining Embedding for multiple instances, supports batched retrieval
        Assumes each object is within its own time embedding

        Args:
            dim (int): _description_
            ts_keyframes (Union[List[float], np.ndarray]): _description_
            v_keyframes (Union[np.ndarray, torch.Tensor], optional): _description_. Defaults to None.
            weight_init (str, optional): _description_. Defaults to 'normal'.
            learnable (bool, optional): _description_. Defaults to True.
            device (_type_, optional): _description_. Defaults to None.
            dtype (_type_, optional): _description_. Defaults to torch.float.
        """
        
        ts_keyframes = check_to_torch(ts_keyframes, device=device)
        if v_keyframes is not None:
            v_keyframes = check_to_torch(v_keyframes, device=device)
        
        # NOTE: Individual sequential embedding for each instance.
        num_embeddings = len(ts_keyframes) * num_instances
        super().__init__(num_embeddings, dim, weight=v_keyframes, weight_init=weight_init, learnable=learnable, device=device, dtype=dtype, **kwargs)
        
        self.register_buffer('ts_keyframes', ts_keyframes, persistent=True)

    def nearest(self, ts: torch.Tensor, ins_inds: torch.Tensor, mode: Literal['nearest', 'ceil', 'floor'] = 'nearest'):
        assert ins_inds is not None, 'Requires `ins_inds`'
        ti = torch_consecutive_nearest1d(self.ts_keyframes, ts, mode=mode)[0]
        vi = ti + ins_inds * len(self.ts_keyframes)
        vals = Embedding.forward(self, vi)
        return vals

    def interp(self, ts: torch.Tensor, ins_inds: torch.Tensor):
        assert ins_inds is not None, 'Requires `ins_inds`'
        
        # NOTE: Similar to `torch_consecutive_interp1d`, but with offsets
        ti = torch.searchsorted(self.ts_keyframes, ts) # in range [0, len]
        ti_below, ti_above = torch.clamp_min(ti-1, 0), torch.clamp_max(ti, len(self.ts_keyframes)-1)
        
        # NOTE `ti_g` and `vi_g` are different;
        #      `ti_g` is the local index of an instance, 
        #      `vi_g` is the global or holistic index of the instance
        ti_g = torch.stack([ti_below, ti_above], -1)
        vi_g = ti_g + ins_inds * len(self.ts_keyframes)
        bins_g = self.ts_keyframes[ti_g]
        vals_g = Embedding.forward(self, vi_g)
        
        denom = bins_g[:, 1] - bins_g[:, 0]
        w  = (ts - bins_g[:, 0]) / denom.clamp_min(1e-5)
        w = w.view(*w.shape, 1)
        vals = vals_g[:, 0] + w * (vals_g[:, 1] - vals_g[:, 0])
        return vals

    def forward(self, ts: Union[Number, torch.Tensor], ins_inds=None, mode: Literal['interp', 'nearest', 'ceil', 'floor'] = 'interp'):
        if mode == 'interp':
            return self.interp(ts, ins_inds=ins_inds)
        else:
            return self.nearest(ts, ins_inds=ins_inds, mode=mode)

    def sample_batched(self, batch_size: int, num_per_batch: int, ts_per_batch: torch.Tensor = None):
        raise NotImplementedError

    def check_or_get_z_per_input_batched(
        self, 
        input_prefix: Union[Tuple, List], 
        bidx: torch.Tensor = None,
        ins_inds_per_batch: torch.Tensor = None, 
        ins_inds_per_input: torch.Tensor = None, 
        ts_per_batch: torch.Tensor = None, 
        ts_per_input: torch.Tensor = None, 
        z_per_batch: torch.Tensor = None, 
        z_per_input: torch.Tensor = None, 
        **unused_kwargs
        ):

        input_prefix = [*input_prefix] # Convert to list of int
        if bidx is not None: 
            # The given `bidx` should correspond to each of the input
            assert [*bidx.shape] == input_prefix, f"`bidx` should be of shape [{','.join([str(i) for i in input_prefix])}]"
        else:
            # If not given `bidx`, then the first dim of x should be batch_size
            batch_size = input_prefix[0]
        
        if z_per_input is not None:
            assert [*z_per_input.shape[:-1]] == input_prefix, f"`z_per_input` should be of shape [{','.join([str(i) for i in input_prefix])},{self.embedding_dim}]"
            pass
        elif z_per_batch is not None:
            if bidx is not None:
                z_per_input = z_per_batch[bidx]
            else:
                z_per_input = z_per_batch.view(batch_size,*[1]*(len(input_prefix)-1),-1).expand(*input_prefix, -1)
        else:
            # If both `z_per_xxx` are not given, we need to get z according to `ts` and `ins_inds`
            if ins_inds_per_input is not None:
                pass
            else:
                assert ins_inds_per_batch is not None, 'Requires `ins_inds_per_batch` to get `ins_inds_per_input`'
                if bidx is not None:
                    ins_inds_per_input = ins_inds_per_batch[bidx]
                else:
                    ins_inds_per_input = ins_inds_per_batch.view(batch_size,*[1]*(len(input_prefix)-1)).expand(*input_prefix)
            
            if ts_per_input is not None:
                assert [*ts_per_input.shape] == input_prefix, f"`ts_per_input` should be of shape [{','.join([str(i) for i in input_prefix])}]"
                z_per_input = self.interp(ts_per_input, ins_inds_per_input)
            elif ts_per_batch is not None:
                if bidx is not None:
                    ts_per_input = ts_per_batch[bidx]
                else:
                    ts_per_input = ts_per_batch.view(batch_size, *[1]*(len(input_prefix)-1)).expand(*input_prefix)
                z_per_input = self.interp(ts_per_input, ins_inds_per_input)
        
        return z_per_input

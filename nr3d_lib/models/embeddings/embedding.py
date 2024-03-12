"""

"""

__all__ = [
    'Embedding', 
]

from numbers import Number
from typing import Literal, Optional, Tuple, Union

import torch
from torch import Tensor
import torch.nn as nn

from nr3d_lib.utils import check_to_torch, torch_consecutive_nearest1d, torch_consecutive_interp1d

class Embedding(nn.Embedding):
    def __init__(
        self, num_embeddings: int, dim: int, *, 
        weight: torch.Tensor = None, 
        learnable: bool = True, 
        weight_init: Union[str, dict] = 'normal', 
        device=None, dtype=torch.float, **kwargs, 
        ) -> None:
        
        if weight is not None:
            weight = check_to_torch(weight, dtype=dtype, device=device)
            assert [*weight.shape] == [num_embeddings, dim]
        else:
            weight = torch.empty([num_embeddings, dim], dtype=dtype, device=device)
            if isinstance(weight_init, str):
                weight_init = {'type': weight_init}
            weight_init_method = weight_init['type']
            if weight_init_method == 'normal':
                mean = weight_init.get('mean', 0)
                std = weight_init.get('std', 1)
                weight.normal_(mean=mean, std=std)
            elif weight_init_method == 'uniform':
                bound = weight_init.get('bound', 1)
                weight.uniform_(-bound, bound)
            elif weight_init_method == 'zero':
                weight.zero_()
            elif weight_init_method == 'linspace':
                start = weight_init.get('start', -1)
                end = weight_init.get('end', 1)
                weight = torch.linspace(start, end, num_embeddings, dtype=dtype, device=device)
                # Expand to all `dim`
                weight = weight.unsqueeze(-1).expand(num_embeddings, dim)
            else:
                raise RuntimeError(f"Invalid weight_init_method={weight_init_method}")
        
        super().__init__(
            num_embeddings=num_embeddings, embedding_dim=dim, 
            _weight=weight, dtype=dtype, device=device)
        
        if not learnable:
            self.weight.requires_grad = False
        
        i_keyframes = torch.arange(num_embeddings, dtype=torch.float, device=device, **kwargs)
        self.register_buffer('i_keyframes', i_keyframes, persistent=False)

    @property
    def dtype(self):
        return self.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.weight.device

    def nearest(
        self, i: Union[Number, torch.Tensor], *, 
        mode: Literal['nearest', 'ceil', 'floor'] = None):
        
        """ More flexible indexing that allow for floating input and different rounding behaviors

        Returns:
            _type_: _description_
        """
        
        i = check_to_torch(i, device=self.device)
        if mode is None:
            i = i.long()
        else:
            if mode == 'ceil':
                i = i.ceil().long()
            elif mode == 'floor':
                i = i.long()
            elif mode == 'nearest':
                i = (i + 0.5).long()
            else:
                raise RuntimeError(f"Invalid mode={mode}")
        
        return nn.Embedding.forward(self, i)
    
    def interp(self, i: torch.Tensor):
        """ Perform 1d interpolation at the specified floating index `i` (no extrapolation)

        Args:
            i (torch.Tensor): The given floating index in range [0, num_embeddings-1]
        """
        
        i = check_to_torch(i, device=self.device, dtype=self.i_keyframes.dtype)
        v = torch_consecutive_interp1d(self.i_keyframes, self.weight, i)
        return v

    def check_or_get_z_per_input_batched(
        self, 
        input_prefix: Tuple, 
        bidx: torch.Tensor = None,
        z_per_batch: torch.Tensor = None, 
        z_per_input: torch.Tensor = None, 
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
            raise RuntimeError("You should specify one of [z_per_batch, z_per_input]")
        
        return z_per_input

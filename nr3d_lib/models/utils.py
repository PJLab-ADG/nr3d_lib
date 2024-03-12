"""
@file   utils.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Neural-network-level utililities.
"""

__all__ = [
    'BufferList', 
    'ParameterList', 
    'ConfigBuffer', 
    'get_jacobian', 
    'zero_weights_init', 
    'calc_grad_norm', 
    'count_trainable_parameters', 
    'get_optimizer', 
    'SimpleScheduler', 
    'get_lr_func_multi_step', 
    'get_lr_func_warmup_cosine', 
    'get_lr_func_exponential', 
    'get_lr_func_exponential_plenoxels', 
    'get_scheduler', 
    'batchify_query', 
    'batchify_query_ray_pts', 
    'clip_norm_', # In-place
]

import re
import math
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from numbers import Number
from bisect import bisect_right
from typing import Callable, Dict, Iterable, List, NamedTuple, Tuple, Union

import torch
import torch.nn as nn
from torch import optim
from torch import autograd

from nr3d_lib.fmt import log
from nr3d_lib.config import ConfigDict
from nr3d_lib.utils import is_scalar, nested_dict, nested_dict_items, nested_dict_keys, nested_dict_values, import_str

class BufferList(nn.Module):
    """
    BufferList, similar to nn.ModuleList but for nn.Buffer
    """
    def __init__(self, lst=[], persistent=False) -> None:
        super().__init__()
        for i, p in enumerate(lst):
            self.register_buffer(f'tensor_{i}', p, persistent=persistent)
        self.n = len(lst)

    def __getitem__(self, index):
        assert is_scalar(index), 'only support scalar index'
        index = int(index)
        return getattr(self, f'tensor_{index}')

    def __setitem__(self, index, val):
        assert is_scalar(index), 'only support scalar index'
        index = int(index)
        setattr(self, f'tensor_{index}', val)

    def append(self, item):
        self.register_buffer(f'tensor_{self.n}', item)
        self.n += 1

    def __iter__(self):
        for i in range(self.n):
            yield self[i]
            
class ParameterList(nn.Module):
    """
    ParameterList, similar to nn.ModuleList but for nn.Parameter
    """
    def __init__(self, lst=[]) -> None:
        super().__init__()
        for i, p in enumerate(lst):
            setattr(self, f'tensor_{i}', torch.nn.Parameter(p))
        self.n = len(lst)

    def __getitem__(self, index):
        assert is_scalar(index), 'only support scalar index'
        index = int(index)
        return getattr(self, f'tensor_{index}')

    def __setitem__(self, index, val):
        assert is_scalar(index), 'only support scalar index'
        index = int(index)
        setattr(self, f'tensor_{index}', val)

    def append(self, item):
        self.register_buffer(f'tensor_{self.n}', item)
        self.n += 1

    def __iter__(self):
        for i in range(self.n):
            yield self[i]

class ConfigBuffer(nn.Module):
    """
    A helper class to save and load model configs, conviniently using pytorch's API and recursive mechanisms
    """
    def __init__(self, config: dict = {}) -> None:
        super().__init__()
        self.config = config
        msg = "Input config must be a (nested) contains only numbers, strings, or lists of number / strings"
        assert isinstance(config, dict), msg + f", but got {type(config)}"
        for *k, v in nested_dict_items(config):
            if isinstance(v, (Number, str)):
                pass
            elif isinstance(v, (list, tuple)) and all([isinstance(i, (Number, str)) for i in v]):
                pass
            else:
                raise RuntimeError(msg + f", but got {'.'.join(k)}: {type(v)}")
        
        self._register_state_dict_hook(self.state_dict_hook)
        self._register_load_state_dict_pre_hook(self.before_load_state_dict)

    @staticmethod # NOTE: "self" is also passed when pytorch call hooks
    def state_dict_hook(self, state_dict: dict, prefix, local_metadata):
        # config_state_dict = {(prefix + '.'.join('config', *k)): v for *k, v in nested_dict_items(self.config)}
        # state_dict.update(config_state_dict)
        state_dict[prefix + 'config'] = self.config
    
    def before_load_state_dict(self, state_dict: dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # config_keys = [k for k in state_dict.keys() if k.startswith(prefix + 'config.')]
        # self.config = {k[len(prefix)+len('config.'):]: state_dict[k] for k in config_keys}
        self.config = state_dict.pop(prefix + 'config') # NOTE: If not popped, will raise unexpected keys error

def get_jacobian(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """ Computes the Jacobian of y wrt x assuming minibatch-mode.

    Args:
        y (torch.Tensor): [..., yyy] with a total of D_y elements in yyy
        x (torch.Tensor): [..., xxx] with a total of D_x elements in xxx

    Returns:
        torch.Tensor: The minibatch Jacobian matrix of shape (..., D_y, D_x)
    """
    assert y.shape[:-1] == x.shape[:-1]
    y = y.view([*y.shape[:-1], -1])

    # Compute Jacobian row by row.
    jac = []
    for j in range(y.shape[-1]):
        dy_j_dx = autograd.grad(
            y[..., j],
            x,
            torch.ones_like(y[..., j], device=y.get_device()),
            retain_graph=True,
            create_graph=True,
        )[0].view([*x.shape[:-1], -1])
        jac.append(torch.unsqueeze(dy_j_dx, -2))
    jac = torch.cat(jac, -2)
    return jac

def zero_weights_init(m):
    from nr3d_lib.models.layers import BatchDenseLayer, DenseLayer
    with torch.no_grad():
        if isinstance(m, (nn.Linear, DenseLayer, BatchDenseLayer)):
            m.weight.zero_()
            if hasattr(m, 'bias') and m.bias is not None:
                m.bias.zero_()

@torch.no_grad()
def calc_grad_norm(norm_type=2.0, debug_gradient_explodes=-1, **named_models):
    if debug_gradient_explodes > 0:
        debug_grad_norms = {}

    gradient_norms = {'total': 0.0}
    for name, model in named_models.items():
        gradient_norms[name] = 0.0
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                if torch.isnan(p.grad).any():
                    log.error(f"NAN! {n}")
                if torch.isinf(p.grad).any():
                    log.error(f"INF! {n}")
                param_norm = p.grad.data.norm(norm_type)
                gradient_norms[name] += param_norm.item() ** norm_type
                if debug_gradient_explodes > 0:
                    debug_grad_norms.setdefault(name, {})[n] = param_norm.item()
        gradient_norms['total'] += gradient_norms[name]
    for k, v in gradient_norms.items():
        gradient_norms[k] = v ** (1.0 / norm_type)
        if (debug_gradient_explodes > 0) and (gradient_norms[k] > debug_gradient_explodes):
            msg = "\n".join([f"{n}: {norm}" for n,norm in debug_grad_norms.items()])
            raise RuntimeError("Gradient explodes:\n" + msg)
    return gradient_norms

@torch.no_grad()
def count_trainable_parameters(model):
    with torch.no_grad():
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        return sum([math.prod(p.size()) for p in model_parameters])

def get_optimizer(param_groups: List[dict], **kwargs) -> optim.Optimizer:
    if isinstance(param_groups, dict):
        param_groups = [param_groups]
    optimizer_type: str = kwargs.pop('type', 'adam').lower()
    if optimizer_type == 'adam':
        optimizer = optim.Adam(param_groups, **kwargs)
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(param_groups, **kwargs)
    elif optimizer_type == 'rmsprop':
        optimizer = optim.RMSprop(param_groups, **kwargs)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(param_groups, **kwargs)
    elif optimizer_type == 'sparse_adam':
        optimizer = optim.SparseAdam(param_groups, **kwargs)
    elif optimizer_type == 'adagrad':
        optimizer = optim.Adagrad(param_groups, **kwargs)
    elif optimizer_type == 'adadelta':
        optimizer = optim.Adadelta(param_groups, **kwargs)
    elif optimizer_type == 'adamax':
        optimizer = optim.Adamax(param_groups, **kwargs)
    else:
        raise RuntimeError(f"Invalid optimizer type={optimizer_type}")
    return optimizer

class SimpleScheduler(object):
    """
    A simple learning rate scheulder, taking current step as input, 
    and directly computes the closed form learning rate to update optimzer param groups.
    """
    def __init__(self, optimizer: optim.Optimizer, lr_fns: Union[Callable, List[Callable]]):
        self.optimizer = optimizer
        if not isinstance(lr_fns, list):
            lr_fns = [lr_fns for _ in range(len(self.optimizer.param_groups))]
        assert len(lr_fns) == len(self.optimizer.param_groups)
        self.lr_fns = lr_fns
    
    def step(self, it: int):
        for group, lr_fn in zip(self.optimizer.param_groups, self.lr_fns):
            group['lr'] = lr_fn(it)

def get_lr_func_multi_step(
    lr_init: float, 
    total_steps: int, 
    *, 
    milestones: List[int], 
    gamma=0.1, 
):
    """
    Mimics the behavior of torch.optim.lr_scheduler.MultiStepLR
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> scheduler = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 80
        >>> # lr = 0.0005   if epoch >= 80
    """
    
    assert 0 < gamma < 1
    def lr_fn(step):
        lr_factor = gamma ** bisect_right(milestones, step)
        return lr_init * lr_factor
    return lr_fn

def get_lr_func_warmup_cosine(
    lr_init: float, 
    total_steps: int, 
    *, 
    warmup_steps: int = 0, 
    decay_start: int=0, 
    decay_interval: int=1, 
    decay_target_factor: float=None, min_factor: float=0.1 # alias for `decay_target_factor`
    ):
    """_summary_
    - First, linear warmup for `warmup_steps` steps (only if `warmup_steps>0`)
    - Then, keeps the intial lr and wait for `decay_start` steps;
    - Then, apply exponential decay every `decay_interval` steps.

    Args:
        total_steps (int): _description_
        warmup_steps (int, optional): _description_. Defaults to 0.
        decay_start (int, optional): _description_. Defaults to 0.
        decay_interval (int, optional): _description_. Defaults to 1.
        decay_target_factor (float, optional): _description_. Defaults to None.
        min_factor (float, optional): _description_. Defaults to 0.1#aliasfor`decay_target_factor`.

    Returns:
        _type_: _description_
    """
    
    # Compatibility issue
    if min_factor is not None:
        decay_target_factor = min_factor
    
    assert 0 < decay_target_factor < 1
    total_stages = (total_steps - decay_start - warmup_steps) // decay_interval
    
    def lr_fn(step):
        """
        Modified from https://github.com/Totoro97/NeuS/blob/main/exp_runner.py
        """
        if (warmup_steps > 0) and (step < warmup_steps):
            lr_factor = max(step / warmup_steps, 1e-3)
        elif step < (warmup_steps + decay_start):
            lr_factor = 1.0
        else:
            cur_stage = (step - decay_start - warmup_steps) // decay_interval
            t = np.clip(cur_stage / total_stages, 0, 1)
            # NOTE: cosine lerp between 1.0 and `decay_target_factor`
            lr_factor = (np.cos(np.pi * t) + 1.0) * 0.5 * (1-decay_target_factor) + decay_target_factor
        return lr_init * lr_factor
    return lr_fn

def get_lr_func_exponential(
    lr_init: float, 
    total_steps: int, 
    *, 
    warmup_steps: int=0, 
    decay_start: int=0, 
    decay_interval: int=1, 
    decay_base: float=None, 
    decay_target_factor: float=None, min_factor: float=None, # alias for `decay_target_factor`
    ):
    """ Exponential decaying scheulder, given the final target lr (`decay_target_factor`) \
        or the exponential base `decay_base`. 
    - First, linear warmup for `warmup_steps` steps (only if `warmup_steps>0`)
    - Then, keeps the intial lr and wait for `decay_start` steps;
    - Then, apply exponential decay every `decay_interval` steps.

    Args:
        total_steps (int): _description_
        warmup_steps (int, optional): _description_. Defaults to 0.
        decay_target_factor (float, optinal), The target decaying 
        min_factor (float, optional): _description_. Defaults to 0.1.

    Returns:
        _type_: _description_
    """
    
    # Compatibility issue
    if min_factor is not None:
        decay_target_factor = min_factor
    
    assert bool(decay_base is not None) != bool(decay_target_factor is not None), \
        "Please give ONLY one of `decay_base` and `decay_target_factor`"
    if decay_base is not None:
        decay_target_factor = decay_base ** total_stages
    assert 0 < decay_target_factor < 1
    total_stages = (total_steps - decay_start - warmup_steps) // decay_interval
    
    def lr_fn(step):
        if (warmup_steps > 0) and (step < warmup_steps):
            lr_factor = max(step / warmup_steps, 1e-3)
        elif step < (warmup_steps + decay_start):
            lr_factor = 1.0
        else:
            cur_stage = (step - decay_start - warmup_steps) // decay_interval
            t = np.clip(cur_stage / total_stages, 0, 1)
            # NOTE: log lerp between 1.0 and `decay_target_factor`
            lr_factor = np.exp(t * np.log(decay_target_factor))
        return lr_init * lr_factor
    return lr_fn

def get_lr_func_exponential_plenoxels(
    lr_init: float, 
    total_steps: int, 
    *, 
    max_steps=None, 
    decay_target_factor: float = None, min_factor: float = None, 
    lr_final: float = None, 
    decay_interval: int = 1, 
    delay_steps=0, 
    delay_mult=1.0, 
    ):
    """
    Modified from Plenoxels
    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    """
    
    # Compatibility issue / aliases
    if max_steps is None:
        max_steps = total_steps # 1000000
    if min_factor is not None:
        decay_target_factor = min_factor
    
    assert bool(decay_target_factor is not None) != bool(lr_final is not None), \
        "Please give ONLY one of `lr_final` and `decay_target_factor`"
    if decay_target_factor is not None:
        lr_final = lr_init * decay_target_factor
    
    total_stages = total_steps // decay_interval
    def lr_fn(step):
        if step < 0:
            # Disable this parameter
            return 0.0
        if delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = delay_mult + (1 - delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        cur_stage = step // decay_interval
        t = np.clip(cur_stage / total_stages, 0, 1)
        # NOTE: log lerp between `lr_init` and `lr_final`
        lr = delay_rate * np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return lr
    return lr_fn

def get_scheduler(optimizer: optim.Optimizer, **kwargs):
    """
    Assume `optimizer` has just been intialized 
    i.e. `lr` in each optimizer.param_groups should represents the initial learning rate.
    """
    kwargs = deepcopy(kwargs)
    scheduler_type = kwargs.pop('type')
    if 'num_iters' in kwargs: # Compatibility issue
        kwargs.setdefault('total_steps', kwargs.pop('num_iters'))
    
    lr_fns = []
    for group in optimizer.param_groups:
        group.setdefault('lr_init', group['lr'])
        lr_init = group['lr_init']
        if scheduler_type == 'multi_step' or scheduler_type == 'multistep':
            lr_fn = get_lr_func_multi_step(lr_init=lr_init, **kwargs)
        elif scheduler_type == 'warmup_cosine' or scheduler_type == 'warmupcosine':
            lr_fn = get_lr_func_warmup_cosine(lr_init=lr_init, **kwargs)
        elif scheduler_type == 'exponential' or scheduler_type == 'exponential_step':
            lr_fn = get_lr_func_exponential(lr_init=lr_init, **kwargs)
        elif scheduler_type == 'exponential_plenoxels':
            lr_fn = get_lr_func_exponential_plenoxels(lr_init=lr_init, **kwargs)
        else:
            raise NotImplementedError
        lr_fns.append(lr_fn)
    
    scheduler = SimpleScheduler(optimizer, lr_fns)
    return scheduler

def batchify_query(
    query_fn: Callable, *args: Iterable[torch.Tensor], dim=0, chunk=4096, 
    show_progress=False, **kwargs: Dict[str, torch.Tensor]):
    """ Query with batchified *args, and automatically concat output tensors in tuples or dicts
        NOTE: 
        - If `query_fn` outputs tensors, each tensor will be concatenated;
        - If `query_fn` outputs tuples of tensors, each tensor will be concatenated, and the tuple structure will be untouched
        - If `query_fn` outputs (nested) dict of tensors, each tensor will be concatenated, and the (nested) dict structure will be untouched
        - If any concatenation failed, the un-concatenated list will be the output

    Args:
        query_fn (Callable): Query function
        dim (int, optional): Which dim in tensors in *args will be batchified. Defaults to 0.
        chunk (int, optional): Chunk-size when batchify. Defaults to 4096.
        show_progress (bool, optional): Whether show progress bar. Defaults to False.

    Returns:
        Any: The automatically concatenated outputs of `query_fn`
    """
    if len(args) != 0:
        _v = args[0]
    else:
        *_, _v = next(nested_dict_items(kwargs))
    N = _v.shape[dim]
    
    ret = []
    for i in tqdm(range(0, N, chunk), disable=not show_progress):
        index = (slice(None),) * dim + (slice(i, i+chunk),)
        args_i = [arg[index] for arg in args]
        vals_i = [v[index] for v in nested_dict_values(kwargs)] 
        kwargs_i = nested_dict(zip(nested_dict_keys(kwargs), vals_i))
        ret_i = query_fn(*args_i, **kwargs_i)
        if not isinstance(ret_i, tuple):
            ret_i = [ret_i]
        ret.append(ret_i)
    
    collate_ret = []
    for idx, entry in enumerate(zip(*ret)):
        if isinstance(entry[0], dict):
            entry_keys = list(nested_dict_keys(entry[0]))
            entry_dict = nested_dict(zip(entry_keys,  [[] for _ in entry_keys ]))
            for entry_item in entry:
                for ks in entry_keys:
                    _d = entry_dict
                    _sel_d = entry_item
                    for _k in ks[:-1]:
                        _d = _d[_k]
                        _sel_d = _sel_d[_k]
                    _d[ks[-1]].append(_sel_d[ks[-1]])
            # entry_vals = [torch.cat(v, dim=dim) for v in nested_dict_values(entry_dict)]
            entry_vals = []
            for v in nested_dict_values(entry_dict):
                try:
                    if isinstance(v[0], torch.Tensor):
                        v = torch.cat(v, dim=dim)
                    elif isinstance(v[0], np.ndarray):
                        v = np.concatenate(v, axis=dim)
                except:
                    pass
                entry_vals.append(v)
            entry_dict = nested_dict(zip(entry_keys, entry_vals))
            collate_ret.append(entry_dict)
        else:
            collate_ret.append(torch.cat(entry, dim=dim))
    if idx==0:
        return collate_ret[0]
    else:
        return tuple(collate_ret)

def batchify_query_ray_pts(
    query_fn, *args: Iterable[torch.Tensor], chunk=4096, dim=0, 
    show_progress=False, **kwargs: Dict[str, torch.Tensor]):
    if len(args) != 0:
        _v = args[0]
    else:
        *_, _v = next(nested_dict_items(kwargs))
    N_rays = _v.shape[dim]
    N_pts = _v.shape[dim+1]
    N = N_rays * N_pts
    
    # [(batch_dims), N_rays, N_pts, ...] -> [(batch_dims), N_rays*N_pts, ...]
    args = [arg.flatten(dim, dim+1) for arg in args]
    vals = [val.flatten(dim, dim+1) for val in nested_dict_values(kwargs)]
    kwargs = nested_dict(zip(nested_dict_keys(kwargs), vals))
    
    ret = []
    for i in tqdm(range(0, N, chunk), disable=not show_progress):
        index = (slice(None),) * dim + (slice(i, i+chunk),)
        args_i = [arg[index] for arg in args]
        vals_i = [v[index] for v in nested_dict_values(kwargs)] 
        kwargs_i = nested_dict(zip(nested_dict_keys(kwargs), vals_i))
        ret_i = query_fn(*args_i, **kwargs_i)
        if not isinstance(ret_i, tuple):
            ret_i = [ret_i]
        ret.append(ret_i)
    
    collate_ret = []
    for idx, entry in enumerate(zip(*ret)):
        # [(batch_dims), N_rays*N_pts, ...] -> [(batch_dims), N_rays, N_pts, ...]
        if isinstance(entry[0], dict):
            entry_keys = list(nested_dict_keys(entry[0]))
            entry_dict = nested_dict(zip(entry_keys,  [[] for _ in entry_keys ]))
            for entry_item in entry:
                for ks in entry_keys:
                    _d = entry_dict
                    _sel_d = entry_item
                    for _k in ks[:-1]:
                        _d = _d[_k]
                        _sel_d = _sel_d[_k]
                    _d[ks[-1]].append(_sel_d[ks[-1]])
            # entry_vals = [torch.cat(v, dim=dim) for v in nested_dict_values(entry_dict)]
            entry_vals = []
            for v in nested_dict_values(entry_dict):
                try:
                    if isinstance(v[0], torch.Tensor):
                        # v = torch.cat(v, dim=dim).unflatten(dim, [N_rays, N_pts])
                        v = torch.cat(v, dim=dim)
                        v = v.reshape([*v.shape[:dim], N_rays, N_pts, *v.shape[dim+1:]])
                    elif isinstance(v[0], np.ndarray):
                        v = np.concatenate(v, axis=dim)
                        v = v.reshape([*v.shape[:dim], N_rays, N_pts, *v.shape[dim+1:]])
                except:
                    pass
                entry_vals.append(v)
            entry_dict = nested_dict(zip(entry_keys, entry_vals))
            collate_ret.append(entry_dict)
        else:
            v = torch.cat(entry, dim=dim)
            v = v.reshape([*v.shape[:dim], N_rays, N_pts, *v.shape[dim+1:]])
            collate_ret.append(v)
    if idx==0:
        return collate_ret[0]
    else:
        return tuple(collate_ret)

def clip_norm_(
    grad: torch.Tensor, max_norm: float, norm_type: float = 2.0,
    error_if_nonfinite: bool = False) -> torch.Tensor:
    """
    Modified from torch/nn/utils/clip_grad.py :: clip_grad_norm_
    """
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    
    if norm_type == np.inf:
        total_norm = grad.data.abs().max()
    else:
        total_norm = grad.data.norm(norm_type)
    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f'The total norm of order {norm_type} for gradients from '
            '`parameters` is non-finite, so it cannot be clipped. To disable '
            'this error and scale the gradients by the non-finite norm anyway, '
            'set `error_if_nonfinite=False`')
    clip_coef = max_norm / (total_norm + 1e-6)
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    grad.mul_(clip_coef_clamped)
    return grad

if __name__ == "__main__":
    def test_schedulers():
        import matplotlib.pyplot as plt
        
        total_steps = 1000
        lr_funcs = {}
        lr_funcs['cosine, warmup=200'] = get_lr_func_warmup_cosine(
            lr_init=0.01, total_steps=total_steps, decay_target_factor=0.06, warmup_steps=200
        )

        lr_funcs['multi_step'] = get_lr_func_multi_step(
            lr_init=0.01, total_steps=total_steps, gamma=0.1, milestones=[300,600]
        )

        lr_funcs['exponential_plenoxels, delay_mult=1'] = get_lr_func_exponential_plenoxels(
            lr_init=0.01, total_steps=total_steps, lr_final=0.001, delay_steps=200, delay_mult=1,
        )
        
        lr_funcs['exponential_plenoxels, delay_mult=0.5'] = get_lr_func_exponential_plenoxels(
            lr_init=0.01, total_steps=total_steps, lr_final=0.001, delay_steps=200, delay_mult=0.5,
        )

        lr_funcs['exponential_plenoxels, delay_mult=0'] = get_lr_func_exponential_plenoxels(
            lr_init=0.01, total_steps=total_steps, lr_final=0.001, delay_steps=200, delay_mult=0,
        )

        num = len(lr_funcs)
        nrow = max(1, int(math.sqrt(num/1.3)+0.5))
        ncol = int((num + nrow - 1) / nrow)
        
        for i, (name, fn) in enumerate(lr_funcs.items()):
            its = np.arange(total_steps)
            lrs = np.array([fn(it) for it in range(total_steps)])
            plt.subplot(nrow, ncol, i+1)
            plt.plot(its, lrs)
            plt.title(name)
        
        plt.show()

    test_schedulers()
    import torch.nn as nn
    import torch.nn.functional as F
    nn.Softplus
    torch._C._nn.softplus
    torch._C._nn.softplus_backward
    torch.autograd.softplus()
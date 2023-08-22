"""
@file   utils.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Neural-network-level utililities.
"""

import re
import math
import numbers
import numpy as np
from tqdm import tqdm
from typing import Callable, Dict, Iterable, List, Union

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
            if isinstance(v, (numbers.Number, str)):
                pass
            elif isinstance(v, (list, tuple)) and all([isinstance(i, (numbers.Number, str)) for i in v]):
                pass
            else:
                raise RuntimeError(msg + f", but got {'.'.join(k)}: {type(v)}")
        
        self._register_state_dict_hook(self.state_dict_hook)
        self._register_load_state_dict_pre_hook(self.load_state_dict_hook)

    @staticmethod # NOTE: "self" is also passed when pytorch call hooks
    def state_dict_hook(self, state_dict: dict, prefix, local_metadata):
        # config_state_dict = {(prefix + '.'.join('config', *k)): v for *k, v in nested_dict_items(self.config)}
        # state_dict.update(config_state_dict)
        state_dict[prefix + 'config'] = self.config
    
    def load_state_dict_hook(self, state_dict: dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
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

def count_trainable_parameters(model):
    with torch.no_grad():
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        return sum([math.prod(p.size()) for p in model_parameters])

def get_optimizer(model: nn.Module, **optim_cfg):
    param_groups = get_param_group(model, optim_cfg['lr'])
    if 'optimizer' not in optim_cfg:
        optimizer = optim.Adam(param_groups)
    else:
        cfg = optim_cfg['optimizer']
        if 'target' in cfg:
            optimizer = import_str(cfg['target'])(param_groups, **cfg['param'])
        else:
            optimizer = optim.Adam(param_groups, **cfg)
    return optimizer

def CosineAnnealWarmUpSchedulerLambda(total_steps: int, warmup_steps: int = 0, min_factor: float=0.1):
    assert 0 <= min_factor < 1
    def lambda_fn(epoch):
        """
        Modified from https://github.com/Totoro97/NeuS/blob/main/exp_runner.py
        """
        if epoch < warmup_steps:
            learning_factor = epoch / warmup_steps
        else:
            learning_factor = (np.cos(np.pi * ((epoch - warmup_steps) / (total_steps - warmup_steps))) + 1.0) * 0.5 * (1-min_factor) + min_factor
        return learning_factor
    return lambda_fn

def ExponentialSchedulerLambda(total_steps: int, warmup_steps: int = 0, min_factor: float=0.1):
    assert 0 <= min_factor < 1
    def lambda_fn(epoch):
        if epoch < warmup_steps:
            learning_factor = epoch / warmup_steps
        else:
            t = np.clip((epoch-warmup_steps) / (total_steps-warmup_steps), 0, 1)
            learning_factor = np.exp(t * np.log(min_factor))
        return learning_factor
    return lambda_fn

def ExponentialDecaySchedulerLambda(total_steps, decay_base: float = 0.9999, decay_interval: int = 1, decay_start: int = 0):
    assert decay_interval >= 1
    num_stages = (total_steps - decay_start) /decay_interval
    min_factor = decay_base ** num_stages
    def lambda_fn(epoch):
        cur_stage = (epoch - decay_start) / decay_interval
        t = np.clip(cur_stage / num_stages, 0, 1)
        learning_factor = np.exp(t * np.log(min_factor))
        return learning_factor
    return lambda_fn

def get_scheduler(config: ConfigDict, optimizer, last_epoch: int=-1):
    stype = config['type']
    if stype == 'multistep':
        scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer, 
                config['milestones'], 
                gamma=config['gamma'], 
                last_epoch=last_epoch)
    elif stype == 'warmupcosine':
        # NOTE: This do not support per-parameter lr
        # from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
        # scheduler = CosineAnnealingWarmupRestarts(
        #     optimizer, 
        #     config['num_iters'], 
        #     max_lr=config['lr'], 
        #     min_lr=config.setdefault('min_lr', 0.1*config.lr), 
        #     warmup_steps=config['warmup_steps'], 
        #     last_epoch=last_epoch)
        # NOTE: Support per-parameter lr
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, 
            CosineAnnealWarmUpSchedulerLambda(
                total_steps=config['num_iters'], 
                warmup_steps=config.setdefault('warmup_steps', 0), 
                min_factor=config.setdefault('min_factor', 0.1)
            ),
            last_epoch=last_epoch)
    elif stype == 'exponential_step':
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            ExponentialSchedulerLambda(
                total_steps=config['num_iters'],
                warmup_steps=config.setdefault('warmup_steps', 0), 
                min_factor=config.setdefault('min_factor', 0.1)
            )
        )
    elif stype == 'exponential_decay':
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            ExponentialDecaySchedulerLambda(
                total_steps=config['num_iters'], 
                decay_base=config['decay_base'], 
                decay_interval=config['decay_interval'], 
                decay_start=config.setdefault('decay_start', 0)
            )
        )
    else:
        raise NotImplementedError
    return scheduler

def get_param_group(
    model: nn.Module, optim_cfg: Union[numbers.Number, dict], prefix: str=''
    ) -> List[dict]:
    """ Pack model parameters into parameter group(s) with given `optim_cfg`
    
    Supported types input `optim_cfg`:
    
    1. A number, indicating shared-by-all learning rate.

    2. A dict with multiple keys including 'lr', indicating optimizer detailed configs of the model;
        e.g. {
            'lr': $item
        },
        
        or {
            'lr': 1.0,
            'betas': 0.9, 0.99,
            'eps': 1.0e-6
        }

    3. A dict with multiple keys including `default`, 
        indicating named-param-specific optimizer detailed configs (NOTE: regex supported.)
        e.g. {
            'default': $item1,
            'model\.backbone\.conv1': $item2,
            'model\.backbone\.conv2': $item3,
            'model\.head\.linear': $item4
        },
        
        in which $item follows the former 1.&2. definition ,
            that could be either learning rate number or a detailed config dict.
        
        e.g. {
                'default': 1.0e-1,
                'model\.backbone\.conv1': {
                    'lr': 1.0e-2,
                    'beta': 0.99
                }
                'model\.backbone\.conv2': 1.0e-3,
                'model\.head\..*linear': {
                    # this is a regex that includes every param matching "model.head.*.linear"
                    'lr': 1.0e-4,
                    'beta': 0.9
                }
            }
    """
    if isinstance(optim_cfg, numbers.Number):
        pg_all = [{
            'name': prefix,
            'params': model.parameters(),
            'capturable': True,
            'lr': optim_cfg
        }]
    elif isinstance(optim_cfg, (dict, ConfigDict)):
        optim_cfg = optim_cfg.copy()
        if 'lr' in optim_cfg.keys():
            lr = optim_cfg.pop('lr')
            pg_all = [{
                'name': prefix,
                'params': model.parameters(),
                'capturable': True,
                'lr': lr, 
                **optim_cfg
            }]
        elif 'default' in optim_cfg.keys():
            def parse_item_cfg(item):
                if isinstance(item, numbers.Number):
                    return {'lr': item}
                elif isinstance(item, dict):
                    return item
                else:
                    raise ValueError(f"Invalid type of input item={type(item)}; the value is {item}")
            
            # `default` parameter group
            pg_default = {
                    'name': '.'.join([prefix, 'default']) if prefix else 'default',
                    'params': [],
                    'capturable': True,
                    **parse_item_cfg(optim_cfg.pop('default'))
                }
            
            # Settings' parameter groups
            pg_cfg = {}
            for pname, pcfg in optim_cfg.items():
                pg_cfg[pname] = {
                    'cnt': 0,
                    'pg': {
                        'name': '.'.join([prefix, pname]) if prefix else pname,
                        'params': [],
                        'capturable': True,
                        **parse_item_cfg(pcfg)
                    }
                }
            
            # Allocate model's parameters
            for pn, p in model.named_parameters():
                matched = False
                for pnp in pg_cfg.keys():
                    # Search for pnp in pn (force matching from the start to prevent unexpected matches)
                    # if re.search('^'+pnp.replace('.', '\.'), pn):
                    if re.search('^'+pnp, pn):
                        # if match any
                        log.debug(pn, '  --->>>  ', pnp)
                        matched = True
                        pg_cfg[pnp]['pg']['params'].append(p)
                        pg_cfg[pnp]['cnt'] += 1
                        break
                if not matched:
                    log.debug(pn, '  --->>>  ' , 'default') # Fallback to `default` if no matches
                    pg_default['params'].append(p)
            
            # check not used settings
            for k, v in pg_cfg.items():
                if v['cnt'] == 0:
                    log.warn(f"key ('{k}') is not used when setting parameter groups.")
            pg_all = [pg_default] + [pgv['pg'] for pgv in pg_cfg.values()]
        else:
            raise RuntimeError(f"Invalid optim_cfg, unable to parse. Should at least contain 'lr' or 'default' in optim_cfg, while current is \n{optim_cfg}.")
    return pg_all

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

def logistic_density(x: torch.Tensor, inv_s: Union[torch.Tensor, numbers.Number]) -> torch.Tensor:
    """ Logistic density function
    Source: https://en.wikipedia.org/wiki/Logistic_distribution

    Args:
        x (torch.Tensor): Input
        inv_s (Union[torch.Tensor, numbers.Number]): The reciprocal of the distribution scaling factor.

    Returns:
        torch.Tensor: Output
    """
    return 0.25*inv_s / (torch.cosh(inv_s*x/2.).clamp_(-20, 20)**2) 

def normalized_logistic_density(x: torch.Tensor, inv_s: Union[torch.Tensor, numbers.Number]) -> torch.Tensor:
    """ Normalized logistic density function (with peak value = 1.0)
    Source: https://en.wikipedia.org/wiki/Logistic_distribution

    Args:
        x (torch.Tensor): Input
        inv_s (Union[torch.Tensor, numbers.Number]): The reciprocal of the distribution scaling factor.

    Returns:
        torch.Tensor: Output
    """
    return (1./torch.cosh((inv_s*x/2.).clamp_(-20, 20)))**2
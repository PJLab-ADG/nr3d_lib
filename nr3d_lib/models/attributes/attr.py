"""
@file   attr.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Basic attr definitions.
"""

_vector_length_list = list(range(2, 15, 1)) # 2-14

__all__ = [
    "Attr", 
    "AttrNested", 
    "AttrBase", 
    "has_common_base", 
    
    "Scalar", 
    # "MatBase", 
    "make_mat", 
    "make_vector", 
    *[f"Vector_{i}" for i in _vector_length_list],
    
    "make_refined_add_cls", 
    
    "ScalarRefinedAdd", 
    *[f"Vector{i}RefinedAdd" for i in _vector_length_list],
    
    "ObjectWithAttr", 
]

import functools
import numpy as np
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from nr3d_lib.utils import check_to_torch, torch_consecutive_interp1d

class Attr(nn.Module):
    def __init__(self, data: Union[np.ndarray, torch.Tensor]=None, *, device=None, dtype=None, learnable=False, persistent=False):
        nn.Module.__init__(self)
        if data is None:
            # data = deepcopy(self.default)
            data = self.default.clone() if self.default is not None else None
        data = check_to_torch(data, device=device, dtype=dtype)
        if learnable:
            assert not data.requires_grad, "input data already carries gradients, no need to set learnable=True"
            self.register_parameter('tensor', nn.Parameter(data, requires_grad=True))
        else:
            self.register_buffer('tensor', data, persistent=persistent)
        assert list(self.datashape) == list(self.default.shape), \
            f"Expect data to have shape suffix = {self.default.shape}, but current shape = {self.datashape}"
        # self.learnable = learnable
        # self.persistent = persistent
    def __getitem__(self, index):
        if len(self.prefix) == 0:
            # if isinstance(index, torch.Tensor):
            #     return self.tile(tuple(index.shape))
            # else:
            #     raise RuntimeError(f"Single item Attr({type(self)}) does not support indexing with type={type(index)}")
            raise RuntimeError(f"Single item Attr({type(self)}) does not support indexing")
        else:
            return type(self)(self.tensor[index])
    def __setitem__(self, index, val):
        if isinstance(val, type(self)):
            self.tensor[index] = val.tensor
        else:
            self.tensor[index] = val
    def reset(self):
        # self.tensor = deepcopy(self.default).to(dtype=self.dtype, device=self.device)
        self.tensor = self.default.clone().to(dtype=self.dtype, device=self.device)
    def new(self, size: Tuple=()):
        """
        Tile and expand the default data to the given prefix size
        """
        return type(self)(self.default.clone().tile([*size, *[1]*len(self.datashape)]), device=self.device, dtype=self.dtype)
    def tile(self, prefix: Tuple):
        """
        Tile and expand current data to the given prefix size
        """
        return type(self)(self.tensor.tile(*prefix, *[1]*len(self.datashape)))
    def clone(self):
        return type(self)(self.tensor.clone())
    def detach(self):
        return type(self)(self.tensor.detach())
    def interp1d(self, ts_keyframes: torch.Tensor, ts: torch.Tensor):
        """ 
        Default behavior: linear interpolation of `self.tensor`. 
        Override this method for custom behaviors in your custom Attr.

        Args:
            ts_keyframes (torch.Tensor): 1D tensor of unique, consecutive keyframe timestamps.
            ts (torch.Tensor): Timestamps at which to perform interpolation.

        Returns:
            type(self): The interpolated Attr.
        """
        assert ts_keyframes.dim() == 1, "`ts_keyframes` should be 1D keyframe timestamps tensor"
        assert [*self.prefix] == [ts_keyframes.size(0)], \
            f"To interpolate Attr of {self.__class__.__name__}, its prefix should be the same with `ts_keyframes`={ts_keyframes.size(0)}"
        ts = check_to_torch(ts, device=self.device)
        v = torch_consecutive_interp1d(ts_keyframes, self.tensor, ts)
        return type(self)(v)
    @property
    def prefix(self):
        dims = self.default.dim()
        return tuple(self.tensor.shape[:-dims]) if dims !=0 else tuple(self.tensor.shape)
    @property
    def datashape(self):
        dims = self.default.dim()
        return tuple(self.tensor.shape[-dims:]) if dims !=0 else ()
    def gather(self, dim: int, index: torch.LongTensor):
        # NOTE: Only the `prefix` dimensions are gathered; `datashape` dimensions should be repeated.
        prefix = [*self.prefix]
        prefix[dim] = 1
        assert [*index.shape] == prefix, f"`index` should be of shape={prefix}, but got {[*index.shape]}"
        index = index.view(*prefix, *[1]*len(self.datashape)).tile(self.datashape)
        return type(self)(self.tensor.gather(dim, index))
    def take_along_dim(self, index: torch.Tensor, dim: int):
        # NOTE: Only the `prefix` dimensions are gathered; `datashape` dimensions should be repeated.
        prefix = [*self.prefix]
        prefix[dim] = 1
        assert [*index.shape] == prefix, f"`index` should be of shape={prefix}, but got {[*index.shape]}"
        index = index.view(*prefix, *[1]*len(self.datashape))
        return type(self)(self.tensor.take_along_dim(index, dim))
    @classmethod
    def stack(cls, list_or_tuple: Union[List, Tuple]):
        assert isinstance(list_or_tuple, (list, tuple))
        assert type(next(iter(list_or_tuple))) == cls
        tensors = torch.stack([it.tensor for it in list_or_tuple], dim=0)
        return cls(tensors)
    @classmethod
    def concat(cls, list_or_tuple: Union[List, Tuple]):
        assert isinstance(list_or_tuple, (list, tuple))
        assert type(next(iter(list_or_tuple))) == cls
        tensors = torch.cat([it.tensor for it in list_or_tuple], dim=0)
        return cls(tensors)
    @property
    def dtype(self):
        return self.tensor.dtype
    @property
    def device(self) -> torch.device:
        return self.tensor.device
    def extra_repr(self) -> str:
        return f"tensor.shape={[*self.prefix]} x {[*self.datashape]}, device={self.device}, dtype={self.dtype}, learnable={self.tensor.requires_grad}"

class AttrNested(nn.Module):
    default: Dict[str,Attr] = {}
    def __init__(
        # Self config
        self, *, allow_new_attr=False, 
        # Bypass subattr configs
        device=None, dtype=None, learnable=False, persistent=False, 
        # Input subattrs
        **subattrs):
        nn.Module.__init__(self)
        
        # NOTE: !!! Important to deepcopy! 
        #           Since class attribute variables like dict/list will only be new-ed ONCE!
        # local_dict = {k:deepcopy(v) for k,v in self.default.items()}
        local_dict = {k:v.clone() for k,v in self.default.items()}
        
        unexpected_keys = []
        for k, v in subattrs.items():
            if k in local_dict:
                if isinstance(v, (Attr, AttrNested)):
                    assert has_common_base(type(subattrs[k]), type(local_dict[k]))
                    # NOTE: For now, `learnable`` and `persistent`` can not bypass already initialized Attr. 
                    #       We will keep it this way --- different learnable settings in different subattrs should be allowed.
                    if device is not None or dtype is not None:
                        v = v.to(device=device, dtype=dtype) 
                else:
                    v = type(local_dict[k])(v, device=device, dtype=dtype, learnable=learnable, persistent=persistent)
            elif not allow_new_attr:
                unexpected_keys.append(k)
                continue
            local_dict[k] = v
        if not allow_new_attr and len(unexpected_keys) > 0:
            raise ValueError(f"Self keys: {list(local_dict.keys())}.\nUnexpected keys: {unexpected_keys}")
        for k,v in local_dict.items():
            local_dict[k] = v.to(device=device, dtype=dtype)
        self.subattr: Dict[str, Union[Attr, AttrNested]] = nn.ModuleDict(local_dict)
    def __getitem__(self, index):
        # return type(self)(allow_new_attr=True, **{ k: v[index] for k,v in self.subattr.items() })
        subattrs = { k: v[index] for k,v in self.subattr.items() }
        # return type(self)(**subattrs, allow_new_attr=(type(self) == AttrNested))
        return type(self)(**subattrs)
    def __setitem__(self, index, val):
        for k,v in self.subattr.items():
            self.subattr[k][index] = val.subattr[k]
    def reset(self):
        for k in self.subattr.keys():
            self.subattr[k].reset()
    def new(self, size: Tuple=()):
        """
        Tile and expand the default data to the given prefix size
        """
        subattrs = {k:v.new(size) for k,v in self.subattr.items()}
        return type(self)(**subattrs)
    def tile(self, prefix: Tuple):
        """
        Tile and expand current data to the given prefix size
        """
        return type(self)(**{k: v.tile(prefix) for k, v in self.subattr.items()})
    def clone(self):
        return type(self)(**{k:v.clone() for k,v in self.subattr.items()})
    def detach(self):
        return type(self)(**{k:v.detach() for k,v in self.subattr.items()})
    def interp1d(self, ts_keyframes: torch.Tensor, ts: torch.Tensor):
        return type(self)(**{k:v.interp1d(ts_keyframes, ts) for k,v in self.subattr.items()})
    @property
    def prefix(self):
        raise NotImplementedError
    def gather(self, dim: int, index: torch.LongTensor):
        local_dict = {}
        for k, v in self.subattr.items():
            local_dict[k] = v.gather(dim, index)
        return type(self)(**local_dict)
    def take_along_dim(self, index: torch.Tensor, dim: int):
        local_dict = {}
        for k, v in self.subattr.items():
            local_dict[k] = v.take_along_dim(index, dim)
        return type(self)(**local_dict)
    @classmethod
    def stack(cls, list_or_tuple: Union[List, Tuple]):
        assert isinstance(list_or_tuple, (list, tuple))
        first = next(iter(list_or_tuple))
        assert type(first) == cls
        local_dict = {}
        for k, v in first.subattr.items():
            local_dict[k] = type(v).stack([it.subattr[k] for it in list_or_tuple])
        return cls(**local_dict)
    @classmethod
    def concat(cls, list_or_tuple: Union[List, Tuple]):
        assert isinstance(list_or_tuple, (list, tuple))
        first = next(iter(list_or_tuple))
        assert type(first) == cls
        local_dict = {}
        for k, v in first.subattr.items():
            local_dict[k] = type(v).concat([it.subattr[k] for it in list_or_tuple])
        return cls(**local_dict)
    @property
    def dtype(self):
        return next(iter(self.subattr.values())).dtype
    @property
    def device(self) -> torch.device:
        return next(iter(self.subattr.values())).device
    def extra_repr(self) -> str:
        return ""

def AttrBase(cls):
    cls.base_cls = cls
    return cls

def has_common_base(cls1, cls2):
    return hasattr(cls1, 'base_cls') and hasattr(cls2, 'base_cls') and (cls1.base_cls == cls2.base_cls)

################################################
########    Scalar/Vectors/Matrices    #########
################################################
@AttrBase
class Scalar(Attr):
    default = torch.zeros([])
    def item(self):
        assert len(self.prefix) == 0, f"item() is only available for zero-len prefix, but current prefix={self.prefix}"
        return self.tensor.item()
    def value(self):
        return self.tensor[:]

# @AttrBase
# class MatBase(Attr):
#     pass

@functools.lru_cache(maxsize=None) # NOTE: lru_scahe to make sure same returns with same id
def make_mat(shape: Tuple[int]):
    assert isinstance(shape, Tuple) and isinstance(shape[0], int), \
        f"Must given tuple of int, but got {shape}"
    @AttrBase
    class Mat(Attr):
        default = torch.zeros([*shape])
        def value(self):
            return self.tensor[:]
    Mat.__name__ = Mat.__name__ + '_' + "x".join([str(i) for i in shape])
    return Mat

@functools.lru_cache(maxsize=None) # NOTE: lru_scahe to make sure same returns with same id
def make_vector(n: int):
    return make_mat((n,))

# class Vector2D(Attr):
#     default = torch.zeros([2,])

""" Define multiple classes
globals().update({f"Vector_{i}": make_vector(i) for i in _vector_length_list})
"""
Vector_2 = make_vector(2)
Vector_3 = make_vector(3)
Vector_4 = make_vector(4)
Vector_5 = make_vector(5)
Vector_6 = make_vector(6)
Vector_7 = make_vector(7)
Vector_8 = make_vector(8)
Vector_9 = make_vector(9)
Vector_10 = make_vector(10)
Vector_11 = make_vector(11)
Vector_12 = make_vector(12)
Vector_13 = make_vector(13)
Vector_14 = make_vector(14)

################################################
##############    Refinement    ################
################################################
class ScalarRefinedAdd(AttrNested, Scalar):
    default = {'attr0': Scalar(), 'delta': Scalar(torch.zeros([]))}
    @property
    def tensor(self):
        return self.subattr.attr0.tensor + self.subattr.delta.tensor
    def value(self):
        return self.tensor
    @property
    def prefix(self):
        return self.subattr.attr0.prefix

#---- Simple addition refinement
def make_refined_add_cls(cls: Attr):
    assert issubclass(cls, Attr) and not issubclass(cls, AttrNested)
    class RefinedAdd(AttrNested, cls):
        default = {'attr0': cls(), 'delta': make_mat((*cls.default.shape,))(torch.zeros([*cls.default.shape]))}
        @property
        def tensor(self):
            return self.subattr.attr0.tensor + self.subattr.delta.tensor
        def value(self):
            return self.tensor
        @property
        def prefix(self):
            return self.subattr.attr0.prefix
    RefinedAdd.__name__ = cls.__name__ + "RefinedAdd"
    return RefinedAdd

Vector2RefinedAdd = make_refined_add_cls(Vector_2)
Vector3RefinedAdd = make_refined_add_cls(Vector_3)
Vector4RefinedAdd = make_refined_add_cls(Vector_4)
Vector5RefinedAdd = make_refined_add_cls(Vector_5)
Vector6RefinedAdd = make_refined_add_cls(Vector_6)
Vector7RefinedAdd = make_refined_add_cls(Vector_7)
Vector8RefinedAdd = make_refined_add_cls(Vector_8)
Vector9RefinedAdd = make_refined_add_cls(Vector_9)
Vector10RefinedAdd = make_refined_add_cls(Vector_10)
Vector11RefinedAdd = make_refined_add_cls(Vector_11)
Vector12RefinedAdd = make_refined_add_cls(Vector_12)
Vector13RefinedAdd = make_refined_add_cls(Vector_13)
Vector14RefinedAdd = make_refined_add_cls(Vector_14)

################################################
###############    Interface    ################
################################################
class ObjectWithAttr(object):
    """
    A mixin for the need of multiple attributes and its management
    """
    _initialized = False
    def __init__(self, device=None, dtype=torch.float) -> None:
        self.device = device
        self.dtype = dtype
        self._attrs: Dict[str, Optional[Union[Attr,AttrNested]]] = dict()
        self._initialized = True
    # NOTE: Mimic pytorch's behavior on nn.Modules and nn.Parameters etc.
    def named_attrs(self, prefix: str = ''):
        for k, v in self._attrs.items():
            yield prefix+k, v
    def __setattr__(self, name: str, value: Any) -> None:
        if self._initialized and ((name in (attrs:=self.__dict__['_attrs'])) or isinstance(value, (Attr, AttrNested))):
            if name in attrs:
                assert has_common_base(type(value), type(attrs[name])), f"Can not assign a {type(value)} object to {type(attrs[name])} attr."
            # Register new attr
            attrs[name] = value
        else:
            object.__setattr__(self, name, value)
    def __getattr__(self, name: str):
        # NOTE: This will not influence attribtues that are set be regular setattr (object.__setattr__)
        if self._initialized and (name in (attrs:=self.__dict__['_attrs'])):
            return attrs[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    def __delattr__(self, name: str):
        if self._initialized and name in self._attrs:
            del self._attrs[name]
        else:
            object.__delattr__(self, name)
    
    def _reset(self):
        for k, v in self._attrs.items():
            v.reset()
    
    def _apply(self, fn):
        with torch.no_grad():
            for k in self._attrs.keys():
                self._attrs[k]._apply(fn)
            self.device = next(iter(self._attrs.values())).device
        return self
    @functools.wraps(nn.Module.to)
    def to(self, *args, **kwargs):
        with torch.no_grad():
            for k in self._attrs.keys():
                self._attrs[k].to(*args, **kwargs)
            self.device = next(iter(self._attrs.values())).device
        return self
    @functools.wraps(nn.Module.cuda)
    def cuda(self, device=None):
        return self._apply(lambda t: t.cuda(device))
    @functools.wraps(nn.Module.cpu)
    def cpu(self):
        return self._apply(lambda t: t.cpu())
    @functools.wraps(nn.Module.float)
    def float(self):
        return self._apply(lambda t: t.float() if t.is_floating_point() else t)
    @functools.wraps(nn.Module.double)
    def double(self):
        return self._apply(lambda t: t.double() if t.is_floating_point() else t)
    @functools.wraps(nn.Module.half)
    def half(self):
        return self._apply(lambda t: t.half() if t.is_floating_point() else t)
"""
@file   utils.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Python utilities.
"""

import os
import glob
import time
import errno
import shutil
import psutil
import numbers
import imageio
import functools
import importlib
import imagesize
import numpy as np
from PIL import Image
from math import prod
from collections import namedtuple
from typing import Iterator, Iterable, List, Dict, Literal, Tuple, Union, Generic, TypeVar

import skimage
from skimage.transform import resize as cpu_resize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import resize as gpu_resize

from nr3d_lib.fmt import log

T = TypeVar('T')  # Any type.
KT = TypeVar('KT')  # Key type.
VT = TypeVar('VT')  # Value type.

# ---------------------------------------------
# --------------     Common     ---------------
# ---------------------------------------------


class IDListedDict(Dict[str, T], Generic[T]):
    """
    A dict that contains a list of object with `id` property.
    - Support integer indexing: indexing the list.
    - Support key/str indexing: indexing with `id`.
    """
    def __init__(self, other: Union[List[T], Dict[str,T]] = (), /):
        if isinstance(other, list):
            dict.__init__(self, **{v.id: v for v in other})
        elif isinstance(other, dict):
            dict.__init__(self, other)
        elif other == ():
            pass
        else:
            raise RuntimeError(f"invalid input type: {type(other)}")

    # NOTE: Force only use dict's setitem API
    def __setitem__(self, k: str, v: T):
        assert isinstance(k, str)
        dict.__setitem__(self, k, v)

    def __getitem__(self, index: Union[str, int, List[str], List[int], torch.LongTensor, np.ndarray]) -> Union[T, 'IDListedDict[T]']:
        if isinstance(index, str):
            return dict.__getitem__(self, index)
        elif isinstance(index, (int, slice)):
            return list(self.values())[index]
        elif isinstance(index, (list, tuple)):  # Could be list of int or list of string
            return IDListedDict([self[i] for i in index])
        elif isinstance(index, (torch.LongTensor, np.ndarray)):
            shape = index.shape
            assert len(shape) <= 1, 'only support 1-D tensor'
            if prod(shape) == 1:
                # scalar
                index = index.item()
                return self[index]
            else:
                # vector
                indices = index.tolist()
                return self[indices]
        else:
            raise RuntimeError(f"Invalid key type: {type(index)}")

    def __delitem__(self, index) -> None:
        if isinstance(index, str):
            return dict.__delitem__(index)
        elif isinstance(index, int):
            return dict.__delitem__(self, list(self.keys())[index])
        elif isinstance(index, (list, tuple)):
            for i in index:
                self.__delitem__(i)
        elif isinstance(index, (torch.LongTensor, np.ndarray)):
            shape = index.shape
            assert len(shape) <= 1, 'only support 1-D tensor'
            if prod(shape) == 1:
                # scalar
                index = index.item()
                self.__delitem__(index)
            else:
                # vector
                indices = index.tolist()
                return self.__delitem__(indices)
        else:
            raise RuntimeError(f"Invalid key type: {type(index)}")

    # Dict-like APIs (inherited and not changed)
    # def __delitem__(self, key, dict_delitem=dict.__delitem__):
    # def clear(self):

    #---------------------------
    #---- List-like APIs
    #---------------------------
    def __iter__(self) -> Iterator[T]:
        for val in self.values():
            yield val

    def __reversed__(self) -> Iterator[T]:
        for val in reversed(self.values()):
            yield val

    def append(self, item) -> None:
        assert hasattr(item, 'id'), 'You can only put objects with ids when using direct append()'
        assert item.id not in self.keys(), f'Item: {item.id} already in dict'
        self[item.id] = item

    def index(self, k: str) -> int:
        return list(self.keys()).index(k)

    def to_list(self) -> List[T]:
        return list(self.values())

    #---------------------------
    #---- Tensor-like APIs
    #---------------------------
    @functools.wraps(torch.Tensor.to)
    def to(self, *args, **kwargs):
        # NOTE: Force inplace
        for k in self.keys():
            self[k] = self[k].to(*args, **kwargs)
        return self

    @property
    def dtype(self):
        assert len(self) > 0, 'You can not get dtype of a empty IDListedDict'
        return next(iter(self.values())).dtype

    @property
    def device(self):
        assert len(self) > 0, 'You can not get device of a empty IDListedDict'
        return next(iter(self.values())).device


def namedtupleTensors(typename: str, field_names: str):
    """
    - `namedtuple` that consists of tensors
    - Should be of the same prefix shapes
    - Sliceable / indexable

    Example:
        Given: 
        >>> x = torch.randn([100, 3, 3])
        >>> y = torch.randn([100, 7])
        >>> z = torch.randn([100, 9])
        >>> a = namedtupleTensors('t', 'x y z')(x, y, z)
        
        The results of:
        >>> print(a.x[0:7].shape)
        >>> print(a.y[0:7].shape)
        >>> print(a.z[0:7].shape)

        Will be the same with:
        >>> print(a[0:7].x.shape)
        >>> print(a[0:7].y.shape)
        >>> print(a[0:7].z.shape)

    """
    cls = namedtuple(typename, field_names)

    def __getitem__(self, index):
        item_dict = {field: getattr(self, field)[index] for field in self._fields}
        return self.__class__(**item_dict)

    @functools.wraps(torch.Tensor.to)
    def to(self, *args, **kwargs):
        # NOTE: not inplace
        item_dict = {field: getattr(self, field).to(*args, **kwargs) for field in self._fields}
        return self.__class__(**item_dict)

    new_cls = type(typename, (cls,), {
        '__getitem__': __getitem__,
        'to': to
    })

    return new_cls

# def import_str(cls_str: str):
#     parts = cls_str.split('.')
#     module = ".".join(parts[:-1])
#     m = __import__(module)
#     for comp in parts[1:]:
#         m = getattr(m, comp)
#     return m


def import_str(string: str):
    """ Import a python module given string paths

    Args:
        string (str): The given paths

    Returns:
        Any: Imported python module / object
    """
    # From https://github.com/CompVis/taming-transformers
    module, cls = string.rsplit(".", 1)
    return getattr(importlib.import_module(module, package=None), cls)


def zip_dict(d: dict):
    # https://stackoverflow.com/a/69578838/11121534
    """
    A generator that zip dict's values
    
    Example:
        >>> d = dict(
        >>>     x=[1,3,5],
        >>>     y=[2,4,6])

        >>> for t in zip_dict(d):
        >>>     print(t)

        The result:
            {'x': 1, 'y': 2}
            {'x': 3, 'y': 4}
            {'x': 5, 'y': 6}
    """
    for vals in zip(*(d.values())):
        yield dict(zip(d.keys(), vals))


def zip_two_dict(d1: dict, d2: dict):
    """
    A generator that zip two dicts' values, similar to `zip_dict`
    """
    for vals1, vals2 in zip(zip(*(d1.values())), zip(*d2.values())):
        yield dict(zip(d1.keys(), vals1)), dict(zip(d2.keys(), vals2))


def nested_dict_keys(d: dict, pre=None):
    """
    A generator for DFS traversal of a nested dict's key tree
    """
    pre = pre[:] if pre else []
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, dict):
                yield from nested_dict_keys(v, pre + [k])
            else:
                yield pre + [k]
    else:
        yield pre


def nested_dict_values(d: dict):
    """
    A generator for DFS traversal of a nested dict's values
    """
    for v in d.values():
        if isinstance(v, dict):
            yield from nested_dict_values(v)
        else:
            yield v


def nested_dict_items(d: dict, pre=None):
    # https://stackoverflow.com/questions/12507206/how-to-completely-traverse-a-complex-dictionary-of-unknown-depth
    """
    A generator for DFS traversal of a nested dict's keytree-value pairs

    Example:
        >>> for *k, v in nested_dict_items(d):
        >>>    print(k, v)
    """
    pre = pre[:] if pre else []
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, dict):
                # for d in nested_dict_items(value, pre + [k]):
                #     yield d
                # NOTE: equivalent
                yield from nested_dict_items(v, pre + [k])
            else:
                yield pre + [k, v]
    else:
        yield pre + [d]


def nested_dict_del(d: dict):
    """
    DFS deletion of a nested dict
    """
    for k in list(d.keys()):
        if isinstance(d[k], dict):
            nested_dict_del(d[k])
        del d[k]


def nested_dict(iterable: Iterable[Tuple[Iterable[KT], VT]]):
    """
    Construct a nested dict given lists of keytree-value pairs
    """
    d = dict()
    for ks, v in iterable:
        _d = d
        for _k in ks[:-1]:
            if _k not in _d:
                _d[_k] = dict()
            _d = _d[_k]
        _d[ks[-1]] = v
    return d


def zip_nested_dict(d: dict):
    """
    `zip` a (nested) dict (also support normal dict)

    Example:
        ====================  nested dict  ======================
        >>> nested_d = dict(
        >>>     x=[1,3,5,7,9],
        >>>     y=[2,4,6,8,10],
        >>>     z=dict(
        >>>         z1=[0.1, 0.2, 0.3, 0.4, 0.5],
        >>>         z2=[-0.1, -0.2, -0.3, -0.4, -0.5]
        >>>     )
        >>> )
        >>> for d in zip_nested_dict(nested_d):
        >>>     print(d)

        => Output
            {'x': 1, 'y': 2, 'z': {'z1': 0.1, 'z2': -0.1}}
            {'x': 3, 'y': 4, 'z': {'z1': 0.2, 'z2': -0.2}}
            {'x': 5, 'y': 6, 'z': {'z1': 0.3, 'z2': -0.3}}
            {'x': 7, 'y': 8, 'z': {'z1': 0.4, 'z2': -0.4}}
            {'x': 9, 'y': 10, 'z': {'z1': 0.5, 'z2': -0.5}}


        ====================  normal dict  ======================
        >>> normal_d = dict(
        >>>     x=[1,3,5,7,9],
        >>>     y=[2,4,6,8,10]
        >>> )
        >>> for d in zip_nested_dict(normal_d):
        >>>     print(d)

        => Output
            {'x': 1, 'y': 2}
            {'x': 3, 'y': 4}
            {'x': 5, 'y': 6}
            {'x': 7, 'y': 8}
            {'x': 9, 'y': 10}

    """
    for vals in zip(*(nested_dict_values(d))):
        yield nested_dict(zip(nested_dict_keys(d), vals))


def zip_two_nested_dict(d1: dict, d2: dict):
    """
    `zip` two nested dicts, similar to `zip_nested_dict`
    """
    for vals1, vals2 in zip(zip(*nested_dict_values(d1)), zip(*nested_dict_values(d2))):
        cur_d1 = nested_dict(zip(nested_dict_keys(d1), vals1))
        cur_d2 = nested_dict(zip(nested_dict_keys(d2), vals2))
        yield cur_d1, cur_d2


def collate_nested_dict(batch: List[dict], stack=True) -> dict:
    """ Collate a list of [nested dict] (each item with the same nested dict structure)

    Args:
        batch (List[dict]): The input list of [nested dict]
        stack (bool, optional): Whether to stack tensor lists. Defaults to True.

    Returns:
        dict: The collated nested dict (of list)
    """
    elem = batch[0]
    if isinstance(elem, dict):
        return {k: collate_nested_dict([d[k] for d in batch], stack=stack) for k in elem.keys()}
    elif isinstance(elem, torch.Tensor) and stack:
        return torch.stack(batch, 0)
    else:
        return batch  # Leave untouched (list)

def collate_tuple_of_nested_dict(batch: List[Tuple[dict]], stack=True) -> Tuple[dict]:
    """ Collate a list of [tuple of nested dicts]

    Args:
        batch (List[Tuple[dict]]): The input list of [tuple of nested dict]
        stack (bool, optional): Whether to stack tensor lists. Defaults to True.

    Returns:
        Tuple[dict]: The tuple of collacted nested dict (of list)
    """
    batch = zip(*batch)
    return tuple([collate_nested_dict(d, stack=stack) for d in batch])

def partialclass(cls, *args, **kwds):
    """
    Construct a class with partially hardcoded __init__
    """
    class NewCls(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwds)
    NewCls.__name__ = cls.__name__  # to preserve old class name.
    return NewCls


def extend_instance(obj, cls):
    """
    Apply mixins to a class instance after creation
    Refer to: https://stackoverflow.com/a/31075641/11121534
    """
    base_cls = obj.__class__
    base_cls_name = obj.__class__.__name__
    obj.__class__ = type(base_cls_name, (cls, base_cls), {})


def torch_dtype(dtype: Union[torch.dtype, str]):
    """
    Convert a None, str, torch.dtype object to corresponding torch.dtype
    """
    if dtype is None:
        return torch.get_default_dtype()
    elif isinstance(dtype, str):
        if dtype == 'float' or dtype == 'float32':
            return torch.float
        elif dtype == 'half' or dtype == 'float16':
            return torch.float16
        elif dtype == 'double' or dtype == 'float64':
            return torch.float64
        else:
            raise RuntimeError(f"Invalid (str) dtype={dtype}")
    elif isinstance(dtype, torch.dtype):
        return dtype
    else:
        raise RuntimeError(f"Invalid type of `dtype`: {type(dtype)}")

def check_to_torch(
    x: Union[np.ndarray, torch.Tensor, List, Tuple],
    ref: torch.Tensor=None, dtype: torch.dtype=None, device: torch.device=None) -> torch.Tensor:
    """ Check and convert input `x` to torch.Tensor

    Args:
        x (Union[np.ndarray, torch.Tensor, List, Tuple]): Input
        ref (torch.Tensor, optional): Reference tensor for dtype and device. Defaults to None.
        dtype (torch.dtype, optional): Target torch.dtype. Defaults to None.
        device (torch.device, optional): Target torch.device. Defaults to None.

    Returns:
        torch.Tensor: Converted torch.Tensor
    """
    if ref is not None:
        if dtype is None:
            dtype = ref.dtype
        if device is None:
            device = ref.device
    if x is None:
        return x
    elif isinstance(x, torch.Tensor):
        return x.to(dtype=dtype or x.dtype, device=device or x.device)
    else:
        return torch.tensor(x, dtype=dtype, device=device)

def img_to_torch_and_downscale(
    x: Union[np.ndarray, torch.Tensor], *, 
    downscale:float=1, use_cpu_downscale=False, antialias=False, 
    dtype=None, device=None):
    """ Check, convert and apply downscale to input image `x`
    
    Args:
        x (Union[np.ndarray, torch.Tensor]): [H, W, (...)] Input image
        downscale (float, optional): Downscaling ratio. Defaults to 1.
        use_cpu_downscale (bool, optional): Whether use CPU downscaling algo (T), or use GPU (F). Defaults to False.
        antialias (bool, optional): Whether use anti-aliasing. Defaults to False.
        dtype (torch.dtype, optional): Output torch.dtype. Defaults to None.
        device (torch.device, optional): Output torch.device. Defaults to None.

    Returns:
        torch.Tensor: [new_H, new_W, (...)] Converted and downscaled torch.Tensor image
    """
    H, W, *_ = x.shape
    if downscale != 1:
        H_ = int(H // downscale)
        W_ = int(W // downscale)
        if use_cpu_downscale:
            x_np = x if isinstance(x, np.ndarray) else x.data.cpu().numpy()
            x = torch.tensor(cpu_resize(x_np, (H_, W_), anti_aliasing=antialias),
                             dtype=dtype, device=device)
        else:
            x = check_to_torch(x, dtype=dtype, device=device)
            x = x.cuda() if not x.is_cuda else x
            if x.dim() == 2:
                x = gpu_resize(x.unsqueeze(0), (H_, W_), antialias=antialias).squeeze(0)
            else:
                x = gpu_resize(x.movedim(-1, 0), (H_, W_), antialias=antialias).movedim(0, -1)
        assert [H_, W_] == [*x.shape[:2]]
    return check_to_torch(x, dtype=dtype, device=device)

def pad_images_to_same_size(
    imgs: List[Union[torch.Tensor, np.ndarray]], value=0, 
    padding: Literal['bottom_right', 'bottom_left', 'top_left', 'top_right']='bottom_right'):
    """ Pad images of possibly different sizes to have the same size
    NOTE: Expects inputs to be HW(C)

    Args:
        imgs (List[Union[torch.Tensor, np.ndarray]]): HW(C) Input images of different sizes
        value (int, optional): The padded value. Defaults to 0.

    Returns:
        List[Union[torch.Tensor, np.ndarray]]: The padded images, all with the same size.
    """
    max_height = max([im.shape[0] for im in imgs])
    max_width = max([im.shape[1] for im in imgs])
    
    if all([(im.shape[0] == max_height) and (im.shape[1] == max_width) for im in imgs]):
        # Already satisfied.
        return imgs
    
    if padding == 'bottom_right':
        if isinstance(imgs[0], torch.Tensor):
            if imgs[0].dim() == 2:
                padded_imgs = [ # NOTE: Axis order is reversed.
                    F.pad(im.unsqueeze(0), (0, max_width-im.shape[1], 0, max_height-im.shape[0]), 
                        mode='constant', value=value).squeeze(0) for im in imgs
                ]
            else:
                padded_imgs = [ # NOTE: Axis order is reversed.
                    F.pad(im.permute(2, 0, 1), (0, max_width-im.shape[1], 0, max_height-im.shape[0]), 
                        mode='constant', value=value).permute(1, 2, 0) for im in imgs
                ]
        elif isinstance(imgs[0], np.ndarray):
            padded_imgs = [
                np.pad(im, ((0, max_height-im.shape[0]), (0, max_width-im.shape[1]), (0, 0)), 
                    mode='constant', constant_values=value) for im in imgs
            ]
        else:
            raise RuntimeError(f"Invalid input type={type(imgs[0])}")
    elif padding == 'bottom_left':
        if isinstance(imgs[0], torch.Tensor):
            if imgs[0].dim() == 2:
                padded_imgs = [
                    F.pad(im.unsqueeze(0), (max_width-im.shape[1], 0, 0, max_height-im.shape[0]), 
                        mode='constant', value=value).squeeze(0) for im in imgs
                ]
            else:
                padded_imgs = [
                    F.pad(im.permute(2, 0, 1), (max_width-im.shape[1], 0, 0, max_height-im.shape[0]), 
                        mode='constant', value=value).permute(1, 2, 0) for im in imgs
                ]
        elif isinstance(imgs[0], np.ndarray):
            padded_imgs = [
                np.pad(im, ((0, max_height-im.shape[0]), (max_width-im.shape[1], 0), (0, 0)), 
                    mode='constant', constant_values=value) for im in imgs
            ]
    elif padding == 'top_left':
        if isinstance(imgs[0], torch.Tensor):
            if imgs[0].dim() == 2:
                padded_imgs = [
                    F.pad(im.unsqueeze(0), (max_width-im.shape[1], 0, max_height-im.shape[0], 0), 
                        mode='constant', value=value).squeeze(0) for im in imgs
                ]
            else:
                padded_imgs = [
                    F.pad(im.permute(2, 0, 1), (max_width-im.shape[1], 0, max_height-im.shape[0], 0), 
                        mode='constant', value=value).permute(1, 2, 0) for im in imgs
                ]
        elif isinstance(imgs[0], np.ndarray):
            padded_imgs = [
                np.pad(im, ((max_height-im.shape[0], 0), (max_width-im.shape[1], 0), (0, 0)), 
                    mode='constant', constant_values=value) for im in imgs
            ]
    elif padding == 'top_right':
        if isinstance(imgs[0], torch.Tensor):
            if imgs[0].dim() == 2:
                padded_imgs = [
                    F.pad(im.unsqueeze(0), (0, max_width-im.shape[1], max_height-im.shape[0], 0), 
                        mode='constant', value=value).squeeze(0) for im in imgs
                ]
            else:
                padded_imgs = [
                    F.pad(im.permute(2, 0, 1), (0, max_width - im.shape[1], max_height - im.shape[0], 0), 
                        mode='constant', value=value).permute(1, 2, 0) for im in imgs
                ]
        elif isinstance(imgs[0], np.ndarray):
            padded_imgs = [
                np.pad(im, ((max_height-im.shape[0], 0), (0, max_width-im.shape[1]), (0, 0)), 
                    mode='constant', constant_values=value) for im in imgs
            ]
    else:
        raise RuntimeError(f"Invalid padding={padding}")
    
    return padded_imgs

# def ModelWithBucket(cls):
#     if not hasattr(cls, 'function_bucket'):
#         cls.function_bucket = dict()

#     cls.__oldinit__ = cls.__init__
#     @functools.wraps(cls.__init__)
#     def wrapper(self, *args, **kwargs):
#         # copy from class variable to self variable
#         self.function_bucket = deepcopy(cls.function_bucket)
#         cls.__oldinit__(self, *args, **kwargs)
#     cls.__init__ = wrapper

#     for basecls in inspect.getmro(cls):
#         for name, method in basecls.__dict__.items():
#             if hasattr(method, "mark_add_to_bucket") and method.mark_add_to_bucket:
#                 # cls.function_bucket[name] = method
#                 cls.function_bucket[method.name_in_bucket] = method
#     return cls

# def mark_add_to_bucket(new_name=None):
#     # https://stackoverflow.com/questions/2366713/can-a-decorator-of-an-instance-method-access-the-class
#     def decorator(method):
#         method.mark_add_to_bucket = True
#         method.name_in_bucket = method.__name__ if new_name is None else new_name
#         # if 'objs' not in inspect.signature(method).parameters:
#         #     @functools.wraps(method)
#         #     def wrapper(*args, obj=None, **kwargs):
#         #         return method(*args, **kwargs)
#         # else:
#         #     wrapper = method
#         return method
#     return decorator


def list_contains(l: list, v):
    """
    Whether any item in `l` contains `v`
    """
    for item in l:
        if v in item:
            return True
    else:
        return False


def key_contains(d: dict, v: str):
    """
    Whether any keys of `d` contains `v`
    """
    for *ks, _ in nested_dict_items(d):
        for k in ks:
            if v in k:
                return True
    return False


def tensor_statistics(t: Union[torch.Tensor, np.ndarray], prefix: str='', metrics: List[str] = None) -> Dict[str, float]:
    """ Generate statistics data dict for a given tensor

    Args:
        t (Union[torch.Tensor, np.ndarray]): The given tensor
        prefix (str, optional): dict key prefix. Defaults to ''.
        metrics (List[str], optional): Optionally manually given statistics metrics. Defaults to None.

    Returns:
        Dict[str, float]: The statistics data dict
    """
    data = t.detach().float() if isinstance(t, torch.Tensor) else t.astype(np.float32)
    if prod(t.shape) != 1:
        metric_fn = {
            "mean": lambda x: x.mean(),
            "std": lambda x: x.std(),
            "min": lambda x: x.min(),
            "max": lambda x: x.max(),
            "absmax": lambda x: x.abs().max(),
            "norm": lambda x: x.norm() if isinstance(x, torch.Tensor) else np.linalg.norm(x)
        }
    else:
        metric_fn = {
            "val": lambda x: x,
            "mean": lambda x: x
        }
    if metrics is None:
        metrics = metric_fn.keys()
    return {f"{prefix}{'.' if prefix and not prefix.endswith('.') else ''}{key}": metric_fn[key](data).item() for key in metrics if key in metric_fn}


def get_shape(t):
    if isinstance(t, (torch.Tensor, np.ndarray)):
        return t.shape
    elif isinstance(t, Iterable):
        return (len(t),)
    elif isinstance(t, numbers.Number):
        return ()
    else:
        raise RuntimeError(f"Can not get shape of type={type(t)}")

def is_scalar(t: Union[torch.Tensor, np.ndarray]):
    return isinstance(t, numbers.Number) or (hasattr(t, 'shape') and prod(t.shape) == 1)

# ---------------------------------------------
# ----------------     File     ---------------
# ---------------------------------------------


def glob_imgs(path: str) -> List[str]:
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
        imgs.extend(glob.glob(os.path.join(path, ext)))
    return imgs


def get_image_size(path: str) -> Tuple[int,int]:
    """ Fast image size getting, **without loading the full image into memory**
        
        Source: https://stackoverflow.com/a/52321252/11121534
        
        Efficiency: (Comparing imagesize.get, magic.from_file, and PIL image)
        >>> imagesize.get (0.019s) > PIL(0.104s) > magic with regex (0.1699s)

    Args:
        path (str): Given image file path

    Returns:
        Tuple[int,int]: W, H
    """
    width, height = imagesize.get(path)
    return width, height


def crop_image(image: Image.Image, bbox: Tuple[int, int, int, int]):
    """
    Crops PIL image using bounding box.

    Args:
        image (PIL.Image): Image to be cropped.
        bbox (tuple): Integer bounding box (xyxy). [xmin, ymin, xmax, ymax] x<=>W y<=>H
    """
    bbox = np.array(bbox)
    if image.mode == "RGB":
        default = (255, 255, 255)
    elif image.mode == "RGBA":
        default = (255, 255, 255, 255)
    else:
        default = 0
    bg = Image.new(image.mode, (bbox[2] - bbox[0], bbox[3] - bbox[1]), default)
    bg.paste(image, (-bbox[0], -bbox[1]))
    return bg


def crop_image_np(image: np.ndarray, bbox: Tuple[int, int, int, int]):
    """
    Crops np.ndarray image using bounding box
    
    Args:
        image: [H, W, ...]
        bbox: [xmin, ymin, xmax, ymax] x<=>W y<=>H
    
    Returns:
        image: [ymax-ymin, xmax-xmin, ...]
    """
    H, W, *C = image.shape
    new_image = np.zeros([bbox[3] - bbox[1], bbox[2] - bbox[0], *C], dtype=image.dtype)

    # new_image[-bbox[1]:H-bbox[1], -bbox[0]:W-bbox[0]] = image
    v0 = np.clip(-bbox[1], 0, bbox[3] - bbox[1] - 1)
    v1 = np.clip(H - bbox[1], 0, bbox[3] - bbox[1] - 1)
    u0 = np.clip(-bbox[0], 0, bbox[2] - bbox[0] - 1)
    u1 = np.clip(W - bbox[0], 0, bbox[2] - bbox[0] - 1)
    new_image[v0:v1, u0:u1, ...] = image[v0 + bbox[1]:v1 + bbox[1], u0 + bbox[0]:u1 + bbox[0], ...]
    return new_image


def load_rgb(path: str, downscale: numbers.Number = 1) -> np.ndarray:
    """ Load image

    Args:
        path (str): Given image file path
        downscale (numbers.Number, optional): Optional downscale ratio. Defaults to 1.

    Returns:
        np.ndarray: [H, W, 3], in range [0,1]
    """
    img = imageio.imread(path)
    img = skimage.img_as_float32(img)
    if downscale != 1:
        H, W, _ = img.shape
        img = cpu_resize(img, (int(H // downscale), int(W // downscale)), anti_aliasing=False)
    # [H, W, 3]
    return img

def cond_mkdir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def backup_folder(
    backup_dir: str,
    source_dir: str="./",
    filetypes_to_copy=[".py", ".h", ".cpp", ".cuh", ".cu", ".sh"]
):
    filetypes_to_copy = tuple(filetypes_to_copy)
    os.makedirs(backup_dir, exist_ok=True)
    for file in os.listdir(source_dir):
        if not file.endswith(filetypes_to_copy):
            continue
        source_file_path = os.path.join(source_dir, file)
        target_file_path = os.path.join(backup_dir, file)
        shutil.copy(source_file_path, target_file_path)

def backup_folder_recursive(
    backup_dir: str,
    source_dir: str="./",
    filetypes_to_copy=[".py", ".h", ".cpp", ".cuh", ".cu", ".sh"]
):
    filetypes_to_copy = tuple(filetypes_to_copy)
    for root, _, files in os.walk(source_dir):
        for file in files:
            if not file.endswith(filetypes_to_copy):
                continue
            source_file_path = os.path.join(root, file)
            # Keeps original directory structure
            target_file_path = os.path.join(backup_dir, os.path.relpath(source_file_path, source_dir))
            target_dir_path = os.path.dirname(target_file_path)
            os.makedirs(target_dir_path, exist_ok=True)
            shutil.copy(source_file_path, target_file_path)

def backup_project(
    backup_dir: str,
    source_dir: str="./",
    subdirs_to_copy=["app", "code_multi", "code_single", "dataio", "nr3d_lib"], 
    filetypes_to_copy=[".py", ".h", ".cpp", ".cuh", ".cu", ".sh"],
):
    filetypes_to_copy = tuple(filetypes_to_copy)
    # Automatic backup codes
    log.info(f"=> Backing up from {source_dir} to {backup_dir}...")
    # Backup project root dir, depth = 1
    backup_folder(backup_dir, source_dir, filetypes_to_copy)
    # Backup cared subdirs, depth = inf
    for subdir in subdirs_to_copy:
        sub_source_dir = os.path.join(source_dir, subdir)
        sub_backup_dir = os.path.join(backup_dir, subdir)
        backup_folder_recursive(sub_backup_dir, sub_source_dir, filetypes_to_copy)
    log.info("done.")

def line_count_project(
    root_dir: str = "./", 
    filetypes_to_count: List[str] = [".py", ".h", ".cpp", ".cuh", ".cu", ".sh"]
    ):
    
    # TODO: Fix behavior in submodules
    
    # NOTE: Ignore files that is configured in gitignore
    from gitignore_parser import parse_gitignore # pip install gitignore-parser
    
    filetypes_to_count = tuple(filetypes_to_count)
    filetypes_line_counts = {k:0 for k in filetypes_to_count}
    
    gitignore_file = os.path.join(root_dir, '.gitignore')
    if os.path.exists(gitignore_file):
        with open(gitignore_file) as f:
            ignore_list = f.read().splitlines()
    ignore_list.append('.git')
    gitignore = parse_gitignore(gitignore_file)
    
    for dirpath, _, filenames in os.walk(root_dir):
        filenames = [f for f in filenames if f.endswith(filetypes_to_count)]
        filenames = [f for f in filenames if not gitignore(os.path.join(dirpath, f))]
        for file in filenames:
            for filetype in filetypes_to_count:
                if file.endswith(filetype):
                    break
            with open(os.path.join(dirpath, file), 'r') as f:
                # https://stackoverflow.com/a/37600991/11121534
                line_count = sum(1 for _ in f)
            filetypes_line_counts[filetype] += line_count
    total_lines = sum(filetypes_line_counts.values())
    filetypes_ratio = {k: v/total_lines for k,v in filetypes_line_counts.items()}
    print(f"=> Total lines: {total_lines}")
    print("=> Different file types:")
    for k in filetypes_line_counts.keys():
        print(f"  {k}: lines: {filetypes_line_counts[k]}, ratio: {filetypes_ratio[k]*100.:.2f}%")

def save_video(imgs, fname, as_gif=False, fps=24, quality=5, already_np=False, gif_scale: int = 512):
    """ Save images to video

    Args:
        imgs ([type]): [0 to 1]
        fname ([type]): [description]
        as_gif (bool, optional): [description]. Defaults to False.
        fps (int, optional): [description]. Defaults to 24.
        quality (int, optional): [description]. Defaults to 5.
    """
    gif_scale = int(gif_scale)
    # convert to np.uint8
    if not already_np:
        imgs = (255 * np.clip(
            imgs.permute(0, 2, 3, 1).detach().cpu().numpy(), 0, 1))\
            .astype(np.uint8)
    imageio.mimwrite(fname, imgs, fps=fps, quality=quality)

    if as_gif:  # save as gif, too
        os.system(f'ffmpeg -i {fname} -r 15 '
                  f'-vf "scale={gif_scale}:-1,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" {os.path.splitext(fname)[0] + ".gif"}')

def wait_for_pid(pid: int):
    """
    Hangs util certain `pid` process ends.
    source: https://stackoverflow.com/a/7654102/11121534
    WARN: This does not work correctly on windows system
    """
    def is_running(pid):        
        try:
            os.kill(pid, 0)
        except OSError as err:
            if err.errno == errno.ESRCH:
                return False
        return True
    while is_running(pid):
        time.sleep(.25)

def is_file_being_written(file_path: str):
    file_path = os.path.abspath(file_path)
    for proc in psutil.process_iter():
        try:
            for item in proc.open_files():
                if file_path == item.path:
                    return True
        except Exception:
            pass
    return False

if __name__ == "__main__":
    def test_line_count():
        line_count_project("./")
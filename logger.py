"""
@file   logger.py
@author Jianfei Guo, Shanghai AI Lab
@brief  A general file & tensorboard logger modified from https://github.com/LMescheder/GAN_stability/blob/master/gan_training/logger.py
"""

import os
import pickle
import imageio
import torchvision
import numpy as np
from math import prod
from numbers import Number
from typing import List, Literal, Optional, Union

import torch
import torch.nn as nn
import torch.distributed as dist

from nr3d_lib.fmt import log
from nr3d_lib.plot import figure_to_image
from nr3d_lib.utils import cond_mkdir, is_scalar, nested_dict_items, tensor_statistics

try:
    # NOTE: Since torch 1.2
    from torch.utils.tensorboard import SummaryWriter
    # from tensorboardX import SummaryWriter
except ImportError:
    log.warning("tensorboard is not installed.")

try:
    open3d_enabled = True
    from open3d.visualization.tensorboard_plugin import summary
    from open3d.visualization.tensorboard_plugin.util import to_dict_batch
except:
    open3d_enabled = False
    log.info("Unable to load open3d's plugin for tensorboard.")

#---------------------------------------------------------------------------
#---------------------- tensorboard / image recorder -----------------------
#---------------------------------------------------------------------------

class Logger(object):
    # https://github.com/LMescheder/GAN_stability/blob/master/gan_training/logger.py
    def __init__(
        self,
        root: str, img_root: str=None, enable_3d = False,
        monitoring: Literal['tensorboard']=None, monitoring_dir: Optional[str]=None,
        rank=0, is_master=True, multi_process_logging=False):
        self.stats = dict()
        self.root = root
        self.img_root = img_root
        self.should_save_imgs = self.img_root is not None
        self.enable_3d = enable_3d
        self.rank = rank
        self.is_master = is_master
        self.multi_process_logging = multi_process_logging

        if self.is_master:
            cond_mkdir(self.root)
            if self.should_save_imgs:
                cond_mkdir(self.img_root)
        if self.multi_process_logging:
            dist.barrier()

        self.monitoring = None
        self.monitoring_dir = None
        self.last_step = None

        # if self.is_master:
        
        # NOTE: For now, we are allowing tensorboard writting on all child processes, 
        #       as it's already nicely supported, 
        #       and the data of different events file of different processes will be automatically aggregated when visualizing.
        #       https://discuss.pytorch.org/t/using-tensorboard-with-distributeddataparallel/102555/7
        self.monitoring = monitoring
        if not (monitoring is None or monitoring == 'none'):
            self.monitoring_dir = monitoring_dir if monitoring_dir is not None else os.path.join(self.root, 'events')
            if monitoring == 'tensorboard':
                self.tb = SummaryWriter(self.monitoring_dir)
            else:
                raise NotImplementedError(f'Monitoring tool "{monitoring}" not supported!')

    def add(self, category: str, k: str, v: Number, it: int):
        self.last_step = it
        tag = '/'.join([category, k])
        
        self.stats.setdefault(category, {}).setdefault(k, []).append((it, v))
        
        if self.monitoring == 'telemetry':
            self.tm.metric_push_async({
                'metric': tag, 'value': v, 'it': it
            })
        elif self.monitoring == 'tensorboard':
            self.tb.add_scalar(tag, v, it)


    def add_vector(self, category: str, k: str, v: Union[np.ndarray, torch.Tensor], it: int):
        self.last_step = it
        
        if isinstance(v, torch.Tensor):
            v = v.data.cpu().numpy()
        self.stats.setdefault(category, {}).setdefault(k, []).append((it, v))

    def add_imgs(self, category: str, k: str, imgs: Union[np.ndarray, torch.Tensor], it: int):
        self.last_step = it
        tag = '/'.join([category, k])
        
        if self.should_save_imgs:
            outdir = os.path.join(self.img_root, tag)
            if self.is_master:
                os.makedirs(outdir, exist_ok=True)
            if self.multi_process_logging:
                dist.barrier()
            outfile = os.path.join(outdir, f'{it:08d}_{self.rank}.png')
            
            if isinstance(imgs, np.ndarray):
                imageio.imwrite(outfile, imgs)
            else:
                # imgs = imgs / 2 + 0.5
                imgs = torchvision.utils.make_grid(imgs)
                torchvision.utils.save_image(imgs.clone(), outfile, nrow=8)

        if self.monitoring == 'tensorboard':
            dataformats = 'HWC' if isinstance(imgs, np.ndarray) else 'CHW'
            self.tb.add_image(tag, imgs, global_step=it, dataformats=dataformats)

    def add_figure(self, category: str, k: str, fig, it: int):
        self.last_step = it
        tag = '/'.join([category, k])
        
        if self.should_save_imgs:
            outdir = os.path.join(self.img_root, tag)
            if self.is_master:
                os.makedirs(outdir, exist_ok=True)
            if self.multi_process_logging:
                dist.barrier()
            outfile = os.path.join(outdir, f'{it:08d}_{self.rank}.png')

            image_hwc = figure_to_image(fig)
            imageio.imwrite(outfile, image_hwc)
            if self.monitoring == 'tensorboard':
                if len(image_hwc.shape) == 3:
                    image_hwc = np.array(image_hwc[None, ...])
                self.tb.add_images(tag, torch.from_numpy(image_hwc), dataformats='NHWC', global_step=it)
        else:
            if self.monitoring == 'tensorboard':
                self.tb.add_figure(tag, fig, it)

    def add_text(self, category: str, k: str, text: str, it: int):
        self.last_step = it
        tag = '/'.join([category, k])
        
        if self.monitoring == 'tensorboard':
            self.tb.add_text(tag, text, it)

    def add_nested_dict(self, category: str, k_prefix: str = '', d: dict = ..., it: int = ..., metrics: List[str] = None):
        self.last_step = it
        for *k, v in nested_dict_items(d):
            if hasattr(v, 'shape') and prod(v.shape) > 1:
                for _k, _v in tensor_statistics(v, metrics=metrics).items():
                    key = '.'.join(k + [_k])
                    self.add(category, f"{k_prefix}{'.' if k_prefix and not k_prefix.endswith('.') else ''}{key}", _v, it)
            elif is_scalar(v):
                key = '.'.join(k)
                self.add(category, f"{k_prefix}{'.' if k_prefix and not k_prefix.endswith('.') else ''}{key}", v, it)

    def add_mesh(self, category: str, k: str, verts: torch.Tensor, *, faces: torch.Tensor = None, colors: torch.Tensor = None, it: int = ...):
        self.last_step = it
        k_name = '/'.join([category, k])
        if self.monitoring == 'tensorboard':
            if verts.dim() == 2:
                verts = verts.unsqueeze(0)
            if faces.dim() == 2:
                faces = faces.unsqueeze(0)
            self.tb.add_mesh(k_name, vertices=verts, colors=colors, faces=faces, global_step=it)

    def add_open3d(self, category: str, k: str, o3d_geo_list: List, it: int):
        self.last_step = it
        k_name = '/'.join([category, k])
        if open3d_enabled and self.monitoring == 'tensorboard':
            if not isinstance(o3d_geo_list, list):
                o3d_geo_list = [o3d_geo_list]
            self.tb.add_open3d(k_name, to_dict_batch(o3d_geo_list), step=it)

    def get_last(self, category: str, k: str, default=0.):
        if category not in self.stats:
            return default
        elif k not in self.stats[category]:
            return default
        else:
            return self.stats[category][k][-1][1]

    def add_module_param(self, module_name: str, module: nn.Module, it: int):
        self.last_step = it
        if self.monitoring == 'tensorboard':
            for name, param in module.named_parameters():
                self.tb.add_histogram(f"{module_name}/{name}", param.detach(), it)

    def save_stats(self, filename: str):
        filename = os.path.join(self.root, filename + f'_{self.rank}')
        with open(filename, 'wb') as f:
            pickle.dump(self.stats, f)

    def load_stats(self, filename: str):
        filename = os.path.join(self.root, filename + f'_{self.rank}')
        if not os.path.exists(filename):
            # log.info(f"=> Not exist: {filename}, will create new after calling save_stats()")
            return

        try:
            with open(filename, 'rb') as f:
                self.stats = pickle.load(f)
                log.info(f"=> Load file: {filename}")
        except EOFError:
            log.info('Warning: log file corrupted!')

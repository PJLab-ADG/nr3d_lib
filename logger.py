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
        log_dir: str, img_dir: str=None, enable_3d = False,
        monitoring: Literal['tensorboard']=None, monitoring_dir: Optional[str]=None,
        rank=0, is_master=True, multi_process_logging=False):
        self.stats = dict()
        self.log_dir = log_dir
        self.img_dir = img_dir
        self.save_imgs = self.img_dir is not None
        self.enable_3d = enable_3d
        self.rank = rank
        self.is_master = is_master
        self.multi_process_logging = multi_process_logging

        if self.is_master:
            cond_mkdir(self.log_dir)
            if self.save_imgs:
                cond_mkdir(self.img_dir)
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
        if not (monitoring is None or monitoring == 'none'):
            self.setup_monitoring(monitoring, monitoring_dir)


    def setup_monitoring(self, monitoring: Literal['tensorboard'], monitoring_dir: str):
        self.monitoring = monitoring
        self.monitoring_dir = monitoring_dir if monitoring_dir is not None else os.path.join(self.log_dir, 'events')
        if monitoring == 'tensorboard':
            self.tb = SummaryWriter(self.monitoring_dir)
        else:
            raise NotImplementedError(f'Monitoring tool "{monitoring}" not supported!')

    def add(self, category: str, k: str, v: Number, it: int):
        self.last_step = it
        if category not in self.stats:
            self.stats[category] = {}
        if k not in self.stats[category]:
            self.stats[category][k] = []

        self.stats[category][k].append((it, v))

        k_name = '/'.join([category, k])
        if self.monitoring == 'telemetry':
            self.tm.metric_push_async({
                'metric': k_name, 'value': v, 'it': it
            })
        elif self.monitoring == 'tensorboard':
            self.tb.add_scalar(k_name, v, it)

    def add_vector(self, category: str, k: str, vec, it: int):
        self.last_step = it
        if category not in self.stats:
            self.stats[category] = {}

        if k not in self.stats[category]:
            self.stats[category][k] = []

        if isinstance(vec, torch.Tensor):
            vec = vec.data.clone().cpu().numpy()

        self.stats[category][k].append((it, vec))

    def add_imgs(self, imgs: Union[np.ndarray, torch.Tensor], class_name: str, it: int):
        self.last_step = it
        outdir = os.path.join(self.img_dir, class_name)
        if not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)
        if self.multi_process_logging:
            dist.barrier()
        outfile = os.path.join(outdir, f'{it:08d}_{self.rank}.png')

        if self.save_imgs:
            if isinstance(imgs, np.ndarray):
                imageio.imwrite(outfile, imgs)
            else:
                # imgs = imgs / 2 + 0.5
                imgs = torchvision.utils.make_grid(imgs)
                torchvision.utils.save_image(imgs.clone(), outfile, nrow=8)

        dataformats = 'HWC' if isinstance(imgs, np.ndarray) else 'CHW'
        if self.monitoring == 'tensorboard':
            self.tb.add_image(class_name, imgs, global_step=it, dataformats=dataformats)

    def add_figure(self, fig, class_name: str, it: int):
        self.last_step = it
        if self.save_imgs:
            outdir = os.path.join(self.img_dir, class_name)
            if self.is_master and not os.path.exists(outdir):
                os.makedirs(outdir)
            if self.multi_process_logging:
                dist.barrier()
            outfile = os.path.join(outdir, f'{it:08d}_{self.rank}.png')

            image_hwc = figure_to_image(fig)
            imageio.imwrite(outfile, image_hwc)
            if self.monitoring == 'tensorboard':
                if len(image_hwc.shape) == 3:
                    image_hwc = np.array(image_hwc[None, ...])
                self.tb.add_images(class_name, torch.from_numpy(image_hwc), dataformats='NHWC', global_step=it)
        else:
            if self.monitoring == 'tensorboard':
                self.tb.add_figure(class_name, fig, it)

    def add_module_param(self, module_name: str, module: nn.Module, it: int):
        self.last_step = it
        if self.monitoring == 'tensorboard':
            for name, param in module.named_parameters():
                self.tb.add_histogram(f"{module_name}/{name}", param.detach(), it)

    def add_text(self, category: str, k: str, text, it: int):
        self.last_step = it
        if self.monitoring == 'tensorboard':
            self.tb.add_text(f"{category}/{k}", text, it)

    def add_nested_dict(self, category: str, prefix: str, d: dict, it: int, metrics: List[str] = None):
        self.last_step = it
        for *k, v in nested_dict_items(d):
            if hasattr(v, 'shape') and prod(v.shape) > 1:
                for _k, _v in tensor_statistics(v, metrics=metrics).items():
                    key = '.'.join(k + [_k])
                    self.add(category, prefix + key, _v, it)
            elif is_scalar(v):
                key = '.'.join(k)
                self.add(category, prefix + key, v, it)

    def add_3d(self, category: str, k: str, o3d_geo_list: List, it: int):
        self.last_step = it
        k_name = '/'.join([category, k])
        if open3d_enabled and self.monitoring == 'tensorboard':
            if not isinstance(o3d_geo_list, list):
                o3d_geo_list = [o3d_geo_list]
            self.tb.add_3d(k_name, to_dict_batch(o3d_geo_list), step=it)

    def add_mesh(self, category: str, k: str, verts: torch.Tensor, *, faces: torch.Tensor = None, colors: torch.Tensor = None, it: int = ...):
        self.last_step = it
        k_name = '/'.join([category, k])
        if self.monitoring == 'tensorboard':
            if verts.dim() == 2:
                verts = verts.unsqueeze(0)
            if faces.dim() == 2:
                faces = faces.unsqueeze(0)
            self.tb.add_mesh(k_name, vertices=verts, colors=colors, faces=faces, global_step=it)

    def get_last(self, category: str, k: str, default=0.):
        if category not in self.stats:
            return default
        elif k not in self.stats[category]:
            return default
        else:
            return self.stats[category][k][-1][1]

    def save_stats(self, filename: str):
        filename = os.path.join(self.log_dir, filename + f'_{self.rank}')
        with open(filename, 'wb') as f:
            pickle.dump(self.stats, f)

    def load_stats(self, filename: str):
        filename = os.path.join(self.log_dir, filename + f'_{self.rank}')
        if not os.path.exists(filename):
            # log.info(f"=> Not exist: {filename}, will create new after calling save_stats()")
            return

        try:
            with open(filename, 'rb') as f:
                self.stats = pickle.load(f)
                log.info(f"=> Load file: {filename}")
        except EOFError:
            log.info('Warning: log file corrupted!')

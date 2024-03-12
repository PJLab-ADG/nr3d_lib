"""
@file   utils.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Utility funcitons for conditional SDF model.
"""

__all__ = [
    'pretrain_conditional_sdf_sphere'
]

from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.cuda.amp.grad_scaler import GradScaler

from nr3d_lib.logger import Logger
from nr3d_lib.models.model_base import ModelMixin
from nr3d_lib.models.loss.safe import safe_mse_loss
from nr3d_lib.utils import check_to_torch, tensor_statistics

def pretrain_conditional_sdf_sphere(
    implicit_surface: ModelMixin, 
    # Training configs
    num_iters=5000, num_points=5000, lr=1.0e-4, w_eikonal=1.0e-3, safe_mse = True, clip_grad_val: float = 0.1, 
    # Shape configs
    target_radius=0.5, # In (uni-scaled) obj coords
    target_radius_in_net: float = None, # In network coords
    target_origin=None, # In (uni-scaled) obj coords
    inside_out=False, 
    # bounding_size=2.0, 
    # aabb=None,
    # Batched (If the model is batched)
    batch_size: int=None, z: torch.Tensor=None, resample_z=False,
    # Debug & logging related
    logger: Logger=None, log_prefix: str=None, debug_param_detail=False):
    """
    Pretrain sdf to be a sphere
    """
    device = next(implicit_surface.parameters()).device

    implicit_surface.training_before_per_step(0)

    # if aabb is None:
    #     aabb_min = torch.ones([3,], device=device) * (-bounding_size/2.)
    #     aabb_max = torch.ones([3,], device=device) * (bounding_size/2.)
    # else:
    #     aabb = check_to_torch(aabb).reshape([2,3])
    #     aabb_min, aabb_max = aabb[0], aabb[1]
    # scale = (aabb_max-aabb_min)/2.
    # center = (aabb_max+aabb_min)/2.
    
    if target_origin is None:
        # target_origin = (aabb_max+aabb_min)/2.
        target_origin = implicit_surface.space.center.clone()
    else:
        target_origin = check_to_torch(target_origin).reshape([3,])

    if log_prefix is None: 
        log_prefix = implicit_surface.__class__.__name__
    if resample_z:
        assert z is not None
        z_dim = z.shape[-1]
        dtype = z.dtype
    prefix_shape = [] if batch_size is None else [batch_size]

    optimizer = optim.Adam(implicit_surface.parameters(), lr=lr)
    scaler = GradScaler(init_scale=128.0)
    if safe_mse:
        loss_eikonal_fn = lambda x: safe_mse_loss(x, x.new_ones(x.shape), reduction='mean', limit=1.0)
    else:
        loss_eikonal_fn = lambda x: F.mse_loss(x, x.new_ones(x.shape), reduction='mean')

    assert bool(target_radius is not None) != bool(target_radius_in_net is not None), \
        "Please specify only one of `target_radius` and `target_radius_in_net`"

    with torch.enable_grad():
        with tqdm(range(num_iters), desc=f"=> Pretraining SDF...") as pbar:
            for it in pbar:
                if resample_z:
                    z = torch.randn([*prefix_shape, z_dim], device=device, dtype=dtype)
                if z is not None and hasattr(implicit_surface, 'set_condition'):
                    implicit_surface.set_condition(z)
                
                samples_in_net = torch.empty([*prefix_shape, num_points, 3], dtype=torch.float, device=device).uniform_(-1+1e-6,1-1e-6)
                
                if target_radius_in_net is not None:
                    #---- `target_radius` defined in net coords
                    sdf_gt = samples_in_net.norm(dim=-1) - target_radius
                else:
                    #---- `target_radius` defined in (uni-scaled) obj coords
                    samples_in_obj = implicit_surface.space.cur_batch__unnormalize_coords(samples_in_net)
                    sdf_gt_in_obj = (samples_in_obj - target_origin).norm(dim=-1) - target_radius
                    # Convert SDF GT value in real-world obj's unit to network's unit
                    sdf_gt = sdf_gt_in_obj / implicit_surface.sdf_scale

                if inside_out:
                    sdf_gt *= -1

                pred = implicit_surface.forward_sdf_nablas(samples_in_net)
                pred_sdf = pred['sdf']
                nablas_norm = pred['nablas'].norm(dim=-1)
                
                loss_sdf = F.l1_loss(pred_sdf, sdf_gt, reduction='mean')
                loss_eikonal = (w_eikonal * loss_eikonal_fn(nablas_norm)) if w_eikonal > 0 else 0
                loss = loss_sdf + loss_eikonal
                
                optimizer.zero_grad()
                
                # loss.backward()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if clip_grad_val is not None:
                    torch.nn.utils.clip_grad.clip_grad_value_(implicit_surface.parameters(), clip_grad_val)
                
                # optimizer.step()
                scaler.step(optimizer)
                scaler.update()
                
                pbar.set_postfix(loss=loss.item())
                if logger is not None:
                    logger.add(f"initialize", log_prefix + '.loss', loss.item(), it)
                    logger.add(f"initialize", log_prefix + '.loss_sdf', loss_sdf.item(), it)
                    if w_eikonal > 0:
                        logger.add(f"initialize", log_prefix + '.loss_eikonal', loss_eikonal.item(), it)
                    logger.add_nested_dict("initialize", log_prefix + '.sdf', tensor_statistics(pred_sdf), it)
                    logger.add_nested_dict("initialize", log_prefix + '.nablas_norm', tensor_statistics(nablas_norm), it)
                    if debug_param_detail:
                        if hasattr(implicit_surface, 'encoding'):
                            logger.add_nested_dict("initialize", log_prefix + '.encoding', implicit_surface.encoding.stat_param(with_grad=True), it)
                        elif hasattr(implicit_surface, 'mll'):
                            logger.add_nested_dict("initialize", log_prefix + '.mll', implicit_surface.mll.stat_param(with_grad=True), it)
                            
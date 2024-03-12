"""
@file   utils.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Forest SDF fields utility functions.
"""

__all__ = [
    'pretrain_forest_sdf_capsule', 
    'pretrain_forest_sdf_flat', 
    'pretrain_forest_sdf_road_surface'
]

from tqdm import tqdm
from typing import Literal

import torch
from torch import optim
import torch.nn.functional as F
from torch.cuda.amp.grad_scaler import GradScaler

from nr3d_lib.logger import Logger
from nr3d_lib.utils import tensor_statistics

from nr3d_lib.models.loss.safe import safe_mse_loss
from nr3d_lib.models.fields_forest.sdf.lotd_forest_sdf import LoTDForestSDF

def pretrain_forest_sdf_capsule(
    forest_sdf_net: LoTDForestSDF, tracks_in_obj: torch.Tensor, *, 
    # Shape configs
    surface_distance: float = 0.2, # In obj's coords.
    surface_distance_in_net: float = None, # In net's coords
    # Training configs
    num_iters: int = 500, num_points: int = 10000, 
    lr: float = 1.0e-3, w_eikonal: float = 3.0e-3, 
    clip_grad_val: float = 0.1, safe_mse = True, 
    logger: Logger = None, log_prefix: str = ''):
    """
    Pretrain SDF to be capsule-shaped, surrounding given tracks.
    """
    optimizer = optim.Adam(forest_sdf_net.parameters(), lr=lr)
    use_half = (forest_sdf_net.dtype == torch.float16) or (forest_sdf_net.encoding.dtype == torch.float16 or forest_sdf_net.decoder.dtype == torch.float16) 
    scaler = GradScaler(init_scale=128.0, enabled=use_half)

    forest_sdf_net.training_before_per_step(0)

    if safe_mse:
        loss_eikonal_fn = lambda x: safe_mse_loss(x, x.new_ones(x.shape), reduction='mean', limit=1.0)
    else:
        loss_eikonal_fn = lambda x: F.mse_loss(x, x.new_ones(x.shape), reduction='mean')
    
    assert bool(surface_distance is not None) != bool(surface_distance_in_net is not None), \
        "Please specify only one of `surface_distance` and `surface_distance_in_net`"
    if surface_distance_in_net is not None:
        surface_distance = surface_distance_in_net * forest_sdf_net.sdf_scale
    
    with torch.enable_grad():
        with tqdm(range(num_iters), desc=f"=> Pretraining SDF...") as pbar:
            for it in pbar:
                block_x, blidx = forest_sdf_net.space.sample_pts_uniform(num_pts=num_points)
                samples_in_obj = forest_sdf_net.space.unnormalize_coords(block_x, blidx)

                #---- For each sample point, find the track point of the minimum distance (measured in 3D space.)
                # [num_points, 1, 3] - [1, num_tracks, 3] = [num_points, num_tracks, 3] -> [num_points, num_tracks] -> [num_samples]
                ret_min_dis_in_obj = (samples_in_obj.unsqueeze(-2) - tracks_in_obj.unsqueeze(0)).norm(dim=-1).min(dim=-1)
                
                # For each sample point, current floor'z coordinate of floor_dim
                sdf_gt_in_obj = surface_distance - ret_min_dis_in_obj.values
                # Convert SDF GT value in real-world obj's unit to network's unit
                sdf_gt = sdf_gt_in_obj / forest_sdf_net.sdf_scale
                
                pred_in_net = forest_sdf_net.forward_sdf_nablas(block_x, blidx, input_normalized=True, nablas_has_grad=True)
                
                pred_sdf = pred_in_net['sdf']
                nablas_norm = pred_in_net['nablas'].norm(dim=-1)
                
                loss_sdf = F.l1_loss(pred_sdf, sdf_gt, reduction='mean')
                loss_eikonal = (w_eikonal * loss_eikonal_fn(nablas_norm)) if w_eikonal > 0 else 0
                loss = loss_sdf + loss_eikonal
                
                optimizer.zero_grad()
                
                # loss.backward()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if clip_grad_val > 0:
                    torch.nn.utils.clip_grad.clip_grad_value_(forest_sdf_net.parameters(), clip_grad_val)
                
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

def pretrain_forest_sdf_flat(
    forest_sdf_net: LoTDForestSDF, *, 
    # Shape configs
    floor_dim: Literal['x','y','z'] = 'z', 
    floor_at: float = 0., # In obj's coords
    floor_up_sign: Literal[1, -1]=-1, # [-1] if (-)dim points to sky else [1]
    # Training configs
    num_iters: int = 500, num_points: int = 10000, 
    lr: float = 1.0e-3, w_eikonal: float = 3.0e-3, 
    clip_grad_val: float = 0.1, safe_mse = True, 
    logger: Logger = None, log_prefix: str = ''):
    
    """
    Pretrain SDF to be flat plane.
    """
    floor_dim: int = ['x','y','z'].index(floor_dim)
    
    optimizer = optim.Adam(forest_sdf_net.parameters(), lr=lr)
    use_half = (forest_sdf_net.dtype == torch.float16) or (forest_sdf_net.encoding.dtype == torch.float16 or forest_sdf_net.decoder.dtype == torch.float16) 
    scaler = GradScaler(init_scale=128.0, enabled=use_half)

    forest_sdf_net.training_before_per_step(0)

    if safe_mse:
        loss_eikonal_fn = lambda x: safe_mse_loss(x, x.new_ones(x.shape), reduction='mean', limit=1.0)
    else:
        loss_eikonal_fn = lambda x: F.mse_loss(x, x.new_ones(x.shape), reduction='mean')
    
    with torch.enable_grad():
        with tqdm(range(num_iters), desc=f"=> Pretraining SDF...") as pbar:
            for it in pbar:
                block_x, blidx = forest_sdf_net.space.sample_pts_uniform(num_pts=num_points)
                samples_in_obj = forest_sdf_net.space.unnormalize_coords(block_x, blidx)
                sdf_gt_in_obj = floor_up_sign * (samples_in_obj[..., floor_dim] - floor_at)
                
                # Convert SDF GT value in real-world obj's unit to network's unit
                sdf_gt = sdf_gt_in_obj / forest_sdf_net.sdf_scale
                
                pred = forest_sdf_net.forward_sdf_nablas(block_x, blidx, input_normalized=True, nablas_has_grad=True)
                pred_sdf = pred['sdf']
                nablas_norm = pred['nablas'].norm(dim=-1)
                
                loss_sdf = F.smooth_l1_loss(pred['sdf'], sdf_gt, reduction='mean')
                loss_eikonal = (w_eikonal * loss_eikonal_fn(nablas_norm)) if w_eikonal > 0 else 0
                loss = loss_sdf + loss_eikonal
                
                optimizer.zero_grad()
                
                # loss.backward()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if clip_grad_val > 0:
                    torch.nn.utils.clip_grad.clip_grad_value_(forest_sdf_net.parameters(), clip_grad_val)
                
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

def pretrain_forest_sdf_road_surface(
    forest_sdf_net: LoTDForestSDF, tracks_in_obj: torch.Tensor, *, 
    # Shape configs
    floor_dim: Literal['x','y','z'] = 'z', # The vertical dimension of obj
    floor_up_sign: Literal[1, -1]=-1, # [-1] if (-)dim points to sky else [1]
    ego_height: float = 0., # Estimated ego's height from road, in obj space
    # Training configs
    num_iters: int = 500, num_points: int = 10000, 
    lr: float = 1.0e-3, w_eikonal: float = 3.0e-3, 
    clip_grad_val: float = 0.1, safe_mse = True, 
    # Debug & logging related
    logger: Logger = None, log_prefix: str = ''):
    """
    Pretrain sdf to be a road surface
    """

    floor_dim: int = ['x','y','z'].index(floor_dim)
    other_dims = [i for i in range(3) if i != floor_dim]
    
    optimizer = optim.Adam(forest_sdf_net.parameters(), lr=lr)
    use_half = (forest_sdf_net.dtype == torch.float16) or (forest_sdf_net.encoding.dtype == torch.float16 or forest_sdf_net.decoder.dtype == torch.float16) 
    scaler = GradScaler(init_scale=128.0, enabled=use_half)

    forest_sdf_net.training_before_per_step(0)

    if safe_mse:
        loss_eikonal_fn = lambda x: safe_mse_loss(x, x.new_ones(x.shape), reduction='mean', limit=1.0)
    else:
        loss_eikonal_fn = lambda x: F.mse_loss(x, x.new_ones(x.shape), reduction='mean')
    
    with torch.enable_grad():
        with tqdm(range(num_iters), desc=f"=> Pretraining SDF...") as pbar:
            for it in pbar:
                block_x, blidx = forest_sdf_net.space.sample_pts_uniform(num_pts=num_points)
                samples_in_obj = forest_sdf_net.space.unnormalize_coords(block_x, blidx)
                
                #---- For each sample point, find the track point of the minimum distance (measured in 3D space.)
                # # [num_points, 1, 3] - [1, num_tracks, 3] = [num_points, num_tracks, 3] -> [num_points, num_tracks] -> [num_samples]
                # ret_min_dis_in_obj = (samples_in_obj.unsqueeze(-2) - tracks_in_obj.unsqueeze(0)).norm(dim=-1).min(dim=-1)
                
                #---- For each sample point, find the track point of the minimum distance (measure at 2D space. i.e. xoy plane for floor_dim=z)
                # [num_points, 1, 3] - [1, num_tracks, 3] = [num_points, num_tracks, 3] -> [num_points, num_tracks] -> [num_samples]
                ret_min_dis_in_obj = (samples_in_obj[..., None, other_dims] - tracks_in_obj[None, ..., other_dims]).norm(dim=-1).min(dim=-1)

                # For each sample point, current floor'z coordinate of floor_dim
                floor_at_in_obj = tracks_in_obj[ret_min_dis_in_obj.indices][..., floor_dim] - floor_up_sign * ego_height
                sdf_gt_in_obj = floor_up_sign * (samples_in_obj[..., floor_dim] - floor_at_in_obj)

                # Convert SDF GT value in real-world obj's unit to network's unit
                sdf_gt = sdf_gt_in_obj / forest_sdf_net.sdf_scale
                
                pred = forest_sdf_net.forward_sdf_nablas(block_x, blidx, input_normalized=True, nablas_has_grad=True)
                pred_sdf = pred['sdf']
                nablas_norm = pred['nablas'].norm(dim=-1)
                
                loss_sdf = F.smooth_l1_loss(pred['sdf'], sdf_gt, reduction='mean')
                loss_eikonal = (w_eikonal * loss_eikonal_fn(nablas_norm)) if w_eikonal > 0 else 0
                loss = loss_sdf + loss_eikonal
                
                optimizer.zero_grad()
                
                # loss.backward()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if clip_grad_val > 0:
                    torch.nn.utils.clip_grad.clip_grad_value_(forest_sdf_net.parameters(), clip_grad_val)
                
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
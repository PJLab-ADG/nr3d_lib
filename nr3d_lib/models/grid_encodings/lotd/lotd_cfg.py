"""
@file   lotd_cfg.py
@author Jianfei Guo, Shanghai AI Lab
@brief  LoTD config generators.
"""

__all__ = [
    'get_lotd_cfg', 
]

import numpy as np
from math import prod
from numbers import Number
from typing import Literal, Union, List, Tuple

import torch

from nr3d_lib.fmt import log
from nr3d_lib.fmt import colored_str, GREEN

def get_lotd_cfg(
    type: str, input_ch: int = ..., 
    stretch: Union[float, List[float]]=None, **kwargs
    ) -> dict:
    if type == 'gen_ngp':
        lotd_cfg = gen_ngp_cfg(dim=input_ch, **kwargs)
    elif type == 'single_res':
        lotd_cfg = single_res_cfg(stretch, **kwargs)
    elif type == 'ngp':
        lotd_cfg = auto_ngp_cfg(stretch, dim=input_ch, **kwargs)
    elif type == 'ngp4d':
        lotd_cfg = auto_ngp4d_cfg(dim=input_ch, stretch=stretch, **kwargs)
    elif type == 'lotd':
        lotd_cfg = auto_lotd_cfg_deprecated(stretch, dim=input_ch, **kwargs)
    else:
        raise RuntimeError(f"Invalid type={type}")
    return lotd_cfg

def single_res_cfg(
    stretch: Union[float, List[float]], 
    voxel_size: float = 0.4, 
    n_feats: int = 8, 
    lotd_type: str = 'Dense', **kwargs
    ):
    level_res = (np.array(stretch) / voxel_size).astype(int)
    return dict(lod_res=[level_res.tolist()], lod_n_feats=[n_feats], lod_types=[lotd_type], **kwargs)

def gen_ngp_cfg(
    min_res: int = 16, dim: int = 3, n_feats: int = 2, log2_hashmap_size: int = 19, 
    per_level_scale: float = 1.382, num_levels: int = 16
):
    hashmap_size = 2**log2_hashmap_size
    level_res = min_res * per_level_scale ** np.arange(num_levels)
    level_res = level_res.astype(int)
    level_n_feats = [n_feats] * num_levels
    level_types = [ ("Dense" if res ** dim <= hashmap_size else "Hash")  for res in level_res]
    return dict(lod_res=level_res.tolist(), lod_n_feats=level_n_feats, lod_types=level_types, hashmap_size=hashmap_size)

def auto_ngp_cfg(
    stretch: Union[float, List[float]], target_num_params: int, 
    *, dim: int = 3, n_feats: int = 2, log2_hashmap_size: int = 19, min_res: int = 4, 
    per_level_scale: float = 1.382, max_num_levels: int = 128
):
    """ Automatically computes LoTD configs for [Dense --> Hash] configs

    Args:
        stretch (Union[float, List[float]], optional): 
            Spatial strech ratio or 3D/4D aspect ratios for cuboid / hyper-cuboid representations. 
            A single number would suggest using cubic representation. Defaults to 1.0.
        target_num_params (int): Target total number of params to generate.
        dim (int, optional): Space dimension. Defaults to 3.
        n_feats (int, optional): Feature width. Defaults to 2.
        log2_hashmap_size (int, optional): log_2^{} of the hashmap size. Defaults to 19.
        min_res (int, optional): Minimum resolution of the shortest side. Defaults to 4.
        per_level_scale (float, optional): Scale factor of the next level's resolution to current level's. Defaults to 1.382.
        max_num_levels (int, optional): Maximum number of layers. Defaults to 128.

    Returns:
        dict: Automatically computed LoTD configs. Composed of `lod_res`, `lod_n_feats`, `lod_types`, `hashmap_size`
        
    Algo:
        1. The parameter growth rate for Dense layers is 1.38^3 ~= 2.62
            -> The total number of parameters in all previous dense layers is roughly equivalent to one hash layer
            -> Determine the number of hash layers based on target_num_params and log2_hashmap_size
        
        2. The number of parameters in the last dense layer is approximately 1 / 2.5~2.6 of the hash layer parameters (i.e., roughly maintaining a constant growth rate for the total number of parameters in each layer)
            -> Determine the resolution of the last dense layer based on log2_hashmap_size
            -> Determine the number of dense layers based on the resolution of the last dense layer and min_res
        
        3. Determine all parameters based on the resolution of the last dense layer, the number of dense layers, the number of hash layers, and per_level_scale
    """
    
    stretch = [stretch]*dim if isinstance(stretch, Number) else stretch
    stretch = np.array(stretch)
    
    hashmap_size = 2**log2_hashmap_size
    hash_num_param = hashmap_size * n_feats
    dense_nparam_factor = (stretch / stretch.min()).prod()
    # per_level_param_scale = per_level_scale ** 3
    
    #----------- The total number of all dense levels should be roughly the same with one hash level    
    dense_last_num_param = hashmap_size / 2.5
    dense_last_min_res = int( (dense_last_num_param / dense_nparam_factor) ** (1/3))
    num_dense_levels = np.exp( np.log(dense_last_min_res/min_res) / per_level_scale)
    num_dense_levels = max(int(num_dense_levels + 1), 1)

    num_hash_levels = target_num_params / hash_num_param - 1
    num_hash_levels = max(int(num_hash_levels + 0.5), 0)

    num_levels = num_dense_levels + num_hash_levels
    if max_num_levels is not None:
        num_levels = min(num_levels, max_num_levels)
        num_hash_levels = num_levels - num_dense_levels
    
    dense_last_res = stretch / (stretch.min() / dense_last_min_res)
    
    ngp_res_dense = dense_last_res[..., None] / (per_level_scale ** np.arange(num_dense_levels))
    ngp_res_dense = ngp_res_dense[:, ::-1].T.astype(int) # NOTE: Convert to int at the last moment
    
    ngp_res_hash = dense_last_res[..., None] * (per_level_scale ** (np.arange(num_hash_levels)+1))
    ngp_res_hash = ngp_res_hash.T.astype(int)
    
    ngp_res_all = np.concatenate([ngp_res_dense, ngp_res_hash], axis=0)
    ngp_type_all = ["Dense"] * num_dense_levels + ["Hash"] * num_hash_levels
    ngp_n_feats_all = [n_feats] * num_levels
    
    ngp_num_param = ngp_res_dense.prod(axis=1).sum() * n_feats + num_hash_levels * hashmap_size * n_feats
    
    log.info(colored_str(f"NGP auto-computed config: layer resolutions: {ngp_res_all.tolist()}", GREEN))
    log.info(colored_str(f"NGP auto-computed config: layer types: {ngp_type_all}", GREEN))
    log.info(colored_str(f"NGP auto-computed config: layer n_feats: {ngp_n_feats_all}", GREEN))
    log.info(colored_str(f"NGP auto-computed config: expected num_params={target_num_params}; generated: {ngp_num_param} [{ngp_num_param/target_num_params:.2f}x]", GREEN))
    return dict(lod_res=ngp_res_all.tolist(), lod_n_feats=ngp_n_feats_all, lod_types=ngp_type_all, hashmap_size=hashmap_size)

def auto_ngp4d_cfg(
    dim: int = 4, n_feats: int = 2, 
    stretch: Union[float, List[float]] = 1.0,
    target_num_params: int = 2**32, max_layers = 128,
    min_dense_layers: int = 0, log2_hashmap_size: int = 19,
    min_res_xyz: int = 4, min_res_w: int = 4, per_level_scale: float = 1.382
    ):
    """ Automatically computes LoTD configs for [Dense --> Hash] configs for NeRF++

    Args:
        dim (int, optional): Space dimension. Defaults to 4.
        n_feats (int, optional): Feature width. Defaults to 2.
        stretch (Union[float, List[float]], optional): 
            Spatial strech ratio or 3D/4D aspect ratios for cuboid / hyper-cuboid representations. 
            A single number would suggest using cubic representation. Defaults to 1.0.
        target_num_params (int, optional): Target total number of params to generate. Defaults to 2**32.
        max_layers (int, optional): Maximum number of layers. Defaults to 128.
        min_dense_layers (int, optional): Minimum number of dense layers. Defaults to 0.
        log2_hashmap_size (int, optional): log_2^{} of the hashmap size. Defaults to 19.
        min_res_xyz (int, optional): Minimum resolution of the shortest side of xyz part of xyzw. Defaults to 4.
        min_res_w (int, optional): Minimum resolution of w part of xyzw. Defaults to 4.
        per_level_scale (float, optional): Scale factor of the next level's resolution to current level's. Defaults to 1.382.

    Returns:
        dict: Automatically computed LoTD configs. Composed of `lod_res`, `lod_n_feats`, `lod_types`, `hashmap_size`
    """
    hashmap_size = 2**log2_hashmap_size
    stretch = np.array([stretch] * dim if isinstance(stretch, Number) else stretch)
    base_res = np.concatenate([
        min_res_xyz * stretch / stretch.min(),
        np.array([min_res_w], dtype=np.float32)
    ])
    num_params = 0
    layers_res = []
    layers_type = []

    for i in range(max_layers):
        layer_res = np.ceil(base_res).astype(np.int64)
        layer_num_grids = layer_res.prod()
        if layer_num_grids > hashmap_size and i >= min_dense_layers:
            layer_type = "Hash"
            layer_num_params = hashmap_size * n_feats
        else:
            layer_type = "Dense"
            layer_num_params = layer_num_grids * n_feats
        if num_params + layer_num_params > target_num_params:
            break
        layers_res.append(layer_res.tolist())
        layers_type.append(layer_type)
        num_params += layer_num_params
        base_res *= per_level_scale

    layers_n_feats = [n_feats] * len(layers_res)

    log.info(colored_str(f"NGP-4D auto-computed config: layer resolutions: {layers_res}", GREEN))
    log.info(colored_str(f"NGP-4D auto-computed config: layer types: {layers_type}", GREEN))
    log.info(colored_str(f"NGP-4D auto-computed config: layer n_feats: {layers_n_feats}", GREEN))
    log.info(colored_str(f"NGP-4D auto-computed config: totally {num_params} parameters " f"[{num_params/target_num_params:.2f}x]", GREEN))
    return dict(lod_res=layers_res, lod_n_feats=layers_n_feats, lod_types=layers_type,
                hashmap_size=hashmap_size)

def auto_lotd_cfg_deprecated(
    stretch: Union[float, List[float]], target_num_params: int, 
    *, dim: int = 3, min_res: int = 4, max_n_feat: int = 8, min_n_feat: int = 2, 
    level_scale: float = 2, ratio_last_nparam: float = 0.3, 
    ratio_dense_nparam: float = 0.2, ratio_dense_nlevel: float = 0.6) -> dict:
    """ Automatically computes LoTD configs for [Dense --> VM] configs

    Args:
        stretch (Union[float, List[float]]): 
            Spatial strech ratio or 3D/4D aspect ratios for cuboid / hyper-cuboid representations. 
            A single number would suggest using cubic representation.
        target_num_params (int): Target total number of params to generate.
        dim (int, optional): Space dimension. Defaults to 3.
        min_res (int, optional): Minimum resolution of the shortest side. Defaults to 4.
            TODO: Perhaps this should be changed to the resolution of the longest side, which might be more accurate to control? To be verified.
                But the previous configurations should also run correctly under these circumstances, so some LoTD configurations should be saved in the ckpt.
                Make this function a backup for the next run, and then start a new function.
            TODO: The resolution of smaller dense layers should be deduced from the last dense layer.
        max_n_feat (int, optional): Maximum feature width. Defaults to 8.
        min_n_feat (int, optional): Minimum feature width. Defaults to 2.
        level_scale (float, optional): Scale factor of the next level's resolution to current level's. Defaults to 2.
        ratio_last_nparam (float, optional): Proportion of n_params of the last VM level. Defaults to 0.3.
        ratio_dense_nparam (float, optional): Proportion of n_params of the dense levels. Defaults to 0.2.
        ratio_dense_nlevel (float, optional): Proportion of n_levels of the dense levels. Defaults to 0.6.

    Returns:
        dict: Automatically computed LoTD configs. Composed of lod_res, lod_n_feats, lod_types

    Algo:
        NOTE: For now, usually generates 0.7x of the requested `target_num_param`
        
        1. Assume the first layer LoDType == Dense, determine the smallest grid resolution and its number of parameters through min_res.
        2. Assume the last layer LoDType == VM, determine the required grid resolution for the last VM layer based on the set `ratio_last_nparam` (proportion of the number of parameters of the largest level).
        3. Determine the number of levels and the grid resolutions of each level based on the resolution of the last level (now only the lodtype and n_feat of each level need to be determined).
        4. Determine the number of layers of dense type and their configurations based on the set ratio_dense_nparam (proportion of the number of parameters of the dense layers).
        5. Place the remaining VM layers.
    """
    stretch = [stretch]*dim if isinstance(stretch, Number) else stretch
    stretch = np.array(stretch)
    
    plane_stretches = np.array([prod(stretch) / s for s in stretch])
    
    target = int(target_num_params * 1.5) # Usally generates 0.6x of requested `target`
    
    grid_size_lv0 = stretch.min() / min_res # stretch per grid
    res_lv0 = (stretch / grid_size_lv0).astype(int)
    
    #----------- Max level [LoDType==VM] consumes up to 40% of param pool
    # - The plane resolutions of grid: plane_stretches / grid_size_lvn^2
    # - To solve: grid_size_lvn
    # - Equation: sum(plane_stretches) / grid_size_lvn^2 * min_n_feat = 0.4 * target
    grid_size_lvn = np.sqrt(plane_stretches.sum() * min_n_feat / (ratio_last_nparam * target))
    
    up_factor = grid_size_lv0 / grid_size_lvn
    # NOTE: log_{ `level_scale` }^{ `up_factor` }
    #       Empirically, +0 will leads to only 0.4x~0.7x of num_params, +0.8 is just OK
    # num_levels = int(np.log(up_factor) / np.log(level_scale) + 0.8) + 1
    num_levels = max(int(np.log(up_factor) / np.log(level_scale)+0.5) + 1, 1)
    level_scale = up_factor ** (1./(num_levels-1))
    
    lotd_res_all = (res_lv0[..., None] * (level_scale ** np.arange(num_levels))).astype(int).T
    lotd_n_feats = [np.nan] * num_levels
    lotd_type_all = ["TODO"] * num_levels
    lotd_type_all[-1] = "VM"
    lotd_n_feats[-1] = min_n_feat

    accumu_num_param = 0
    accumu_lvl = 0

    #----------- [LoDType==Dense] consumes up to 20% of param pool
    for lvl in range(0, num_levels-1, 1):
        cur_res = lotd_res_all[lvl]
        
        cur_n_feat = (target*ratio_dense_nparam - accumu_num_param) / prod(cur_res) / 2
        cur_n_feat = np.clip(cur_n_feat, min_n_feat, max_n_feat)
        cur_n_feat = int( max(cur_n_feat // min_n_feat,1) * min_n_feat)
        
        cur_n_param = prod(cur_res) * cur_n_feat
        
        if (cur_n_param + accumu_num_param) > (ratio_dense_nparam * target) or \
            (lvl+1)/num_levels > ratio_dense_nlevel:
            break
        
        accumu_num_param += cur_n_param
        lotd_n_feats[lvl] = cur_n_feat
        lotd_type_all[lvl] = "Dense"
        accumu_lvl += 1
    
    
    #----------- [LoDType==VM]
    lvn_plane_size = [prod(lotd_res_all[-1])//r for r in lotd_res_all[-1]]
    lvn_n_param = sum(lvn_plane_size) * min_n_feat
    accumu_num_param += lvn_n_param
    
    for lvl in range(accumu_lvl, num_levels-1, 1):
        cur_res = lotd_res_all[lvl]
        
        plane_size = [prod(cur_res) // r for r in cur_res]
        
        cur_n_feat = (target - accumu_num_param) / sum(plane_size) / 2
        cur_n_feat = np.clip(cur_n_feat, min_n_feat, max_n_feat)
        cur_n_feat = int(max(cur_n_feat // min_n_feat,1) * min_n_feat)
        
        cur_n_param = sum(plane_size) * cur_n_feat
        
        if (cur_n_param + accumu_num_param) > target:
            break
        
        accumu_num_param += cur_n_param
        lotd_n_feats[lvl] = cur_n_feat
        lotd_type_all[lvl] = "VM"
        accumu_lvl += 1
    
    #----------- Fill all the remaining levels with VM (if any) 
    # Only happens when the above loop already exceeds the target num_params
    for lvl in range(accumu_lvl, num_levels-1, 1):
        cur_res = lotd_res_all[lvl]
        cur_n_feat = min_n_feat
        plane_size = [prod(cur_res) // r for r in cur_res]
        cur_n_param = sum(plane_size) * cur_n_feat
        
        accumu_num_param += cur_n_param
        lotd_n_feats[lvl] = cur_n_feat
        lotd_type_all[lvl] = "VM"

    print(colored_str(f"Expected num_params={target_num_params}; auto-computed LoTD config makes {accumu_num_param} [{accumu_num_param/target_num_params:.2f}x]", GREEN))
    return dict(lod_res=lotd_res_all.tolist(), lod_n_feats=lotd_n_feats, lod_types=lotd_type_all)

if __name__ == "__main__":
    
    def test_generate_lotd_cfg(device=torch.device('cuda')):
        # 32 MiB of float params
        from icecream import ic
        from nr3d_lib.models.grid_encodings.lotd import LoTD
        lotd_cfg = auto_lotd_cfg_deprecated([100., 20., 20.], dim=3, target_num_params=16*(2**20)//4, max_n_feat=8, min_res=16, min_n_feat=2)
        m = LoTD(3, **lotd_cfg, device=device, dtype=torch.float)
        print(m)
        
        lotd_cfg = auto_lotd_cfg_deprecated([20., 20., 20.], dim=3, target_num_params=16*(2**20)//4, max_n_feat=8, min_res=8, min_n_feat=2, ratio_dense_nparam=0.5)
        m = LoTD(3, **lotd_cfg, device=device, dtype=torch.float)
        print(m)

        lotd_cfg = auto_lotd_cfg_deprecated([20., 20., 20.], dim=3, target_num_params=16*(2**10)//4, max_n_feat=8, min_res=8, min_n_feat=2, ratio_dense_nparam=0.5)
        m = LoTD(3, **lotd_cfg, device=device, dtype=torch.float)
        print(m)

    def test_generate_ngp_cfg(device=torch.device('cuda')):
        from icecream import ic
        from nr3d_lib.models.grid_encodings.lotd import LoTD
        ngp_cfg = auto_ngp_cfg([100., 20., 20.], 16*(2**20)//4, dim=3, n_feats=2, log2_hashmap_size=18, min_res=16)
        m = LoTD(3, **ngp_cfg, device=device, dtype=torch.float)
        ic(m)

        ngp_cfg = auto_ngp_cfg([100., 20., 20.], 16*(2**20)//2, dim=3, n_feats=2, log2_hashmap_size=19, min_res=16)
        m = LoTD(3, **ngp_cfg, device=device, dtype=torch.float16)
        ic(m)

        ngp_cfg = auto_ngp_cfg([149.458, 92.742, 39.2999], 40*(2**20)//2, dim=3, n_feats=2, log2_hashmap_size=20, min_res=16)
        m = LoTD(3, **ngp_cfg, device=device, dtype=torch.float16)
        ic(m)
        
        ngp_cfg = auto_ngp4d_cfg(4, 2, [149.458, 92.742, 39.2999], 16*(2**20), min_res_xyz=16, min_res_w=4, log2_hashmap_size=20, per_level_scale=1.382)
        m = LoTD(4, **ngp_cfg, device=device, dtype=torch.float16)
        ic(m)
    
    def test_generate_cuboid_vs_cubic(device=torch.device('cuda')):
        from icecream import ic
        from nr3d_lib.models.grid_encodings.lotd import LoTD
        cfg1 = auto_ngp_cfg([20, 4, 1], 16*(2**20), dim=3, n_feats=2, log2_hashmap_size=20, min_res=16)
        cfg2 = auto_ngp_cfg(1, 16*(2**20), dim=3, n_feats=2, log2_hashmap_size=20, min_res=16)
        ic(len(cfg1['lod_n_feats']), len(cfg2['lod_n_feats']))
        print(np.array([200., 40., 10.]) / np.array(cfg1['lod_res'][-1]))
        print(np.array([200., 40., 10.]) / np.array(cfg2['lod_res'][-1]))
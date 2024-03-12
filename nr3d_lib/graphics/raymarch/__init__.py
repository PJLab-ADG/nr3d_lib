"""
@file   __init__.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Basic raymarch definitions
"""

import torch
from dataclasses import dataclass, fields

@dataclass
class RaymarchRetBase:
    num_hit_rays: int
    ridx_hit: torch.LongTensor          # [num_hit_ray_packs]       Unique idx for input rays, marking those actually hit
    samples: torch.FloatTensor          # [num_samples, 3]          Sampled coordinates
    depth_samples: torch.FloatTensor    # [num_samples]             Sampled ray depths
    deltas: torch.FloatTensor           # [num_samples]             Sampled ray depth deltas
    ridx: torch.LongTensor              # [num_samples]             Ray idx of each element
    pack_infos: torch.LongTensor        # [num_hit_ray_packs, 2]    Pack infos for `ridx`. (Indices of first element of hit ray pack & Number of samples each hit ray pack)
    def __iter__(self):
        return iter(tuple(getattr(self, field.name) for field in fields(self)))
    def __getitem__(self, name: str):
        return getattr(self, name)

@dataclass
class RaymarchRetSingle(RaymarchRetBase):
    # NOTE: The whole packed [num_samples, ...] elements can be viewed as:
    #       1. One pack for each hit ray
    #           ridx, pack_infos
    #       2. One pack for each grid voxel intersection
    #           gidx, gidx_pack_infos
    gidx: torch.LongTensor              # [num_samples]             Grid voxel idx of each element
    gidx_pack_infos: torch.LongTensor   # [num_grid_packs, 2]       Pack infos for `gidx`

@dataclass
class RaymarchRetBatched(RaymarchRetBase):
    # NOTE: The whole packed [num_samples, ...] elements can be viewed as:
    #       1. One pack for each hit ray
    #           ridx, pack_infos
    #       2. One pack for each grid voxel intersection
    #           gidx, gidx_pack_infos
    bidx: torch.LongTensor              # [num_samples]             Batch idx of each element
    gidx: torch.LongTensor              # [num_samples]             Grid voxel idx of each element
    gidx_pack_infos: torch.LongTensor   # [num_grid_packs, 2]       Pack infos for `gidx`

@dataclass
class RaymarchRetDynamic(RaymarchRetBase):
    # NOTE: The whole packed [num_samples, ...] elements can be viewed as:
    #       1. One pack for each hit ray
    #           ridx, pack_infos
    #       2. One pack for each grid voxel intersection
    #           gidx, gidx_pack_infos
    ts: torch.Tensor                    # [num_samples]             Timestamp of each element
    gidx: torch.LongTensor              # [num_samples]             Grid voxel idx of each element
    gidx_pack_infos: torch.LongTensor   # [num_grid_packs, 2]       Pack infos for `gidx`

@dataclass
class RaymarchRetBatchedDynamic(RaymarchRetBase):
    # NOTE: The whole packed [num_samples, ...] elements can be viewed as:
    #       1. One pack for each hit ray
    #           ridx, pack_infos
    #       2. One pack for each grid voxel intersection
    #           gidx, gidx_pack_infos
    bidx: torch.LongTensor              # [num_samples]             Batch idx of each element
    ts: torch.Tensor                    # [num_samples]             Timestamp of each element
    gidx: torch.LongTensor              # [num_samples]             Grid voxel idx of each element
    gidx_pack_infos: torch.LongTensor   # [num_grid_packs, 2]       Pack infos for `gidx`

@dataclass
class RaymarchRetForest(RaymarchRetBase):
    # NOTE: The whole packed [num_samples, ...] elements can be viewed as:
    #       1. One pack for each hit ray
    #           ridx, pack_infos
    #       2. One pack for each block intersection
    #           bidx, bidx_pack_infos
    #       3. One pack for each voxel grid intersection
    #           gidx, gidx_pack_infos
    blidx: torch.LongTensor             # [num_samples]             Block idx of each element
    blidx_pack_infos: torch.LongTensor  # [num_block_packs, 2]      Pack infos of `blidx`
    gidx: torch.LongTensor              # [num_samples]             Grid voxel idx of each element
    gidx_pack_infos: torch.LongTensor   # [num_grid_packs, 2]       Pack infos for `gidx`

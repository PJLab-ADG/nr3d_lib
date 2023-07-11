"""
@file   __init__.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Basic raymarch definitions
"""

import torch
from dataclasses import dataclass, fields

@dataclass
class dataclass_raymarch_ret:
    # NOTE: The whole packed [num_samples, ...] elements can be seen as results of two kinds of packing:
    #       1. One pack for each hit ray
    #           ridx_hit, pack_infos
    #       2. One pack for each continuous segment on one ray (there could be multiple continuous segments on one ray)
    #           con_pack_infos
    ridx_hit: torch.LongTensor          # [num_ray_packs]       Unique input ray inds that actually hit (with ray-trace to the octree)
    samples: torch.FloatTensor          # [num_samples, 3]      Sampled coordinates
    depth_samples: torch.FloatTensor    # [num_samples]         Sampled ray depths
    deltas: torch.FloatTensor           # [num_samples]         Sampled ray depth deltas
    ridx: torch.LongTensor              # [num_samples]         Ray inds of each element
    pack_infos: torch.LongTensor        # [num_ray_packs, 2]    Indices of first element of hit ray pack &  Number of samples each hit ray pack
    con_idx: torch.LongTensor           # [num_samples]         Continuous segments indices of each element
    con_pack_infos: torch.LongTensor    # [num_continuous_pack, 2]  Indices of first element of each continuous pack & Number of samples each continuous pack
    def __iter__(self):
        return iter(tuple(getattr(self, field.name) for field in fields(self)))

@dataclass
class dataclass_batched_raymarch_ret:
    # NOTE: The whole packed [num_samples, ...] elements can be seen as results of two kinds of packing:
    #       1. One pack for each hit ray
    #           ridx_hit, pack_infos
    #       2. One pack for each continuous segment on one ray (there could be multiple continuous segments on one ray)
    #           con_pack_infos
    ridx_hit: torch.LongTensor          # [num_ray_packs]       Unique input ray inds that actually hit (with ray-trace to the octree)
    samples: torch.FloatTensor          # [num_samples, 3]      Sampled coordinates
    depth_samples: torch.FloatTensor    # [num_samples]         Sampled ray depths
    deltas: torch.FloatTensor           # [num_samples]         Sampled ray depth deltas
    ridx: torch.LongTensor              # [num_samples]         Ray inds of each element
    bidx: torch.LongTensor              # [num_samples]         Batch inds of each element
    pack_infos: torch.LongTensor        # [num_ray_packs, 2]    Indices of first element of hit ray pack &  Number of samples each hit ray pack
    con_idx: torch.LongTensor           # [num_samples]         Continuous segments indices of each element
    con_pack_infos: torch.LongTensor    # [num_continuous_pack, 2]  Indices of first element of each continuous pack & Number of samples each continuous pack
    def __iter__(self):
        return iter(tuple(getattr(self, field.name) for field in fields(self)))

@dataclass
class dataclass_forest_raymarch_ret:
    # NOTE: The whole packed [num_samples, ...] elements can be seen as results of two kinds of packing:
    #       1. One pack for each hit ray
    #           ridx_hit, pack_infos
    #       2. One pack for each block intersection
    #           block_pack_infos
    ridx_hit: torch.LongTensor          # [num_ray_packs]       Unique input ray inds that actually hit (with ray-trace to the octree)
    samples: torch.FloatTensor          # [num_samples, 3]      Sampled coordinates
    depth_samples: torch.FloatTensor    # [num_samples]         Sampled ray depths
    deltas: torch.FloatTensor           # [num_samples]         Sampled ray depth deltas
    ridx: torch.LongTensor              # [num_samples]         Ray inds of each element
    pack_infos: torch.LongTensor        # [num_ray_packs, 2]    Pack infos of ray (ridx)
    bidx: torch.LongTensor              # [num_samples]         Block inds of each element
    block_pack_infos: torch.LongTensor  # [num_block_packs, 2]  Pack infos of block (bidx)
    def __iter__(self):
        return iter(tuple(getattr(self, field.name) for field in fields(self)))
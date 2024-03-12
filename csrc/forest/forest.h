/** @file   forest.h
 *  @author Jianfei Guo, Shanghai AI Lab
 *  @brief
 */

#pragma once

#include <stdint.h>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <vector>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>

#include "forest_cpp_api.h"

// #include "kaolin_spc_raytrace_fixed/spc_utils.cuh"
static __device__ __forceinline__ int32_t identify(
    const short3 k,
    const uint32_t       level,
    const int32_t*      prefix_sum,
    const uint8_t*     octree)
{
    // Modified from kaolin
    int maxval = (0x1 << level) - 1; // seems you could do this better using Morton codes
    // Check if in bounds
    if (k.x < 0 || k.y < 0 || k.z < 0 || k.x > maxval || k.y > maxval || k.z > maxval) {
        return -1;
    }
    int ord = 0;
    for (uint l = 0; l < level; l++)
    {
        uint depth = level - l - 1;
        uint mask = (0x1 << depth);
        uint child_idx = ((mask & k.x) << 2 | (mask & k.y) << 1 | (mask & k.z)) >> depth;
        uint8_t bits = octree[ord];
        // if bit set, keep going
        if (bits & (0x1 << child_idx))
        {
            // count set bits up to child - inclusive sum
            uint cnt = __popc(bits & ((0x2 << child_idx) - 1));
            ord = prefix_sum[ord] + cnt;
            if (depth == 0) {
                return ord;
            }
        }
        else {
            return -1;
        }
    }
    return ord; // only if called with Level=0
}

struct ForestMetaRef {
    uint8_t* octree;
    int32_t* exsum;
    // int16_t* block_ks;
    short3* block_ks;

    float3 world_block_size;
    float3 world_origin;
    int3 resolution;
    uint32_t n_trees;
    uint32_t level;
	uint32_t level_poffset=0;

    bool continuity_enabled=true;

    ForestMetaRef(ForestMeta meta): 
    level{meta.level}, level_poffset{meta.level_poffset}, n_trees{meta.n_trees}, continuity_enabled{meta.continuity_enabled} {
        world_block_size = make_float3((float)meta.world_block_size[0], (float)meta.world_block_size[1], (float)meta.world_block_size[2]);
        world_origin = make_float3((float)meta.world_origin[0], (float)meta.world_origin[1], (float)meta.world_origin[2]);
        resolution = make_int3(meta.resolution[0], meta.resolution[1], meta.resolution[2]);
        octree = meta.octree.data_ptr<uint8_t>();
        exsum = meta.exsum.data_ptr<int32_t>();
        block_ks = reinterpret_cast<short3*>(meta.block_ks.data_ptr<int16_t>());
    }

    __device__ int32_t map_block_ind(const int16_t idx[3]) const {
        short3 k = make_short3(idx[0], idx[1], idx[2]);
		return map_block_ind(k);
    }

    __device__ int32_t map_block_ind(const short3 k) const {
        int32_t pidx = identify(k, level, exsum, octree);
        // int32_t pidx = kaolin::identify(k, level, exsum, octree);
        return (pidx == -1) ? -1 : (pidx - (int32_t)level_poffset);
    }

};


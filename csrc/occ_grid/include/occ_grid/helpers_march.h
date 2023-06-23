/** @file   helpers_march.h
 *  @brief  Ray marching of occupancy grid
 *  Modified from https://github.com/KAIR-BAIR/nerfacc
 *  Copyright (c) 2022 Ruilong Li, UC Berkeley.
 */

#include "helpers_cuda.h"
#include "helpers_math.h"
#include "helpers_contraction.h"

inline __device__ __host__ float calc_dt(const float t, const float dt_gamma, const float dt_min, const float dt_max)
{
    return clamp(t * dt_gamma, dt_min, dt_max);
}

inline __device__ __host__ int grid_idx_at(const float3 xyz_unit, const int3 grid_res)
{
    // xyz should be always in [0, 1]^3.
    int3 ixyz = make_int3(xyz_unit * make_float3(grid_res));
    ixyz = clamp(ixyz, make_int3(0, 0, 0), grid_res - 1);
    int3 grid_offset = make_int3(grid_res.y * grid_res.z, grid_res.z, 1); // Contiguous memory on z-dim
    int idx = dot(ixyz, grid_offset);
    return idx;
}

template <typename scalar_t>
inline __device__ __host__ scalar_t grid_occupied_at(
    const float3 xyz,
    const float3 roi_min, const float3 roi_max,
    ContractionType type,
    const int3 grid_res, const scalar_t *grid_value, int* grid_idx_out)
{
    if (type == ContractionType::AABB &&
        (xyz.x < roi_min.x || xyz.x > roi_max.x ||
         xyz.y < roi_min.y || xyz.y > roi_max.y ||
         xyz.z < roi_min.z || xyz.z > roi_max.z))
    {
        return false;
    }
    float3 xyz_unit = apply_contraction(xyz, roi_min, roi_max, type);
    int idx = grid_idx_at(xyz_unit, grid_res);
    *grid_idx_out = idx;
    return grid_value[idx];
}

// dda like step
inline __device__ __host__ float distance_to_next_voxel(
    const float3 xyz, const float3 dir, const float3 inv_dir,
    const float3 roi_min, const float3 roi_max, const int3 grid_res)
{
    float3 _occ_res = make_float3(grid_res);
    float3 _xyz = roi_to_unit(xyz, roi_min, roi_max) * _occ_res;
    float3 txyz = ((floorf(_xyz + 0.5f + 0.5f * sign(dir)) - _xyz) * inv_dir) / _occ_res * (roi_max - roi_min);
    float t = min(min(txyz.x, txyz.y), txyz.z);
    return fmaxf(t, 0.0f);
}

inline __device__ __host__ float advance_to_next_voxel(
    const float t, const float dt_min,
    const float3 xyz, const float3 dir, const float3 inv_dir,
    const float3 roi_min, const float3 roi_max, const int3 grid_res)
{
    // Regular stepping (may be slower but matches non-empty space)
    float t_target = t + distance_to_next_voxel(xyz, dir, inv_dir, roi_min, roi_max, grid_res);
    float _t = t;
    do
    {
        _t += dt_min;
    } while (_t < t_target);
    
    // float _t = t;
    // if (_t < t_target) {
    //     _t += dt_min * ( (int)( (t_target-t) / dt_min ) + 1 );
    // }

    return _t;
}
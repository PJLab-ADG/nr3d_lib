/** @file   occ_grid.h
 *  @brief  
 *  Modified from https://github.com/KAIR-BAIR/nerfacc
 * Copyright (c) 2022 Ruilong Li, UC Berkeley.
 */

#pragma once

#include <stdint.h>
#include <torch/torch.h>

// #include <forest_cpp_api.h>

enum ContractionType
{
    AABB = 0,
    UN_BOUNDED_TANH = 1,
    UN_BOUNDED_SPHERE = 2,
};

std::vector<at::Tensor> ray_marching(
    const at::Tensor rays_o,
    const at::Tensor rays_d,
    const at::Tensor t_min,
    const at::Tensor t_max,
    const at::Tensor roi,
    const at::Tensor grid_binary,
    const ContractionType contraction_type,
    const float step_size,
    const float max_step_size, 
    const float dt_gamma, 
    const uint32_t max_steps,
    const bool return_gidx);

std::vector<at::Tensor> batched_ray_marching(
    const at::Tensor rays_o,
    const at::Tensor rays_d,
    const at::Tensor t_min,
    const at::Tensor t_max,
    const at::optional<at::Tensor> batch_inds_, 
    const at::optional<uint32_t> batch_data_size_, 
    const at::Tensor roi,
    const at::Tensor grid_binary,
    const ContractionType type,
    const float step_size,
    const float max_step_size, 
    const float dt_gamma, 
    const uint32_t max_steps,
    const bool return_gidx);

// std::vector<at::Tensor> forest_ray_marching(
//     const ForestMeta& forest,
//     const at::Tensor rays_o,
//     const at::Tensor rays_d,
//     const at::Tensor t_min, 
//     const at::Tensor t_max,
//     const at::Tensor seg_block_inds,
//     const at::Tensor seg_entries,
//     const at::Tensor seg_exits,
//     const at::Tensor seg_pack_infos,
//     const at::Tensor grid_binary,
//     const float step_size,
//     const float max_step_size, 
//     const float dt_gamma, 
//     const uint32_t max_steps,
//     const bool return_gidx);
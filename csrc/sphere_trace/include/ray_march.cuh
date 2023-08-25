#pragma once
#include "common.cuh"
#include "dense_grid.cuh"

std::tuple<at::Tensor, at::Tensor, at::Tensor, std::map<std::string, at::Tensor>>
ray_march(const DenseGrid &grid, at::Tensor rays_o, at::Tensor rays_d, at::Tensor rays_near,
          at::Tensor rays_far, bool enable_debug = false);
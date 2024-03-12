// Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//    http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <ATen/ATen.h>

#include <vector>

#include "kaolin_spc_raytrace_fixed/check.h"
#ifdef WITH_CUDA
#include "kaolin_spc_raytrace_fixed/utils.h"
#include "kaolin_spc_raytrace_fixed/spc_math.h"
#endif

namespace kaolin {

#define CHECK_TRIPLE(x) TORCH_CHECK(x.dim() == 1 && x.size(0) == 3, #x " must be a triplet")
#define CHECK_CPU_COORDS(x) CHECK_CONTIGUOUS(x); CHECK_CPU(x); CHECK_FLOAT(x); CHECK_TRIPLE(x)

using namespace std;
using namespace at::indexing;

#ifdef WITH_CUDA

std::vector<at::Tensor> raytrace_cuda_impl(
    at::Tensor octree,
    at::Tensor points,
    at::Tensor pyramid,
    at::Tensor exclusive_sum,
    at::Tensor ray_o,
    at::Tensor ray_d,
    uint32_t max_level,
    uint32_t target_level,
    bool return_depth,
    bool with_exit,
    bool include_head);

#endif

std::vector<at::Tensor> raytrace_cuda_fixed(
    at::Tensor octree,
    at::Tensor points,
    at::Tensor pyramid,
    at::Tensor exclusive_sum,
    at::Tensor ray_o,
    at::Tensor ray_d,
    uint32_t target_level,
    bool return_depth,
    bool with_exit,
    bool include_head) {
#ifdef WITH_CUDA
  at::TensorArg octree_arg{octree, "octree", 1};
  at::TensorArg points_arg{points, "points", 2};
  at::TensorArg pyramid_arg{pyramid, "pyramid", 3};
  at::TensorArg exclusive_sum_arg{exclusive_sum, "exclusive_sum", 4};
  at::TensorArg ray_o_arg{ray_o, "ray_o", 5};
  at::TensorArg ray_d_arg{ray_d, "ray_d", 6};
  at::checkAllSameGPU(__func__, {octree_arg, points_arg, exclusive_sum_arg, ray_o_arg, ray_d_arg});
  at::checkAllContiguous(__func__,  {octree_arg, points_arg, exclusive_sum_arg, ray_o_arg, ray_d_arg});
  at::checkDeviceType(__func__, {pyramid}, at::DeviceType::CPU);
  
  CHECK_SHORT(points);
  at::checkDim(__func__, points_arg, 2);
  at::checkSize(__func__, points_arg, 1, 3);
  at::checkDim(__func__, pyramid_arg, 2);
  at::checkSize(__func__, pyramid_arg, 0, 2);
  uint32_t max_level = pyramid.size(1)-2;
  TORCH_CHECK(max_level < KAOLIN_SPC_MAX_LEVELS, "SPC pyramid too big");

  uint32_t* pyramid_ptr = (uint32_t*)pyramid.data_ptr<int>();
  uint32_t osize = pyramid_ptr[2*max_level+2];
  uint32_t psize = pyramid_ptr[2*max_level+3];
  at::checkSize(__func__, octree_arg, 0, osize);
  at::checkSize(__func__, points_arg, 0, psize);
  TORCH_CHECK(pyramid_ptr[max_level+1] == 0 && pyramid_ptr[max_level+2] == 0, 
              "SPC pyramid corrupt, check if the SPC pyramid has been sliced");

  // do cuda
  return raytrace_cuda_impl(octree, points, pyramid, exclusive_sum, ray_o, ray_d, 
                                max_level, target_level, return_depth, with_exit, include_head);

#else
  KAOLIN_NO_CUDA_ERROR(__func__);
#endif  // WITH_CUDA
}

}  // namespace kaolin

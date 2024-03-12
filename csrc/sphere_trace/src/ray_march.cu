/** @file   ray_march.cu
 *  @author Nianchen Deng, Shanghai AI Lab
 *  @brief  The implementation of ray march along a dense occupancy grid.
 */
#include <sphere_trace/ray_march.cuh>
#include <ATen/cuda/CUDAContext.h>

using namespace at::indexing;

template <bool ENABLE_DEBUG>
__global__ void dense_grid_ray_march_phase1_kernel(uint32_t n_elements, uint32_t max_segs_per_ray,
                                                   DenseGrid grid,
                                                   const glm::vec3 *__restrict__ rays_o,
                                                   const glm::vec3 *__restrict__ rays_d,
                                                   const float *__restrict__ rays_near,
                                                   const float *__restrict__ rays_far,
                                                   int32_t *__restrict__ o_rays_n_segs, //
                                                   int *o_debug_rays_flag,              //
                                                   glm::vec3 *o_debug_rays_march_poses, //
                                                   glm::ivec3 *o_debug_rays_march_voxels) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements)
        return;
    glm::vec3 space_scale = 0.5f * glm::vec3(grid.res());
    glm::vec3 origin = (rays_o[i] + 1.0f) * space_scale;
    glm::vec3 dir = rays_d[i] * space_scale;
    auto new_seg_callback = [](glm::vec2 seg) {};
    o_rays_n_segs[i] = grid.ray_march<ENABLE_DEBUG>(
        origin, dir, rays_near[i], rays_far[i], nullptr, new_seg_callback, max_segs_per_ray,
        o_debug_rays_flag + i, o_debug_rays_march_poses + i * max_segs_per_ray,
        o_debug_rays_march_voxels + i * max_segs_per_ray);
}

__global__ void dense_grid_ray_march_phase2_kernel(
    uint32_t n_elements, DenseGrid grid, const glm::vec3 *__restrict__ rays_o,
    const glm::vec3 *__restrict__ rays_d, const float *__restrict__ rays_near,
    const float *__restrict__ rays_far, const int64_t *__restrict__ rays_idx,
    const int32_t *__restrict__ rays_segs_offset, glm::vec2 *__restrict__ o_rays_segs,
    glm::vec3 *__restrict__ o_rays_segs_endpoints) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements)
        return;
    int32_t ray_idx = rays_idx[i];
    glm::vec3 space_scale = 0.5f * glm::vec3(grid.res());
    glm::vec3 origin = rays_o[ray_idx];
    glm::vec3 dir = rays_d[ray_idx];
    glm::vec3 origin_grid = (origin + 1.0f) * space_scale;
    glm::vec3 dir_grid = dir * space_scale;
    if (o_rays_segs_endpoints) {
        auto new_seg_callback = [&](glm::vec2 seg) {
            *(o_rays_segs_endpoints++) = origin + dir * seg.x;
            *(o_rays_segs_endpoints++) = origin + dir * seg.y;
        };
        o_rays_segs_endpoints += rays_segs_offset[i] * 2;
        grid.ray_march<false>(origin_grid, dir_grid, rays_near[ray_idx], rays_far[ray_idx],
                              o_rays_segs + rays_segs_offset[i], new_seg_callback);
    } else {
        auto new_seg_callback = [](glm::vec2 seg) {};
        grid.ray_march<false>(origin_grid, dir_grid, rays_near[ray_idx], rays_far[ray_idx],
                              o_rays_segs + rays_segs_offset[i], new_seg_callback);
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, std::map<std::string, at::Tensor>>
ray_march(const DenseGrid &grid, at::Tensor rays_o, at::Tensor rays_d, at::Tensor rays_near,
          at::Tensor rays_far, bool return_pts, bool enable_debug) {
    int64_t n_rays = rays_o.size(0);
    uint32_t max_segs_per_ray = std::max(grid.res().x, std::max(grid.res().y, grid.res().z)) * 3;
    at::Tensor rays_n_segs = at::empty({n_rays}, at::dtype(at::kInt).device(rays_o.device()));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Phase 1: compute number of segments for each ray
    std::map<std::string, at::Tensor> debug_info;
    if (enable_debug) {
        at::Tensor debug_rays_flag =
            at::empty({n_rays}, at::dtype(at::kInt).device(rays_o.device()));
        at::Tensor debug_rays_march_poses =
            at::zeros({n_rays, max_segs_per_ray, 3}, at::dtype(at::kFloat).device(rays_o.device()));
        at::Tensor debug_rays_march_voxels =
            at::zeros({n_rays, max_segs_per_ray, 3}, at::dtype(at::kInt).device(rays_o.device()));
        linear_kernel(dense_grid_ray_march_phase1_kernel<false>, 0, stream, n_rays,
                      max_segs_per_ray, grid, (glm::vec3 *)rays_o.data_ptr(),
                      (glm::vec3 *)rays_d.data_ptr(), rays_near.data_ptr<float>(),
                      rays_far.data_ptr<float>(), rays_n_segs.data_ptr<int32_t>(),
                      debug_rays_flag.data_ptr<int>(),
                      (glm::vec3 *)debug_rays_march_poses.data_ptr(),
                      (glm::ivec3 *)debug_rays_march_voxels.data_ptr());
        debug_info = {{"flag", debug_rays_flag},
                      {"march_poses", debug_rays_march_poses},
                      {"march_voxels", debug_rays_march_voxels}};
    } else {
        linear_kernel(dense_grid_ray_march_phase1_kernel<false>, 0, stream, n_rays,
                      max_segs_per_ray, grid, (glm::vec3 *)rays_o.data_ptr(),
                      (glm::vec3 *)rays_d.data_ptr(), rays_near.data_ptr<float>(),
                      rays_far.data_ptr<float>(), rays_n_segs.data_ptr<int32_t>(), nullptr, nullptr,
                      nullptr);
    }

    // Filter out rays with no intersection
    at::Tensor candidate_rays_idx = rays_n_segs.nonzero().index({Slice(), 0});
    rays_n_segs = rays_n_segs.index({candidate_rays_idx});
    n_rays = rays_n_segs.size(0);

    // Compute pack info
    at::Tensor rays_n_segs_cumsum = rays_n_segs.cumsum(0, at::kInt);
    at::Tensor rays_segs_offset = rays_n_segs_cumsum - rays_n_segs;
    at::Tensor pack_info = at::stack({rays_segs_offset, rays_n_segs}, -1);

    // Allocate rays segments buffer
    int32_t total_segs =
        rays_n_segs_cumsum.numel() > 0 ? rays_n_segs_cumsum[-1].item<int32_t>() : 0;
    at::Tensor rays_segs = rays_o.new_empty({total_segs, 2});

    at::Tensor rays_segs_endpoints;
    if (return_pts)
        rays_segs_endpoints = rays_o.new_empty({total_segs, 2, 3});

    // Phase 2: compute ray segments
    if (total_segs)
        linear_kernel(dense_grid_ray_march_phase2_kernel, 0, stream, n_rays, grid,
                      (glm::vec3 *)rays_o.data_ptr(), (glm::vec3 *)rays_d.data_ptr(),
                      rays_near.data_ptr<float>(), rays_far.data_ptr<float>(),
                      candidate_rays_idx.data_ptr<int64_t>(), rays_segs_offset.data_ptr<int32_t>(),
                      (glm::vec2 *)rays_segs.data_ptr(),
                      return_pts ? (glm::vec3 *)rays_segs_endpoints.data_ptr() : nullptr);

    return std::make_tuple(candidate_rays_idx, pack_info, rays_segs, rays_segs_endpoints,
                           debug_info);
}
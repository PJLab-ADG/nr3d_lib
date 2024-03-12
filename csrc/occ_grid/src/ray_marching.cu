/** @file   ray_marching.cu
 *  @brief  Ray marching of occupancy grid
 *  Modified from https://github.com/KAIR-BAIR/nerfacc
 *  Copyright (c) 2022 Ruilong Li, UC Berkeley.
 */

#include <occ_grid/helpers_cuda.h>
#include <occ_grid/helpers_math.h>
#include <occ_grid/helpers_contraction.h>
#include <occ_grid/helpers_march.h>
#include <occ_grid/cpp_api.h>

// -------------------------------------------------------------------------------
// Raymarching
// -------------------------------------------------------------------------------

__global__ void ray_marching_kernel(
    // rays info
    const uint32_t n_rays,
    const float* __restrict__ rays_o, // shape (n_rays, 3)
    const float* __restrict__ rays_d, // shape (n_rays, 3)
    const float* __restrict__ t_min,  // shape (n_rays,)
    const float* __restrict__ t_max,  // shape (n_rays,)
    // occupancy grid & contraction
    const float* __restrict__ roi,
    const int3 grid_res,
    const bool* __restrict__ grid_binary, // shape (reso_x, reso_y, reso_z)
    const ContractionType type,
    // sampling
    const float step_size,
    const float max_step_size, 
    const float dt_gamma,
    uint32_t max_steps,
    const int32_t* __restrict__ packed_info,
    // first round outputs
    int32_t* __restrict__ num_steps,
    // second round outputs
    float* __restrict__ t_starts,
    float* __restrict__ t_ends, 
    int32_t* __restrict__ ridx_out, 
    int32_t* __restrict__ gidx_out
) {
    CUDA_GET_THREAD_ID(i, n_rays);

    bool is_first_round = (packed_info == nullptr);

    // locate
    rays_o += i * 3;
    rays_d += i * 3;
    t_min += i;
    t_max += i;

    if (is_first_round)
    {
        num_steps += i;
    }
    else
    {
        uint32_t base = packed_info[i * 2 + 0];
        max_steps = packed_info[i * 2 + 1];
        t_starts += base;
        t_ends += base;
        ridx_out += base;
        if (gidx_out) gidx_out += base;
    }

    const float3 origin = make_float3(rays_o[0], rays_o[1], rays_o[2]);
    const float3 dir = make_float3(rays_d[0], rays_d[1], rays_d[2]);
    const float3 inv_dir = 1.0f / dir;
    const float near = t_min[0], far = t_max[0];

    const float3 roi_min = make_float3(roi[0], roi[1], roi[2]);
    const float3 roi_max = make_float3(roi[3], roi[4], roi[5]);

    // TODO: compute dt_max from occ resolution.
    float dt_min = step_size;
    float dt_max = max_step_size;

    uint32_t j = 0;
    float t0 = near;
    float dt = calc_dt(t0, dt_gamma, dt_min, dt_max);
    float t1 = t0 + dt;
    float t_mid = (t0 + t1) * 0.5f;

    while ((t_mid < far) && (j < max_steps))
    {
        // current center 
        const float3 xyz = origin + t_mid * dir;
        int32_t grid_idx = -1;
        if (grid_occupied_at(xyz, roi_min, roi_max, type, grid_res, grid_binary, &grid_idx))
        {
            if (!is_first_round)
            {
                t_starts[j] = t0;
                t_ends[j] = t1;
                ridx_out[j] = i;
                if (gidx_out) gidx_out[j] = grid_idx;
            }
            ++j;
            // march to next sample
            t0 = t1;
            t1 = t0 + calc_dt(t0, dt_gamma, dt_min, dt_max);
            t_mid = (t0 + t1) * 0.5f;
        }
        else
        {
            // march to next sample
            switch (type)
            {
            case ContractionType::AABB:
                // no contraction
                t_mid = advance_to_next_voxel(
                    t_mid, dt_min, xyz, dir, inv_dir, roi_min, roi_max, grid_res);
                dt = calc_dt(t_mid, dt_gamma, dt_min, dt_max);
                t0 = t_mid - dt * 0.5f;
                t1 = t_mid + dt * 0.5f;
                break;

            default:
                // any type of scene contraction does not work with DDA.
                t0 = t1;
                t1 = t0 + calc_dt(t0, dt_gamma, dt_min, dt_max);
                t_mid = (t0 + t1) * 0.5f;
                break;
            }
        }
    }

    if (is_first_round)
    {
        *num_steps = j;
    }
    return;
}

std::vector<at::Tensor> ray_marching(
    // rays
    const at::Tensor rays_o,
    const at::Tensor rays_d,
    const at::Tensor t_min,
    const at::Tensor t_max,
    // occupancy grid & contraction
    const at::Tensor roi,
    const at::Tensor grid_binary,
    const ContractionType type,
    // sampling
    const float step_size,
    const float max_step_size, 
    const float dt_gamma, 
    const uint32_t max_steps,
    const bool return_gidx)
{
    DEVICE_GUARD(rays_o);

    CHECK_INPUT(rays_o);
    CHECK_INPUT(rays_d);
    CHECK_INPUT(t_min);
    CHECK_INPUT(t_max);
    CHECK_INPUT(roi);
    CHECK_INPUT(grid_binary);
    TORCH_CHECK(rays_o.ndimension() == 2 & rays_o.size(1) == 3)
    TORCH_CHECK(rays_d.ndimension() == 2 & rays_d.size(1) == 3)
    TORCH_CHECK(t_min.ndimension() == 1)
    TORCH_CHECK(t_max.ndimension() == 1)
    TORCH_CHECK(roi.ndimension() == 1 & roi.size(0) == 6)
    TORCH_CHECK(grid_binary.ndimension() == 3)

    const uint32_t n_rays = rays_o.size(0);
    const int3 grid_res = make_int3(
        grid_binary.size(0), grid_binary.size(1), grid_binary.size(2));

    const uint32_t threads = 256;
    const uint32_t blocks = CUDA_N_BLOCKS_NEEDED(n_rays, threads);

    // helper counter
    at::Tensor num_steps = at::empty({n_rays}, rays_o.options().dtype(at::kInt));

    // count number of samples per ray
    ray_marching_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        // rays
        n_rays,
        rays_o.data_ptr<float>(),
        rays_d.data_ptr<float>(),
        t_min.data_ptr<float>(),
        t_max.data_ptr<float>(),
        // occupancy grid & contraction
        roi.data_ptr<float>(),
        grid_res,
        grid_binary.data_ptr<bool>(),
        type,
        // sampling
        step_size,
        max_step_size, 
        dt_gamma,
        max_steps,
        nullptr, /* packed_info */
        // outputs
        num_steps.data_ptr<int32_t>(),
        nullptr, /* t_starts */
        nullptr, /* t_ends */
        nullptr, /* ridx_out */
        nullptr  /* gidx_out */
    );

    at::Tensor cum_steps = num_steps.cumsum(0, at::kInt);
    at::Tensor packed_info = at::stack({cum_steps - num_steps, num_steps}, 1);

    // output samples starts and ends
    uint32_t total_steps = cum_steps[cum_steps.size(0) - 1].item<int32_t>();
    at::Tensor t_starts = at::empty({total_steps, 1}, rays_o.options());
    at::Tensor t_ends = at::empty({total_steps, 1}, rays_o.options());
    at::Tensor ridx_out = at::empty({total_steps}, rays_o.options().dtype(at::kInt));
    at::Tensor gidx_out;
    if (return_gidx) {
        gidx_out = at::empty({total_steps}, rays_o.options().dtype(at::kInt));
    }

    ray_marching_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        // rays
        n_rays,
        rays_o.data_ptr<float>(),
        rays_d.data_ptr<float>(),
        t_min.data_ptr<float>(),
        t_max.data_ptr<float>(),
        // occupancy grid & contraction
        roi.data_ptr<float>(),
        grid_res,
        grid_binary.data_ptr<bool>(),
        type,
        // sampling
        step_size,
        max_step_size, 
        dt_gamma,
        max_steps, 
        packed_info.data_ptr<int32_t>(),
        // outputs
        nullptr, /* num_steps */
        t_starts.data_ptr<float>(),
        t_ends.data_ptr<float>(), 
        ridx_out.data_ptr<int32_t>(), 
        return_gidx ? gidx_out.data_ptr<int32_t>() : nullptr);

    return {packed_info, t_starts, t_ends, ridx_out, gidx_out};
}
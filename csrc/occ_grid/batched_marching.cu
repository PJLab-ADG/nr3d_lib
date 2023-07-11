/** @file   batched_marching.cu
 *  @author Jianfei Guo, Shanghai AI Lab
 *  @brief  Ray marching of a batch of occupancy grids
 *  Modified from https://github.com/KAIR-BAIR/nerfacc
 */

#include <occ_grid/helpers_cuda.h>
#include <occ_grid/helpers_math.h>
#include <occ_grid/helpers_contraction.h>
#include <occ_grid/helpers_march.h>
#include <occ_grid/cpp_api.h>

template<typename T>
inline bool is_divisible(const T number, const T n) {
	return ((number - (number / n) * n) == (T)0);
}

__global__ void batched_ray_marching_kernel(
    // Rays info
    const uint32_t n_rays, 
    const float* __restrict__ rays_o, // Flattened [n_rays, 3]
    const float* __restrict__ rays_d, // Flattened [n_rays, 3]
    const float* __restrict__ t_min, // Flattened [n_rays, 3]
    const float* __restrict__ t_max, // Flattened [n_rays, 3]
    const uint32_t batch_data_size, 
    const int32_t* __restrict__ batch_inds, // Optional; flattened [n_rays, 3]
    // Occupancy grid & contraction
    const float *roi,
    const int3 grid_res, // [reso_x, reso_y, reso_z]
    const bool* __restrict__ grid_binary, // shape [B, reso_x, reso_y, reso_z], 
    const ContractionType type,
    // Sampling
    const float step_size,
    const float max_step_size, 
    const float dt_gamma,
    uint32_t max_steps,
    const int32_t* __restrict__ packed_info,
    // First round outputs
    int32_t* __restrict__ num_steps,
    // Second round outputs
    float* t_starts,
    float* t_ends, 
    int32_t* __restrict__ ridx_out, 
    int32_t* __restrict__ bidx_out, 
    int32_t* __restrict__ gidx_out=nullptr
) {
    CUDA_GET_THREAD_ID(i, n_rays);

    bool is_first_round = (packed_info == nullptr);

	// For batched
	uint32_t batch_ind = 0;
	if (batch_inds) {
		// NOTE: pass in batch_ind=-1 to ignore certain points.
		if (batch_inds[i] < 0) { return;}
		batch_ind = batch_inds[i];
	} else if (batch_data_size) {
		batch_ind = i / batch_data_size;
	}

    // Locate
    uint32_t grid_offset = batch_ind * (grid_res.x * grid_res.y * grid_res.z);
    grid_binary += grid_offset;
    roi += batch_ind * 6;
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
        bidx_out += base;
        if (gidx_out) gidx_out += base;
    }

    const float3 origin = make_float3(rays_o[0], rays_o[1], rays_o[2]);
    const float3 dir = make_float3(rays_d[0], rays_d[1], rays_d[2]);
    const float3 inv_dir = 1.0f / dir;
    const float near = t_min[0], far = t_max[0];

    const float3 roi_min = make_float3(roi[0], roi[1], roi[2]);
    const float3 roi_max = make_float3(roi[3], roi[4], roi[5]);

    float dt_min = step_size;
    float dt_max = max_step_size;

    uint32_t j = 0;
    float t0 = near;
    float dt = calc_dt(t0, dt_gamma, dt_min, dt_max);
    float t1 = t0 + dt;
    float t_mid = (t0 + t1) * 0.5f;

    while ((t_mid < far) && (j < max_steps))
    {
        // Current center
        const float3 xyz = origin + t_mid * dir;
        int32_t grid_idx = -1;
        if (grid_occupied_at(xyz, roi_min, roi_max, type, grid_res, grid_binary, &grid_idx))
        {
            if (!is_first_round)
            {
                t_starts[j] = t0;
                t_ends[j] = t1;
                ridx_out[j] = i;
                bidx_out[j] = batch_ind;
                if (gidx_out) gidx_out[j] = grid_idx + grid_offset;
            }
            ++j;
            // March to next sample
            t0 = t1;
            t1 = t0 + calc_dt(t0, dt_gamma, dt_min, dt_max);
            t_mid = (t0 + t1) * 0.5f;
        }
        else
        {
            // March to next sample
            switch (type)
            {
            case ContractionType::AABB:
                // No contraction
                t_mid = advance_to_next_voxel(
                    t_mid, dt_min, xyz, dir, inv_dir, roi_min, roi_max, grid_res);
                dt = calc_dt(t_mid, dt_gamma, dt_min, dt_max);
                t0 = t_mid - dt * 0.5f;
                t1 = t_mid + dt * 0.5f;
                break;

            default:
                // Any type of scene contraction does not work with DDA.
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

std::vector<at::Tensor> batched_ray_marching(
    // Rays
    const at::Tensor rays_o,
    const at::Tensor rays_d,
    const at::Tensor t_min,
    const at::Tensor t_max,
    const at::optional<at::Tensor> batch_inds_, 
    const at::optional<uint32_t> batch_data_size_, 
    // Occupancy grid & contraction
    const at::Tensor roi,
    const at::Tensor grid_binary,
    const ContractionType type,
    // Sampling
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
    TORCH_CHECK(rays_o.ndimension() == 2 && rays_o.size(1) == 3)
    TORCH_CHECK(rays_d.ndimension() == 2 && rays_d.size(1) == 3)
    TORCH_CHECK(t_min.ndimension() == 1)
    TORCH_CHECK(t_max.ndimension() == 1)

    TORCH_CHECK(roi.ndimension() == 2 && roi.size(1) == 6)
    TORCH_CHECK(grid_binary.ndimension() == 4)
    TORCH_CHECK(grid_binary.size(0) == roi.size(0)) // Batch dim

    const uint32_t n_rays = rays_o.size(0);
    const int3 grid_res = make_int3(grid_binary.size(1), grid_binary.size(2), grid_binary.size(3));

    const uint32_t threads = 256;
    const uint32_t blocks = CUDA_N_BLOCKS_NEEDED(n_rays, threads);

	at::Tensor batch_inds;
	if (batch_inds_.has_value()) {
		batch_inds = batch_inds_.value();
        CHECK_INPUT(batch_inds);
        TORCH_CHECK(batch_inds.ndimension() == 1 && batch_inds.size(0) == n_rays)
	}

	uint32_t batch_data_size = 0;
	if (batch_data_size_.has_value()) {
		batch_data_size = batch_data_size_.value();
		if (! (batch_data_size == 0 || is_divisible(n_rays, batch_data_size))) {
			throw std::runtime_error("batched_ray_marching: Expect nonzero `batch_data_size`=" + std::to_string(batch_data_size) + " to be a divisor of `n_rays`=" + std::to_string(n_rays));
		}
	}

    // Helper counter
    at::Tensor num_steps = at::empty({n_rays}, rays_o.options().dtype(at::kInt));

    // Count number of samples per ray
    batched_ray_marching_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        // Rays
        n_rays,
        rays_o.data_ptr<float>(),
        rays_d.data_ptr<float>(),
        t_min.data_ptr<float>(),
        t_max.data_ptr<float>(), 
        batch_data_size, 
        batch_inds_.has_value() ? batch_inds_.value().data_ptr<int32_t>() : nullptr, 
        // Occupancy grid & contraction
        roi.data_ptr<float>(),
        grid_res,
        grid_binary.data_ptr<bool>(),
        type,
        // Sampling
        step_size,
        max_step_size, 
        dt_gamma,
        max_steps, 
        nullptr, /* packed_info */
        // Outputs
        num_steps.data_ptr<int32_t>(),
        nullptr, /* t_starts */
        nullptr, /* t_ends */
        nullptr, /* ridx_out */
        nullptr, /* bidx_out */
        nullptr  /* gidx_out */
    );

    at::Tensor cum_steps = num_steps.cumsum(0, at::kInt);
    at::Tensor packed_info = at::stack({cum_steps - num_steps, num_steps}, 1);

    // Output samples starts and ends
    uint32_t total_steps = cum_steps[cum_steps.size(0) - 1].item<int32_t>();
    at::Tensor t_starts = at::empty({total_steps, 1}, rays_o.options());
    at::Tensor t_ends = at::empty({total_steps, 1}, rays_o.options());
    at::Tensor ridx_out = at::empty({total_steps}, rays_o.options().dtype(at::kInt));
    at::Tensor bidx_out = at::empty({total_steps}, rays_o.options().dtype(at::kInt));
    at::Tensor gidx_out;
    if (return_gidx) {
        gidx_out = at::empty({total_steps}, rays_o.options().dtype(at::kInt));
    }

    batched_ray_marching_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        // Rays
        n_rays,
        rays_o.data_ptr<float>(),
        rays_d.data_ptr<float>(),
        t_min.data_ptr<float>(),
        t_max.data_ptr<float>(),
        batch_data_size, 
        batch_inds_.has_value() ? batch_inds_.value().data_ptr<int32_t>() : nullptr, 
        // Occupancy grid & contraction
        roi.data_ptr<float>(),
        grid_res,
        grid_binary.data_ptr<bool>(),
        type,
        // Sampling
        step_size,
        max_step_size, 
        dt_gamma,
        max_steps, 
        packed_info.data_ptr<int32_t>(),
        // Outputs
        nullptr, /* num_steps */
        t_starts.data_ptr<float>(),
        t_ends.data_ptr<float>(), 
        ridx_out.data_ptr<int32_t>(), 
        bidx_out.data_ptr<int32_t>(), 
        return_gidx ? gidx_out.data_ptr<int32_t>() : nullptr
    );

    return {packed_info, t_starts, t_ends, ridx_out, bidx_out, gidx_out};
}
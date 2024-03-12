/** @file   forest_marching.cu
 *  @author Jianfei Guo, Shanghai AI Lab
 *  @brief  Ray marching of a forest of occupancy grid
 *  Modified from https://github.com/KAIR-BAIR/nerfacc
 */

#include <occ_grid/helpers_cuda.h>
#include <occ_grid/helpers_math.h>
#include <occ_grid/helpers_contraction.h>
#include <occ_grid/helpers_march.h>
#include <occ_grid/cpp_api.h>

#include <forest.h>
#include <forest_cpp_api.h>

inline __device__ bool block_grid_occupied_at(
    const float3 xyz,
    const float3 roi_min, const float3 roi_max,
    const int3 grid_res, const bool *grid_value, int32_t* grid_idx_out)
{
    float3 xyz_unit = roi_to_unit(xyz, roi_min, roi_max);
    int32_t idx = grid_idx_at(xyz_unit, grid_res);
    *grid_idx_out = idx;
    return grid_value[idx];
}

__global__ void forest_marching_kernel(
    // Forest info
    const ForestMetaRef forest, 
    // Rays info
    const uint32_t n_rays,
    const float* __restrict__ rays_o, // [n_rays, 3]
    const float* __restrict__ rays_d, // [n_rays, 3]
    const float* __restrict__ t_min,  // [n_rays,]
    const float* __restrict__ t_max,  // [n_rays,]
    // Block segments info
    const int32_t* __restrict__ seg_block_inds, // [n_segments]
    const float* __restrict__ seg_entries,      // [n_segments]
    const float* __restrict__ seg_exits,        // [n_segments]
    const int32_t* __restrict__ seg_pack_infos, // [n_rays, 2] Block-segments pack info
    // Occupancy grid
    const int3 grid_res, // [reso_x, reso_y, reso_z]
    const bool* __restrict__ grid_binary, // [B, reso_x, reso_y, reso_z], 
    // Sampling
    const float step_size,
    const float max_step_size,
    const float dt_gamma,
    uint32_t max_steps,
    const int32_t* __restrict__ pack_infos, // [n_rays, 2]
    // First round outputs
    int32_t* num_steps,
    // Second round outputs
    float* __restrict__ t_starts,
    float* __restrict__ t_ends, 
    int32_t* __restrict__ ridx_out, 
    int32_t* __restrict__ blidx_out,
    int32_t* __restrict__ gidx_out
) {
    CUDA_GET_THREAD_ID(i, n_rays);

    bool is_first_round = (pack_infos == nullptr);

    // Locate
    const uint32_t seg_begin = seg_pack_infos[i * 2], seg_length = seg_pack_infos[i * 2 + 1];
    const uint32_t grid_size = grid_res.x * grid_res.y * grid_res.z;
    rays_o += i * 3;
    rays_d += i * 3;
    t_min += i;
    t_max += i;
    seg_block_inds += seg_begin;
    seg_entries += seg_begin;
    seg_exits += seg_begin;

    if (is_first_round)
    {
        num_steps += i;
    }
    else
    {
        uint32_t base = pack_infos[i * 2 + 0];
        max_steps = pack_infos[i * 2 + 1];
        t_starts += base;
        t_ends += base;
        ridx_out += base;
        blidx_out += base;
        if (gidx_out) gidx_out += base;
    }

    const float3 origin = make_float3(rays_o[0], rays_o[1], rays_o[2]);
    const float3 dir = make_float3(rays_d[0], rays_d[1], rays_d[2]);
    const float3 inv_dir = 1.0f / dir;
    const float near = t_min[0], far = t_max[0];

    const float dt_min = step_size;
    const float dt_max = max_step_size;

    // // NOTE: roi of the whole forest
    // const float3 roi_min = forest.world_origin;
    // const float3 roi_max = forest.world_origin + forest.world_block_size * (1<<forest.level);

    uint32_t j = 0; // j: current step
    float t0 = near;
    float dt = calc_dt(t0, dt_gamma, dt_min, dt_max);
    float t1 = t0 + dt;
    float t_mid = (t0 + t1) * 0.5f;

    for (uint32_t s=0; s < seg_length; ++s) {
        const float cur_entry = seg_entries[s], cur_exit = seg_exits[s];
        const uint32_t block_ind = seg_block_inds[s];
        const short3 k = forest.block_ks[block_ind];
        const float3 local_roi_min = forest.world_origin + make_float3(k.x, k.y, k.z) * forest.world_block_size;
        const float3 local_roi_max = local_roi_min + forest.world_block_size;
        const uint32_t grid_offset = block_ind * grid_size;
        const bool* local_grid_binary = grid_binary + grid_offset;

        if (cur_entry >= far || cur_exit <= near) break;
        
        // March to current block-segment entry
        /*--- v1 ---*/
        do {
            t_mid += step_size;
        } while(t_mid < cur_entry);
        dt = calc_dt(t_mid, dt_gamma, dt_min, dt_max);
        t0 = t_mid - dt * 0.5f;
        t1 = t_mid + dt * 0.5f;

        // March all steps in current block-segment
        while ((t_mid <= cur_exit) && (t_mid <= far) && (j < max_steps)) {
            // Current center
            const float3 xyz = origin + t_mid * dir;
            int32_t grid_idx = -1;

            if (block_grid_occupied_at(xyz, local_roi_min, local_roi_max, grid_res, local_grid_binary, &grid_idx)) {
                if (!is_first_round) {
                    t_starts[j] = t0;
                    t_ends[j] = t1;
                    ridx_out[j] = i;
                    blidx_out[j] = block_ind;
                    if (gidx_out) gidx_out[j] = grid_idx + grid_offset;
                }
                ++j;
                // March to next sample
                t0 = t1;
                t1 = t0 + calc_dt(t0, dt_gamma, dt_min, dt_max);
                t_mid = (t0 + t1) * 0.5f;
            } else {
                t_mid = advance_to_next_voxel(t_mid, dt_min, xyz, dir, inv_dir, local_roi_min, local_roi_max, grid_res);
                dt = calc_dt(t_mid, dt_gamma, dt_min, dt_max);
                t0 = t_mid - dt * 0.5f;
                t1 = t_mid + dt * 0.5f;
            }
        }
    }

    if (is_first_round) {
        *num_steps = j;
    }
}

std::vector<at::Tensor> forest_ray_marching(
    // Forest
    const ForestMeta& forest,
    // Rays
    const at::Tensor rays_o, // [n_rays, 3]
    const at::Tensor rays_d, // [n_rays, 3]
    const at::Tensor t_min,  // [n_rays,]
    const at::Tensor t_max,  // [n_rays,]
    // Block segments info
    const at::Tensor seg_block_inds,// [n_segments]
    const at::Tensor seg_entries,   // [n_segments]
    const at::Tensor seg_exits,     // [n_segments]
    const at::Tensor seg_pack_infos,// [n_rays, 2]
    // Occupancy grid
    const at::Tensor grid_binary, // [B, reso_x, reso_y, reso_z]
    // Sampling
    const float step_size,
    const float max_step_size, 
    const float dt_gamma, 
    const uint32_t max_steps,
    const bool return_gidx
) {
    const at::cuda::OptionalCUDAGuard device_guard(at::device_of(rays_o));

	at::TensorArg rays_o_arg(rays_o, "rays_o", 2);
	at::TensorArg rays_d_arg(rays_d, "rays_d", 3);
	at::TensorArg t_min_arg(t_min, "t_min", 4);
	at::TensorArg t_max_arg(t_max, "t_max", 5);
	at::TensorArg seg_block_inds_arg(seg_block_inds, "seg_block_inds", 6);
	at::TensorArg seg_entries_arg(seg_entries, "seg_entries", 7);
	at::TensorArg seg_exits_arg(seg_exits, "seg_exits", 8);
	at::TensorArg seg_pack_infos_arg(seg_pack_infos, "seg_pack_infos", 9);
	at::TensorArg grid_binary_arg(grid_binary, "grid_binary", 10);

    at::checkDim(__func__, rays_o_arg, 2);
    at::checkDim(__func__, rays_d_arg, 2);
    at::checkDim(__func__, t_min_arg, 1);
    at::checkDim(__func__, t_max_arg, 1);
    at::checkDim(__func__, seg_block_inds_arg, 1);
    at::checkDim(__func__, seg_entries_arg, 1);
    at::checkDim(__func__, seg_exits_arg, 1);
    at::checkDim(__func__, seg_pack_infos_arg, 2);
    at::checkDim(__func__, grid_binary_arg, 4);
    at::checkAllSameGPU(__func__, {rays_o_arg, rays_d_arg, t_min_arg, t_max_arg, seg_block_inds_arg, seg_entries_arg, seg_exits_arg, seg_pack_infos_arg, grid_binary_arg});
    at::checkAllContiguous(__func__, {rays_o_arg, rays_d_arg, t_min_arg, t_max_arg, seg_block_inds_arg, seg_entries_arg, seg_exits_arg, seg_pack_infos_arg, grid_binary_arg});
    at::checkScalarType(__func__, rays_o_arg, at::kFloat);
    at::checkAllSameType(__func__, {rays_o_arg, rays_d_arg, t_min_arg, t_max_arg, seg_entries_arg, seg_exits_arg});
    at::checkScalarType(__func__, seg_block_inds_arg, at::kInt);
    at::checkScalarType(__func__, seg_pack_infos_arg, at::kInt);
    at::checkScalarType(__func__, grid_binary_arg, at::kBool);

    at::checkSameSize(__func__, rays_o_arg, rays_d_arg);
    at::checkSameSize(__func__, t_min_arg, t_max_arg);
    at::checkSameSize(__func__, seg_block_inds_arg, seg_entries_arg);
    at::checkSameSize(__func__, seg_block_inds_arg, seg_exits_arg);
    TORCH_CHECK(rays_o.size(0) == seg_pack_infos.size(0));
    TORCH_CHECK(rays_o.size(0) == t_min.size(0));

    const uint32_t n_rays = rays_o.size(0);
    const int3 grid_res = make_int3(grid_binary.size(1), grid_binary.size(2), grid_binary.size(3));

    const uint32_t threads = 256;
    const uint32_t blocks = CUDA_N_BLOCKS_NEEDED(n_rays, threads);

    // Helper counter
    at::Tensor num_steps = at::empty({n_rays}, rays_o.options().dtype(at::kInt));

    forest_marching_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        // Forest info
        {forest}, 
        // Rays
        n_rays, 
        rays_o.data_ptr<float>(), 
        rays_d.data_ptr<float>(), 
        t_min.data_ptr<float>(), 
        t_max.data_ptr<float>(), 
        // Block segments info
        seg_block_inds.data_ptr<int32_t>(), 
        seg_entries.data_ptr<float>(), 
        seg_exits.data_ptr<float>(), 
        seg_pack_infos.data_ptr<int32_t>(), 
        grid_res, 
        grid_binary.data_ptr<bool>(), 
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
        nullptr, /* blidx_out */
        nullptr  /* gidx_out */
    );

    at::Tensor cum_steps = num_steps.cumsum(0, at::kInt);
    at::Tensor packed_info = at::stack({cum_steps - num_steps, num_steps}, 1);

    // Output samples starts and ends
    uint32_t total_steps = cum_steps[cum_steps.size(0) - 1].item<int32_t>();
    at::Tensor t_starts = at::empty({total_steps, 1}, rays_o.options());
    at::Tensor t_ends = at::empty({total_steps, 1}, rays_o.options());
    at::Tensor ridx_out = at::empty({total_steps}, rays_o.options().dtype(at::kInt));
    at::Tensor blidx_out = at::empty({total_steps}, rays_o.options().dtype(at::kInt));
    at::Tensor gidx_out;
    if (return_gidx) {
        gidx_out = at::empty({total_steps}, rays_o.options().dtype(at::kInt));
    }

    forest_marching_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        // Forest info
        {forest}, 
        // Rays
        n_rays, 
        rays_o.data_ptr<float>(), 
        rays_d.data_ptr<float>(), 
        t_min.data_ptr<float>(), 
        t_max.data_ptr<float>(), 
        // Block segments info
        seg_block_inds.data_ptr<int32_t>(), 
        seg_entries.data_ptr<float>(), 
        seg_exits.data_ptr<float>(), 
        seg_pack_infos.data_ptr<int32_t>(), 
        grid_res, 
        grid_binary.data_ptr<bool>(), 
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
        blidx_out.data_ptr<int32_t>(), 
        return_gidx ? gidx_out.data_ptr<int32_t>() : nullptr
    );

    return {packed_info, t_starts, t_ends, ridx_out, blidx_out, gidx_out};
}
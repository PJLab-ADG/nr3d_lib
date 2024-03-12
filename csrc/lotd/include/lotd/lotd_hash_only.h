/** @file   lotd_hash_only.h
 *  @author Nianchen Deng, Shanghai AI Lab
 *  @brief  A simplified lotd-encoding kernel containing only Dense and Hash type implementation,
			much faster than the full kernel (tested on 3090).
 */

#include "lotd_torch_api.h"
#include "linear_interpolate.cuh"

using namespace lotd;

namespace lotd {
namespace torch {

template <typename INPUT_T, typename PARAM_T, typename COMPUTE_T, uint32_t N_POS_DIMS, uint32_t N_FEATS>
__global__ void
kernel_lod_hash_only(
	const uint32_t num_elements, 
	const LoDMetaRef lod_meta, 
	int32_t max_level,
	// inputs
	const PARAM_T *__restrict__ grid,         // [n_params]
	const INPUT_T *__restrict__ positions_in, // [n_points, 3]
	// Optional inputs for multi-batch data
	const int64_t *__restrict__ batch_inds,    // [n_points]
	const int64_t *__restrict__ batch_offsets, // [n_batch]
	const uint32_t batch_data_size,
	// outputs
	PARAM_T *__restrict__ encoded_positions
) {
	const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num_elements)
		return;

	const uint32_t pseudo_lvl = blockIdx.y;
	const uint32_t level = lod_meta.map_levels[pseudo_lvl];
	const uint32_t grid_feat_offset = lod_meta.map_cnt[pseudo_lvl] * N_FEATS;
	const uint32_t out_feat_offset = pseudo_lvl * N_FEATS;

	if (level > max_level) {
		return;
	}

	// For batched
	uint32_t batch_ind = 0;
	if (batch_inds) {
		// NOTE: pass in batch_ind=-1 to ignore certain points.
		if (batch_inds[i] < 0)
			return;
		batch_ind = batch_inds[i];
	} else if (batch_data_size) {
		batch_ind = i / batch_data_size;
	}
	const uint32_t batch_offset = batch_offsets ? batch_offsets[batch_ind] : (batch_ind * lod_meta.level_offsets[lod_meta.n_levels]);
	grid += (batch_offset + lod_meta.level_offsets[level]);

	const uint32_t grid_size = lod_meta.level_sizes[level];
	const uint32_t n_feat_cur_lvl = lod_meta.level_n_feats[level];
	const LoDType lod_type_cur_lvl = (LoDType)lod_meta.level_types[level];
	const InterpolationType interpolation_type = (InterpolationType)lod_meta.interpolation_type;

	uint32_t grid_resolution[N_POS_DIMS];
	COMPUTE_T scale[N_POS_DIMS];
	#pragma unroll
	for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
		grid_resolution[dim] = lod_meta.level_res[level][dim];
		scale[dim] = grid_resolution[dim] - 2; // Why not grid_resolution[dim] - 1?
	}
	COMPUTE_T pos[N_POS_DIMS];
	uint32_t pos_grid[N_POS_DIMS];

	positions_in += i * N_POS_DIMS;
	pos_fract<N_POS_DIMS, INPUT_T, COMPUTE_T>(positions_in, scale, interpolation_type, pos_grid, pos);

	vector_t<PARAM_T, N_FEATS> result = {};
	PARAM_T *result_ptr = (PARAM_T *)&result;

	{
		// auto grid_val_impl = lod_type_cur_lvl == LoDType::Dense ? grid_val_dense_impl<PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEATS>
		// 					: lod_type_cur_lvl == LoDType::Hash ? grid_val_hash_impl<PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEATS>
		// 					: nullptr; // Should never happen

		// linear_interpolate<INPUT_T, PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEATS, false>(
		// 	grid_val_impl, scale, pos, nullptr,
		// 	pos_grid, grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, grid,
		// 	result_ptr, nullptr);
		
		//---- NOTE: Using switch-case is 2.5x faster than altering function pointers
		switch (lod_type_cur_lvl) {
			case LoDType::Dense: 
			{
				linear_interpolate<INPUT_T, PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEATS, false>(
					grid_val_dense_impl<PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEATS>, scale, pos, nullptr,
					pos_grid, grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, grid,
					result_ptr, nullptr);
			} break;
			case LoDType::Hash: 
			{
				linear_interpolate<INPUT_T, PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEATS, false>(
					grid_val_hash_impl<PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEATS>, scale, pos, nullptr,
					pos_grid, grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, grid,
					result_ptr, nullptr);
			} break;
			default: break; // Should never happen
		}
	}

	#pragma unroll
	for (uint32_t f = 0; f < N_FEATS; ++f)
		encoded_positions[i + (out_feat_offset + f) * num_elements] = ((PARAM_T *)&result)[f];
}

template <typename INPUT_T, typename PARAM_T, typename COMPUTE_T, uint32_t N_POS_DIMS, uint32_t N_FEATS, bool PREFETCH_GRID_VALUE, bool PERMUTE_DYDX>
__global__ 
typename std::enable_if<PREFETCH_GRID_VALUE, void>::type // Prefetch
kernel_lod_hash_only_with_dydx(
	const uint32_t num_elements, 
	const LoDMetaRef lod_meta, 
	int32_t max_level,
	// inputs
	const PARAM_T *__restrict__ grid,         // [n_params]
	const INPUT_T *__restrict__ positions_in, // [n_points, 3]
	// Optional inputs for multi-batch data
	const int64_t *__restrict__ batch_inds,    // [n_points]
	const int64_t *__restrict__ batch_offsets, // [n_batch]
	const uint32_t batch_data_size,
	// outputs
	PARAM_T *__restrict__ encoded_positions, INPUT_T *__restrict__ dy_dx
) {
	const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num_elements)
		return;

	const uint32_t pseudo_lvl = blockIdx.y;
	const uint32_t level = lod_meta.map_levels[pseudo_lvl];
	const uint32_t grid_feat_offset = lod_meta.map_cnt[pseudo_lvl] * N_FEATS;
	const uint32_t out_feat_offset = pseudo_lvl * N_FEATS;

	if (level > max_level) {
		return;
	}

	// For batched
	uint32_t batch_ind = 0;
	if (batch_inds) {
		// NOTE: pass in batch_ind=-1 to ignore certain points.
		if (batch_inds[i] < 0)
			return;
		batch_ind = batch_inds[i];
	} else if (batch_data_size) {
		batch_ind = i / batch_data_size;
	}
	const uint32_t batch_offset = batch_offsets ? batch_offsets[batch_ind] : (batch_ind * lod_meta.level_offsets[lod_meta.n_levels]);
	grid += (batch_offset + lod_meta.level_offsets[level]);

	const uint32_t grid_size = lod_meta.level_sizes[level];
	const uint32_t n_feat_cur_lvl = lod_meta.level_n_feats[level];
	const LoDType lod_type_cur_lvl = (LoDType)lod_meta.level_types[level];
	const InterpolationType interpolation_type = (InterpolationType)lod_meta.interpolation_type;

	uint32_t grid_resolution[N_POS_DIMS];
	COMPUTE_T scale[N_POS_DIMS];
	#pragma unroll
	for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
		grid_resolution[dim] = lod_meta.level_res[level][dim];
		scale[dim] = grid_resolution[dim] - 2; // Why not grid_resolution[dim] - 1?
	}
	COMPUTE_T pos[N_POS_DIMS];
	COMPUTE_T pos_derivative[N_POS_DIMS];
	uint32_t pos_grid[N_POS_DIMS];

	positions_in += i * N_POS_DIMS;
	pos_fract<N_POS_DIMS, INPUT_T, COMPUTE_T>(positions_in, scale, interpolation_type, pos_grid, pos, pos_derivative);

	vector_t<PARAM_T, N_FEATS> result = {};
	PARAM_T *result_ptr = (PARAM_T *)&result;
	vector_t<INPUT_T, N_POS_DIMS> grads[N_FEATS] = {};

	//---- Prefetch grid values
	{
		// vector_t<PARAM_T, N_FEATS> grid_values[1 << N_POS_DIMS];
		// auto grid_val_impl = lod_type_cur_lvl == LoDType::Dense ? grid_val_dense_impl<PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEATS>
		// 					: lod_type_cur_lvl == LoDType::Hash ? grid_val_hash_impl<PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEATS>
		// 					: nullptr; // Should never happen

		// #pragma unroll
		// for (uint32_t idx = 0; idx < (1 << N_POS_DIMS); ++idx) {
		// 	uint32_t pos_grid_local[N_POS_DIMS];

		// 	#pragma unroll
		// 	for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
		// 		if ((idx & (1 << dim)) == 0)
		// 			pos_grid_local[dim] = pos_grid[dim];
		// 		else
		// 			pos_grid_local[dim] = pos_grid[dim] + 1;
		// 	}
		// 	grid_val_impl(pos_grid_local, grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl,
		// 		grid, (PARAM_T *)&grid_values[idx]);
		// }
		// linear_interpolate<INPUT_T, PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEATS, true>(
		// 	scale, pos, pos_derivative, grid_values, result_ptr, grads);
		

		//---- NOTE: Using switch-case is 2.5x faster than altering function pointers
		vector_t<PARAM_T, N_FEATS> grid_values[1 << N_POS_DIMS];
		switch(lod_type_cur_lvl) {
			case LoDType::Dense:
			{
				#pragma unroll
				for (uint32_t idx = 0; idx < (1 << N_POS_DIMS); ++idx) {
					uint32_t pos_grid_local[N_POS_DIMS];

					#pragma unroll
					for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
						if ((idx & (1 << dim)) == 0)
							pos_grid_local[dim] = pos_grid[dim];
						else
							pos_grid_local[dim] = pos_grid[dim] + 1;
					}
					grid_val_dense_impl<PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEATS>(
						pos_grid_local, grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl,
						grid, (PARAM_T *)&grid_values[idx]);
				}
			}
			break;

			case LoDType::Hash:
			{
				#pragma unroll
				for (uint32_t idx = 0; idx < (1 << N_POS_DIMS); ++idx) {
					uint32_t pos_grid_local[N_POS_DIMS];

					#pragma unroll
					for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
						if ((idx & (1 << dim)) == 0)
							pos_grid_local[dim] = pos_grid[dim];
						else
							pos_grid_local[dim] = pos_grid[dim] + 1;
					}
					grid_val_hash_impl<PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEATS>(
						pos_grid_local, grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl,
						grid, (PARAM_T *)&grid_values[idx]);
				}
			}
			break;
		}
		linear_interpolate<INPUT_T, PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEATS, true>(
			scale, pos, pos_derivative, grid_values, result_ptr, grads);

	}

	#pragma unroll
	for (uint32_t f = 0; f < N_FEATS; ++f)
		encoded_positions[i + (out_feat_offset + f) * num_elements] = ((PARAM_T *)&result)[f];

	if (PERMUTE_DYDX) {
		dy_dx += (out_feat_offset * num_elements + i) * N_POS_DIMS;
		#pragma unroll
		for (uint32_t f = 0; f < N_FEATS; ++f)
			((vector_t<INPUT_T, N_POS_DIMS> *)dy_dx)[f * num_elements] = grads[f];
	} else {
		dy_dx += N_POS_DIMS * (i * lod_meta.n_encoded_dims + out_feat_offset);
		#pragma unroll
		for (uint32_t f = 0; f < N_FEATS; ++f)
			((vector_t<INPUT_T, N_POS_DIMS> *)dy_dx)[f] = grads[f];
	}
}

template <typename INPUT_T, typename PARAM_T, typename COMPUTE_T, uint32_t N_POS_DIMS, uint32_t N_FEATS, bool PREFETCH_GRID_VALUE, bool PERMUTE_DYDX>
__global__ 
typename std::enable_if<!PREFETCH_GRID_VALUE, void>::type // No prefetch
kernel_lod_hash_only_with_dydx(
	const uint32_t num_elements, 
	const LoDMetaRef lod_meta, 
	int32_t max_level,
	// inputs
	const PARAM_T *__restrict__ grid,         // [n_params]
	const INPUT_T *__restrict__ positions_in, // [n_points, 3]
	// Optional inputs for multi-batch data
	const int64_t *__restrict__ batch_inds,    // [n_points]
	const int64_t *__restrict__ batch_offsets, // [n_batch]
	const uint32_t batch_data_size,
	// outputs
	PARAM_T *__restrict__ encoded_positions, INPUT_T *__restrict__ dy_dx
) {
	const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num_elements)
		return;

	const uint32_t pseudo_lvl = blockIdx.y;
	const uint32_t level = lod_meta.map_levels[pseudo_lvl];
	const uint32_t grid_feat_offset = lod_meta.map_cnt[pseudo_lvl] * N_FEATS;
	const uint32_t out_feat_offset = pseudo_lvl * N_FEATS;

	if (level > max_level) {
		return;
	}

	// For batched
	uint32_t batch_ind = 0;
	if (batch_inds) {
		// NOTE: pass in batch_ind=-1 to ignore certain points.
		if (batch_inds[i] < 0)
			return;
		batch_ind = batch_inds[i];
	} else if (batch_data_size) {
		batch_ind = i / batch_data_size;
	}
	const uint32_t batch_offset = batch_offsets ? batch_offsets[batch_ind] : (batch_ind * lod_meta.level_offsets[lod_meta.n_levels]);
	grid += (batch_offset + lod_meta.level_offsets[level]);

	const uint32_t grid_size = lod_meta.level_sizes[level];
	const uint32_t n_feat_cur_lvl = lod_meta.level_n_feats[level];
	const LoDType lod_type_cur_lvl = (LoDType)lod_meta.level_types[level];
	const InterpolationType interpolation_type = (InterpolationType)lod_meta.interpolation_type;

	uint32_t grid_resolution[N_POS_DIMS];
	COMPUTE_T scale[N_POS_DIMS];
	#pragma unroll
	for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
		grid_resolution[dim] = lod_meta.level_res[level][dim];
		scale[dim] = grid_resolution[dim] - 2; // Why not grid_resolution[dim] - 1?
	}
	COMPUTE_T pos[N_POS_DIMS];
	COMPUTE_T pos_derivative[N_POS_DIMS];
	uint32_t pos_grid[N_POS_DIMS];

	positions_in += i * N_POS_DIMS;
	pos_fract<N_POS_DIMS, INPUT_T, COMPUTE_T>(positions_in, scale, interpolation_type, pos_grid, pos, pos_derivative);

	vector_t<PARAM_T, N_FEATS> result = {};
	PARAM_T *result_ptr = (PARAM_T *)&result;
	vector_t<INPUT_T, N_POS_DIMS> grads[N_FEATS] = {};

	//---- No prefetch
	{
		// auto grid_val_impl = lod_type_cur_lvl == LoDType::Dense ? grid_val_dense_impl<PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEATS>
		// 					: lod_type_cur_lvl == LoDType::Hash ? grid_val_hash_impl<PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEATS>
		// 					: nullptr; // Should never happen

		// linear_interpolate<INPUT_T, PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEATS, true>(
		// 	grid_val_impl, scale, pos, pos_derivative,
		// 	pos_grid, grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, grid,
		// 	result_ptr, grads);

		//---- NOTE: Using switch-case is 2.5x faster than altering function pointers
		switch (lod_type_cur_lvl) {
		case LoDType::Dense:
			linear_interpolate<INPUT_T, PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEATS, true>(
				grid_val_dense_impl<PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEATS>, scale, pos, pos_derivative,
				pos_grid, grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, grid,
				result_ptr, grads);
			break;
		case LoDType::Hash:
			linear_interpolate<INPUT_T, PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEATS, true>(
				grid_val_hash_impl<PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEATS>, scale, pos, pos_derivative,
				pos_grid, grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, grid,
				result_ptr, grads);
			break;
		}
	}

	#pragma unroll
	for (uint32_t f = 0; f < N_FEATS; ++f)
		encoded_positions[i + (out_feat_offset + f) * num_elements] = ((PARAM_T *)&result)[f];

	if (PERMUTE_DYDX) {
		dy_dx += (out_feat_offset * num_elements + i) * N_POS_DIMS;
		#pragma unroll
		for (uint32_t f = 0; f < N_FEATS; ++f)
			((vector_t<INPUT_T, N_POS_DIMS> *)dy_dx)[f * num_elements] = grads[f];
	} else {
		dy_dx += N_POS_DIMS * (i * lod_meta.n_encoded_dims + out_feat_offset);
		#pragma unroll
		for (uint32_t f = 0; f < N_FEATS; ++f)
			((vector_t<INPUT_T, N_POS_DIMS> *)dy_dx)[f] = grads[f];
	}
}

template <typename INPUT_T, typename PARAM_T, typename GRAD_T, typename COMPUTE_T, uint32_t N_POS_DIMS, uint32_t N_FEATS, uint32_t N_FEAT_PER_THREAD>
__global__ void kernel_lod_hashonly_backward_grid(
	const uint32_t num_elements, 
	const LoDMetaRef lod_meta, 
	int32_t max_level, 
	const int32_t *__restrict__ max_level_gpu,
	// inputs
	at::PackedTensorAccessor32<PARAM_T, 2> dL_dy, const PARAM_T *__restrict__ grid,
	const INPUT_T *__restrict__ positions,
	// Optional inputs for multi-batch data
	const int64_t *__restrict__ batch_inds, const int64_t *__restrict__ batch_offsets,
	const uint32_t batch_data_size,
	// outputs
	GRAD_T *__restrict__ grid_gradient
) {
	const uint32_t i = ((blockIdx.x * blockDim.x + threadIdx.x) * N_FEAT_PER_THREAD) / N_FEATS;
	if (i >= num_elements)
		return;

	const uint32_t pseudo_lvl = blockIdx.y;
	const uint32_t level = lod_meta.map_levels[pseudo_lvl];
	const uint32_t feature = (blockIdx.x * blockDim.x + threadIdx.x) * N_FEAT_PER_THREAD - i * N_FEATS;
	const uint32_t grid_feat_offset = lod_meta.map_cnt[pseudo_lvl] * N_FEATS + feature;
	const uint32_t out_feat_offset = pseudo_lvl * N_FEATS + feature;

	if (max_level_gpu) {
		max_level = max_level_gpu[i];
	}

	if (level > max_level) {
		return;
	}

	// For batched
	uint32_t batch_ind = 0;
	if (batch_inds) {
		// NOTE: pass in batch_ind=-1 to ignore certain points.
		if (batch_inds[i] < 0) {
			return;
		}
		batch_ind = batch_inds[i];
	} else if (batch_data_size) {
		batch_ind = i / batch_data_size;
	}
	const uint32_t batch_offset = batch_offsets ? batch_offsets[batch_ind] : (batch_ind * lod_meta.level_offsets[lod_meta.n_levels]);
	grid += (batch_offset + lod_meta.level_offsets[level]);
	grid_gradient += (batch_offset + lod_meta.level_offsets[level]);

	const uint32_t grid_size = lod_meta.level_sizes[level];
	const uint32_t n_feat_cur_lvl = lod_meta.level_n_feats[level];
	const LoDType lod_type_cur_lvl = (LoDType)lod_meta.level_types[level];
	const InterpolationType interpolation_type = (InterpolationType)lod_meta.interpolation_type;

	uint32_t grid_resolution[N_POS_DIMS];
	COMPUTE_T scale[N_POS_DIMS];
	#pragma unroll
	for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
		grid_resolution[dim] = lod_meta.level_res[level][dim];
		scale[dim] = grid_resolution[dim] - 2;
	}
	COMPUTE_T pos[N_POS_DIMS];
	uint32_t pos_grid[N_POS_DIMS];

	positions += i * N_POS_DIMS;
	pos_fract<N_POS_DIMS, INPUT_T, COMPUTE_T>(positions, scale, interpolation_type, pos_grid, pos);

	// dL_dy += i * lod_meta.n_encoded_dims + out_feat_offset;
	// dL_dy += out_feat_offset * num_elements + i;
	auto dL_dy_i = dL_dy[i];
	vector_t<PARAM_T, N_FEAT_PER_THREAD> grad;
	#pragma unroll
	for (uint32_t f = 0; f < N_FEAT_PER_THREAD; ++f) {
		// grad[f] = dL_dy[f];
		grad[f] = dL_dy_i[out_feat_offset + f];
	}

	//---- NOTE: Using switch-case is 2.5x faster than altering function pointers
	switch (lod_type_cur_lvl) {
	case LoDType::Dense: {
		bwd_n_linear<PARAM_T, GRAD_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD>(
			add_grid_gridient_dense_impl<PARAM_T, GRAD_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD, std::is_same<GRAD_T, __half>::value>, 
			pos, pos_grid, grid_feat_offset, grid_resolution, grid_size,
			n_feat_cur_lvl, grid, grad, grid_gradient);
	} break;
	case LoDType::Hash: {
		bwd_n_linear<PARAM_T, GRAD_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD>(
			add_grid_gridient_hash_impl<PARAM_T, GRAD_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD, std::is_same<GRAD_T, __half>::value>, pos, pos_grid, grid_feat_offset, grid_resolution, grid_size,
			n_feat_cur_lvl, grid, grad, grid_gradient);
	} break;
	}
}

template <typename INPUT_T, typename PARAM_T, typename GRAD_T, typename COMPUTE_T, uint32_t N_POS_DIMS, uint32_t N_FEATS, uint32_t N_FEAT_PER_THREAD>
__global__ void kernel_lod_hashonly_backward_input_backward_grid(
	const uint32_t num_elements, 
	const LoDMetaRef lod_meta, 
	int32_t max_level, 
	const int32_t *__restrict__ max_level_gpu,
	// inputs
	const INPUT_T *__restrict__ dL_ddLdx, 
	at::PackedTensorAccessor32<PARAM_T, 2> dL_dy,
	const PARAM_T *__restrict__ grid, 
	const INPUT_T *__restrict__ positions,
	// Optional inputs for multi-batch data
	const int64_t *__restrict__ batch_inds, 
	const int64_t *__restrict__ batch_offsets,
	const uint32_t batch_data_size,
	// outputs
	GRAD_T *__restrict__ grid_gradient) {
	const uint32_t i = ((blockIdx.x * blockDim.x + threadIdx.x) * N_FEAT_PER_THREAD) / N_FEATS;
	if (i >= num_elements)
		return;

	const uint32_t pseudo_lvl = blockIdx.y;
	const uint32_t level = lod_meta.map_levels[pseudo_lvl];
	const uint32_t feature = (blockIdx.x * blockDim.x + threadIdx.x) * N_FEAT_PER_THREAD - i * N_FEATS;
	const uint32_t grid_feat_offset = lod_meta.map_cnt[pseudo_lvl] * N_FEATS + feature;
	const uint32_t out_feat_offset = pseudo_lvl * N_FEATS + feature;

	if (max_level_gpu) {
		max_level = max_level_gpu[i];
	}

	if (level > max_level) {
		return;
	}

	// For batched
	uint32_t batch_ind = 0;
	if (batch_inds) {
		// NOTE: pass in batch_ind=-1 to ignore certain points.
		if (batch_inds[i] < 0) {
			return;
		}
		batch_ind = batch_inds[i];
	} else if (batch_data_size) {
		batch_ind = i / batch_data_size;
	}
	const uint32_t batch_offset = batch_offsets ? batch_offsets[batch_ind] : (batch_ind * lod_meta.level_offsets[lod_meta.n_levels]);
	grid += (batch_offset + lod_meta.level_offsets[level]);
	grid_gradient += (batch_offset + lod_meta.level_offsets[level]);

	const uint32_t grid_size = lod_meta.level_sizes[level];
	const uint32_t n_feat_cur_lvl = lod_meta.level_n_feats[level];
	const LoDType lod_type_cur_lvl = (LoDType)lod_meta.level_types[level];
	const InterpolationType interpolation_type = (InterpolationType)lod_meta.interpolation_type;

	uint32_t grid_resolution[N_POS_DIMS];
	COMPUTE_T scale[N_POS_DIMS];
	#pragma unroll
	for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
		grid_resolution[dim] = lod_meta.level_res[level][dim];
		scale[dim] = grid_resolution[dim] - 2;
	}
	COMPUTE_T pos[N_POS_DIMS];
	COMPUTE_T pos_derivative[N_POS_DIMS];
	uint32_t pos_grid[N_POS_DIMS];

	positions += i * N_POS_DIMS;
	pos_fract<N_POS_DIMS, INPUT_T, COMPUTE_T>(positions, scale, interpolation_type, pos_grid, pos, pos_derivative);

	// dL_dy += i * lod_meta.n_encoded_dims + out_feat_offset;
	auto dL_dy_i = dL_dy[i];
	vector_t<PARAM_T, N_FEAT_PER_THREAD> grad;
	#pragma unroll
	for (uint32_t f = 0; f < N_FEAT_PER_THREAD; ++f) {
		// grad[f] = dL_dy[i + (out_feat_offset + f) * num_elements];
		// grad[f] = dL_dy[f];
		grad[f] = dL_dy_i[out_feat_offset + f];
	}

	dL_ddLdx += i * N_POS_DIMS;
	vector_t<INPUT_T, N_POS_DIMS> grad_input;
	#pragma unroll
	for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
		grad_input[dim] = dL_ddLdx[dim];
	}

	//---- NOTE: Using switch-case is 2.5x faster than altering function pointers
	switch (lod_type_cur_lvl) {
	case LoDType::Dense: {
		bwd_input_bwd_grid_n_linear<INPUT_T, PARAM_T, GRAD_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD>(
			add_grid_gridient_dense_impl<PARAM_T, GRAD_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD, std::is_same<GRAD_T, __half>::value>, 
			scale, pos, pos_derivative, pos_grid, grid_feat_offset,
			grid_resolution, grid_size, n_feat_cur_lvl, grid, grad, grad_input, grid_gradient);
	} break;

	case LoDType::Hash: {
		bwd_input_bwd_grid_n_linear<INPUT_T, PARAM_T, GRAD_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD>(
			add_grid_gridient_hash_impl<PARAM_T, GRAD_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD, std::is_same<GRAD_T, __half>::value>, 
			scale, pos, pos_derivative, pos_grid, grid_feat_offset,
			grid_resolution, grid_size, n_feat_cur_lvl, grid, grad, grad_input, grid_gradient);
	} break;
	}
}

template <typename INPUT_T, typename PARAM_T, typename COMPUTE_T, uint32_t N_POS_DIMS, uint32_t N_FEATS, uint32_t N_FEAT_PER_THREAD>
__global__ std::enable_if_t<std::is_same<INPUT_T, float>::value, void>
kernel_lod_hashonly_backward_input_backward_input(
	const uint32_t num_elements, 
	const LoDMetaRef lod_meta, 
	int32_t max_level, 
	const int32_t *__restrict__ max_level_gpu,
	// inputs
	const INPUT_T *__restrict__ dL_ddLdx, 
	at::PackedTensorAccessor32<PARAM_T, 2> dL_dy,
	const PARAM_T *__restrict__ grid, 
	const INPUT_T *__restrict__ positions,
	// Optional inputs for multi-batch data
	const int64_t *__restrict__ batch_inds, const int64_t *__restrict__ batch_offsets,
	const uint32_t batch_data_size,
	// outputs
	INPUT_T *dL_dx
) {
	const uint32_t i = ((blockIdx.x * blockDim.x + threadIdx.x) * N_FEAT_PER_THREAD) / N_FEATS;
	if (i >= num_elements)
		return;

	const uint32_t pseudo_lvl = blockIdx.y;
	const uint32_t level = lod_meta.map_levels[pseudo_lvl];
	const uint32_t feature = (blockIdx.x * blockDim.x + threadIdx.x) * N_FEAT_PER_THREAD - i * N_FEATS;
	const uint32_t grid_feat_offset = lod_meta.map_cnt[pseudo_lvl] * N_FEATS + feature;
	const uint32_t out_feat_offset = pseudo_lvl * N_FEATS + feature;

	if (max_level_gpu) {
		max_level = max_level_gpu[i];
	}

	if (level > max_level) {
		return;
	}

	// For batched
	uint32_t batch_ind = 0;
	if (batch_inds) {
		// NOTE: pass in batch_ind=-1 to ignore certain points.
		if (batch_inds[i] < 0) {
			return;
		}
		batch_ind = batch_inds[i];
	} else if (batch_data_size) {
		batch_ind = i / batch_data_size;
	}
	const uint32_t batch_offset = batch_offsets ? batch_offsets[batch_ind] : (batch_ind * lod_meta.level_offsets[lod_meta.n_levels]);
	grid += (batch_offset + lod_meta.level_offsets[level]);

	const uint32_t grid_size = lod_meta.level_sizes[level];
	const uint32_t n_feat_cur_lvl = lod_meta.level_n_feats[level];
	const LoDType lod_type_cur_lvl = (LoDType)lod_meta.level_types[level];
	const InterpolationType interpolation_type = (InterpolationType)lod_meta.interpolation_type;

	uint32_t grid_resolution[N_POS_DIMS];
	COMPUTE_T scale[N_POS_DIMS];
	#pragma unroll
	for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
		grid_resolution[dim] = lod_meta.level_res[level][dim];
		scale[dim] = grid_resolution[dim] - 2;
	}
	COMPUTE_T pos[N_POS_DIMS];
	COMPUTE_T pos_derivative[N_POS_DIMS];
	COMPUTE_T pos_2nd_derivative[N_POS_DIMS];
	uint32_t pos_grid[N_POS_DIMS];

	positions += i * N_POS_DIMS;
	pos_fract<N_POS_DIMS, INPUT_T, COMPUTE_T>(positions, scale, interpolation_type, pos_grid, pos, pos_derivative, pos_2nd_derivative);

	// dL_dy += i * lod_meta.n_encoded_dims + out_feat_offset;
	auto dL_dy_i = dL_dy[i];
	vector_t<PARAM_T, N_FEAT_PER_THREAD> grad;
	#pragma unroll
	for (uint32_t f = 0; f < N_FEAT_PER_THREAD; ++f) {
		// grad[f] = dL_dy[i + (out_feat_offset + f) * num_elements];
		// grad[f] = dL_dy[f];
		grad[f] = dL_dy_i[out_feat_offset + f];
	}

	dL_ddLdx += i * N_POS_DIMS;
	vector_t<INPUT_T, N_POS_DIMS> grad_input;
	#pragma unroll
	for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
		grad_input[dim] = dL_ddLdx[dim];
	}

	vector_t<INPUT_T, N_POS_DIMS> grad_result = {};

	//---- NOTE: Using switch-case is 2.5x faster than altering function pointers
	switch (lod_type_cur_lvl) {
	case LoDType::Dense: {
		bwd_input_bwd_input_n_linear<INPUT_T, PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD>(
			calc_dLdx_dim_dense_impl<PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD>, 
			scale, interpolation_type, pos, pos_derivative, pos_2nd_derivative,
			pos_grid, grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, grid, grad,
			grad_input, (INPUT_T *)&grad_result);
	} break;

	case LoDType::Hash: {
		bwd_input_bwd_input_n_linear<INPUT_T, PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD>(
			calc_dLdx_dim_hash_impl<PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD>, 
			scale, interpolation_type, pos, pos_derivative, pos_2nd_derivative,
			pos_grid, grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, grid, grad,
			grad_input, (INPUT_T *)&grad_result);
	} break;
	}

	dL_dx += i * N_POS_DIMS;

	#pragma unroll
	for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
		atomicAdd((INPUT_T *)(dL_dx + dim), (INPUT_T)grad_result[dim]);
	}

	// using input_t = typename std::conditional<std::is_same<INPUT_T, __half>::value, at::Half,
	// INPUT_T>::type; #pragma unroll for (uint32_t dim=0; dim < N_POS_DIMS; ++dim) {
	// 	gpuAtomicAdd((input_t*)(dL_dx + dim), (input_t)grad_result[dim]);
	// }
}

template <typename INPUT_T, typename PARAM_T, typename COMPUTE_T, uint32_t N_POS_DIMS, uint32_t N_FEATS>
inline void lod_hash_only_fwd_dispatched(
	LoDMeta &lod_meta, 
	at::Tensor input,
	at::Tensor params, 
	at::optional<at::Tensor> batch_inds_,
	at::optional<at::Tensor> batch_offsets_, 
	uint32_t batch_data_size,
	int32_t max_level, 
	bool need_input_grad, 
	at::Tensor output,
	at::Tensor dy_dx
) {
	const uint32_t batch_size = input.size(0);

	const dim3 blocks_lod = {div_round_up(batch_size, N_THREADS), lod_meta.n_pseudo_levels, 1};

	const at::cuda::OptionalCUDAGuard device_guard(at::device_of(params));
	cudaStream_t stream = at::cuda::getCurrentCUDAStream();
	cudaEvent_t start, stop;

	if (lod_meta.c_profile) {
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, stream);
	}

	if (need_input_grad) {
		auto fn = (lod_meta.c_prefetch && lod_meta.c_permute_dydx) ? kernel_lod_hash_only_with_dydx<INPUT_T, PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEATS, true, true>
				: (lod_meta.c_prefetch && !lod_meta.c_permute_dydx) ? kernel_lod_hash_only_with_dydx<INPUT_T, PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEATS, true, false>
				: (!lod_meta.c_prefetch && lod_meta.c_permute_dydx) ? kernel_lod_hash_only_with_dydx<INPUT_T, PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEATS, false, true>
				: (!lod_meta.c_prefetch && !lod_meta.c_permute_dydx) ? kernel_lod_hash_only_with_dydx<INPUT_T, PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEATS, false, false>
				: nullptr; // Should never happen

		fn<<<blocks_lod, N_THREADS, 0, stream>>>(
			batch_size, {lod_meta}, max_level, 
			data_ptr<PARAM_T>(params), data_ptr<INPUT_T>(input),
			batch_inds_.has_value() ? batch_inds_.value().data_ptr<int64_t>() : nullptr,
			batch_offsets_.has_value() ? batch_offsets_.value().data_ptr<int64_t>() : nullptr,
			batch_data_size, data_ptr<PARAM_T>(output), data_ptr<INPUT_T>(dy_dx)
		);
	} else {
		kernel_lod_hash_only<INPUT_T, PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEATS>
			<<<blocks_lod, N_THREADS, 0, stream>>>(
				batch_size, {lod_meta}, max_level, 
				data_ptr<PARAM_T>(params), data_ptr<INPUT_T>(input),
				batch_inds_.has_value() ? batch_inds_.value().data_ptr<int64_t>() : nullptr,
				batch_offsets_.has_value() ? batch_offsets_.value().data_ptr<int64_t>() : nullptr,
				batch_data_size, data_ptr<PARAM_T>(output));
	}

	if (lod_meta.c_profile) {
		cudaEventRecord(stop, stream);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		std::cout << "Hash-LOD" << input.size(1) << (need_input_grad ? "-grad" : "")
				  << (need_input_grad && lod_meta.c_prefetch ? "-prefetch" : "") << ", "
				  << (need_input_grad && lod_meta.c_permute_dydx ? "-perm" : "") << ", "
				  << input.size(0) << ", " << milliseconds << ", "
				  << milliseconds / input.size(0) * 1000000.0f << std::endl;
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}
}

template <uint32_t N_POS_DIMS, uint32_t N_FEATS>
inline void lod_hash_only_fwd_templated(
	LoDMeta &lod_meta, 
	at::Tensor input, 
	at::Tensor params,
	at::optional<at::Tensor> batch_inds_,
	at::optional<at::Tensor> batch_offsets_, 
	uint32_t batch_data_size,
	int32_t max_level, 
	bool need_input_grad, 
	at::Tensor output,
	at::Tensor dy_dx
) {
	auto fn = (input.scalar_type() == at::kHalf && params.scalar_type() == at::kHalf) ? lod_hash_only_fwd_dispatched<__half, __half, __half, N_POS_DIMS, N_FEATS>
			: (input.scalar_type() == at::kFloat && params.scalar_type() == at::kHalf) ? lod_hash_only_fwd_dispatched<float, __half, float, N_POS_DIMS, N_FEATS>
			: (input.scalar_type() == at::kFloat && params.scalar_type() == at::kFloat) ? lod_hash_only_fwd_dispatched<float, float, float, N_POS_DIMS, N_FEATS>
			: nullptr;
	if (!fn)
		throw std::runtime_error(
			"LoTDEncoding: Input type combination not supported. Supported types are: "
			"<input,param> -> (half, half), (float, half), (float, float)");
	fn(lod_meta, input, params, batch_inds_, batch_offsets_, batch_data_size, max_level, need_input_grad, output, dy_dx);
}

template <uint32_t N_POS_DIMS>
void lod_hash_only_fwd_impl(
	LoDMeta &lod_meta, 
	at::Tensor input, 
	at::Tensor params,
	at::optional<at::Tensor> batch_inds_, 
	at::optional<at::Tensor> batch_offsets_,
	uint32_t batch_data_size, 
	int32_t max_level, 
	bool need_input_grad,
	at::Tensor output, 
	at::Tensor dy_dx
) {
	auto fn = (lod_meta.n_feat_per_pseudo_lvl == 2) ? lod_hash_only_fwd_templated<N_POS_DIMS, 2>
			: (lod_meta.n_feat_per_pseudo_lvl == 4) ? lod_hash_only_fwd_templated<N_POS_DIMS, 4>
			: (lod_meta.n_feat_per_pseudo_lvl == 8) ? lod_hash_only_fwd_templated<N_POS_DIMS, 8>
			: nullptr;
	if (!fn)
		throw std::runtime_error("LoTDEncoding: `n_feat_per_pseudo_lvl` must be one of [2,4,8]");
	fn(lod_meta, input, params, batch_inds_, batch_offsets_, batch_data_size, max_level, need_input_grad, output, dy_dx);
}

template <typename INPUT_T, typename PARAM_T, typename COMPUTE_T, uint32_t N_POS_DIMS, uint32_t N_FEATS>
inline std::tuple<at::Tensor, at::Tensor>
lod_hash_only_bwd_dispatched(
	LoDMeta &lod_meta, 
	at::Tensor dL_dy, 
	at::Tensor input, 
	at::Tensor params,
	at::optional<at::Tensor> dy_dx_, 
	at::optional<at::Tensor> batch_inds_,
	at::optional<at::Tensor> batch_offsets_, 
	uint32_t batch_data_size,
	int32_t max_level, 
	bool need_input_grad, 
	bool need_param_grad,
	at::Tensor dL_dx, 
	at::Tensor dL_dparam
) {
	const uint32_t batch_size = input.size(0);
	const uint32_t num_elements = input.size(0);

	const at::cuda::OptionalCUDAGuard device_guard(at::device_of(params));
	cudaStream_t stream = at::cuda::getCurrentCUDAStream();
	cudaEvent_t start, stop;

	if (lod_meta.c_profile) {
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, stream);
	}

	if (need_input_grad) {
		// dL_dy, dy_dx -> dL_dx
		// [batch_size, n_encoded_dims], [batch_size, n_encoded_dims, n_dims_to_encode]
		// -> [batch_size, n_encoded_dims]

		auto dy_dx = dy_dx_.value().view({batch_size, lod_meta.n_encoded_dims, lod_meta.n_dims_to_encode});
		auto dL_dy_1 = dL_dy.to(scalar_type<INPUT_T>());
		
		if (lod_meta.c_bmm_backend == 0) {
			dL_dy_1 = dL_dy_1.unsqueeze(-2);
			dL_dx.unsqueeze_(-2);
			at::bmm_out(dL_dx, dL_dy_1, dy_dx);
			dL_dx.squeeze_(-2);
		} else if (lod_meta.c_bmm_backend == 1) {
			// Fastest
			dL_dy_1 = dL_dy_1.unsqueeze(-1);
			at::sum_out(dL_dx, at::mul(dL_dy_1, dy_dx), -2);
		} else if (lod_meta.c_bmm_backend == 2) {
			dL_dx = at::einsum("ij,ijk->ik", {dL_dy_1, dy_dx});
		}

		// dL_dx.unsqueeze_(-2);
		// at::bmm_out(dL_dx, dL_dy.view({batch_size, 1, lod_meta.n_encoded_dims}).to(scalar_type<INPUT_T>()), dy_dx_.value().view({batch_size, lod_meta.n_encoded_dims, lod_meta.n_dims_to_encode}));
		// dL_dx.squeeze_(-2);
	}

	if (need_param_grad) {
		static constexpr uint32_t N_FEAT_PER_THREAD = std::min(2u, N_FEATS);
		const dim3 blocks_lod = {
			div_round_up(batch_size * N_FEATS / N_FEAT_PER_THREAD, N_THREADS_BACK),
			lod_meta.n_pseudo_levels, 1};

		kernel_lod_hashonly_backward_grid<INPUT_T, PARAM_T, PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEATS, N_FEAT_PER_THREAD>
			<<<blocks_lod, N_THREADS_BACK, 0, stream>>>(
				batch_size, {lod_meta}, max_level, (int32_t *)nullptr, 
				packed_accessor32<PARAM_T, 2>(dL_dy), 
				data_ptr<PARAM_T>(params), data_ptr<INPUT_T>(input),
				batch_inds_.has_value() ? batch_inds_.value().data_ptr<int64_t>() : nullptr,
				batch_offsets_.has_value() ? batch_offsets_.value().data_ptr<int64_t>() : nullptr,
				batch_data_size, data_ptr<PARAM_T>(dL_dparam));
	}

	if (lod_meta.c_profile) {
		cudaEventRecord(stop, stream);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		std::cout << "Hash-LOD" << input.size(1) << "-bwd-"
				  << (need_input_grad ? "dx" : "") << (need_param_grad ? "dp" : "") << ", "
				  << input.size(0) << ", " << milliseconds << ", "
				  << milliseconds / input.size(0) * 1000000.0f << std::endl;
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	return std::make_tuple(dL_dx, dL_dparam);
}

template <uint32_t N_POS_DIMS, uint32_t N_FEATS>
inline std::tuple<at::Tensor, at::Tensor>
lod_hash_only_bwd_templated(
	LoDMeta &lod_meta, 
	at::Tensor dL_dy, 
	at::Tensor input, 
	at::Tensor params,
	at::optional<at::Tensor> dy_dx_, 
	at::optional<at::Tensor> batch_inds_,
	at::optional<at::Tensor> batch_offsets_, 
	uint32_t batch_data_size, 
	int32_t max_level,
	bool need_input_grad, 
	bool need_param_grad, 
	at::Tensor dL_dx, 
	at::Tensor dL_dparam
) {
	auto fn = (input.scalar_type() == at::kHalf && params.scalar_type() == at::kHalf) ? lod_hash_only_bwd_dispatched<__half, __half, __half, N_POS_DIMS, N_FEATS>
			: (input.scalar_type() == at::kFloat && params.scalar_type() == at::kHalf) ? lod_hash_only_bwd_dispatched<float, __half, float, N_POS_DIMS, N_FEATS>
			: (input.scalar_type() == at::kFloat && params.scalar_type() == at::kFloat) ? lod_hash_only_bwd_dispatched<float, float, float, N_POS_DIMS, N_FEATS>
			: nullptr;
	if (!fn)
		throw std::runtime_error(
			"LoTDEncoding: Input type combination not supported. Supported types are: "
			"<input,param> -> (half, half), (float, half), (float, float)");
	return fn(lod_meta, dL_dy, input, params, dy_dx_, batch_inds_, batch_offsets_, batch_data_size, max_level, need_input_grad, need_param_grad, dL_dx, dL_dparam);
}

template <uint32_t N_POS_DIMS>
std::tuple<at::Tensor, at::Tensor>
lod_hash_only_bwd_impl(
	LoDMeta &lod_meta, 
	at::Tensor dL_dy, 
	at::Tensor input, 
	at::Tensor params,
	at::optional<at::Tensor> dy_dx_, 
	at::optional<at::Tensor> batch_inds_,
	at::optional<at::Tensor> batch_offsets_, 
	uint32_t batch_data_size, 
	int32_t max_level,
	bool need_input_grad, 
	bool need_param_grad, 
	at::Tensor dL_dx, 
	at::Tensor dL_dparam
) {
	auto fn = (lod_meta.n_feat_per_pseudo_lvl == 2) ? lod_hash_only_bwd_templated<N_POS_DIMS, 2>
			: (lod_meta.n_feat_per_pseudo_lvl == 4) ? lod_hash_only_bwd_templated<N_POS_DIMS, 4>
			: (lod_meta.n_feat_per_pseudo_lvl == 8) ? lod_hash_only_bwd_templated<N_POS_DIMS, 8>
			: nullptr;
	if (!fn)
		throw std::runtime_error("LoTDEncoding: `n_feat_per_pseudo_lvl` must be one of [2,4,8]");
	return fn(lod_meta, dL_dy, input, params, dy_dx_, batch_inds_, batch_offsets_, batch_data_size, max_level, need_input_grad, need_param_grad, dL_dx, dL_dparam);
}

template <typename INPUT_T, typename PARAM_T, typename COMPUTE_T, uint32_t N_POS_DIMS, uint32_t N_FEATS>
inline void lod_hash_only_bwd_bwd_input_dispatched(
	LoDMeta &lod_meta, 
	at::Tensor dL_ddLdx, 
	at::Tensor dL_dy, 
	at::Tensor input, 
	at::Tensor params,
	at::optional<at::Tensor> dy_dx_, 
	at::optional<at::Tensor> batch_inds_,
	at::optional<at::Tensor> batch_offsets_, 
	uint32_t batch_data_size, 
	int32_t max_level,
	bool need_dLdy_grad, 
	bool need_input_grad, 
	bool need_param_grad, 
	at::Tensor dL_ddLdy,
	at::Tensor dL_dx, 
	at::Tensor dL_dparams
) {
	const uint32_t batch_size = input.size(0);

	const at::cuda::OptionalCUDAGuard device_guard(at::device_of(params));
	cudaStream_t stream = at::cuda::getCurrentCUDAStream();
	cudaEvent_t start, stop;

	if (lod_meta.c_profile) {
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, stream);
	}

	if (need_dLdy_grad) {
		// dL_ddLdx, dy_dx -> dL_ddLdy
		// [batch_size, n_dims_to_encode], [batch_size, n_encoded_dims, n_dims_to_encode]
		// -> [batch_size, n_encoded_dims]

		auto dy_dx = dy_dx_.value().view({batch_size, lod_meta.n_encoded_dims, lod_meta.n_dims_to_encode}).to(scalar_type<PARAM_T>());
		auto dL_ddLdx_1 = dL_ddLdx.to(scalar_type<PARAM_T>()); 

		if (lod_meta.c_bmm_backend == 0) {
			dL_ddLdx_1 = dL_ddLdx_1.unsqueeze(-1);
			dL_ddLdy.unsqueeze_(-1);
			at::bmm_out(dL_ddLdy, dy_dx, dL_ddLdx_1);
			dL_ddLdy.squeeze_(-1);
		} else if (lod_meta.c_bmm_backend == 1) {
			// Fastest
			dL_ddLdx_1 = dL_ddLdx_1.unsqueeze(-2);
			at::sum_out(dL_ddLdy, at::mul(dL_ddLdx_1, dy_dx), -1);
		} else if (lod_meta.c_bmm_backend == 2) {
			dL_ddLdy = at::einsum("ik,ijk->ij", {dL_ddLdx_1, dy_dx});
		}

		// dL_ddLdy.unsqueeze_(-1);
		// at::bmm_out(dL_ddLdy, dy_dx_.value().view({ batch_size, lod_meta.n_encoded_dims, lod_meta.n_dims_to_encode }).to(scalar_type<PARAM_T>()), dL_ddLdx.view({ batch_size, lod_meta.n_dims_to_encode, 1 }).to(scalar_type<PARAM_T>()) );
		// dL_ddLdy.squeeze_(-1);
	}

	if (need_input_grad) {
		static constexpr uint32_t N_FEAT_PER_THREAD = std::min(2u, N_FEATS);
		const dim3 blocks_lod = {
			div_round_up(batch_size * N_FEATS / N_FEAT_PER_THREAD, N_THREADS_BACK),
			lod_meta.n_pseudo_levels, 1};

		// Safer to use COMPUTE_T=float
		kernel_lod_hashonly_backward_input_backward_input<INPUT_T, PARAM_T, float, N_POS_DIMS, N_FEATS, N_FEAT_PER_THREAD>
			<<<blocks_lod, N_THREADS_BACK, 0, stream>>>(
				batch_size, {lod_meta}, max_level, (int32_t *)nullptr, 
				data_ptr<INPUT_T>(dL_ddLdx), packed_accessor32<PARAM_T, 2>(dL_dy), 
				data_ptr<PARAM_T>(params), data_ptr<INPUT_T>(input),
				batch_inds_.has_value() ? batch_inds_.value().data_ptr<int64_t>() : nullptr,
				batch_offsets_.has_value() ? batch_offsets_.value().data_ptr<int64_t>() : nullptr,
				batch_data_size, data_ptr<INPUT_T>(dL_dx));
	}

	if (need_param_grad) {
		static constexpr uint32_t N_FEAT_PER_THREAD = std::min(2u, N_FEATS);
		const dim3 blocks_lod = {
			div_round_up(batch_size * N_FEATS / N_FEAT_PER_THREAD, N_THREADS_BACK),
			lod_meta.n_pseudo_levels, 1};

		const at::cuda::OptionalCUDAGuard device_guard(at::device_of(params));
		cudaStream_t stream = at::cuda::getCurrentCUDAStream();

		kernel_lod_hashonly_backward_input_backward_grid<INPUT_T, PARAM_T, PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEATS, N_FEAT_PER_THREAD>
			<<<blocks_lod, N_THREADS_BACK, 0, stream>>>(
				batch_size, {lod_meta}, max_level, (int32_t *)nullptr, 
				data_ptr<INPUT_T>(dL_ddLdx), packed_accessor32<PARAM_T, 2>(dL_dy),
				data_ptr<PARAM_T>(params), data_ptr<INPUT_T>(input),
				batch_inds_.has_value() ? batch_inds_.value().data_ptr<int64_t>() : nullptr,
				batch_offsets_.has_value() ? batch_offsets_.value().data_ptr<int64_t>() : nullptr,
				batch_data_size, data_ptr<PARAM_T>(dL_dparams));
	}

	if (lod_meta.c_profile) {
		cudaEventRecord(stop, stream);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		std::cout << "Hash-LOD" << input.size(1) << "-bwd2-" // << (lod_meta.c_prefetch ? "-prefetch" : "")
				  << (need_input_grad ? "dx" : "") << (need_param_grad ? "dp" : "")
				  << (need_dLdy_grad ? "dLdy" : "") << ", " << input.size(0) << ", " << milliseconds
				  << ", " << milliseconds / input.size(0) * 1000000.0f << std::endl;
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}
}

template <uint32_t N_POS_DIMS, uint32_t N_FEATS>
inline void lod_hash_only_bwd_bwd_input_templated(
	LoDMeta &lod_meta, 
	at::Tensor dL_ddLdx, 
	at::Tensor dL_dy,
	at::Tensor input, 
	at::Tensor params, 
	at::optional<at::Tensor> dy_dx_,
	at::optional<at::Tensor> batch_inds_,
	at::optional<at::Tensor> batch_offsets_, 
	uint32_t batch_data_size,
	int32_t max_level, 
	bool need_dLdy_grad, 
	bool need_input_grad,
	bool need_param_grad, 
	at::Tensor dL_ddLdy, 
	at::Tensor dL_dx,
	at::Tensor dL_dparams
) {
	auto fn = (input.scalar_type() == at::kFloat && params.scalar_type() == at::kHalf) ? lod_hash_only_bwd_bwd_input_dispatched<float, __half, float, N_POS_DIMS, N_FEATS>
			: (input.scalar_type() == at::kFloat && params.scalar_type() == at::kFloat) ? lod_hash_only_bwd_bwd_input_dispatched<float, float, float, N_POS_DIMS, N_FEATS>
			: nullptr;
	if (!fn)
		throw std::runtime_error(
			"LoTDEncoding: Input type combination not supported. Supported types are: "
			"<input,param> -> (half, half), (float, half), (float, float)");
	fn(lod_meta, dL_ddLdx, dL_dy, input, params, dy_dx_, batch_inds_, batch_offsets_, batch_data_size, max_level, need_dLdy_grad, need_input_grad, need_param_grad, dL_ddLdy, dL_dx, dL_dparams);
}

template <uint32_t N_POS_DIMS>
void lod_hash_only_bwd_bwd_input_impl(
	LoDMeta &lod_meta, 
	at::Tensor dL_ddLdx, 
	at::Tensor dL_dy, 
	at::Tensor input,
	at::Tensor params, 
	at::optional<at::Tensor> dy_dx_,
	at::optional<at::Tensor> batch_inds_,
	at::optional<at::Tensor> batch_offsets_, 
	uint32_t batch_data_size,
	int32_t max_level, 
	bool need_dLdy_grad, 
	bool need_input_grad,
	bool need_param_grad, 
	at::Tensor dL_ddLdy, 
	at::Tensor dL_dx,
	at::Tensor dL_dparams
) {
	auto fn = (lod_meta.n_feat_per_pseudo_lvl == 2) ? lod_hash_only_bwd_bwd_input_templated<N_POS_DIMS, 2>
			: (lod_meta.n_feat_per_pseudo_lvl == 4) ? lod_hash_only_bwd_bwd_input_templated<N_POS_DIMS, 4>
			: (lod_meta.n_feat_per_pseudo_lvl == 8) ? lod_hash_only_bwd_bwd_input_templated<N_POS_DIMS, 8>
			: nullptr;
	if (!fn)
		throw std::runtime_error("LoTDEncoding: `n_feat_per_pseudo_lvl` must be one of [2,4,8]");
	fn(lod_meta, dL_ddLdx, dL_dy, input, params, dy_dx_, batch_inds_, batch_offsets_, batch_data_size, max_level, need_dLdy_grad, need_input_grad, need_param_grad, dL_ddLdy, dL_dx, dL_dparams);
}

} // namespace torch
} // namespace lotd
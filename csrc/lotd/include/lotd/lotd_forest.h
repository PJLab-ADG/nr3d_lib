/** @file   lotd_forest.h
 *  @author Jianfei Guo, Shanghai AI Lab
 *  @brief  LoTD-Forest. Pytorch-CUDA basd implementation.
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

#include <ATen/Functions.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>
// #include <THC/THCAtomics.cuh>

#include "lotd_cuda.h"
#include "lotd_torch_api.h"
#include "lotd_encoding.h"
#include "if_constexpr.hpp"
#include <forest.h>

namespace lotd {
namespace torch {

template <typename INPUT_T, typename PARAM_T, typename COMPUTE_T, uint32_t N_POS_DIMS, uint32_t N_FEAT, typename F>
__device__ __forceinline__ void forest_fwd_n_linear(
    F get_grid_val,

	const ForestMetaRef& forest,
	const uint32_t block_offset,
	const int16_t block_idx[N_POS_DIMS],
	const int64_t* __restrict__ block_offsets,
	const uint32_t block_n_params,

	const COMPUTE_T scale[N_POS_DIMS],
	const COMPUTE_T pos[N_POS_DIMS],
	const COMPUTE_T pos_derivative[N_POS_DIMS],
	const uint32_t pos_grid[N_POS_DIMS], 
	const uint32_t grid_feat_offset,
	const uint32_t grid_resolution[N_POS_DIMS],
	const uint32_t grid_size,
	const uint32_t n_feat_cur_lvl,
	const PARAM_T* __restrict__ grid, 
	PARAM_T* __restrict__ result_ptr, 
	vector_t<INPUT_T, N_POS_DIMS>* __restrict__ grads_ptr
) {
	auto grid_val = [&](const uint32_t local_pos[N_POS_DIMS]) {
		vector_t<PARAM_T, N_FEAT> val = {};

		// Continuity fixing for lod-forest-encoding: mapping index
		int16_t block_idx_local[N_POS_DIMS];
		uint32_t local_pos_local[N_POS_DIMS];
		bool block_changed = false;
		
		#pragma unroll
		for (uint32_t dim=0; dim<N_POS_DIMS; ++dim) {
			const uint32_t local_pos_dim = local_pos[dim];
			if (local_pos_dim == 0) {
				// (0) => left-neighbor's (grid_res-1)
				block_idx_local[dim] = block_idx[dim] - 1;
				local_pos_local[dim] = grid_resolution[dim] - 1;
				block_changed = true;
			} else if (local_pos_dim == grid_resolution[dim]+1) {
				// (grid_res+1) => right-neighbor's (0)
				block_idx_local[dim] = block_idx[dim] + 1;
				local_pos_local[dim] = 0;
				block_changed = true;
			} else { 
				// (1,2,3,...,grid_res) => self's (0,1,2,...,grid_res-1)
				block_idx_local[dim] = block_idx[dim];
				local_pos_local[dim] = local_pos_dim-1;
			}
		}

		uint32_t block_offset_local = block_offset;
		if (block_changed) {
			if (!forest.continuity_enabled) return val;
			// Re-check validness & find neighbor block's ind
			int32_t block_ind_local = forest.map_block_ind(block_idx_local);
			if (block_ind_local < 0) return val;
			block_offset_local = block_offsets ? block_offsets[block_ind_local] : (block_ind_local * block_n_params);
		}

		get_grid_val(local_pos_local, grid_feat_offset+block_offset_local, grid_resolution, grid_size, n_feat_cur_lvl, grid, (PARAM_T*)&val);
		return val;
	};

	if (result_ptr) {
		#pragma unroll 1 // Force skip unrolling; significantly reduce kernel code size & actually increase speed;
		for (uint32_t idx = 0; idx < (1 << N_POS_DIMS); ++idx) {
			COMPUTE_T weight = 1;
			uint32_t pos_grid_local[N_POS_DIMS];

			#pragma unroll
			for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
				if ((idx & (1<<dim)) == 0) {
					weight *= (COMPUTE_T)1 - (COMPUTE_T)pos[dim];
					pos_grid_local[dim] = pos_grid[dim];
				} else {
					weight *= pos[dim];
					pos_grid_local[dim] = pos_grid[dim] + 1;
				}
			}

			auto val = grid_val(pos_grid_local);

			#pragma unroll
			for (uint32_t f = 0; f < N_FEAT; ++f) {
				result_ptr[f] += (PARAM_T)(weight * (COMPUTE_T)((PARAM_T*)&val)[f]);
			}
		}
	}

	if (grads_ptr) { 
		#pragma unroll 1 // Force skip unrolling; significantly reduce kernel code size & actually increase speed;
		for (uint32_t grad_dim = 0; grad_dim < N_POS_DIMS; ++grad_dim) {
			#pragma unroll
			for (uint32_t idx = 0; idx < (1 << (N_POS_DIMS-1)); ++idx) {

				COMPUTE_T weight = scale[grad_dim] * pos_derivative[grad_dim];
				uint32_t pos_grid_local[N_POS_DIMS];

				#pragma unroll
				for (uint32_t non_grad_dim = 0; non_grad_dim < N_POS_DIMS-1; ++non_grad_dim) {
					const uint32_t dim = non_grad_dim >= grad_dim ? (non_grad_dim+1) : non_grad_dim;

					if ((idx & 1<<non_grad_dim) == 0) {
						weight *= (COMPUTE_T)1 - (COMPUTE_T)pos[dim];
						pos_grid_local[dim] = pos_grid[dim];
					} else {
						weight *= pos[dim];
						pos_grid_local[dim] = pos_grid[dim] + 1;
					}
				}

				pos_grid_local[grad_dim] = pos_grid[grad_dim];
				auto val_left = grid_val(pos_grid_local);
				pos_grid_local[grad_dim] = pos_grid[grad_dim] + 1;
				auto val_right = grid_val(pos_grid_local);

				#pragma unroll
				for (uint32_t f = 0; f < N_FEAT; ++f) {
					grads_ptr[f][grad_dim] += weight * ((COMPUTE_T)val_right[f] - (COMPUTE_T)val_left[f]);
				}
			}
		}
	}
}

template <typename INPUT_T, typename PARAM_T, typename COMPUTE_T, uint32_t N_POS_DIMS, uint32_t N_FEAT_PER_PSEUDO_LVL>
__global__ 
typename std::enable_if<N_POS_DIMS==3, void>::type 
kernel_lod_forest(
	const uint32_t num_elements,
	const LoDMetaRef lod_meta,
	const ForestMetaRef forest,
	int32_t max_level,
	const int32_t* __restrict__ max_level_gpu,   // [n_points]
	// inputs
	const PARAM_T* __restrict__ grid,          // [n_params]
	const INPUT_T* __restrict__ positions,  // [n_points, 3]
	// Optional inputs for multi-batch data
	const int64_t* __restrict__ block_inds,    // [n_points]
	const int64_t* __restrict__ block_offsets, // [n_batch]
	const uint32_t batch_data_size,
	// outputs
	PARAM_T* __restrict__ encoded,
	INPUT_T* __restrict__ dy_dx
) {
	const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num_elements) return;

	const uint32_t pseudo_lvl = blockIdx.y;
	const uint32_t level = lod_meta.map_levels[pseudo_lvl];
	const uint32_t grid_feat_offset = lod_meta.map_cnt[pseudo_lvl] * N_FEAT_PER_PSEUDO_LVL;
	const uint32_t out_feat_offset = pseudo_lvl * N_FEAT_PER_PSEUDO_LVL;

	if (max_level_gpu) {
		max_level = max_level_gpu[i];
	}

	auto set_zero = [&] () {
		if (encoded) {
			#pragma unroll
			for (uint32_t f = 0; f < N_FEAT_PER_PSEUDO_LVL; ++f) {
				// encoded[i + (out_feat_offset + f) * num_elements] = (PARAM_T)0.0f;
				encoded[i * lod_meta.n_encoded_dims + out_feat_offset + f] = (PARAM_T)0.0f;
			}
		}

		// Gradient is zero for zeroed-out dimensions.
		if (dy_dx) {
			#pragma unroll
			for (uint32_t f = 0; f < N_FEAT_PER_PSEUDO_LVL; ++f) {
				// ((vector_t<INPUT_T, N_POS_DIMS>*)dy_dx)[i + (out_feat_offset + f) * num_elements] = {0};
				((vector_t<INPUT_T, N_POS_DIMS>*)dy_dx)[i * lod_meta.n_encoded_dims + out_feat_offset + f] = {};
			}
		}
	};

	if (level > max_level) {
		set_zero();
		return;
	}

	// For batched
	uint32_t block_ind = 0;
	if (block_inds) {
		// NOTE: pass in block_ind=-1 to ignore certain points.
		if (block_inds[i] < 0) { set_zero(); return;}
		block_ind = block_inds[i];
	} else if (batch_data_size) {
		block_ind = i / batch_data_size;
	}
	const uint32_t block_n_params = lod_meta.level_offsets[lod_meta.n_levels];
	const uint32_t block_offset = block_offsets ? block_offsets[block_ind] : block_ind * block_n_params;
	const short3 block_k = forest.block_ks[block_ind];
	const int16_t block_idx[N_POS_DIMS] = {block_k.x, block_k.y, block_k.z};

	grid += lod_meta.level_offsets[level];

	const uint32_t grid_size = lod_meta.level_sizes[level];
	const uint32_t n_feat_cur_lvl = lod_meta.level_n_feats[level];
	const LoDType lod_type_cur_lvl = (LoDType)lod_meta.level_types[level];
	const InterpolationType interpolation_type = (InterpolationType)lod_meta.interpolation_type;

	uint32_t grid_resolution[N_POS_DIMS];
	COMPUTE_T scale[N_POS_DIMS];
	#pragma unroll
	for (uint32_t dim=0; dim < N_POS_DIMS; ++dim) {
		grid_resolution[dim] = lod_meta.level_res[level][dim];
		scale[dim] = grid_resolution[dim]; // NOTE: for forest
	}

	COMPUTE_T pos[N_POS_DIMS];
	COMPUTE_T pos_derivative[N_POS_DIMS];
	uint32_t pos_grid[N_POS_DIMS];
	positions += i * N_POS_DIMS;
	
	if (dy_dx) {
		pos_fract<N_POS_DIMS, INPUT_T, COMPUTE_T>(positions, scale, interpolation_type, pos_grid, pos, pos_derivative);
	} else {
		pos_fract<N_POS_DIMS, INPUT_T, COMPUTE_T>(positions, scale, interpolation_type, pos_grid, pos);
	}

	// PARAM_T* result_ptr = encoded ? (encoded + i * lod_meta.n_encoded_dims + out_feat_offset) : nullptr;
	vector_t<PARAM_T, N_FEAT_PER_PSEUDO_LVL> result = {};
	PARAM_T* result_ptr = encoded ? (PARAM_T*)&result : nullptr;

	// vector_t<INPUT_T, N_POS_DIMS>* grads_ptr = dy_dx ? ( ((vector_t<INPUT_T, N_POS_DIMS>*)dy_dx) + i * lod_meta.n_encoded_dims + out_feat_offset ) : nullptr;
	vector_t<INPUT_T, N_POS_DIMS> grads[N_FEAT_PER_PSEUDO_LVL] = {};
	vector_t<INPUT_T, N_POS_DIMS>* grads_ptr = dy_dx ? grads : nullptr;

	//---- NOTE: Using switch-case is 2.5x faster than altering function pointers
	switch (lod_type_cur_lvl)
	{
		case LoDType::Dense:
		{
			forest_fwd_n_linear<INPUT_T, PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL>(
				grid_val_dense_impl<PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL>, 
				forest, block_offset, block_idx, block_offsets, block_n_params, scale, pos, pos_derivative, pos_grid, 
				grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, grid, result_ptr, grads_ptr);
		}
		break;

		case LoDType::VectorMatrix:
		{
			forest_fwd_n_linear<INPUT_T, PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL>(
				grid_val_vm_impl<PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL>, 
				forest, block_offset, block_idx, block_offsets, block_n_params, scale, pos, pos_derivative, pos_grid, 
				grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, grid, result_ptr, grads_ptr);
		}
		break;

		case LoDType::NPlaneMul:
		{
			forest_fwd_n_linear<INPUT_T, PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL>(
				grid_val_nplane_mul_impl<PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL>, 
				forest, block_offset, block_idx, block_offsets, block_n_params, scale, pos, pos_derivative, pos_grid, 
				grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, grid, result_ptr, grads_ptr);
		}
		break;

		case LoDType::CP:
		{
			forest_fwd_n_linear<INPUT_T, PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL>(
				grid_val_cp_eq_impl<PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL>, 
				forest, block_offset, block_idx, block_offsets, block_n_params, scale, pos, pos_derivative, pos_grid, 
				grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, grid, result_ptr, grads_ptr);
		}
		break;

		case LoDType::Hash:
		{
			forest_fwd_n_linear<INPUT_T, PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL>(
				grid_val_hash_impl<PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL>, 
				forest, block_offset, block_idx, block_offsets, block_n_params, scale, pos, pos_derivative, pos_grid, 
				grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, grid, result_ptr, grads_ptr);
		}
		break;

		default:
			// Should never happen
			break;
	}

	// NOTE: 1.8x slower compared with above running in seperate switch case.
	// forest_fwd_n_linear<INPUT_T, PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL>(get_grid_val, forest, block_offset, block_idx, block_offsets, block_n_params, scale, pos, pos_grid, grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, grid, result_ptr, grads_ptr);

	if (encoded) {
		encoded += i * lod_meta.n_encoded_dims + out_feat_offset;
		#pragma unroll
		for (uint32_t f=0; f<N_FEAT_PER_PSEUDO_LVL; ++f) {
			encoded[f] = ((PARAM_T*)&result)[f];
		}
	}

	if (dy_dx) {
		dy_dx += N_POS_DIMS * (i * lod_meta.n_encoded_dims + out_feat_offset);
		#pragma unroll
		for (uint32_t f=0; f<N_FEAT_PER_PSEUDO_LVL; ++f) {
			((vector_t<INPUT_T, N_POS_DIMS>*)dy_dx)[f] = grads[f];
		}
	}			
}

template <typename PARAM_T, typename GRAD_T, typename COMPUTE_T, uint32_t N_POS_DIMS, uint32_t N_FEAT, typename F>
__device__ __forceinline__ void forest_bwd_n_linear(
	F add_grid_grad, 

	const ForestMetaRef& forest,
	const uint32_t block_offset,
	const int16_t block_idx[N_POS_DIMS],
	const int64_t* __restrict__ block_offsets,
	const uint32_t block_n_params,

	const COMPUTE_T pos[N_POS_DIMS],
	const uint32_t pos_grid[N_POS_DIMS], 
	const uint32_t grid_feat_offset,
	const uint32_t grid_resolution[N_POS_DIMS],
	const uint32_t grid_size,
	const uint32_t n_feat_cur_lvl,
	const PARAM_T* __restrict__ grid, 
	const vector_t<PARAM_T, N_FEAT>&  grad,
	GRAD_T* __restrict__ grid_gradient
) {
	auto add_grid_gradient = [&](const uint32_t local_pos[N_POS_DIMS], const vector_t<PARAM_T, N_FEAT>& grad, const COMPUTE_T weight) {
		// Continuity fixing for lod-forest-encoding: mapping index
		int16_t block_idx_local[N_POS_DIMS];
		uint32_t local_pos_local[N_POS_DIMS];
		bool block_changed = false;

		#pragma unroll
		for (uint32_t dim=0; dim<N_POS_DIMS; ++dim) {
			const uint32_t local_pos_dim = local_pos[dim];
			if (local_pos_dim == 0) {
				// (0) => left-neighbor's (grid_res-1)
				block_idx_local[dim] = block_idx[dim] - 1;
				local_pos_local[dim] = grid_resolution[dim] - 1;
				block_changed = true;
			} else if (local_pos_dim == grid_resolution[dim]+1) {
				// (grid_res+1) => right-neighbor's (0)
				block_idx_local[dim] = block_idx[dim] + 1;
				local_pos_local[dim] = 0;
				block_changed = true;
			} else { 
				// (1,2,3,...,grid_res) => self's (0,1,2,...,grid_res-1)
				block_idx_local[dim] = block_idx[dim];
				local_pos_local[dim] = local_pos_dim-1;
			}
		}

		uint32_t block_offset_local = block_offset;
		if (block_changed) {
			if (!forest.continuity_enabled) return;
			// Re-check validness & find neighbor block's ind
			int32_t block_ind_local = forest.map_block_ind(block_idx_local);
			if (block_ind_local < 0) return;
			block_offset_local = block_offsets ? block_offsets[block_ind_local] : (block_ind_local * block_n_params);
		}

		add_grid_grad(local_pos_local, grid_feat_offset+block_offset_local, grid_resolution, grid_size, n_feat_cur_lvl, grid, (PARAM_T*)&grad, weight, grid_gradient);
	};

	// N-linear interpolation
	#pragma unroll 1 // Force skip unrolling; significantly reduce kernel code size & actually increase speed;
	for (uint32_t idx = 0; idx < (1 << N_POS_DIMS); ++idx) {
		COMPUTE_T weight = 1;
		uint32_t pos_grid_local[N_POS_DIMS];

		#pragma unroll
		for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
			if ((idx & (1<<dim)) == 0) {
				weight *= (COMPUTE_T)1 - (COMPUTE_T)pos[dim];
				pos_grid_local[dim] = pos_grid[dim];
			} else {
				weight *= pos[dim];
				pos_grid_local[dim] = pos_grid[dim] + 1;
			}
		}

		add_grid_gradient(pos_grid_local, grad, weight);
	}
}

template <typename INPUT_T, typename PARAM_T, typename GRAD_T, typename COMPUTE_T, uint32_t N_POS_DIMS, uint32_t N_FEAT_PER_PSEUDO_LVL, uint32_t N_FEAT_PER_THREAD>
__global__ 
typename std::enable_if<N_POS_DIMS==3, void>::type 
kernel_lod_forest_backward_grid(
	const uint32_t num_elements,
	const LoDMetaRef lod_meta,
	const ForestMetaRef forest,
	int32_t max_level,
	const int32_t* __restrict__ max_level_gpu,   // [n_points]
	// inputs
	const PARAM_T* __restrict__ dL_dy,         // [n_points, n_lod_features]
	const PARAM_T* __restrict__ grid,          // [n_params]
	const INPUT_T* __restrict__ positions,  // [n_points, 3]
	// Optional inputs for multi-batch data
	const int64_t* __restrict__ block_inds,    // [n_points]
	const int64_t* __restrict__ block_offsets, // [n_batch]
	const uint32_t batch_data_size,
	// outputs
	GRAD_T* __restrict__ grid_gradient
) {
	const uint32_t i = ((blockIdx.x * blockDim.x + threadIdx.x) * N_FEAT_PER_THREAD) / N_FEAT_PER_PSEUDO_LVL;
	if (i >= num_elements) return;

	const uint32_t pseudo_lvl = blockIdx.y;
	const uint32_t level = lod_meta.map_levels[pseudo_lvl];
	const uint32_t feature = (blockIdx.x * blockDim.x + threadIdx.x) * N_FEAT_PER_THREAD - i * N_FEAT_PER_PSEUDO_LVL;
	const uint32_t grid_feat_offset = lod_meta.map_cnt[pseudo_lvl] * N_FEAT_PER_PSEUDO_LVL + feature;
	const uint32_t out_feat_offset = pseudo_lvl * N_FEAT_PER_PSEUDO_LVL + feature;

	if (max_level_gpu) {
		max_level = max_level_gpu[i];
	}

	if (level > max_level) {
		return;
	}

	// For batched
	uint32_t block_ind = 0;
	if (block_inds) {
		// NOTE: pass in block_ind=-1 to ignore certain points.
		if (block_inds[i] < 0) return;
		block_ind = block_inds[i];
	} else if (batch_data_size) {
		block_ind = i / batch_data_size;
	}
	const uint32_t block_n_params = lod_meta.level_offsets[lod_meta.n_levels];
	const uint32_t block_offset = block_offsets ? block_offsets[block_ind] : block_ind * block_n_params;
	const short3 block_k = forest.block_ks[block_ind];
	const int16_t block_idx[N_POS_DIMS] = {block_k.x, block_k.y, block_k.z};

	grid += lod_meta.level_offsets[level];
	grid_gradient += lod_meta.level_offsets[level];

	const uint32_t grid_size = lod_meta.level_sizes[level];
	const uint32_t n_feat_cur_lvl = lod_meta.level_n_feats[level];
	const LoDType lod_type_cur_lvl = (LoDType)lod_meta.level_types[level];
	const InterpolationType interpolation_type = (InterpolationType)lod_meta.interpolation_type;

	uint32_t grid_resolution[N_POS_DIMS];
	COMPUTE_T scale[N_POS_DIMS];
	#pragma unroll
	for (uint32_t dim=0; dim < N_POS_DIMS; ++dim) {
		grid_resolution[dim] = lod_meta.level_res[level][dim];
		scale[dim] = grid_resolution[dim]; // NOTE: for forest
	}

	COMPUTE_T pos[N_POS_DIMS];
	uint32_t pos_grid[N_POS_DIMS];
	positions += i * N_POS_DIMS;
	pos_fract<N_POS_DIMS, INPUT_T, COMPUTE_T>(positions, scale, interpolation_type, pos_grid, pos);

	dL_dy += i * lod_meta.n_encoded_dims + out_feat_offset;
	vector_t<PARAM_T, N_FEAT_PER_THREAD> grad;
	#pragma unroll
	for (uint32_t f = 0; f < N_FEAT_PER_THREAD; ++f) {
		// grad[f] = dL_dy[i + (out_feat_offset + f) * num_elements];
		grad[f] = dL_dy[f];
	}

	//---- NOTE: Using switch-case is 2.5x faster than altering function pointers
	switch (lod_type_cur_lvl)
	{
		case LoDType::Dense:
		{
			forest_bwd_n_linear<PARAM_T, GRAD_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD>(
				add_grid_gridient_dense_impl<PARAM_T, GRAD_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD, std::is_same<GRAD_T, __half>::value>, 
				forest, block_offset, block_idx, block_offsets, block_n_params, pos, pos_grid, 
				grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, grid, grad, grid_gradient);
		}
		break;

		case LoDType::NPlaneMul:
		{
			forest_bwd_n_linear<PARAM_T, GRAD_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD>(
				add_grid_gridient_nplane_mul_impl<PARAM_T, GRAD_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD, std::is_same<GRAD_T, __half>::value>, 
				forest, block_offset, block_idx, block_offsets, block_n_params, pos, pos_grid, 
				grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, grid, grad, grid_gradient);
		}
		break;

		case LoDType::VectorMatrix:
		{
			forest_bwd_n_linear<PARAM_T, GRAD_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD>(
				add_grid_gridient_vm_impl<PARAM_T, GRAD_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD, std::is_same<GRAD_T, __half>::value>, 
				forest, block_offset, block_idx, block_offsets, block_n_params, pos, pos_grid, 
				grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, grid, grad, grid_gradient);
		}
		break;

		case LoDType::CP:
		{
			forest_bwd_n_linear<PARAM_T, GRAD_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD>(
				add_grid_gridient_cp_eq_impl<PARAM_T, GRAD_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD, std::is_same<GRAD_T, __half>::value>, 
				forest, block_offset, block_idx, block_offsets, block_n_params, pos, pos_grid, 
				grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, grid, grad, grid_gradient);
		}
		break;

		case LoDType::Hash:
		{
			forest_bwd_n_linear<PARAM_T, GRAD_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD>(
				add_grid_gridient_hash_impl<PARAM_T, GRAD_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD, std::is_same<GRAD_T, __half>::value>, 
				forest, block_offset, block_idx, block_offsets, block_n_params, pos, pos_grid, 
				grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, grid, grad, grid_gradient);
		}
		break;
	}
}

template <typename INPUT_T, typename PARAM_T, typename GRAD_T, typename COMPUTE_T, uint32_t N_POS_DIMS, uint32_t N_FEAT, typename F>
__device__ __forceinline__ void forest_bwd_input_bwd_grid_n_linear(
	F add_grid_grad, 

	const ForestMetaRef& forest,
	const uint32_t block_offset,
	const int16_t block_idx[N_POS_DIMS],
	const int64_t* __restrict__ block_offsets,
	const uint32_t block_n_params,

	const COMPUTE_T scale[N_POS_DIMS],
	const COMPUTE_T pos[N_POS_DIMS],
	const COMPUTE_T pos_derivative[N_POS_DIMS],
	const uint32_t pos_grid[N_POS_DIMS], 
	const uint32_t grid_feat_offset,
	const uint32_t grid_resolution[N_POS_DIMS], 
	const uint32_t grid_size,
	const uint32_t n_feat_cur_lvl,
	const PARAM_T* __restrict__ grid, 
	const vector_t<PARAM_T, N_FEAT>& grad,
	const vector_t<INPUT_T, N_POS_DIMS>& grad_input,
	GRAD_T* __restrict__ grid_gradient
) {
	auto add_grid_gradient = [&](const uint32_t local_pos[N_POS_DIMS], const vector_t<PARAM_T, N_FEAT>& grad, const COMPUTE_T weight) {
		// Continuity fixing for lod-forest-encoding: mapping index
		int16_t block_idx_local[N_POS_DIMS];
		uint32_t local_pos_local[N_POS_DIMS];
		bool block_changed = false;

		#pragma unroll
		for (uint32_t dim=0; dim<N_POS_DIMS; ++dim) {
			const uint32_t local_pos_dim = local_pos[dim];
			if (local_pos_dim == 0) {
				// (0) => left-neighbor's (grid_res-1)
				block_idx_local[dim] = block_idx[dim] - 1;
				local_pos_local[dim] = grid_resolution[dim] - 1;
				block_changed = true;
			} else if (local_pos_dim == grid_resolution[dim]+1) {
				// (grid_res+1) => right-neighbor's (0)
				block_idx_local[dim] = block_idx[dim] + 1;
				local_pos_local[dim] = 0;
				block_changed = true;
			} else { 
				// (1,2,3,...,grid_res) => self's (0,1,2,...,grid_res-1)
				block_idx_local[dim] = block_idx[dim];
				local_pos_local[dim] = local_pos_dim-1;
			}
		}

		uint32_t block_offset_local = block_offset;
		if (block_changed) {
			if (!forest.continuity_enabled) return;
			// Re-check validness & find neighbor block's ind
			int32_t block_ind_local = forest.map_block_ind(block_idx_local);
			if (block_ind_local < 0) return;
			block_offset_local = block_offsets ? block_offsets[block_ind_local] : (block_ind_local * block_n_params);
		}

		add_grid_grad(local_pos_local, grid_feat_offset+block_offset_local, grid_resolution, grid_size, n_feat_cur_lvl, grid, (PARAM_T*)&grad, weight, grid_gradient);
	};

	#pragma unroll 1 // Force skip unrolling; significantly reduce kernel code size & actually increase speed;
	for (uint32_t grad_dim = 0; grad_dim < N_POS_DIMS; ++grad_dim) {
		COMPUTE_T grad_in = scale[grad_dim] * (COMPUTE_T)grad_input[grad_dim] * pos_derivative[grad_dim];
		#pragma unroll
		for (uint32_t idx = 0; idx < (1 << (N_POS_DIMS-1)); ++idx) {
			COMPUTE_T weight = grad_in;
			uint32_t pos_grid_local[N_POS_DIMS];
		
			#pragma unroll
			for (uint32_t non_grad_dim = 0; non_grad_dim < N_POS_DIMS-1; ++non_grad_dim) {
				const uint32_t dim = non_grad_dim >= grad_dim ? (non_grad_dim+1) : non_grad_dim;

				if ((idx & 1<<non_grad_dim) == 0) {
					weight *= (COMPUTE_T)1 - pos[dim];
					pos_grid_local[dim] = pos_grid[dim];
				} else {
					weight *= pos[dim];
					pos_grid_local[dim] = pos_grid[dim] + 1;
				}
			}

			// left
			pos_grid_local[grad_dim] = pos_grid[grad_dim];
			add_grid_gradient(pos_grid_local, grad, -weight);
			// right
			pos_grid_local[grad_dim] = pos_grid[grad_dim] + 1;
			add_grid_gradient(pos_grid_local, grad, weight);
		}
	}
}

template <typename INPUT_T, typename PARAM_T, typename GRAD_T, typename COMPUTE_T, uint32_t N_POS_DIMS, uint32_t N_FEAT_PER_PSEUDO_LVL, uint32_t N_FEAT_PER_THREAD>
__global__ 
typename std::enable_if<N_POS_DIMS==3, void>::type
kernel_lod_forest_backward_input_backward_grid(
	const uint32_t num_elements,
	const LoDMetaRef lod_meta,
	const ForestMetaRef forest,
	int32_t max_level,
	const int32_t* __restrict__ max_level_gpu,
	// inputs
	const INPUT_T* __restrict__ dL_ddLdx,      // [n_points, 3]
	const PARAM_T* __restrict__ dL_dy,         // [n_points, n_lod_features]
	const PARAM_T* __restrict__ grid,          // [n_params]
	const INPUT_T* __restrict__ positions,  // [n_points, 3]
	// Optional inputs for multi-batch data
	const int64_t* __restrict__ block_inds,    // [n_points]
	const int64_t* __restrict__ block_offsets, // [n_batch]
	const uint32_t batch_data_size,
	// outputs
	GRAD_T* __restrict__ grid_gradient
) {
	const uint32_t i = ((blockIdx.x * blockDim.x + threadIdx.x) * N_FEAT_PER_THREAD) / N_FEAT_PER_PSEUDO_LVL;
	if (i >= num_elements) return;

	const uint32_t pseudo_lvl = blockIdx.y;
	const uint32_t level = lod_meta.map_levels[pseudo_lvl];
	const uint32_t feature = (blockIdx.x * blockDim.x + threadIdx.x) * N_FEAT_PER_THREAD - i * N_FEAT_PER_PSEUDO_LVL;
	const uint32_t grid_feat_offset = lod_meta.map_cnt[pseudo_lvl] * N_FEAT_PER_PSEUDO_LVL + feature;
	const uint32_t out_feat_offset = pseudo_lvl * N_FEAT_PER_PSEUDO_LVL + feature;

	if (max_level_gpu) {
		max_level = max_level_gpu[i];
	}

	if (level > max_level) {
		return;
	}

	// For batched
	uint32_t block_ind = 0;
	if (block_inds) {
		// NOTE: pass in block_ind=-1 to ignore certain points.
		if (block_inds[i] < 0) return;
		block_ind = block_inds[i];
	} else if (batch_data_size) {
		block_ind = i / batch_data_size;
	}
	const uint32_t block_n_params = lod_meta.level_offsets[lod_meta.n_levels];
	const uint32_t block_offset = block_offsets ? block_offsets[block_ind] : block_ind * block_n_params;
	const short3 block_k = forest.block_ks[block_ind];
	const int16_t block_idx[N_POS_DIMS] = {block_k.x, block_k.y, block_k.z};

	grid += lod_meta.level_offsets[level];
	grid_gradient += lod_meta.level_offsets[level];

	const uint32_t grid_size = lod_meta.level_sizes[level];
	const uint32_t n_feat_cur_lvl = lod_meta.level_n_feats[level];
	const LoDType lod_type_cur_lvl = (LoDType)lod_meta.level_types[level];
	const InterpolationType interpolation_type = (InterpolationType)lod_meta.interpolation_type;

	uint32_t grid_resolution[N_POS_DIMS];
	COMPUTE_T scale[N_POS_DIMS];
	#pragma unroll
	for (uint32_t dim=0; dim < N_POS_DIMS; ++dim) {
		grid_resolution[dim] = lod_meta.level_res[level][dim];
		scale[dim] = grid_resolution[dim]; // NOTE: for forest
	}

	COMPUTE_T pos[N_POS_DIMS];
	COMPUTE_T pos_derivative[N_POS_DIMS];
	uint32_t pos_grid[N_POS_DIMS];
	positions += i * N_POS_DIMS;
	pos_fract<N_POS_DIMS, INPUT_T, COMPUTE_T>(positions, scale, interpolation_type, pos_grid, pos, pos_derivative);

	dL_dy += i * lod_meta.n_encoded_dims + out_feat_offset;
	vector_t<PARAM_T, N_FEAT_PER_THREAD> grad;
	#pragma unroll
	for (uint32_t f = 0; f < N_FEAT_PER_THREAD; ++f) {
		// grad[f] = dL_dy[i + (out_feat_offset + f) * num_elements];
		grad[f] = dL_dy[f];
	}

	dL_ddLdx += i * N_POS_DIMS;
	vector_t<INPUT_T, N_POS_DIMS> grad_input;
	#pragma unroll
	for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
		grad_input[dim] = dL_ddLdx[dim];
	}

	//---- NOTE: Using switch-case is 2.5x faster than altering function pointers
	switch (lod_type_cur_lvl)
	{
		case LoDType::Dense:
		{
			forest_bwd_input_bwd_grid_n_linear<INPUT_T, PARAM_T, GRAD_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD>(
				add_grid_gridient_dense_impl<PARAM_T, GRAD_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD, std::is_same<GRAD_T, __half>::value>, 
				forest, block_offset, block_idx, block_offsets, block_n_params, scale, pos, pos_derivative, pos_grid, 
				grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, grid, grad, grad_input, grid_gradient);
		}
		break;

		case LoDType::NPlaneMul:
		{
			forest_bwd_input_bwd_grid_n_linear<INPUT_T, PARAM_T, GRAD_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD>(
				add_grid_gridient_nplane_mul_impl<PARAM_T, GRAD_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD, std::is_same<GRAD_T, __half>::value>, 
				forest, block_offset, block_idx, block_offsets, block_n_params, scale, pos, pos_derivative, pos_grid, 
				grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, grid, grad, grad_input, grid_gradient);
		}
		break;

		case LoDType::VectorMatrix:
		{
			forest_bwd_input_bwd_grid_n_linear<INPUT_T, PARAM_T, GRAD_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD>(
				add_grid_gridient_vm_impl<PARAM_T, GRAD_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD, std::is_same<GRAD_T, __half>::value>, 
				forest, block_offset, block_idx, block_offsets, block_n_params, scale, pos, pos_derivative, pos_grid, 
				grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, grid, grad, grad_input, grid_gradient);
		}
		break;

		case LoDType::CP:
		{
			forest_bwd_input_bwd_grid_n_linear<INPUT_T, PARAM_T, GRAD_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD>(
				add_grid_gridient_cp_eq_impl<PARAM_T, GRAD_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD, std::is_same<GRAD_T, __half>::value>, 
				forest, block_offset, block_idx, block_offsets, block_n_params, scale, pos, pos_derivative, pos_grid, 
				grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, grid, grad, grad_input, grid_gradient);
		}
		break;

		case LoDType::Hash:
		{
			forest_bwd_input_bwd_grid_n_linear<INPUT_T, PARAM_T, GRAD_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD>(
				add_grid_gridient_hash_impl<PARAM_T, GRAD_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD, std::is_same<GRAD_T, __half>::value>, 
				forest, block_offset, block_idx, block_offsets, block_n_params, scale, pos, pos_derivative, pos_grid, 
				grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, grid, grad, grad_input, grid_gradient);
		}
		break;
	}
}

template <typename INPUT_T, typename PARAM_T, typename COMPUTE_T, uint32_t N_POS_DIMS, uint32_t N_FEAT, typename F>
__device__ __forceinline__ void forest_bwd_input_bwd_input_n_linear(
	F calc_dLdx_dim,

	const ForestMetaRef& forest,
	const uint32_t block_offset,
	const int16_t block_idx[N_POS_DIMS],
	const int64_t* __restrict__ block_offsets,
	const uint32_t block_n_params,

	const COMPUTE_T scale[N_POS_DIMS],
	const InterpolationType interpolation_type, 
	const COMPUTE_T pos[N_POS_DIMS],
	const COMPUTE_T pos_derivative[N_POS_DIMS], 
	const COMPUTE_T pos_2nd_derivative[N_POS_DIMS], 
	const uint32_t pos_grid[N_POS_DIMS], 
	const uint32_t grid_feat_offset,
	const uint32_t grid_resolution[N_POS_DIMS],
	const uint32_t grid_size,
	const uint32_t n_feat_cur_lvl,
	const PARAM_T* __restrict__ grid, 
	const vector_t<PARAM_T, N_FEAT>&  grad,
	const vector_t<INPUT_T, N_POS_DIMS>& grad_input,
	INPUT_T* __restrict__ grad_result
) {
	auto calc_dLdx = [&](const uint32_t local_pos[N_POS_DIMS], const COMPUTE_T weight) {
		// Continuity fixing for lod-forest-encoding: mapping index
		int16_t block_idx_local[N_POS_DIMS];
		uint32_t local_pos_local[N_POS_DIMS];
		bool block_changed = false;
		
		#pragma unroll
		for (uint32_t dim=0; dim<N_POS_DIMS; ++dim) {
			const uint32_t local_pos_dim = local_pos[dim];
			if (local_pos_dim == 0) {
				// (0) => left-neighbor's (grid_res-1)
				block_idx_local[dim] = block_idx[dim] - 1;
				local_pos_local[dim] = grid_resolution[dim] - 1;
				block_changed = true;
			} else if (local_pos_dim == grid_resolution[dim]+1) {
				// (grid_res+1) => right-neighbor's (0)
				block_idx_local[dim] = block_idx[dim] + 1;
				local_pos_local[dim] = 0;
				block_changed = true;
			} else { 
				// (1,2,3,...,grid_res) => self's (0,1,2,...,grid_res-1)
				block_idx_local[dim] = block_idx[dim];
				local_pos_local[dim] = local_pos_dim-1;
			}
		}

		uint32_t block_offset_local = block_offset;
		if (block_changed) {
			if (!forest.continuity_enabled) return (COMPUTE_T)0.f;
			// Re-check validness & find neighbor block's ind
			int32_t block_ind_local = forest.map_block_ind(block_idx_local);
			if (block_ind_local < 0) return (COMPUTE_T)0.f;
			block_offset_local = block_offsets ? block_offsets[block_ind_local] : (block_ind_local * block_n_params);
		}
		return calc_dLdx_dim(local_pos_local, grid_feat_offset+block_offset_local, grid_resolution, grid_size, n_feat_cur_lvl, grid, (PARAM_T*)&grad, weight);
	};

	vector_t<COMPUTE_T, N_POS_DIMS> grad_in_diag;
	vector_t<COMPUTE_T, N_POS_DIMS> grad_in_other; 

	// From diagonal part of Hessian
	// NOTE: LinearInterpolations' diagonal part is 0.
	if (interpolation_type == InterpolationType::Smoothstep) {
		#pragma unroll
		for (uint32_t grad_dim = 0; grad_dim < N_POS_DIMS; ++grad_dim) {
			grad_in_diag[grad_dim] = (scale[grad_dim] * (COMPUTE_T)grad_input[grad_dim]) * (scale[grad_dim] * pos_2nd_derivative[grad_dim]);
		}
	}

	// From other part of Hessian
	#pragma unroll
	for (uint32_t grad_dim = 0; grad_dim < N_POS_DIMS; ++grad_dim) {
		grad_in_other[grad_dim] = scale[grad_dim] * (COMPUTE_T)grad_input[grad_dim] * pos_derivative[grad_dim]; // * (pos_derivative[other_grad_dim] * scale[other_grad_dim]);
	}

	#pragma unroll 1 // Force skip unrolling; significantly reduce kernel code size & actually increase speed;
	for (uint32_t grad_dim = 0; grad_dim < N_POS_DIMS; ++grad_dim) {
		COMPUTE_T grad_out = 0;
		#pragma unroll
		for (uint32_t idx = 0; idx < (1 << (N_POS_DIMS-1)); ++idx) {
			// From diagonal part of Hessian; d(doutput_d[grad_dim])_d[grad_dim]
			// NOTE: LinearInterpolations' diagonal part is 0.
			if (interpolation_type == InterpolationType::Smoothstep) {
				COMPUTE_T weight_2nd_diag = grad_in_diag[grad_dim];
				uint32_t pos_grid_local[N_POS_DIMS];

				#pragma unroll
				for (uint32_t non_grad_dim = 0; non_grad_dim < N_POS_DIMS-1; ++non_grad_dim) {
					const uint32_t dim = non_grad_dim >= grad_dim ? (non_grad_dim+1) : non_grad_dim;
					// real non_grad_dim
					if ((idx & 1<<non_grad_dim) == 0) {
						weight_2nd_diag *= (COMPUTE_T)1 - pos[dim];
						pos_grid_local[dim] = pos_grid[dim];
					} else {
						weight_2nd_diag *= pos[dim];
						pos_grid_local[dim] = pos_grid[dim] + 1;
					}
				}

				// left
				pos_grid_local[grad_dim] = pos_grid[grad_dim];
				grad_out += calc_dLdx(pos_grid_local, -weight_2nd_diag);
				// right
				pos_grid_local[grad_dim] = pos_grid[grad_dim] + 1;
				grad_out += calc_dLdx(pos_grid_local, weight_2nd_diag);
			}

			// From other part of Hessian; d(doutput_d[real_other_grad_dim])_d[grad_dim]
			// if constexpr (N_POS_DIMS > 1) // NOTE: Valid after c++17 (CUDA>=11)
			ic::if_<(N_POS_DIMS>1)>([&]{
				#pragma unroll
				for (uint32_t other_grad_dim = 0; other_grad_dim < N_POS_DIMS-1; ++other_grad_dim) {
					const uint32_t real_other_grad_dim = other_grad_dim >= grad_dim ? (other_grad_dim+1) : other_grad_dim;
					COMPUTE_T weight_2nd_other = grad_in_other[real_other_grad_dim] * (scale[grad_dim] * pos_derivative[grad_dim]);
					uint32_t pos_grid_local[N_POS_DIMS];

					#pragma unroll
					for (uint32_t non_grad_dim = 0; non_grad_dim < N_POS_DIMS-1; ++non_grad_dim) {
						// real non_grad_dim
						const uint32_t dim = non_grad_dim >= real_other_grad_dim ? (non_grad_dim+1) : non_grad_dim;
						if ((idx & 1<<non_grad_dim) == 0) {
							if (dim != grad_dim) {
								weight_2nd_other *= (COMPUTE_T)1 - pos[dim];
							} else {
								weight_2nd_other *= -1;
							}
							pos_grid_local[dim] = pos_grid[dim];
						} else {
							if (dim != grad_dim) {
								weight_2nd_other *= pos[dim];
							}
							pos_grid_local[dim] = pos_grid[dim] + 1;
						}
					}

					// left
					pos_grid_local[real_other_grad_dim] = pos_grid[real_other_grad_dim];
					grad_out += (COMPUTE_T)calc_dLdx(pos_grid_local, -weight_2nd_other);
					// right
					pos_grid_local[real_other_grad_dim] = pos_grid[real_other_grad_dim] + 1;
					grad_out += (COMPUTE_T)calc_dLdx(pos_grid_local, weight_2nd_other);
				}
			}, []{});
		}

		grad_result[grad_dim] = (INPUT_T)grad_out;
	}
}

template <typename INPUT_T, typename PARAM_T, typename COMPUTE_T, uint32_t N_POS_DIMS, uint32_t N_FEAT_PER_PSEUDO_LVL, uint32_t N_FEAT_PER_THREAD>
__global__ 
// typename std::enable_if<N_POS_DIMS==3, void>::type
typename std::enable_if<N_POS_DIMS==3 && std::is_same<INPUT_T, float>::value, void>::type
kernel_lod_forest_backward_input_backward_input(
	const uint32_t num_elements,
	const LoDMetaRef lod_meta,
	const ForestMetaRef forest,
	int32_t max_level,
	const int32_t* __restrict__ max_level_gpu,
	// inputs
	const INPUT_T* __restrict__ dL_ddLdx,      // [n_points, 3]
	const PARAM_T* __restrict__ dL_dy,         // [n_points, n_lod_features]
	const PARAM_T* __restrict__ grid,          // [n_params]
	const INPUT_T* __restrict__ positions,  // [n_points, 3]
	// Optional inputs for multi-batch data
	const int64_t* __restrict__ block_inds,    // [n_points]
	const int64_t* __restrict__ block_offsets, // [n_batch]
	const uint32_t batch_data_size,
	// outputs
	INPUT_T* dL_dx
) {
	const uint32_t i = ((blockIdx.x * blockDim.x + threadIdx.x) * N_FEAT_PER_THREAD) / N_FEAT_PER_PSEUDO_LVL;
	if (i >= num_elements) return;

	const uint32_t pseudo_lvl = blockIdx.y;
	const uint32_t level = lod_meta.map_levels[pseudo_lvl];
	const uint32_t feature = (blockIdx.x * blockDim.x + threadIdx.x) * N_FEAT_PER_THREAD - i * N_FEAT_PER_PSEUDO_LVL;
	const uint32_t grid_feat_offset = lod_meta.map_cnt[pseudo_lvl] * N_FEAT_PER_PSEUDO_LVL + feature;
	const uint32_t out_feat_offset = pseudo_lvl * N_FEAT_PER_PSEUDO_LVL + feature;

	if (max_level_gpu) {
		max_level = max_level_gpu[i];
	}

	if (level > max_level) {
		return;
	}

	// For batched
	uint32_t block_ind = 0;
	if (block_inds) {
		// NOTE: pass in block_ind=-1 to ignore certain points.
		if (block_inds[i] < 0) return;
		block_ind = block_inds[i];
	} else if (batch_data_size) {
		block_ind = i / batch_data_size;
	}
	const uint32_t block_n_params = lod_meta.level_offsets[lod_meta.n_levels];
	const uint32_t block_offset = block_offsets ? block_offsets[block_ind] : block_ind * block_n_params;
	const short3 block_k = forest.block_ks[block_ind];
	const int16_t block_idx[N_POS_DIMS] = {block_k.x, block_k.y, block_k.z};

	grid += lod_meta.level_offsets[level];

	const uint32_t grid_size = lod_meta.level_sizes[level];
	const uint32_t n_feat_cur_lvl = lod_meta.level_n_feats[level];
	const LoDType lod_type_cur_lvl = (LoDType)lod_meta.level_types[level];
	const InterpolationType interpolation_type = (InterpolationType)lod_meta.interpolation_type;

	uint32_t grid_resolution[N_POS_DIMS];
	COMPUTE_T scale[N_POS_DIMS];
	#pragma unroll
	for (uint32_t dim=0; dim < N_POS_DIMS; ++dim) {
		grid_resolution[dim] = lod_meta.level_res[level][dim];
		scale[dim] = grid_resolution[dim]; // NOTE: for forest
	}

	COMPUTE_T pos[N_POS_DIMS];
	COMPUTE_T pos_derivative[N_POS_DIMS];
	COMPUTE_T pos_2nd_derivative[N_POS_DIMS];
	uint32_t pos_grid[N_POS_DIMS];
	positions += i * N_POS_DIMS;
	pos_fract<N_POS_DIMS, INPUT_T, COMPUTE_T>(positions, scale, interpolation_type, pos_grid, pos, pos_derivative, pos_2nd_derivative);

	dL_dy += i * lod_meta.n_encoded_dims + out_feat_offset;
	vector_t<PARAM_T, N_FEAT_PER_THREAD> grad;
	#pragma unroll
	for (uint32_t f = 0; f < N_FEAT_PER_THREAD; ++f) {
		// grad[f] = dL_dy[i + (out_feat_offset + f) * num_elements];
		grad[f] = dL_dy[f];
	}

	dL_ddLdx += i * N_POS_DIMS;
	vector_t<INPUT_T, N_POS_DIMS> grad_input;
	#pragma unroll
	for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
		grad_input[dim] = dL_ddLdx[dim];
	}

	vector_t<INPUT_T, N_POS_DIMS> grad_result = {};

	//---- NOTE: Using switch-case is 2.5x faster than altering function pointers
	switch (lod_type_cur_lvl)
	{
		case LoDType::Dense:
		{
			forest_bwd_input_bwd_input_n_linear<INPUT_T, PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD>(
				calc_dLdx_dim_dense_impl<PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD>, 
				forest, block_offset, block_idx, block_offsets, block_n_params, scale, interpolation_type, pos, pos_derivative, pos_2nd_derivative, pos_grid, 
				grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, grid, grad, grad_input, (INPUT_T*)&grad_result);
		}
		break;

		case LoDType::VectorMatrix:
		{
			forest_bwd_input_bwd_input_n_linear<INPUT_T, PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD>(
				calc_dLdx_dim_vm_impl<PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD>, 
				forest, block_offset, block_idx, block_offsets, block_n_params, scale, interpolation_type, pos, pos_derivative, pos_2nd_derivative, pos_grid, 
				grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, grid, grad, grad_input, (INPUT_T*)&grad_result);
		}
		break;

		case LoDType::Hash:
		{
			forest_bwd_input_bwd_input_n_linear<INPUT_T, PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD>(
				calc_dLdx_dim_hash_impl<PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD>, 
				forest, block_offset, block_idx, block_offsets, block_n_params, scale, interpolation_type, pos, pos_derivative, pos_2nd_derivative, pos_grid, 
				grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, grid, grad, grad_input, (INPUT_T*)&grad_result);
		}
		break;
	}

	dL_dx += i * N_POS_DIMS;

	#pragma unroll
	for (uint32_t dim=0; dim < N_POS_DIMS; ++dim) {
		atomicAdd((INPUT_T*)(dL_dx + dim), (INPUT_T)grad_result[dim]);
	}

	// using input_t = typename std::conditional<std::is_same<INPUT_T, __half>::value, at::Half, INPUT_T>::type;
	// #pragma unroll
	// for (uint32_t dim=0; dim < N_POS_DIMS; ++dim) {
	// 	gpuAtomicAdd((input_t*)(dL_dx + dim), (input_t)grad_result[dim]);
	// }
}


template <typename INPUT_T, typename PARAM_T, typename COMPUTE_T, uint32_t N_POS_DIMS, uint32_t N_FEAT_PER_PSEUDO_LVL>
inline void lod_forest_fwd_impl_dispatched(
	LoDMeta& lod_meta,
	ForestMeta& forest_meta,
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

	static constexpr uint32_t n_threads = 512;
	// static constexpr uint32_t n_threads = 128;
	const dim3 blocks_lod = { div_round_up(batch_size, n_threads), lod_meta.n_pseudo_levels, 1 };

	const at::cuda::OptionalCUDAGuard device_guard(at::device_of(params));
	cudaStream_t stream = at::cuda::getCurrentCUDAStream();

	kernel_lod_forest<INPUT_T, PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL><<<blocks_lod, n_threads, 0, stream>>>(
		batch_size, 
		{lod_meta}, 
		{forest_meta}, 
		max_level, (int32_t*)nullptr, 
		data_ptr<PARAM_T>(params), 
		data_ptr<INPUT_T>(input), 
		batch_inds_.has_value() ? batch_inds_.value().data_ptr<int64_t>() : nullptr, 
		batch_offsets_.has_value() ? batch_offsets_.value().data_ptr<int64_t>() : nullptr, 
		batch_data_size, 
		data_ptr<PARAM_T>(output),
		need_input_grad ? data_ptr<INPUT_T>(dy_dx) : nullptr
	);
}

template <uint32_t N_POS_DIMS, uint32_t N_FEAT_PER_PSEUDO_LVL>
inline void lod_forest_fwd_impl_templated(
	LoDMeta& lod_meta,
	ForestMeta& forest_meta,
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
	auto fn = (input.scalar_type() == at::kHalf && params.scalar_type() == at::kHalf) ? lod_forest_fwd_impl_dispatched<__half, __half, __half, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL>
			: (input.scalar_type() == at::kFloat && params.scalar_type() == at::kHalf) ? lod_forest_fwd_impl_dispatched<float, __half, float, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL>
			: (input.scalar_type() == at::kFloat && params.scalar_type() == at::kFloat) ? lod_forest_fwd_impl_dispatched<float, float, float, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL>
			: nullptr;
	if (!fn) {
		throw std::runtime_error("LoTDEncoding: Input type combination not supported. Supported types are: <input,param> -> (half, half), (float, half), (float, float)");
	}
	fn(lod_meta, forest_meta, input, params, batch_inds_, batch_offsets_, batch_data_size, max_level, need_input_grad, output, dy_dx); 
}

template <uint32_t N_POS_DIMS>
typename std::enable_if<N_POS_DIMS==3, void>::type 
lod_forest_fwd_impl(
	LoDMeta& lod_meta,
	ForestMeta& forest_meta,
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
	auto fn = (lod_meta.n_feat_per_pseudo_lvl == 2) ? lod_forest_fwd_impl_templated<N_POS_DIMS, 2>
			: (lod_meta.n_feat_per_pseudo_lvl == 4) ? lod_forest_fwd_impl_templated<N_POS_DIMS, 4>
			: (lod_meta.n_feat_per_pseudo_lvl == 8) ? lod_forest_fwd_impl_templated<N_POS_DIMS, 8>
			: nullptr; 
	if (!fn) {
		throw std::runtime_error("LoTDEncoding: `n_feat_per_pseudo_lvl` must be one of [2,4,8]");
	}
	fn(lod_meta, forest_meta, input, params, batch_inds_, batch_offsets_, batch_data_size, max_level, need_input_grad, output, dy_dx); 
}

template <typename INPUT_T, typename PARAM_T, typename COMPUTE_T,  uint32_t N_POS_DIMS, uint32_t N_FEAT_PER_PSEUDO_LVL>
inline void lod_forest_bwd_impl_dispatched(
	LoDMeta& lod_meta, 
	ForestMeta& forest_meta,
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
		static constexpr uint32_t N_THREADS_LOD = 256;
		static constexpr uint32_t N_FEAT_PER_THREAD = std::min(2u, N_FEAT_PER_PSEUDO_LVL);
		const dim3 blocks_lod = { div_round_up(batch_size * N_FEAT_PER_PSEUDO_LVL / N_FEAT_PER_THREAD, N_THREADS_LOD), lod_meta.n_pseudo_levels, 1 };

		const at::cuda::OptionalCUDAGuard device_guard(at::device_of(params));
		cudaStream_t stream = at::cuda::getCurrentCUDAStream();

		kernel_lod_forest_backward_grid<INPUT_T, PARAM_T, PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL, N_FEAT_PER_THREAD><<<blocks_lod, N_THREADS_LOD, 0, stream>>> (
			batch_size, 
			{lod_meta}, 
			{forest_meta},
			max_level, (int32_t*)nullptr,
			data_ptr<PARAM_T>(dL_dy), 
			data_ptr<PARAM_T>(params), 
			data_ptr<INPUT_T>(input), 
			batch_inds_.has_value() ? batch_inds_.value().data_ptr<int64_t>() : nullptr, 
			batch_offsets_.has_value() ? batch_offsets_.value().data_ptr<int64_t>() : nullptr, 
			batch_data_size, 
			data_ptr<PARAM_T>(dL_dparam)
		);
	}
}

template <uint32_t N_POS_DIMS, uint32_t N_FEAT_PER_PSEUDO_LVL>
inline void lod_forest_bwd_impl_templated(
	LoDMeta& lod_meta, 
	ForestMeta& forest_meta,
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
	auto fn = (input.scalar_type() == at::kHalf && params.scalar_type() == at::kHalf) ? lod_forest_bwd_impl_dispatched<__half, __half, __half, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL>
			: (input.scalar_type() == at::kFloat && params.scalar_type() == at::kHalf) ? lod_forest_bwd_impl_dispatched<float, __half, float, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL>
			: (input.scalar_type() == at::kFloat && params.scalar_type() == at::kFloat) ? lod_forest_bwd_impl_dispatched<float, float, float, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL>
			: nullptr;
	if (!fn) {
		throw std::runtime_error("LoTDEncoding: Input type combination not supported. Supported types are: <input,param> -> (half, half), (float, half), (float, float)");
	}
	fn(lod_meta, forest_meta, dL_dy, input, params, dy_dx_, batch_inds_, batch_offsets_, batch_data_size, max_level, need_input_grad, need_param_grad, dL_dx, dL_dparam); 
}

template <uint32_t N_POS_DIMS>
typename std::enable_if<N_POS_DIMS==3, void>::type 
lod_forest_bwd_impl(
	LoDMeta& lod_meta, 
	ForestMeta& forest_meta,
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
	auto fn = (lod_meta.n_feat_per_pseudo_lvl == 2) ? lod_forest_bwd_impl_templated<N_POS_DIMS, 2>
			: (lod_meta.n_feat_per_pseudo_lvl == 4) ? lod_forest_bwd_impl_templated<N_POS_DIMS, 4>
			: (lod_meta.n_feat_per_pseudo_lvl == 8) ? lod_forest_bwd_impl_templated<N_POS_DIMS, 8>
			: nullptr; 
	if (!fn) {
		throw std::runtime_error("LoTDEncoding: `n_feat_per_pseudo_lvl` must be one of [2,4,8]");
	}
	fn(lod_meta, forest_meta, dL_dy, input, params, dy_dx_, batch_inds_, batch_offsets_, batch_data_size, max_level, need_input_grad, need_param_grad, dL_dx, dL_dparam); 
}

template <typename INPUT_T, typename PARAM_T, typename COMPUTE_T, uint32_t N_POS_DIMS, uint32_t N_FEAT_PER_PSEUDO_LVL>
inline void lod_forest_bwd_bwd_input_impl_dispatched(
	LoDMeta& lod_meta, 
	ForestMeta& forest_meta,
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
		static constexpr uint32_t N_THREADS_LOD = 256;
		static constexpr uint32_t N_FEAT_PER_THREAD = std::min(2u, N_FEAT_PER_PSEUDO_LVL);
		const dim3 blocks_lod = { div_round_up(batch_size * N_FEAT_PER_PSEUDO_LVL / N_FEAT_PER_THREAD, N_THREADS_LOD), lod_meta.n_pseudo_levels, 1 };

		const at::cuda::OptionalCUDAGuard device_guard(at::device_of(params));
		cudaStream_t stream = at::cuda::getCurrentCUDAStream();

		// Safer to use COMPUTE_T=float
		kernel_lod_forest_backward_input_backward_input<INPUT_T, PARAM_T, float, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL, N_FEAT_PER_THREAD><<<blocks_lod, N_THREADS_LOD, 0, stream>>>(
			batch_size, 
			{lod_meta}, 
			{forest_meta},
			max_level, (int32_t*)nullptr, 
			data_ptr<INPUT_T>(dL_ddLdx), 
			data_ptr<PARAM_T>(dL_dy), 
			data_ptr<PARAM_T>(params), 
			data_ptr<INPUT_T>(input), 
			batch_inds_.has_value() ? batch_inds_.value().data_ptr<int64_t>() : nullptr, 
			batch_offsets_.has_value() ? batch_offsets_.value().data_ptr<int64_t>() : nullptr, 
			batch_data_size, 
			data_ptr<INPUT_T>(dL_dx)
		);
	}

	if (need_param_grad) {
		static constexpr uint32_t N_THREADS_LOD = 256;
		static constexpr uint32_t N_FEAT_PER_THREAD = std::min(2u, N_FEAT_PER_PSEUDO_LVL);
		const dim3 blocks_lod = { div_round_up(batch_size * N_FEAT_PER_PSEUDO_LVL / N_FEAT_PER_THREAD, N_THREADS_LOD), lod_meta.n_pseudo_levels, 1 };

		const at::cuda::OptionalCUDAGuard device_guard(at::device_of(params));
		cudaStream_t stream = at::cuda::getCurrentCUDAStream();

		kernel_lod_forest_backward_input_backward_grid<INPUT_T, PARAM_T, PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL, N_FEAT_PER_THREAD><<<blocks_lod, N_THREADS_LOD, 0, stream>>>(
			batch_size, 
			{lod_meta}, 
			{forest_meta},
			max_level, (int32_t*)nullptr, 
			data_ptr<INPUT_T>(dL_ddLdx), 
			data_ptr<PARAM_T>(dL_dy), 
			data_ptr<PARAM_T>(params), 
			data_ptr<INPUT_T>(input), 
			batch_inds_.has_value() ? batch_inds_.value().data_ptr<int64_t>() : nullptr, 
			batch_offsets_.has_value() ? batch_offsets_.value().data_ptr<int64_t>() : nullptr, 
			batch_data_size, 
			data_ptr<PARAM_T>(dL_dparams)
		);
	}
}

template <uint32_t N_POS_DIMS, uint32_t N_FEAT_PER_PSEUDO_LVL>
inline void lod_forest_bwd_bwd_input_impl_templated(
	LoDMeta& lod_meta, 
	ForestMeta& forest_meta,
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
	auto fn = (input.scalar_type() == at::kFloat && params.scalar_type() == at::kHalf) ? lod_forest_bwd_bwd_input_impl_dispatched<float, __half, float, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL>
			: (input.scalar_type() == at::kFloat && params.scalar_type() == at::kFloat) ? lod_forest_bwd_bwd_input_impl_dispatched<float, float, float, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL>
			: nullptr; 
	if (!fn) {
		throw std::runtime_error("LoTDEncoding: Input type combination not supported. Supported types are: <input,param> -> (half, half), (float, half), (float, float)");
	}
	fn(lod_meta, forest_meta, dL_ddLdx, dL_dy, input, params, dy_dx_, batch_inds_, batch_offsets_, batch_data_size, max_level, need_dLdy_grad, need_input_grad, need_param_grad, dL_ddLdy, dL_dx, dL_dparams); 
}

template <uint32_t N_POS_DIMS>
typename std::enable_if<N_POS_DIMS==3, void>::type 
lod_forest_bwd_bwd_input_impl(
	LoDMeta& lod_meta, 
	ForestMeta& forest_meta,
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
	auto fn = (lod_meta.n_feat_per_pseudo_lvl == 2) ?  lod_forest_bwd_bwd_input_impl_templated<N_POS_DIMS, 2>
			: (lod_meta.n_feat_per_pseudo_lvl == 4) ?  lod_forest_bwd_bwd_input_impl_templated<N_POS_DIMS, 4>
			: (lod_meta.n_feat_per_pseudo_lvl == 8) ?  lod_forest_bwd_bwd_input_impl_templated<N_POS_DIMS, 8>
			: nullptr; 
	if (!fn) {
		throw std::runtime_error("LoTDEncoding: `n_feat_per_pseudo_lvl` must be one of [2,4,8]");
	}
	fn(lod_meta, forest_meta, dL_ddLdx, dL_dy, input, params, dy_dx_, batch_inds_, batch_offsets_, batch_data_size, max_level, need_dLdy_grad, need_input_grad, need_param_grad, dL_ddLdy, dL_dx, dL_dparams); 
}


} // namespace lotd::torch

} // namespace lotd
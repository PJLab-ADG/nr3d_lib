/** @file   lotd_encoding.h
 *  @author Jianfei Guo, Shanghai AI Lab
 *  @brief  LoTD-Encoding; Pytorch-CUDA basd implementation.
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

#include "lotd.h"
#include "lotd_torch_api.h"
#include "if_constexpr.hpp" // pre-c++17 constexpr if

namespace lotd {
namespace torch {

static constexpr uint32_t MAX_N_LEVELS = 32;
static constexpr uint32_t MAX_N_POS_DIMS = 4;

struct LoDMetaRef {
	uint32_t level_res[MAX_N_LEVELS][MAX_N_POS_DIMS] = {};
	uint32_t level_n_feats[MAX_N_LEVELS];
	LoDType level_types[MAX_N_LEVELS];
	uint32_t level_n_params[MAX_N_LEVELS];
	uint32_t level_sizes[MAX_N_LEVELS];
	uint32_t level_offsets[MAX_N_LEVELS+1];

	uint32_t map_levels[MAX_N_LEVELS * 8];
	uint32_t map_cnt[MAX_N_LEVELS * 8];

	uint32_t n_levels = 0;
	uint32_t n_pseudo_levels = 0; 
	uint32_t n_feat_per_pseudo_lvl = 0;
	uint32_t n_dims_to_encode = 3;
	uint32_t n_encoded_dims = 0;

	InterpolationType interpolation_type = InterpolationType::Linear;

	__host__ LoDMetaRef(const LoDMeta& meta): 
		n_levels{meta.n_levels}, n_pseudo_levels{meta.n_pseudo_levels}, n_feat_per_pseudo_lvl{meta.n_feat_per_pseudo_lvl}, 
		n_dims_to_encode{meta.n_dims_to_encode}, n_encoded_dims{meta.n_encoded_dims}, interpolation_type{meta.interpolation_type} {
			for (uint32_t l=0; l<meta.n_levels; ++l) {
				for (uint32_t dim=0; dim < meta.n_dims_to_encode; ++dim) {
					level_res[l][dim] = meta.level_res_multi_dim[l][dim];
				}
				// level_res[l] = meta.level_res[l];
				level_n_feats[l] = meta.level_n_feats[l];
				level_types[l] = (LoDType)meta.level_types[l];
				level_n_params[l] = meta.level_n_params[l];
				level_sizes[l] = meta.level_sizes[l];
				level_offsets[l] = meta.level_offsets[l];
			}
			level_offsets[meta.n_levels] = meta.level_offsets[meta.n_levels];
			for (uint32_t psl=0; psl<meta.n_pseudo_levels; ++psl) {
				map_levels[psl] = meta.map_levels[psl];
				map_cnt[psl] = meta.map_cnt[psl];
			}
		};
};

template <typename INPUT_T, typename PARAM_T, typename COMPUTE_T, uint32_t N_POS_DIMS, uint32_t N_FEAT, typename F>
__device__ __forceinline__ void fwd_n_linear(
    F get_grid_val,
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
		get_grid_val(local_pos, grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, grid, (PARAM_T*)&val);
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
					// printf("dydx[%d][%d] += weight(%f) * ( val_right(%f) - val_left(%f) ) = %f\n", f, grad_dim, (float)weight, (float)val_right[f], (float)val_left[f], (float)grads_ptr[f][grad_dim]);
				}
			}
		}
	}
}

template <typename INPUT_T, typename PARAM_T, typename COMPUTE_T, uint32_t N_POS_DIMS, uint32_t N_FEAT_PER_PSEUDO_LVL>
__global__ void kernel_lod(
	const uint32_t num_elements,
	const uint32_t num_lod_features, 
	const uint32_t num_levels, 
	const LoDMetaRef lod_meta,
	int32_t max_level,
	const int32_t* __restrict__ max_level_gpu,   // [n_points]
	// inputs
	const PARAM_T* __restrict__ grid,          // [n_params]
	const INPUT_T* __restrict__ positions_in,  // [n_points, 3]
	// Optional inputs for multi-batch data
	const int64_t* __restrict__ batch_inds,    // [n_points]
	const int64_t* __restrict__ batch_offsets, // [n_batch]
	const uint32_t batch_data_size,
	// outputs
	PARAM_T* __restrict__ encoded_positions,
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
		if (encoded_positions) {
			#pragma unroll
			for (uint32_t f = 0; f < N_FEAT_PER_PSEUDO_LVL; ++f) {
				// encoded_positions[i + (out_feat_offset + f) * num_elements] = (PARAM_T)0.0f;
				encoded_positions[i * num_lod_features + out_feat_offset + f] = (PARAM_T)0.0f;
			}
		}

		// Gradient is zero for zeroed-out dimensions.
		if (dy_dx) {
			#pragma unroll
			for (uint32_t f = 0; f < N_FEAT_PER_PSEUDO_LVL; ++f) {
				// ((vector_t<INPUT_T, N_POS_DIMS>*)dy_dx)[i + (out_feat_offset + f) * num_elements] = {0};
				((vector_t<INPUT_T, N_POS_DIMS>*)dy_dx)[i * num_lod_features + out_feat_offset + f] = {};
			}
		}
	};

	if (level > max_level) {
		set_zero();
		return;
	}

	// For batched
	uint32_t batch_ind = 0;
	if (batch_inds) {
		// NOTE: pass in batch_ind=-1 to ignore certain points.
		if (batch_inds[i] < 0) { set_zero(); return;}
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
	for (uint32_t dim=0; dim < N_POS_DIMS; ++dim) {
		grid_resolution[dim] = lod_meta.level_res[level][dim];
		scale[dim] = grid_resolution[dim] - 2;
	}
	COMPUTE_T pos[N_POS_DIMS];
	COMPUTE_T pos_derivative[N_POS_DIMS];
	uint32_t pos_grid[N_POS_DIMS];

	positions_in += i * N_POS_DIMS;
	if (dy_dx) {
		pos_fract<N_POS_DIMS, INPUT_T, COMPUTE_T>(positions_in, scale, interpolation_type, pos_grid, pos, pos_derivative);
	} else {
		pos_fract<N_POS_DIMS, INPUT_T, COMPUTE_T>(positions_in, scale, interpolation_type, pos_grid, pos);
	}

	// PARAM_T* result_ptr = encoded_positions ? (encoded_positions + i * num_lod_features + out_feat_offset) : nullptr;
	vector_t<PARAM_T, N_FEAT_PER_PSEUDO_LVL> result = {};
	PARAM_T* result_ptr = encoded_positions ? (PARAM_T*)&result : nullptr;

	// vector_t<INPUT_T, N_POS_DIMS>* grads_ptr = dy_dx ? ( ((vector_t<INPUT_T, N_POS_DIMS>*)dy_dx) + i * num_lod_features + out_feat_offset ) : nullptr;
	vector_t<INPUT_T, N_POS_DIMS> grads[N_FEAT_PER_PSEUDO_LVL] = {};
	vector_t<INPUT_T, N_POS_DIMS>* grads_ptr = dy_dx ? grads : nullptr;
	
	void (*get_grid_val)(const uint32_t[N_POS_DIMS], const uint32_t, const uint32_t[N_POS_DIMS], const uint32_t, const uint32_t, const PARAM_T*, PARAM_T*);

	switch (lod_type_cur_lvl)
	{
		case LoDType::Dense:
		{
			get_grid_val = grid_val_dense_impl<PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL>;
			fwd_n_linear<INPUT_T, PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL>(get_grid_val, scale, pos, pos_derivative, pos_grid, grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, grid, result_ptr, grads_ptr);
		}
		break;

		case LoDType::VectorMatrix:
		{
			ic::if_<N_POS_DIMS==3>([&]{
				get_grid_val = grid_val_vm_impl<PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL>;
				fwd_n_linear<INPUT_T, PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL>(get_grid_val, scale, pos, pos_derivative, pos_grid, grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, grid, result_ptr, grads_ptr);
			}, []{});
			// if constexpr (N_POS_DIMS==3) // NOTE: Valid after c++17 (CUDA>=11)
		}
		break;

		case LoDType::VecZMatXoY:
		{
			ic::if_<N_POS_DIMS==3>([&]{
				get_grid_val = grid_val_vec_z_mat_xoy_impl<PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL>;
				fwd_n_linear<INPUT_T, PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL>(get_grid_val, scale, pos, pos_derivative, pos_grid, grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, grid, result_ptr, grads_ptr);
			}, []{});
			// if constexpr (N_POS_DIMS==3) // NOTE: Valid after c++17 (CUDA>=11)
		}
		break;

		case LoDType::NPlaneMul:
		{
			get_grid_val = grid_val_nplane_mul_impl<PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL>;
			fwd_n_linear<INPUT_T, PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL>(get_grid_val, scale, pos, pos_derivative, pos_grid, grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, grid, result_ptr, grads_ptr);
		}
		break;

		case LoDType::CP:
		{
			get_grid_val = grid_val_cp_eq_impl<PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL>;
			fwd_n_linear<INPUT_T, PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL>(get_grid_val, scale, pos, pos_derivative, pos_grid, grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, grid, result_ptr, grads_ptr);
		}
		break;

		case LoDType::Hash:
		{
			get_grid_val = grid_val_hash_impl<PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL>;
			fwd_n_linear<INPUT_T, PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL>(get_grid_val, scale, pos, pos_derivative, pos_grid, grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, grid, result_ptr, grads_ptr);
		}
		break;

		case LoDType::NPlaneSum:
		{
			// if constexpr (N_POS_DIMS > 2) // NOTE: Valid after c++17 (CUDA>=11)
			ic::if_<(N_POS_DIMS>2)>([&]{
				auto plane_val = [&](const uint32_t local_plane_pos[N_POS_DIMS-1], const uint32_t jump_dim) {
					uint32_t index = grid_index_nplane_sub<N_POS_DIMS>(grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, local_plane_pos, jump_dim);
					return *(vector_t<PARAM_T, N_FEAT_PER_PSEUDO_LVL>*)&grid[index];
				};

				if (encoded_positions) {
					COMPUTE_T wplane = 1; // Or, set to 1.0f / N_POS_DIMS
					#pragma unroll 1 // Force skip unrolling; significantly reduce kernel code size & actually increase speed;
					for (uint32_t jump_dim=0; jump_dim < N_POS_DIMS; ++jump_dim) {
						
						// (N-1) - linear interpolation
						#pragma unroll
						for (uint32_t idx = 0; idx < (1 << (N_POS_DIMS-1)); ++idx) {
							COMPUTE_T weight = wplane;
							uint32_t pos_plane_local[N_POS_DIMS-1];
							
							#pragma unroll
							for (uint32_t dim2D=0; dim2D < N_POS_DIMS-1; ++dim2D) {
								uint32_t dim3D = dim2D >= jump_dim ? (dim2D+1) : dim2D;
								if ((idx & (1<<dim2D)) == 0) {
									weight *= (COMPUTE_T)1 - pos[dim3D];
									pos_plane_local[dim2D] = pos_grid[dim3D];
								} else {
									weight *= pos[dim3D];
									pos_plane_local[dim2D] = pos_grid[dim3D] + 1;
								}
							}

							auto val = plane_val(pos_plane_local, jump_dim);
							#pragma unroll
							for (uint32_t f = 0; f < N_FEAT_PER_PSEUDO_LVL; ++f) {
								((PARAM_T*)&result)[f] += (PARAM_T)(weight * (COMPUTE_T)((PARAM_T*)&val)[f]);
							}
						}
					}
				}

				if (dy_dx) {
					#pragma unroll 1 // Force skip unrolling; significantly reduce kernel code size & actually increase speed;
					for (uint32_t jump_dim = 0; jump_dim < N_POS_DIMS; ++jump_dim) {
						
						#pragma unroll
						for (uint32_t grad_dim_2D = 0; grad_dim_2D < N_POS_DIMS-1; ++grad_dim_2D) {
							const uint32_t grad_dim_3D = grad_dim_2D >= jump_dim ? (grad_dim_2D+1) : grad_dim_2D;

							#pragma unroll
							for (uint32_t idx=0; idx < (1<<(N_POS_DIMS-2)); ++idx) {

								COMPUTE_T weight = scale[grad_dim_3D] * pos_derivative[grad_dim_3D]; // divide by N_POS_DIMS ?
								uint32_t pos_plane_local[N_POS_DIMS-1];

								#pragma unroll
								for (int32_t non_grad_dim = 0; non_grad_dim < N_POS_DIMS-2; ++non_grad_dim) {
									const uint32_t dim2D = non_grad_dim >= grad_dim_2D ? (non_grad_dim + 1) : non_grad_dim;
									const uint32_t dim3D = dim2D >= jump_dim ? (dim2D+1) : dim2D;
									if ((idx & 1<<non_grad_dim) == 0) {
										weight *= (COMPUTE_T)1 - pos[dim3D];
										pos_plane_local[dim2D] = pos_grid[dim3D];
									} else {
										weight *= pos[dim3D];
										pos_plane_local[dim2D] = pos_grid[dim3D] + 1;
									}
								}

								pos_plane_local[grad_dim_2D] = pos_grid[grad_dim_3D];
								auto val_left = plane_val(pos_plane_local, jump_dim);
								pos_plane_local[grad_dim_2D] = pos_grid[grad_dim_3D] + 1;
								auto val_right = plane_val(pos_plane_local, jump_dim);

								#pragma unroll
								for (uint32_t f = 0; f < N_FEAT_PER_PSEUDO_LVL; ++f) {
									grads[f][grad_dim_3D] += weight * ((COMPUTE_T)val_right[f] - (COMPUTE_T)val_left[f]);
								}
							}
						}
					}
				}
			}, []{});
		}
		break;

		case LoDType::CPfast:
		{
			auto line_val = [&](const uint32_t local_line_pos, const uint32_t line_dim) {
				uint32_t index = grid_index_cp_line<N_POS_DIMS>(grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, local_line_pos, line_dim);
				return *(vector_t<PARAM_T, N_FEAT_PER_PSEUDO_LVL>*)&grid[index];
			};

			if (encoded_positions) {
				vector_t<COMPUTE_T, N_FEAT_PER_PSEUDO_LVL> result_;
				#pragma unroll
				for (uint32_t f=0; f < N_FEAT_PER_PSEUDO_LVL; ++f) ((COMPUTE_T*)&result_)[f] = (COMPUTE_T)1.0f;

				#pragma unroll 1 // Force skip unrolling; significantly reduce kernel code size & actually increase speed;
				for (uint32_t line_dim=0; line_dim<N_POS_DIMS; ++line_dim) {
					// Linear-interpolation
					auto val_left = line_val(pos_grid[line_dim], line_dim);
					auto val_right = line_val(pos_grid[line_dim]+1, line_dim);
					COMPUTE_T weight = pos[line_dim];

					#pragma unroll
					for (uint32_t f=0; f < N_FEAT_PER_PSEUDO_LVL; ++f) {
						((COMPUTE_T*)&result_)[f] *= ((COMPUTE_T)1-weight) * (COMPUTE_T)val_left[f] + weight * (COMPUTE_T)val_right[f];
					}
				}
				#pragma unroll
				for (uint32_t f=0; f<N_FEAT_PER_PSEUDO_LVL; ++f) {
					((PARAM_T*)&result)[f] = ((COMPUTE_T*)&result_)[f];
				}
			}

			if (dy_dx) {
				#pragma unroll 1 // Force skip unrolling; significantly reduce kernel code size & actually increase speed;
				for (uint32_t grad_dim=0; grad_dim < N_POS_DIMS; ++grad_dim) {
					COMPUTE_T weight = scale[grad_dim] * pos_derivative[grad_dim];
					auto val_left = line_val(pos_grid[grad_dim], grad_dim);
					auto val_right = line_val(pos_grid[grad_dim]+1, grad_dim);
					
					#pragma unroll
					for (uint32_t f=0; f<N_FEAT_PER_PSEUDO_LVL; ++f) {
						grads[f][grad_dim] = weight * ((COMPUTE_T)val_right[f] - (COMPUTE_T)val_left[f]);
					}

					#pragma unroll
					for (uint32_t non_grad_dim=0; non_grad_dim < N_POS_DIMS - 1; ++non_grad_dim) {
						const uint32_t dim = non_grad_dim >= grad_dim ? (non_grad_dim+1) : non_grad_dim;
						auto ng_val_left = line_val(pos_grid[dim], dim);
						auto ng_val_right = line_val(pos_grid[dim]+1, dim);
						COMPUTE_T ng_weight = pos[dim];
						
						#pragma unroll
						for (uint32_t f=0; f<N_FEAT_PER_PSEUDO_LVL; ++f) {
							grads[f][grad_dim] *= ((COMPUTE_T)1-ng_weight) * (COMPUTE_T)ng_val_left[f] + ng_weight * (COMPUTE_T)ng_val_right[f];
						}
					}
				}
			}
		}
		break;
	}

	if (encoded_positions) {
		encoded_positions += i * num_lod_features + out_feat_offset;
		#pragma unroll
		for (uint32_t f=0; f<N_FEAT_PER_PSEUDO_LVL; ++f) {
			encoded_positions[f] = ((PARAM_T*)&result)[f];
		}
	}

	if (dy_dx) {
		dy_dx += N_POS_DIMS * (i * num_lod_features + out_feat_offset);
		#pragma unroll
		for (uint32_t f=0; f<N_FEAT_PER_PSEUDO_LVL; ++f) {
			((vector_t<INPUT_T, N_POS_DIMS>*)dy_dx)[f] = grads[f];
		}
	}
}

template <typename PARAM_T, typename GRAD_T, typename COMPUTE_T, uint32_t N_POS_DIMS, uint32_t N_FEAT, typename F>
__device__ __forceinline__ void bwd_n_linear(
	F add_grid_grad, 
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
		add_grid_grad(local_pos, grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, grid, (PARAM_T*)&grad, weight, grid_gradient);
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
__global__ void kernel_lod_backward(
	const uint32_t num_elements,
	const uint32_t num_lod_features, 
	const uint32_t num_levels, 
	const LoDMetaRef lod_meta,
	int32_t max_level,
	const int32_t* __restrict__ max_level_gpu,
	// inputs
	const PARAM_T* __restrict__ dL_dy,
	const PARAM_T* __restrict__ grid,
	const INPUT_T* __restrict__ positions_in,
	// Optional inputs for multi-batch data
	const int64_t* __restrict__ batch_inds,
	const int64_t* __restrict__ batch_offsets,
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
	uint32_t batch_ind = 0;
	if (batch_inds) {
		// NOTE: pass in batch_ind=-1 to ignore certain points.
		if (batch_inds[i] < 0) { return;}
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
	for (uint32_t dim=0; dim < N_POS_DIMS; ++dim) {
		grid_resolution[dim] = lod_meta.level_res[level][dim];
		scale[dim] = grid_resolution[dim] - 2;
	}
	COMPUTE_T pos[N_POS_DIMS];
	uint32_t pos_grid[N_POS_DIMS];

	positions_in += i * N_POS_DIMS;
	pos_fract<N_POS_DIMS, INPUT_T, COMPUTE_T>(positions_in, scale, interpolation_type, pos_grid, pos);

	dL_dy += i * num_lod_features + out_feat_offset;
	vector_t<PARAM_T, N_FEAT_PER_THREAD> grad;
	#pragma unroll
	for (uint32_t f = 0; f < N_FEAT_PER_THREAD; ++f) {
		// grad[f] = dL_dy[i + (out_feat_offset + f) * num_elements];
		grad[f] = dL_dy[f];
	}

	switch (lod_type_cur_lvl)
	{
		case LoDType::Dense:
		{
			auto add_grid_grad_impl = add_grid_gridient_dense_impl<PARAM_T, GRAD_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD, std::is_same<GRAD_T, __half>::value>;
			bwd_n_linear<PARAM_T, GRAD_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD>(add_grid_grad_impl, pos, pos_grid, grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, grid, grad, grid_gradient);
		}
		break;

		case LoDType::NPlaneMul:
		{
			auto add_grid_grad_impl = add_grid_gridient_nplane_mul_impl<PARAM_T, GRAD_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD, std::is_same<GRAD_T, __half>::value>;
			bwd_n_linear<PARAM_T, GRAD_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD>(add_grid_grad_impl, pos, pos_grid, grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, grid, grad, grid_gradient);
		}
		break;

		case LoDType::VectorMatrix:
		{
			ic::if_<N_POS_DIMS==3>([&]{
				auto add_grid_grad_impl = add_grid_gridient_vm_impl<PARAM_T, GRAD_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD, std::is_same<GRAD_T, __half>::value>;
				bwd_n_linear<PARAM_T, GRAD_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD>(add_grid_grad_impl, pos, pos_grid, grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, grid, grad, grid_gradient);
			}, []{});
			// if constexpr (N_POS_DIMS==3) // NOTE: Valid after c++17 (CUDA>=11)
		}
		break;

		case LoDType::VecZMatXoY:
		{
			ic::if_<N_POS_DIMS==3>([&]{
				auto add_grid_grad_impl = add_grid_gridient_vec_z_mat_xoy_impl<PARAM_T, GRAD_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD, std::is_same<GRAD_T, __half>::value>;
				bwd_n_linear<PARAM_T, GRAD_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD>(add_grid_grad_impl, pos, pos_grid, grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, grid, grad, grid_gradient);
			}, []{});
			// if constexpr (N_POS_DIMS==3) // NOTE: Valid after c++17 (CUDA>=11)
		}
		break;

		case LoDType::CP:
		{
			auto add_grid_grad_impl = add_grid_gridient_cp_eq_impl<PARAM_T, GRAD_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD, std::is_same<GRAD_T, __half>::value>;
			bwd_n_linear<PARAM_T, GRAD_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD>(add_grid_grad_impl, pos, pos_grid, grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, grid, grad, grid_gradient);
		}
		break;

		case LoDType::Hash:
		{
			auto add_grid_grad_impl = add_grid_gridient_hash_impl<PARAM_T, GRAD_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD, std::is_same<GRAD_T, __half>::value>;
			bwd_n_linear<PARAM_T, GRAD_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD>(add_grid_grad_impl, pos, pos_grid, grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, grid, grad, grid_gradient);
		}
		break;

		case LoDType::NPlaneSum:
		{
			// if constexpr (N_POS_DIMS > 2) // NOTE: Valid after c++17 (CUDA>=11)
			ic::if_<(N_POS_DIMS>2)>([&]{
				auto add_grid_gradient = [&](const uint32_t local_plane_pos[N_POS_DIMS-1], const uint32_t jump_dim, const vector_t<PARAM_T, N_FEAT_PER_THREAD>& grad, const COMPUTE_T weight) {
					uint32_t index = grid_index_nplane_sub<N_POS_DIMS>(grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, local_plane_pos, jump_dim);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600 // atomicAdd(__half2) is only supported with compute capability 60 and above
					if (N_FEAT_PER_THREAD > 1 && std::is_same<GRAD_T, __half>::value) {
						for (uint32_t f = 0; f < N_FEAT_PER_THREAD; f += 2) {
							__half2 v = {(__half)((COMPUTE_T)grad[f] * weight), (__half)((COMPUTE_T)grad[f+1] * weight)};
							atomicAdd((__half2*)&grid_gradient[index + f], v);
						}
					} else
#endif
					{
						if (std::is_same<GRAD_T, __half>::value) {
							// Should never happen
							//printf("Attempted to use atomicAdd(__half)\n")
						} else {
							#pragma unroll
							for (uint32_t f = 0; f < N_FEAT_PER_THREAD; ++f) {
								atomicAdd((float*)&grid_gradient[index + f], (float)((COMPUTE_T)grad[f] * weight));
							}
						}
					}
				};

				// sum of N * [(N-1) - linear interpolation]s
				COMPUTE_T wplane = 1; // Or, set to 1.0f / N_POS_DIMS
				#pragma unroll 1 // Force skip unrolling; significantly reduce kernel code size & actually increase speed;
				for (uint32_t jump_dim=0; jump_dim < N_POS_DIMS; ++jump_dim) {
					// (N-1) - linear interpolation
					#pragma unroll
					for (uint32_t idx = 0; idx < (1 << (N_POS_DIMS-1)); ++idx) {
						COMPUTE_T weight = wplane;
						uint32_t pos_plane_local[N_POS_DIMS-1];

						#pragma unroll
						for (uint32_t dim2D=0; dim2D < N_POS_DIMS-1; ++dim2D) {
							uint32_t dim3D = dim2D >= jump_dim ? (dim2D+1) : dim2D;
							if ((idx & (1<<dim2D)) == 0) {
								weight *= (COMPUTE_T)1 - pos[dim3D];
								pos_plane_local[dim2D] = pos_grid[dim3D];
							} else {
								weight *= pos[dim3D];
								pos_plane_local[dim2D] = pos_grid[dim3D] + 1;
							}
						}

						add_grid_gradient(pos_plane_local, jump_dim, grad, weight);
					}
				}
			}, []{});
		}
		break;

		case LoDType::CPfast:
		{
			auto line_val = [&](const uint32_t local_line_pos, const uint32_t line_dim) {
				uint32_t index = grid_index_cp_line<N_POS_DIMS>(grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, local_line_pos, line_dim);
				return *(vector_t<PARAM_T, N_FEAT_PER_THREAD>*)&grid[index];
			};

			auto add_grid_gradient = [&](const uint32_t local_line_pos, const uint32_t line_dim, const vector_t<COMPUTE_T, N_FEAT_PER_THREAD>& grad_local, const COMPUTE_T weight) {
				uint32_t index = grid_index_cp_line<N_POS_DIMS>(grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, local_line_pos, line_dim);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600 // atomicAdd(__half2) is only supported with compute capability 60 and above
				if (N_FEAT_PER_THREAD > 1 && std::is_same<GRAD_T, __half>::value) {
					for (uint32_t f = 0; f < N_FEAT_PER_THREAD; f += 2) {
						__half2 v = {(__half)(grad_local[f] * weight), (__half)(grad_local[f+1] * weight)};
						atomicAdd((__half2*)&grid_gradient[index + f], v);
					}
				} else
#endif
				{
					if (std::is_same<GRAD_T, __half>::value) {
						// Should never happen
						//printf("Attempted to use atomicAdd(__half)\n")
					} else {
						#pragma unroll
						for (uint32_t f = 0; f < N_FEAT_PER_THREAD; ++f) {
							atomicAdd((float*)&grid_gradient[index + f], (float)(grad_local[f] * weight));
						}
					}
				}
			};

			#pragma unroll 1 // Force skip unrolling; significantly reduce kernel code size & actually increase speed;
			for (uint32_t grad_dim=0; grad_dim < N_POS_DIMS; ++grad_dim) {
				vector_t<COMPUTE_T, N_FEAT_PER_THREAD> grad_local;
				#pragma unroll
				for (uint32_t f=0; f<N_FEAT_PER_THREAD; ++f) grad_local[f] = (COMPUTE_T)grad[f];

				#pragma unroll
				for (uint32_t non_grad_dim=0; non_grad_dim < N_POS_DIMS - 1; ++non_grad_dim) {
					const uint32_t dim = non_grad_dim >= grad_dim ? (non_grad_dim+1) : non_grad_dim;
					auto ng_val_left = line_val(pos_grid[dim], dim);
					auto ng_val_right = line_val(pos_grid[dim]+1, dim);
					COMPUTE_T ng_weight = pos[dim];
					
					#pragma unroll
					for (uint32_t f=0; f<N_FEAT_PER_THREAD; ++f) {
						grad_local[f] *= ((COMPUTE_T)1-ng_weight) * (COMPUTE_T)ng_val_left[f] + ng_weight * (COMPUTE_T)ng_val_right[f];
					}
				}
				add_grid_gradient(pos_grid[grad_dim], grad_dim, grad_local, (COMPUTE_T)1-pos[grad_dim]);
				add_grid_gradient(pos_grid[grad_dim]+1, grad_dim, grad_local, pos[grad_dim]);
			}
		}
		break;

		default:
			// should never happen
			break;
	}
}

template <typename INPUT_T, typename PARAM_T, typename GRAD_T, typename COMPUTE_T, uint32_t N_POS_DIMS, uint32_t N_FEAT, typename F>
__device__ __forceinline__ void bwd_input_bwd_grid_n_linear(
	F add_grid_grad, 
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
		add_grid_grad(local_pos, grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, grid, (PARAM_T*)&grad, weight, grid_gradient);
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
__global__ void kernel_lod_backward_input_backward_grid(
	const uint32_t num_elements,
	const uint32_t num_lod_features, 
	const uint32_t num_levels, 
	const LoDMetaRef lod_meta,
	int32_t max_level,
	const int32_t* __restrict__ max_level_gpu,
	// inputs
	const INPUT_T* __restrict__ dL_ddLdx,
	const PARAM_T* __restrict__ dL_dy,
	const PARAM_T* __restrict__ grid,
	const INPUT_T* __restrict__ positions_in,
	// Optional inputs for multi-batch data
	const int64_t* __restrict__ batch_inds,
	const int64_t* __restrict__ batch_offsets,
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
	uint32_t batch_ind = 0;
	if (batch_inds) {
		// NOTE: pass in batch_ind=-1 to ignore certain points.
		if (batch_inds[i] < 0) {return;}
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
	for (uint32_t dim=0; dim < N_POS_DIMS; ++dim) {
		grid_resolution[dim] = lod_meta.level_res[level][dim];
		scale[dim] = grid_resolution[dim] - 2;
	}
	COMPUTE_T pos[N_POS_DIMS];
	COMPUTE_T pos_derivative[N_POS_DIMS];
	uint32_t pos_grid[N_POS_DIMS];

	positions_in += i * N_POS_DIMS;
	pos_fract<N_POS_DIMS, INPUT_T, COMPUTE_T>(positions_in, scale, interpolation_type, pos_grid, pos, pos_derivative);

	dL_dy += i * num_lod_features + out_feat_offset;
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

	switch (lod_type_cur_lvl)
	{
		case LoDType::Dense:
		{
			auto add_grid_grad_impl = add_grid_gridient_dense_impl<PARAM_T, GRAD_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD, std::is_same<GRAD_T, __half>::value>;
			bwd_input_bwd_grid_n_linear<INPUT_T, PARAM_T, GRAD_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD>(add_grid_grad_impl, scale, pos, pos_derivative, pos_grid, grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, grid, grad, grad_input, grid_gradient);
		}
		break;

		case LoDType::NPlaneMul:
		{
			auto add_grid_grad_impl = add_grid_gridient_nplane_mul_impl<PARAM_T, GRAD_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD, std::is_same<GRAD_T, __half>::value>;
			bwd_input_bwd_grid_n_linear<INPUT_T, PARAM_T, GRAD_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD>(add_grid_grad_impl, scale, pos, pos_derivative, pos_grid, grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, grid, grad, grad_input, grid_gradient);
		}
		break;

		case LoDType::VectorMatrix:
		{
			ic::if_<N_POS_DIMS==3>([&]{
				auto add_grid_grad_impl = add_grid_gridient_vm_impl<PARAM_T, GRAD_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD, std::is_same<GRAD_T, __half>::value>;
				bwd_input_bwd_grid_n_linear<INPUT_T, PARAM_T, GRAD_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD>(add_grid_grad_impl, scale, pos, pos_derivative, pos_grid, grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, grid, grad, grad_input, grid_gradient);
			}, []{});
			// if constexpr (N_POS_DIMS==3) // NOTE: Valid after c++17 (CUDA>=11)
		}
		break;

		case LoDType::VecZMatXoY:
		{
			ic::if_<N_POS_DIMS==3>([&]{
				auto add_grid_grad_impl = add_grid_gridient_vec_z_mat_xoy_impl<PARAM_T, GRAD_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD, std::is_same<GRAD_T, __half>::value>;
				bwd_input_bwd_grid_n_linear<INPUT_T, PARAM_T, GRAD_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD>(add_grid_grad_impl, scale, pos, pos_derivative, pos_grid, grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, grid, grad, grad_input, grid_gradient);
			}, []{});
			// if constexpr (N_POS_DIMS==3) // NOTE: Valid after c++17 (CUDA>=11)
		}
		break;

		case LoDType::CP:
		{
			auto add_grid_grad_impl = add_grid_gridient_cp_eq_impl<PARAM_T, GRAD_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD, std::is_same<GRAD_T, __half>::value>;
			bwd_input_bwd_grid_n_linear<INPUT_T, PARAM_T, GRAD_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD>(add_grid_grad_impl, scale, pos, pos_derivative, pos_grid, grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, grid, grad, grad_input, grid_gradient);
		}
		break;

		case LoDType::Hash:
		{
			auto add_grid_grad_impl = add_grid_gridient_hash_impl<PARAM_T, GRAD_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD, std::is_same<GRAD_T, __half>::value>;
			bwd_input_bwd_grid_n_linear<INPUT_T, PARAM_T, GRAD_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD>(add_grid_grad_impl, scale, pos, pos_derivative, pos_grid, grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, grid, grad, grad_input, grid_gradient);
		}
		break;

		case LoDType::NPlaneSum:
		{
// 			// if constexpr (N_POS_DIMS > 2) // NOTE: Valid after c++17 (CUDA>=11)
			ic::if_<(N_POS_DIMS>2)>([&]{
				auto add_grid_gradient = [&](const uint32_t local_plane_pos[N_POS_DIMS-1], const uint32_t jump_dim, const vector_t<PARAM_T, N_FEAT_PER_THREAD>& grad, const COMPUTE_T weight) {
					uint32_t index = grid_index_nplane_sub<N_POS_DIMS>(grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, local_plane_pos, jump_dim);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600 // atomicAdd(__half2) is only supported with compute capability 60 and above
					if (N_FEAT_PER_THREAD > 1 && std::is_same<GRAD_T, __half>::value) {
						for (uint32_t f = 0; f < N_FEAT_PER_THREAD; f += 2) {
							__half2 v = {(__half)((COMPUTE_T)grad[f] * weight), (__half)((COMPUTE_T)grad[f+1] * weight)};
							atomicAdd((__half2*)&grid_gradient[index + f], v);
						}
					} else
#endif
					{
						if (std::is_same<GRAD_T, __half>::value) {
							// Should never happen
							//printf("Attempted to use atomicAdd(__half)\n")
						} else {
							#pragma unroll
							for (uint32_t f = 0; f < N_FEAT_PER_THREAD; ++f) {
								atomicAdd((float*)&grid_gradient[index + f], (float)((COMPUTE_T)grad[f] * weight));
							}
						}
					}
				};

				COMPUTE_T wplane = 1; // Or, set to 1.0f / N_POS_DIMS
				#pragma unroll 1 // Force skip unrolling; significantly reduce kernel code size & actually increase speed;
				for (uint32_t jump_dim = 0; jump_dim < N_POS_DIMS; ++jump_dim) {
					
					#pragma unroll
					for (uint32_t grad_dim_2D = 0; grad_dim_2D < N_POS_DIMS-1; ++grad_dim_2D) {
						const uint32_t grad_dim_3D = grad_dim_2D >= jump_dim ? (grad_dim_2D+1) : grad_dim_2D;
						COMPUTE_T grad_in = scale[grad_dim_3D] * wplane * (COMPUTE_T)grad_input[grad_dim_3D] * pos_derivative[grad_dim_3D];

						#pragma unroll
						for (uint32_t idx=0; idx < (1<<(N_POS_DIMS-2)); ++idx) {
							COMPUTE_T weight = grad_in;
							uint32_t pos_plane_local[N_POS_DIMS-1];

							#pragma unroll
							for (int32_t non_grad_dim = 0; non_grad_dim < N_POS_DIMS-2; ++non_grad_dim) {
								const uint32_t dim2D = non_grad_dim >= grad_dim_2D ? (non_grad_dim + 1) : non_grad_dim;
								const uint32_t dim3D = dim2D >= jump_dim ? (dim2D+1) : dim2D;
								if ((idx & 1<<non_grad_dim) == 0) {
									weight *= (COMPUTE_T)1 - pos[dim3D];
									pos_plane_local[dim2D] = pos_grid[dim3D];
								} else {
									weight *= pos[dim3D];
									pos_plane_local[dim2D] = pos_grid[dim3D] + 1;
								}
							}

							// left
							pos_plane_local[grad_dim_2D] = pos_grid[grad_dim_3D];
							add_grid_gradient(pos_plane_local, jump_dim, grad, -weight);
							// right
							pos_plane_local[grad_dim_2D] = pos_grid[grad_dim_3D] + 1;
							add_grid_gradient(pos_plane_local, jump_dim, grad, weight);
						}
					}
				}
			}, []{});
		}
		break;

		case LoDType::CPfast:
		{
			auto line_val = [&](const uint32_t local_line_pos, const uint32_t line_dim) {
				uint32_t index = grid_index_cp_line<N_POS_DIMS>(grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, local_line_pos, line_dim);
				return *(vector_t<PARAM_T, N_FEAT_PER_THREAD>*)&grid[index];
			};

			auto add_grid_gradient = [&](const uint32_t local_line_pos, const uint32_t line_dim, const vector_t<COMPUTE_T, N_FEAT_PER_THREAD>& grad_local, const COMPUTE_T weight) {
				uint32_t index = grid_index_cp_line<N_POS_DIMS>(grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, local_line_pos, line_dim);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600 // atomicAdd(__half2) is only supported with compute capability 60 and above
				if (N_FEAT_PER_THREAD > 1 && std::is_same<GRAD_T, __half>::value) {
					for (uint32_t f = 0; f < N_FEAT_PER_THREAD; f += 2) {
						__half2 v = {(__half)(grad_local[f] * weight), (__half)(grad_local[f+1] * weight)};
						atomicAdd((__half2*)&grid_gradient[index + f], v);
					}
				} else
#endif
				{
					if (std::is_same<GRAD_T, __half>::value) {
						// Should never happen
						//printf("Attempted to use atomicAdd(__half)\n")
					} else {
						#pragma unroll
						for (uint32_t f = 0; f < N_FEAT_PER_THREAD; ++f) {
							atomicAdd((float*)&grid_gradient[index + f], (float)(grad_local[f] * weight));
						}
					}
				}
			};

			#pragma unroll 1 // Force skip unrolling; significantly reduce kernel code size & actually increase speed;
			for (uint32_t line_dim=0; line_dim < N_POS_DIMS; ++line_dim) {
				// line_dim=0: h1,h2;  line_dim=1: h3,h4; line_dim=2: h5,h6

				#pragma unroll
				for (uint32_t grad_dim=0; grad_dim < N_POS_DIMS; ++grad_dim) {
					// doutput_d[grad_dim]
					vector_t<COMPUTE_T, N_FEAT_PER_THREAD> grad_local;
					#pragma unroll
					for (uint32_t f=0; f<N_FEAT_PER_THREAD; ++f) grad_local[f] = (COMPUTE_T)grad[f] * scale[grad_dim] * (COMPUTE_T)grad_input[grad_dim] * pos_derivative[grad_dim];

					COMPUTE_T w_l = -1.0f, w_r = 1.0f;
					if (line_dim != grad_dim) {
						w_l = (COMPUTE_T)1 - pos[line_dim];
						w_r = pos[line_dim];
					}

					#pragma unroll
					for (uint32_t other_line_dim=0; other_line_dim < N_POS_DIMS - 1; ++other_line_dim) {
						const uint32_t dim = other_line_dim >= line_dim ? (other_line_dim+1) : other_line_dim;
						auto ng_val_left = line_val(pos_grid[dim], dim);
						auto ng_val_right = line_val(pos_grid[dim]+1, dim);

						COMPUTE_T ng_weight_l = -1.0f, ng_weight_r = 1.0f;
						if (dim != grad_dim) {
							ng_weight_l = (COMPUTE_T)1 - pos[dim];
							ng_weight_r = pos[dim];
						}
						
						#pragma unroll
						for (uint32_t f=0; f<N_FEAT_PER_PSEUDO_LVL; ++f) {
							grad_local[f] *= ng_weight_l * (COMPUTE_T)ng_val_left[f] + ng_weight_r * (COMPUTE_T)ng_val_right[f];
						}
					}
					add_grid_gradient(pos_grid[line_dim], line_dim, grad_local, w_l);
					add_grid_gradient(pos_grid[line_dim]+1, line_dim, grad_local, w_r);
				}
			}
		}
		break;
	}
}

template <typename INPUT_T, typename PARAM_T, typename COMPUTE_T, uint32_t N_POS_DIMS, uint32_t N_FEAT, typename F>
__device__ __forceinline__ void bwd_input_bwd_input_n_linear(
	F calc_dLdx_dim,
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
		return calc_dLdx_dim(local_pos, grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, grid, (PARAM_T*)&grad, weight);
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
					COMPUTE_T weight_2nd_other = grad_in_other[real_other_grad_dim] * (pos_derivative[grad_dim] * scale[grad_dim]);
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
// void
typename std::enable_if<std::is_same<INPUT_T, float>::value, void>::type
kernel_lod_backward_input_backward_input(
	const uint32_t num_elements,
	const uint32_t num_lod_features, 
	const uint32_t num_levels, 
	const LoDMetaRef lod_meta,
	int32_t max_level,
	const int32_t* __restrict__ max_level_gpu,
	// inputs
	const INPUT_T* __restrict__ dL_ddLdx,
	const PARAM_T* __restrict__ dL_dy,
	const PARAM_T* __restrict__ grid,
	const INPUT_T* __restrict__ positions_in,
	// Optional inputs for multi-batch data
	const int64_t* __restrict__ batch_inds,
	const int64_t* __restrict__ batch_offsets,
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
	uint32_t batch_ind = 0;
	if (batch_inds) {
		// NOTE: pass in batch_ind=-1 to ignore certain points.
		if (batch_inds[i] < 0) { return;}
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
	for (uint32_t dim=0; dim < N_POS_DIMS; ++dim) {
		grid_resolution[dim] = lod_meta.level_res[level][dim];
		scale[dim] = grid_resolution[dim] - 2;
	}
	COMPUTE_T pos[N_POS_DIMS];
	COMPUTE_T pos_derivative[N_POS_DIMS];
	COMPUTE_T pos_2nd_derivative[N_POS_DIMS];
	uint32_t pos_grid[N_POS_DIMS];

	positions_in += i * N_POS_DIMS;
	pos_fract<N_POS_DIMS, INPUT_T, COMPUTE_T>(positions_in, scale, interpolation_type, pos_grid, pos, pos_derivative, pos_2nd_derivative);

	dL_dy += i * num_lod_features + out_feat_offset;
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

	switch (lod_type_cur_lvl)
	{
		case LoDType::Dense:
		{
			auto calc_dLdx_dim_impl = calc_dLdx_dim_dense_impl<PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD>;
			bwd_input_bwd_input_n_linear<INPUT_T, PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD>(calc_dLdx_dim_impl, scale, interpolation_type, pos, pos_derivative, pos_2nd_derivative, pos_grid, grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, grid, grad, grad_input, (INPUT_T*)&grad_result);
		}
		break;

		case LoDType::VectorMatrix:
		{
			ic::if_<N_POS_DIMS==3>([&]{
				auto calc_dLdx_dim_impl = calc_dLdx_dim_vm_impl<PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD>;
				bwd_input_bwd_input_n_linear<INPUT_T, PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD>(calc_dLdx_dim_impl, scale, interpolation_type, pos, pos_derivative, pos_2nd_derivative, pos_grid, grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, grid, grad, grad_input, (INPUT_T*)&grad_result);
			}, []{});
			// if constexpr (N_POS_DIMS==3) // NOTE: Valid after c++17 (CUDA>=11)
		}
		break;

		case LoDType::VecZMatXoY:
		{
			ic::if_<N_POS_DIMS==3>([&]{
				auto calc_dLdx_dim_impl = calc_dLdx_dim_vec_z_mat_xoy_impl<PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD>;
				bwd_input_bwd_input_n_linear<INPUT_T, PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD>(calc_dLdx_dim_impl, scale, interpolation_type, pos, pos_derivative, pos_2nd_derivative, pos_grid, grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, grid, grad, grad_input, (INPUT_T*)&grad_result);
			}, []{});
			// if constexpr (N_POS_DIMS==3) // NOTE: Valid after c++17 (CUDA>=11)
		}
		break;

		case LoDType::Hash:
		{
			auto calc_dLdx_dim_impl = calc_dLdx_dim_hash_impl<PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD>;
			bwd_input_bwd_input_n_linear<INPUT_T, PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_THREAD>(calc_dLdx_dim_impl, scale, interpolation_type, pos, pos_derivative, pos_2nd_derivative, pos_grid, grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, grid, grad, grad_input, (INPUT_T*)&grad_result);
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

template <typename INPUT_T, typename COMPUTE_T, uint32_t N_POS_DIMS, uint32_t N_FEAT_PER_PSEUDO_LVL>
__global__ void kernel_lod_get_grid_index(
	const uint32_t num_elements,
	const uint32_t num_lod_features, 
	const uint32_t num_levels, 
	const LoDMetaRef lod_meta,
	int32_t max_level,
	const int32_t* __restrict__ max_level_gpu,   // [n_points]
	// inputs
	const INPUT_T* __restrict__ positions_in,  // [n_points, 3]
	// Optional inputs for multi-batch data
	const int64_t* __restrict__ batch_inds,    // [n_points]
	const int64_t* __restrict__ batch_offsets, // [n_batch]
	const uint32_t batch_data_size,
	// outputs
	int64_t* __restrict__ grid_inds // [n_points, num_lod_features, 2^N_POS_DIMS]
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

	if (level > max_level) {
		return;
	}

	// For batched
	uint32_t batch_ind = 0;
	if (batch_inds) {
		// NOTE: pass in batch_ind=-1 to ignore certain points.
		if (batch_inds[i] < 0) return;
		batch_ind = batch_inds[i];
	} else if (batch_data_size) {
		batch_ind = i / batch_data_size;
	}
	const uint32_t batch_offset = batch_offsets ? batch_offsets[batch_ind] : (batch_ind * lod_meta.level_offsets[lod_meta.n_levels]);
	uint32_t grid_ind_offset = batch_offset + lod_meta.level_offsets[level];

	const uint32_t grid_size = lod_meta.level_sizes[level];
	const uint32_t n_feat_cur_lvl = lod_meta.level_n_feats[level];
	const LoDType lod_type_cur_lvl = (LoDType)lod_meta.level_types[level];
	const InterpolationType interpolation_type = (InterpolationType)lod_meta.interpolation_type;

	uint32_t grid_resolution[N_POS_DIMS];
	COMPUTE_T scale[N_POS_DIMS];
	#pragma unroll
	for (uint32_t dim=0; dim < N_POS_DIMS; ++dim) {
		grid_resolution[dim] = lod_meta.level_res[level][dim];
		scale[dim] = grid_resolution[dim] - 2;
	}
	COMPUTE_T pos[N_POS_DIMS];
	uint32_t pos_grid[N_POS_DIMS];

	positions_in += i * N_POS_DIMS;
	pos_fract<N_POS_DIMS, INPUT_T, COMPUTE_T>(positions_in, scale, interpolation_type, pos_grid, pos);

	grid_inds += (i * num_lod_features + out_feat_offset) * (1<<N_POS_DIMS);

	switch (lod_type_cur_lvl)
	{
		case LoDType::Dense:
		{
			#pragma unroll 1 // Force skip unrolling; significantly reduce kernel code size & actually increase speed;
			for (uint32_t idx = 0; idx < (1 << N_POS_DIMS); ++idx) {
				// COMPUTE_T weight = 1;
				uint32_t pos_grid_local[N_POS_DIMS];

				#pragma unroll
				for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
					if ((idx & (1<<dim)) == 0) {
						// weight *= (COMPUTE_T)1 - (COMPUTE_T)pos[dim];
						pos_grid_local[dim] = pos_grid[dim];
					} else {
						// weight *= pos[dim];
						pos_grid_local[dim] = pos_grid[dim] + 1;
					}
				}

				uint32_t ind = grid_index_dense<N_POS_DIMS>(grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, pos_grid_local);
				#pragma unroll
				for (uint32_t f = 0; f < N_FEAT_PER_PSEUDO_LVL; ++f) {
					grid_inds[idx + f * (1<<N_POS_DIMS)] = grid_ind_offset + ind + f;
				}

				// auto val = grid_val(pos_grid_local);

				// #pragma unroll
				// for (uint32_t f = 0; f < N_FEAT; ++f) {
				// 	result_ptr[f] += (PARAM_T)(weight * (COMPUTE_T)((PARAM_T*)&val)[f]);
				// }
			}
		}
		break;

		case LoDType::Hash:
		{
			#pragma unroll 1 // Force skip unrolling; significantly reduce kernel code size & actually increase speed;
			for (uint32_t idx = 0; idx < (1 << N_POS_DIMS); ++idx) {
				// COMPUTE_T weight = 1;
				uint32_t pos_grid_local[N_POS_DIMS];

				#pragma unroll
				for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
					if ((idx & (1<<dim)) == 0) {
						// weight *= (COMPUTE_T)1 - (COMPUTE_T)pos[dim];
						pos_grid_local[dim] = pos_grid[dim];
					} else {
						// weight *= pos[dim];
						pos_grid_local[dim] = pos_grid[dim] + 1;
					}
				}

				uint32_t ind = grid_index_hash<N_POS_DIMS>(grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, pos_grid_local);
				#pragma unroll
				for (uint32_t f = 0; f < N_FEAT_PER_PSEUDO_LVL; ++f) {
					grid_inds[idx + f * (1<<N_POS_DIMS)] = grid_ind_offset + ind + f;
				}

				// auto val = grid_val(pos_grid_local);

				// #pragma unroll
				// for (uint32_t f = 0; f < N_FEAT; ++f) {
				// 	result_ptr[f] += (PARAM_T)(weight * (COMPUTE_T)((PARAM_T*)&val)[f]);
				// }
			}
		}
		break;
	}
}

template <typename INPUT_T, typename PARAM_T, typename COMPUTE_T, uint32_t N_POS_DIMS, uint32_t N_FEAT_PER_PSEUDO_LVL>
inline void lod_fwd_impl_dispatched(
	LoDMeta& lod_meta,
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

	kernel_lod<INPUT_T, PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL><<<blocks_lod, n_threads, 0, stream>>>(
		batch_size, 
		lod_meta.n_encoded_dims, 
		lod_meta.n_levels, 
		{lod_meta}, 
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
inline void lod_fwd_impl_templated(
	LoDMeta& lod_meta,
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
	if (input.scalar_type() == at::kHalf && params.scalar_type() == at::kHalf) {
		lod_fwd_impl_dispatched<__half, __half, __half, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL>(lod_meta, input, params, batch_inds_, batch_offsets_, batch_data_size, max_level, need_input_grad, output, dy_dx);
	} else if (input.scalar_type() == at::kFloat && params.scalar_type() == at::kHalf) {
		lod_fwd_impl_dispatched<float, __half, float, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL>(lod_meta, input, params, batch_inds_, batch_offsets_, batch_data_size, max_level, need_input_grad, output, dy_dx);
	} else if (input.scalar_type() == at::kFloat && params.scalar_type() == at::kFloat) {
		lod_fwd_impl_dispatched<float, float, float, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL>(lod_meta, input, params, batch_inds_, batch_offsets_, batch_data_size, max_level, need_input_grad, output, dy_dx);
	} else {
		throw std::runtime_error("LoTDEncoding: Input type combination not supported. Supported types are: <input,param> -> (half, half), (float, half), (float, float)");
	}
}

template <uint32_t N_POS_DIMS>
void lod_fwd_impl(
	LoDMeta& lod_meta,
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
	switch (lod_meta.n_feat_per_pseudo_lvl) {
		case 2: lod_fwd_impl_templated<N_POS_DIMS, 2>(lod_meta, input, params, batch_inds_, batch_offsets_, batch_data_size, max_level, need_input_grad, output, dy_dx); break;
		case 4: lod_fwd_impl_templated<N_POS_DIMS, 4>(lod_meta, input, params, batch_inds_, batch_offsets_, batch_data_size, max_level, need_input_grad, output, dy_dx); break;
		case 8: lod_fwd_impl_templated<N_POS_DIMS, 8>(lod_meta, input, params, batch_inds_, batch_offsets_, batch_data_size, max_level, need_input_grad, output, dy_dx); break;
		default: throw std::runtime_error("LoTDEncoding: `n_feat_per_pseudo_lvl` must be one of [2,4,8]"); break;
	}
}

template <typename INPUT_T, typename PARAM_T, typename COMPUTE_T,  uint32_t N_POS_DIMS, uint32_t N_FEAT_PER_PSEUDO_LVL>
inline void lod_bwd_impl_dispatched(
	LoDMeta& lod_meta, 
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
		// ([batch_size, n_encoded_dims, 1] * [batch_size, n_encoded_dims * n_dims_to_encode]).sum(-2) -> [batch_size, n_encoded_dims]

		dL_dx.unsqueeze_(-2);
		at::bmm_out(dL_dx, dL_dy.view({batch_size, 1, lod_meta.n_encoded_dims}).to(scalar_type<INPUT_T>()), dy_dx_.value().view({batch_size, lod_meta.n_encoded_dims, lod_meta.n_dims_to_encode}));
		dL_dx.squeeze_(-2);
	}

	if (need_param_grad) {
		static constexpr uint32_t N_THREADS_LOD = 256;
		static constexpr uint32_t N_FEAT_PER_THREAD = std::min(2u, N_FEAT_PER_PSEUDO_LVL);
		const dim3 blocks_lod = { div_round_up(batch_size * N_FEAT_PER_PSEUDO_LVL / N_FEAT_PER_THREAD, N_THREADS_LOD), lod_meta.n_pseudo_levels, 1 };

		const at::cuda::OptionalCUDAGuard device_guard(at::device_of(params));
		cudaStream_t stream = at::cuda::getCurrentCUDAStream();

		kernel_lod_backward<INPUT_T, PARAM_T, PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL, N_FEAT_PER_THREAD><<<blocks_lod, N_THREADS_LOD, 0, stream>>> (
			batch_size, 
			lod_meta.n_encoded_dims, 
			lod_meta.n_levels, 
			{lod_meta}, 
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
inline void lod_bwd_impl_templated(
	LoDMeta& lod_meta, 
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
	if (input.scalar_type() == at::kHalf && params.scalar_type() == at::kHalf) {
		lod_bwd_impl_dispatched<__half, __half, __half, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL>(lod_meta, dL_dy, input, params, dy_dx_, batch_inds_, batch_offsets_, batch_data_size, max_level, need_input_grad, need_param_grad, dL_dx, dL_dparam); 
	} else if (input.scalar_type() == at::kFloat && params.scalar_type() == at::kHalf) {
		lod_bwd_impl_dispatched<float, __half, float, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL>(lod_meta, dL_dy, input, params, dy_dx_, batch_inds_, batch_offsets_, batch_data_size, max_level, need_input_grad, need_param_grad, dL_dx, dL_dparam); 
	} else if (input.scalar_type() == at::kFloat && params.scalar_type() == at::kFloat) {
		lod_bwd_impl_dispatched<float, float, float, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL>(lod_meta, dL_dy, input, params, dy_dx_, batch_inds_, batch_offsets_, batch_data_size, max_level, need_input_grad, need_param_grad, dL_dx, dL_dparam); 
	} else {
		throw std::runtime_error("LoTDEncoding: Input type combination not supported. Supported types are: <input,param> -> (half, half), (float, half), (float, float)");
	}
}

template <uint32_t N_POS_DIMS>
void lod_bwd_impl(
	LoDMeta& lod_meta, 
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
	switch (lod_meta.n_feat_per_pseudo_lvl) {
		case 2: lod_bwd_impl_templated<N_POS_DIMS, 2>(lod_meta, dL_dy, input, params, dy_dx_, batch_inds_, batch_offsets_, batch_data_size, max_level, need_input_grad, need_param_grad, dL_dx, dL_dparam); break;
		case 4: lod_bwd_impl_templated<N_POS_DIMS, 4>(lod_meta, dL_dy, input, params, dy_dx_, batch_inds_, batch_offsets_, batch_data_size, max_level, need_input_grad, need_param_grad, dL_dx, dL_dparam); break;
		case 8: lod_bwd_impl_templated<N_POS_DIMS, 8>(lod_meta, dL_dy, input, params, dy_dx_, batch_inds_, batch_offsets_, batch_data_size, max_level, need_input_grad, need_param_grad, dL_dx, dL_dparam); break;
		default: throw std::runtime_error("LoTDEncoding: `n_feat_per_pseudo_lvl` must be one of [2,4,8]"); break;
	}
}

template <typename INPUT_T, typename PARAM_T, typename COMPUTE_T, uint32_t N_POS_DIMS, uint32_t N_FEAT_PER_PSEUDO_LVL>
inline void lod_bwd_bwd_input_impl_dispatched(
	LoDMeta& lod_meta, 
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
		// [batch_size, n_dims_to_encode], [batch_size, n_encoded_dims, n_dims_to_encode] -> [batch_size, n_encoded_dims]

		dL_ddLdy.unsqueeze_(-1);
		at::bmm_out(dL_ddLdy, dy_dx_.value().view({ batch_size, lod_meta.n_encoded_dims, lod_meta.n_dims_to_encode }).to(scalar_type<PARAM_T>()), dL_ddLdx.view({ batch_size, lod_meta.n_dims_to_encode, 1 }).to(scalar_type<PARAM_T>()) );
		dL_ddLdy.squeeze_(-1);
	}

	if (need_input_grad) {
		static constexpr uint32_t N_THREADS_LOD = 256;
		static constexpr uint32_t N_FEAT_PER_THREAD = std::min(2u, N_FEAT_PER_PSEUDO_LVL);
		const dim3 blocks_lod = { div_round_up(batch_size * N_FEAT_PER_PSEUDO_LVL / N_FEAT_PER_THREAD, N_THREADS_LOD), lod_meta.n_pseudo_levels, 1 };

		const at::cuda::OptionalCUDAGuard device_guard(at::device_of(params));
		cudaStream_t stream = at::cuda::getCurrentCUDAStream();

		// Safer to use COMPUTE_T=float
		kernel_lod_backward_input_backward_input<INPUT_T, PARAM_T, float, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL, N_FEAT_PER_THREAD><<<blocks_lod, N_THREADS_LOD, 0, stream>>>(
			batch_size, 
			lod_meta.n_encoded_dims, 
			lod_meta.n_levels, 
			{lod_meta}, 
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

		kernel_lod_backward_input_backward_grid<INPUT_T, PARAM_T, PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL, N_FEAT_PER_THREAD><<<blocks_lod, N_THREADS_LOD, 0, stream>>>(
			batch_size, 
			lod_meta.n_encoded_dims, 
			lod_meta.n_levels, 
			{lod_meta}, 
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
inline void lod_bwd_bwd_input_impl_templated(
	LoDMeta& lod_meta, 
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
	if (input.scalar_type() == at::kHalf && params.scalar_type() == at::kHalf) {
		throw std::runtime_error("LoTDEncoding: Currently do not support input type combination = (half,half)");
		// lod_bwd_bwd_input_impl_dispatched<__half, __half, __half, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL>(lod_meta, dL_ddLdx, dL_dy, input, params, dy_dx_, batch_inds_, batch_offsets_, batch_data_size, max_level, need_dLdy_grad, need_input_grad, need_param_grad, dL_ddLdy, dL_dx, dL_dparams); 
	} else if (input.scalar_type() == at::kFloat && params.scalar_type() == at::kHalf) {
		lod_bwd_bwd_input_impl_dispatched<float, __half, float, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL>(lod_meta, dL_ddLdx, dL_dy, input, params, dy_dx_, batch_inds_, batch_offsets_, batch_data_size, max_level, need_dLdy_grad, need_input_grad, need_param_grad, dL_ddLdy, dL_dx, dL_dparams); 
	} else if (input.scalar_type() == at::kFloat && params.scalar_type() == at::kFloat) {
		lod_bwd_bwd_input_impl_dispatched<float, float, float, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL>(lod_meta, dL_ddLdx, dL_dy, input, params, dy_dx_, batch_inds_, batch_offsets_, batch_data_size, max_level, need_dLdy_grad, need_input_grad, need_param_grad, dL_ddLdy, dL_dx, dL_dparams); 
	} else {
		throw std::runtime_error("LoTDEncoding: Input type combination not supported. Supported types are: <input,param> -> (half, half), (float, half), (float, float)");
	}
}

template <uint32_t N_POS_DIMS>
void lod_bwd_bwd_input_impl(
	LoDMeta& lod_meta, 
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
	switch (lod_meta.n_feat_per_pseudo_lvl) {
		case 2: lod_bwd_bwd_input_impl_templated<N_POS_DIMS, 2>(lod_meta, dL_ddLdx, dL_dy, input, params, dy_dx_, batch_inds_, batch_offsets_, batch_data_size, max_level, need_dLdy_grad, need_input_grad, need_param_grad, dL_ddLdy, dL_dx, dL_dparams); break;
		case 4: lod_bwd_bwd_input_impl_templated<N_POS_DIMS, 4>(lod_meta, dL_ddLdx, dL_dy, input, params, dy_dx_, batch_inds_, batch_offsets_, batch_data_size, max_level, need_dLdy_grad, need_input_grad, need_param_grad, dL_ddLdy, dL_dx, dL_dparams); break;
		case 8: lod_bwd_bwd_input_impl_templated<N_POS_DIMS, 8>(lod_meta, dL_ddLdx, dL_dy, input, params, dy_dx_, batch_inds_, batch_offsets_, batch_data_size, max_level, need_dLdy_grad, need_input_grad, need_param_grad, dL_ddLdy, dL_dx, dL_dparams); break;
		default: throw std::runtime_error("LoTDEncoding: `n_feat_per_pseudo_lvl` must be one of [2,4,8]"); break;
	}
}

template <typename INPUT_T, typename COMPUTE_T, uint32_t N_POS_DIMS, uint32_t N_FEAT_PER_PSEUDO_LVL>
inline void lod_get_grid_index_dispatched(
	LoDMeta& lod_meta,
	at::Tensor input,
	at::optional<at::Tensor> batch_inds_,
	at::optional<at::Tensor> batch_offsets_,
	uint32_t batch_data_size, 
	int32_t max_level, 
	at::Tensor grid_inds
) {
	const uint32_t batch_size = input.size(0);
	// static constexpr uint32_t n_threads = 512;
	static constexpr uint32_t n_threads = 128;
	const dim3 blocks_lod = { div_round_up(batch_size, n_threads), lod_meta.n_pseudo_levels, 1 };
	const at::cuda::OptionalCUDAGuard device_guard(at::device_of(input));
	cudaStream_t stream = at::cuda::getCurrentCUDAStream();

	kernel_lod_get_grid_index<INPUT_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL><<<blocks_lod, n_threads, 0, stream>>>(
		batch_size, 
		lod_meta.n_encoded_dims, 
		lod_meta.n_levels, 
		{lod_meta}, 
		max_level, (int32_t*)nullptr, 
		data_ptr<INPUT_T>(input), 
		batch_inds_.has_value() ? batch_inds_.value().data_ptr<int64_t>() : nullptr, 
		batch_offsets_.has_value() ? batch_offsets_.value().data_ptr<int64_t>() : nullptr, 
		batch_data_size, 
		grid_inds.data_ptr<int64_t>()
	);
}

template <uint32_t N_POS_DIMS, uint32_t N_FEAT_PER_PSEUDO_LVL>
inline void lod_get_grid_index_templated(
	LoDMeta& lod_meta,
	at::Tensor input,
	at::optional<at::Tensor> batch_inds_,
	at::optional<at::Tensor> batch_offsets_,
	uint32_t batch_data_size, 
	int32_t max_level, 
	at::Tensor grid_inds
) {
	if (input.scalar_type() == at::kHalf) {
		lod_get_grid_index_dispatched<__half, __half, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL>(lod_meta, input, batch_inds_, batch_offsets_, batch_data_size, max_level, grid_inds);
	} else if (input.scalar_type() == at::kFloat) {
		lod_get_grid_index_dispatched<float, float, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL>(lod_meta, input, batch_inds_, batch_offsets_, batch_data_size, max_level, grid_inds);
	} else {
		throw std::runtime_error("LoTDEncoding: Input type not supported. Supported types are: float, half");
	}
}

template <uint32_t N_POS_DIMS>
void lod_get_grid_index_impl(
	LoDMeta& lod_meta,
	at::Tensor input,
	at::optional<at::Tensor> batch_inds_,
	at::optional<at::Tensor> batch_offsets_,
	uint32_t batch_data_size, 
	int32_t max_level, 
	at::Tensor grid_inds
) {
	switch (lod_meta.n_feat_per_pseudo_lvl) {
		case 2: lod_get_grid_index_templated<N_POS_DIMS, 2>(lod_meta, input, batch_inds_, batch_offsets_, batch_data_size, max_level, grid_inds); break;
		case 4: lod_get_grid_index_templated<N_POS_DIMS, 4>(lod_meta, input, batch_inds_, batch_offsets_, batch_data_size, max_level, grid_inds); break;
		case 8: lod_get_grid_index_templated<N_POS_DIMS, 8>(lod_meta, input, batch_inds_, batch_offsets_, batch_data_size, max_level, grid_inds); break;
		default: throw std::runtime_error("LoTDEncoding: `n_feat_per_pseudo_lvl` must be one of [2,4,8]"); break;
	}
}

// 2D instantiation
extern template void lod_fwd_impl<2>(
	LoDMeta& lod_meta,
	at::Tensor input,
	at::Tensor params,
	at::optional<at::Tensor> batch_inds_,
	at::optional<at::Tensor> batch_offsets_,
	uint32_t batch_data_size, 
	int32_t max_level, 
	bool need_input_grad, 
	at::Tensor output, 
	at::Tensor dy_dx
);

extern template void lod_bwd_impl<2>(
	LoDMeta& lod_meta, 
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
);

extern template void lod_bwd_bwd_input_impl<2>(
	LoDMeta& lod_meta, 
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
);

// 3D instantiation
extern template void lod_fwd_impl<3>(
	LoDMeta& lod_meta,
	at::Tensor input,
	at::Tensor params,
	at::optional<at::Tensor> batch_inds_,
	at::optional<at::Tensor> batch_offsets_,
	uint32_t batch_data_size, 
	int32_t max_level, 
	bool need_input_grad, 
	at::Tensor output, 
	at::Tensor dy_dx
);

extern template void lod_bwd_impl<3>(
	LoDMeta& lod_meta, 
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
);

extern template void lod_bwd_bwd_input_impl<3>(
	LoDMeta& lod_meta, 
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
);

// 4D instantiation
extern template void lod_fwd_impl<4>(
	LoDMeta& lod_meta,
	at::Tensor input,
	at::Tensor params,
	at::optional<at::Tensor> batch_inds_,
	at::optional<at::Tensor> batch_offsets_,
	uint32_t batch_data_size, 
	int32_t max_level, 
	bool need_input_grad, 
	at::Tensor output, 
	at::Tensor dy_dx
);

extern template void lod_bwd_impl<4>(
	LoDMeta& lod_meta, 
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
);

extern template void lod_bwd_bwd_input_impl<4>(
	LoDMeta& lod_meta, 
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
);

} // namespace lotd::torch

} // namespace lotd

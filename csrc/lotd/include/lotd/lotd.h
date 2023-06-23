/** @file   lotd.h
 *  @author Jianfei Guo, Shanghai AI Lab
 *  @brief  LoTD basic definitions and maths. Independent of implementation frameworks.
 * 			Relying on neither of pytorch or tiny-cuda-nn frameworks.
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

#include "lotd_types.h"
namespace lotd {

template <typename T, uint32_t N_ELEMS>
struct vector_t {
	__host__ __device__ T& operator[](uint32_t idx) {
		return data[idx];
	}

	__host__ __device__ T operator [](uint32_t idx) const {
		return data[idx];
	}

	T data[N_ELEMS];
	static constexpr uint32_t N = N_ELEMS;
};

template <uint32_t N_POS_DIMS>
__device__ __forceinline__ uint32_t grid_index_dense(
	const uint32_t feature, 
	const uint32_t grid_res[N_POS_DIMS], 
	const uint32_t grid_size, 
	const uint32_t n_feature_cur_level,
	const uint32_t pos_grid[N_POS_DIMS]
) {
	static constexpr uint32_t N_1 = N_POS_DIMS-1;
	uint32_t stride = 1;
	uint32_t index = 0;
	
	// NOTE: contiguous memory on the last dim (e.g. z of xyz)
	#pragma unroll
	for (int32_t dim = 0; dim < N_POS_DIMS; ++dim) {
		index += pos_grid[N_1-dim] * stride;
		stride *= grid_res[N_1-dim];
	}
	// NOTE: contiguous memory on the first dim (e.g. x of xyz)
	// #pragma unroll
	// for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
	// 	index += pos_grid[dim] * stride;
	// 	stride *= grid_res;
	// }

	return index * n_feature_cur_level + feature;
}

template <uint32_t N_POS_DIMS>
__device__ __forceinline__
typename std::enable_if< N_POS_DIMS<=7, uint32_t >::type
fast_hash(const uint32_t pos_grid[N_POS_DIMS]) {
	constexpr uint32_t primes[7] = { 1u, 2654435761u, 805459861u, 3674653429u, 2097192037u, 1434869437u, 2165219737u };
	uint32_t result = 0;
	#pragma unroll
	for (uint32_t i=0; i < N_POS_DIMS; ++i ) {
		result ^= pos_grid[i] * primes[i];
	}
	return result;
}

template <uint32_t N_POS_DIMS>
__device__ __forceinline__ uint32_t grid_index_hash(
	const uint32_t feature, 
	const uint32_t grid_res[N_POS_DIMS], 
	const uint32_t grid_size, 
	const uint32_t n_feature_cur_level,
	const uint32_t pos_grid[N_POS_DIMS]
) {
	uint32_t index = fast_hash<N_POS_DIMS>(pos_grid);
	return (index % grid_size) * n_feature_cur_level + feature;
}

template <uint32_t N_POS_DIMS>
__device__ __forceinline__ uint32_t grid_index_nplane_sub(
	const uint32_t feature, 
	const uint32_t grid_res[N_POS_DIMS], 
	const uint32_t grid_size, 
	const uint32_t n_feature_cur_level,
	const uint32_t pos_plane[N_POS_DIMS-1],
	const uint32_t jump_dim
) {
	static constexpr uint32_t N_1 = N_POS_DIMS-1;
	uint32_t stride = 1;
	uint32_t index = 0;
	
	// NOTE: contiguous memory on the last dim2D (e.g. z of xyz)
	#pragma unroll
	for (uint32_t dim2D = 0; dim2D < N_1; ++dim2D) {
		uint32_t dim3D = dim2D >= jump_dim ? (dim2D+1) : dim2D;
		index += pos_plane[N_1-1-dim2D] * stride;
		stride *= grid_res[N_1-dim3D];
	}

	// NOTE: contiguous memory on the first dim (e.g. x of xyz)
	// #pragma unroll
	// for (uint32_t dim = 0; dim < N_POS_DIMS-1; ++dim) {
	// 	index += pos_plane[dim] * stride;
	// 	stride *= grid_res;
	// }

	return (jump_dim * stride + index) * n_feature_cur_level + feature;
}

template <uint32_t N_POS_DIMS>
__device__ __forceinline__ uint32_t grid_index_nplane(
	const uint32_t feature, 
	const uint32_t grid_res[N_POS_DIMS], 
	const uint32_t grid_size, 
	const uint32_t n_feature_cur_level,
	const uint32_t pos_grid[N_POS_DIMS],
	const uint32_t jump_dim
) {
	static constexpr uint32_t N_1 = N_POS_DIMS-1;
	uint32_t stride = 1;
	uint32_t index = 0;
	
	// NOTE: contiguous memory on the last dim (e.g. z of xyz)
	#pragma unroll
	for (uint32_t dim2D = 0; dim2D < N_1; ++dim2D) {
		uint32_t dim3D = dim2D >= jump_dim ? (dim2D+1) : dim2D;
		index += pos_grid[N_1-dim3D] * stride;
		stride *= grid_res[N_1-dim3D];
	}

	// NOTE: contiguous memory on the first dim (e.g. x of xyz)
	// #pragma unroll
	// for (uint32_t dim = 0; dim < N_POS_DIMS-1; ++dim) {
	// 	uint32_t real_dim = dim >= jump_dim ? (dim+1) : dim;
	// 	index += pos_grid[real_dim] * stride;
	// 	stride *= grid_res;
	// }

	return index * n_feature_cur_level + feature;
}

template <uint32_t N_POS_DIMS>
__device__ __forceinline__ uint32_t grid_index_cp_line(
	const uint32_t feature, 
	const uint32_t grid_res[N_POS_DIMS], 
	const uint32_t grid_size, 
	const uint32_t n_feature_cur_level,
	const uint32_t pos_line,
	const uint32_t line_dim
) {
	uint32_t accum_grid_line = 0;
	for (uint32_t dim = 0; dim < line_dim; ++dim) {
		accum_grid_line += grid_res[dim];
	}
	return (accum_grid_line + pos_line) * n_feature_cur_level + feature;
}

template <uint32_t N_POS_DIMS>
__device__ __forceinline__ void grid_index_cp(
	const uint32_t feature, 
	const uint32_t grid_res[N_POS_DIMS], 
	const uint32_t grid_size, 
	const uint32_t n_feature_cur_level,
	const uint32_t pos_grid[N_POS_DIMS],
	uint32_t* index_line
) {
	uint32_t accum_grid_line = 0;
	#pragma unroll
	for (uint32_t line_dim=0; line_dim < N_POS_DIMS; ++line_dim) {
		index_line[line_dim] = (accum_grid_line + pos_grid[line_dim]) * n_feature_cur_level + feature;
		accum_grid_line += grid_res[line_dim];
	}
}

template <uint32_t N_POS_DIMS>
__device__ __forceinline__ void grid_index_vm(
	const uint32_t feature, 
	const uint32_t grid_res[N_POS_DIMS], 
	const uint32_t grid_size, 
	const uint32_t n_feature_cur_level,
	const uint32_t pos_grid[N_POS_DIMS],
	uint32_t index_plane[N_POS_DIMS],
	uint32_t index_line[N_POS_DIMS]
) {
	static constexpr uint32_t N_1 = N_POS_DIMS-1;

	uint32_t accum_grid_line = 0;
	#pragma unroll
	for (uint32_t line_dim=0; line_dim < N_POS_DIMS; ++line_dim) {
		index_line[line_dim] = (accum_grid_line + pos_grid[line_dim]) * n_feature_cur_level + feature;
		accum_grid_line += grid_res[line_dim];
	}

	uint32_t accum_grid_plane = 0;
	#pragma unroll
	for (uint32_t line_dim=0; line_dim < N_POS_DIMS; ++line_dim) {
		uint32_t stride = 1;
		uint32_t index = 0;

		// NOTE: contiguous memory on the last dim (e.g. z of xyz)
		const uint32_t rev_jump_dim = N_1 - line_dim;
		#pragma unroll
		for (uint32_t dim2D = 0; dim2D < N_1; ++dim2D) {
			uint32_t dim3D = dim2D >= rev_jump_dim ? (dim2D+1) : dim2D;
			index += pos_grid[N_1-dim3D] * stride;
			stride *= grid_res[N_1-dim3D];
		}
		index_plane[line_dim] = (accum_grid_line + accum_grid_plane + index) * n_feature_cur_level + feature;
		accum_grid_plane += stride;
	}
}

template <uint32_t N_POS_DIMS>
__device__ __forceinline__
typename std::enable_if<N_POS_DIMS==3, void>::type 
grid_index_vm_xoy(
	const uint32_t feature, 
	const uint32_t grid_res[N_POS_DIMS], 
	const uint32_t grid_size, 
	const uint32_t n_feature_cur_level,
	const uint32_t pos_grid[N_POS_DIMS],
	uint32_t* __restrict__ index_plane,
	uint32_t* __restrict__ index_line
) {
	*index_line = pos_grid[2] * n_feature_cur_level + feature;

	// grid_res[2]: offset for vector length
	// pos_grid[1] +  pos_grid[0] * grid_res[0]: current position in flatten 2D grid
	*index_plane = (grid_res[2] + pos_grid[1] +  pos_grid[0] * grid_res[0]) * n_feature_cur_level + feature;
}

template<typename PARAM_T, typename COMPUTE_T, uint32_t N_POS_DIMS, uint32_t N_FEAT>
__device__ __forceinline__ void grid_val_dense_impl(
	const uint32_t local_pos[N_POS_DIMS],
	const uint32_t grid_feat_offset,
	const uint32_t grid_resolution[N_POS_DIMS],
	const uint32_t grid_size,
	const uint32_t n_feat_cur_lvl,
	const PARAM_T* __restrict__ grid,
	PARAM_T* __restrict__ val
) {
	uint32_t index = grid_index_dense<N_POS_DIMS>(grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, local_pos);
	// #pragma unroll
	// for (uint32_t f=0; f < N_FEAT; ++f) {
	// 	val[f] = grid[index+f];
	// }
	*(vector_t<PARAM_T, N_FEAT>*)val = *(vector_t<PARAM_T, N_FEAT>*)(&grid[index]);
}

template<typename PARAM_T, typename COMPUTE_T, uint32_t N_POS_DIMS, uint32_t N_FEAT>
__device__ __forceinline__ void grid_val_hash_impl(
	const uint32_t local_pos[N_POS_DIMS],
	const uint32_t grid_feat_offset,
	const uint32_t grid_resolution[N_POS_DIMS],
	const uint32_t grid_size,
	const uint32_t n_feat_cur_lvl,
	const PARAM_T* __restrict__ grid,
	PARAM_T* __restrict__ val
) {
	uint32_t index = grid_index_hash<N_POS_DIMS>(grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, local_pos);
	#pragma unroll
	for (uint32_t f=0; f < N_FEAT; ++f) {
		val[f] = grid[index+f];
	}
}


template<typename PARAM_T, typename COMPUTE_T, uint32_t N_POS_DIMS, uint32_t N_FEAT>
__device__ __forceinline__ void grid_val_vm_impl(
	const uint32_t local_pos[N_POS_DIMS],
	const uint32_t grid_feat_offset,
	const uint32_t grid_resolution[N_POS_DIMS],
	const uint32_t grid_size,
	const uint32_t n_feat_cur_lvl,
	const PARAM_T* __restrict__ grid,
	PARAM_T* __restrict__ val
) {
	// Uncomment for performance. Always remember to zero-init `val`
	#pragma unroll
	for (uint32_t f=0; f<N_FEAT; ++f) {
		val[f] = 0; // Important! sometimes `val` is not properly initialized.
	}

	uint32_t index_lines[N_POS_DIMS], index_planes[N_POS_DIMS];
	grid_index_vm<N_POS_DIMS>(grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, local_pos, index_planes, index_lines);

	#pragma unroll
	for (uint32_t line_dim=0; line_dim < N_POS_DIMS; ++line_dim) {
		uint32_t index_plane = index_planes[line_dim];
		uint32_t index_line = index_lines[line_dim];
		#pragma unroll
		for (uint32_t f=0; f<N_FEAT; ++f) {
			val[f] += (PARAM_T)((COMPUTE_T)grid[index_plane+f] * (COMPUTE_T)grid[index_line+f]);
		}
	}

	// uint32_t index_plane[N_POS_DIMS];
	// uint32_t index_line[N_POS_DIMS];
	// #pragma unroll
	// for (uint32_t line_dim=0; line_dim < N_POS_DIMS; ++line_dim) {
	// 	grid_index_vm<N_POS_DIMS>(grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, local_pos, line_dim, &index_plane[line_dim], &index_line[line_dim]);
	// }

	// #pragma unroll
	// for (uint32_t f=0; f<N_FEAT; ++f) {
	// 	COMPUTE_T result = ((COMPUTE_T)grid[index_plane[0]+f] * (COMPUTE_T)grid[index_line[0]+f]);
	// 	#pragma unroll
	// 	for (uint32_t dim=1; dim < N_POS_DIMS; ++dim) {
	// 		result += ((COMPUTE_T)grid[index_plane[dim]+f] * (COMPUTE_T)grid[index_line[dim]+f]);
	// 	}
	// 	val[f] = result;
	// }
}

template<typename PARAM_T, typename COMPUTE_T, uint32_t N_POS_DIMS, uint32_t N_FEAT>
__device__ __forceinline__ 
typename std::enable_if<N_POS_DIMS!=3, void>::type 
grid_val_vec_z_mat_xoy_impl(
	const uint32_t local_pos[N_POS_DIMS],
	const uint32_t grid_feat_offset,
	const uint32_t grid_resolution[N_POS_DIMS],
	const uint32_t grid_size,
	const uint32_t n_feat_cur_lvl,
	const PARAM_T* __restrict__ grid,
	PARAM_T* __restrict__ val
) {
	// Should never happen: N_POS_DIMS!=3
}

template<typename PARAM_T, typename COMPUTE_T, uint32_t N_POS_DIMS, uint32_t N_FEAT>
__device__ __forceinline__ 
typename std::enable_if<N_POS_DIMS==3, void>::type 
grid_val_vec_z_mat_xoy_impl(
	const uint32_t local_pos[N_POS_DIMS],
	const uint32_t grid_feat_offset,
	const uint32_t grid_resolution[N_POS_DIMS],
	const uint32_t grid_size,
	const uint32_t n_feat_cur_lvl,
	const PARAM_T* __restrict__ grid,
	PARAM_T* __restrict__ val
) {
	uint32_t index_plane, index_line;
	grid_index_vm_xoy<N_POS_DIMS>(grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, local_pos, &index_plane, &index_line);
	#pragma unroll
	for (uint32_t f=0; f<N_FEAT; ++f) {
		val[f] = (PARAM_T)((COMPUTE_T)grid[index_plane+f] * (COMPUTE_T)grid[index_line+f]);
	}
}

template<typename PARAM_T, typename COMPUTE_T, uint32_t N_POS_DIMS, uint32_t N_FEAT>
__device__ __forceinline__ void grid_val_nplane_sum_impl(
	const uint32_t local_pos[N_POS_DIMS],
	const uint32_t grid_feat_offset,
	const uint32_t grid_resolution[N_POS_DIMS],
	const uint32_t grid_size,
	const uint32_t n_feat_cur_lvl,
	const PARAM_T* __restrict__ grid,
	PARAM_T* __restrict__ val
) {
	uint32_t index_planes[N_POS_DIMS];
	#pragma unroll
	for (uint32_t jump_dim=0; jump_dim < N_POS_DIMS; ++jump_dim) {
		index_planes[jump_dim] = grid_index_nplane<N_POS_DIMS>(grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, local_pos, jump_dim);
	}

	#pragma unroll
	for (uint32_t f=0; f<N_FEAT; ++f) {
		COMPUTE_T result = (COMPUTE_T)grid[index_planes[0]+f];
		#pragma unroll
		for (uint32_t jump_dim=1; jump_dim < N_POS_DIMS; ++jump_dim) {
			result += (COMPUTE_T)grid[index_planes[jump_dim]+f];
		}
		val[f] = (PARAM_T)result;
	}
}

template<typename PARAM_T, typename COMPUTE_T, uint32_t N_POS_DIMS, uint32_t N_FEAT>
__device__ __forceinline__ void grid_val_nplane_mul_impl(
	const uint32_t local_pos[N_POS_DIMS],
	const uint32_t grid_feat_offset,
	const uint32_t grid_resolution[N_POS_DIMS],
	const uint32_t grid_size,
	const uint32_t n_feat_cur_lvl,
	const PARAM_T* __restrict__ grid,
	PARAM_T* __restrict__ val
) {
	uint32_t index_planes[N_POS_DIMS];
	#pragma unroll
	for (uint32_t jump_dim=0; jump_dim < N_POS_DIMS; ++jump_dim) {
		index_planes[jump_dim] = grid_index_nplane<N_POS_DIMS>(grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, local_pos, jump_dim);
	}

	#pragma unroll
	for (uint32_t f=0; f<N_FEAT; ++f) {
		COMPUTE_T result = (COMPUTE_T)grid[index_planes[0]+f];
		#pragma unroll
		for (uint32_t jump_dim=1; jump_dim < N_POS_DIMS; ++jump_dim) {
			result *= (COMPUTE_T)grid[index_planes[jump_dim]+f];
		}
		val[f] = (PARAM_T)result;
	}
}

template<typename PARAM_T, typename COMPUTE_T, uint32_t N_POS_DIMS, uint32_t N_FEAT>
__device__ __forceinline__ void grid_val_cp_eq_impl(
	const uint32_t local_pos[N_POS_DIMS],
	const uint32_t grid_feat_offset,
	const uint32_t grid_resolution[N_POS_DIMS],
	const uint32_t grid_size,
	const uint32_t n_feat_cur_lvl,
	const PARAM_T* __restrict__ grid,
	PARAM_T* __restrict__ val
) {
	uint32_t index_lines[N_POS_DIMS];
	grid_index_cp<N_POS_DIMS>(grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, local_pos, index_lines);

	#pragma unroll
	for (uint32_t f=0; f<N_FEAT; ++f) {
		COMPUTE_T result = (COMPUTE_T)grid[index_lines[0]+f];
		#pragma unroll
		for (uint32_t line_dim=1; line_dim < N_POS_DIMS; ++line_dim) {
			result *= (COMPUTE_T)grid[index_lines[line_dim]+f];
		}
		val[f] = (PARAM_T)result;
	}
}

template<typename PARAM_T, typename GRAD_T, typename COMPUTE_T, uint32_t N_POS_DIMS, uint32_t N_FEAT, bool grad_is_half>
__device__ __forceinline__ void add_grid_gridient_dense_impl(
	const uint32_t local_pos[N_POS_DIMS],
	const uint32_t grid_feat_offset,
	const uint32_t grid_resolution[N_POS_DIMS],
	const uint32_t grid_size,
	const uint32_t n_feat_cur_lvl,
	const PARAM_T* __restrict__ grid,
	const PARAM_T* __restrict__ grad,
	const COMPUTE_T weight,
	GRAD_T* __restrict__ grid_gradient
) {
	uint32_t index = grid_index_dense<N_POS_DIMS>(grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, local_pos);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600 // atomicAdd(__half2) is only supported with compute capability 60 and above
	if (N_FEAT > 1 && grad_is_half) {
		for (uint32_t f = 0; f < N_FEAT; f += 2) {
			__half2 v = {(__half)((COMPUTE_T)grad[f] * weight), (__half)((COMPUTE_T)grad[f+1] * weight)};
			atomicAdd((__half2*)&grid_gradient[index + f], v);
		}
	} else
#endif
	{
		if (grad_is_half) {
			// Should never happen
			//printf("Attempted to use atomicAdd(__half)\n")
		} else {
			// NOTE: Here we perform a reduction on `grid_gradient` using `atomicAdd`.
			#pragma unroll
			for (uint32_t f = 0; f < N_FEAT; ++f) {
				atomicAdd((float*)&grid_gradient[index + f], (float)((COMPUTE_T)grad[f] * weight));
			}
		}
	}
}

template<typename PARAM_T, typename GRAD_T, typename COMPUTE_T, uint32_t N_POS_DIMS, uint32_t N_FEAT, bool grad_is_half>
__device__ __forceinline__ void add_grid_gridient_hash_impl(
	const uint32_t local_pos[N_POS_DIMS],
	const uint32_t grid_feat_offset,
	const uint32_t grid_resolution[N_POS_DIMS],
	const uint32_t grid_size,
	const uint32_t n_feat_cur_lvl,
	const PARAM_T* __restrict__ grid,
	const PARAM_T* __restrict__ grad,
	const COMPUTE_T weight,
	GRAD_T* __restrict__ grid_gradient
) {
	uint32_t index = grid_index_hash<N_POS_DIMS>(grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, local_pos);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600 // atomicAdd(__half2) is only supported with compute capability 60 and above
	if (N_FEAT > 1 && grad_is_half) {
		for (uint32_t f = 0; f < N_FEAT; f += 2) {
			__half2 v = {(__half)((COMPUTE_T)grad[f] * weight), (__half)((COMPUTE_T)grad[f+1] * weight)};
			atomicAdd((__half2*)&grid_gradient[index + f], v);
		}
	} else
#endif
	{
		if (grad_is_half) {
			// Should never happen
			//printf("Attempted to use atomicAdd(__half)\n")
		} else {
			// NOTE: Here we perform a reduction on `grid_gradient` using `atomicAdd`.
			#pragma unroll
			for (uint32_t f = 0; f < N_FEAT; ++f) {
				atomicAdd((float*)&grid_gradient[index + f], (float)((COMPUTE_T)grad[f] * weight));
			}
		}
	}
}

template<typename PARAM_T, typename GRAD_T, typename COMPUTE_T, uint32_t N_POS_DIMS, uint32_t N_FEAT, bool grad_is_half>
__device__ __forceinline__ 
typename std::enable_if<N_POS_DIMS!=3, void>::type 
add_grid_gridient_vm_impl(
	const uint32_t local_pos[N_POS_DIMS],
	const uint32_t grid_feat_offset,
	const uint32_t grid_resolution[N_POS_DIMS],
	const uint32_t grid_size,
	const uint32_t n_feat_cur_lvl,
	const PARAM_T* __restrict__ grid,
	const PARAM_T* __restrict__ grad,
	const COMPUTE_T weight,
	GRAD_T* __restrict__ grid_gradient
) {
	// Should never happen
}

template<typename PARAM_T, typename GRAD_T, typename COMPUTE_T, uint32_t N_POS_DIMS, uint32_t N_FEAT, bool grad_is_half>
__device__ __forceinline__ 
typename std::enable_if<N_POS_DIMS==3, void>::type 
add_grid_gridient_vm_impl(
	const uint32_t local_pos[N_POS_DIMS],
	const uint32_t grid_feat_offset,
	const uint32_t grid_resolution[N_POS_DIMS],
	const uint32_t grid_size,
	const uint32_t n_feat_cur_lvl,
	const PARAM_T* __restrict__ grid,
	const PARAM_T* __restrict__ grad,
	const COMPUTE_T weight,
	GRAD_T* __restrict__ grid_gradient
) {
	COMPUTE_T weighted_grad[N_FEAT];
	#pragma unroll
	for (uint32_t f = 0; f < N_FEAT; ++f) {
		weighted_grad[f] = (COMPUTE_T)grad[f] * weight;
	}

	uint32_t index_lines[N_POS_DIMS], index_planes[N_POS_DIMS];
	grid_index_vm<N_POS_DIMS>(grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, local_pos, index_planes, index_lines);

	#pragma unroll
	for (uint32_t line_dim=0; line_dim < N_POS_DIMS; ++line_dim) {
		uint32_t index_plane = index_planes[line_dim];
		uint32_t index_line = index_lines[line_dim];
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600 // atomicAdd(__half2) is only supported with compute capability 60 and above
		if (N_FEAT > 1 && grad_is_half) {
			for (uint32_t f = 0; f < N_FEAT; f += 2) {
				{
					__half2 v = {(__half)(weighted_grad[f] * (COMPUTE_T)grid[index_line+f]), (__half)(weighted_grad[f+1] * (COMPUTE_T)grid[index_line+f+1])};
					atomicAdd((__half2*)&grid_gradient[index_plane + f], v);
				}
				{
					__half2 v = {(__half)(weighted_grad[f] * (COMPUTE_T)grid[index_plane+f]), (__half)(weighted_grad[f+1] * (COMPUTE_T)grid[index_plane+f+1])};
					atomicAdd((__half2*)&grid_gradient[index_line + f], v);
				}
			}
		} else
#endif
		{
			if (grad_is_half) {
				// Should never happen
				//printf("Attempted to use atomicAdd(__half)\n")
			} else {
				// NOTE: Here we perform a reduction on `grid_gradient` using `atomicAdd`.
				#pragma unroll
				for (uint32_t f = 0; f < N_FEAT; ++f) {
					atomicAdd((float*)&grid_gradient[index_plane + f], (float)(weighted_grad[f] * (COMPUTE_T)grid[index_line+f]));
					atomicAdd((float*)&grid_gradient[index_line + f], (float)(weighted_grad[f] * (COMPUTE_T)grid[index_plane+f]));
				}
			}
		}
	}
}

template<typename PARAM_T, typename GRAD_T, typename COMPUTE_T, uint32_t N_POS_DIMS, uint32_t N_FEAT, bool grad_is_half>
__device__ __forceinline__
typename std::enable_if<N_POS_DIMS!=3, void>::type 
add_grid_gridient_vec_z_mat_xoy_impl(
	const uint32_t local_pos[N_POS_DIMS],
	const uint32_t grid_feat_offset,
	const uint32_t grid_resolution[N_POS_DIMS],
	const uint32_t grid_size,
	const uint32_t n_feat_cur_lvl,
	const PARAM_T* __restrict__ grid,
	const PARAM_T* __restrict__ grad,
	const COMPUTE_T weight,
	GRAD_T* __restrict__ grid_gradient
) {
	// Should never happen
}

template<typename PARAM_T, typename GRAD_T, typename COMPUTE_T, uint32_t N_POS_DIMS, uint32_t N_FEAT, bool grad_is_half>
__device__ __forceinline__
typename std::enable_if<N_POS_DIMS==3, void>::type 
add_grid_gridient_vec_z_mat_xoy_impl(
	const uint32_t local_pos[N_POS_DIMS],
	const uint32_t grid_feat_offset,
	const uint32_t grid_resolution[N_POS_DIMS],
	const uint32_t grid_size,
	const uint32_t n_feat_cur_lvl,
	const PARAM_T* __restrict__ grid,
	const PARAM_T* __restrict__ grad,
	const COMPUTE_T weight,
	GRAD_T* __restrict__ grid_gradient
) {
	COMPUTE_T weighted_grad[N_FEAT];
	#pragma unroll
	for (uint32_t f = 0; f < N_FEAT; ++f) {
		weighted_grad[f] = (COMPUTE_T)grad[f] * weight;
	}

	uint32_t index_plane, index_line;
	grid_index_vm_xoy<N_POS_DIMS>(grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, local_pos, &index_plane, &index_line);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600 // atomicAdd(__half2) is only supported with compute capability 60 and above
	if (N_FEAT > 1 && grad_is_half) {
		for (uint32_t f = 0; f < N_FEAT; f += 2) {
			{
				__half2 v = {(__half)(weighted_grad[f] * (COMPUTE_T)grid[index_line+f]), (__half)(weighted_grad[f+1] * (COMPUTE_T)grid[index_line+f+1])};
				atomicAdd((__half2*)&grid_gradient[index_plane + f], v);
			}
			{
				__half2 v = {(__half)(weighted_grad[f] * (COMPUTE_T)grid[index_plane+f]), (__half)(weighted_grad[f+1] * (COMPUTE_T)grid[index_plane+f+1])};
				atomicAdd((__half2*)&grid_gradient[index_line + f], v);
			}
		}
	} else
#endif
	{
		if (grad_is_half) {
			// Should never happen
			//printf("Attempted to use atomicAdd(__half)\n")
		} else {
			// NOTE: Here we perform a reduction on `grid_gradient` using `atomicAdd`.
			#pragma unroll
			for (uint32_t f = 0; f < N_FEAT; ++f) {
				atomicAdd((float*)&grid_gradient[index_plane + f], (float)(weighted_grad[f] * (COMPUTE_T)grid[index_line+f]));
				atomicAdd((float*)&grid_gradient[index_line + f], (float)(weighted_grad[f] * (COMPUTE_T)grid[index_plane+f]));
			}
		}
	}
}

template<typename PARAM_T, typename GRAD_T, typename COMPUTE_T, uint32_t N_POS_DIMS, uint32_t N_FEAT, bool grad_is_half>
__device__ __forceinline__ void add_grid_gridient_nplane_mul_impl(
	const uint32_t local_pos[N_POS_DIMS],
	const uint32_t grid_feat_offset,
	const uint32_t grid_resolution[N_POS_DIMS],
	const uint32_t grid_size,
	const uint32_t n_feat_cur_lvl,
	const PARAM_T* __restrict__ grid,
	const PARAM_T* __restrict__ grad,
	const COMPUTE_T weight,
	GRAD_T* __restrict__ grid_gradient
) {
	uint32_t index_planes[N_POS_DIMS];
	#pragma unroll
	for (uint32_t jump_dim=0; jump_dim < N_POS_DIMS; ++jump_dim) {
		index_planes[jump_dim] = grid_index_nplane<N_POS_DIMS>(grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, local_pos, jump_dim);
	}

	COMPUTE_T weighted_grad[N_FEAT];
	#pragma unroll
	for (uint32_t f = 0; f < N_FEAT; ++f) {
		weighted_grad[f] = (COMPUTE_T)grad[f] * weight;
	}

	#pragma unroll
	for (uint32_t grad_dim = 0; grad_dim < N_POS_DIMS; ++grad_dim) {
		COMPUTE_T grad_product[N_FEAT];
		const uint32_t index = index_planes[grad_dim];

		#pragma unroll
		for (uint32_t f = 0; f < N_FEAT; ++f) {
			COMPUTE_T cur_grad = weighted_grad[f];
			
			#pragma unroll
			for (uint32_t non_grad_dim = 0; non_grad_dim < N_POS_DIMS-1; ++non_grad_dim) {
				uint32_t dim = non_grad_dim >= grad_dim ? (non_grad_dim+1) : non_grad_dim;
				cur_grad *= (COMPUTE_T)grid[index_planes[dim]+f];
			}
			grad_product[f] = cur_grad;
		}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600 // atomicAdd(__half2) is only supported with compute capability 60 and above
		if (N_FEAT > 1 && grad_is_half) {
			for (uint32_t f = 0; f < N_FEAT; f += 2) {
				__half2 v = {(__half)grad_product[f], (__half)grad_product[f+1]};
				atomicAdd((__half2*)&grid_gradient[index + f], v);
			}
		} else
#endif
		{
			if (grad_is_half) {
				// Should never happen
				//printf("Attempted to use atomicAdd(__half)\n")
			} else {
				// NOTE: Here we perform a reduction on `grid_gradient` using `atomicAdd`.
				#pragma unroll
				for (uint32_t f = 0; f < N_FEAT; ++f) {
					atomicAdd((float*)&grid_gradient[index + f], (float)grad_product[f]);
				}
			}
		}
	}
}

template<typename PARAM_T, typename GRAD_T, typename COMPUTE_T, uint32_t N_POS_DIMS, uint32_t N_FEAT, bool grad_is_half>
__device__ __forceinline__ void add_grid_gridient_cp_eq_impl(
	const uint32_t local_pos[N_POS_DIMS],
	const uint32_t grid_feat_offset,
	const uint32_t grid_resolution[N_POS_DIMS],
	const uint32_t grid_size,
	const uint32_t n_feat_cur_lvl,
	const PARAM_T* __restrict__ grid,
	const PARAM_T* __restrict__ grad,
	const COMPUTE_T weight,
	GRAD_T* __restrict__ grid_gradient
) {
	uint32_t index_lines[N_POS_DIMS];
	grid_index_cp<N_POS_DIMS>(grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, local_pos, index_lines);

	COMPUTE_T weighted_grad[N_FEAT];
	#pragma unroll
	for (uint32_t f = 0; f < N_FEAT; ++f) {
		weighted_grad[f] = (COMPUTE_T)grad[f] * weight;
	}

	#pragma unroll
	for (uint32_t grad_dim = 0; grad_dim < N_POS_DIMS; ++grad_dim) {
		COMPUTE_T grad_product[N_FEAT];
		const uint32_t index = index_lines[grad_dim];

		#pragma unroll
		for (uint32_t f = 0; f < N_FEAT; ++f) {
			COMPUTE_T cur_grad = weighted_grad[f];
			
			#pragma unroll
			for (uint32_t non_grad_dim = 0; non_grad_dim < N_POS_DIMS-1; ++non_grad_dim) {
				uint32_t dim = non_grad_dim >= grad_dim ? (non_grad_dim+1) : non_grad_dim;
				cur_grad *= (COMPUTE_T)grid[index_lines[dim]+f];
			}
			grad_product[f] = cur_grad;
		}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600 // atomicAdd(__half2) is only supported with compute capability 60 and above
		if (N_FEAT > 1 && grad_is_half) {
			for (uint32_t f = 0; f < N_FEAT; f += 2) {
				__half2 v = {(__half)grad_product[f], (__half)grad_product[f+1]};
				atomicAdd((__half2*)&grid_gradient[index + f], v);
			}
		} else
#endif
		{
			if (grad_is_half) {
				// Should never happen
				//printf("Attempted to use atomicAdd(__half)\n")
			} else {
				// NOTE: Here we perform a reduction on `grid_gradient` using `atomicAdd`.
				#pragma unroll
				for (uint32_t f = 0; f < N_FEAT; ++f) {
					atomicAdd((float*)&grid_gradient[index + f], (float)grad_product[f]);
				}
			}
		}
	}
}

template<typename PARAM_T, typename COMPUTE_T, uint32_t N_POS_DIMS, uint32_t N_FEAT>
__device__ __forceinline__ COMPUTE_T calc_dLdx_dim_dense_impl( 
	const uint32_t local_pos[N_POS_DIMS],
	const uint32_t grid_feat_offset,
	const uint32_t grid_resolution[N_POS_DIMS],
	const uint32_t grid_size,
	const uint32_t n_feat_cur_lvl,
	const PARAM_T* __restrict__ grid,
	const PARAM_T* __restrict__ grad,
	const COMPUTE_T weight
) {
	uint32_t index = grid_index_dense<N_POS_DIMS>(grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, local_pos);
	COMPUTE_T dL_dx_dim = 0;
	#pragma unroll
	for (uint32_t f=0; f < N_FEAT; ++f) {
		dL_dx_dim += (COMPUTE_T)grid[index + f] * (COMPUTE_T)grad[f] * weight;
	}
	return dL_dx_dim;
}

template<typename PARAM_T, typename COMPUTE_T, uint32_t N_POS_DIMS, uint32_t N_FEAT>
__device__ __forceinline__ COMPUTE_T calc_dLdx_dim_hash_impl( 
	const uint32_t local_pos[N_POS_DIMS],
	const uint32_t grid_feat_offset,
	const uint32_t grid_resolution[N_POS_DIMS],
	const uint32_t grid_size,
	const uint32_t n_feat_cur_lvl,
	const PARAM_T* __restrict__ grid,
	const PARAM_T* __restrict__ grad,
	const COMPUTE_T weight
) {
	uint32_t index = grid_index_hash<N_POS_DIMS>(grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, local_pos);
	COMPUTE_T dL_dx_dim = 0;
	#pragma unroll
	for (uint32_t f=0; f < N_FEAT; ++f) {
		dL_dx_dim += (COMPUTE_T)grid[index + f] * (COMPUTE_T)grad[f] * weight;
	}
	return dL_dx_dim;
}

template<typename PARAM_T, typename COMPUTE_T, uint32_t N_POS_DIMS, uint32_t N_FEAT>
__device__ __forceinline__ 
typename std::enable_if<N_POS_DIMS!=3, COMPUTE_T>::type 
calc_dLdx_dim_vm_impl(
	const uint32_t local_pos[N_POS_DIMS],
	const uint32_t grid_feat_offset,
	const uint32_t grid_resolution[N_POS_DIMS],
	const uint32_t grid_size,
	const uint32_t n_feat_cur_lvl,
	const PARAM_T* __restrict__ grid,
	const PARAM_T* __restrict__ grad,
	const COMPUTE_T weight
) {
	// Should never happen
}

template<typename PARAM_T, typename COMPUTE_T, uint32_t N_POS_DIMS, uint32_t N_FEAT>
__device__ __forceinline__ 
typename std::enable_if<N_POS_DIMS==3, COMPUTE_T>::type 
calc_dLdx_dim_vm_impl(
	const uint32_t local_pos[N_POS_DIMS],
	const uint32_t grid_feat_offset,
	const uint32_t grid_resolution[N_POS_DIMS],
	const uint32_t grid_size,
	const uint32_t n_feat_cur_lvl,
	const PARAM_T* __restrict__ grid,
	const PARAM_T* __restrict__ grad,
	const COMPUTE_T weight
) {
	COMPUTE_T weighted_grad[N_FEAT];
	#pragma unroll
	for (uint32_t f = 0; f < N_FEAT; ++f) {
		weighted_grad[f] = (COMPUTE_T)grad[f] * weight;
	}
	uint32_t index_lines[N_POS_DIMS], index_planes[N_POS_DIMS];
	grid_index_vm<N_POS_DIMS>(grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, local_pos, index_planes, index_lines);
	COMPUTE_T dL_dx_dim = 0;
	#pragma unroll
	for (uint32_t line_dim=0; line_dim<N_POS_DIMS; ++line_dim) {
		uint32_t index_plane = index_planes[line_dim];
		uint32_t index_line = index_lines[line_dim];
		#pragma unroll
		for (uint32_t f=0; f < N_FEAT; ++f) {
			dL_dx_dim += (COMPUTE_T)grid[index_plane + f] * (COMPUTE_T)grid[index_line+f] * weighted_grad[f];
		}
	}
	return dL_dx_dim;
}

template<typename PARAM_T, typename COMPUTE_T, uint32_t N_POS_DIMS, uint32_t N_FEAT>
__device__ __forceinline__
typename std::enable_if<N_POS_DIMS!=3, COMPUTE_T>::type 
calc_dLdx_dim_vec_z_mat_xoy_impl(
	const uint32_t local_pos[N_POS_DIMS],
	const uint32_t grid_feat_offset,
	const uint32_t grid_resolution[N_POS_DIMS],
	const uint32_t grid_size,
	const uint32_t n_feat_cur_lvl,
	const PARAM_T* __restrict__ grid,
	const PARAM_T* __restrict__ grad,
	const COMPUTE_T weight
) {
	// Should never happen
}

template<typename PARAM_T, typename COMPUTE_T, uint32_t N_POS_DIMS, uint32_t N_FEAT>
__device__ __forceinline__
typename std::enable_if<N_POS_DIMS==3, COMPUTE_T>::type 
calc_dLdx_dim_vec_z_mat_xoy_impl(
	const uint32_t local_pos[N_POS_DIMS],
	const uint32_t grid_feat_offset,
	const uint32_t grid_resolution[N_POS_DIMS],
	const uint32_t grid_size,
	const uint32_t n_feat_cur_lvl,
	const PARAM_T* __restrict__ grid,
	const PARAM_T* __restrict__ grad,
	const COMPUTE_T weight
) {
	uint32_t index_line, index_plane;
	grid_index_vm_xoy<N_POS_DIMS>(grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, local_pos, &index_plane, &index_line);
	COMPUTE_T dL_dx_dim = 0;
	#pragma unroll
	for (uint32_t f = 0; f < N_FEAT; ++f) {
		dL_dx_dim += (COMPUTE_T)grid[index_plane + f] * (COMPUTE_T)grid[index_line+f] * (COMPUTE_T)grad[f] * weight;
	}
	return dL_dx_dim;
}

template <uint32_t N_POS_DIMS, typename INPUT_T, typename COMPUTE_T>
__device__ inline void pos_fract(
	const INPUT_T positions_in[N_POS_DIMS], 
	const COMPUTE_T scale[N_POS_DIMS], 
	const InterpolationType interpolation_type, 
	uint32_t pos_grid[N_POS_DIMS], 
	COMPUTE_T pos[N_POS_DIMS]
) {
	switch (interpolation_type)
	{
	case InterpolationType::Linear:
	{
		#pragma unroll
		for (uint32_t dim=0; dim<N_POS_DIMS; ++dim) {
			COMPUTE_T val = positions_in[dim] * scale[dim] + (COMPUTE_T)0.5f;
			val -= (COMPUTE_T)(pos_grid[dim] = floorf(val));
			pos[dim] = val;
		}
	}
	break;

	case InterpolationType::Smoothstep:
	{
		#pragma unroll
		for (uint32_t dim=0; dim<N_POS_DIMS; ++dim) {
			COMPUTE_T val = positions_in[dim] * scale[dim] + (COMPUTE_T)0.5f;
			val -= (COMPUTE_T)(pos_grid[dim] = floorf(val));
			pos[dim] = val * val * ((COMPUTE_T)3.0f - (COMPUTE_T)2.0f * (COMPUTE_T)val);
		}
	}
	break;
	
	default:
		break;
	}
}

template <uint32_t N_POS_DIMS, typename INPUT_T, typename COMPUTE_T>
__device__ inline void pos_fract(
	const INPUT_T positions_in[N_POS_DIMS], 
	const COMPUTE_T scale[N_POS_DIMS], 
	const InterpolationType interpolation_type, 
	uint32_t pos_grid[N_POS_DIMS], 
	COMPUTE_T pos[N_POS_DIMS], 
	COMPUTE_T pos_derivative[N_POS_DIMS]
) {
	switch (interpolation_type)
	{
	case InterpolationType::Linear:
	{
		#pragma unroll
		for (uint32_t dim=0; dim<N_POS_DIMS; ++dim) {
			COMPUTE_T val = positions_in[dim] * scale[dim] + (COMPUTE_T)0.5f;
			val -= (COMPUTE_T)(pos_grid[dim] = floorf(val));
			pos[dim] = val;
			pos_derivative[dim] = 1.0f;
		}
	}
	break;

	case InterpolationType::Smoothstep:
	{
		#pragma unroll
		for (uint32_t dim=0; dim<N_POS_DIMS; ++dim) {
			COMPUTE_T val = positions_in[dim] * scale[dim] + (COMPUTE_T)0.5f;
			val -= (COMPUTE_T)(pos_grid[dim] = floorf(val));
			pos[dim] = val * val * ((COMPUTE_T)3.0f - (COMPUTE_T)2.0f * val);
			pos_derivative[dim] = (COMPUTE_T)6.0f * val * ((COMPUTE_T)1.0f - val);
		}
	}
	break;
	
	default:
		break;
	}
}

template <uint32_t N_POS_DIMS, typename INPUT_T, typename COMPUTE_T>
__device__ inline void pos_fract(
	const INPUT_T positions_in[N_POS_DIMS], 
	const COMPUTE_T scale[N_POS_DIMS], 
	const InterpolationType interpolation_type, 
	uint32_t pos_grid[N_POS_DIMS], 
	COMPUTE_T pos[N_POS_DIMS], 
	COMPUTE_T pos_derivative[N_POS_DIMS], 
	COMPUTE_T pos_2nd_derivative[N_POS_DIMS]
) {
	switch (interpolation_type)
	{
	case InterpolationType::Linear:
	{
		#pragma unroll
		for (uint32_t dim=0; dim<N_POS_DIMS; ++dim) {
			COMPUTE_T val = positions_in[dim] * scale[dim] + (COMPUTE_T)0.5f;
			val -= (COMPUTE_T)(pos_grid[dim] = floorf(val));
			pos[dim] = val;
			pos_derivative[dim] = 1.0f;
			pos_2nd_derivative[dim] = 0;
		}
	}
	break;

	case InterpolationType::Smoothstep:
	{
		#pragma unroll
		for (uint32_t dim=0; dim<N_POS_DIMS; ++dim) {
			COMPUTE_T val = positions_in[dim] * scale[dim] + (COMPUTE_T)0.5f;
			val -= (COMPUTE_T)(pos_grid[dim] = floorf(val));
			pos[dim] = val * val * ((COMPUTE_T)3.0f - (COMPUTE_T)2.0f * val);
			pos_derivative[dim] = (COMPUTE_T)6.0f * val * ((COMPUTE_T)1.0f - val);
			pos_2nd_derivative[dim] = (COMPUTE_T)6.0f - (COMPUTE_T)12.0f * val;
		}
	}
	break;
	
	default:
		break;
	}
}

// #include <stdint.h>
// #include <string>
// #include <algorithm>
// #include <stdexcept>
// #include <vector>

template <typename scalar_t>
inline __host__ __device__ scalar_t div_round_up(scalar_t val, scalar_t divisor) {
    return (val + divisor - 1) / divisor;
}

template<typename T>
inline bool is_divisible(const T number, const T n) {
	return ((number - (number / n) * n) == (T)0);
}

template<typename T>
inline bool is_all_divisible(const std::vector<T>& numbers, const T n) {
	for (auto item: numbers) {
		if ( !is_divisible(item, n) ) return false;
	}
	return true;
}

inline uint32_t powi(uint32_t base, uint32_t exponent) {
	uint32_t result = 1;
	for (uint32_t i = 0; i < exponent; ++i) {
		result *= base;
	}
	return result;
}


} // namespace lotd



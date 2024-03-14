/** @file   permuto_cuda.h
 *  @author Jianfei Guo, Shanghai AI Lab
 *  @brief  A re-implementation of the permutohedral encoding.

New features:
- Support half(float16) param dtype
- Support 2 <= n_levels <= 20
- Support n_feats >= 2
- Support different layers using different widths (n_feats)
- Support batched inference with batch inds or batched input

Original: https://github.com/RaduAlexandru/permutohedral_encoding

Citation: 
@inproceedings{rosu2023permutosdf,
	title={PermutoSDF: Fast Multi-View Reconstruction with 
			Implicit Surfaces using Permutohedral Lattices  },
	author={Radu Alexandru Rosu and Sven Behnke},
	booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
	year={2023}
}
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

#include "permuto.h"

namespace permuto {

static constexpr uint32_t MAX_N_LEVELS = 24;
static constexpr uint32_t N_THREADS = 128;
static constexpr uint32_t N_THREADS_BACK = 128;

struct PermutoEncMetaRef { // GPU Reference structure for PermutoEncMeta
	// Do not exceeds 4096 Bytes of constant memory
	float level_scales0[MAX_N_LEVELS];	        // [n_levels]	Original raw positions scales per `level`
	uint32_t level_n_feats[MAX_N_LEVELS];	// [n_levels]	Feature width for each actual `level`
	uint32_t level_n_params[MAX_N_LEVELS];	// [n_levels]	Parameter sizes for each actual `level` (considering feature width)
	uint32_t level_offsets[MAX_N_LEVELS+1];	// [n_levels+1]	Parameter offsets for each actual `level` (considering feature width)
	uint32_t level_sizes[MAX_N_LEVELS];		// [n_levels]	Hashmap sizes for each actual `level` (NOT considering feature width)

	uint8_t map_levels[MAX_N_LEVELS * 4];	// [n_pseudo_levels]	Actual `level` corresponding to each pseudo `level`
	uint8_t map_cnt[MAX_N_LEVELS * 4];		// [n_pseudo_levels]	Index of the current pseudo `level` in all the pseudo `level`s corresponding to current actual `level`

	uint32_t n_levels = 0;					// Number of actual levels (allow non-equal feature width)
	uint32_t n_pseudo_levels = 0; 			// Number of pseudo levels (all equal feature width = `n_feat_per_pseudo_lvl`)
	uint32_t n_feat_per_pseudo_lvl = 0;		// Feature width of each pseudo level = the greatest common divisor of all acutal levels' widths 
	uint32_t n_dims_to_encode = 3;			// Number of dims to encode (in_features)
	uint32_t n_encoded_dims = 0;			// Number of encoded dims (out_features)

	__host__ PermutoEncMetaRef(const PermutoEncMeta& meta): 
		n_levels{meta.n_levels}, 
		n_pseudo_levels{meta.n_pseudo_levels}, 
		n_feat_per_pseudo_lvl{meta.n_feat_per_pseudo_lvl}, 
		n_dims_to_encode{meta.n_dims_to_encode}, 
		n_encoded_dims{meta.n_encoded_dims} 
	{
		for (uint32_t l=0; l<meta.n_levels; ++l) {
			level_scales0[l] = meta.level_scales0[l]; 
			level_n_feats[l] = meta.level_n_feats[l];
			level_n_params[l] = meta.level_n_params[l];
			level_offsets[l] = meta.level_offsets[l];
			level_sizes[l] = meta.level_sizes[l];
		}
		level_offsets[meta.n_levels] = meta.level_offsets[meta.n_levels];
		for (uint32_t psl=0; psl<meta.n_pseudo_levels; ++psl) {
			map_levels[psl] = meta.map_levels[psl];
			map_cnt[psl] = meta.map_cnt[psl]; 
		}
	}
};


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
__device__ __forceinline__
uint32_t hash(const int32_t key[N_POS_DIMS]) {
	uint32_t k = 0;
	#pragma unroll
	for (uint32_t dim=0; dim<N_POS_DIMS;++dim) {
		k += key[dim];
		k *= 2531011;
	}
	return k;
}

template <uint32_t N_POS_DIMS>
__device__ __forceinline__ uint32_t index_hash(
	const int32_t key[N_POS_DIMS], 
	const uint32_t hashmap_size
) {
	return (hash<N_POS_DIMS>(key) % hashmap_size);
}


template <typename INPUT_T, typename PARAM_T, typename COMPUTE_T, uint32_t N_POS_DIMS, uint32_t N_FEAT_PER_PSEUDO_LVL>
__global__ void 
__launch_bounds__(N_THREADS) 
kernel_permutohedral(
	const uint32_t num_elements, 
	const PermutoEncMetaRef meta, 
	int32_t max_level, 
	const int32_t* __restrict__ max_level_gpu,  // [n_points]
	// Inputs
	const INPUT_T* __restrict__ positions,      // [n_points, N_POS_DIMS]
	const PARAM_T* __restrict__ lattice_values, 
	// Fixed inputs
	const INPUT_T* __restrict__ level_scales_multidim, // [n_levels, N_POS_DIMS]
	const INPUT_T* __restrict__ level_random_shifts, // [n_levels, N_POS_DIMS]
	// Optional inputs
	const int64_t* __restrict__ batch_inds,     // [n_points]
	const int64_t* __restrict__ batch_offsets,  // [n_batch]
	const uint32_t batch_data_size,
	// Outputs
	PARAM_T* __restrict__ encoded               // [n_points, n_encoded_dims]
	// int32_t* __restrict__ rank_,                // [n_points, n_levels, N_POS_DIMS+1] Optional extra stored rank
	// int32_t* __restrict__ rem0_                 // [n_points, n_levels, N_POS_DIMS+1] Optional extra stored remainder-0 point coords
	// INPUT_T* __restrict__ elevated_             // [n_points, n_levels, N_POS_DIMS+1] Optional extra stored elevated (d+1) coords
) {
	const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num_elements) return;

	const uint32_t pseudo_lvl = blockIdx.y; // Pseudo level index for the lease common multiple of `level_n_feats`
	const uint32_t level = meta.map_levels[pseudo_lvl]; // Actual level index for `level_xxx`
	// The lattice offset of Current actual level (usally related to accumulated parameter sizes)
	// + the feature offset of current pseudo level in all the pseudo levels corresponding to current actual level.
	const uint32_t lattice_feat_offset = meta.level_offsets[level] + meta.map_cnt[pseudo_lvl] * N_FEAT_PER_PSEUDO_LVL; // Offset in the lattice features
	const uint32_t out_feat_offset = pseudo_lvl * N_FEAT_PER_PSEUDO_LVL; // Offset in the output features
	const uint32_t size_cur_lvl = meta.level_sizes[level]; 

	// printf("i[%d]: pslvl=%d, lvl=%d, lattice_feat_offset=%d, out_feat_offset=%d\n", i, pseudo_lvl, level, lattice_feat_offset, out_feat_offset);

	if (max_level_gpu) {
		max_level = max_level_gpu[i];
	}

	if (level > max_level)
		return;

	//---- For batched:
	// 1. First calculate `batch_ind` of current point
	// 2. Then offset `lattice_values` to the caculated batch
	uint32_t batch_ind = 0;
	if (batch_inds) {
		// NOTE: pass in batch_ind=-1 to ignore certain points.
		if (batch_inds[i] < 0)
			return;
		batch_ind = batch_inds[i];
	} else if (batch_data_size) {
		batch_ind = i / batch_data_size;
	}
	const uint32_t batch_offset = batch_offsets ? batch_offsets[batch_ind] : (batch_ind * meta.level_offsets[meta.n_levels]);
	//---- Offset `lattice_values` to the current batch + current actual level + current pseudo level in local pseudo levels
	lattice_values += (batch_offset + lattice_feat_offset);

	//---- Offset `positions` to the current point index
	positions += i * N_POS_DIMS; 

	const uint32_t n_feat_cur_lvl = meta.level_n_feats[level]; // Number of feature of current actual level
	COMPUTE_T scale_cur_lvl[N_POS_DIMS];
	COMPUTE_T pos[N_POS_DIMS];
	#pragma unroll
	for (uint32_t dim=0; dim<N_POS_DIMS; ++dim) {
		scale_cur_lvl[dim] = level_scales_multidim[level * N_POS_DIMS + dim]; 
		pos[dim] = positions[dim]; 
	}
	COMPUTE_T random_shifts_cur_lvl[N_POS_DIMS]{0};
	if (level_random_shifts) {
		#pragma unroll
		for (uint32_t dim=0; dim<N_POS_DIMS; ++dim) {
			random_shifts_cur_lvl[dim] = level_random_shifts[level * N_POS_DIMS + dim]; 
		}
	}

	//---- Elevate d-dimension vector to (d+1)-dimension homogeneous vector on hyperplane H_d
	// 1. `level_random_shifts` is to randomly shifts positions on different levels and different dims, 
	//    to minimize collision
	// 2. `elevated` is calculated in a way that:
	//    a) The sum of the components of `elevated` is zero, ensuring it within hyperplane H_d
	//    b) The magnitudes of the components of `elevated` are similar to each other.
	COMPUTE_T elevated[N_POS_DIMS + 1];
	COMPUTE_T sm = 0;
	#pragma unroll
	for (int32_t dim = N_POS_DIMS; dim > 0; dim--) {
		COMPUTE_T cf = (pos[dim-1] + (COMPUTE_T)random_shifts_cur_lvl[dim-1]) * (COMPUTE_T)scale_cur_lvl[dim-1]; 
		elevated[dim] = sm - (COMPUTE_T)dim * cf; 
		sm += cf;
	}
	elevated[0] = sm;



	//---- Find the closest remainder-0 point through rounding
	int32_t rem0[N_POS_DIMS+1]; // The coords of remainder-0 point
	int32_t rank[N_POS_DIMS+1]{0};
	int32_t sum = 0;
	#pragma unroll
	for (uint32_t dim=0; dim<=N_POS_DIMS; ++dim) {
		float v = elevated[dim] / (COMPUTE_T)(N_POS_DIMS+1); 
		int32_t down = (int32_t)floorf(v) * (int32_t)(N_POS_DIMS+1); 
		int32_t up = down + (int32_t)(N_POS_DIMS+1); 
		if ((COMPUTE_T)up - elevated[dim] < elevated[dim] - (COMPUTE_T)down) {
			rem0[dim] = up;
		} else {
			rem0[dim] = down;
		}
		sum += rem0[dim];
	}
	sum /= (int32_t)(N_POS_DIMS+1); // Must convert to int32_t first

	// Find the simplex we are in and store it in rank
	//  (where rank describes what position coordinate i has in the sorted order of the features values)
	#pragma unroll
	for (uint32_t dim=0; dim<N_POS_DIMS; ++dim) {
		COMPUTE_T di = elevated[dim] - (COMPUTE_T)rem0[dim]; // \vec{\Delta} = \vec{x}-\vec{\nu}_0
		for (uint32_t other_dim=dim+1; other_dim<=N_POS_DIMS; ++other_dim) {
			if (di < elevated[other_dim] - (COMPUTE_T)rem0[other_dim]) {
				rank[dim]++; 
			} else {
				rank[other_dim]++; 
			}
		}
	}

	// If the point doesn't lie on the plane (sum != 0) bring it back
	#pragma unroll
	for (uint32_t dim=0; dim<=N_POS_DIMS; ++dim) {
		rank[dim] += sum;
		if (rank[dim] < 0) {
			rank[dim] += (int32_t)(N_POS_DIMS+1); 
			rem0[dim] += (int32_t)(N_POS_DIMS+1); 
		} else if (rank[dim] > (int32_t)N_POS_DIMS) {
			rank[dim] -= (int32_t)(N_POS_DIMS+1);
			rem0[dim] -= (int32_t)(N_POS_DIMS+1); 
		}
	}



	// //---- Optionally store the calculated `rem0`, `rank` and `elevated`
	// if (rank_ && rem0_) {
	// 	const uint32_t intermediate_offset = (i * meta.n_levels + level) * (N_POS_DIMS+1); 
	// 	rank_ += intermediate_offset;
	// 	rem0_ += intermediate_offset; 
	// 	// elevated_ += intermediate_offset; 
	// 	#pragma unroll
	// 	for (uint32_t dim=0; dim<=N_POS_DIMS; ++dim) {
	// 		rem0_[dim] = rem0[dim];
	// 		rank_[dim] = rank[dim];
	// 		// elevated_[dim] = elevated[dim]; 
	// 	}
	// }


	//---- Compute the barycentric coordinates (p.10 in [Adams etal 2010])
	COMPUTE_T barycentric[N_POS_DIMS + 2]{0};
	#pragma unroll
	for (uint32_t dim=0; dim<=N_POS_DIMS; ++dim) {
		COMPUTE_T delta = (elevated[dim] - (COMPUTE_T)rem0[dim]) / (COMPUTE_T)(N_POS_DIMS+1); 
		barycentric[(int32_t)N_POS_DIMS - rank[dim]] += delta;
		barycentric[(int32_t)N_POS_DIMS + 1 - rank[dim]] -= delta; 
	}
	// TODO? Wrap around
	barycentric[0] += (COMPUTE_T)1.0f + barycentric[N_POS_DIMS + 1];


	//---- Interpolate the values to calculate encoded
	vector_t<PARAM_T, N_FEAT_PER_PSEUDO_LVL> result{0};
	int32_t key[N_POS_DIMS]; 

	#pragma unroll 1 // Force skip unrolling
	for (uint32_t k=0; k<=N_POS_DIMS; ++k) { // remainder-k vertex, for k \in {0,1,...,d}
		// Compute the coordinates of the remainder-k vertex explicitly
		// (all but the last coordinate - it's redundant because they sum to zero)
		#pragma unroll
		for (uint32_t dim=0; dim < N_POS_DIMS; ++dim) {
			key[dim] = rem0[dim] + (int32_t)k; 
			if (rank[dim] > (int32_t)(N_POS_DIMS-k))
				key[dim] -= (int32_t)(N_POS_DIMS+1);
		}

		// Retrieve pointer to the value at this vertex.
		uint32_t index = index_hash<N_POS_DIMS>(key, size_cur_lvl) * n_feat_cur_lvl;

		// Accumulate vertex's feature value by the barycentric weight
		COMPUTE_T weight = barycentric[k];

		// Vectorized loads
		auto val = *(vector_t<PARAM_T, N_FEAT_PER_PSEUDO_LVL>*)&lattice_values[index];

		#pragma unroll
		for (uint32_t f=0; f<N_FEAT_PER_PSEUDO_LVL; ++f) {
			((PARAM_T*)&result)[f] += (PARAM_T)( weight * (COMPUTE_T)((PARAM_T*)&val)[f] );
		}
	}

	encoded += i * meta.n_encoded_dims + out_feat_offset;
	#pragma unroll
	for (uint32_t f=0; f<N_FEAT_PER_PSEUDO_LVL; ++f) {
		encoded[f] = ((PARAM_T*)&result)[f];
	}
}


template <typename INPUT_T, typename PARAM_T, typename GRAD_T, typename COMPUTE_T, uint32_t N_POS_DIMS, uint32_t N_FEAT_PER_PSEUDO_LVL, uint32_t N_FEAT_PER_THREAD>
__global__ void 
__launch_bounds__(N_THREADS_BACK) 
kernel_permutohedral_backward_lattice(
	const uint32_t num_elements, 
	const PermutoEncMetaRef meta, 
	int32_t max_level, 
	const int32_t* __restrict__ max_level_gpu,  // [n_points]
	// Inputs
	const PARAM_T* __restrict__ dL_dy,          // [n_points, n_encoded_dims]
	const INPUT_T* __restrict__ positions,      // [n_points, N_POS_DIMS]
	const PARAM_T* __restrict__ lattice_values, 
	// Fixed inputs
	const INPUT_T* __restrict__ level_scales_multidim, // [n_levels, N_POS_DIMS]
	const INPUT_T* __restrict__ level_random_shifts, // [n_levels, N_POS_DIMS]
	// Stored data
	// const int32_t* __restrict__ rank_,          // [n_points, N_POS_DIMS+1] Optional extra stored rank
	// const int32_t* __restrict__ rem0_,          // [n_points, N_POS_DIMS+1] Optional extra stored remainder-0 point coords
	// const INPUT_T* __restrict__ elevated_,      // [n_points, N_POS_DIMS+1] Optional extra stored elevated (d+1) coords
	// Optional inputs
	const int64_t* __restrict__ batch_inds,     // [n_points]
	const int64_t* __restrict__ batch_offsets,  // [n_batch]
	const uint32_t batch_data_size,
	// Outputs
	GRAD_T* __restrict__ grad_lattice_values
) {
	const uint32_t i = ((blockIdx.x * blockDim.x + threadIdx.x) * N_FEAT_PER_THREAD) / N_FEAT_PER_PSEUDO_LVL;
	if (i >= num_elements) return;

	const uint32_t pseudo_lvl = blockIdx.y; // Pseudo level index for the lease common multiple of `level_n_feats`
	const uint32_t level = meta.map_levels[pseudo_lvl]; // Actual level index for `level_xxx`
	// The lattice offset of Current actual level (usally related to accumulated parameter sizes)
	// + the feature offset of current pseudo level in all the pseudo levels corresponding to current actual level.
	const uint32_t thread_feat_offset = (blockIdx.x * blockDim.x + threadIdx.x) * N_FEAT_PER_THREAD - i * N_FEAT_PER_PSEUDO_LVL;
	const uint32_t lattice_feat_offset = thread_feat_offset + meta.level_offsets[level] + meta.map_cnt[pseudo_lvl] * N_FEAT_PER_PSEUDO_LVL; // Offset in the lattice features
	const uint32_t out_feat_offset = thread_feat_offset + pseudo_lvl * N_FEAT_PER_PSEUDO_LVL; // Offset in the output features
	const uint32_t size_cur_lvl = meta.level_sizes[level]; 

	if (max_level_gpu) {
		max_level = max_level_gpu[i];
	}

	if (level > max_level) {
		return;
	}

	//---- For batched:
	// 1. First calculate `batch_ind` of current point
	// 2. Then offset `lattice_values`, `grad_lattice_values` to the caculated batch
	uint32_t batch_ind = 0;
	if (batch_inds) {
		// NOTE: pass in batch_ind=-1 to ignore certain points.
		if (batch_inds[i] < 0) { return;}
		batch_ind = batch_inds[i];
	} else if (batch_data_size) {
		batch_ind = i / batch_data_size;
	}
	const uint32_t batch_offset = batch_offsets ? batch_offsets[batch_ind] : (batch_ind * meta.level_offsets[meta.n_levels]);
	//---- Offset `lattice_values` to the current batch + current actual level + current pseudo level in local pseudo levels
	lattice_values += (batch_offset + lattice_feat_offset);
	grad_lattice_values += (batch_offset + lattice_feat_offset);

	//---- Offset `positions` to the current point index
	positions += i * N_POS_DIMS; 

	const uint32_t n_feat_cur_lvl = meta.level_n_feats[level]; // Number of feature of current actual level
	COMPUTE_T scale_cur_lvl[N_POS_DIMS];
	COMPUTE_T pos[N_POS_DIMS];
	#pragma unroll
	for (uint32_t dim=0; dim<N_POS_DIMS; ++dim) {
		scale_cur_lvl[dim] = level_scales_multidim[level * N_POS_DIMS + dim]; 
		pos[dim] = positions[dim]; 
	}
	COMPUTE_T random_shifts_cur_lvl[N_POS_DIMS]{0};
	if (level_random_shifts) {
		#pragma unroll
		for (uint32_t dim=0; dim<N_POS_DIMS; ++dim) {
			random_shifts_cur_lvl[dim] = level_random_shifts[level * N_POS_DIMS + dim]; 
		}
	}

	//---- Offset `dL_dy` to the current point index and out_feat_offset
	dL_dy += i * meta.n_encoded_dims + out_feat_offset;
	vector_t<COMPUTE_T, N_FEAT_PER_THREAD> dL_dy_i;
	// COMPUTE_T dL_dy_i[N_FEAT_PER_THREAD];
	#pragma unroll
	for (uint32_t f=0; f<N_FEAT_PER_THREAD; ++f) {
		dL_dy_i[f] = dL_dy[f];
	}


	//---- Elevate d-dimension vector to (d+1)-dimension homogeneous vector on hyperplane H_d
	// 1. `level_random_shifts` is to randomly shifts positions on different levels and different dims, 
	//    to minimize collision
	// 2. `elevated` is calculated in a way that:
	//    a) The sum of the components of `elevated` is zero, ensuring it within hyperplane H_d
	//    b) The magnitudes of the components of `elevated` are similar to each other.
	COMPUTE_T elevated[N_POS_DIMS + 1];
	COMPUTE_T sm = 0;
	#pragma unroll
	for (int32_t dim = N_POS_DIMS; dim > 0; dim--) {
		COMPUTE_T cf = (pos[dim-1] + (COMPUTE_T)random_shifts_cur_lvl[dim-1]) * (COMPUTE_T)scale_cur_lvl[dim-1]; 
		elevated[dim] = sm - (COMPUTE_T)dim * cf; 
		sm += cf;
	}
	elevated[0] = sm;

	//---- Find the closest remainder-0 point through rounding
	int32_t rem0[N_POS_DIMS+1]; // The coords of remainder-0 point
	int32_t rank[N_POS_DIMS+1]{0};
	int32_t sum = 0;
	#pragma unroll
	for (uint32_t dim=0; dim<=N_POS_DIMS; ++dim) {
		float v = elevated[dim] / (COMPUTE_T)(N_POS_DIMS+1); 
		int32_t down = (int32_t)floorf(v) * (int32_t)(N_POS_DIMS+1); 
		int32_t up = down + (int32_t)(N_POS_DIMS+1); 
		if ((COMPUTE_T)up - elevated[dim] < elevated[dim] - (COMPUTE_T)down) {
			rem0[dim] = up;
		} else {
			rem0[dim] = down;
		}
		sum += rem0[dim];
	}
	sum /= (int32_t)(N_POS_DIMS+1); // Must convert to int32_t first

	// Find the simplex we are in and store it in rank
	//  (where rank describes what position coordinate i has in the sorted order of the features values)
	#pragma unroll
	for (uint32_t dim=0; dim<N_POS_DIMS; ++dim) {
		COMPUTE_T di = elevated[dim] - (COMPUTE_T)rem0[dim]; // \vec{\Delta} = \vec{x}-\vec{\nu}_0
		for (uint32_t other_dim=dim+1; other_dim<=N_POS_DIMS; ++other_dim) {
			if (di < elevated[other_dim] - (COMPUTE_T)rem0[other_dim]) {
				rank[dim]++; 
			} else {
				rank[other_dim]++; 
			}
		}
	}

	// If the point doesn't lie on the plane (sum != 0) bring it back
	#pragma unroll
	for (uint32_t dim=0; dim<=N_POS_DIMS; ++dim) {
		rank[dim] += sum;
		if (rank[dim] < 0) {
			rank[dim] += (int32_t)(N_POS_DIMS+1); 
			rem0[dim] += (int32_t)(N_POS_DIMS+1); 
		} else if (rank[dim] > (int32_t)N_POS_DIMS) {
			rank[dim] -= (int32_t)(N_POS_DIMS+1);
			rem0[dim] -= (int32_t)(N_POS_DIMS+1); 
		}
	}



	//---- Compute the barycentric coordinates (p.10 in [Adams etal 2010])
	COMPUTE_T barycentric[N_POS_DIMS + 2]{0};
	#pragma unroll
	for (uint32_t dim=0; dim<=N_POS_DIMS; ++dim) {
		COMPUTE_T delta = (elevated[dim] - (COMPUTE_T)rem0[dim]) / (COMPUTE_T)(N_POS_DIMS+1); 
		barycentric[(int32_t)N_POS_DIMS - rank[dim]] += delta;
		barycentric[(int32_t)N_POS_DIMS + 1 - rank[dim]] -= delta; 
	}
	// TODO? Wrap around
	barycentric[0] += (COMPUTE_T)1.0f + barycentric[N_POS_DIMS + 1];


	auto add_lattice_gradient = [&](const uint32_t index, const vector_t<COMPUTE_T, N_FEAT_PER_THREAD>& grad, COMPUTE_T weight) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600 // atomicAdd(__half2) is only supported with compute capability 60 and above
		if (N_FEAT_PER_THREAD > 1 && std::is_same<GRAD_T, __half>::value)
		{
			for (uint32_t f = 0; f < N_FEAT_PER_THREAD; f += 2) {
				__half2 v = {(__half)((COMPUTE_T)grad[f] * weight), (__half)((COMPUTE_T)grad[f+1] * weight)};
				atomicAdd((__half2*)&grad_lattice_values[index + f], v);
			}
		} else 
#endif
		{
			if (std::is_same<GRAD_T, __half>::value) {
				// Should never happen
				//printf("Attempted to use atomicAdd(__half)\n")
			} else {
				// NOTE: Here we perform a reduction on `grad_lattice_values` using `atomicAdd`.
				#pragma unroll
				for (uint32_t f=0; f<N_FEAT_PER_THREAD; ++f) {
					atomicAdd((float*)&grad_lattice_values[index + f], (float)((COMPUTE_T)grad[f] * weight));
				}
			}
		}
	};

	//---- Interpolate the values to calculate dL_dlattice
	int32_t key[N_POS_DIMS]; 

	#pragma unroll 1 // Force skip unrolling
	for (uint32_t k=0; k<=N_POS_DIMS; ++k) { // remainder-k vertex, for k \in {0,1,...,d}
		// Compute the coordinates of the remainder-k vertex explicitly
		// (all but the last coordinate - it's redundant because they sum to zero)
		#pragma unroll
		for (uint32_t dim=0; dim < N_POS_DIMS; ++dim) {
			key[dim] = rem0[dim] + (int32_t)k; 
			if (rank[dim] > (int32_t)(N_POS_DIMS-k))
				key[dim] -= (int32_t)(N_POS_DIMS+1);
		}

		// Retrieve pointer to the value at this vertex.
		uint32_t index = index_hash<N_POS_DIMS>(key, size_cur_lvl) * n_feat_cur_lvl;

		// Accumulate vertex's gradients by the barycentric weight
		COMPUTE_T weight = barycentric[k];

		add_lattice_gradient(index, dL_dy_i, weight); 
	}
}


template <typename INPUT_T, typename PARAM_T, typename GRAD_T, typename COMPUTE_T, uint32_t N_POS_DIMS, uint32_t N_FEAT_PER_PSEUDO_LVL, uint32_t N_FEAT_PER_THREAD>
__global__ void 
__launch_bounds__(N_THREADS_BACK) 
kernel_permutohedral_backward_input(
	const uint32_t num_elements, 
	const PermutoEncMetaRef meta, 
	int32_t max_level, 
	const int32_t* __restrict__ max_level_gpu,  // [n_points]
	const uint32_t max_pos_dims, 
	// Inputs
	const PARAM_T* __restrict__ dL_dy,          // [n_points, n_encoded_dims]
	const INPUT_T* __restrict__ positions,      // [n_points, N_POS_DIMS]
	const PARAM_T* __restrict__ lattice_values, 
	// Fixed inputs
	const INPUT_T* __restrict__ level_scales_multidim, // [n_levels, N_POS_DIMS]
	const INPUT_T* __restrict__ level_random_shifts, // [n_levels, N_POS_DIMS]
	// Stored data
	// const int32_t* __restrict__ rank_,          // [n_points, N_POS_DIMS+1] Optional extra stored rank
	// const int32_t* __restrict__ rem0_,          // [n_points, N_POS_DIMS+1] Optional extra stored remainder-0 point coords
	// const INPUT_T* __restrict__ elevated_,      // [n_points, N_POS_DIMS+1] Optional extra stored elevated (d+1) coords
	// Optional inputs
	const int64_t* __restrict__ batch_inds,     // [n_points]
	const int64_t* __restrict__ batch_offsets,  // [n_batch]
	const uint32_t batch_data_size,
	// Outputs
	INPUT_T* __restrict__ dL_dx                 // [n_points, N_POS_DIMS]
) {
	const uint32_t i = ((blockIdx.x * blockDim.x + threadIdx.x) * N_FEAT_PER_THREAD) / N_FEAT_PER_PSEUDO_LVL;
	if (i >= num_elements) return;

	const uint32_t pseudo_lvl = blockIdx.y; // Pseudo level index for the lease common multiple of `level_n_feats`
	const uint32_t level = meta.map_levels[pseudo_lvl]; // Actual level index for `level_xxx`
	// The lattice offset of Current actual level (usally related to accumulated parameter sizes)
	// + the feature offset of current pseudo level in all the pseudo levels corresponding to current actual level.
	const uint32_t thread_feat_offset = (blockIdx.x * blockDim.x + threadIdx.x) * N_FEAT_PER_THREAD - i * N_FEAT_PER_PSEUDO_LVL;
	const uint32_t lattice_feat_offset = thread_feat_offset + meta.level_offsets[level] + meta.map_cnt[pseudo_lvl] * N_FEAT_PER_PSEUDO_LVL; // Offset in the lattice features
	const uint32_t out_feat_offset = thread_feat_offset + pseudo_lvl * N_FEAT_PER_PSEUDO_LVL; // Offset in the output features
	const uint32_t size_cur_lvl = meta.level_sizes[level]; 

	// printf("i[%d]: pslvl=%d, lvl=%d, thread_feat_offset=%d, lattice_feat_offset=%d, out_feat_offset=%d\n", i, pseudo_lvl, level, thread_feat_offset, lattice_feat_offset, out_feat_offset);

	if (max_level_gpu) {
		max_level = max_level_gpu[i];
	}

	if (level > max_level) {
		return;
	}

	//---- For batched:
	// 1. First calculate `batch_ind` of current point
	// 2. Then offset `lattice_values`, `grad_lattice_values` to the caculated batch
	uint32_t batch_ind = 0;
	if (batch_inds) {
		// NOTE: pass in batch_ind=-1 to ignore certain points.
		if (batch_inds[i] < 0) { return;}
		batch_ind = batch_inds[i];
	} else if (batch_data_size) {
		batch_ind = i / batch_data_size;
	}
	const uint32_t batch_offset = batch_offsets ? batch_offsets[batch_ind] : (batch_ind * meta.level_offsets[meta.n_levels]);
	//---- Offset `lattice_values` to the current batch + current actual level + current pseudo level in local pseudo levels
	lattice_values += (batch_offset + lattice_feat_offset);

	//---- Offset `positions` and `dL_dx` to the current point index
	positions += i * N_POS_DIMS;
	dL_dx += i * N_POS_DIMS;

	const uint32_t n_feat_cur_lvl = meta.level_n_feats[level]; // Number of feature of current actual level
	COMPUTE_T scale_cur_lvl[N_POS_DIMS];
	COMPUTE_T pos[N_POS_DIMS];
	#pragma unroll
	for (uint32_t dim=0; dim<N_POS_DIMS; ++dim) {
		scale_cur_lvl[dim] = level_scales_multidim[level * N_POS_DIMS + dim]; 
		pos[dim] = positions[dim]; 
	}
	COMPUTE_T random_shifts_cur_lvl[N_POS_DIMS]{0};
	if (level_random_shifts) {
		#pragma unroll
		for (uint32_t dim=0; dim<N_POS_DIMS; ++dim) {
			random_shifts_cur_lvl[dim] = level_random_shifts[level * N_POS_DIMS + dim]; 
		}
	}

	//---- Offset `dL_dy` to the current point index and out_feat_offset
	dL_dy += i * meta.n_encoded_dims + out_feat_offset;
	vector_t<COMPUTE_T, N_FEAT_PER_THREAD> dL_dy_i;
	// COMPUTE_T dL_dy_i[N_FEAT_PER_THREAD];
	#pragma unroll
	for (uint32_t f=0; f<N_FEAT_PER_THREAD; ++f) {
		dL_dy_i[f] = dL_dy[f];
	}


	//---- Elevate d-dimension vector to (d+1)-dimension homogeneous vector on hyperplane H_d
	// 1. `level_random_shifts` is to randomly shifts positions on different levels and different dims, 
	//    to minimize collision
	// 2. `elevated` is calculated in a way that:
	//    a) The sum of the components of `elevated` is zero, ensuring it within hyperplane H_d
	//    b) The magnitudes of the components of `elevated` are similar to each other.
	COMPUTE_T elevated[N_POS_DIMS + 1];
	COMPUTE_T sm = 0;
	#pragma unroll
	for (int32_t dim = N_POS_DIMS; dim > 0; dim--) {
		COMPUTE_T cf = (pos[dim-1] + (COMPUTE_T)random_shifts_cur_lvl[dim-1]) * (COMPUTE_T)scale_cur_lvl[dim-1]; 
		elevated[dim] = sm - (COMPUTE_T)dim * cf; 
		sm += cf;
	}
	elevated[0] = sm;



	//---- Find the closest remainder-0 point through rounding
	int32_t rem0[N_POS_DIMS+1]; // The coords of remainder-0 point
	int32_t rank[N_POS_DIMS+1]{0};
	int32_t sum = 0;
	#pragma unroll
	for (uint32_t dim=0; dim<=N_POS_DIMS; ++dim) {
		float v = elevated[dim] / (COMPUTE_T)(N_POS_DIMS+1); 
		int32_t down = (int32_t)floorf(v) * (int32_t)(N_POS_DIMS+1); 
		int32_t up = down + (int32_t)(N_POS_DIMS+1); 
		if ((COMPUTE_T)up - elevated[dim] < elevated[dim] - (COMPUTE_T)down) {
			rem0[dim] = up;
		} else {
			rem0[dim] = down;
		}
		sum += rem0[dim];
	}
	sum /= (int32_t)(N_POS_DIMS+1); // Must convert to int32_t first

	// Find the simplex we are in and store it in rank
	//  (where rank describes what position coordinate i has in the sorted order of the features values)
	#pragma unroll
	for (uint32_t dim=0; dim<N_POS_DIMS; ++dim) {
		COMPUTE_T di = elevated[dim] - (COMPUTE_T)rem0[dim]; // \vec{\Delta} = \vec{x}-\vec{\nu}_0
		for (uint32_t other_dim=dim+1; other_dim<=N_POS_DIMS; ++other_dim) {
			if (di < elevated[other_dim] - (COMPUTE_T)rem0[other_dim]) {
				rank[dim]++; 
			} else {
				rank[other_dim]++; 
			}
		}
	}

	// If the point doesn't lie on the plane (sum != 0) bring it back
	#pragma unroll
	for (uint32_t dim=0; dim<=N_POS_DIMS; ++dim) {
		rank[dim] += sum;
		if (rank[dim] < 0) {
			rank[dim] += (int32_t)(N_POS_DIMS+1); 
			rem0[dim] += (int32_t)(N_POS_DIMS+1); 
		} else if (rank[dim] > (int32_t)N_POS_DIMS) {
			rank[dim] -= (int32_t)(N_POS_DIMS+1);
			rem0[dim] -= (int32_t)(N_POS_DIMS+1); 
		}
	}



	//---- Calculate dL_dx
	// We have from upstrema grad the dL/dy which is the derivative of the loss wrt to the sliced (encoded) value
	// If we require positions grad we want to obtain dL/dx

	//---- dL/dx = dL/dy * dy/dB * dB/dE * dE/dx
	// We need dy/dB which is the derivative of the sliced value wrt to the barycentric coords
	// We need dB/dE which is the derivative of the barycentric wrt to the elevated value
	// We need dE/dx which is the derivative of the elevated wrt to the position in xyz
	COMPUTE_T dL_dbarycentric[N_POS_DIMS + 2]{0};
	int32_t key[N_POS_DIMS]; 

	#pragma unroll 1 // Force skip unrolling
	for (uint32_t k=0; k<=N_POS_DIMS; ++k) { // remainder-k vertex, for k \in {0,1,...,d}
		// Compute the coordinates of the remainder-k vertex explicitly
		// (all but the last coordinate - it's redundant because they sum to zero)
		#pragma unroll
		for (uint32_t dim=0; dim < N_POS_DIMS; ++dim) {
			key[dim] = rem0[dim] + (int32_t)k; 
			if (rank[dim] > (int32_t)(N_POS_DIMS-k))
				key[dim] -= (int32_t)(N_POS_DIMS+1);
		}

		// Retrieve pointer to the value at this vertex.
		uint32_t index = index_hash<N_POS_DIMS>(key, size_cur_lvl) * n_feat_cur_lvl;

		// Vectorized loads
		auto val = *(vector_t<PARAM_T, N_FEAT_PER_THREAD>*)&lattice_values[index];
		
		// Add to dL_d_barycentric
		#pragma unroll
		for (uint32_t f=0; f<N_FEAT_PER_THREAD; ++f) {
			dL_dbarycentric[k] += (COMPUTE_T)val[f] * dL_dy_i[f]; 
		}
	}
	dL_dbarycentric[N_POS_DIMS + 1] += dL_dbarycentric[0];

	//---- dL/dE = dL/dB *dB/dE
	COMPUTE_T dL_delevated[N_POS_DIMS + 1]{0};
	#pragma unroll
	for (uint32_t dim=0; dim<=N_POS_DIMS; ++dim) {
		dL_delevated[dim] += dL_dbarycentric[(int32_t)N_POS_DIMS - rank[dim]] / (COMPUTE_T)(N_POS_DIMS+1);
		dL_delevated[dim] -= dL_dbarycentric[(int32_t)(N_POS_DIMS + 1) - rank[dim]] / (COMPUTE_T)(N_POS_DIMS+1);
	}

	//---- dL/dx = dL/dE * dE/dx
	INPUT_T dL_dx_i[N_POS_DIMS]{0};
	#pragma unroll 1 // Force skip unrolling
	for (uint32_t dim=0; dim<max_pos_dims; ++dim) {
		COMPUTE_T dL_dx_dim = 0;
		#pragma unroll
		for (uint32_t other_dim=0; other_dim<=dim; ++other_dim) {
			dL_dx_dim += dL_delevated[other_dim] * scale_cur_lvl[dim];
		}
		dL_dx_dim -= dL_delevated[dim+1] * scale_cur_lvl[dim] * (COMPUTE_T)(dim+1);
		dL_dx_i[dim] = (INPUT_T)dL_dx_dim; 
	}

	//---- Finish
	#pragma unroll
	for (uint32_t dim=0; dim<max_pos_dims; ++dim) {
		// Should be atomic, since different levels of dL_dy can backward to the same input.
		atomicAdd(&dL_dx[dim], dL_dx_i[dim]); 
	}
}

template <typename INPUT_T, typename PARAM_T, typename GRAD_T, typename COMPUTE_T, uint32_t N_POS_DIMS, uint32_t N_FEAT_PER_PSEUDO_LVL, uint32_t N_FEAT_PER_THREAD>
__global__ void 
__launch_bounds__(N_THREADS_BACK) 
kernel_permutohedral_backward_backward_input(
	const uint32_t num_elements, 
	const PermutoEncMetaRef meta, 
	int32_t max_level, 
	const int32_t* __restrict__ max_level_gpu,  // [n_points]
	// Inputs
	const INPUT_T* __restrict__ dL_ddLdx,       // [n_points, N_POS_DIMS]
	const PARAM_T* __restrict__ dL_dy,          // [n_points, n_encoded_dims]
	const INPUT_T* __restrict__ positions,      // [n_points, N_POS_DIMS]
	const PARAM_T* __restrict__ lattice_values, 
	// Fixed inputs
	const INPUT_T* __restrict__ level_scales_multidim, // [n_levels, N_POS_DIMS]
	const INPUT_T* __restrict__ level_random_shifts, // [n_levels, N_POS_DIMS]
	// Stored data
	// const int32_t* __restrict__ rank_,          // [n_points, N_POS_DIMS+1] Optional extra stored rank
	// const int32_t* __restrict__ rem0_,          // [n_points, N_POS_DIMS+1] Optional extra stored remainder-0 point coords
	// const INPUT_T* __restrict__ elevated_,      // [n_points, N_POS_DIMS+1] Optional extra stored elevated (d+1) coords
	// Optional inputs
	const int64_t* __restrict__ batch_inds,     // [n_points]
	const int64_t* __restrict__ batch_offsets,  // [n_batch]
	const uint32_t batch_data_size,
	// Outputs
	GRAD_T* __restrict__ grad_lattice_values,   
	PARAM_T* __restrict__ dL_ddLdy              // [n_points, n_encoded_dims]
) {
	const uint32_t i = ((blockIdx.x * blockDim.x + threadIdx.x) * N_FEAT_PER_THREAD) / N_FEAT_PER_PSEUDO_LVL;
	if (i >= num_elements) return;

	const uint32_t pseudo_lvl = blockIdx.y; // Pseudo level index for the lease common multiple of `level_n_feats`
	const uint32_t level = meta.map_levels[pseudo_lvl]; // Actual level index for `level_xxx`
	// The lattice offset of Current actual level (usally related to accumulated parameter sizes)
	// + the feature offset of current pseudo level in all the pseudo levels corresponding to current actual level.
	const uint32_t thread_feat_offset = (blockIdx.x * blockDim.x + threadIdx.x) * N_FEAT_PER_THREAD - i * N_FEAT_PER_PSEUDO_LVL; // Nonzero iff `N_FEAT_PER_THREAD` != `N_FEAT_PER_PSEUDO_LVL`
	const uint32_t lattice_feat_offset = thread_feat_offset + meta.level_offsets[level] + meta.map_cnt[pseudo_lvl] * N_FEAT_PER_PSEUDO_LVL; // Offset in the lattice features
	const uint32_t out_feat_offset = thread_feat_offset + pseudo_lvl * N_FEAT_PER_PSEUDO_LVL; // Offset in the output features
	const uint32_t size_cur_lvl = meta.level_sizes[level]; 

	if (max_level_gpu) {
		max_level = max_level_gpu[i];
	}

	if (level > max_level) {
		return;
	}

	//---- For batched:
	// 1. First calculate `batch_ind` of current point
	// 2. Then offset `lattice_values`, `grad_lattice_values` to the caculated batch
	uint32_t batch_ind = 0;
	if (batch_inds) {
		// NOTE: pass in batch_ind=-1 to ignore certain points.
		if (batch_inds[i] < 0) { return;}
		batch_ind = batch_inds[i];
	} else if (batch_data_size) {
		batch_ind = i / batch_data_size;
	}
	const uint32_t batch_offset = batch_offsets ? batch_offsets[batch_ind] : (batch_ind * meta.level_offsets[meta.n_levels]);
	//---- Offset `lattice_values` to the current batch + current actual level + current pseudo level in local pseudo levels
	lattice_values += (batch_offset + lattice_feat_offset);

	//---- Offset `positions`, `dL_ddLdx` to the current point
	positions += i*N_POS_DIMS;
	dL_ddLdx += i*N_POS_DIMS;

	const uint32_t n_feat_cur_lvl = meta.level_n_feats[level]; // Number of feature of current actual level
	COMPUTE_T scale_cur_lvl[N_POS_DIMS];
	COMPUTE_T pos[N_POS_DIMS];
	#pragma unroll
	for (uint32_t dim=0; dim<N_POS_DIMS; ++dim) {
		scale_cur_lvl[dim] = level_scales_multidim[level * N_POS_DIMS + dim]; 
		pos[dim] = positions[dim]; 
	}
	COMPUTE_T random_shifts_cur_lvl[N_POS_DIMS]{0};
	if (level_random_shifts) {
		#pragma unroll
		for (uint32_t dim=0; dim<N_POS_DIMS; ++dim) {
			random_shifts_cur_lvl[dim] = level_random_shifts[level * N_POS_DIMS + dim]; 
		}
	}

	//---- Offset `dL_dy` to the current point index and out_feat_offset
	dL_dy += i * meta.n_encoded_dims + out_feat_offset;
	vector_t<COMPUTE_T, N_FEAT_PER_THREAD> dL_dy_i;
	// COMPUTE_T dL_dy_i[N_FEAT_PER_THREAD];
	#pragma unroll
	for (uint32_t f=0; f<N_FEAT_PER_THREAD; ++f) {
		dL_dy_i[f] = dL_dy[f]; 
	}

	COMPUTE_T dL_ddLdx_i[N_POS_DIMS];
	#pragma unroll
	for (uint32_t dim=0; dim<N_POS_DIMS; ++dim) {
		dL_ddLdx_i[dim] = dL_ddLdx[dim]; 
	}


	//---- Elevate d-dimension vector to (d+1)-dimension homogeneous vector on hyperplane H_d
	// 1. `level_random_shifts` is to randomly shifts positions on different levels and different dims, 
	//    to minimize collision
	// 2. `elevated` is calculated in a way that:
	//    a) The sum of the components of `elevated` is zero, ensuring it within hyperplane H_d
	//    b) The magnitudes of the components of `elevated` are similar to each other.
	COMPUTE_T elevated[N_POS_DIMS + 1];
	COMPUTE_T sm = 0;
	#pragma unroll
	for (int32_t dim = N_POS_DIMS; dim > 0; dim--) {
		COMPUTE_T cf = (pos[dim-1] + (COMPUTE_T)random_shifts_cur_lvl[dim-1]) * (COMPUTE_T)scale_cur_lvl[dim-1]; 
		elevated[dim] = sm - (COMPUTE_T)dim * cf; 
		sm += cf;
	}
	elevated[0] = sm;



	//---- Find the closest remainder-0 point through rounding
	int32_t rem0[N_POS_DIMS+1]; // The coords of remainder-0 point
	int32_t rank[N_POS_DIMS+1]{0};
	int32_t sum = 0;
	#pragma unroll
	for (uint32_t dim=0; dim<=N_POS_DIMS; ++dim) {
		float v = elevated[dim] / (COMPUTE_T)(N_POS_DIMS+1); 
		int32_t down = (int32_t)floorf(v) * (int32_t)(N_POS_DIMS+1); 
		int32_t up = down + (int32_t)(N_POS_DIMS+1); 
		if ((COMPUTE_T)up - elevated[dim] < elevated[dim] - (COMPUTE_T)down) {
			rem0[dim] = up;
		} else {
			rem0[dim] = down;
		}
		sum += rem0[dim];
	}
	sum /= (int32_t)(N_POS_DIMS+1); // Must convert to int32_t first

	// Find the simplex we are in and store it in rank
	//  (where rank describes what position coordinate i has in the sorted order of the features values)
	#pragma unroll
	for (uint32_t dim=0; dim<N_POS_DIMS; ++dim) {
		COMPUTE_T di = elevated[dim] - (COMPUTE_T)rem0[dim]; // \vec{\Delta} = \vec{x}-\vec{\nu}_0
		for (uint32_t other_dim=dim+1; other_dim<=N_POS_DIMS; ++other_dim) {
			if (di < elevated[other_dim] - (COMPUTE_T)rem0[other_dim]) {
				rank[dim]++; 
			} else {
				rank[other_dim]++; 
			}
		}
	}

	// If the point doesn't lie on the plane (sum != 0) bring it back
	#pragma unroll
	for (uint32_t dim=0; dim<=N_POS_DIMS; ++dim) {
		rank[dim] += sum;
		if (rank[dim] < 0) {
			rank[dim] += (int32_t)(N_POS_DIMS+1); 
			rem0[dim] += (int32_t)(N_POS_DIMS+1); 
		} else if (rank[dim] > (int32_t)N_POS_DIMS) {
			rank[dim] -= (int32_t)(N_POS_DIMS+1);
			rem0[dim] -= (int32_t)(N_POS_DIMS+1); 
		}
	}


	// We have upstream gradient `dL/dx`,
	// And we want to backward the gradient to `grad_lattice_values`, `dL_ddLdy`
	// dL/dy = dL/dx * dx/dE * dE/dB * dB/dy
	// dL/dV = dL/dx * dx/dE * dE/dB * dB/dV
	
	//---- dx/dE
	COMPUTE_T dL_delevated[N_POS_DIMS+1]{0};
	#pragma unroll 1 // Force skip unrolling
	for (uint32_t dim=0; dim<N_POS_DIMS; ++dim) {
		COMPUTE_T grad = dL_ddLdx_i[dim] * scale_cur_lvl[dim]; 
		#pragma unroll
		for (uint32_t other_dim=0; other_dim<=dim; ++other_dim) {
			dL_delevated[other_dim] += grad;
		}
		dL_delevated[dim+1] -= dL_ddLdx_i[dim] * scale_cur_lvl[dim] * (COMPUTE_T)(dim+1); 
	}

	//---- dE/dB
	COMPUTE_T dL_dbarycentric[N_POS_DIMS+2]{0};
	#pragma unroll
	for (uint32_t dim=0; dim<=N_POS_DIMS; ++dim) {
		COMPUTE_T dL_dE = dL_delevated[dim] / (COMPUTE_T)(N_POS_DIMS+1); 
		dL_dbarycentric[(int32_t)N_POS_DIMS - rank[dim]] += dL_dE;
		dL_dbarycentric[(int32_t)N_POS_DIMS + 1 - rank[dim]] -= dL_dE;
	}
	dL_dbarycentric[0] += dL_dbarycentric[N_POS_DIMS + 1];
	
	//---- Push gradients into `grad_lattice_values`, `dL_ddLdy`

	// Prepare for `dL_ddLdy`
	COMPUTE_T dL_ddLdy_i[N_FEAT_PER_THREAD]{0}; 

	// Prepare for `grad_lattice_values`
	if (grad_lattice_values) {
		grad_lattice_values += (batch_offset + lattice_feat_offset);
	}
	auto add_lattice_gradient = [&](const uint32_t index, const vector_t<COMPUTE_T, N_FEAT_PER_THREAD>& grad, COMPUTE_T weight) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600 // atomicAdd(__half2) is only supported with compute capability 60 and above
		if (N_FEAT_PER_THREAD > 1 && std::is_same<GRAD_T, __half>::value)
		{
			for (uint32_t f = 0; f < N_FEAT_PER_THREAD; f += 2) {
				__half2 v = {(__half)((COMPUTE_T)grad[f] * weight), (__half)((COMPUTE_T)grad[f+1] * weight)};
				atomicAdd((__half2*)&grad_lattice_values[index + f], v);
			}
		} else 
#endif
		{
			if (std::is_same<GRAD_T, __half>::value) {
				// Should never happen
				//printf("Attempted to use atomicAdd(__half)\n")
			} else {
				// NOTE: Here we perform a reduction on `grad_lattice_values` using `atomicAdd`.
				#pragma unroll
				for (uint32_t f=0; f<N_FEAT_PER_THREAD; ++f) {
					atomicAdd((float*)&grad_lattice_values[index + f], (float)((COMPUTE_T)grad[f] * weight));
				}
			}
		}
	};

	int32_t key[N_POS_DIMS]; 
	for (uint32_t k=0; k <= N_POS_DIMS; ++k) { // remainder-k vertex, for k \in {0,1,...,d}
		// Compute the coordinates of the remainder-k vertex explicitly
		// (all but the last coordinate - it's redundant because they sum to zero)
		#pragma unroll
		for (uint32_t dim=0; dim < N_POS_DIMS; ++dim) {
			key[dim] = rem0[dim] + (int32_t)k; 
			if (rank[dim] > (int32_t)(N_POS_DIMS-k))
				key[dim] -= (int32_t)(N_POS_DIMS+1);
		}

		// Retrieve pointer to the value at this vertex.
		uint32_t index = index_hash<N_POS_DIMS>(key, size_cur_lvl) * n_feat_cur_lvl;

		if (dL_ddLdy) {
			// Vectorized loads
			auto val = *(vector_t<PARAM_T, N_FEAT_PER_THREAD>*)&lattice_values[index];
			#pragma unroll
			for (uint32_t f=0; f<N_FEAT_PER_THREAD; ++f) {
				dL_ddLdy_i[f] += dL_dbarycentric[k] * (COMPUTE_T)val[f]; 
			}
		}

		if (grad_lattice_values) {
			add_lattice_gradient(index, dL_dy_i, dL_dbarycentric[k]); 
		}
	}

	if (dL_ddLdy) {
		dL_ddLdy += (i*meta.n_encoded_dims + out_feat_offset);
		#pragma unroll
		for (uint32_t f=0; f<N_FEAT_PER_THREAD; ++f) {
			dL_ddLdy[f] = dL_ddLdy_i[f]; 
		}
	}
}

template <typename INPUT_T, typename PARAM_T, typename COMPUTE_T,  uint32_t N_POS_DIMS, uint32_t N_FEAT_PER_PSEUDO_LVL>
void permuto_enc_fwd_impl_dispatched(
	// Input
	PermutoEncMeta& meta, 
	at::Tensor& positions, 
	at::Tensor& lattice_values, 
	// Optional
	at::optional<at::Tensor> level_random_shifts_,
	at::optional<at::Tensor> batch_inds_,
	at::optional<at::Tensor> batch_offsets_,
	uint32_t batch_data_size, 
	int32_t max_level, 
	// Output
	at::Tensor& encoded
) {
	const uint32_t batch_size = positions.size(0);
	at::Tensor level_scales_multidim = meta.level_scales_multidim.to(at::TensorOptions().dtype(at::kFloat).device(positions.device()), false, true); 

	const dim3 blocks = { div_round_up(batch_size, N_THREADS), meta.n_pseudo_levels, 1 };

	const at::cuda::OptionalCUDAGuard device_guard(at::device_of(lattice_values));
	cudaStream_t stream = at::cuda::getCurrentCUDAStream();

	kernel_permutohedral<INPUT_T, PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL><<<blocks, N_THREADS, 0, stream>>>(
		batch_size, 
		{meta}, 
		max_level, (int32_t*)nullptr, 
		data_ptr<INPUT_T>(positions), 
		data_ptr<PARAM_T>(lattice_values), 
		level_scales_multidim.data_ptr<float>(), 
		level_random_shifts_.has_value() ? data_ptr<float>(level_random_shifts_.value()) : nullptr, 
		batch_inds_.has_value() ? batch_inds_.value().data_ptr<int64_t>() : nullptr, 
		batch_offsets_.has_value() ? batch_offsets_.value().data_ptr<int64_t>() : nullptr, 
		batch_data_size, 
		data_ptr<PARAM_T>(encoded)
		// need_intermediate ? data_ptr<int32_t>(rank) : nullptr, 
		// need_intermediate ? data_ptr<int32_t>(rem0) : nullptr
		// need_intermediate ? data_ptr<INPUT_T>(elevated) : nullptr
	);
	// cudaDeviceSynchronize(); // For debug
}

template <uint32_t N_POS_DIMS, uint32_t N_FEAT_PER_PSEUDO_LVL>
void permuto_enc_fwd_impl_templated(
	// Input
	PermutoEncMeta& meta, 
	at::Tensor& positions, 
	at::Tensor& lattice_values, 
	// Optional
	at::optional<at::Tensor> level_random_shifts_,
	at::optional<at::Tensor> batch_inds_,
	at::optional<at::Tensor> batch_offsets_,
	uint32_t batch_data_size, 
	int32_t max_level, 
	// Output
	at::Tensor& encoded
) {
	if (positions.scalar_type() == at::kHalf && lattice_values.scalar_type() == at::kHalf) {
		throw std::runtime_error("PermutoEncImpl: Currently, do not support input type combination = <positions,lattice_values> -> (half,half)");
		// permuto_enc_fwd_impl_dispatched<__half, __half, __half, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL>(meta, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, encoded);
	} else if (positions.scalar_type() == at::kFloat && lattice_values.scalar_type() == at::kHalf) {
		permuto_enc_fwd_impl_dispatched<float, __half, float, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL>(meta, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, encoded);
	} else if (positions.scalar_type() == at::kFloat && lattice_values.scalar_type() == at::kFloat) {
		permuto_enc_fwd_impl_dispatched<float, float, float, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL>(meta, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, encoded);
	} else {
		throw std::runtime_error("PermutoEncImpl: Input type combination not supported. Supported types are: <positions,lattice_values> -> (half, half), (float, half), (float, float)");
	}
}

template <uint32_t N_POS_DIMS>
void permuto_enc_fwd_impl(
	// Input
	PermutoEncMeta& meta, 
	at::Tensor& positions, 
	at::Tensor& lattice_values, 
	// Optional
	at::optional<at::Tensor> level_random_shifts_,
	at::optional<at::Tensor> batch_inds_,
	at::optional<at::Tensor> batch_offsets_,
	uint32_t batch_data_size, 
	int32_t max_level, 
	// Output
	at::Tensor& encoded
) {
	switch (meta.n_feat_per_pseudo_lvl) {
		case 2: permuto_enc_fwd_impl_templated<N_POS_DIMS,2>(meta, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, encoded); break;
		case 4: permuto_enc_fwd_impl_templated<N_POS_DIMS,4>(meta, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, encoded); break;
		// case 8: permuto_enc_fwd_impl_templated<N_POS_DIMS,8>(meta, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, encoded); break;
		default: throw std::runtime_error("PermutoEncImpl: `n_feat_per_pseudo_lvl` must be one of [2,4,8]"); break;
	}
}


template <typename INPUT_T, typename PARAM_T, typename COMPUTE_T,  uint32_t N_POS_DIMS, uint32_t N_FEAT_PER_PSEUDO_LVL>
void permuto_enc_bwd_impl_dispatched(
	// Input
	PermutoEncMeta meta, 
	at::Tensor& dL_dy,
	at::Tensor& positions, 
	at::Tensor& lattice_values, 
	// Optional
	at::optional<at::Tensor> level_random_shifts_,
	at::optional<at::Tensor> batch_inds_,
	at::optional<at::Tensor> batch_offsets_,
	uint32_t batch_data_size, 
	int32_t max_level, 
	uint32_t max_pos_dims, 
	bool need_input_grad, 
	bool need_param_grad, 
	// Output
	at::Tensor& dL_dx, 
	at::Tensor& dL_dlattice_val
) {
	const uint32_t batch_size = positions.size(0);
	at::Tensor level_scales_multidim = meta.level_scales_multidim.to(at::TensorOptions().dtype(at::kFloat).device(positions.device()), false, true); 

	static constexpr uint32_t N_FEAT_PER_THREAD = std::min(2u, N_FEAT_PER_PSEUDO_LVL);
	const dim3 blocks = { div_round_up(batch_size * N_FEAT_PER_PSEUDO_LVL / N_FEAT_PER_THREAD, N_THREADS_BACK), meta.n_pseudo_levels, 1 };

	const at::cuda::OptionalCUDAGuard device_guard(at::device_of(lattice_values));
	cudaStream_t stream = at::cuda::getCurrentCUDAStream();

	if (need_input_grad) {
		kernel_permutohedral_backward_input<INPUT_T, PARAM_T, PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL, N_FEAT_PER_THREAD><<<blocks, N_THREADS_BACK, 0, stream>>> (
			batch_size, 
			{meta}, 
			max_level, (int32_t*)nullptr, 
			max_pos_dims, 
			data_ptr<PARAM_T>(dL_dy), 
			data_ptr<INPUT_T>(positions), 
			data_ptr<PARAM_T>(lattice_values), 
			level_scales_multidim.data_ptr<float>(), 
			level_random_shifts_.has_value() ? data_ptr<float>(level_random_shifts_.value()) : nullptr, 
			batch_inds_.has_value() ? batch_inds_.value().data_ptr<int64_t>() : nullptr, 
			batch_offsets_.has_value() ? batch_offsets_.value().data_ptr<int64_t>() : nullptr, 
			batch_data_size, 
			data_ptr<INPUT_T>(dL_dx)
		);
	}

	if (need_param_grad) {
		kernel_permutohedral_backward_lattice<INPUT_T, PARAM_T, PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL, N_FEAT_PER_THREAD><<<blocks, N_THREADS_BACK, 0, stream>>> (
			batch_size, 
			{meta}, 
			max_level, (int32_t*)nullptr, 
			data_ptr<PARAM_T>(dL_dy), 
			data_ptr<INPUT_T>(positions), 
			data_ptr<PARAM_T>(lattice_values), 
			level_scales_multidim.data_ptr<float>(), 
			level_random_shifts_.has_value() ? data_ptr<float>(level_random_shifts_.value()) : nullptr, 
			batch_inds_.has_value() ? batch_inds_.value().data_ptr<int64_t>() : nullptr, 
			batch_offsets_.has_value() ? batch_offsets_.value().data_ptr<int64_t>() : nullptr, 
			batch_data_size, 
			data_ptr<PARAM_T>(dL_dlattice_val)
		); 
	}
	// cudaDeviceSynchronize(); // For debug
}

template <uint32_t N_POS_DIMS, uint32_t N_FEAT_PER_PSEUDO_LVL>
void permuto_enc_bwd_impl_templated(
	// Input
	PermutoEncMeta meta, 
	at::Tensor& dL_dy,
	at::Tensor& positions, 
	at::Tensor& lattice_values, 
	// Optional
	at::optional<at::Tensor> level_random_shifts_,
	at::optional<at::Tensor> batch_inds_,
	at::optional<at::Tensor> batch_offsets_,
	uint32_t batch_data_size, 
	int32_t max_level, 
	uint32_t max_pos_dims, 
	bool need_input_grad, 
	bool need_param_grad, 
	// Output
	at::Tensor& dL_dx, 
	at::Tensor& dL_dlattice_val
) {
	if (positions.scalar_type() == at::kHalf && lattice_values.scalar_type() == at::kHalf) {
		throw std::runtime_error("PermutoEncImpl: Currently, do not support input type combination = <positions,lattice_values> -> (half,half)");
		// permuto_enc_bwd_impl_dispatched<__half, __half, __half, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL>(meta, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, max_pos_dims, need_input_grad, need_param_grad, dL_dx, dL_dlattice_val); 
	} else if (positions.scalar_type() == at::kFloat && lattice_values.scalar_type() == at::kHalf) {
		permuto_enc_bwd_impl_dispatched<float, __half, float, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL>(meta, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, max_pos_dims, need_input_grad, need_param_grad, dL_dx, dL_dlattice_val); 
	} else if (positions.scalar_type() == at::kFloat && lattice_values.scalar_type() == at::kFloat) {
		permuto_enc_bwd_impl_dispatched<float, float, float, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL>(meta, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, max_pos_dims, need_input_grad, need_param_grad, dL_dx, dL_dlattice_val); 
	} else {
		throw std::runtime_error("PermutoEncImpl: Input type combination not supported. Supported types are: <positions,lattice_values> -> (half, half), (float, half), (float, float)");
	}
}

template <uint32_t N_POS_DIMS>
void permuto_enc_bwd_impl(
	// Input
	PermutoEncMeta meta, 
	at::Tensor& dL_dy,
	at::Tensor& positions, 
	at::Tensor& lattice_values, 
	// Optional
	at::optional<at::Tensor> level_random_shifts_,
	at::optional<at::Tensor> batch_inds_,
	at::optional<at::Tensor> batch_offsets_,
	uint32_t batch_data_size, 
	int32_t max_level, 
	uint32_t max_pos_dims, 
	bool need_input_grad, 
	bool need_param_grad, 
	// Output
	at::Tensor& dL_dx, 
	at::Tensor& dL_dlattice_val
) {
	switch (meta.n_feat_per_pseudo_lvl) {
		case 2: permuto_enc_bwd_impl_templated<N_POS_DIMS,2>(meta, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, max_pos_dims, need_input_grad, need_param_grad, dL_dx, dL_dlattice_val); break;
		case 4: permuto_enc_bwd_impl_templated<N_POS_DIMS,4>(meta, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, max_pos_dims, need_input_grad, need_param_grad, dL_dx, dL_dlattice_val); break;
		// case 8: permuto_enc_bwd_impl_templated<N_POS_DIMS,8>(meta, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, max_pos_dims, need_input_grad, need_param_grad, dL_dx, dL_dlattice_val); break;
		default: throw std::runtime_error("PermutoEncImpl: `n_feat_per_pseudo_lvl` must be one of [2,4,8]"); break;
	}
}

template <typename INPUT_T, typename PARAM_T, typename COMPUTE_T,  uint32_t N_POS_DIMS, uint32_t N_FEAT_PER_PSEUDO_LVL>
void permuto_enc_bwd_bwd_input_impl_dispatched(
	// Input
	PermutoEncMeta meta, 
	at::Tensor& dL_ddLdx,
	at::Tensor& dL_dy,
	at::Tensor& positions, 
	at::Tensor& lattice_values, 
	// Optional
	at::optional<at::Tensor> level_random_shifts_,
	at::optional<at::Tensor> batch_inds_,
	at::optional<at::Tensor> batch_offsets_,
	uint32_t batch_data_size, 
	int32_t max_level, 
	bool need_dL_ddLdy, 
	bool need_dL_dparams, 
	// Output
	at::Tensor& dL_ddLdy, 
	at::Tensor& dL_dlattice_val
) {
	const uint32_t batch_size = positions.size(0);
	at::Tensor level_scales_multidim = meta.level_scales_multidim.to(at::TensorOptions().dtype(at::kFloat).device(positions.device()), false, true); 

	static constexpr uint32_t N_FEAT_PER_THREAD = std::min(2u, N_FEAT_PER_PSEUDO_LVL);
	const dim3 blocks = { div_round_up(batch_size * N_FEAT_PER_PSEUDO_LVL / N_FEAT_PER_THREAD, N_THREADS_BACK), meta.n_pseudo_levels, 1 };

	const at::cuda::OptionalCUDAGuard device_guard(at::device_of(lattice_values));
	cudaStream_t stream = at::cuda::getCurrentCUDAStream();

	kernel_permutohedral_backward_backward_input<INPUT_T, PARAM_T, PARAM_T, COMPUTE_T, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL, N_FEAT_PER_PSEUDO_LVL><<<blocks, N_THREADS_BACK, 0, stream>>> (
		batch_size, 
		{meta}, 
		max_level, (int32_t*)nullptr, 
		data_ptr<INPUT_T>(dL_ddLdx), 
		data_ptr<PARAM_T>(dL_dy), 
		data_ptr<INPUT_T>(positions), 
		data_ptr<PARAM_T>(lattice_values), 
		level_scales_multidim.data_ptr<float>(), 
		level_random_shifts_.has_value() ? data_ptr<float>(level_random_shifts_.value()) : nullptr, 
		batch_inds_.has_value() ? batch_inds_.value().data_ptr<int64_t>() : nullptr, 
		batch_offsets_.has_value() ? batch_offsets_.value().data_ptr<int64_t>() : nullptr, 
		batch_data_size, 
		need_dL_dparams ? data_ptr<PARAM_T>(dL_dlattice_val) : nullptr, 
		need_dL_ddLdy ? data_ptr<PARAM_T>(dL_ddLdy) : nullptr
	);
	// cudaDeviceSynchronize(); // For debug
}

template <uint32_t N_POS_DIMS, uint32_t N_FEAT_PER_PSEUDO_LVL>
void permuto_enc_bwd_bwd_input_impl_templated(
	// Input
	PermutoEncMeta meta, 
	at::Tensor& dL_ddLdx,
	at::Tensor& dL_dy,
	at::Tensor& positions, 
	at::Tensor& lattice_values, 
	// Optional
	at::optional<at::Tensor> level_random_shifts_,
	at::optional<at::Tensor> batch_inds_,
	at::optional<at::Tensor> batch_offsets_,
	uint32_t batch_data_size, 
	int32_t max_level, 
	bool need_dL_ddLdy, 
	bool need_dL_dparams, 
	// Output
	at::Tensor& dL_ddLdy, 
	at::Tensor& dL_dlattice_val
) {
	if (positions.scalar_type() == at::kHalf && lattice_values.scalar_type() == at::kHalf) {
		throw std::runtime_error("PermutoEncImpl: Currently, do not support input type combination = <positions,lattice_values> -> (half,half)");
		// permuto_enc_bwd_bwd_input_impl_dispatched<__half, __half, __half, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL>(meta, dL_ddLdx, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, need_dL_ddLdy, need_dL_dparams, dL_ddLdy, dL_dlattice_val); 
	} else if (positions.scalar_type() == at::kFloat && lattice_values.scalar_type() == at::kHalf) {
		permuto_enc_bwd_bwd_input_impl_dispatched<float, __half, float, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL>(meta, dL_ddLdx, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, need_dL_ddLdy, need_dL_dparams, dL_ddLdy, dL_dlattice_val); 
	} else if (positions.scalar_type() == at::kFloat && lattice_values.scalar_type() == at::kFloat) {
		permuto_enc_bwd_bwd_input_impl_dispatched<float, float, float, N_POS_DIMS, N_FEAT_PER_PSEUDO_LVL>(meta, dL_ddLdx, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, need_dL_ddLdy, need_dL_dparams, dL_ddLdy, dL_dlattice_val); 
	} else {
		throw std::runtime_error("PermutoEncImpl: Input type combination not supported. Supported types are: <positions,lattice_values> -> (half, half), (float, half), (float, float)");
	}
}

template <uint32_t N_POS_DIMS>
void permuto_enc_bwd_bwd_input_impl(
	// Input
	PermutoEncMeta meta, 
	at::Tensor& dL_ddLdx,
	at::Tensor& dL_dy,
	at::Tensor& positions, 
	at::Tensor& lattice_values, 
	// Optional
	at::optional<at::Tensor> level_random_shifts_,
	at::optional<at::Tensor> batch_inds_,
	at::optional<at::Tensor> batch_offsets_,
	uint32_t batch_data_size, 
	int32_t max_level, 
	bool need_dL_ddLdy, 
	bool need_dL_dparams, 
	// Output
	at::Tensor& dL_ddLdy, 
	at::Tensor& dL_dlattice_val
) {
	switch (meta.n_feat_per_pseudo_lvl) {
		case 2: permuto_enc_bwd_bwd_input_impl_templated<N_POS_DIMS,2>(meta, dL_ddLdx, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, need_dL_ddLdy, need_dL_dparams, dL_ddLdy, dL_dlattice_val); break;
		case 4: permuto_enc_bwd_bwd_input_impl_templated<N_POS_DIMS,4>(meta, dL_ddLdx, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, need_dL_ddLdy, need_dL_dparams, dL_ddLdy, dL_dlattice_val); break;
		// case 8: permuto_enc_bwd_bwd_input_impl_templated<N_POS_DIMS,8>(meta, dL_ddLdx, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, need_dL_ddLdy, need_dL_dparams, dL_ddLdy, dL_dlattice_val); break;
		default: throw std::runtime_error("PermutoEncImpl: `n_feat_per_pseudo_lvl` must be one of [2,4,8]"); break;
	}
}



// Explicit Template Instantiation (to allow for compiling seperate files in parallel to speed up)
extern template void permuto_enc_fwd_impl< 2>(PermutoEncMeta&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,at::Tensor&); 
extern template void permuto_enc_fwd_impl< 3>(PermutoEncMeta&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,at::Tensor&); 
extern template void permuto_enc_fwd_impl< 4>(PermutoEncMeta&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,at::Tensor&); 
extern template void permuto_enc_fwd_impl< 5>(PermutoEncMeta&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,at::Tensor&); 
extern template void permuto_enc_fwd_impl< 6>(PermutoEncMeta&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,at::Tensor&); 
extern template void permuto_enc_fwd_impl< 7>(PermutoEncMeta&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,at::Tensor&); 
extern template void permuto_enc_fwd_impl< 8>(PermutoEncMeta&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,at::Tensor&); 
extern template void permuto_enc_fwd_impl< 9>(PermutoEncMeta&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,at::Tensor&); 
extern template void permuto_enc_fwd_impl<10>(PermutoEncMeta&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,at::Tensor&); 
extern template void permuto_enc_fwd_impl<11>(PermutoEncMeta&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,at::Tensor&); 
extern template void permuto_enc_fwd_impl<12>(PermutoEncMeta&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,at::Tensor&); 
extern template void permuto_enc_fwd_impl<13>(PermutoEncMeta&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,at::Tensor&); 
extern template void permuto_enc_fwd_impl<14>(PermutoEncMeta&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,at::Tensor&); 
extern template void permuto_enc_fwd_impl<15>(PermutoEncMeta&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,at::Tensor&); 
extern template void permuto_enc_fwd_impl<16>(PermutoEncMeta&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,at::Tensor&); 
extern template void permuto_enc_fwd_impl<17>(PermutoEncMeta&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,at::Tensor&); 
extern template void permuto_enc_fwd_impl<18>(PermutoEncMeta&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,at::Tensor&); 
extern template void permuto_enc_fwd_impl<19>(PermutoEncMeta&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,at::Tensor&); 
extern template void permuto_enc_fwd_impl<20>(PermutoEncMeta&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,at::Tensor&); 
extern template void permuto_enc_fwd_impl<24>(PermutoEncMeta&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,at::Tensor&); 
extern template void permuto_enc_fwd_impl<28>(PermutoEncMeta&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,at::Tensor&); 
extern template void permuto_enc_fwd_impl<32>(PermutoEncMeta&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,at::Tensor&); 
extern template void permuto_enc_fwd_impl<40>(PermutoEncMeta&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,at::Tensor&); 
extern template void permuto_enc_fwd_impl<48>(PermutoEncMeta&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,at::Tensor&); 
extern template void permuto_enc_fwd_impl<56>(PermutoEncMeta&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,at::Tensor&); 
extern template void permuto_enc_fwd_impl<64>(PermutoEncMeta&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,at::Tensor&); 

extern template void permuto_enc_bwd_impl< 2>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,uint32_t,bool,bool,at::Tensor&,at::Tensor&); 
extern template void permuto_enc_bwd_impl< 3>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,uint32_t,bool,bool,at::Tensor&,at::Tensor&); 
extern template void permuto_enc_bwd_impl< 4>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,uint32_t,bool,bool,at::Tensor&,at::Tensor&); 
extern template void permuto_enc_bwd_impl< 5>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,uint32_t,bool,bool,at::Tensor&,at::Tensor&); 
extern template void permuto_enc_bwd_impl< 6>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,uint32_t,bool,bool,at::Tensor&,at::Tensor&); 
extern template void permuto_enc_bwd_impl< 7>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,uint32_t,bool,bool,at::Tensor&,at::Tensor&); 
extern template void permuto_enc_bwd_impl< 8>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,uint32_t,bool,bool,at::Tensor&,at::Tensor&); 
extern template void permuto_enc_bwd_impl< 9>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,uint32_t,bool,bool,at::Tensor&,at::Tensor&); 
extern template void permuto_enc_bwd_impl<10>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,uint32_t,bool,bool,at::Tensor&,at::Tensor&); 
extern template void permuto_enc_bwd_impl<11>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,uint32_t,bool,bool,at::Tensor&,at::Tensor&); 
extern template void permuto_enc_bwd_impl<12>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,uint32_t,bool,bool,at::Tensor&,at::Tensor&); 
extern template void permuto_enc_bwd_impl<13>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,uint32_t,bool,bool,at::Tensor&,at::Tensor&); 
extern template void permuto_enc_bwd_impl<14>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,uint32_t,bool,bool,at::Tensor&,at::Tensor&); 
extern template void permuto_enc_bwd_impl<15>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,uint32_t,bool,bool,at::Tensor&,at::Tensor&); 
extern template void permuto_enc_bwd_impl<16>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,uint32_t,bool,bool,at::Tensor&,at::Tensor&); 
extern template void permuto_enc_bwd_impl<17>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,uint32_t,bool,bool,at::Tensor&,at::Tensor&); 
extern template void permuto_enc_bwd_impl<18>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,uint32_t,bool,bool,at::Tensor&,at::Tensor&); 
extern template void permuto_enc_bwd_impl<19>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,uint32_t,bool,bool,at::Tensor&,at::Tensor&); 
extern template void permuto_enc_bwd_impl<20>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,uint32_t,bool,bool,at::Tensor&,at::Tensor&); 
extern template void permuto_enc_bwd_impl<24>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,uint32_t,bool,bool,at::Tensor&,at::Tensor&); 
extern template void permuto_enc_bwd_impl<28>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,uint32_t,bool,bool,at::Tensor&,at::Tensor&); 
extern template void permuto_enc_bwd_impl<32>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,uint32_t,bool,bool,at::Tensor&,at::Tensor&); 
extern template void permuto_enc_bwd_impl<36>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,uint32_t,bool,bool,at::Tensor&,at::Tensor&); 
extern template void permuto_enc_bwd_impl<40>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,uint32_t,bool,bool,at::Tensor&,at::Tensor&); 
extern template void permuto_enc_bwd_impl<48>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,uint32_t,bool,bool,at::Tensor&,at::Tensor&); 
extern template void permuto_enc_bwd_impl<56>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,uint32_t,bool,bool,at::Tensor&,at::Tensor&); 
extern template void permuto_enc_bwd_impl<64>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,uint32_t,bool,bool,at::Tensor&,at::Tensor&); 

extern template void permuto_enc_bwd_bwd_input_impl< 2>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,bool,bool,at::Tensor&,at::Tensor&); 
extern template void permuto_enc_bwd_bwd_input_impl< 3>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,bool,bool,at::Tensor&,at::Tensor&); 
extern template void permuto_enc_bwd_bwd_input_impl< 4>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,bool,bool,at::Tensor&,at::Tensor&); 
extern template void permuto_enc_bwd_bwd_input_impl< 5>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,bool,bool,at::Tensor&,at::Tensor&); 
extern template void permuto_enc_bwd_bwd_input_impl< 6>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,bool,bool,at::Tensor&,at::Tensor&); 
extern template void permuto_enc_bwd_bwd_input_impl< 7>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,bool,bool,at::Tensor&,at::Tensor&); 
extern template void permuto_enc_bwd_bwd_input_impl< 8>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,bool,bool,at::Tensor&,at::Tensor&); 
extern template void permuto_enc_bwd_bwd_input_impl< 9>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,bool,bool,at::Tensor&,at::Tensor&); 
extern template void permuto_enc_bwd_bwd_input_impl<10>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,bool,bool,at::Tensor&,at::Tensor&); 
extern template void permuto_enc_bwd_bwd_input_impl<11>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,bool,bool,at::Tensor&,at::Tensor&); 
extern template void permuto_enc_bwd_bwd_input_impl<12>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,bool,bool,at::Tensor&,at::Tensor&); 
extern template void permuto_enc_bwd_bwd_input_impl<13>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,bool,bool,at::Tensor&,at::Tensor&); 
extern template void permuto_enc_bwd_bwd_input_impl<14>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,bool,bool,at::Tensor&,at::Tensor&); 
extern template void permuto_enc_bwd_bwd_input_impl<15>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,bool,bool,at::Tensor&,at::Tensor&); 
extern template void permuto_enc_bwd_bwd_input_impl<16>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,bool,bool,at::Tensor&,at::Tensor&); 
extern template void permuto_enc_bwd_bwd_input_impl<17>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,bool,bool,at::Tensor&,at::Tensor&); 
extern template void permuto_enc_bwd_bwd_input_impl<18>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,bool,bool,at::Tensor&,at::Tensor&); 
extern template void permuto_enc_bwd_bwd_input_impl<19>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,bool,bool,at::Tensor&,at::Tensor&); 
extern template void permuto_enc_bwd_bwd_input_impl<20>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,bool,bool,at::Tensor&,at::Tensor&); 
extern template void permuto_enc_bwd_bwd_input_impl<24>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,bool,bool,at::Tensor&,at::Tensor&); 
extern template void permuto_enc_bwd_bwd_input_impl<28>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,bool,bool,at::Tensor&,at::Tensor&); 
extern template void permuto_enc_bwd_bwd_input_impl<32>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,bool,bool,at::Tensor&,at::Tensor&); 
extern template void permuto_enc_bwd_bwd_input_impl<36>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,bool,bool,at::Tensor&,at::Tensor&); 
extern template void permuto_enc_bwd_bwd_input_impl<40>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,bool,bool,at::Tensor&,at::Tensor&); 
extern template void permuto_enc_bwd_bwd_input_impl<48>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,bool,bool,at::Tensor&,at::Tensor&); 
extern template void permuto_enc_bwd_bwd_input_impl<56>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,bool,bool,at::Tensor&,at::Tensor&); 
extern template void permuto_enc_bwd_bwd_input_impl<64>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,bool,bool,at::Tensor&,at::Tensor&); 

}

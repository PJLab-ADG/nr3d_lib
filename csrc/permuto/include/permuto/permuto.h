/** @file   permuto.h
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

namespace permuto {

template <typename T>
inline
typename std::enable_if<!std::is_same<T, __half>::value, T*>::type
data_ptr(at::Tensor& t) {
	return t.data_ptr<T>();
}

template <typename T>
inline
typename std::enable_if<std::is_same<T, __half>::value, T*>::type
data_ptr(at::Tensor& t) {
	return (T*)(t.data_ptr<at::Half>());
}

template <typename T>
inline
typename std::enable_if<!std::is_same<T, __half>::value, at::ScalarType>::type
scalar_type() {
	return at::CppTypeToScalarType<T>::value;
}

template <typename T>
inline
typename std::enable_if<std::is_same<T, __half>::value, at::ScalarType>::type
scalar_type() {
	return at::kHalf;
}

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

extern const std::vector<uint32_t> supported_n_input_dims; 

struct PermutoEncMeta {
	std::vector<double> level_scales0;      // [n_levels]   Original raw positions scales per `level`
	std::vector<uint32_t> level_n_feats;    // [n_levels]   Feature width for each actual `level`
	std::vector<uint32_t> level_n_params;   // [n_levels]   Grid parameter sizes for each actual `level` (considering feature width)
	std::vector<uint32_t> level_offsets;    // [n_levels+1] Parameter offsets for each actual `level` in a single entry (considering feature width)
	std::vector<uint32_t> level_sizes;      // [n_levels]   Grid sizes for each actual `level` (not considering feature width)
	std::vector<uint32_t> map_levels;       // [n_pseudo_levels]    Actual `level` corresponding to each pseudo `level`
	std::vector<uint32_t> map_cnt;          // [n_pseudo_levels]    Index of the current pseudo `level` in all the pseudo `level`s corresponding to current actual `level`

	uint32_t n_levels = 0;                  // Number of actual levels (allow non-equal feature width)
	uint32_t n_pseudo_levels = 0;           // Number of pseudo levels (all equal feature width = `n_feat_per_pseudo_lvl`)
	uint32_t n_feat_per_pseudo_lvl = 2;     // Feature width of each pseudo level = the greatest common divisor of all acutal levels' widths 
	uint32_t n_dims_to_encode = 3;          // Number of dims to encode (in_features)
	uint32_t n_encoded_dims = 0;            // Number of encoded dims (out_features)
	uint32_t n_params = 0;                  // Number of total params

	at::Tensor level_scales_multidim;       // [n_levels], each [n_dims_to_encode]:     Grid side lengths for each actual `level`

	PermutoEncMeta(
		const int32_t n_input_dim,
		const int32_t hashmap_size, 
		const std::vector<double>& res_list,
		const std::vector<int32_t>& n_feats_list
	) {
		create_meta(n_input_dim, hashmap_size, res_list, n_feats_list);
	}

	void create_meta(
		const uint32_t n_input_dim,
		const uint32_t hashmap_size, 
		const std::vector<double>& res_list,
		const std::vector<int32_t>& n_feats_list
	);
};



//---- Forward
// Returns: encoded, (rank, rem0)
at::Tensor permuto_enc_fwd(
	// Input
	PermutoEncMeta meta, 
	at::Tensor positions, 
	at::Tensor lattice_values, 
	// Optional
	at::optional<at::Tensor> level_random_shifts_, 
	at::optional<at::Tensor> batch_inds_,
	at::optional<at::Tensor> batch_offsets_,
	at::optional<uint32_t> batch_data_size_, 
	at::optional<int32_t> max_level_
);


//---- Backward
// Returns: dL_dx, dL_dlattice
std::tuple<at::Tensor, at::Tensor> permuto_enc_bwd(
	// Input
	PermutoEncMeta meta, 
	at::Tensor dL_dy,
	at::Tensor positions, 
	at::Tensor lattice_values, 
	// Optional
	at::optional<at::Tensor> level_random_shifts_, 
	at::optional<at::Tensor> batch_inds_,
	at::optional<at::Tensor> batch_offsets_,
	at::optional<uint32_t> batch_data_size, 
	at::optional<int32_t> max_level_, 
	at::optional<uint32_t> max_pos_dims_, 
	at::optional<bool> need_input_grad_, 
	at::optional<bool> need_param_grad_
);

std::tuple<at::Tensor, at::Tensor> permuto_enc_bwd_bwd_input(
	// Input
	PermutoEncMeta meta, 
	at::Tensor dL_ddLdx, 
	at::Tensor dL_dy,
	at::Tensor positions, 
	at::Tensor lattice_values, 
	// Optional
	at::optional<at::Tensor> level_random_shifts_, 
	at::optional<at::Tensor> batch_inds_,
	at::optional<at::Tensor> batch_offsets_,
	at::optional<uint32_t> batch_data_size_,
	at::optional<int32_t> max_level_, 
	at::optional<bool> need_dLdinput_ddLdoutput_,
	at::optional<bool> need_dLdinput_dparams_
);

}

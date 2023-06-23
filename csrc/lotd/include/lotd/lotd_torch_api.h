/** @file   lotd_torch_api.cu
 *  @author Jianfei Guo, Shanghai AI Lab
 *  @brief  LoTD Pytorch APIs (declaration).
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

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>

#include "lotd_types.h"

namespace lotd {
namespace torch {

using LoDType = lotd::LoDType;

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

struct LoDMeta {
	std::vector<uint32_t> level_res;			// [n_levels]	Grid side lengths for each actual `level` (Only valid when cubic)
	std::vector<std::vector<uint32_t>> level_res_multi_dim; // [n_levels], each [n_dims_to_encode]:		Grid side lengths for each actual `level`
	std::vector<uint32_t> level_n_feats; 		// [n_levels]	Feature width for each actual `level`
	std::vector<uint32_t> level_types; 			// [n_levels]	lod types for each actual `level`
	std::vector<std::string> level_types_str;
	std::vector<uint32_t> level_n_params; 		// [n_levels]	Grid parameter sizes for each actual `level` (considering feature width)
	std::vector<uint32_t> level_offsets; 		// [n_levels+1]	Parameter offsets for each actual `level` in a single entry (considering feature width)
	std::vector<uint32_t> level_sizes; 			// [n_levels]	Grid sizes for each actual `level` (not considering feature width)
	std::vector<uint32_t> map_levels; 			// [n_pseudo_levels]	Actual `level` corresponding to each pseudo `level`
	std::vector<uint32_t> map_cnt; 				// [n_pseudo_levels]	Index of the current pseudo `level` in all the pseudo `level`s corresponding to current actual `level`
	
	uint32_t n_levels = 0;
	uint32_t n_pseudo_levels = 0; 
	uint32_t n_feat_per_pseudo_lvl = 2;
	uint32_t n_dims_to_encode = 3;
	uint32_t n_encoded_dims = 0;
	uint32_t n_params = 0;
	InterpolationType interpolation_type = InterpolationType::Linear;

	LoDMeta(
		const int32_t n_input_dim,
		const std::vector<std::vector<int32_t>>& lod_res_multidim,
		const std::vector<int32_t>& lod_n_feats,
		const std::vector<std::string>& lod_str_types,
		const at::optional<uint32_t> hashmap_size_, 
		const at::optional<bool> use_smooth_step_
	) {
		create_meta(n_input_dim, lod_res_multidim, lod_n_feats, lod_str_types, hashmap_size_, use_smooth_step_);
	}

	LoDMeta(
		const int32_t n_input_dim,
		const std::vector<int32_t>& lod_res,
		const std::vector<int32_t>& lod_n_feats,
		const std::vector<std::string>& lod_str_types,
		const at::optional<uint32_t> hashmap_size_, 
		const at::optional<bool> use_smooth_step_
	) {
		std::vector<std::vector<int32_t>> lod_res_multidim(lod_res.size(), std::vector<int32_t> (n_input_dim, 0) );
		for (uint32_t l=0; l < lod_res.size(); ++l) {
			for (uint32_t dim=0; dim < (uint32_t)n_input_dim; ++dim) {
				lod_res_multidim[l][dim] = lod_res[l];
			}
		}
		create_meta(n_input_dim, lod_res_multidim, lod_n_feats, lod_str_types, hashmap_size_, use_smooth_step_);
	}

	void create_meta(
		const int32_t n_input_dim,
		const std::vector<std::vector<int32_t>>& lod_res,
		const std::vector<int32_t>& lod_n_feats,
		const std::vector<std::string>& lod_str_types,
		const at::optional<uint32_t> hashmap_size_, 
		const at::optional<bool> use_smooth_step_
	);
};

std::tuple<at::Tensor, at::Tensor> lod_fwd(
	LoDMeta lod_meta,
	at::Tensor input,
	at::Tensor params,
	// Optional
	at::optional<at::Tensor> batch_inds_,
	at::optional<at::Tensor> batch_offsets_,
	at::optional<uint32_t> batch_data_size_, 
	at::optional<int32_t> max_level_, 
	at::optional<bool> need_input_grad_
);

std::tuple<at::Tensor, at::Tensor> lod_bwd(
	LoDMeta lod_meta,
	at::Tensor dL_dy,
	at::Tensor input,
	at::Tensor params,
	at::optional<at::Tensor> dy_dx_,
	at::optional<at::Tensor> batch_inds_,
	at::optional<at::Tensor> batch_offsets_,
	at::optional<uint32_t> batch_data_size_, 
	at::optional<int32_t> max_level_, 
	at::optional<bool> need_input_grad_, 
	at::optional<bool> need_param_grad_
);

std::tuple<at::Tensor, at::Tensor, at::Tensor> lod_bwd_bwd_input(
	LoDMeta lod_meta,
	at::Tensor dL_ddLdx,
	at::Tensor dL_dy,
	at::Tensor input,
	at::Tensor params,
	// Optional
	at::optional<at::Tensor> dy_dx_, // Needed when `need_dLdy_grad`
	at::optional<at::Tensor> batch_inds_,
	at::optional<at::Tensor> batch_offsets_,
	at::optional<uint32_t> batch_data_size_,
	at::optional<int32_t> max_level_, 
	at::optional<bool> need_dLdinput_ddLdoutput_,
	at::optional<bool> need_dLdinput_dparams_,
	at::optional<bool> need_dLdinput_dinput_
);

at::Tensor lod_get_grid_index(
	LoDMeta lod_meta,
	at::Tensor input,
	// Optional
	at::optional<at::Tensor> batch_inds_,
	at::optional<at::Tensor> batch_offsets_,
	at::optional<uint32_t> batch_data_size_,
	at::optional<int32_t> max_level_
);


} // namespace: lotd::torch
} // namespace: lotd


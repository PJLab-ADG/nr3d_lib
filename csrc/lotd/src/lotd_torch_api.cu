/** @file   lotd_torch_api.cu
 *  @author Jianfei Guo, Shanghai AI Lab
 *  @brief  LoTD Pytorch APIs (definition).
 */

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

#include <lotd/lotd.h>
#include <lotd/lotd_torch_api.h>
#include <lotd/lotd_encoding.h>

namespace lotd {
namespace torch {

void LoDMeta::create_meta(
	const int32_t n_input_dim,
	const std::vector<std::vector<int32_t>>& lod_res_multidim, 
	const std::vector<int32_t>& lod_n_feats, 
	const std::vector<std::string>& lod_str_types, 
	const at::optional<uint32_t> hashmap_size_, 
	const at::optional<bool> use_smooth_step_
) {
	level_types_str = lod_str_types;
	interpolation_type = use_smooth_step_.value_or(false) ? InterpolationType::Smoothstep : InterpolationType::Linear;

	if (! (lod_res_multidim.size() == lod_n_feats.size() && lod_res_multidim.size() == lod_str_types.size()) ){
		throw std::runtime_error("LoTDEncoding: Expect los_res, lod_n_feats, lod_str_types to have the same length");
	}

	if (! (n_input_dim == 2 || n_input_dim == 3 || n_input_dim == 4) ) {
		throw std::runtime_error("LoTDEncoding: `n_input_dim` must be 2/3/4.");
	}

	n_dims_to_encode = (uint32_t)n_input_dim;
	n_levels = lod_res_multidim.size();

	if (n_levels > MAX_N_LEVELS) {
		throw std::runtime_error(std::string{"LoTDEncoding:` num_level`="} + std::to_string(n_levels) + std::string{" exceeds maximum level="} + std::to_string(MAX_N_LEVELS));
	}

	level_res.resize(n_levels);
	level_res_multi_dim.resize(n_levels);
	level_n_feats.resize(n_levels);
	level_types.resize(n_levels);
	level_n_params.resize(n_levels);
	level_sizes.resize(n_levels);
	level_offsets.resize(n_levels+1);
	
	if (is_all_divisible(lod_n_feats, 8)) {
		n_feat_per_pseudo_lvl = 8;
	} else if (is_all_divisible(lod_n_feats, 4) ) {
		n_feat_per_pseudo_lvl = 4;
	} else if (is_all_divisible(lod_n_feats, 2)) {
		n_feat_per_pseudo_lvl = 2;
	} else {
		throw std::runtime_error("LoTDEncoding: the greatest common divisor of `lod_n_feats` must be at least 2");
	}

	const uint32_t hashmap_size = hashmap_size_.value_or(0u);

	n_pseudo_levels = 0;
	n_encoded_dims = 0;
	uint32_t max_params = std::numeric_limits<uint32_t>::max()/2;
	uint32_t accumulated_num_params = 0;
	float accumulated_num_params_float = 0.0f;
	for (uint32_t lvl = 0; lvl < n_levels; ++lvl ) {
		const uint32_t n_feat = lod_n_feats[lvl];
		const LoDType lod_type = string_to_lod_type(lod_str_types[lvl]);
		
		level_n_feats[lvl] = (uint32_t)n_feat;
		level_types[lvl] = (uint32_t)lod_type;

		uint32_t n = n_feat / n_feat_per_pseudo_lvl;
		n_pseudo_levels += n;
		n_encoded_dims += n_feat;

		std::vector<uint32_t> resolution(n_dims_to_encode);
		bool equal_res = true; 
		for (uint32_t dim=0; dim < n_dims_to_encode; ++dim) {
		 	resolution[dim] = lod_res_multidim[lvl][dim];
			if (resolution[dim] <= 2) {
				throw std::runtime_error("LoTDEncoding: only support grid resolutions >= 3");
			}
			if (resolution[dim] != resolution[0]) {
				equal_res = false;
			}
		}
		level_res_multi_dim[lvl] = resolution;
		if (equal_res) {
			level_res[lvl] = resolution[0];
		} else {
			level_res[lvl] = 0u;
		}

		// Decide grid offset and size
		uint32_t size = 0;
		float size_float = 0;
		switch (lod_type)
		{
			case lotd::LoDType::Dense:
				size = 1;
				size_float = 1.0f;
				for (uint32_t dim = 0; dim < n_dims_to_encode; ++dim) {
					size *= resolution[dim];
					size_float *= resolution[dim];
				}
				break;
			
			case lotd::LoDType::NPlaneMul:
			case lotd::LoDType::NPlaneSum:
				if (n_dims_to_encode < 2) {
					throw std::runtime_error("LoTDEncoding: NPlane mode only support n_dims_to_encode() >= 2");
				}
				size = 0;
				size_float = 0.f;
				for (uint32_t line_dim = 0; line_dim < n_dims_to_encode; ++line_dim) {
					uint32_t plane_size = 1;
					float plane_size_float = 1.0f;
					for (uint32_t dim = 0; dim < n_dims_to_encode-1; ++dim) {
						uint32_t real_dim = dim >= line_dim ? (dim+1) : dim;
						plane_size *= resolution[real_dim];
						plane_size_float *= resolution[real_dim];
					}
					size += plane_size;
					size_float += plane_size_float;
				}
				break;
			
			case lotd::LoDType::VectorMatrix:
				if (n_dims_to_encode != 3) {
					throw std::runtime_error("LoTDEncoding: VectorMatrix mode only support 3D encoding.");
				}
				size = 0;
				size_float = 0.f;
				for (uint32_t line_dim = 0; line_dim < n_dims_to_encode; ++line_dim) {
					uint32_t plane_size = 1;
					float plane_size_float = 1.0f;
					for (uint32_t dim = 0; dim < n_dims_to_encode-1; ++dim) {
						uint32_t real_dim = dim >= line_dim ? (dim+1) : dim;
						plane_size *= resolution[real_dim];
						plane_size_float *= resolution[real_dim];
					}
					size += (plane_size + resolution[line_dim]);
					size_float += (plane_size + resolution[line_dim]);
				}
				break;
			
			case lotd::LoDType::VecZMatXoY:
				if (n_dims_to_encode != 3) {
					throw std::runtime_error("LoTDEncoding: VecZMatXoY mode only support 3D encoding.");
				}
				size = resolution[0] * resolution[1] + resolution[2];
				size_float = (float)resolution[0] * (float)resolution[1] + (float)resolution[2];
				break;

			case lotd::LoDType::CPfast:
			case lotd::LoDType::CP:
				size = 0;
				size_float = 0.f;
				for (uint32_t dim=0; dim < n_dims_to_encode; ++dim) {
					size += resolution[dim];
					size_float += resolution[dim];
				}
				break;
			
			case lotd::LoDType::Hash:
				if (!hashmap_size) {
					throw std::runtime_error("LoTDEncoding: Hash mode need `hashmap_size`");
				}
				size = hashmap_size;
				size_float = hashmap_size;
				break;
			default:
				// Should never happen
				break;
		}

		accumulated_num_params_float += size_float * n_feat;
		if (accumulated_num_params_float > (float)max_params) {
			throw std::runtime_error("LoTDEncoding: param size too large.");
		}

		uint32_t params_in_level = size * n_feat;

		level_sizes[lvl] = (uint32_t)size;
		level_n_params[lvl] = (uint32_t)params_in_level;
		
		level_offsets[lvl] = (uint32_t)accumulated_num_params;
		accumulated_num_params += params_in_level;
	}
	level_offsets[n_levels] = (uint32_t)accumulated_num_params;
	n_params = accumulated_num_params;

	// map_levels, map_cnt
	map_levels.resize(n_pseudo_levels);
	map_cnt.resize(n_pseudo_levels);
	uint32_t acc_n_pseudo_levels = 0;
	for (uint32_t lvl = 0; lvl < n_levels; lvl++) {
		uint32_t n = lod_n_feats[lvl] / n_feat_per_pseudo_lvl;
		for (uint32_t j=0; j<n; ++j) {
			map_levels[acc_n_pseudo_levels + j] = (uint32_t)lvl;
			map_cnt[acc_n_pseudo_levels + j] = (uint32_t)j;
		}
		acc_n_pseudo_levels += n;
	}

	// NOTE: the maximum number of threads of a block
	if (n_encoded_dims > 1024) {
		throw std::runtime_error("LoTDEncoding: total number of features too large. Shoule be <= 1024.");
	}
}

std::tuple<at::Tensor, at::Tensor> lod_fwd_common(
	LoDMeta lod_meta,
	at::Tensor input,
	at::Tensor params,
	// Optional
	at::optional<at::Tensor> batch_inds_,
	at::optional<at::Tensor> batch_offsets_,
	at::optional<uint32_t> batch_data_size_, 
	at::optional<int32_t> max_level_, 
	at::optional<bool> need_input_grad_
) {
	at::TensorArg input_arg(input, "x", 1); 
	at::TensorArg params_arg(params, "grid", 2);

	at::checkDim(__func__, input_arg, 2);
	at::checkDim(__func__, params_arg, 1);
	at::checkAllSameGPU(__func__, {input_arg, params_arg});
	at::checkAllContiguous(__func__, {input_arg, params_arg});
	at::checkScalarTypes(__func__, input_arg, {at::kHalf, at::kFloat});
	at::checkScalarTypes(__func__, params_arg, {at::kHalf, at::kFloat});

	const uint32_t batch_size = input.size(0);
	at::checkSize(__func__, input_arg, 1, lod_meta.n_dims_to_encode);
	const uint32_t num_params = params.size(0);
	if (!is_divisible(num_params, lod_meta.n_params)) {
		throw std::runtime_error(std::string{"LoTDEncoding::fwd: Expect size of `params`="} + std::to_string(num_params) + std::string{" to be an integral multiple of `n_param`="} + std::to_string(lod_meta.n_params));
	}

	const int32_t max_level = max_level_.value_or(lod_meta.n_levels);

	at::Tensor batch_inds;
	at::TensorArg batch_inds_arg(batch_inds, "batch_inds", 3);
	if (batch_inds_.has_value()) {
		batch_inds = batch_inds_.value();
		at::checkDim(__func__, batch_inds_arg, 1);
		at::checkSameGPU(__func__, input_arg, batch_inds_arg);
		at::checkContiguous(__func__, batch_inds_arg);
		at::checkScalarType(__func__, batch_inds_arg, at::kLong);
		at::checkSize(__func__, batch_inds_arg, {batch_size});
	}

	at::Tensor batch_offset;
	if (batch_offsets_.has_value()) {
		batch_offset = batch_offsets_.value();
		at::TensorArg batch_offset_arg(batch_offset, "batch_offset", 4);
		at::checkDim(__func__, batch_offset_arg, 1);
		at::checkSameGPU(__func__, input_arg, batch_offset_arg);
		at::checkContiguous(__func__, batch_offset_arg);
		at::checkScalarType(__func__, batch_offset_arg, at::kLong);
	}

	uint32_t batch_data_size = 0;
	if (batch_data_size_.has_value()) {
		batch_data_size = batch_data_size_.value();
		if (! (batch_data_size == 0 || is_divisible(batch_size, batch_data_size))) {
			throw std::runtime_error("LoTDEncoding::fwd: Expect nonzero `batch_data_size`=" + std::to_string(batch_data_size) + " to be a divisor of `batch_size`=" + std::to_string(batch_size));
		}
	}

	bool need_input_grad = need_input_grad_.value_or(input.requires_grad());
	at::Tensor dy_dx;
	if (need_input_grad) {
		dy_dx = at::zeros( {batch_size, lod_meta.n_encoded_dims * lod_meta.n_dims_to_encode}, input.options());
	}

	at::Tensor output = at::zeros({ batch_size, lod_meta.n_encoded_dims}, params.options());
	if (max_level <= -1) return {output, dy_dx};

	switch (lod_meta.n_dims_to_encode) {
		// case 1: lod_fwd_impl<1>(lod_meta, input, params, batch_inds_, batch_offsets_, batch_data_size, max_level, need_input_grad, output, dy_dx); break;
		case 2: lod_fwd_impl<2>(lod_meta, input, params, batch_inds_, batch_offsets_, batch_data_size, max_level, need_input_grad, output, dy_dx); break;
		case 3: lod_fwd_impl<3>(lod_meta, input, params, batch_inds_, batch_offsets_, batch_data_size, max_level, need_input_grad, output, dy_dx); break;
		case 4: lod_fwd_impl<4>(lod_meta, input, params, batch_inds_, batch_offsets_, batch_data_size, max_level, need_input_grad, output, dy_dx); break;
		default: throw std::runtime_error("LoTDEncoding::fwd: `n_dims_to_encode` must be 2 or 3."); break;
	}

	return {output, dy_dx};
}

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
) {
	return lod_fwd_common(lod_meta, input, params, batch_inds_, batch_offsets_, batch_data_size_, max_level_, need_input_grad_);
}

// std::tuple<at::Tensor, at::Tensor> lod_forest_fwd(
// 	std::tuple<LoDMeta, ForestMeta> metas,
// 	at::Tensor input,
// 	at::Tensor params,
// 	// Optional
// 	at::optional<at::Tensor> batch_inds_,
// 	at::optional<at::Tensor> batch_offsets_,
// 	at::optional<uint32_t> batch_data_size_, 
// 	at::optional<int32_t> max_level_, 
// 	at::optional<bool> need_input_grad_
// ) {
// 	LoDMeta& lod_meta = std::get<0>(metas);
// 	ForestMeta& forest_meta = std::get<1>(metas);
// 	return lod_fwd_common(lod_meta, forest_meta, input, params, batch_inds_, batch_offsets_, batch_data_size_, max_level_, need_input_grad_);
// }

std::tuple<at::Tensor, at::Tensor> lod_bwd_common(
	LoDMeta lod_meta,
	at::Tensor dL_dy,
	at::Tensor input,
	at::Tensor params,
	// Optional
	at::optional<at::Tensor> dy_dx_, // Needed when `need_input_grad`
	at::optional<at::Tensor> batch_inds_,
	at::optional<at::Tensor> batch_offsets_,
	at::optional<uint32_t> batch_data_size_, 
	at::optional<int32_t> max_level_, 
	at::optional<bool> need_input_grad_, 
	at::optional<bool> need_param_grad_
) {
	at::TensorArg dL_dy_arg(dL_dy, "dL_dy", 1);
	at::TensorArg input_arg(input, "x", 2);
	at::TensorArg params_arg(params, "grid", 3);

	at::checkDim(__func__, dL_dy_arg, 2);
	at::checkDim(__func__, input_arg, 2);
	at::checkDim(__func__, params_arg, 1);
	at::checkAllSameGPU(__func__, {input_arg, params_arg, dL_dy_arg});
	at::checkAllContiguous(__func__, {input_arg, params_arg, dL_dy_arg});
	at::checkScalarTypes(__func__, dL_dy_arg, {at::kHalf, at::kFloat});
	at::checkScalarTypes(__func__, input_arg, {at::kHalf, at::kFloat});
	at::checkScalarTypes(__func__, params_arg, {at::kHalf, at::kFloat});
	at::checkSameType(__func__, dL_dy_arg, params_arg);

	const uint32_t batch_size = input.size(0);
	at::checkSize(__func__, input_arg, 1, lod_meta.n_dims_to_encode);
	at::checkSize(__func__, dL_dy_arg, {batch_size, lod_meta.n_encoded_dims});
	const uint32_t num_params = params.size(0);
	if (!is_divisible(num_params, lod_meta.n_params)) {
		throw std::runtime_error(std::string{"LoTDEncoding::bwd: Expect size of `params`="} + std::to_string(num_params) + std::string{" to be an integral multiple of `n_param`="} + std::to_string(lod_meta.n_params));
	}

	const int32_t max_level = max_level_.value_or(lod_meta.n_levels);

	at::Tensor dy_dx;
	if (dy_dx_.has_value()) {
		dy_dx = dy_dx_.value();
		at::TensorArg dy_dx_arg(dy_dx, "dy_dx", 4);
		at::checkDim(__func__, dy_dx_arg, 2);
		at::checkSameGPU(__func__, input_arg, dy_dx_arg);
		at::checkContiguous(__func__, dy_dx_arg);
		at::checkScalarTypes(__func__, dy_dx_arg, {at::kHalf, at::kFloat});
		at::checkSameType(__func__, input_arg, dy_dx_arg);
		at::checkSize(__func__, dy_dx_arg, {batch_size, lod_meta.n_encoded_dims * lod_meta.n_dims_to_encode });
	}

	at::Tensor batch_inds;
	if (batch_inds_.has_value()) {
		batch_inds = batch_inds_.value();
		at::TensorArg batch_inds_arg(batch_inds, "batch_inds", 5);
		at::checkDim(__func__, batch_inds_arg, 1);
		at::checkSameGPU(__func__, input_arg, batch_inds_arg);
		at::checkContiguous(__func__, batch_inds_arg);
		at::checkScalarType(__func__, batch_inds_arg, at::kLong);
		at::checkSize(__func__, batch_inds_arg, {batch_size});
	}

	at::Tensor batch_offset;
	if (batch_offsets_.has_value()) {
		batch_offset = batch_offsets_.value();
		at::TensorArg batch_offset_arg(batch_offset, "batch_offset", 6);
		at::checkDim(__func__, batch_offset_arg, 1);
		at::checkSameGPU(__func__, input_arg, batch_offset_arg);
		at::checkContiguous(__func__, batch_offset_arg);
		at::checkScalarType(__func__, batch_offset_arg, at::kLong);
	}

	uint32_t batch_data_size = 0;
	if (batch_data_size_.has_value()) {
		batch_data_size = batch_data_size_.value();
		if (! (batch_data_size == 0 || is_divisible(batch_size, batch_data_size))) {
			throw std::runtime_error("LoTDEncoding::bwd: Expect nonzero `batch_data_size`=" + std::to_string(batch_data_size) + " to be a divisor of `batch_size`=" + std::to_string(batch_size));
		}
	}

	bool need_input_grad = need_input_grad_.value_or(input.requires_grad());
	at::Tensor dL_dx;
	if (need_input_grad) {
		if (!dy_dx_.has_value()) {
			throw std::runtime_error("LoTDEncoding::bwd: need `dy_dx` to comput `dL_dx`.");
		}
		dL_dx = at::zeros( {batch_size, lod_meta.n_dims_to_encode}, input.options());
	}

	bool need_param_grad = need_param_grad_.value_or(params.requires_grad());
	at::Tensor dL_dparam;
	if (need_param_grad) {
		dL_dparam = at::zeros( {params.size(0)}, params.options() );
	}

	if (max_level <= -1) return {dL_dx, dL_dparam};

	if (need_input_grad || need_param_grad) {
		switch (lod_meta.n_dims_to_encode) {
			// case 1: lod_bwd_impl<1>(lod_meta, dL_dy, input, params, dy_dx_, batch_inds_, batch_offsets_, batch_data_size, max_level, need_input_grad, need_param_grad, dL_dx, dL_dparam); break;
			case 2: lod_bwd_impl<2>(lod_meta, dL_dy, input, params, dy_dx_, batch_inds_, batch_offsets_, batch_data_size, max_level, need_input_grad, need_param_grad, dL_dx, dL_dparam); break;
			case 3: lod_bwd_impl<3>(lod_meta, dL_dy, input, params, dy_dx_, batch_inds_, batch_offsets_, batch_data_size, max_level, need_input_grad, need_param_grad, dL_dx, dL_dparam); break;
			case 4: lod_bwd_impl<4>(lod_meta, dL_dy, input, params, dy_dx_, batch_inds_, batch_offsets_, batch_data_size, max_level, need_input_grad, need_param_grad, dL_dx, dL_dparam); break;
			default: throw std::runtime_error("LoTDEncoding::bwd: `n_dims_to_encode` must be 2 or 3."); break;
		}
	}
	return {dL_dx, dL_dparam};
} 

std::tuple<at::Tensor, at::Tensor> lod_bwd(
	LoDMeta lod_meta,
	at::Tensor dL_dy,
	at::Tensor input,
	at::Tensor params,
	// Optional
	at::optional<at::Tensor> dy_dx_, // Needed when `need_input_grad`
	at::optional<at::Tensor> batch_inds_,
	at::optional<at::Tensor> batch_offsets_,
	at::optional<uint32_t> batch_data_size_, 
	at::optional<int32_t> max_level_, 
	at::optional<bool> need_input_grad_, 
	at::optional<bool> need_param_grad_
) {
	return lod_bwd_common(lod_meta, dL_dy, input, params, dy_dx_, batch_inds_, batch_offsets_, batch_data_size_, max_level_, need_input_grad_, need_param_grad_);
}

// std::tuple<at::Tensor, at::Tensor> lod_forest_bwd(
// 	std::tuple<LoDMeta, ForestMeta> metas,
// 	at::Tensor dL_dy,
// 	at::Tensor input,
// 	at::Tensor params,
// 	// Optional
// 	at::optional<at::Tensor> dy_dx_, // Needed when `need_input_grad`
// 	at::optional<at::Tensor> batch_inds_,
// 	at::optional<at::Tensor> batch_offsets_,
// 	at::optional<uint32_t> batch_data_size_, 
// 	at::optional<int32_t> max_level_, 
// 	at::optional<bool> need_input_grad_, 
// 	at::optional<bool> need_param_grad_
// ) {
// 	LoDMeta& lod_meta = std::get<0>(metas);
// 	ForestMeta& forest_meta = std::get<1>(metas);
// 	return lod_bwd_common(lod_meta, forest_meta, dL_dy, input, params, dy_dx_, batch_inds_, batch_offsets_, batch_data_size_, max_level_, need_input_grad_, need_param_grad_);
// }

std::tuple<at::Tensor, at::Tensor, at::Tensor> lod_bwd_bwd_input_common(
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
) {
	at::TensorArg dL_ddLdx_arg(dL_ddLdx, "dL_ddLdx", 1);
	at::TensorArg dL_dy_arg(dL_dy, "dL_dy", 2);
	at::TensorArg input_arg(input, "x", 3);
	at::TensorArg params_arg(params, "grid", 4);

	at::checkDim(__func__, dL_ddLdx_arg, 2);
	at::checkDim(__func__, dL_dy_arg, 2);
	at::checkDim(__func__, input_arg, 2);
	at::checkDim(__func__, params_arg, 1);
	at::checkAllSameGPU(__func__, {input_arg, params_arg, dL_dy_arg, dL_ddLdx_arg});
	at::checkAllContiguous(__func__, {input_arg, params_arg, dL_dy_arg, dL_ddLdx_arg});
	at::checkScalarTypes(__func__, dL_ddLdx_arg, {at::kHalf, at::kFloat});
	at::checkScalarTypes(__func__, dL_dy_arg, {at::kHalf, at::kFloat});
	at::checkScalarTypes(__func__, input_arg, {at::kHalf, at::kFloat});
	at::checkScalarTypes(__func__, params_arg, {at::kHalf, at::kFloat});
	at::checkSameType(__func__, dL_ddLdx_arg, input_arg);
	at::checkSameType(__func__, dL_dy_arg, params_arg);

	const uint32_t batch_size = input.size(0);
	at::checkSize(__func__, input_arg, 1, lod_meta.n_dims_to_encode);
	at::checkSize(__func__, dL_ddLdx_arg, {batch_size, lod_meta.n_dims_to_encode});
	at::checkSize(__func__, dL_dy_arg, {batch_size, lod_meta.n_encoded_dims});
	const uint32_t num_params = params.size(0);
	if (!is_divisible(num_params, lod_meta.n_params)) {
		throw std::runtime_error(std::string{"LoTDEncoding::bwd_bwd_input: Expect size of `params`="} + std::to_string(num_params) + std::string{" to be an integral multiple of `n_param`="} + std::to_string(lod_meta.n_params));
	}

	const int32_t max_level = max_level_.value_or(lod_meta.n_levels);

	at::Tensor dy_dx;
	if (dy_dx_.has_value()) {
		dy_dx = dy_dx_.value();
		at::TensorArg dy_dx_arg(dy_dx, "dy_dx", 5);
		at::checkDim(__func__, dy_dx_arg, 2);
		at::checkSameGPU(__func__, input_arg, dy_dx_arg);
		at::checkContiguous(__func__, dy_dx_arg);
		at::checkScalarTypes(__func__, dy_dx_arg, {at::kHalf, at::kFloat});
		at::checkSameType(__func__, input_arg, dy_dx_arg);
		at::checkSize(__func__, dy_dx_arg, {batch_size, lod_meta.n_encoded_dims * lod_meta.n_dims_to_encode });
	}

	at::Tensor batch_inds;
	if (batch_inds_.has_value()) {
		batch_inds = batch_inds_.value();
		at::TensorArg batch_inds_arg(batch_inds, "batch_inds", 6);
		at::checkDim(__func__, batch_inds_arg, 1);
		at::checkSameGPU(__func__, input_arg, batch_inds_arg);
		at::checkContiguous(__func__, batch_inds_arg);
		at::checkScalarType(__func__, batch_inds_arg, at::kLong);
		at::checkSize(__func__, batch_inds_arg, {batch_size});
	}

	at::Tensor batch_offset;
	if (batch_offsets_.has_value()) {
		batch_offset = batch_offsets_.value();
		at::TensorArg batch_offset_arg(batch_offset, "batch_offset", 7);
		at::checkDim(__func__, batch_offset_arg, 1);
		at::checkSameGPU(__func__, input_arg, batch_offset_arg);
		at::checkContiguous(__func__, batch_offset_arg);
		at::checkScalarType(__func__, batch_offset_arg, at::kLong);
	}

	uint32_t batch_data_size = 0;
	if (batch_data_size_.has_value()) {
		batch_data_size = batch_data_size_.value();
		if (! (batch_data_size == 0 || is_divisible(batch_size, batch_data_size))) {
			throw std::runtime_error("LoTDEncoding::bwd_bwd_input: Expect nonzero `batch_data_size`=" + std::to_string(batch_data_size) + " to be a divisor of `batch_size`=" + std::to_string(batch_size));
		}
	}

	bool need_dLdy_grad = need_dLdinput_ddLdoutput_.value_or(dL_dy.requires_grad());
	at::Tensor dL_ddLdy;
	if (need_dLdy_grad) {
		if (!dy_dx_.has_value()) {
			throw std::runtime_error("LoTDEncoding::bwd_bwd_input: need `dy_dx` to compute `dL_d(dLdy)`.");
		}
		dL_ddLdy = at::zeros({ batch_size, lod_meta.n_encoded_dims}, dL_dy.options());
	}

	bool need_input_grad = need_dLdinput_dinput_.value_or(input.requires_grad());
	at::Tensor dL_dx;
	if (need_input_grad) {
		dL_dx = at::zeros({ batch_size, lod_meta.n_dims_to_encode }, input.options());
	}

	bool need_param_grad = need_dLdinput_dparams_.value_or(params.requires_grad());
	at::Tensor dL_dparams;
	if (need_param_grad) {
		dL_dparams = at::zeros({ params.size(0) }, params.options());
	}

	if (max_level <= -1) return {dL_ddLdy, dL_dparams, dL_dx};

	if (need_dLdy_grad || need_input_grad || need_param_grad) {
		switch (lod_meta.n_dims_to_encode) {
			// case 1: lod_bwd_bwd_input_impl<1>(lod_meta, dL_ddLdx, dL_dy, input, params, dy_dx_, batch_inds_, batch_offsets_, batch_data_size, max_level, need_dLdy_grad, need_input_grad, need_param_grad, dL_ddLdy, dL_dx, dL_dparams); break;
			case 2: lod_bwd_bwd_input_impl<2>(lod_meta, dL_ddLdx, dL_dy, input, params, dy_dx_, batch_inds_, batch_offsets_, batch_data_size, max_level, need_dLdy_grad, need_input_grad, need_param_grad, dL_ddLdy, dL_dx, dL_dparams); break;
			case 3: lod_bwd_bwd_input_impl<3>(lod_meta, dL_ddLdx, dL_dy, input, params, dy_dx_, batch_inds_, batch_offsets_, batch_data_size, max_level, need_dLdy_grad, need_input_grad, need_param_grad, dL_ddLdy, dL_dx, dL_dparams); break;
			case 4: lod_bwd_bwd_input_impl<4>(lod_meta, dL_ddLdx, dL_dy, input, params, dy_dx_, batch_inds_, batch_offsets_, batch_data_size, max_level, need_dLdy_grad, need_input_grad, need_param_grad, dL_ddLdy, dL_dx, dL_dparams); break;
			default: throw std::runtime_error("LoTDEncoding::bwd_bwd_input: `n_dims_to_encode` must be 2 or 3."); break;
		}
	}

	return {dL_ddLdy, dL_dparams, dL_dx};
}

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
) {
	return lod_bwd_bwd_input_common(lod_meta, dL_ddLdx, dL_dy, input, params, dy_dx_, batch_inds_, batch_offsets_, batch_data_size_, max_level_, need_dLdinput_ddLdoutput_, need_dLdinput_dparams_, need_dLdinput_dinput_);
}

at::Tensor lod_get_grid_index_common(
	LoDMeta lod_meta,
	at::Tensor input,
	// Optional
	at::optional<at::Tensor> batch_inds_,
	at::optional<at::Tensor> batch_offsets_,
	at::optional<uint32_t> batch_data_size_, 
	at::optional<int32_t> max_level_
) {
	at::TensorArg input_arg(input, "x", 1); 

	at::checkDim(__func__, input_arg, 2);
	at::checkScalarTypes(__func__, input_arg, {at::kHalf, at::kFloat});

	const uint32_t batch_size = input.size(0);
	at::checkSize(__func__, input_arg, 1, lod_meta.n_dims_to_encode);

	const int32_t max_level = max_level_.value_or(lod_meta.n_levels);

	for (uint32_t l=0; l < lod_meta.n_levels; ++l) {
		LoDType tp = (LoDType)lod_meta.level_types[l];
		if (!(tp == LoDType::Dense || tp == LoDType::Hash)) {
			throw std::runtime_error("LoTDEncoding::get_grid_index: Only support Dense/Hash type.");
		}
	}

	at::Tensor batch_inds;
	at::TensorArg batch_inds_arg(batch_inds, "batch_inds", 2);
	if (batch_inds_.has_value()) {
		batch_inds = batch_inds_.value();
		at::checkDim(__func__, batch_inds_arg, 1);
		at::checkSameGPU(__func__, input_arg, batch_inds_arg);
		at::checkContiguous(__func__, batch_inds_arg);
		at::checkScalarType(__func__, batch_inds_arg, at::kLong);
		at::checkSize(__func__, batch_inds_arg, {batch_size});
	}

	at::Tensor batch_offset;
	if (batch_offsets_.has_value()) {
		batch_offset = batch_offsets_.value();
		at::TensorArg batch_offset_arg(batch_offset, "batch_offset", 3);
		at::checkDim(__func__, batch_offset_arg, 1);
		at::checkSameGPU(__func__, input_arg, batch_offset_arg);
		at::checkContiguous(__func__, batch_offset_arg);
		at::checkScalarType(__func__, batch_offset_arg, at::kLong);
	}

	uint32_t batch_data_size = 0;
	if (batch_data_size_.has_value()) {
		batch_data_size = batch_data_size_.value();
		if (! (batch_data_size == 0 || is_divisible(batch_size, batch_data_size))) {
			throw std::runtime_error("LoTDEncoding::get_grid_index: Expect nonzero `batch_data_size`=" + std::to_string(batch_data_size) + " to be a divisor of `batch_size`=" + std::to_string(batch_size));
		}
	}

	// For Dense & Hash type only
	at::Tensor grid_inds = at::zeros({ batch_size, lod_meta.n_encoded_dims, (1<<lod_meta.n_dims_to_encode)}, input.options().dtype(at::kLong));
	if (max_level <= -1) return grid_inds;

	switch (lod_meta.n_dims_to_encode) {
		// case 1: lod_get_grid_index_impl<1>(lod_meta, input, batch_inds_, batch_offsets_, batch_data_size, max_level, grid_inds); break;
		case 2: lod_get_grid_index_impl<2>(lod_meta, input, batch_inds_, batch_offsets_, batch_data_size, max_level, grid_inds); break;
		case 3: lod_get_grid_index_impl<3>(lod_meta, input, batch_inds_, batch_offsets_, batch_data_size, max_level, grid_inds); break;
		case 4: lod_get_grid_index_impl<4>(lod_meta, input, batch_inds_, batch_offsets_, batch_data_size, max_level, grid_inds); break;
		default: throw std::runtime_error("LoTDEncoding::get_grid_index: `n_dims_to_encode` must be 2 or 3."); break;
	}
	return grid_inds;
}

// std::tuple<at::Tensor, at::Tensor, at::Tensor> lod_forest_bwd_bwd_input(
// 	std::tuple<LoDMeta, ForestMeta> metas,
// 	at::Tensor dL_ddLdx,
// 	at::Tensor dL_dy,
// 	at::Tensor input,
// 	at::Tensor params,
// 	// Optional
// 	at::optional<at::Tensor> dy_dx_, // Needed when `need_dLdy_grad`
// 	at::optional<at::Tensor> batch_inds_,
// 	at::optional<at::Tensor> batch_offsets_,
// 	at::optional<uint32_t> batch_data_size_,
// 	at::optional<int32_t> max_level_, 
// 	at::optional<bool> need_dLdinput_ddLdoutput_,
// 	at::optional<bool> need_dLdinput_dparams_,
// 	at::optional<bool> need_dLdinput_dinput_
// ) {
// 	LoDMeta &lod_meta = std::get<0>(metas);
// 	ForestMeta &forest_meta = std::get<1>(metas);
// 	return lod_bwd_bwd_input_common(lod_meta, forest_meta, dL_ddLdx, dL_dy, input, params, dy_dx_, batch_inds_, batch_offsets_, batch_data_size_, max_level_, need_dLdinput_ddLdoutput_, need_dLdinput_dparams_, need_dLdinput_dinput_);
// }

at::Tensor lod_get_grid_index(
	LoDMeta lod_meta,
	at::Tensor input,
	// Optional
	at::optional<at::Tensor> batch_inds_,
	at::optional<at::Tensor> batch_offsets_,
	at::optional<uint32_t> batch_data_size_,
	at::optional<int32_t> max_level_
) {
	return lod_get_grid_index_common(lod_meta, input, batch_inds_, batch_offsets_, batch_data_size_, max_level_);
}

} // namespace lotd::torch

} // namespace lotd


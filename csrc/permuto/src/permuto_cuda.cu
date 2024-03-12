/** @file   permuto_cuda.cu
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

#include <stdint.h>
#include <string>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <vector>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>

#include <permuto/permuto.h>
#include <permuto/permuto_cuda.h>

namespace permuto {

const std::vector<uint32_t> supported_n_input_dims{2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,24,28,32,36,40,48,56,64}; 

void PermutoEncMeta::create_meta(
	const uint32_t n_input_dim, 
	const uint32_t hashmap_size, 
	const std::vector<double>& res_list,
	const std::vector<int32_t>& n_feats_list
) {

	if (! (res_list.size() == n_feats_list.size())) {
		throw std::runtime_error("PermutoEncImpl: Expect `res_list` and `n_feats_list` to have the same length");
	}

	n_dims_to_encode = (uint32_t)n_input_dim;
	if (std::find(supported_n_input_dims.begin(), supported_n_input_dims.end(), n_dims_to_encode) != supported_n_input_dims.end()) {
	} else {
		// Not found
		std::stringstream ss; 
		ss << "["; 
		for (int i=0; i<supported_n_input_dims.size(); ++i) {
			if (i != 0) ss << ","; 
			ss << supported_n_input_dims[i]; 
		}
		ss << "]"; 
		std::string supported_n_input_dims_str = ss.str(); 
		throw std::runtime_error(std::string("PermutoEncImpl: Currently not supported n_dims_to_encode=") + std::to_string(n_dims_to_encode)
			+ std::string(", while what's supported are " + supported_n_input_dims_str)); 
	}

	n_levels = res_list.size(); 

	if (n_levels > MAX_N_LEVELS) {
		throw std::runtime_error(std::string{"PermutoEncImpl: num_level="} + std::to_string(n_levels) + std::string{" exceeds maximum level="} + std::to_string(MAX_N_LEVELS));
	}

	level_scales0.resize(n_levels); 
	level_n_feats.resize(n_levels); 
	level_n_params.resize(n_levels); 
	level_offsets.resize(n_levels+1); 
	level_sizes.resize(n_levels);

	level_scales_multidim = at::zeros({n_levels, n_input_dim}, at::TensorOptions().dtype(at::kFloat).device(at::kCPU));

	// if (is_all_divisible(n_feats_list, 8)) {
	// 	n_feat_per_pseudo_lvl = 8;
	// } else 
	if (is_all_divisible(n_feats_list, 4) ) {
		n_feat_per_pseudo_lvl = 4;
	} else if (is_all_divisible(n_feats_list, 2)) {
		n_feat_per_pseudo_lvl = 2;
	} else {
		throw std::runtime_error("PermutoEncImpl: the greatest common divisor of `n_feats_list` must be at least 2");
	}

	n_pseudo_levels = 0;
	n_encoded_dims = 0;
	uint32_t max_params = std::numeric_limits<uint32_t>::max()/2;
	uint32_t accumulated_num_params = 0;
	double accumulated_num_params_float = 0.0f;

	double invStdDev = 1.0;
	for (uint32_t lvl=0; lvl < n_levels; ++lvl) {
		const uint32_t n_feat = n_feats_list[lvl];

		level_n_feats[lvl] = (uint32_t)n_feat;

		// Number of pseudo levels in current actual level
		uint32_t n_pseudo_levels_cur_lvl = n_feat / n_feat_per_pseudo_lvl;
		n_pseudo_levels += n_pseudo_levels_cur_lvl;
		n_encoded_dims += n_feat;

		double res = res_list[lvl];
		level_scales0[lvl] = res; 

		// Compute level_scales_multidim
		for (uint32_t dim=0; dim<n_dims_to_encode; ++dim) {
			level_scales_multidim[lvl][dim] = (double)res / std::sqrt((double)(dim+1)*(dim+2)) * invStdDev;
		}

		accumulated_num_params_float += (double)hashmap_size * n_feat;
		if (accumulated_num_params_float > (double)max_params) {
			throw std::runtime_error("PermutoEncImpl: param size too large.");
		}

		uint32_t n_params_cur_lvl = hashmap_size * n_feat;

		level_sizes[lvl] = (uint32_t)hashmap_size; 
		level_n_params[lvl] = (uint32_t)n_params_cur_lvl;

		level_offsets[lvl] = (uint32_t)accumulated_num_params; 
		accumulated_num_params += n_params_cur_lvl;
	}
	level_offsets[n_levels] = (uint32_t)accumulated_num_params;
	n_params = accumulated_num_params;

	// Compute map_levels, map_cnt
	map_levels.resize(n_pseudo_levels);
	map_cnt.resize(n_pseudo_levels);
	uint32_t acc_n_pseudo_levels = 0;
	for (uint32_t lvl = 0; lvl < n_levels; lvl++) {
		uint32_t n_pseudo_levels_cur_lvl = level_n_feats[lvl] / n_feat_per_pseudo_lvl;
		for (uint32_t j=0; j<n_pseudo_levels_cur_lvl; ++j) {
			map_levels[acc_n_pseudo_levels + j] = (uint32_t)lvl;
			map_cnt[acc_n_pseudo_levels + j] = (uint32_t)j;
		}
		acc_n_pseudo_levels += n_pseudo_levels_cur_lvl; 
	}

	if (n_encoded_dims > 1024) {
		throw std::runtime_error("PermutoEncImpl: total number of features too large. Shoule be <= 1024.");
	}
}

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
) {	
	at::TensorArg positions_arg(positions, "positions", 1); 
	at::TensorArg lattice_values_arg(lattice_values, "lattice_values", 2);

	at::checkDim(__func__, positions_arg, 2);
	at::checkDim(__func__, lattice_values_arg, 1);
	at::checkAllSameGPU(__func__, {positions_arg, lattice_values_arg});
	at::checkAllContiguous(__func__, {positions_arg, lattice_values_arg});
	at::checkScalarTypes(__func__, positions_arg, {at::kHalf, at::kFloat});
	at::checkScalarTypes(__func__, lattice_values_arg, {at::kHalf, at::kFloat});

	const uint32_t batch_size = positions.size(0);
	at::checkSize(__func__, positions_arg, 1, meta.n_dims_to_encode);
	const uint32_t num_params = lattice_values.size(0);
	if (!is_divisible(num_params, meta.n_params)) {
		throw std::runtime_error(std::string{"PermutoEncImpl::fwd: Expect size of `params`="} + std::to_string(num_params) + std::string{" to be an integral multiple of `n_param`="} + std::to_string(meta.n_params));
	}

	const int32_t max_level = max_level_.value_or(meta.n_levels);

	at::Tensor level_random_shifts;
	at::TensorArg level_random_shifts_arg(level_random_shifts, "level_random_shifts", 3);
	if (level_random_shifts_.has_value()) {
		level_random_shifts = level_random_shifts_.value().to(at::kFloat);
		at::checkSize(__func__, level_random_shifts_arg, {meta.n_levels, meta.n_dims_to_encode});
		at::checkSameGPU(__func__, positions_arg, level_random_shifts_arg);
		at::checkContiguous(__func__, level_random_shifts_arg);
	}

	at::Tensor batch_inds;
	at::TensorArg batch_inds_arg(batch_inds, "batch_inds", 4);
	if (batch_inds_.has_value()) {
		batch_inds = batch_inds_.value();
		at::checkDim(__func__, batch_inds_arg, 1);
		at::checkSameGPU(__func__, positions_arg, batch_inds_arg);
		at::checkContiguous(__func__, batch_inds_arg);
		at::checkScalarType(__func__, batch_inds_arg, at::kLong);
		at::checkSize(__func__, batch_inds_arg, {batch_size});
	}

	at::Tensor batch_offset;
	if (batch_offsets_.has_value()) {
		batch_offset = batch_offsets_.value();
		at::TensorArg batch_offset_arg(batch_offset, "batch_offset", 5);
		at::checkDim(__func__, batch_offset_arg, 1);
		at::checkSameGPU(__func__, positions_arg, batch_offset_arg);
		at::checkContiguous(__func__, batch_offset_arg);
		at::checkScalarType(__func__, batch_offset_arg, at::kLong);
	}

	uint32_t batch_data_size = 0;
	if (batch_data_size_.has_value()) {
		batch_data_size = batch_data_size_.value();
		if (! (batch_data_size == 0 || is_divisible(batch_size, batch_data_size))) {
			throw std::runtime_error("PermutoEncImpl::fwd: Expect nonzero `batch_data_size`=" + std::to_string(batch_data_size) + " to be a divisor of `batch_size`=" + std::to_string(batch_size));
		}
	}

	at::Tensor encoded = at::zeros({batch_size, meta.n_encoded_dims}, lattice_values.options());
	if (max_level <= -1) {
		return encoded;
	}
	
	switch (meta.n_dims_to_encode) {
		case 2:  permuto_enc_fwd_impl< 2>(meta, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, encoded); break;
		case 3:  permuto_enc_fwd_impl< 3>(meta, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, encoded); break;
		case 4:  permuto_enc_fwd_impl< 4>(meta, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, encoded); break;
		case 5:  permuto_enc_fwd_impl< 5>(meta, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, encoded); break;
		case 6:  permuto_enc_fwd_impl< 6>(meta, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, encoded); break;
		case 7:  permuto_enc_fwd_impl< 7>(meta, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, encoded); break;
		case 8:  permuto_enc_fwd_impl< 8>(meta, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, encoded); break;
		case 9:  permuto_enc_fwd_impl< 9>(meta, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, encoded); break;
		case 10: permuto_enc_fwd_impl<10>(meta, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, encoded); break;
		case 11: permuto_enc_fwd_impl<11>(meta, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, encoded); break;
		case 12: permuto_enc_fwd_impl<12>(meta, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, encoded); break;
		case 13: permuto_enc_fwd_impl<13>(meta, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, encoded); break;
		case 14: permuto_enc_fwd_impl<14>(meta, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, encoded); break;
		case 15: permuto_enc_fwd_impl<15>(meta, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, encoded); break;
		case 16: permuto_enc_fwd_impl<16>(meta, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, encoded); break;
		case 17: permuto_enc_fwd_impl<17>(meta, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, encoded); break;
		case 18: permuto_enc_fwd_impl<18>(meta, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, encoded); break;
		case 19: permuto_enc_fwd_impl<19>(meta, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, encoded); break;
		case 20: permuto_enc_fwd_impl<20>(meta, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, encoded); break;
		case 24: permuto_enc_fwd_impl<24>(meta, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, encoded); break;
		case 28: permuto_enc_fwd_impl<28>(meta, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, encoded); break;
		case 32: permuto_enc_fwd_impl<32>(meta, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, encoded); break;
		case 36: permuto_enc_fwd_impl<36>(meta, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, encoded); break;
		case 40: permuto_enc_fwd_impl<40>(meta, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, encoded); break;
		case 48: permuto_enc_fwd_impl<48>(meta, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, encoded); break;
		case 56: permuto_enc_fwd_impl<56>(meta, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, encoded); break;
		case 64: permuto_enc_fwd_impl<64>(meta, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, encoded); break;
		default: throw std::runtime_error(std::string("PermutoEncImpl::fwd: `n_dims_to_encode`=") + std::to_string(meta.n_dims_to_encode) + std::string(" is not yet implemented.")); break;
	}

	return encoded; 
}

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
	at::optional<uint32_t> batch_data_size_, 
	at::optional<int32_t> max_level_,
	at::optional<uint32_t> max_pos_dims_, 
	at::optional<bool> need_input_grad_, 
	at::optional<bool> need_param_grad_
) {
	at::TensorArg dL_dy_arg(dL_dy, "dL_dy", 1);
	at::TensorArg positions_arg(positions, "positions", 2); 
	at::TensorArg lattice_values_arg(lattice_values, "lattice_values", 3);

	at::checkDim(__func__, dL_dy_arg, 2);
	at::checkDim(__func__, positions_arg, 2);
	at::checkDim(__func__, lattice_values_arg, 1);
	at::checkAllSameGPU(__func__, {positions_arg, lattice_values_arg, dL_dy_arg});
	at::checkAllContiguous(__func__, {positions_arg, lattice_values_arg, dL_dy_arg});
	at::checkScalarTypes(__func__, dL_dy_arg, {at::kHalf, at::kFloat});
	at::checkScalarTypes(__func__, positions_arg, {at::kHalf, at::kFloat});
	at::checkScalarTypes(__func__, lattice_values_arg, {at::kHalf, at::kFloat});
	at::checkSameType(__func__, dL_dy_arg, lattice_values_arg);

	const uint32_t batch_size = positions.size(0);
	at::checkSize(__func__, dL_dy_arg, {batch_size, meta.n_encoded_dims});
	at::checkSize(__func__, positions_arg, 1, meta.n_dims_to_encode);
	const uint32_t num_params = lattice_values.size(0);
	if (!is_divisible(num_params, meta.n_params)) {
		throw std::runtime_error(std::string{"PermutoEncImpl::bwd: Expect size of `params`="} + std::to_string(num_params) + std::string{" to be an integral multiple of `n_param`="} + std::to_string(meta.n_params));
	}

	const int32_t max_level = max_level_.value_or(meta.n_levels);
	const uint32_t max_pos_dims = max_pos_dims_.value_or(meta.n_dims_to_encode); 

	at::Tensor level_random_shifts;
	at::TensorArg level_random_shifts_arg(level_random_shifts, "level_random_shifts", 4);
	if (level_random_shifts_.has_value()) {
		level_random_shifts = level_random_shifts_.value();
		at::checkSize(__func__, level_random_shifts_arg, {meta.n_levels, meta.n_dims_to_encode});
		at::checkSameGPU(__func__, positions_arg, level_random_shifts_arg);
		at::checkSameType(__func__, positions_arg, level_random_shifts_arg);
		at::checkContiguous(__func__, level_random_shifts_arg);
	}

	at::Tensor batch_inds;
	at::TensorArg batch_inds_arg(batch_inds, "batch_inds", 5);
	if (batch_inds_.has_value()) {
		batch_inds = batch_inds_.value();
		at::checkDim(__func__, batch_inds_arg, 1);
		at::checkSameGPU(__func__, positions_arg, batch_inds_arg);
		at::checkContiguous(__func__, batch_inds_arg);
		at::checkScalarType(__func__, batch_inds_arg, at::kLong);
		at::checkSize(__func__, batch_inds_arg, {batch_size});
	}

	at::Tensor batch_offset;
	if (batch_offsets_.has_value()) {
		batch_offset = batch_offsets_.value();
		at::TensorArg batch_offset_arg(batch_offset, "batch_offset", 6);
		at::checkDim(__func__, batch_offset_arg, 1);
		at::checkSameGPU(__func__, positions_arg, batch_offset_arg);
		at::checkContiguous(__func__, batch_offset_arg);
		at::checkScalarType(__func__, batch_offset_arg, at::kLong);
	}

	uint32_t batch_data_size = 0;
	if (batch_data_size_.has_value()) {
		batch_data_size = batch_data_size_.value();
		if (! (batch_data_size == 0 || is_divisible(batch_size, batch_data_size))) {
			throw std::runtime_error("PermutoEncImpl::bwd: Expect nonzero `batch_data_size`=" + std::to_string(batch_data_size) + " to be a divisor of `batch_size`=" + std::to_string(batch_size));
		}
	}

	at::Tensor dL_dx, dL_dlattice_val;
	bool need_input_grad = need_input_grad_.value_or(positions.requires_grad());
	bool need_param_grad = need_param_grad_.value_or(lattice_values.requires_grad());
	if ((max_level <= -1) || (!need_input_grad && !need_param_grad)) {
		return {dL_dx, dL_dlattice_val}; 
	}

	if (need_input_grad) {
		dL_dx = at::zeros({ batch_size, meta.n_dims_to_encode }, positions.options()); 
	}

	if (need_param_grad) {
		dL_dlattice_val = at::zeros({lattice_values.size(0)}, lattice_values.options());
	}

	switch (meta.n_dims_to_encode) {
		case 2:  permuto_enc_bwd_impl< 2>(meta, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, max_pos_dims, need_input_grad, need_param_grad, dL_dx, dL_dlattice_val); break;
		case 3:  permuto_enc_bwd_impl< 3>(meta, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, max_pos_dims, need_input_grad, need_param_grad, dL_dx, dL_dlattice_val); break;
		case 4:  permuto_enc_bwd_impl< 4>(meta, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, max_pos_dims, need_input_grad, need_param_grad, dL_dx, dL_dlattice_val); break;
		case 5:  permuto_enc_bwd_impl< 5>(meta, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, max_pos_dims, need_input_grad, need_param_grad, dL_dx, dL_dlattice_val); break;
		case 6:  permuto_enc_bwd_impl< 6>(meta, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, max_pos_dims, need_input_grad, need_param_grad, dL_dx, dL_dlattice_val); break;
		case 7:  permuto_enc_bwd_impl< 7>(meta, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, max_pos_dims, need_input_grad, need_param_grad, dL_dx, dL_dlattice_val); break;
		case 8:  permuto_enc_bwd_impl< 8>(meta, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, max_pos_dims, need_input_grad, need_param_grad, dL_dx, dL_dlattice_val); break;
		case 9:  permuto_enc_bwd_impl< 9>(meta, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, max_pos_dims, need_input_grad, need_param_grad, dL_dx, dL_dlattice_val); break;
		case 10: permuto_enc_bwd_impl<10>(meta, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, max_pos_dims, need_input_grad, need_param_grad, dL_dx, dL_dlattice_val); break;
		case 11: permuto_enc_bwd_impl<11>(meta, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, max_pos_dims, need_input_grad, need_param_grad, dL_dx, dL_dlattice_val); break;
		case 12: permuto_enc_bwd_impl<12>(meta, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, max_pos_dims, need_input_grad, need_param_grad, dL_dx, dL_dlattice_val); break;
		case 13: permuto_enc_bwd_impl<13>(meta, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, max_pos_dims, need_input_grad, need_param_grad, dL_dx, dL_dlattice_val); break;
		case 14: permuto_enc_bwd_impl<14>(meta, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, max_pos_dims, need_input_grad, need_param_grad, dL_dx, dL_dlattice_val); break;
		case 15: permuto_enc_bwd_impl<15>(meta, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, max_pos_dims, need_input_grad, need_param_grad, dL_dx, dL_dlattice_val); break;
		case 16: permuto_enc_bwd_impl<16>(meta, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, max_pos_dims, need_input_grad, need_param_grad, dL_dx, dL_dlattice_val); break;
		case 17: permuto_enc_bwd_impl<17>(meta, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, max_pos_dims, need_input_grad, need_param_grad, dL_dx, dL_dlattice_val); break;
		case 18: permuto_enc_bwd_impl<18>(meta, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, max_pos_dims, need_input_grad, need_param_grad, dL_dx, dL_dlattice_val); break;
		case 19: permuto_enc_bwd_impl<19>(meta, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, max_pos_dims, need_input_grad, need_param_grad, dL_dx, dL_dlattice_val); break;
		case 20: permuto_enc_bwd_impl<20>(meta, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, max_pos_dims, need_input_grad, need_param_grad, dL_dx, dL_dlattice_val); break;
		case 24: permuto_enc_bwd_impl<24>(meta, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, max_pos_dims, need_input_grad, need_param_grad, dL_dx, dL_dlattice_val); break;
		case 28: permuto_enc_bwd_impl<28>(meta, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, max_pos_dims, need_input_grad, need_param_grad, dL_dx, dL_dlattice_val); break;
		case 32: permuto_enc_bwd_impl<32>(meta, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, max_pos_dims, need_input_grad, need_param_grad, dL_dx, dL_dlattice_val); break;
		case 36: permuto_enc_bwd_impl<36>(meta, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, max_pos_dims, need_input_grad, need_param_grad, dL_dx, dL_dlattice_val); break;
		case 40: permuto_enc_bwd_impl<40>(meta, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, max_pos_dims, need_input_grad, need_param_grad, dL_dx, dL_dlattice_val); break;
		case 48: permuto_enc_bwd_impl<48>(meta, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, max_pos_dims, need_input_grad, need_param_grad, dL_dx, dL_dlattice_val); break;
		case 56: permuto_enc_bwd_impl<56>(meta, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, max_pos_dims, need_input_grad, need_param_grad, dL_dx, dL_dlattice_val); break;
		case 64: permuto_enc_bwd_impl<64>(meta, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, max_pos_dims, need_input_grad, need_param_grad, dL_dx, dL_dlattice_val); break;
		default: throw std::runtime_error(std::string("PermutoEncImpl::fwd: `n_dims_to_encode`=") + std::to_string(meta.n_dims_to_encode) + std::string(" is not yet implemented.")); break;
	}

	return {dL_dx, dL_dlattice_val}; 
}

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
	at::optional<bool> need_dL_ddLdy_,
	at::optional<bool> need_dL_dparams_
) {
	at::TensorArg dL_ddLdx_arg(dL_ddLdx, "dL_ddLdx", 1);
	at::TensorArg dL_dy_arg(dL_dy, "dL_dy", 2);
	at::TensorArg positions_arg(positions, "positions", 3); 
	at::TensorArg lattice_values_arg(lattice_values, "lattice_values", 4);

	at::checkDim(__func__, dL_ddLdx_arg, 2);
	at::checkDim(__func__, dL_dy_arg, 2);
	at::checkDim(__func__, positions_arg, 2);
	at::checkDim(__func__, lattice_values_arg, 1);
	at::checkAllSameGPU(__func__, {positions_arg, lattice_values_arg, dL_dy_arg, dL_ddLdx_arg});
	at::checkAllContiguous(__func__, {positions_arg, lattice_values_arg, dL_dy_arg, dL_ddLdx_arg});
	at::checkScalarTypes(__func__, dL_ddLdx_arg, {at::kHalf, at::kFloat});
	at::checkScalarTypes(__func__, dL_dy_arg, {at::kHalf, at::kFloat});
	at::checkScalarTypes(__func__, positions_arg, {at::kHalf, at::kFloat});
	at::checkScalarTypes(__func__, lattice_values_arg, {at::kHalf, at::kFloat});
	at::checkSameType(__func__, dL_dy_arg, lattice_values_arg);
	at::checkSameType(__func__, dL_ddLdx_arg, positions_arg);

	const uint32_t batch_size = positions.size(0);
	at::checkSize(__func__, dL_ddLdx_arg, {batch_size, meta.n_dims_to_encode});
	at::checkSize(__func__, dL_dy_arg, {batch_size, meta.n_encoded_dims});
	at::checkSize(__func__, positions_arg, 1, meta.n_dims_to_encode);
	const uint32_t num_params = lattice_values.size(0);
	if (!is_divisible(num_params, meta.n_params)) {
		throw std::runtime_error(std::string{"PermutoEncImpl::bwdbwd: Expect size of `params`="} + std::to_string(num_params) + std::string{" to be an integral multiple of `n_param`="} + std::to_string(meta.n_params));
	}

	const int32_t max_level = max_level_.value_or(meta.n_levels);

	at::Tensor level_random_shifts;
	at::TensorArg level_random_shifts_arg(level_random_shifts, "level_random_shifts", 5);
	if (level_random_shifts_.has_value()) {
		level_random_shifts = level_random_shifts_.value();
		at::checkSize(__func__, level_random_shifts_arg, {meta.n_levels, meta.n_dims_to_encode});
		at::checkSameGPU(__func__, positions_arg, level_random_shifts_arg);
		at::checkSameType(__func__, positions_arg, level_random_shifts_arg);
		at::checkContiguous(__func__, level_random_shifts_arg);
	}

	at::Tensor batch_inds;
	at::TensorArg batch_inds_arg(batch_inds, "batch_inds", 6);
	if (batch_inds_.has_value()) {
		batch_inds = batch_inds_.value();
		at::checkDim(__func__, batch_inds_arg, 1);
		at::checkSameGPU(__func__, positions_arg, batch_inds_arg);
		at::checkContiguous(__func__, batch_inds_arg);
		at::checkScalarType(__func__, batch_inds_arg, at::kLong);
		at::checkSize(__func__, batch_inds_arg, {batch_size});
	}

	at::Tensor batch_offset;
	if (batch_offsets_.has_value()) {
		batch_offset = batch_offsets_.value();
		at::TensorArg batch_offset_arg(batch_offset, "batch_offset", 7);
		at::checkDim(__func__, batch_offset_arg, 1);
		at::checkSameGPU(__func__, positions_arg, batch_offset_arg);
		at::checkContiguous(__func__, batch_offset_arg);
		at::checkScalarType(__func__, batch_offset_arg, at::kLong);
	}

	uint32_t batch_data_size = 0;
	if (batch_data_size_.has_value()) {
		batch_data_size = batch_data_size_.value();
		if (! (batch_data_size == 0 || is_divisible(batch_size, batch_data_size))) {
			throw std::runtime_error("PermutoEncImpl::bwdbwd: Expect nonzero `batch_data_size`=" + std::to_string(batch_data_size) + " to be a divisor of `batch_size`=" + std::to_string(batch_size));
		}
	}

	at::Tensor dL_ddLdy, dL_dlattice_val;
	bool need_dL_ddLdy = need_dL_ddLdy_.value_or(dL_dy.requires_grad());
	bool need_dL_dparams = need_dL_dparams_.value_or(lattice_values.requires_grad());
	if ((max_level <= -1) || (!need_dL_ddLdy && !need_dL_dparams)) {
		return {dL_ddLdy, dL_dlattice_val};
	}

	if (need_dL_ddLdy) {
		dL_ddLdy = at::zeros({ batch_size, meta.n_encoded_dims }, dL_dy.options());
	}

	if (need_dL_dparams) {
		dL_dlattice_val = at::zeros({ lattice_values.size(0) }, lattice_values.options()); 
	}

	switch (meta.n_dims_to_encode) {
		case  2: permuto_enc_bwd_bwd_input_impl< 2>(meta, dL_ddLdx, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, need_dL_ddLdy, need_dL_dparams, dL_ddLdy, dL_dlattice_val); break;
		case  3: permuto_enc_bwd_bwd_input_impl< 3>(meta, dL_ddLdx, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, need_dL_ddLdy, need_dL_dparams, dL_ddLdy, dL_dlattice_val); break;
		case  4: permuto_enc_bwd_bwd_input_impl< 4>(meta, dL_ddLdx, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, need_dL_ddLdy, need_dL_dparams, dL_ddLdy, dL_dlattice_val); break;
		case  5: permuto_enc_bwd_bwd_input_impl< 5>(meta, dL_ddLdx, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, need_dL_ddLdy, need_dL_dparams, dL_ddLdy, dL_dlattice_val); break;
		case  6: permuto_enc_bwd_bwd_input_impl< 6>(meta, dL_ddLdx, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, need_dL_ddLdy, need_dL_dparams, dL_ddLdy, dL_dlattice_val); break;
		case  7: permuto_enc_bwd_bwd_input_impl< 7>(meta, dL_ddLdx, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, need_dL_ddLdy, need_dL_dparams, dL_ddLdy, dL_dlattice_val); break;
		case  8: permuto_enc_bwd_bwd_input_impl< 8>(meta, dL_ddLdx, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, need_dL_ddLdy, need_dL_dparams, dL_ddLdy, dL_dlattice_val); break;
		case  9: permuto_enc_bwd_bwd_input_impl< 9>(meta, dL_ddLdx, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, need_dL_ddLdy, need_dL_dparams, dL_ddLdy, dL_dlattice_val); break;
		case 10: permuto_enc_bwd_bwd_input_impl<10>(meta, dL_ddLdx, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, need_dL_ddLdy, need_dL_dparams, dL_ddLdy, dL_dlattice_val); break;
		case 11: permuto_enc_bwd_bwd_input_impl<11>(meta, dL_ddLdx, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, need_dL_ddLdy, need_dL_dparams, dL_ddLdy, dL_dlattice_val); break;
		case 12: permuto_enc_bwd_bwd_input_impl<12>(meta, dL_ddLdx, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, need_dL_ddLdy, need_dL_dparams, dL_ddLdy, dL_dlattice_val); break;
		case 13: permuto_enc_bwd_bwd_input_impl<13>(meta, dL_ddLdx, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, need_dL_ddLdy, need_dL_dparams, dL_ddLdy, dL_dlattice_val); break;
		case 14: permuto_enc_bwd_bwd_input_impl<14>(meta, dL_ddLdx, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, need_dL_ddLdy, need_dL_dparams, dL_ddLdy, dL_dlattice_val); break;
		case 15: permuto_enc_bwd_bwd_input_impl<15>(meta, dL_ddLdx, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, need_dL_ddLdy, need_dL_dparams, dL_ddLdy, dL_dlattice_val); break;
		case 16: permuto_enc_bwd_bwd_input_impl<16>(meta, dL_ddLdx, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, need_dL_ddLdy, need_dL_dparams, dL_ddLdy, dL_dlattice_val); break;
		case 17: permuto_enc_bwd_bwd_input_impl<17>(meta, dL_ddLdx, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, need_dL_ddLdy, need_dL_dparams, dL_ddLdy, dL_dlattice_val); break;
		case 18: permuto_enc_bwd_bwd_input_impl<18>(meta, dL_ddLdx, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, need_dL_ddLdy, need_dL_dparams, dL_ddLdy, dL_dlattice_val); break;
		case 19: permuto_enc_bwd_bwd_input_impl<19>(meta, dL_ddLdx, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, need_dL_ddLdy, need_dL_dparams, dL_ddLdy, dL_dlattice_val); break;
		case 20: permuto_enc_bwd_bwd_input_impl<20>(meta, dL_ddLdx, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, need_dL_ddLdy, need_dL_dparams, dL_ddLdy, dL_dlattice_val); break;
		case 24: permuto_enc_bwd_bwd_input_impl<24>(meta, dL_ddLdx, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, need_dL_ddLdy, need_dL_dparams, dL_ddLdy, dL_dlattice_val); break;
		case 28: permuto_enc_bwd_bwd_input_impl<28>(meta, dL_ddLdx, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, need_dL_ddLdy, need_dL_dparams, dL_ddLdy, dL_dlattice_val); break;
		case 32: permuto_enc_bwd_bwd_input_impl<32>(meta, dL_ddLdx, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, need_dL_ddLdy, need_dL_dparams, dL_ddLdy, dL_dlattice_val); break;
		case 36: permuto_enc_bwd_bwd_input_impl<36>(meta, dL_ddLdx, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, need_dL_ddLdy, need_dL_dparams, dL_ddLdy, dL_dlattice_val); break;
		case 40: permuto_enc_bwd_bwd_input_impl<40>(meta, dL_ddLdx, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, need_dL_ddLdy, need_dL_dparams, dL_ddLdy, dL_dlattice_val); break;
		case 48: permuto_enc_bwd_bwd_input_impl<48>(meta, dL_ddLdx, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, need_dL_ddLdy, need_dL_dparams, dL_ddLdy, dL_dlattice_val); break;
		case 56: permuto_enc_bwd_bwd_input_impl<56>(meta, dL_ddLdx, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, need_dL_ddLdy, need_dL_dparams, dL_ddLdy, dL_dlattice_val); break;
		case 64: permuto_enc_bwd_bwd_input_impl<64>(meta, dL_ddLdx, dL_dy, positions, lattice_values, level_random_shifts_, batch_inds_, batch_offsets_, batch_data_size, max_level, need_dL_ddLdy, need_dL_dparams, dL_ddLdy, dL_dlattice_val); break;
		default: throw std::runtime_error(std::string("PermutoEncImpl::fwd: `n_dims_to_encode`=") + std::to_string(meta.n_dims_to_encode) + std::string(" is not yet implemented.")); break;
	}

	return {dL_ddLdy, dL_dlattice_val}; 
}

}
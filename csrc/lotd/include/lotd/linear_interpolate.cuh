#include "lotd_cuda.h"
#include "if_constexpr.hpp" // pre-c++17 constexpr if

using namespace lotd;

namespace lotd {
namespace torch {

template <typename INPUT_T, typename PARAM_T, typename COMPUTE_T, uint32_t N_POS_DIMS, uint32_t N_FEATS, bool WITH_GRAD, typename F>
__device__ __forceinline__ void
linear_interpolate( // Without pre-fetched `grid_values`
	F get_grid_val, 
	const COMPUTE_T scale[N_POS_DIMS], 
	const COMPUTE_T pos[N_POS_DIMS],
	const COMPUTE_T pos_derivative[N_POS_DIMS], 
	const uint32_t pos_grid[N_POS_DIMS],
	const uint32_t grid_feat_offset, 
	const uint32_t grid_resolution[N_POS_DIMS],
	const uint32_t grid_size, 
	const uint32_t n_feat_cur_lvl,
	const PARAM_T *__restrict__ grid, 
	PARAM_T *__restrict__ result_ptr,
	vector_t<INPUT_T, N_POS_DIMS> *__restrict__ grads_ptr
) {

	auto grid_val = [&](const uint32_t local_pos[N_POS_DIMS]) {
		vector_t<PARAM_T, N_FEATS> val = {};
		get_grid_val(local_pos, grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, grid, (PARAM_T *)&val);
		return val;
	};

	#pragma unroll
	for (uint32_t idx = 0; idx < (1 << N_POS_DIMS); ++idx) {
		COMPUTE_T weight = 1;
		uint32_t pos_grid_local[N_POS_DIMS];

		#pragma unroll
		for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
			if ((idx & (1 << dim)) == 0) {
				weight *= (COMPUTE_T)1 - pos[dim];
				pos_grid_local[dim] = pos_grid[dim];
			} else {
				weight *= pos[dim];
				pos_grid_local[dim] = pos_grid[dim] + 1;
			}
		}

		auto val = grid_val(pos_grid_local);

		#pragma unroll
		for (uint32_t f = 0; f < N_FEATS; ++f) {
			result_ptr[f] += (PARAM_T)(weight * (COMPUTE_T)((PARAM_T *)&val)[f]);
		}
	}

	if (WITH_GRAD) {
		#pragma unroll 1
		for (uint32_t grad_dim = 0; grad_dim < N_POS_DIMS; ++grad_dim) {
			#pragma unroll
			for (uint32_t idx = 0; idx < (1 << (N_POS_DIMS - 1)); ++idx) {

				COMPUTE_T weight = scale[grad_dim] * pos_derivative[grad_dim];
				uint32_t pos_grid_local[N_POS_DIMS];

				#pragma unroll
				for (uint32_t non_grad_dim = 0; non_grad_dim < N_POS_DIMS - 1; ++non_grad_dim) {
					const uint32_t dim = non_grad_dim >= grad_dim ? (non_grad_dim + 1) : non_grad_dim;

					if ((idx & 1 << non_grad_dim) == 0) {
						weight *= (COMPUTE_T)1 - pos[dim];
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
				for (uint32_t f = 0; f < N_FEATS; ++f) {
					grads_ptr[f][grad_dim] += weight * ((COMPUTE_T)val_right[f] - (COMPUTE_T)val_left[f]);
				}
			}
		}
	}
}

template <typename INPUT_T, typename PARAM_T, typename COMPUTE_T, uint32_t N_POS_DIMS, uint32_t N_FEATS, bool WITH_GRAD>
__device__ __forceinline__ void
linear_interpolate( // With pre-fetched `grid_values`
	const COMPUTE_T scale[N_POS_DIMS], 
	const COMPUTE_T pos[N_POS_DIMS],
	const COMPUTE_T pos_derivative[N_POS_DIMS],
	const vector_t<PARAM_T, N_FEATS> *__restrict__ grid_values, // Pre-fetched grid values
	PARAM_T *__restrict__ result_ptr,
	vector_t<INPUT_T, N_POS_DIMS> *__restrict__ grads_ptr
) {
	#pragma unroll
	for (uint32_t idx = 0; idx < (1 << N_POS_DIMS); ++idx) {
		COMPUTE_T weight = 1;

		#pragma unroll
		for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
			if ((idx & (1 << dim)) == 0)
				weight *= (COMPUTE_T)1 - pos[dim];
			else
				weight *= pos[dim];
		}

		auto val = grid_values[idx];

		#pragma unroll
		for (uint32_t f = 0; f < N_FEATS; ++f) {
			result_ptr[f] += (PARAM_T)(weight * (COMPUTE_T)((PARAM_T *)&val)[f]);
		}
	}

	if (WITH_GRAD) {
		#pragma unroll
		for (uint32_t grad_dim = 0; grad_dim < N_POS_DIMS; ++grad_dim) {
			#pragma unroll
			for (uint32_t idx = 0; idx < (1 << (N_POS_DIMS - 1)); ++idx) {
				COMPUTE_T weight = scale[grad_dim] * pos_derivative[grad_dim];
				uint32_t left_idx = 0;
				#pragma unroll
				for (uint32_t non_grad_dim = 0; non_grad_dim < N_POS_DIMS - 1; ++non_grad_dim) {
					const uint32_t dim = non_grad_dim >= grad_dim ? (non_grad_dim + 1) : non_grad_dim;
					if ((idx & 1 << non_grad_dim) == 0) {
						weight *= (COMPUTE_T)1 - pos[dim];
					} else {
						weight *= pos[dim];
						left_idx += 1 << dim;
					}
				}
				uint32_t right_idx = left_idx + (1 << grad_dim);
				auto val_left = grid_values[left_idx];
				auto val_right = grid_values[right_idx];

				#pragma unroll
				for (uint32_t f = 0; f < N_FEATS; ++f) {
					grads_ptr[f][grad_dim] += weight * ((COMPUTE_T)val_right[f] - (COMPUTE_T)val_left[f]);
				}
			}
		}
	}
}

template <typename PARAM_T, typename GRAD_T, typename COMPUTE_T, uint32_t N_POS_DIMS, uint32_t N_FEAT, typename F>
__device__ __forceinline__ void linear_interpolate_backward_grid(
	F add_grid_grad, 
	const COMPUTE_T pos[N_POS_DIMS], 
	const uint32_t pos_grid[N_POS_DIMS],
	const uint32_t grid_feat_offset, 
	const uint32_t grid_resolution[N_POS_DIMS],
	const uint32_t grid_size, 
	const uint32_t n_feat_cur_lvl, 
	const PARAM_T *__restrict__ grid,
	const vector_t<PARAM_T, N_FEAT> &grad, 
	GRAD_T *__restrict__ grid_gradient
) {
	auto add_grid_gradient = [&](const uint32_t local_pos[N_POS_DIMS], const vector_t<PARAM_T, N_FEAT> &grad, const COMPUTE_T weight) {
		add_grid_grad(local_pos, grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, grid, (PARAM_T *)&grad, weight, grid_gradient);
	};
	#pragma unroll 1
	for (uint32_t idx = 0; idx < (1 << N_POS_DIMS); ++idx) {
		COMPUTE_T weight = 1;
		uint32_t pos_grid_local[N_POS_DIMS];

		#pragma unroll
		for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
			if ((idx & (1 << dim)) == 0) {
				weight *= (COMPUTE_T)1 - pos[dim];
				pos_grid_local[dim] = pos_grid[dim];
			} else {
				weight *= pos[dim];
				pos_grid_local[dim] = pos_grid[dim] + 1;
			}
		}

		add_grid_gradient(pos_grid_local, grad, weight);
	}
}

template <typename INPUT_T, typename PARAM_T, typename GRAD_T, typename COMPUTE_T, uint32_t N_POS_DIMS, uint32_t N_FEAT, typename F>
__device__ __forceinline__ void linear_interpolate_backward_input_backward_grid(
	F add_grid_grad, 
	const COMPUTE_T scale[N_POS_DIMS], 
	const COMPUTE_T pos[N_POS_DIMS],
	const COMPUTE_T pos_derivative[N_POS_DIMS], 
	const uint32_t pos_grid[N_POS_DIMS],
	const uint32_t grid_feat_offset, 
	const uint32_t grid_resolution[N_POS_DIMS],
	const uint32_t grid_size, 
	const uint32_t n_feat_cur_lvl, 
	const PARAM_T *__restrict__ grid,
	const vector_t<PARAM_T, N_FEAT> &grad, 
	const vector_t<INPUT_T, N_POS_DIMS> &grad_input,
	GRAD_T *__restrict__ grid_gradient
) {
	auto add_grid_gradient = [&](const uint32_t local_pos[N_POS_DIMS], const vector_t<PARAM_T, N_FEAT> &grad, const COMPUTE_T weight) {
		add_grid_grad(local_pos, grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, grid, (PARAM_T *)&grad, weight, grid_gradient);
	};

	#pragma unroll 1 // Force skip unrolling; significantly reduce kernel code size & actually increase speed;
	for (uint32_t grad_dim = 0; grad_dim < N_POS_DIMS; ++grad_dim) {
		COMPUTE_T grad_in = scale[grad_dim] * (COMPUTE_T)grad_input[grad_dim] * pos_derivative[grad_dim];
		#pragma unroll
		for (uint32_t idx = 0; idx < (1 << (N_POS_DIMS - 1)); ++idx) {
			COMPUTE_T weight = grad_in;
			uint32_t pos_grid_local[N_POS_DIMS];

			#pragma unroll
			for (uint32_t non_grad_dim = 0; non_grad_dim < N_POS_DIMS - 1; ++non_grad_dim) {
				const uint32_t dim = non_grad_dim >= grad_dim ? (non_grad_dim + 1) : non_grad_dim;

				if ((idx & 1 << non_grad_dim) == 0) {
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

template <typename INPUT_T, typename PARAM_T, typename COMPUTE_T, uint32_t N_POS_DIMS, uint32_t N_FEAT, typename F>
__device__ __forceinline__ void linear_interpolate_backward_input_backward_input(
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
	const PARAM_T *__restrict__ grid,
	const vector_t<PARAM_T, N_FEAT> &grad, 
	const vector_t<INPUT_T, N_POS_DIMS> &grad_input,
	INPUT_T *__restrict__ grad_result
) {
	auto calc_dLdx = [&](const uint32_t local_pos[N_POS_DIMS], const COMPUTE_T weight) {
		return calc_dLdx_dim(local_pos, grid_feat_offset, grid_resolution, grid_size, n_feat_cur_lvl, grid, (PARAM_T *)&grad, weight);
	};

	vector_t<COMPUTE_T, N_POS_DIMS> grad_in_diag;
	vector_t<COMPUTE_T, N_POS_DIMS> grad_in_other;

	// From diagonal part of Hessian
	// NOTE: LinearInterpolations' diagonal part is 0.
	if (interpolation_type == InterpolationType::Smoothstep) {
		#pragma unroll
		for (uint32_t grad_dim = 0; grad_dim < N_POS_DIMS; ++grad_dim) {
			grad_in_diag[grad_dim] = (scale[grad_dim] * (COMPUTE_T)grad_input[grad_dim]) *
									 (scale[grad_dim] * pos_2nd_derivative[grad_dim]);
		}
	}

	// From other part of Hessian
	#pragma unroll
	for (uint32_t grad_dim = 0; grad_dim < N_POS_DIMS; ++grad_dim) {
		grad_in_other[grad_dim] =
			scale[grad_dim] * (COMPUTE_T)grad_input[grad_dim] * pos_derivative[grad_dim]; // * (pos_derivative[other_grad_dim] * scale[other_grad_dim]);
	}

	#pragma unroll 1 // Force skip unrolling; significantly reduce kernel code size & actually increase speed;
	for (uint32_t grad_dim = 0; grad_dim < N_POS_DIMS; ++grad_dim) {
		COMPUTE_T grad_out = 0;
		#pragma unroll
		for (uint32_t idx = 0; idx < (1 << (N_POS_DIMS - 1)); ++idx) {
			// From diagonal part of Hessian; d(doutput_d[grad_dim])_d[grad_dim]
			// NOTE: LinearInterpolations' diagonal part is 0.
			if (interpolation_type == InterpolationType::Smoothstep) {
				COMPUTE_T weight_2nd_diag = grad_in_diag[grad_dim];
				uint32_t pos_grid_local[N_POS_DIMS];

				#pragma unroll
				for (uint32_t non_grad_dim = 0; non_grad_dim < N_POS_DIMS - 1; ++non_grad_dim) {
					const uint32_t dim = non_grad_dim >= grad_dim ? (non_grad_dim + 1) : non_grad_dim;
					// real non_grad_dim
					if ((idx & 1 << non_grad_dim) == 0) {
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
			ic::if_<(N_POS_DIMS > 1)>(
				[&] {
					#pragma unroll
					for (uint32_t other_grad_dim = 0; other_grad_dim < N_POS_DIMS - 1; ++other_grad_dim) {
						const uint32_t real_other_grad_dim = other_grad_dim >= grad_dim ? (other_grad_dim + 1) : other_grad_dim;
						COMPUTE_T weight_2nd_other = grad_in_other[real_other_grad_dim] * (pos_derivative[grad_dim] * scale[grad_dim]);
						uint32_t pos_grid_local[N_POS_DIMS];

						#pragma unroll
						for (uint32_t non_grad_dim = 0; non_grad_dim < N_POS_DIMS - 1; ++non_grad_dim) {
							// real non_grad_dim
							const uint32_t dim = non_grad_dim >= real_other_grad_dim ? (non_grad_dim + 1) : non_grad_dim;
							if ((idx & 1 << non_grad_dim) == 0) {
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
				},
				[] {});
		}

		grad_result[grad_dim] = (INPUT_T)grad_out;
	}
}

} // namespace torch
} // namespace lotd
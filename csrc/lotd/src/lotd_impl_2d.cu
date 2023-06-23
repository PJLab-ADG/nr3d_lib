/** @file   lotd_impl_2d.cu
 *  @author Jianfei Guo, Shanghai AI Lab
 *  @brief  This file is for parallel compilation acceleration only.
 */

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

#include <lotd/lotd.h>
#include <lotd/lotd_torch_api.h>
#include <lotd/lotd_encoding.h>

namespace lotd {
namespace torch {

template void lod_fwd_impl<2>(
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

template void lod_bwd_impl<2>(
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

template void lod_bwd_bwd_input_impl<2>(
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
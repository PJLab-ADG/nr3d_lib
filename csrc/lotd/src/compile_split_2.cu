/** @file   compile_split_2.cu
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

#include <lotd/lotd_cuda.h>
#include <lotd/lotd_torch_api.h>
#include <lotd/lotd_encoding.h>

namespace lotd {
namespace torch {

template void lod_fwd_impl<3>(LoDMeta&,at::Tensor,at::Tensor,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t, int32_t, bool, at::Tensor, at::Tensor);
template void lod_bwd_impl<3>(LoDMeta&,at::Tensor,at::Tensor,at::Tensor,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t, bool,bool, at::Tensor,at::Tensor);
template void lod_bwd_bwd_input_impl<3>(LoDMeta&,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,bool,bool,bool,at::Tensor,at::Tensor,at::Tensor);

} // namespace lotd::torch

} // namespace lotd
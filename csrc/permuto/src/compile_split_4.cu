
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

#include <permuto/permuto.h>
#include <permuto/permuto_cuda.h>

namespace permuto {

// Explicit Template Instantiation
template void permuto_enc_bwd_impl<24>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,uint32_t,bool,bool,at::Tensor&,at::Tensor&); 
template void permuto_enc_bwd_impl<28>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,uint32_t,bool,bool,at::Tensor&,at::Tensor&); 
template void permuto_enc_bwd_impl<32>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,uint32_t,bool,bool,at::Tensor&,at::Tensor&); 
template void permuto_enc_bwd_impl<36>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,uint32_t,bool,bool,at::Tensor&,at::Tensor&); 
template void permuto_enc_bwd_impl<40>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,uint32_t,bool,bool,at::Tensor&,at::Tensor&); 
template void permuto_enc_bwd_impl<48>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,uint32_t,bool,bool,at::Tensor&,at::Tensor&); 
template void permuto_enc_bwd_impl<56>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,uint32_t,bool,bool,at::Tensor&,at::Tensor&); 
template void permuto_enc_bwd_impl<64>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,uint32_t,bool,bool,at::Tensor&,at::Tensor&); 

}
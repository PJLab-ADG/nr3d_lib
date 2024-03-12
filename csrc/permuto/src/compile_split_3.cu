
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
template void permuto_enc_bwd_impl< 2>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,uint32_t,bool,bool,at::Tensor&,at::Tensor&); 
template void permuto_enc_bwd_impl< 3>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,uint32_t,bool,bool,at::Tensor&,at::Tensor&); 
template void permuto_enc_bwd_impl< 4>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,uint32_t,bool,bool,at::Tensor&,at::Tensor&); 
template void permuto_enc_bwd_impl< 5>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,uint32_t,bool,bool,at::Tensor&,at::Tensor&); 
template void permuto_enc_bwd_impl< 6>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,uint32_t,bool,bool,at::Tensor&,at::Tensor&); 
template void permuto_enc_bwd_impl< 7>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,uint32_t,bool,bool,at::Tensor&,at::Tensor&); 
template void permuto_enc_bwd_impl< 8>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,uint32_t,bool,bool,at::Tensor&,at::Tensor&); 
template void permuto_enc_bwd_impl< 9>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,uint32_t,bool,bool,at::Tensor&,at::Tensor&); 
template void permuto_enc_bwd_impl<10>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,uint32_t,bool,bool,at::Tensor&,at::Tensor&); 
template void permuto_enc_bwd_impl<11>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,uint32_t,bool,bool,at::Tensor&,at::Tensor&); 
template void permuto_enc_bwd_impl<12>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,uint32_t,bool,bool,at::Tensor&,at::Tensor&); 
template void permuto_enc_bwd_impl<13>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,uint32_t,bool,bool,at::Tensor&,at::Tensor&); 
template void permuto_enc_bwd_impl<14>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,uint32_t,bool,bool,at::Tensor&,at::Tensor&); 
template void permuto_enc_bwd_impl<15>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,uint32_t,bool,bool,at::Tensor&,at::Tensor&); 
template void permuto_enc_bwd_impl<16>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,uint32_t,bool,bool,at::Tensor&,at::Tensor&); 
template void permuto_enc_bwd_impl<17>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,uint32_t,bool,bool,at::Tensor&,at::Tensor&); 
template void permuto_enc_bwd_impl<18>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,uint32_t,bool,bool,at::Tensor&,at::Tensor&); 
template void permuto_enc_bwd_impl<19>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,uint32_t,bool,bool,at::Tensor&,at::Tensor&); 
template void permuto_enc_bwd_impl<20>(PermutoEncMeta,at::Tensor&,at::Tensor&,at::Tensor&,at::optional<at::Tensor>,at::optional<at::Tensor>,at::optional<at::Tensor>,uint32_t,int32_t,uint32_t,bool,bool,at::Tensor&,at::Tensor&); 

}
/** @file   pack_ops_cuda.cu
 *  @author Jianfei Guo, Shanghai AI Lab
 *  @brief  pack_ops: Pytorch CUDA implementation for pack-wise operations.
 */

#include <stdint.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>

#include <cmath>
#include <algorithm>
#include <stdexcept>

#include <cstdio>

#include "pack_ops.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")

/// Checks the result of a cudaXXXXXX call and throws an error on failure
#define STRINGIFY(x) #x
#define STR(x) STRINGIFY(x)
#define FILE_LINE __FILE__ ":" STR(__LINE__)
#define CUDA_CHECK_THROW(x)                                                                                               \
	do {                                                                                                                  \
		cudaError_t result = x;                                                                                           \
		if (result != cudaSuccess)                                                                                        \
			throw std::runtime_error(std::string(FILE_LINE " " #x " failed with error ") + cudaGetErrorString(result));  \
	} while(0)

template <typename scalar_t>
inline __host__ __device__ scalar_t div_round_up(scalar_t val, scalar_t divisor) {
    return (val + divisor - 1) / divisor;
}

template <typename scalar_t>
__device__ scalar_t clamp(scalar_t val, scalar_t lower, scalar_t upper) {
	return val < lower ? lower : (upper < val ? upper : val);
}

template <typename scalar_t>
__global__ void kernel_interleave_arange(
    // Inputs
    const uint32_t num_packs,
    const int64_t* __restrict__ num_steps,
    const int64_t* __restrict__ num_steps_cumsum,
    const scalar_t* __restrict__ starts,
    const scalar_t* __restrict__ step_sizes,
    scalar_t start,
    scalar_t step_size,
    // Outputs
    scalar_t* out,
    int64_t* nidx=nullptr
) {
    const uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidx >= num_packs) return;

    uint32_t begin = tidx > 0 ? num_steps_cumsum[tidx-1] : 0;
    uint32_t num_step = num_steps[tidx];

    if (starts) start = starts[tidx];
    if (step_sizes) step_size = step_sizes[tidx];

    if (nidx) {
        out += begin;
        nidx += begin;
        for (uint32_t j=0; j<num_step; ++j) {
            out[j] = start + (scalar_t)j * step_size;
            nidx[j] = tidx;
        }
    } else {
        out += begin;
        for (uint32_t j=0; j<num_step; ++j) {
            out[j] = start + (scalar_t)j * step_size;
        }
    }
}

std::tuple<at::Tensor, at::Tensor> interleave_arange(
    at::Tensor stop,
    bool return_idx
) {
    at::TensorArg stop_arg{stop, "stop", 1};
    at::checkDim(__func__, stop_arg, 1);
    at::checkContiguous(__func__, stop_arg);
    at::checkScalarType(__func__, stop_arg, at::kLong);
    at::checkDeviceType(__func__, {stop}, at::kCUDA);

    uint32_t num_packs = stop.size(0);
    at::Tensor num_steps_cumsum = stop.cumsum(0);
    uint32_t num = num_steps_cumsum[-1].item<int64_t>();

    at::Tensor out = at::empty({ num }, stop.options());
    at::Tensor nidx;
    if (return_idx) {
        nidx = at::empty({ num }, stop.options());
    }
    
    static constexpr uint32_t num_threads = 128;

    const at::cuda::OptionalCUDAGuard device_guard(at::device_of(stop));
    auto stream = at::cuda::getCurrentCUDAStream();
    kernel_interleave_arange<int64_t><<<div_round_up(num_packs, num_threads), num_threads, 0, stream>>>(
        num_packs, stop.data_ptr<int64_t>(), num_steps_cumsum.data_ptr<int64_t>(), 
        nullptr, nullptr, 0, 1, out.data_ptr<int64_t>(),
        return_idx ? nidx.data_ptr<int64_t>() : nullptr
    );
    return {out, nidx};
}

std::tuple<at::Tensor, at::Tensor> interleave_linstep_impl(
    at::Tensor start, at::Tensor num_steps, at::Tensor step_size, bool return_idx
) {
    at::Tensor num_steps_cumsum = num_steps.cumsum(0);
    uint32_t num_packs = start.size(0);
    uint32_t num = num_steps_cumsum[-1].item<int64_t>();

    at::Tensor out = at::empty({ num }, start.options());
    at::Tensor nidx;
    if (return_idx) {
        nidx = at::empty({ num }, num_steps.options());
    }

    static constexpr uint32_t num_threads = 128;
    
    AT_DISPATCH_ALL_TYPES_AND_HALF(start.scalar_type(), "interleave_arange_tensor", ([&] {
        const at::cuda::OptionalCUDAGuard device_guard(at::device_of(start));
        auto stream = at::cuda::getCurrentCUDAStream();
        kernel_interleave_arange<scalar_t><<<div_round_up(num_packs, num_threads), num_threads, 0, stream>>>(
            num_packs, num_steps.data_ptr<int64_t>(), num_steps_cumsum.data_ptr<int64_t>(), 
            start.data_ptr<scalar_t>(), step_size.data_ptr<scalar_t>(), 0, 0, out.data_ptr<scalar_t>(),
            return_idx ? nidx.data_ptr<int64_t>() : nullptr
        );
    }));

    return {out, nidx};
};

std::tuple<at::Tensor, at::Tensor> interleave_linstep_impl(
    at::Tensor start, at::Tensor num_steps, at::Scalar step_size, bool return_idx
) {
    at::Tensor num_steps_cumsum = num_steps.cumsum(0);
    uint32_t num_packs = start.size(0);
    uint32_t num = num_steps_cumsum[-1].item<int64_t>();

    at::Tensor out = at::empty({ num }, start.options());
    at::Tensor nidx;
    if (return_idx) {
        nidx = at::empty({ num }, num_steps.options());
    }

    static constexpr uint32_t num_threads = 128;
    
    AT_DISPATCH_ALL_TYPES_AND_HALF(start.scalar_type(), "interleave_arange_scalar", ([&] {
        const at::cuda::OptionalCUDAGuard device_guard(at::device_of(start));
        auto stream = at::cuda::getCurrentCUDAStream();
        kernel_interleave_arange<scalar_t><<<div_round_up(num_packs, num_threads), num_threads, 0, stream>>>(
            num_packs, num_steps.data_ptr<int64_t>(), num_steps_cumsum.data_ptr<int64_t>(), 
            start.data_ptr<scalar_t>(), nullptr, 0, step_size.to<scalar_t>(), out.data_ptr<scalar_t>(),
            return_idx ? nidx.data_ptr<int64_t>() : nullptr
        );
    }));

    return {out, nidx};
};

std::tuple<at::Tensor, at::Tensor> interleave_linstep(
    at::Tensor start, at::Tensor num_steps, at::Tensor step_size,
    bool return_idx
) {
    at::TensorArg start_arg{start, "start", 1};
    at::TensorArg num_steps_arg{num_steps, "num_steps", 2};
    at::TensorArg step_size_arg{step_size, "step_size", 3};

    at::checkDim(__func__, start_arg, 1);
    at::checkDim(__func__, num_steps_arg, 1);
    at::checkDim(__func__, step_size_arg, 1);
    at::checkSameSize(__func__, start_arg, num_steps_arg);
    at::checkAllContiguous(__func__, {start_arg, num_steps_arg, step_size_arg});
    at::checkAllSameGPU(__func__, {start_arg, num_steps_arg, step_size_arg});
    at::checkSameType(__func__, start_arg, step_size_arg);
    at::checkScalarType(__func__, num_steps_arg, at::kLong);
    return interleave_linstep_impl(start, num_steps, step_size, return_idx);
}

std::tuple<at::Tensor, at::Tensor> interleave_linstep(
    at::Tensor start, at::Tensor num_steps, double step_size,
    bool return_idx
) {
    at::TensorArg start_arg{start, "start", 1};
    at::TensorArg num_steps_arg{num_steps, "num_steps", 2};
    at::checkDim(__func__, start_arg, 1);
    at::checkDim(__func__, num_steps_arg, 1);
    at::checkSameSize(__func__, start_arg, num_steps_arg);
    at::checkSameGPU(__func__, start_arg, num_steps_arg);
    at::checkScalarType(__func__, num_steps_arg, at::kLong);
    return interleave_linstep_impl(start, num_steps, step_size, return_idx);
}

std::tuple<at::Tensor, at::Tensor> interleave_linstep(
    at::Tensor start, at::Tensor num_steps, int32_t step_size,
    bool return_idx
) {
    at::TensorArg start_arg{start, "start", 1};
    at::TensorArg num_steps_arg{num_steps, "num_steps", 2};
    at::checkDim(__func__, start_arg, 1);
    at::checkDim(__func__, num_steps_arg, 1);
    at::checkSameSize(__func__, start_arg, num_steps_arg);
    at::checkSameGPU(__func__, start_arg, num_steps_arg);
    at::checkScalarType(__func__, num_steps_arg, at::kLong);
    return interleave_linstep_impl(start, num_steps, step_size, return_idx);
}

/*
Buffered & sliced version;
- requires two sets of tensor storage
- slower (because of addtional compactifing)
- pack order preserved
*/
template <typename scalar_t>
__global__ void kernel_interleave_sample_step_wrt_depth_clamp_v1_round1(
    // Inputs
    const uint32_t num_packs,
    const uint32_t max_steps,
    const scalar_t dt_gamma,
    const scalar_t min_step_size,
    const scalar_t max_step_size,
    const scalar_t* __restrict__ nears,
    const scalar_t* __restrict__ fars,
    // Outputs
    scalar_t* __restrict__ t_samples,
    scalar_t* __restrict__ deltas,
    int32_t* __restrict__ nidx,
    int64_t* __restrict__ n_per_pack
) {
    const uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidx >= num_packs) return;

    const scalar_t near = nears[tidx], far = fars[tidx];

    uint32_t begin = tidx * max_steps;

    t_samples += begin;
    nidx += begin;
    deltas += begin;

    // Init
    scalar_t t = near;
    uint32_t step = 0;

    while (t <= far && step < max_steps) {
        t_samples[step] = t;
        nidx[step] = tidx;
        scalar_t dt = clamp(t * dt_gamma, min_step_size, max_step_size);
        deltas[step] = dt;
        t += dt;
        step++;
    }
    n_per_pack[tidx] = step;
}

template <typename scalar_t>
__global__ void interleave_sample_step_wrt_depth_clamp_v1_round2(
    const uint32_t num_packs,
    const uint32_t num_feats,
    const uint32_t max_steps,
    const int64_t* pack_infos,
    const int32_t* __restrict__ pidx_in,
    int64_t* __restrict__ pidx_out,
    const scalar_t* __restrict__ samples_in,
    scalar_t* __restrict__ samples_out,
    const scalar_t* __restrict__ deltas_in,
    scalar_t* __restrict__ deltas_out
) {
    const uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidx >= num_packs) return;

    uint32_t begin_in = tidx * max_steps;
    uint32_t begin_out = pack_infos[tidx * 2];
    uint32_t num_out = pack_infos[tidx * 2 + 1];

    pidx_out += begin_out;
    samples_out += begin_out;
    deltas_out += begin_out;
    
    pidx_in += begin_in;
    samples_in += begin_in;
    deltas_in += begin_in;

    for(uint32_t j=0; j < num_out; ++j) {
        pidx_out[j] = pidx_in[j];
        samples_out[j] = samples_in[j];
        deltas_out[j] = deltas_in[j];
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> interleave_sample_step_wrt_depth_clamp_deprecated(
    at::Tensor near, // [num_packs]
    at::Tensor far, // [num_packs]
    int32_t max_steps,
    double dt_gamma,
    double min_step_size,
    double max_step_size
) {
    at::TensorArg near_arg{near, "near", 1};
    at::TensorArg far_arg{far, "far", 2};

    at::checkDim(__func__, near_arg, 1);
    at::checkDim(__func__, far_arg, 1);

    at::checkAllSameGPU(__func__, {near_arg, far_arg});
    at::checkAllContiguous(__func__, {near_arg, far_arg});
    at::checkScalarTypes(__func__, near_arg, {at::kHalf, at::kFloat, at::kDouble});
    at::checkAllSameType(__func__, {near_arg, far_arg});

    at::checkSameSize(__func__, near_arg, far_arg);

    uint32_t num_packs = near.size(0);

    at::Tensor t_samples = at::empty({ num_packs * max_steps }, near.options());
    at::Tensor dt_samples = at::empty({ num_packs * max_steps }, near.options());
    at::Tensor nidx = at::empty({ num_packs * max_steps }, near.options().dtype(at::kInt));
    at::Tensor n_per_pack = at::zeros({ num_packs }, near.options().dtype(at::kLong));

    static constexpr uint32_t num_threads = 128;

    // Sample and store into sparse tensors
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(near.scalar_type(), "interleave_sample_step_wrt_depth_clamp_deprecated", ([&] {
        const at::cuda::OptionalCUDAGuard device_guard(at::device_of(near));
        auto stream = at::cuda::getCurrentCUDAStream();
        kernel_interleave_sample_step_wrt_depth_clamp_v1_round1<scalar_t><<<div_round_up(num_packs, num_threads), num_threads, 0, stream>>>(
            num_packs, (uint32_t)max_steps, (scalar_t)dt_gamma, (scalar_t)min_step_size, (scalar_t)max_step_size, 
            near.data_ptr<scalar_t>(), far.data_ptr<scalar_t>(), t_samples.data_ptr<scalar_t>(), dt_samples.data_ptr<scalar_t>(), 
            nidx.data_ptr<int32_t>(), n_per_pack.data_ptr<int64_t>()
        );
    }));

    at::Tensor cumsum = n_per_pack.cumsum(0);
    at::Tensor pack_infos = at::stack({ cumsum-n_per_pack, n_per_pack }, 1);
    const uint32_t num = cumsum[-1].item<int64_t>();

    // Allocate compact tensors
    at::Tensor t_compact = at::empty({ num }, t_samples.options());
    at::Tensor dt_compact = at::empty({ num }, dt_samples.options());
    at::Tensor nidx_compact = at::empty({ num }, nidx.options().dtype(at::kLong));

    // Compactify tensors (toss away not-used zeros)
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(near.scalar_type(), "interleave_sample_step_wrt_depth_clamp_deprecated", ([&] {
        const at::cuda::OptionalCUDAGuard device_guard(at::device_of(near));
        auto stream = at::cuda::getCurrentCUDAStream();
        interleave_sample_step_wrt_depth_clamp_v1_round2<scalar_t><<<div_round_up(num_packs, num_threads), num_threads, 0, stream>>>(
            num_packs, num, (uint32_t)max_steps, pack_infos.data_ptr<int64_t>(), 
            nidx.data_ptr<int32_t>(), nidx_compact.data_ptr<int64_t>(), 
            t_samples.data_ptr<scalar_t>(), t_compact.data_ptr<scalar_t>(),
            dt_samples.data_ptr<scalar_t>(), dt_compact.data_ptr<scalar_t>()
        );
    }));

    return {t_compact, dt_compact, nidx_compact, pack_infos};
}

/*
Atomic version;
- requires only one set of tensor storage
- faster (because of no additional compactifing)
- pack order not preserved
*/
/*
template <typename scalar_t>
__global__ void kernel_sample_step_wrt_gamma_depth_clamp_v2(
    const uint32_t num_packs,
    const uint32_t max_steps,
    const scalar_t dt_gamma,
    const scalar_t min_step_size,
    const scalar_t max_step_size,
    const scalar_t* __restrict__ nears,
    const scalar_t* __restrict__ fars,

    scalar_t* __restrict__ t_samples,
    scalar_t* __restrict__ deltas,
    // scalar_t* __restrict__ noise_buffer,

    int64_t* __restrict__ nidx,
    int64_t* __restrict__ pack_infos,
    int32_t* __restrict__ counter // [current pack, current point]
) {
    const uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidx >= num_packs) return;

    const scalar_t near = nears[tidx], far = fars[tidx];

    // First pass to get accurate number of steps
    scalar_t t = near;
    uint32_t num_steps = 0;

    while (t <= far && num_steps < max_steps) {
        t += clamp(t * dt_gamma, min_step_size, max_step_size);
        num_steps++;
    }

    int32_t pack_order_idx = atomicAdd(counter, 1);
    int32_t begin = atomicAdd(counter+1, num_steps);

    pack_infos[pack_order_idx * 2] = begin; // This is wrong! Will cause pack indices not consecutive!
    pack_infos[pack_order_idx * 2 + 1] = num_steps; // This is confusing! not actual pack order!

    t_samples += begin;
    nidx += begin;
    deltas += begin;

    // Actual process
    t = near;
    uint32_t step = 0;
    while (t <= far && step < num_steps) {
        t_samples[step] = t;
        nidx[step] = tidx;
        scalar_t dt = clamp(t * dt_gamma, min_step_size, max_step_size);
        deltas[step] = dt;
        t += dt;
        step++;
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> sample_step_wrt_gamma_depth_clamp_v2(
    at::Tensor near, // [num_packs]
    at::Tensor far, // [num_packs]
    int32_t max_steps,
    double dt_gamma,
    double min_step_size,
    double max_step_size
    // bool perturb
) {
    at::TensorArg near_arg{near, "near", 1};
    at::TensorArg far_arg{far, "far", 2};

    at::checkDim(__func__, near_arg, 1);
    at::checkDim(__func__, far_arg, 1);

    at::checkAllSameGPU(__func__, {near_arg, far_arg});
    at::checkAllContiguous(__func__, {near_arg, far_arg});
    at::checkScalarTypes(__func__, near_arg, {at::kHalf, at::kFloat, at::kDouble});
    at::checkAllSameType(__func__, {near_arg, far_arg});

    at::checkSameSize(__func__, near_arg, far_arg);

    uint32_t num_packs = near.size(0);

    at::Tensor t_samples = at::empty({ num_packs * max_steps }, near.options());
    at::Tensor dt_samples = at::empty({ num_packs * max_steps }, near.options());
    at::Tensor nidx = at::empty({ num_packs * max_steps }, near.options().dtype(at::kLong));
    at::Tensor noises;

    at::Tensor pack_infos = at::zeros({ num_packs, 2 }, near.options().dtype(at::kLong));

    at::Tensor counter = at::zeros({ 2 }, near.options().dtype(at::kInt));

    static constexpr uint32_t num_threads = 128;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(near.scalar_type(), "sample_step_wrt_gamma_depth_clamp_v2", ([&] {
        const at::cuda::OptionalCUDAGuard device_guard(at::device_of(near));
        auto stream = at::cuda::getCurrentCUDAStream();
        kernel_sample_step_wrt_gamma_depth_clamp_v2<scalar_t><<<div_round_up(num_packs, num_threads), num_threads, 0, stream>>>(
            num_packs, (uint32_t)max_steps, (scalar_t)dt_gamma, (scalar_t)min_step_size, (scalar_t)max_step_size, 
            near.data_ptr<scalar_t>(), far.data_ptr<scalar_t>(), t_samples.data_ptr<scalar_t>(), dt_samples.data_ptr<scalar_t>(), 
            nidx.data_ptr<int64_t>(), pack_infos.data_ptr<int64_t>(), counter.data_ptr<int32_t>()
        );
    }));

    const uint32_t num = counter[1].item<int32_t>();
    return {t_samples.slice(0, at::nullopt, num), dt_samples.slice(0, at::nullopt, num), nidx.slice(0, at::nullopt, num), n_per_pack, pack_infos};
}
*/

template <typename scalar_t>
__global__ void kernel_interleave_sample_step_wrt_depth_clamp_v3_round1(
    // Inputs
    const uint32_t num_packs,
    const uint32_t max_steps,
    const scalar_t dt_gamma,
    const scalar_t min_step_size,
    const scalar_t max_step_size,
    const scalar_t* __restrict__ nears,
    const scalar_t* __restrict__ fars,
    // Outputs
    int64_t* __restrict__ n_per_pack
) {
    const uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidx >= num_packs) return;

    const scalar_t near = nears[tidx], far = fars[tidx];

    scalar_t t = near;
    uint32_t num_steps = 0;

    while (t <= far && num_steps < max_steps) {
        t += clamp(t * dt_gamma, min_step_size, max_step_size);
        num_steps++;
    }
    n_per_pack[tidx] = num_steps;
}

template <typename scalar_t>
__global__ void kernel_interleave_sample_step_wrt_depth_clamp_v3_round2(
    // Inputs
    const uint32_t num_packs,
    const uint32_t max_steps,
    const scalar_t dt_gamma,
    const scalar_t min_step_size,
    const scalar_t max_step_size,
    const scalar_t* __restrict__ nears,
    const scalar_t* __restrict__ fars,
    const int64_t* __restrict__ pack_infos,
    // Outputs
    scalar_t* __restrict__ t_samples,
    scalar_t* __restrict__ deltas,
    int64_t* __restrict__ nidx
) {
    const uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidx >= num_packs) return;

    const scalar_t near = nears[tidx]; //, far = fars[tidx];
    const uint32_t begin = pack_infos[tidx * 2];
    const uint32_t num_steps = pack_infos[tidx * 2 + 1];

    t_samples += begin;
    nidx += begin;
    deltas += begin;

    scalar_t t = near;
    uint32_t step = 0;
    while (step < num_steps) {
        t_samples[step] = t;
        nidx[step] = tidx;
        scalar_t dt = clamp(t * dt_gamma, min_step_size, max_step_size);
        deltas[step] = dt;
        t += dt;
        step++;
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> interleave_sample_step_wrt_depth_clamped(
    at::Tensor near, // [num_packs]
    at::Tensor far, // [num_packs]
    int32_t max_steps,
    double dt_gamma,
    double min_step_size,
    double max_step_size
) {
    at::TensorArg near_arg{near, "near", 1};
    at::TensorArg far_arg{far, "far", 2};

    at::checkDim(__func__, near_arg, 1);
    at::checkDim(__func__, far_arg, 1);

    at::checkAllSameGPU(__func__, {near_arg, far_arg});
    at::checkAllContiguous(__func__, {near_arg, far_arg});
    at::checkScalarTypes(__func__, near_arg, {at::kHalf, at::kFloat, at::kDouble});
    at::checkAllSameType(__func__, {near_arg, far_arg});

    at::checkSameSize(__func__, near_arg, far_arg);

    uint32_t num_packs = near.size(0);

    at::Tensor n_per_pack = at::empty({ num_packs }, near.options().dtype(at::kLong));

    static constexpr uint32_t num_threads = 128;

    // First pass to get accurate number of steps
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(near.scalar_type(), "interleave_sample_step_wrt_depth_clamped", ([&] {
        const at::cuda::OptionalCUDAGuard device_guard(at::device_of(near));
        auto stream = at::cuda::getCurrentCUDAStream();
        kernel_interleave_sample_step_wrt_depth_clamp_v3_round1<scalar_t><<<div_round_up(num_packs, num_threads), num_threads, 0, stream>>>(
            num_packs, (uint32_t)max_steps, (scalar_t)dt_gamma, (scalar_t)min_step_size, (scalar_t)max_step_size, 
            near.data_ptr<scalar_t>(), far.data_ptr<scalar_t>(), n_per_pack.data_ptr<int64_t>()
        );
    }));

    at::Tensor cumsum = n_per_pack.cumsum(0);
    at::Tensor pack_infos = at::stack({cumsum-n_per_pack, n_per_pack}, 1);
    const uint32_t num = cumsum[-1].item<int64_t>();

    at::Tensor t_samples = at::empty({ num }, near.options());
    at::Tensor dt_samples = at::empty({ num }, near.options());
    at::Tensor nidx = at::empty({ num }, near.options().dtype(at::kLong));

    // Actual process
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(near.scalar_type(), "interleave_sample_step_wrt_depth_clamped", ([&] {
        const at::cuda::OptionalCUDAGuard device_guard(at::device_of(near));
        auto stream = at::cuda::getCurrentCUDAStream();
        kernel_interleave_sample_step_wrt_depth_clamp_v3_round2<scalar_t><<<div_round_up(num_packs, num_threads), num_threads, 0, stream>>>(
            num_packs, (uint32_t)max_steps, (scalar_t)dt_gamma, (scalar_t)min_step_size, (scalar_t)max_step_size,
            near.data_ptr<scalar_t>(), far.data_ptr<scalar_t>(), pack_infos.data_ptr<int64_t>(), 
            t_samples.data_ptr<scalar_t>(), dt_samples.data_ptr<scalar_t>(), nidx.data_ptr<int64_t>()
        );
    }));

    return {t_samples, dt_samples, nidx, pack_infos};
}

template <typename scalar_t>
__global__ void kernel_interleave_sample_step_wrt_depth_in_packed_segments_round1(
    // Inputs
    const uint32_t num_packs,
    const uint32_t num_segments,
    const uint32_t max_steps,
    const scalar_t dt_gamma,
    const scalar_t min_step_size,
    const scalar_t max_step_size,
    const scalar_t* __restrict__ nears, // [num_packs]
    const scalar_t* __restrict__ fars, // [num_packs]
    const scalar_t* __restrict__ entries, // [num_segments]
    const scalar_t* __restrict__ exits, // [num_segments]
    const int64_t* __restrict__ seg_pack_infos, // [num_packs, 2] Segments pack info

    // Outputs
    int64_t* __restrict__ n_per_pack
) {
    const uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidx >= num_packs) return;

    const scalar_t near = nears[tidx], far = fars[tidx];
    const uint32_t seg_begin = seg_pack_infos[tidx * 2];
    const uint32_t seg_end = seg_begin + seg_pack_infos[tidx * 2 + 1];

    scalar_t t = near;
    uint32_t num_steps = 0;
    for (uint32_t i = seg_begin; i < seg_end; ++i) {
        const scalar_t cur_entry = entries[i], cur_exit = exits[i];
        if (cur_entry >= far || cur_exit <= near) break;

        // March to current segment entry
        // do {
        //     t += clamp(t * dt_gamma, min_step_size, max_step_size);
        // } while (t < cur_entry);
        do {
            t += min_step_size;
        } while (t < cur_entry);

        // March all steps in current segment
        while (t <= cur_exit && t <= far && num_steps < max_steps) {
            t += clamp(t * dt_gamma, min_step_size, max_step_size);
            num_steps++;
        }
    }
    n_per_pack[tidx] = num_steps;
}

template <typename scalar_t>
__global__ void kernel_interleave_sample_step_wrt_depth_in_packed_segments_round2(
    // Inputs
    const uint32_t num_packs,
    const uint32_t num_segments,
    const uint32_t max_steps,
    const scalar_t dt_gamma,
    const scalar_t min_step_size,
    const scalar_t max_step_size,
    const scalar_t* __restrict__ nears,
    const scalar_t* __restrict__ fars,
    const scalar_t* __restrict__ entries,
    const scalar_t* __restrict__ exits,
    const int64_t* __restrict__ seg_pack_infos, // Input segments pack info
    const int64_t* __restrict__ pack_infos, // Output samples pack info
    // Outputs
    scalar_t* __restrict__ t_samples,
    scalar_t* __restrict__ deltas,
    int64_t* __restrict__ nidx,
    int64_t* __restrict__ sidx
) {
    const uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidx >= num_packs) return;

    const scalar_t near = nears[tidx], far = fars[tidx];

    const uint32_t seg_begin = seg_pack_infos[tidx * 2];
    const uint32_t seg_end = seg_begin + seg_pack_infos[tidx * 2 + 1];

    const uint32_t begin = pack_infos[tidx * 2];
    const uint32_t num_steps = pack_infos[tidx * 2 + 1];

    t_samples += begin;
    nidx += begin;
    sidx += begin;
    deltas += begin;

    scalar_t t = near;
    uint32_t step = 0;
    for (uint32_t i = seg_begin; i < seg_end; ++i) {
        const scalar_t cur_entry = entries[i], cur_exit = exits[i];
        if (cur_entry >= far || cur_exit <= near) break;

        // March to current segment entry
        /*--- v1 ---*/
        // do {
        //     t += clamp(t * dt_gamma, min_step_size, max_step_size);
        // } while (t < cur_entry);
        /*--- v2 ---*/
        do {
            t += min_step_size;
        } while (t < cur_entry);

        // March all steps in current segment
        while (t <= cur_exit && t <= far && step < num_steps) {
            t_samples[step] = t;
            nidx[step] = tidx;
            sidx[step] = i;
            scalar_t dt = clamp(t * dt_gamma, min_step_size, max_step_size);
            deltas[step] = dt;
            t += dt;
            step++;
        }
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> interleave_sample_step_wrt_depth_in_packed_segments(
    at::Tensor near, // [num_packs]
    at::Tensor far, // [num_packs]
    at::Tensor entry, // [num_segments]
    at::Tensor exit, // [num_segments]
    at::Tensor seg_pack_infos, // [num_packs, 2] segment pack infos
    int32_t max_steps,
    double dt_gamma,
    double min_step_size,
    double max_step_size
) {
    at::TensorArg near_arg{near, "near", 1};
    at::TensorArg far_arg{far, "far", 2};
    at::TensorArg entry_arg{entry, "entry", 3};
    at::TensorArg exit_arg{exit, "exit", 4};
    at::TensorArg seg_pack_infos_arg{seg_pack_infos, "seg_pack_infos", 5};

    at::checkDim(__func__, near_arg, 1);
    at::checkDim(__func__, far_arg, 1);
    at::checkDim(__func__, entry_arg, 1);
    at::checkDim(__func__, exit_arg, 1);
    at::checkDim(__func__, seg_pack_infos_arg, 2);
    at::checkAllSameGPU(__func__, {near_arg, far_arg, entry_arg, exit_arg, seg_pack_infos_arg});
    at::checkAllContiguous(__func__, {near_arg, far_arg, entry_arg, exit_arg, seg_pack_infos_arg});
    at::checkScalarTypes(__func__, near_arg, {at::kHalf, at::kFloat, at::kDouble});
    at::checkAllSameType(__func__, {near_arg, far_arg, entry_arg, exit_arg});
    at::checkScalarType(__func__, seg_pack_infos_arg, at::kLong);

    at::checkSameSize(__func__, near_arg, far_arg);
    at::checkSameSize(__func__, entry_arg, exit_arg);
    at::checkSize(__func__, seg_pack_infos_arg, {near.size(0), 2});

    const int64_t num_feats = seg_pack_infos.index({-1,0}).item<int64_t>() + seg_pack_infos.index({-1,1}).item<int64_t>();
    at::checkSize(__func__, entry_arg, 0, num_feats);
    at::checkSize(__func__, exit_arg, 0, num_feats);

    const uint32_t num_packs = near.size(0);
    const uint32_t num_segments = entry.size(0);
    at::Tensor n_per_pack = at::empty({ num_packs }, near.options().dtype(at::kLong));

    static constexpr uint32_t num_threads = 128;

    // First pass to get accurate number of steps
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(near.scalar_type(), "interleave_sample_step_wrt_depth_in_packed_segments", ([&] {
        const at::cuda::OptionalCUDAGuard device_guard(at::device_of(near));
        auto stream = at::cuda::getCurrentCUDAStream();
        kernel_interleave_sample_step_wrt_depth_in_packed_segments_round1<scalar_t><<<div_round_up(num_packs, num_threads), num_threads, 0, stream>>>(
            num_packs, num_segments, (uint32_t)max_steps, (scalar_t)dt_gamma, (scalar_t)min_step_size, (scalar_t)max_step_size, 
            near.data_ptr<scalar_t>(), far.data_ptr<scalar_t>(), entry.data_ptr<scalar_t>(), exit.data_ptr<scalar_t>(), 
            seg_pack_infos.data_ptr<int64_t>(), n_per_pack.data_ptr<int64_t>()
        );
    }));
    
    at::Tensor cumsum = n_per_pack.cumsum(0);
    at::Tensor pack_infos = at::stack({cumsum-n_per_pack, n_per_pack}, 1);
    const uint32_t num = cumsum[-1].item<int64_t>();

    at::Tensor t_samples = at::empty({ num }, near.options());
    at::Tensor dt_samples = at::empty({ num }, near.options());
    at::Tensor nidx = at::empty({ num }, near.options().dtype(at::kLong)); // The pack ind each sample belongs to
    at::Tensor sidx = at::empty({ num }, near.options().dtype(at::kLong)); // The segment ind each sample belongs to

    // Actual process
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(near.scalar_type(), "interleave_sample_step_wrt_depth_in_packed_segments", ([&] {
        const at::cuda::OptionalCUDAGuard device_guard(at::device_of(near));
        auto stream = at::cuda::getCurrentCUDAStream();
        kernel_interleave_sample_step_wrt_depth_in_packed_segments_round2<scalar_t><<<div_round_up(num_packs, num_threads), num_threads, 0, stream>>>(
            num_packs, num_segments, (uint32_t)max_steps, (scalar_t)dt_gamma, (scalar_t)min_step_size, (scalar_t)max_step_size, 
            near.data_ptr<scalar_t>(), far.data_ptr<scalar_t>(), entry.data_ptr<scalar_t>(), exit.data_ptr<scalar_t>(), 
            seg_pack_infos.data_ptr<int64_t>(), pack_infos.data_ptr<int64_t>(), 
            t_samples.data_ptr<scalar_t>(), dt_samples.data_ptr<scalar_t>(), nidx.data_ptr<int64_t>(), sidx.data_ptr<int64_t>()
        );
    }));
    
    return {t_samples, dt_samples, sidx, nidx, pack_infos};
}

// Modified from https://github.com/NVIDIAGameWorks/kaolin
template<typename scalar_t>
__global__ void kernel_packed_sum(
    // Inputs
    const uint32_t num_packs,
    const uint32_t num_feats, 
    const uint32_t feat_dim, 
    const scalar_t* __restrict__ feats_in, 
    const int64_t* __restrict__ pack_infos, 
    // Outputs
    scalar_t* __restrict__ feats_out
) {

    uint32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= num_packs) return;

    uint32_t begin = pack_infos[tidx * 2];
    uint32_t end = begin + pack_infos[tidx * 2 + 1];
    scalar_t result;
    // For loop on feat_dim first.
    for (uint32_t j=0; j<feat_dim; ++j) {
        result = feats_in[begin * feat_dim + j];
        for (uint32_t i=begin+1; i < end; ++i) {
            result += feats_in[i * feat_dim + j];
        }
        feats_out[tidx * feat_dim + j] = result;
    }
}

at::Tensor packed_sum(
    at::Tensor feats, // [num_feats, feat_dim] or [num_feats]
    at::Tensor pack_infos // [num_packs, 2]
) {
    at::TensorArg feats_arg{feats, "feats", 1};
    at::TensorArg pack_infos_arg{pack_infos, "pack_infos", 2};

    at::checkDimRange(__func__, feats_arg, 1, 3); // [1, 2] is allowed.
    at::checkDim(__func__, pack_infos_arg, 2);
    at::checkAllSameGPU(__func__, {feats_arg, pack_infos_arg});
    at::checkAllContiguous(__func__, {feats_arg, pack_infos_arg});
    at::checkScalarType(__func__, pack_infos_arg, at::kLong);

    at::checkSize(__func__, feats_arg, 0, pack_infos.index({-1,0}).item<int64_t>() + pack_infos.index({-1,1}).item<int64_t>());

    uint32_t num_packs = pack_infos.size(0);
    uint32_t num_feats = feats.size(0);

    int64_t* pack_infos_ptr = pack_infos.data_ptr<int64_t>();
    
    uint32_t feat_dim = feats.dim() == 1 ? 1 : feats.size(1);
    at::Tensor feats_out = feats.dim() == 1 ? at::zeros({num_packs}, feats.options()) : at::zeros({num_packs, feat_dim}, feats.options());

    static constexpr uint32_t num_threads = 256;

    AT_DISPATCH_ALL_TYPES_AND_HALF(feats.scalar_type(), "packed_sum", ([&] {
        const at::cuda::OptionalCUDAGuard device_guard(at::device_of(feats_out));
        auto stream = at::cuda::getCurrentCUDAStream();
        kernel_packed_sum<scalar_t><<<div_round_up((uint32_t)num_packs, num_threads), num_threads, 0, stream>>>(
            num_packs, num_feats, feat_dim, 
            feats.data_ptr<scalar_t>(), pack_infos_ptr, feats_out.data_ptr<scalar_t>()
        );
    }));

    return feats_out;
}

// Modified from https://github.com/NVIDIAGameWorks/kaolin
template<typename scalar_t>
__global__ void
kernel_packed_cumprod(
    // Inputs
    const uint32_t num_packs,
    const uint32_t num_feats,
    const uint32_t feat_dim, 
    const scalar_t* __restrict__ feats_in, 
    const int64_t* __restrict__ pack_infos,  // maps idx of pack -> beginning of global idx
    const int32_t offset,
    // Outputs
    scalar_t* __restrict__ feats_out
) {
    uint32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= num_packs) {
        return;
    }

    uint32_t begin = pack_infos[tidx * 2];
    uint32_t end = begin + pack_infos[tidx * 2 + 1];
    if (offset == 0) {
        for (uint32_t j=0; j<feat_dim; ++j){
            feats_out[begin * feat_dim + j] = feats_in[begin * feat_dim + j];
        }
    }
    // For loop on feat_dim first.
    for (uint32_t j=0; j<feat_dim; ++j){
        for (uint32_t i=begin+1; i<end; ++i) {
            feats_out[i * feat_dim + j] = feats_in[(i-offset) * feat_dim + j] * feats_out[(i-1) * feat_dim + j];
        }
    }
}

// Modified from https://github.com/NVIDIAGameWorks/kaolin
template<typename scalar_t>
__global__ void
kernel_packed_cumprod_reverse(
    // Inputs
    const uint32_t num_packs,
    const uint32_t num_feats,
    const uint32_t feat_dim, 
    const scalar_t* __restrict__ feats_in, 
    const int64_t* __restrict__ pack_infos,  // maps idx of pack -> beginning of global idx
    const int32_t offset,
    // Outputs
    scalar_t* __restrict__ feats_out
) {
    uint32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= num_packs) {
        return;
    }

    int32_t begin = pack_infos[tidx * 2];
    int32_t end = begin + pack_infos[tidx * 2 + 1];
    if (offset == 0) {
        for (uint32_t j=0; j<feat_dim; ++j){
            feats_out[(end-1) * feat_dim + j] = feats_in[(end-1) * feat_dim + j];
        }
    }
    // For loop on feat_dim first.
    for (uint32_t j=0; j<feat_dim; ++j){
        for (int32_t i=end-2; i>=(int32_t)begin; --i) {
            feats_out[i * feat_dim + j] = feats_in[(i+offset) * feat_dim + j] * feats_out[(i+1) * feat_dim + j];
        }
    }  
}

at::Tensor packed_cumprod(
    at::Tensor feats, // [num_feats, feat_dim] or [num_feats]
    at::Tensor pack_infos, // [num_packs, 2]
    bool exclusive,
    bool reverse
) {
    at::TensorArg feats_arg{feats, "feats", 1};
    at::TensorArg pack_infos_arg{pack_infos, "pack_infos", 2};
    at::checkDimRange(__func__, feats_arg, 1, 3); // [1, 2] is allowed.
    at::checkDim(__func__, pack_infos_arg, 2);
    at::checkAllSameGPU(__func__, {feats_arg, pack_infos_arg});
    at::checkAllContiguous(__func__,  {feats_arg, pack_infos_arg});
    at::checkScalarType(__func__, pack_infos_arg, at::kLong);

    at::checkSize(__func__, feats_arg, 0, pack_infos.index({-1,0}).item<int64_t>() + pack_infos.index({-1,1}).item<int64_t>());

    uint32_t num_feats = feats.size(0);
    uint32_t num_packs = pack_infos.size(0);
    
    uint32_t feat_dim = feats.dim() == 1 ? 1 : feats.size(1);
    at::Tensor feats_out = feats.dim() == 1 ? at::zeros({num_feats}, feats.options()) : at::zeros({num_feats, feat_dim}, feats.options());

    int64_t* pack_infos_ptr = pack_infos.data_ptr<int64_t>();
    int32_t offset = exclusive ? 1 : 0;

    const uint32_t num_threads = 128;
    if (reverse) {
        AT_DISPATCH_ALL_TYPES_AND_HALF(feats.scalar_type(), "packed_cumprod_reverse", ([&] {
            const at::cuda::OptionalCUDAGuard device_guard(at::device_of(feats_out));
            auto stream = at::cuda::getCurrentCUDAStream();
            kernel_packed_cumprod_reverse<scalar_t><<<div_round_up(num_packs, num_threads), num_threads, 0, stream>>>(
                num_packs, num_feats, feat_dim, 
                feats.data_ptr<scalar_t>(), pack_infos_ptr, offset, feats_out.data_ptr<scalar_t>()
            );
        }));
    } else {
        AT_DISPATCH_ALL_TYPES_AND_HALF(feats.scalar_type(), "packed_cumprod", ([&] {
            const at::cuda::OptionalCUDAGuard device_guard(at::device_of(feats_out));
            auto stream = at::cuda::getCurrentCUDAStream();
            kernel_packed_cumprod<scalar_t><<<div_round_up(num_packs, num_threads), num_threads, 0, stream>>>(
                num_packs, num_feats, feat_dim, 
                feats.data_ptr<scalar_t>(), pack_infos_ptr, offset, feats_out.data_ptr<scalar_t>()
            );
        }));
    }

    return feats_out;
}

// Modified from https://github.com/NVIDIAGameWorks/kaolin
template<typename scalar_t>
__global__ void
kernel_packed_cumsum(
    // Inputs
    const uint32_t num_packs,
    const uint32_t num_feats,
    const uint32_t feat_dim, 
    const scalar_t* __restrict__ feats_in, 
    const int64_t* __restrict__ pack_infos,  // maps idx of pack -> beginning of global idx
    const int32_t offset,
    // Outputs
    scalar_t* __restrict__ feats_out
) {
    uint32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= num_packs) {
        return;
    }

    uint32_t begin = pack_infos[tidx * 2];
    uint32_t end = begin + pack_infos[tidx * 2 + 1];
    if (offset == 0) {
        for (uint32_t j=0; j<feat_dim; ++j){
            feats_out[begin * feat_dim + j] = feats_in[begin * feat_dim + j];
        }
    }
    // For loop on feat_dim first.
    for (uint32_t j=0; j<feat_dim; ++j){
        for (uint32_t i=begin+1; i<end; ++i) {
            feats_out[i * feat_dim + j] = feats_in[(i-offset) * feat_dim + j] + feats_out[(i-1) * feat_dim + j];
        }
    }  
}

// Modified from https://github.com/NVIDIAGameWorks/kaolin
template<typename scalar_t>
__global__ void
kernel_packed_cumsum_reverse(
    // Inputs
    const uint32_t num_packs,
    const uint32_t num_feats,
    const uint32_t feat_dim, 
    const scalar_t* __restrict__ feats_in, 
    const int64_t* __restrict__ pack_infos,  // maps idx of pack -> beginning of global idx
    const int32_t offset,
    // Outputs
    scalar_t* __restrict__ feats_out
) {
    uint32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= num_packs) {
        return;
    }

    uint32_t begin = pack_infos[tidx * 2];
    uint32_t end = begin + pack_infos[tidx * 2 + 1];
    if (offset == 0) {
        for (uint32_t j=0; j<feat_dim; ++j){
            feats_out[(end-1) * feat_dim + j] = feats_in[(end-1) * feat_dim + j];
        }
    }
    // For loop on feat_dim first.
    for (uint32_t j=0; j<feat_dim; ++j){
        for (int32_t i=end-2; i>=(int32_t)begin; --i) {
            feats_out[i * feat_dim + j] = feats_in[(i+offset) * feat_dim + j] + feats_out[(i+1) * feat_dim + j];
        }
    }  
}

at::Tensor packed_cumsum(
    at::Tensor feats, // [num_feats, feat_dim] or [num_feats]
    at::Tensor pack_infos, // [num_packs, 2]
    bool exclusive,
    bool reverse
) {
    at::TensorArg feats_arg{feats, "feats", 1};
    at::TensorArg pack_infos_arg{pack_infos, "pack_infos", 2};
    at::checkDimRange(__func__, feats_arg, 1, 3); // [1, 2] is allowed.
    at::checkDim(__func__, pack_infos_arg, 2);
    at::checkAllSameGPU(__func__, {feats_arg, pack_infos_arg});
    at::checkAllContiguous(__func__,  {feats_arg, pack_infos_arg});
    at::checkScalarType(__func__, pack_infos_arg, at::kLong);

    at::checkSize(__func__, feats_arg, 0, pack_infos.index({-1,0}).item<int64_t>() + pack_infos.index({-1,1}).item<int64_t>());

    uint32_t num_feats = feats.size(0);
    uint32_t num_packs = pack_infos.size(0);

    uint32_t feat_dim = feats.dim() == 1 ? 1 : feats.size(1);
    at::Tensor feats_out = feats.dim() == 1 ? at::zeros({num_feats}, feats.options()) : at::zeros({num_feats, feat_dim}, feats.options());

    int64_t* pack_infos_ptr = pack_infos.data_ptr<int64_t>();
    int32_t offset = exclusive ? 1 : 0;

    const uint32_t num_threads = 256;
    if (reverse) {
        AT_DISPATCH_ALL_TYPES_AND_HALF(feats.scalar_type(), "packed_cumsum_reverse", ([&] {
            const at::cuda::OptionalCUDAGuard device_guard(at::device_of(feats_out));
            auto stream = at::cuda::getCurrentCUDAStream();
            kernel_packed_cumsum_reverse<scalar_t><<<div_round_up(num_packs, num_threads), num_threads, 0, stream>>>(
                num_packs, num_feats, feat_dim, 
                feats.data_ptr<scalar_t>(), pack_infos_ptr, offset, feats_out.data_ptr<scalar_t>()
            );
        }));
    } else {
        AT_DISPATCH_ALL_TYPES_AND_HALF(feats.scalar_type(), "packed_cumsum", ([&] {
            const at::cuda::OptionalCUDAGuard device_guard(at::device_of(feats_out));
            auto stream = at::cuda::getCurrentCUDAStream();
            kernel_packed_cumsum<scalar_t><<<div_round_up(num_packs, num_threads), num_threads, 0, stream>>>(
                num_packs, num_feats, feat_dim, 
                feats.data_ptr<scalar_t>(), pack_infos_ptr, offset, feats_out.data_ptr<scalar_t>()
            );
        }));
    }

    return feats_out;
}

// Modified from https://github.com/NVIDIAGameWorks/kaolin
template <typename scalar_t>
__global__ void kernel_packed_diff(
    // Inputs
    const uint32_t num_packs,
    const uint32_t num_feats, 
    const uint32_t feat_dim, 
    const scalar_t* __restrict__ feats_in, 

    const scalar_t* __restrict__ appends,
    const scalar_t* __restrict__ last_fills,

    const int64_t* __restrict__ pack_infos,
    // Outputs
    scalar_t* __restrict__ feats_out
) {
    uint32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= num_packs) return;

    uint32_t begin = pack_infos[tidx * 2];
    uint32_t end = begin + pack_infos[tidx * 2 + 1];

    // For loop on feat_dim first.
    for (uint32_t j=0; j<feat_dim; ++j) {
        for (uint32_t i=begin; i < end-1; ++i) {
            feats_out[i * feat_dim + j] = feats_in[(i+1) * feat_dim + j] - feats_in[i * feat_dim + j];
        }
    }

    if (appends) {
        for (uint32_t j=0; j<feat_dim; ++j) {
            feats_out[(end-1) * feat_dim + j] = appends[tidx * feat_dim + j] - feats_in[(end-1) * feat_dim + j];
        }
    } else if (last_fills) {
        for (uint32_t j=0; j<feat_dim; ++j) {
            feats_out[(end-1) * feat_dim + j] = last_fills[tidx * feat_dim + j];
        }
    } 
    // else {
    //     // By default, the last diff in the pack is zero.
    //     for (uint32_t j=0; j<feat_dim; ++j) {
    //         feats_out[(end-1) * feat_dim + j] = (scalar_t)0;
    //     }
    // }
}

template <typename scalar_t>
__global__ void kernel_packed_backward_diff(
    // Inputs
    const uint32_t num_packs,
    const uint32_t num_feats, 
    const uint32_t feat_dim, 
    const scalar_t* __restrict__ feats_in, 
    const scalar_t* __restrict__ prepends,
    const scalar_t* __restrict__ first_fill,
    const int64_t* __restrict__ pack_infos,
    // Outputs
    scalar_t* __restrict__ feats_out
) {
    uint32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= num_packs) return;

    uint32_t begin = pack_infos[tidx * 2];
    uint32_t end = begin + pack_infos[tidx * 2 + 1];

    // For loop on feat_dim first.
    for (uint32_t j=0; j<feat_dim; ++j) {
        for (uint32_t i=begin+1; i < end; ++i) {
            feats_out[i * feat_dim + j] = feats_in[i * feat_dim + j] - feats_in[(i-1) * feat_dim + j];
        }
    }

    if (prepends) {
        for (uint32_t j=0; j<feat_dim; ++j) {
            feats_out[begin * feat_dim + j] = feats_in[begin * feat_dim + j] - prepends[tidx * feat_dim + j];
        }
    } else if (first_fill) {
        for (uint32_t j=0; j<feat_dim; ++j) {
            feats_out[begin * feat_dim + j] = first_fill[tidx * feat_dim + j];
        }
    } 
    // else {
    //     // By default, the first diff in the pack is zero.
    //     for (uint32_t j=0; j<feat_dim; ++j) {
    //         feats_out[begin * feat_dim + j] = (scalar_t)0;
    //     }
    // }
}

at::Tensor packed_diff(
    at::Tensor feats,  // [num_feats, feat_dim] or [num_feats]
    at::Tensor pack_infos, // [num_packs, 2]
    at::optional<at::Tensor> pack_appends_, // [num_packs, feat_dim] or [num_packs]
    at::optional<at::Tensor> pack_last_fill_ // [num_packs, feat_dim] or [num_packs]
) {
    // https://en.wikipedia.org/wiki/Finite_difference
    // forward difference: [next value] - [this value]

    at::TensorArg feats_arg{feats, "feats", 1};
    at::TensorArg pack_infos_arg{pack_infos, "pack_infos", 2};

    at::checkDimRange(__func__, feats_arg, 1, 3); // [1, 2] is allowed.
    at::checkDim(__func__, pack_infos_arg, 2);
    at::checkAllContiguous(__func__, {feats_arg, pack_infos_arg});
    at::checkSameGPU(__func__, feats_arg, pack_infos_arg);
    at::checkScalarType(__func__, pack_infos_arg, at::kLong);

    if ((int32_t)pack_appends_.has_value() + (int32_t)pack_last_fill_.has_value() > 1) {
        throw std::runtime_error("You should only specify AT MOST one of [appends, prepends, last_fill, first_fill]");
    }

    at::checkSize(__func__, feats_arg, 0, pack_infos.index({-1,0}).item<int64_t>() + pack_infos.index({-1,1}).item<int64_t>());

    uint32_t num_packs = pack_infos.size(0);
    uint32_t num_feats = feats.size(0);
    uint32_t feat_dim = feats.dim() == 1 ? 1 : feats.size(1);

    at::Tensor pack_appends;
    if (pack_appends_.has_value()) {
        pack_appends = pack_appends_.value();
        at::TensorArg pack_appends_arg{pack_appends, "pack_appends", 3};
        at::checkContiguous(__func__, pack_appends_arg);
        at::checkSameGPU(__func__, feats_arg, pack_appends_arg);
        at::checkSameType(__func__, feats_arg, pack_appends_arg);
        if (feats.dim() == 1) {
            at::checkSize(__func__, pack_appends_arg, {num_packs});
        } else {
            at::checkSize(__func__, pack_appends_arg, {num_packs, feat_dim});
        }
    }

    at::Tensor pack_last_fill;
    if (pack_last_fill_.has_value()) {
        pack_last_fill = pack_last_fill_.value();
        at::TensorArg pack_last_fill_arg{pack_last_fill, "pack_last_fill", 4};
        at::checkContiguous(__func__, pack_last_fill_arg);
        at::checkSameGPU(__func__, feats_arg, pack_last_fill_arg);
        at::checkSameType(__func__, feats_arg, pack_last_fill_arg);
        if (feats.dim() == 1) {
            at::checkSize(__func__, pack_last_fill_arg, {num_packs});
        } else {
            at::checkSize(__func__, pack_last_fill_arg, {num_packs, feat_dim});
        }
    }

    at::Tensor feats_out = feats.dim() == 1 ? at::zeros({num_feats}, feats.options()) : at::zeros({num_feats, feat_dim}, feats.options());

    int64_t* pack_infos_ptr = pack_infos.data_ptr<int64_t>();

    static constexpr uint32_t num_threads = 128;

    AT_DISPATCH_ALL_TYPES_AND_HALF(feats.scalar_type(), "packed_diff", ([&] {
        const at::cuda::OptionalCUDAGuard device_guard(at::device_of(feats));
        auto stream = at::cuda::getCurrentCUDAStream();
        kernel_packed_diff<scalar_t><<<div_round_up(num_packs, num_threads), num_threads, 0, stream>>>(
            num_packs, num_feats, feat_dim, 
            feats.data_ptr<scalar_t>(), 
            pack_appends_.has_value() ? pack_appends.data_ptr<scalar_t>() : nullptr, 
            pack_last_fill_.has_value() ? pack_last_fill.data_ptr<scalar_t>() : nullptr, 
            pack_infos_ptr, feats_out.data_ptr<scalar_t>()
        );
    }));
    return feats_out;
}

at::Tensor packed_backward_diff(
    at::Tensor feats,  // [num_feats, feat_dim]
    at::Tensor pack_infos, // [num_packs, 2]
    at::optional<at::Tensor> pack_prepends_, // [num_packs, feat_dim]
    at::optional<at::Tensor> pack_first_fill_  // [num_packs, feat_dim]
) {
    // https://en.wikipedia.org/wiki/Finite_difference
    // backward difference: [this value] - [prev value]

    at::TensorArg feats_arg{feats, "feats", 1};
    at::TensorArg pack_infos_arg{pack_infos, "pack_infos", 2};

    at::checkDimRange(__func__, feats_arg, 1, 3); // [1, 2] is allowed.
    at::checkDim(__func__, pack_infos_arg, 2);
    at::checkAllContiguous(__func__, {feats_arg, pack_infos_arg});
    at::checkSameGPU(__func__, feats_arg, pack_infos_arg);
    at::checkScalarType(__func__, pack_infos_arg, at::kLong);
    TORCH_CHECK(!(pack_prepends_.has_value() and pack_first_fill_.has_value()));

    at::checkSize(__func__, feats_arg, 0, pack_infos.index({-1,0}).item<int64_t>() + pack_infos.index({-1,1}).item<int64_t>());

    uint32_t num_packs = pack_infos.size(0);
    uint32_t num_feats = feats.size(0);
    uint32_t feat_dim = feats.dim() == 1 ? 1 : feats.size(1);

    at::Tensor pack_prepends;
    if (pack_prepends_.has_value()) {
        pack_prepends = pack_prepends_.value();
        at::TensorArg pack_prepends_arg{pack_prepends, "pack_prepends", 3};
        at::checkContiguous(__func__, pack_prepends_arg);
        at::checkSameGPU(__func__, feats_arg, pack_prepends_arg);
        at::checkSameType(__func__, feats_arg, pack_prepends_arg);
        if (feats.dim() == 1) {
            at::checkSize(__func__, pack_prepends_arg, {num_packs});
        } else {
            at::checkSize(__func__, pack_prepends_arg, {num_packs, feat_dim});
        }
    }

    at::Tensor pack_first_fill;
    if (pack_first_fill_.has_value()) {
        pack_first_fill = pack_first_fill_.value();
        at::TensorArg pack_first_fill_arg{pack_first_fill, "pack_first_fill", 4};
        at::checkContiguous(__func__, pack_first_fill_arg);
        at::checkSameGPU(__func__, feats_arg, pack_first_fill_arg);
        at::checkSameType(__func__, feats_arg, pack_first_fill_arg);
        if (feats.dim() == 1) {
            at::checkSize(__func__, pack_first_fill_arg, {num_packs});
        } else {
            at::checkSize(__func__, pack_first_fill_arg, {num_packs, feat_dim});
        }
    }

    at::Tensor feats_out = feats.dim() == 1 ? at::zeros({num_feats}, feats.options()) : at::zeros({num_feats, feat_dim}, feats.options());

    int64_t* pack_infos_ptr = pack_infos.data_ptr<int64_t>();

    static constexpr uint32_t num_threads = 128;

    AT_DISPATCH_ALL_TYPES_AND_HALF(feats.scalar_type(), "packed_backward_diff", ([&] {
        const at::cuda::OptionalCUDAGuard device_guard(at::device_of(feats));
        auto stream = at::cuda::getCurrentCUDAStream();
        kernel_packed_backward_diff<scalar_t><<<div_round_up(num_packs, num_threads), num_threads, 0, stream>>>(
            num_packs, num_feats, feat_dim, 
            feats.data_ptr<scalar_t>(), 
            pack_prepends_.has_value() ? pack_prepends.data_ptr<scalar_t>() : nullptr, 
            pack_first_fill_.has_value() ? pack_first_fill.data_ptr<scalar_t>() : nullptr, 
            pack_infos_ptr, feats_out.data_ptr<scalar_t>()
        );
    }));
    return feats_out;
}


template <typename scalar_t>
inline __device__ uint32_t binary_search_unsafe(scalar_t val, const scalar_t* data, uint32_t length) {
    // Borrowed from intant-ngp
    // Returns "right" bound index of the found interval.
    // (None or data[return-1]) <= val < data[return]
    // Allows val less than the minimum data.
    // Disallows val larger than the maximum data. (will return wrong index)
    if (length == 0) {
		return 0;
	}
	uint32_t it;
	uint32_t count, step;
	count = length;

	uint32_t first = 0;
	while (count > 0) {
		it = first;
		step = count / 2;
		it += step;
		if (data[it] < val) {
			first = ++it;
			count -= step + 1;
		} else {
			count = step;
		}
	}
	return first;
}

template <typename scalar_t>
inline __device__ uint32_t binary_search(scalar_t val, const scalar_t* data, uint32_t length) {
	if (length == 0) {
		return 0;
	}
	uint32_t first = binary_search_unsafe<scalar_t>(val, data, length);
	return std::min(first, length-1);
}

template <typename scalar_t>
__global__ void kernel_packed_searchsorted(
    // Inputs
    const uint32_t num_packs,
    const uint32_t num_feats,
    const scalar_t* __restrict__ bins,
    const scalar_t* __restrict__ vals,
    const int64_t* __restrict__ pack_infos,
    uint32_t num_to_search,
    const int64_t* __restrict__ pack_infos_to_search, // Optional, if different num_to_search each pack.
    // Outputs
    int64_t* __restrict__ pidx
) {
    uint32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= num_packs) return;

    const uint32_t begin = pack_infos[tidx * 2];
    const uint32_t length = pack_infos[tidx * 2 + 1];

    uint32_t out_begin;
    if (pack_infos_to_search) {
        out_begin = pack_infos_to_search[tidx * 2];
        num_to_search = pack_infos_to_search[tidx * 2 + 1];
    } else {
        out_begin = tidx * num_to_search;
    }

    bins += begin;
    pidx += out_begin;
    vals += out_begin;
    for (uint32_t i=0; i < num_to_search; ++i) {
        pidx[i] = begin + binary_search(vals[i], bins, length); // `packed_searchsorted`
    }
}

at::Tensor packed_searchsorted(
    at::Tensor bins, // [num_feats]
    at::Tensor vals, // [num_pack, num_to_search]
    at::Tensor pack_infos // [num_pack, 2]
) {

    at::TensorArg bins_arg{bins, "bins", 1};
    at::TensorArg vals_arg{vals, "vals", 2};
    at::TensorArg pack_infos_arg{pack_infos, "pack_infos", 3};

    at::checkDim(__func__, bins_arg, 1);
    at::checkDim(__func__, vals_arg, 2);
    at::checkDim(__func__, pack_infos_arg, 2);
    at::checkAllSameGPU(__func__, {bins_arg, vals_arg, pack_infos_arg});
    at::checkAllContiguous(__func__, {bins_arg, vals_arg, pack_infos_arg});
    at::checkSameType(__func__, bins_arg, vals_arg);
    at::checkScalarType(__func__, pack_infos_arg, at::kLong);
    at::checkSize(__func__, pack_infos_arg, {vals.size(0), 2});

    at::checkSize(__func__, bins_arg, 0, pack_infos.index({-1,0}).item<int64_t>() + pack_infos.index({-1,1}).item<int64_t>());

    const uint32_t num_packs = pack_infos.size(0);
    const uint32_t num_feats = bins.size(0);
    const uint32_t num_to_search = vals.size(1);

    // pidx should always of the same size as vals;
    at::Tensor pidx = at::full_like(vals, -1, vals.options().dtype(at::kLong));
    int64_t* pidx_ptr = pidx.data_ptr<int64_t>();

    const int64_t* pack_infos_ptr = pack_infos.data_ptr<int64_t>();

    static constexpr uint32_t num_threads = 128;

    AT_DISPATCH_ALL_TYPES_AND_HALF(bins.scalar_type(), "packed_searchsorted", ([&] {
        const at::cuda::OptionalCUDAGuard device_guard(at::device_of(pidx));
        auto stream = at::cuda::getCurrentCUDAStream();
        kernel_packed_searchsorted<scalar_t><<<div_round_up(num_packs, num_threads), num_threads, 0, stream>>>(
            num_packs, num_feats, 
            bins.data_ptr<scalar_t>(), vals.data_ptr<scalar_t>(), 
            pack_infos_ptr, num_to_search, nullptr, pidx_ptr
        );
    }));

    return pidx;
}

at::Tensor packed_searchsorted_packed_vals(
    at::Tensor bins, // [num_feats]
    at::Tensor pack_infos, // [num_pack, 2]
    at::Tensor vals, // [num_feats_to_search]
    at::Tensor val_pack_infos // [num_pack, 2]
) {
    at::TensorArg bins_arg{bins, "bins", 1};
    at::TensorArg pack_infos_arg{pack_infos, "pack_infos", 2};
    at::TensorArg vals_arg{vals, "vals", 3};
    at::TensorArg val_pack_infos_arg{val_pack_infos, "val_pack_infos", 4};

    at::checkDim(__func__, bins_arg, 1);
    at::checkDim(__func__, vals_arg, 1);
    at::checkDim(__func__, pack_infos_arg, 2);
    at::checkDim(__func__, val_pack_infos_arg, 2);
    at::checkAllSameGPU(__func__, {bins_arg, vals_arg, pack_infos_arg, val_pack_infos_arg});
    at::checkAllContiguous(__func__, {bins_arg, vals_arg, pack_infos_arg, val_pack_infos_arg});
    at::checkSameType(__func__, bins_arg, vals_arg);
    at::checkScalarType(__func__, pack_infos_arg, at::kLong);
    at::checkScalarType(__func__, val_pack_infos_arg, at::kLong);
    at::checkSize(__func__, val_pack_infos_arg, {pack_infos.size(0), 2});

    at::checkSize(__func__, bins_arg, 0, pack_infos.index({-1,0}).item<int64_t>() + pack_infos.index({-1,1}).item<int64_t>());
    at::checkSize(__func__, vals_arg, 0, val_pack_infos.index({-1,0}).item<int64_t>() + val_pack_infos.index({-1,1}).item<int64_t>());

    const uint32_t num_packs = pack_infos.size(0);
    const uint32_t num_feats = bins.size(0);

    // pidx should always of the same size as vals;
    at::Tensor pidx = at::full_like(vals, -1, vals.options().dtype(at::kLong));
    int64_t* pidx_ptr = pidx.data_ptr<int64_t>();

    const int64_t* pack_infos_ptr = pack_infos.data_ptr<int64_t>();
    const int64_t* pack_infos_to_search_ptr = val_pack_infos.data_ptr<int64_t>();

    static constexpr uint32_t num_threads = 128;

    AT_DISPATCH_ALL_TYPES_AND_HALF(bins.scalar_type(), "packed_searchsorted_packed_vals", ([&] {
        const at::cuda::OptionalCUDAGuard device_guard(at::device_of(pidx));
        auto stream = at::cuda::getCurrentCUDAStream();
        kernel_packed_searchsorted<scalar_t><<<div_round_up(num_packs, num_threads), num_threads, 0, stream>>>(
            num_packs, num_feats, 
            bins.data_ptr<scalar_t>(), vals.data_ptr<scalar_t>(), 
            pack_infos_ptr, 0, pack_infos_to_search_ptr, pidx_ptr
        );
    }));

    return pidx;
}

template <typename scalar_t>
__global__ void kernel_try_merge_two_packs_sorted_aligned(
    // Inputs
    const uint32_t num_packs, 
    const uint32_t num_feats_a, 
    const scalar_t* __restrict__ vals_a, 
    const int64_t* __restrict__ pack_infos_a, 
    const uint32_t num_feats_b, 
    const scalar_t* __restrict__ vals_b, 
    const int64_t* __restrict__ pack_infos_b, 
    const int64_t* __restrict__ pack_infos_merged,
    // Outputs
    int64_t* __restrict__ pidx_a,
    int64_t* __restrict__ pidx_b, 
    bool b_sorted = true
) {
    uint32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= num_packs) return;

    const uint32_t begin = pack_infos_a[tidx * 2];
    const uint32_t length = pack_infos_a[tidx * 2 + 1];

    const uint32_t b_begin = pack_infos_b[tidx * 2];
    const uint32_t b_length = pack_infos_b[tidx * 2 + 1];

    const uint32_t out_begin = pack_infos_merged[tidx * 2];

    vals_a += begin;
    pidx_a += begin;

    vals_b += b_begin;
    pidx_b += b_begin;
    
    // Put search result i in `pidx_b`; put i count into `pidx_a`
    if (b_sorted) {
        // NOTE: Faster: since vals_b is also sorted, we can skip last found left bound.
        int last_i = 0;
        for (uint32_t j=0; j < b_length; ++j) {
            // int i = binary_search_unsafe<scalar_t>(vals_b[j], vals_a, length);
            int i = binary_search_unsafe<scalar_t>(vals_b[j], vals_a+last_i, length-last_i) + last_i;
            pidx_b[j] = i;
            if (i < length) pidx_a[i]++;
            last_i = i;
        }
    } else {
        for (uint32_t j=0; j < b_length; ++j) {
            int i = binary_search_unsafe<scalar_t>(vals_b[j], vals_a, length);
            pidx_b[j] = i;
            if (i < length) pidx_a[i]++;
        }
    }

    // From i count to `pidx_a` offset
    pidx_a[0] += out_begin;
    for (uint32_t i=1; i < length; ++i) {
        pidx_a[i] += pidx_a[i-1] + 1;
    }

    // From i to `pidx_b` offset
    uint32_t acc = 1;
    int last_i = -1;
    for (uint32_t j=0; j < b_length; ++j) {
        int i = pidx_b[j];
        pidx_b[j] = ((i==last_i) ? (++acc) : (acc=0)) + ((i==0) ? out_begin : (pidx_a[i-1]+1));
        last_i = i;
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> try_merge_two_packs_sorted_aligned(
    at::Tensor vals_a, 
    at::Tensor pack_infos_a,
    at::Tensor vals_b,
    at::Tensor pack_infos_b, 
    bool b_sorted
) {
    // Merge two already sorted packs
    
    at::TensorArg vals_a_arg{vals_a, "vals_a", 1};
    at::TensorArg pack_infos_a_arg{pack_infos_a, "pack_infos_a", 2};
    at::TensorArg vals_b_arg{vals_b, "vals_b", 3};
    at::TensorArg pack_infos_b_arg{pack_infos_b, "pack_infos_b", 4};

    at::checkDim(__func__, vals_a_arg, 1);
    at::checkDim(__func__, vals_b_arg, 1);
    at::checkDim(__func__, pack_infos_a_arg, 2);
    at::checkDim(__func__, pack_infos_b_arg, 2);
    at::checkAllSameGPU(__func__, {vals_a_arg, vals_b_arg, pack_infos_a_arg, pack_infos_b_arg});
    at::checkAllContiguous(__func__, {vals_a_arg, vals_b_arg, pack_infos_a_arg, pack_infos_b_arg});
    at::checkSameType(__func__, vals_a_arg, vals_b_arg);
    at::checkScalarType(__func__, pack_infos_a_arg, at::kLong);
    at::checkScalarType(__func__, pack_infos_b_arg, at::kLong);
    at::checkSize(__func__, pack_infos_b_arg, {pack_infos_a.size(0), 2});

    // Check vals' size with pack_infos
    at::checkSize(__func__, vals_a_arg, 0, pack_infos_a.index({-1,0}).item<int64_t>() + pack_infos_a.index({-1,1}).item<int64_t>());
    at::checkSize(__func__, vals_b_arg, 0, pack_infos_b.index({-1,0}).item<int64_t>() + pack_infos_b.index({-1,1}).item<int64_t>());

    const uint32_t num_packs = pack_infos_a.size(0);
    const uint32_t num_feats_a = vals_a.size(0);
    const uint32_t num_feats_b = vals_b.size(0);

    const int64_t* pack_infos_a_ptr = pack_infos_a.data_ptr<int64_t>();
    const int64_t* pack_infos_b_ptr = pack_infos_b.data_ptr<int64_t>();

    at::Tensor n_per_pack = pack_infos_a.select(1, 1) + pack_infos_b.select(1, 1);
    at::Tensor cumsum = n_per_pack.cumsum(0);
    at::Tensor pack_infos = at::stack({cumsum-n_per_pack, n_per_pack}, 1);

    at::Tensor pidx_a = at::zeros({ num_feats_a }, pack_infos_a.options());
    at::Tensor pidx_b = at::zeros({ num_feats_b }, pack_infos_b.options());

    static constexpr uint32_t num_threads = 128;

    AT_DISPATCH_ALL_TYPES_AND_HALF(vals_a.scalar_type(), "try_merge_two_packs_sorted_aligned", ([&] {
        const at::cuda::OptionalCUDAGuard device_guard(at::device_of(vals_a));
        auto stream = at::cuda::getCurrentCUDAStream();
        kernel_try_merge_two_packs_sorted_aligned<scalar_t><<<div_round_up(num_packs, num_threads), num_threads, 0, stream>>>(
            num_packs, 
            num_feats_a, vals_a.data_ptr<scalar_t>(), pack_infos_a_ptr, 
            num_feats_b, vals_b.data_ptr<scalar_t>(), pack_infos_b_ptr, 
            pack_infos.data_ptr<int64_t>(), pidx_a.data_ptr<int64_t>(), pidx_b.data_ptr<int64_t>(), 
            b_sorted
        );
    }));

    return {pidx_a, pidx_b, pack_infos};
}

template <typename scalar_t>
__global__ void kernel_packed_invert_cdf(
    // Inputs
    const uint32_t num_packs,
    const uint32_t num_feats,
    const scalar_t* __restrict__ bins,
    const scalar_t* __restrict__ cdfs,
    const int64_t* __restrict__ pack_infos,
    const scalar_t* __restrict__ u_vals,
    uint32_t num_to_sample,
    const int64_t* __restrict__ out_pack_infos, // Optional, if different num_to_sample each pack.
    // Outputs
    scalar_t* __restrict__ samples,
    int64_t* __restrict__ bin_idx
) {
    uint32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= num_packs) return;

    const scalar_t eps = 1.0e-5f;

    const uint32_t begin = pack_infos[tidx * 2];
    const uint32_t length = pack_infos[tidx * 2 + 1];

    uint32_t out_begin;
    if (out_pack_infos) {
        out_begin = out_pack_infos[tidx * 2];
        num_to_sample = out_pack_infos[tidx * 2 + 1];
    } else {
        out_begin = tidx * num_to_sample;
    }

    bins += begin;
    cdfs += begin;
    u_vals += out_begin;
    bin_idx += out_begin;
    samples += out_begin;
    for (uint32_t i=0; i < num_to_sample; ++i) {
        scalar_t u = u_vals[i];
        uint32_t pos = binary_search(u, cdfs, length); // `packed_searchsorted`
        bin_idx[i] = pos + begin;
        if (pos == 0) {
            samples[i] = bins[0];
        } else {
            uint32_t pos_prev = pos - 1;
            scalar_t pmf = cdfs[pos] - cdfs[pos_prev];
            samples[i] = pmf < eps ? bins[pos_prev] : (bins[pos_prev] + ((u - cdfs[pos_prev]) / pmf) * (bins[pos] - bins[pos_prev]));
        }
    }
}

std::tuple<at::Tensor, at::Tensor> packed_invert_cdf(
    at::Tensor bins, // [num_feats]
    at::Tensor cdfs, // [num_feats]
    at::Tensor u_vals, // [num_packs, num_to_sample]
    at::Tensor pack_infos // [num_packs, 2]
) {
    at::TensorArg bins_arg{bins, "bins", 1};
    at::TensorArg cdfs_arg{cdfs, "cdfs", 2};
    at::TensorArg u_vals_arg{u_vals, "u_vals", 3};
    at::TensorArg pack_infos_arg{pack_infos, "pack_infos", 4};

    at::checkDim(__func__, bins_arg, 1);
    at::checkDim(__func__, cdfs_arg, 1);
    at::checkDim(__func__, u_vals_arg, 2);
    at::checkDim(__func__, pack_infos_arg, 2);
    at::checkAllSameGPU(__func__, {bins_arg, cdfs_arg, u_vals_arg, pack_infos_arg});
    at::checkAllContiguous(__func__, {bins_arg, cdfs_arg, u_vals_arg, pack_infos_arg});
    at::checkScalarTypes(__func__, u_vals_arg, {at::kHalf, at::kFloat, at::kDouble});
    at::checkAllSameType(__func__, {bins_arg, cdfs_arg, u_vals_arg});
    at::checkScalarType(__func__, pack_infos_arg, at::kLong);
    at::checkSameSize(__func__, bins_arg, cdfs_arg);
    at::checkSize(__func__, pack_infos_arg, {u_vals.size(0), 2});

    at::checkSize(__func__, bins_arg, 0, pack_infos.index({-1,0}).item<int64_t>() + pack_infos.index({-1,1}).item<int64_t>());

    const uint32_t num_feats = bins.size(0);
    const uint32_t num_packs = pack_infos.size(0);
    const uint32_t num_to_sample = u_vals.size(1);

    // `bin_idx` should always of the same size as u_vals;
    at::Tensor bin_idx = at::full_like(u_vals, -1, u_vals.options().dtype(at::kLong));
    at::Tensor t_samples = at::zeros_like(u_vals, u_vals.options());
    int64_t* pidx_ptr = bin_idx.data_ptr<int64_t>();

    const int64_t* pack_infos_ptr = pack_infos.data_ptr<int64_t>();

    static constexpr uint32_t num_threads = 128;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(bins.scalar_type(), "packed_invert_cdf", ([&] {
        const at::cuda::OptionalCUDAGuard device_guard(at::device_of(t_samples));
        auto stream = at::cuda::getCurrentCUDAStream();
        kernel_packed_invert_cdf<scalar_t><<<div_round_up(num_packs, num_threads), num_threads, 0, stream>>>(
            num_packs, num_feats, 
            bins.data_ptr<scalar_t>(), cdfs.data_ptr<scalar_t>(), pack_infos_ptr, 
            u_vals.data_ptr<scalar_t>(), num_to_sample, nullptr, 
            t_samples.data_ptr<scalar_t>(), pidx_ptr
        );
    }));

    return {t_samples, bin_idx};
}

template <typename scalar_t>
__global__ void kernel_packed_alpha_to_vw_forward(
	// Inputs
	const uint32_t num_packs, 
	const uint32_t num_feats,
	const scalar_t* __restrict__ alphas,
	const scalar_t early_stop_eps, 
	const scalar_t alpha_thre, 
	const int64_t* __restrict__ pack_infos, 
	// Outputs
	scalar_t* __restrict__ weights, 
	int64_t* __restrict__ num_steps,
	bool* __restrict__ compact_selector // the samples that we needs to compute the gradients
) {
	// Modified from https://github.com/KAIR-BAIR/nerfacc
	uint32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
	if (tidx >= num_packs) return;

	const uint32_t begin = pack_infos[tidx * 2];
	const uint32_t length = pack_infos[tidx * 2 + 1];

	alphas += begin;
	if (weights) weights += begin;
	if (compact_selector) compact_selector += begin;

	// Accumulated rendering
	scalar_t T = 1.f;
	int cnt = 0;
	for (int j = 0; j < length; ++j)
	{
		if (T < early_stop_eps)
		{
			break;
		}

		scalar_t alpha = alphas[j];

		if (alpha <= alpha_thre)
		{
			// Empty space
			continue;
		}

		const scalar_t weight = alpha * T;
		T *= (1.f - alpha);

		if (weights) {
			weights[j] = weight;
		}

		if (compact_selector) {
			compact_selector[j] = true;
		}
        cnt += 1;
	}
	if (num_steps) {
		num_steps[tidx] = cnt;
	}
}

template <typename scalar_t>
__global__ void kernel_packed_alpha_to_vw_backward(
	// Inputs
	const uint32_t num_packs, 
	const uint32_t num_feats,
	const scalar_t* __restrict__ alphas,
	const scalar_t* __restrict__ weights,
	const scalar_t* __restrict__ grad_weights,
	const scalar_t early_stop_eps, 
	const scalar_t alpha_thre, 
    const int64_t* __restrict__ pack_infos,

	// Outputs
	scalar_t* __restrict__ grad_alphas
) {
	// Modified from https://github.com/KAIR-BAIR/nerfacc
	uint32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
	if (tidx >= num_packs) return;

	const uint32_t begin = pack_infos[tidx * 2];
	const uint32_t length = pack_infos[tidx * 2 + 1];

	alphas += begin;
	grad_alphas += begin;
	
	weights += begin;
	grad_weights += begin;

	scalar_t accum = 0;
	for (int j = 0; j < length; ++j)
	{
		accum += grad_weights[j] * weights[j];
	}

	// Backward of accumulated rendering
	scalar_t T = 1.f;
	for (int j = 0; j < length; ++j) {
		if (T < early_stop_eps)
		{
			break;
		}

		scalar_t alpha = alphas[j];
		if (alpha < alpha_thre)
		{
			// Empty space
			continue;
		}
		grad_alphas[j] = (grad_weights[j] * T - accum) / fmaxf(1.f - alpha, 1e-10f);

		accum -= grad_weights[j] * weights[j];
		T *= (1.f - alpha);
	}
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> packed_alpha_to_vw_forward(
	at::Tensor alphas,
	at::Tensor pack_infos,
	float early_stop_eps, 
	float alpha_thre, 
	bool compression
) {
	// Modified from https://github.com/KAIR-BAIR/nerfacc
	at::TensorArg alphas_arg(alphas, "alphas", 1);
	at::TensorArg pack_infos_arg(pack_infos, "pack_infos", 2);
	
	at::checkDim(__func__, alphas_arg, 1);
	at::checkDim(__func__, pack_infos_arg, 2);
	at::checkAllSameGPU(__func__, {alphas_arg, pack_infos_arg});
	at::checkAllContiguous(__func__, {alphas_arg, pack_infos_arg});
	at::checkScalarTypes(__func__, alphas_arg,  {at::kHalf, at::kFloat, at::kDouble});
	at::checkScalarTypes(__func__, pack_infos_arg, at::kLong);

    at::checkSize(__func__, alphas_arg, 0, pack_infos.index({-1,0}).item<int64_t>() + pack_infos.index({-1,1}).item<int64_t>());

	const uint32_t num_feats = alphas.size(0);
	const uint32_t num_packs = pack_infos.size(0);

	at::Tensor weights;
	at::Tensor compact_pack_info;
	at::Tensor compact_selector;

	static constexpr uint32_t num_threads = 128;

	if (compression) {
		at::Tensor num_steps = at::zeros({ num_packs }, pack_infos.options());
		compact_selector = at::zeros({ num_feats }, alphas.options().dtype(at::kBool));
		AT_DISPATCH_FLOATING_TYPES_AND_HALF(alphas.scalar_type(), "packed_alpha_to_vw_forward", ([&] {
			const at::cuda::OptionalCUDAGuard device_guard(at::device_of(alphas));
			auto stream = at::cuda::getCurrentCUDAStream();
			kernel_packed_alpha_to_vw_forward<scalar_t><<<div_round_up(num_packs, num_threads), num_threads, 0, stream>>>(
				num_packs, num_feats, 
				alphas.data_ptr<scalar_t>(), (scalar_t)early_stop_eps, (scalar_t)alpha_thre, pack_infos.data_ptr<int64_t>(), 
				nullptr, num_steps.data_ptr<int64_t>(), compact_selector.data_ptr<bool>()
			);
		}));
        at::Tensor cumsum = num_steps.cumsum(0, at::kInt);
        compact_pack_info = at::stack({cumsum - num_steps, num_steps}, 1);
	} else {
		weights = at::zeros({ num_feats }, alphas.options());
		AT_DISPATCH_FLOATING_TYPES_AND_HALF(alphas.scalar_type(), "packed_alpha_to_vw_forward", ([&] {
			const at::cuda::OptionalCUDAGuard device_guard(at::device_of(alphas));
			auto stream = at::cuda::getCurrentCUDAStream();
			kernel_packed_alpha_to_vw_forward<scalar_t><<<div_round_up(num_packs, num_threads), num_threads, 0, stream>>>(
				num_packs, num_feats, 
				alphas.data_ptr<scalar_t>(), (scalar_t)early_stop_eps, (scalar_t)alpha_thre, pack_infos.data_ptr<int64_t>(), 
				weights.data_ptr<scalar_t>(), nullptr, nullptr
			);
		}));
	}

	return {weights, compact_pack_info, compact_selector};
}


at::Tensor packed_alpha_to_vw_backward(
	at::Tensor weights,
	at::Tensor grad_weights,
	at::Tensor alphas,
	at::Tensor pack_infos,
    float early_stop_eps,
    float alpha_thre
) {
	// Modified from https://github.com/KAIR-BAIR/nerfacc
	at::TensorArg weights_arg(weights, "weights", 1);
	at::TensorArg grad_weights_arg(grad_weights, "weights", 2);
	at::TensorArg alphas_arg(alphas, "alpha", 3);
	at::TensorArg pack_infos_arg(pack_infos, "pack_infos", 4);
	
	at::checkDim(__func__, weights_arg, 1);
	at::checkDim(__func__, grad_weights_arg, 1);
	at::checkDim(__func__, alphas_arg, 1);
	at::checkDim(__func__, pack_infos_arg, 2);
	at::checkAllSameGPU(__func__, {weights_arg, grad_weights_arg, alphas_arg, pack_infos_arg});
	at::checkAllContiguous(__func__, {weights_arg, grad_weights_arg, alphas_arg, pack_infos_arg});
	at::checkScalarTypes(__func__, weights_arg,  {at::kHalf, at::kFloat, at::kDouble});
	at::checkAllSameType(__func__, {weights_arg, grad_weights_arg, alphas_arg});
	at::checkScalarTypes(__func__, pack_infos_arg, at::kLong);
	at::checkSameSize(__func__, weights_arg, grad_weights_arg);
	at::checkSameSize(__func__, weights_arg, alphas_arg);

    const int64_t num_feats_ = pack_infos.index({-1,0}).item<int64_t>() + pack_infos.index({-1,1}).item<int64_t>();
    at::checkSize(__func__, weights_arg, 0, num_feats_);

	const uint32_t num_packs = pack_infos.size(0);
	const uint32_t num_feats = weights.size(0);

	static constexpr uint32_t num_threads = 128;

	at::Tensor grad_alphas = at::zeros({ num_feats }, alphas.options());

	AT_DISPATCH_FLOATING_TYPES_AND_HALF(weights.scalar_type(), "packed_alpha_to_vw_backward", ([&] {
		const at::cuda::OptionalCUDAGuard device_guard(at::device_of(weights));
		auto stream = at::cuda::getCurrentCUDAStream();
		kernel_packed_alpha_to_vw_backward<scalar_t><<<div_round_up(num_packs, num_threads), num_threads, 0, stream>>>(
			num_packs, num_feats, 
			alphas.data_ptr<scalar_t>(), weights.data_ptr<scalar_t>(), grad_weights.data_ptr<scalar_t>(), 
            (scalar_t)early_stop_eps, (scalar_t)alpha_thre, pack_infos.data_ptr<int64_t>(), 
			grad_alphas.data_ptr<scalar_t>()
		);
	}));

	return grad_alphas;
}

template<typename scalar_t>
__global__ void kernel_packed_add(
    // Inputs
    const uint32_t num_packs,
    const uint32_t num_feats, 
    const uint32_t feat_dim, 
    const scalar_t* __restrict__ feats_in, // [num_feats, feat_dim]
    const scalar_t* __restrict__ other_in, // [num_packs, feat_dim]
    const int64_t* __restrict__ pack_infos, 
    // Outputs
    scalar_t* __restrict__ feats_out
) {
    uint32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= num_packs) return;

    other_in += tidx * feat_dim;
    uint32_t begin = pack_infos[tidx * 2];
    uint32_t end = begin + pack_infos[tidx * 2 + 1];
    // For loop on feat_dim first.
    for (uint32_t j=0; j<feat_dim; ++j) {
        for (uint32_t i=begin; i < end; ++i) {
            feats_out[i * feat_dim + j] = feats_in[i * feat_dim + j] + other_in[j];
        }
    }
}

template<typename scalar_t>
__global__ void kernel_packed_sub(
    // Inputs
    const uint32_t num_packs,
    const uint32_t num_feats, 
    const uint32_t feat_dim, 
    const scalar_t* __restrict__ feats_in, // [num_feats, feat_dim]
    const scalar_t* __restrict__ other_in, // [num_packs, feat_dim]
    const int64_t* __restrict__ pack_infos, 
    // Outputs
    scalar_t* __restrict__ feats_out
) {
    uint32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= num_packs) return;

    other_in += tidx * feat_dim;
    uint32_t begin = pack_infos[tidx * 2];
    uint32_t end = begin + pack_infos[tidx * 2 + 1];
    // For loop on feat_dim first.
    for (uint32_t j=0; j<feat_dim; ++j) {
        for (uint32_t i=begin; i < end; ++i) {
            feats_out[i * feat_dim + j] = feats_in[i * feat_dim + j] - other_in[j];
        }
    }
}

template<typename scalar_t>
__global__ void kernel_packed_mul(
    // Inputs
    const uint32_t num_packs,
    const uint32_t num_feats, 
    const uint32_t feat_dim, 
    const scalar_t* __restrict__ feats_in, // [num_feats, feat_dim]
    const scalar_t* __restrict__ other_in, // [num_packs, feat_dim]
    const int64_t* __restrict__ pack_infos, 
    // Outputs
    scalar_t* __restrict__ feats_out
) {
    uint32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= num_packs) return;

    other_in += tidx * feat_dim;
    uint32_t begin = pack_infos[tidx * 2];
    uint32_t end = begin + pack_infos[tidx * 2 + 1];
    // For loop on feat_dim first.
    for (uint32_t j=0; j<feat_dim; ++j) {
        for (uint32_t i=begin; i < end; ++i) {
            feats_out[i * feat_dim + j] = feats_in[i * feat_dim + j] * other_in[j];
        }
    }
}

template<typename scalar_t>
__global__ void kernel_packed_div(
    // Inputs
    const uint32_t num_packs,
    const uint32_t num_feats, 
    const uint32_t feat_dim, 
    const scalar_t* __restrict__ feats_in, // [num_feats, feat_dim]
    const scalar_t* __restrict__ other_in, // [num_packs, feat_dim]
    const int64_t* __restrict__ pack_infos, 
    // Outputs
    scalar_t* __restrict__ feats_out
) {
    uint32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= num_packs) return;

    other_in += tidx * feat_dim;
    uint32_t begin = pack_infos[tidx * 2];
    uint32_t end = begin + pack_infos[tidx * 2 + 1];
    // For loop on feat_dim first.
    for (uint32_t j=0; j<feat_dim; ++j) {
        for (uint32_t i=begin; i < end; ++i) {
            feats_out[i * feat_dim + j] = feats_in[i * feat_dim + j] / other_in[j];
        }
    }
}

template<typename scalar_t>
__global__ void kernel_packed_matmul(
    // Inputs
    const uint32_t num_packs,
    const uint32_t num_feats, 
    const uint32_t feat_dim, 
    const uint32_t out_feat_dim, 
    const scalar_t* __restrict__ feats_in, // [num_feats, feat_dim]
    const scalar_t* __restrict__ other_in, // [num_packs, out_feat_dim, feat_dim]
    const int64_t* __restrict__ pack_infos, 
    // Outputs
    scalar_t* __restrict__ feats_out // [num_feats, out_feat_dim]
) {
    uint32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= num_packs) return;

    other_in += tidx * out_feat_dim * feat_dim;
    uint32_t begin = pack_infos[tidx * 2];
    uint32_t end = begin + pack_infos[tidx * 2 + 1];
    // For loop on out_feat_dim first.
    for (uint32_t j=0; j<out_feat_dim; ++j) {
        for (uint32_t i=begin; i < end; ++i) {
            scalar_t result = 0;
            for (uint32_t k=0; k<feat_dim; ++k) {
                result += feats_in[i * feat_dim + k] * other_in[j * feat_dim + k];
            }
            feats_out[i * out_feat_dim + j] = result;
        }
    }
}

template<typename scalar_t>
__global__ void kernel_packed_gt(
    // Inputs
    const uint32_t num_packs,
    const uint32_t num_feats, 
    const uint32_t feat_dim, 
    const scalar_t* __restrict__ feats_in, // [num_feats, feat_dim]
    const scalar_t* __restrict__ other_in, // [num_packs, feat_dim]
    const int64_t* __restrict__ pack_infos, 
    // Outputs
    bool* __restrict__ feats_out
) {
    uint32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= num_packs) return;

    other_in += tidx * feat_dim;
    uint32_t begin = pack_infos[tidx * 2];
    uint32_t end = begin + pack_infos[tidx * 2 + 1];
    // For loop on feat_dim first.
    for (uint32_t j=0; j<feat_dim; ++j) {
        for (uint32_t i=begin; i < end; ++i) {
            feats_out[i * feat_dim + j] = feats_in[i * feat_dim + j] > other_in[j];
        }
    }
}

template<typename scalar_t>
__global__ void kernel_packed_geq(
    // Inputs
    const uint32_t num_packs,
    const uint32_t num_feats, 
    const uint32_t feat_dim, 
    const scalar_t* __restrict__ feats_in, // [num_feats, feat_dim]
    const scalar_t* __restrict__ other_in, // [num_packs, feat_dim]
    const int64_t* __restrict__ pack_infos, 
    // Outputs
    bool* __restrict__ feats_out
) {
    uint32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= num_packs) return;

    other_in += tidx * feat_dim;
    uint32_t begin = pack_infos[tidx * 2];
    uint32_t end = begin + pack_infos[tidx * 2 + 1];
    // For loop on feat_dim first.
    for (uint32_t j=0; j<feat_dim; ++j) {
        for (uint32_t i=begin; i < end; ++i) {
            feats_out[i * feat_dim + j] = feats_in[i * feat_dim + j] >= other_in[j];
        }
    }
}

template<typename scalar_t>
__global__ void kernel_packed_lt(
    // Inputs
    const uint32_t num_packs,
    const uint32_t num_feats, 
    const uint32_t feat_dim, 
    const scalar_t* __restrict__ feats_in, // [num_feats, feat_dim]
    const scalar_t* __restrict__ other_in, // [num_packs, feat_dim]
    const int64_t* __restrict__ pack_infos, 
    // Outputs
    bool* __restrict__ feats_out
) {
    uint32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= num_packs) return;

    other_in += tidx * feat_dim;
    uint32_t begin = pack_infos[tidx * 2];
    uint32_t end = begin + pack_infos[tidx * 2 + 1];
    // For loop on feat_dim first.
    for (uint32_t j=0; j<feat_dim; ++j) {
        for (uint32_t i=begin; i < end; ++i) {
            feats_out[i * feat_dim + j] = feats_in[i * feat_dim + j] < other_in[j];
        }
    }
}

template<typename scalar_t>
__global__ void kernel_packed_leq(
    // Inputs
    const uint32_t num_packs,
    const uint32_t num_feats, 
    const uint32_t feat_dim, 
    const scalar_t* __restrict__ feats_in, // [num_feats, feat_dim]
    const scalar_t* __restrict__ other_in, // [num_packs, feat_dim]
    const int64_t* __restrict__ pack_infos, 
    // Outputs
    bool* __restrict__ feats_out
) {
    uint32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= num_packs) return;

    other_in += tidx * feat_dim;
    uint32_t begin = pack_infos[tidx * 2];
    uint32_t end = begin + pack_infos[tidx * 2 + 1];
    // For loop on feat_dim first.
    for (uint32_t j=0; j<feat_dim; ++j) {
        for (uint32_t i=begin; i < end; ++i) {
            feats_out[i * feat_dim + j] = feats_in[i * feat_dim + j] <= other_in[j];
        }
    }
}

template<typename scalar_t>
__global__ void kernel_packed_eq(
    // Inputs
    const uint32_t num_packs,
    const uint32_t num_feats, 
    const uint32_t feat_dim, 
    const scalar_t* __restrict__ feats_in, // [num_feats, feat_dim]
    const scalar_t* __restrict__ other_in, // [num_packs, feat_dim]
    const int64_t* __restrict__ pack_infos, 
    // Outputs
    bool* __restrict__ feats_out
) {
    uint32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= num_packs) return;

    other_in += tidx * feat_dim;
    uint32_t begin = pack_infos[tidx * 2];
    uint32_t end = begin + pack_infos[tidx * 2 + 1];
    // For loop on feat_dim first.
    for (uint32_t j=0; j<feat_dim; ++j) {
        for (uint32_t i=begin; i < end; ++i) {
            feats_out[i * feat_dim + j] = feats_in[i * feat_dim + j] == other_in[j];
        }
    }
}

template<typename scalar_t>
__global__ void kernel_packed_neq(
    // Inputs
    const uint32_t num_packs,
    const uint32_t num_feats, 
    const uint32_t feat_dim, 
    const scalar_t* __restrict__ feats_in, // [num_feats, feat_dim]
    const scalar_t* __restrict__ other_in, // [num_packs, feat_dim]
    const int64_t* __restrict__ pack_infos, 
    // Outputs
    bool* __restrict__ feats_out
) {
    uint32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= num_packs) return;

    other_in += tidx * feat_dim;
    uint32_t begin = pack_infos[tidx * 2];
    uint32_t end = begin + pack_infos[tidx * 2 + 1];
    // For loop on feat_dim first.
    for (uint32_t j=0; j<feat_dim; ++j) {
        for (uint32_t i=begin; i < end; ++i) {
            feats_out[i * feat_dim + j] = feats_in[i * feat_dim + j] != other_in[j];
        }
    }
}

at::Tensor packed_binary_ops(
    at::Tensor feats, // [num_feats, feat_dim] or [num_feats]
    at::Tensor other, // [num_packs, ...]
    at::Tensor pack_infos, // [num_packs, 2]
    PackBinaryOpType op
) {
	at::TensorArg feats_arg(feats, "feats", 1);
	at::TensorArg other_arg(other, "other", 2);
	at::TensorArg pack_infos_arg(pack_infos, "pack_infos", 3);

	at::checkDimRange(__func__, feats_arg, 1, 3); // [1, 2] is allowed.
    at::checkDim(__func__, pack_infos_arg, 2);
    at::checkAllSameGPU(__func__, {feats_arg, other_arg, pack_infos_arg});
    at::checkAllContiguous(__func__, {feats_arg, other_arg, pack_infos_arg});
    at::checkScalarType(__func__, pack_infos_arg, at::kLong);

    at::checkSize(__func__, other_arg, 0, pack_infos.size(0));
    at::checkSize(__func__, feats_arg, 0, pack_infos.index({-1,0}).item<int64_t>() + pack_infos.index({-1,1}).item<int64_t>());

    uint32_t num_packs = pack_infos.size(0);
    uint32_t num_feats = feats.size(0);

    int64_t* pack_infos_ptr = pack_infos.data_ptr<int64_t>();

    uint32_t feat_dim = feats.dim() == 1 ? 1 : feats.size(1);
    uint32_t out_feat_dim;
    at::TensorOptions out_options = feats.options();
    switch (op) {
        case PackBinaryOpType::Add:
        case PackBinaryOpType::Subtract:
        case PackBinaryOpType::Multiply:
        case PackBinaryOpType::Division:
        {
            at::checkSameDim(__func__, feats_arg, other_arg);
            out_feat_dim = feat_dim;
            break;
        }
        case PackBinaryOpType::Matmul:
        {
            // feats [num_feats, feat_dim]
            // other [num_packs, out_feat_dim, feat_dim]
            at::checkDim(__func__, feats_arg, 2);
            at::checkDim(__func__, other_arg, 3);
            at::checkSize(__func__, other_arg, 2, feat_dim);
            out_feat_dim = other.size(1);
            break;
        }
        case PackBinaryOpType::Gt:
        case PackBinaryOpType::Geq:
        case PackBinaryOpType::Lt:
        case PackBinaryOpType::Leq:
        case PackBinaryOpType::Eq:
        case PackBinaryOpType::Neq:
        {
            at::checkSameDim(__func__, feats_arg, other_arg);
            out_feat_dim = feat_dim;
            out_options = out_options.dtype(at::kBool);
        }
        default:
            // Should never happen
            break;
    }

    at::Tensor feats_out = feats.dim() == 1 ? at::zeros({num_feats}, out_options) : at::zeros({num_feats, out_feat_dim}, out_options);
    
    static constexpr uint32_t num_threads = 256;

    const at::cuda::OptionalCUDAGuard device_guard(at::device_of(feats_out));
    auto stream = at::cuda::getCurrentCUDAStream();

    switch (op) {
        case PackBinaryOpType::Add:
        {
            AT_DISPATCH_ALL_TYPES_AND_HALF(feats.scalar_type(), "packed_add", ([&] {
                kernel_packed_add<scalar_t><<<div_round_up((uint32_t)num_packs, num_threads), num_threads, 0, stream>>>(
                    num_packs, num_feats, feat_dim, 
                    feats.data_ptr<scalar_t>(), other.data_ptr<scalar_t>(), pack_infos_ptr, feats_out.data_ptr<scalar_t>()
                );
            }));
        }
        break;

        case PackBinaryOpType::Subtract:
        {
            AT_DISPATCH_ALL_TYPES_AND_HALF(feats.scalar_type(), "packed_sub", ([&] {
                kernel_packed_sub<scalar_t><<<div_round_up((uint32_t)num_packs, num_threads), num_threads, 0, stream>>>(
                    num_packs, num_feats, feat_dim, 
                    feats.data_ptr<scalar_t>(), other.data_ptr<scalar_t>(), pack_infos_ptr, feats_out.data_ptr<scalar_t>()
                );
            }));
        }
        break;

        case PackBinaryOpType::Multiply:
        {
            AT_DISPATCH_ALL_TYPES_AND_HALF(feats.scalar_type(), "packed_mul", ([&] {
                kernel_packed_mul<scalar_t><<<div_round_up((uint32_t)num_packs, num_threads), num_threads, 0, stream>>>(
                    num_packs, num_feats, feat_dim, 
                    feats.data_ptr<scalar_t>(), other.data_ptr<scalar_t>(), pack_infos_ptr, feats_out.data_ptr<scalar_t>()
                );
            }));
        }
        break;

        case PackBinaryOpType::Division:
        {
            AT_DISPATCH_ALL_TYPES_AND_HALF(feats.scalar_type(), "packed_div", ([&] {
                kernel_packed_div<scalar_t><<<div_round_up((uint32_t)num_packs, num_threads), num_threads, 0, stream>>>(
                    num_packs, num_feats, feat_dim, 
                    feats.data_ptr<scalar_t>(), other.data_ptr<scalar_t>(), pack_infos_ptr, feats_out.data_ptr<scalar_t>()
                );
            }));
        }
        break;

        case PackBinaryOpType::Matmul:
        {
            AT_DISPATCH_ALL_TYPES_AND_HALF(feats.scalar_type(), "packed_matmul", ([&] {
                kernel_packed_matmul<scalar_t><<<div_round_up((uint32_t)num_packs, num_threads), num_threads, 0, stream>>>(
                    num_packs, num_feats, feat_dim, out_feat_dim, 
                    feats.data_ptr<scalar_t>(), other.data_ptr<scalar_t>(), pack_infos_ptr, feats_out.data_ptr<scalar_t>()
                );
            }));
        }
        break;

        case PackBinaryOpType::Gt:
        {
            AT_DISPATCH_ALL_TYPES_AND_HALF(feats.scalar_type(), "packed_gt", ([&] {
                kernel_packed_gt<scalar_t><<<div_round_up((uint32_t)num_packs, num_threads), num_threads, 0, stream>>>(
                    num_packs, num_feats, feat_dim, 
                    feats.data_ptr<scalar_t>(), other.data_ptr<scalar_t>(), pack_infos_ptr, feats_out.data_ptr<bool>()
                );
            }));
        }
        break;

        case PackBinaryOpType::Geq:
        {
            AT_DISPATCH_ALL_TYPES_AND_HALF(feats.scalar_type(), "packed_geq", ([&] {
                kernel_packed_geq<scalar_t><<<div_round_up((uint32_t)num_packs, num_threads), num_threads, 0, stream>>>(
                    num_packs, num_feats, feat_dim, 
                    feats.data_ptr<scalar_t>(), other.data_ptr<scalar_t>(), pack_infos_ptr, feats_out.data_ptr<bool>()
                );
            }));
        }
        break;

        case PackBinaryOpType::Lt:
        {
            AT_DISPATCH_ALL_TYPES_AND_HALF(feats.scalar_type(), "packed_lt", ([&] {
                kernel_packed_lt<scalar_t><<<div_round_up((uint32_t)num_packs, num_threads), num_threads, 0, stream>>>(
                    num_packs, num_feats, feat_dim, 
                    feats.data_ptr<scalar_t>(), other.data_ptr<scalar_t>(), pack_infos_ptr, feats_out.data_ptr<bool>()
                );
            }));
        }
        break;

        case PackBinaryOpType::Leq:
        {
            AT_DISPATCH_ALL_TYPES_AND_HALF(feats.scalar_type(), "packed_leq", ([&] {
                kernel_packed_leq<scalar_t><<<div_round_up((uint32_t)num_packs, num_threads), num_threads, 0, stream>>>(
                    num_packs, num_feats, feat_dim, 
                    feats.data_ptr<scalar_t>(), other.data_ptr<scalar_t>(), pack_infos_ptr, feats_out.data_ptr<bool>()
                );
            }));
        }
        break;

        case PackBinaryOpType::Eq:
        {
            AT_DISPATCH_ALL_TYPES_AND_HALF(feats.scalar_type(), "packed_eq", ([&] {
                kernel_packed_eq<scalar_t><<<div_round_up((uint32_t)num_packs, num_threads), num_threads, 0, stream>>>(
                    num_packs, num_feats, feat_dim, 
                    feats.data_ptr<scalar_t>(), other.data_ptr<scalar_t>(), pack_infos_ptr, feats_out.data_ptr<bool>()
                );
            }));
        }
        break;

        case PackBinaryOpType::Neq:
        {
            AT_DISPATCH_ALL_TYPES_AND_HALF(feats.scalar_type(), "packed_neq", ([&] {
                kernel_packed_neq<scalar_t><<<div_round_up((uint32_t)num_packs, num_threads), num_threads, 0, stream>>>(
                    num_packs, num_feats, feat_dim, 
                    feats.data_ptr<scalar_t>(), other.data_ptr<scalar_t>(), pack_infos_ptr, feats_out.data_ptr<bool>()
                );
            }));
        }
        break;

        default:
            // Should never happen
            break;
    }

    return feats_out;
}

at::Tensor packed_add(
    at::Tensor feats, // [num_feats, feat_dim] or [num_feats]
    at::Tensor other, // [num_packs, feat_dim] or [num_packs]
    at::Tensor pack_infos // [num_packs, 2]
) {
    return packed_binary_ops(feats, other, pack_infos, PackBinaryOpType::Add);
}

at::Tensor packed_sub(
    at::Tensor feats, // [num_feats, feat_dim] or [num_feats]
    at::Tensor other, // [num_packs, feat_dim] or [num_packs]
    at::Tensor pack_infos // [num_packs, 2]
) {
    return packed_binary_ops(feats, other, pack_infos, PackBinaryOpType::Subtract);
}

at::Tensor packed_mul(
    at::Tensor feats, // [num_feats, feat_dim] or [num_feats]
    at::Tensor other, // [num_packs, feat_dim] or [num_packs]
    at::Tensor pack_infos // [num_packs, 2]
) {
    return packed_binary_ops(feats, other, pack_infos, PackBinaryOpType::Multiply);
}

at::Tensor packed_div(
    at::Tensor feats, // [num_feats, feat_dim] or [num_feats]
    at::Tensor other, // [num_packs, feat_dim] or [num_packs]
    at::Tensor pack_infos // [num_packs, 2]
) {
    return packed_binary_ops(feats, other, pack_infos, PackBinaryOpType::Division);
}

at::Tensor packed_matmul(
    at::Tensor feats, // [num_feats, feat_dim]
    at::Tensor other, // [num_packs, out_feat_dim, feat_dim]
    at::Tensor pack_infos // [num_packs, 2]
) {
    return packed_binary_ops(feats, other, pack_infos, PackBinaryOpType::Matmul);
}

at::Tensor packed_gt(
    at::Tensor feats, // [num_feats, feat_dim] or [num_feats]
    at::Tensor other, // [num_packs, feat_dim] or [num_packs]
    at::Tensor pack_infos // [num_packs, 2]
) {
    return packed_binary_ops(feats, other, pack_infos, PackBinaryOpType::Gt);
}

at::Tensor packed_geq(
    at::Tensor feats, // [num_feats, feat_dim] or [num_feats]
    at::Tensor other, // [num_packs, feat_dim] or [num_packs]
    at::Tensor pack_infos // [num_packs, 2]
) {
    return packed_binary_ops(feats, other, pack_infos, PackBinaryOpType::Geq);
}

at::Tensor packed_lt(
    at::Tensor feats, // [num_feats, feat_dim] or [num_feats]
    at::Tensor other, // [num_packs, feat_dim] or [num_packs]
    at::Tensor pack_infos // [num_packs, 2]
) {
    return packed_binary_ops(feats, other, pack_infos, PackBinaryOpType::Lt);
}

at::Tensor packed_leq(
    at::Tensor feats, // [num_feats, feat_dim] or [num_feats]
    at::Tensor other, // [num_packs, feat_dim] or [num_packs]
    at::Tensor pack_infos // [num_packs, 2]
) {
    return packed_binary_ops(feats, other, pack_infos, PackBinaryOpType::Leq);
}

at::Tensor packed_eq(
    at::Tensor feats, // [num_feats, feat_dim] or [num_feats]
    at::Tensor other, // [num_packs, feat_dim] or [num_packs]
    at::Tensor pack_infos // [num_packs, 2]
) {
    return packed_binary_ops(feats, other, pack_infos, PackBinaryOpType::Eq);
}

at::Tensor packed_neq(
    at::Tensor feats, // [num_feats, feat_dim] or [num_feats]
    at::Tensor other, // [num_packs, feat_dim] or [num_packs]
    at::Tensor pack_infos // [num_packs, 2]
) {
    return packed_binary_ops(feats, other, pack_infos, PackBinaryOpType::Neq);
}


#if defined(PACK_OPS_USE_THRUST_SORT)

// Used by packed_sort
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

static uint32_t log2_max_num_feats_to_sort = 25; // 32 Mi feats
int set_device_heap_size() {
    // NOTE: Set max device heap size for thrust::sort. 
    // https://stackoverflow.com/a/64446330/11121534
    // 32 Mi * 16 B = 512 MiB GPU mem.
    CUDA_CHECK_THROW(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 16ull * (1 << log2_max_num_feats_to_sort)));
    return 1;
}
static int call_once = set_device_heap_size();

template<typename scalar_t>
__global__ void kernel_packed_sort_thrust(
    // Inputs
    const uint32_t num_packs, 
    const uint32_t num_feats,
    scalar_t* __restrict__ vals, // Will be modified in-place
    int64_t* __restrict__ ids,  // Will be modified in-place
    const int64_t* __restrict__ pack_infos
) {
    uint32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= num_packs) return;

    uint32_t begin = pack_infos[tidx * 2];
    uint32_t end = begin + pack_infos[tidx * 2 + 1];

    if (ids) {
        thrust::sort_by_key(thrust::seq, vals + begin, vals + end, ids + begin);
    } else {
        thrust::sort(thrust::seq, vals + begin, vals + end);
    }
}

at::Tensor packed_sort_thrust(
    at::Tensor vals, // [num_feats]
    at::Tensor pack_infos, // [num_packs, 2]
    bool return_idx
) {
	at::TensorArg vals_arg(vals, "vals", 1);
	at::TensorArg pack_infos_arg(pack_infos, "pack_infos", 2);

    at::checkDim(__func__, vals_arg, 1);
    at::checkDim(__func__, pack_infos_arg, 2);
    at::checkAllSameGPU(__func__, {vals_arg, pack_infos_arg});
    at::checkAllContiguous(__func__, {vals_arg, pack_infos_arg});
    at::checkScalarType(__func__, pack_infos_arg, at::kLong);
    at::checkSize(__func__, vals_arg, 0, pack_infos.index({-1,0}).item<int64_t>() + pack_infos.index({-1,1}).item<int64_t>());

    uint32_t num_packs = pack_infos.size(0);
    uint32_t num_feats = vals.size(0);

    if (num_feats > (1 << log2_max_num_feats_to_sort)) {
        throw std::runtime_error(std::string("Can only handle number of feats less than ") + std::to_string(1 << log2_max_num_feats_to_sort) + std::string(", while current = ") + std::to_string(num_feats));
    }

    at::Tensor idx;
    if (return_idx) {
        idx = at::arange((int64_t)num_feats, pack_infos.options());
    }
    
    static constexpr uint32_t num_threads = 256;

    AT_DISPATCH_ALL_TYPES_AND_HALF(vals.scalar_type(), "packed_sort_thrust", ([&] {
        const at::cuda::OptionalCUDAGuard device_guard(at::device_of(vals));
        auto stream = at::cuda::getCurrentCUDAStream();
        kernel_packed_sort_thrust<scalar_t><<<div_round_up((uint32_t)num_packs, num_threads), num_threads, 0, stream>>>(
            num_packs, num_feats, 
            vals.data_ptr<scalar_t>(), 
            return_idx ? idx.data_ptr<int64_t>() : nullptr, 
            pack_infos.data_ptr<int64_t>()
        );
    }));

    return idx;
}

#else

at::Tensor packed_sort_thrust(
    at::Tensor vals, // [num_feats]
    at::Tensor pack_infos, // [num_packs, 2]
    bool return_idx
) {
	throw std::runtime_error("pack_ops:: Please set PACK_OPS_USE_THRUST_SORT to use thrust::sort.");
}

#endif


template<typename scalar_t>
inline __device__ int64_t qsort_partition(
    scalar_t* __restrict__ vals, 
    int64_t* __restrict__ ids, 
    int64_t l, 
    int64_t h
) {
    // Index of smaller element
    int64_t i = l - 1;
    scalar_t pivot = vals[h];
    if (ids) {
        for (uint32_t j=l; j<h; ++j) {
            // If current element is smaller than or equal to pivot
            if (vals[j] <= pivot) {
                // Increment index of smaller element
                i++;
                { scalar_t tmp = vals[j]; vals[j] = vals[i]; vals[i] = tmp; }
                { int64_t tmp = ids[j]; ids[j] = ids[i]; ids[i] = tmp; }
            }
        }
        { scalar_t tmp = vals[i+1]; vals[i+1] = vals[h]; vals[h] = tmp; }
        { int64_t tmp = ids[i+1]; ids[i+1] = ids[h]; ids[h] = tmp; }
    } else {
        for (uint32_t j=l; j<h; ++j) {
            // If current element is smaller than or equal to pivot
            if (vals[j] <= pivot) {
                // Increment index of smaller element
                i++;
                { scalar_t tmp = vals[j]; vals[j] = vals[i]; vals[i] = tmp; }
            }
        }
        { scalar_t tmp = vals[i+1]; vals[i+1] = vals[h]; vals[h] = tmp; }
    }
    return i+1;
}

template<typename scalar_t>
__global__ void kernel_packed_sort_qsort(
    // Inputs
    const uint32_t num_packs, 
    const uint32_t num_feats,
    scalar_t* __restrict__ vals, // [num_feats] Will be modified in-place
    int64_t* __restrict__ stack, // [num_feats] Init as zeros; will be modified in-place
    int64_t* __restrict__ ids,  // [num_feats] Will be modified in-place
    const int64_t* __restrict__ pack_infos
) {
    // Modified from https://github.com/numba/numba/issues/4283#issuecomment-908511793
    uint32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= num_packs) return;

    uint32_t begin = pack_infos[tidx * 2];
    uint32_t num = pack_infos[tidx * 2 + 1];

    vals += begin;
    stack += begin;
    if (ids) ids += begin;

    // Low and high indices.
    int64_t l = 0, h = num - 1;
    // Initialize top of stack
    int64_t top = -1;
    // Push initial values of l and h to stack
    stack[++top] = l;
    stack[++top] = h;

    // Keep popping from stack while is not empty
    while (top >= 0) {
        // Pop h and l
        h = stack[top--];
        l = stack[top--];

        // Set pivot element at its correct position in sorted array
        int64_t p = qsort_partition<scalar_t>(vals, ids, l, h);

        // If there are elements on left side of pivot, then push left side to stack
        if (p-1 > l) {
            stack[++top] = l;
            stack[++top] = p-1;
        }

        // If there are elements on right side of pivot, then push right side to stack
        if (p+1 < h) {
            stack[++top] = p + 1;
            stack[++top] = h;
        }
    }
}

at::Tensor packed_sort_qsort(
    at::Tensor vals, // [num_feats]
    at::Tensor pack_infos, // [num_packs, 2]
    bool return_idx
) {
	at::TensorArg vals_arg(vals, "vals", 1);
	at::TensorArg pack_infos_arg(pack_infos, "pack_infos", 2);

    at::checkDim(__func__, vals_arg, 1);
    at::checkDim(__func__, pack_infos_arg, 2);
    at::checkAllSameGPU(__func__, {vals_arg, pack_infos_arg});
    at::checkAllContiguous(__func__, {vals_arg, pack_infos_arg});
    at::checkScalarType(__func__, pack_infos_arg, at::kLong);

    at::checkSize(__func__, vals_arg, 0, pack_infos.index({-1,0}).item<int64_t>() + pack_infos.index({-1,1}).item<int64_t>());

    uint32_t num_packs = pack_infos.size(0);
    uint32_t num_feats = vals.size(0);

    at::Tensor idx;
    if (return_idx) {
        idx = at::arange((int64_t)num_feats, pack_infos.options());
    }

    at::Tensor stack = at::zeros({num_feats}, pack_infos.options());
    
    static constexpr uint32_t num_threads = 256;

    AT_DISPATCH_ALL_TYPES_AND_HALF(vals.scalar_type(), "packed_sort_qsort", ([&] {
        const at::cuda::OptionalCUDAGuard device_guard(at::device_of(vals));
        auto stream = at::cuda::getCurrentCUDAStream();
        kernel_packed_sort_qsort<scalar_t><<<div_round_up((uint32_t)num_packs, num_threads), num_threads, 0, stream>>>(
            num_packs, num_feats, 
            vals.data_ptr<scalar_t>(), 
            stack.data_ptr<int64_t>(), 
            return_idx ? idx.data_ptr<int64_t>() : nullptr, 
            pack_infos.data_ptr<int64_t>()
        );
    }));

    return idx;
}

template<typename scalar_t>
__global__ void
mark_pack_boundaries_cuda_kernel(
    const int64_t num, 
    const scalar_t* __restrict__ pack_ids, 
    uint* __restrict__ boundaries
) {
    // Borrowed from https://github.com/NVIDIAGameWorks/kaolin
    uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

    if (tidx < num) {
        if (tidx == 0) {
            boundaries[tidx] = 1;
        } else {
            boundaries[tidx] = pack_ids[tidx - 1] == pack_ids[tidx] ? 0 : 1;
        }
    }
}

at::Tensor mark_pack_boundaries_cuda(
    at::Tensor pack_ids
) {
    // Borrowed from https://github.com/NVIDIAGameWorks/kaolin
    at::TensorArg pack_ids_arg{pack_ids, "pack_ids", 1};
    at::checkDim(__func__, pack_ids_arg, 1);
    at::checkAllSameGPU(__func__, {pack_ids_arg});
    at::checkAllContiguous(__func__,  {pack_ids_arg});
    at::checkScalarTypes(__func__, pack_ids_arg, {at::kByte, at::kChar, at::kInt, at::kLong, at::kShort});
    int num_ids = pack_ids.size(0);
    at::Tensor boundaries = at::zeros({num_ids}, pack_ids.options().dtype(at::kInt));
    static constexpr uint32_t num_threads = 1024;
    AT_DISPATCH_INTEGRAL_TYPES(pack_ids.type(), "mark_pack_boundaries_cuda", ([&] {
        const at::cuda::OptionalCUDAGuard device_guard(at::device_of(boundaries));
        auto stream = at::cuda::getCurrentCUDAStream();
        mark_pack_boundaries_cuda_kernel<<<(num_ids + num_threads - 1) / num_threads, num_threads, 0, stream>>>(
            num_ids,
            pack_ids.data_ptr<scalar_t>(),
            reinterpret_cast<uint*>(boundaries.data_ptr<int>()));
    }));
    return boundaries;
}

__global__ void kernel_mark_consecutive_segments(
    // Inputs
    const uint32_t num_packs,
    const uint32_t num_nuggets,
    const int64_t* __restrict__ pack_infos,
    const int32_t* __restrict__ point_indices,
    const short3* __restrict__ point_hiearchies,
    // Outputs
    bool* __restrict__ mark_start,
    bool* __restrict__ mark_end
) {
    const uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidx >= num_packs) return;

    const uint32_t begin = pack_infos[tidx * 2];
    const uint32_t length = pack_infos[tidx * 2 + 1];

    mark_start += begin;
    mark_end += begin;

    short3 prev_point, next_point;
    mark_start[0] = true; // The first nugget in the pack must be the starting nugget.

    // Loop over every nugget in this pack
    for (uint32_t j=1; j < length; ++j) {
        prev_point = point_hiearchies[point_indices[j-1]];
        next_point = point_hiearchies[point_indices[j]];
        if ((abs(next_point.x-prev_point.x) + abs(next_point.y-prev_point.y) + abs(next_point.z-prev_point.z)) > 1) {
            // Changed, not consecutive
            mark_end[j-1] = true;
            mark_start[j] = true;
        }
    }
    mark_end[length-1] = true; // The last nugget in the pack must be the ending nugget.
}

std::tuple<at::Tensor, at::Tensor> octree_mark_consecutive_segments(
    at::Tensor pidx, // [num_nuggets]
    at::Tensor pack_infos, // [num_packs, 2]
    at::Tensor point_hierarchies // [num_active_voxels (from kaolin.rep.spc)]
) {
    at::TensorArg pidx_arg{pidx, "pidx", 1};
    at::TensorArg pack_infos_arg{pack_infos, "pack_infos", 2};
    at::TensorArg point_hierarchies_arg{point_hierarchies, "point_hierarchies", 3};

    at::checkDim(__func__, pidx_arg, 1);
    at::checkDim(__func__, pack_infos_arg, 2);
    at::checkAllSameGPU(__func__, {pidx_arg, pack_infos_arg, point_hierarchies_arg});
    at::checkAllContiguous(__func__, {pidx_arg, pack_infos_arg, point_hierarchies_arg});
    at::checkScalarTypes(__func__, pidx_arg, at::kInt);
    at::checkScalarTypes(__func__, pack_infos_arg, at::kLong);
    at::checkScalarTypes(__func__, point_hierarchies_arg, at::kShort);

    at::checkSize(__func__, pidx_arg, 0, pack_infos.index({-1,0}).item<int64_t>() + pack_infos.index({-1,1}).item<int64_t>());

    uint32_t num_nuggets = pidx.size(0);
    uint32_t num_packs = pack_infos.size(0);
    // uint32_t depth_dim = depths.size(1);

    int32_t* pidx_ptr = pidx.data_ptr<int32_t>();
    int64_t* pack_infos_ptr = pack_infos.data_ptr<int64_t>();
    short3* points_ptr = reinterpret_cast<short3*>(point_hierarchies.data_ptr<short>());

    at::Tensor mark_start = at::zeros({ num_nuggets }, pidx.options().dtype(at::kBool));
    at::Tensor mark_end = at::zeros({ num_nuggets }, pidx.options().dtype(at::kBool));
    bool* mark_start_ptr = mark_start.data_ptr<bool>();
    bool* mark_end_ptr = mark_end.data_ptr<bool>();

    static constexpr uint32_t num_threads = 256;
    const at::cuda::OptionalCUDAGuard device_guard(at::device_of(pidx));
    auto stream = at::cuda::getCurrentCUDAStream();
    kernel_mark_consecutive_segments<<<div_round_up(num_packs, num_threads), num_threads, 0, stream>>>(
        num_packs, num_nuggets, 
        pack_infos_ptr, pidx_ptr, points_ptr, 
        mark_start_ptr, mark_end_ptr
    );

    return {mark_start, mark_end};

    // at::Tensor ridx_out = ridx.masked_select(mark_start);
}

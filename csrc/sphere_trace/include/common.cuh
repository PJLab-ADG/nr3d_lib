#pragma once
#include <ATen/ATen.h>
#include <glm/glm.hpp>

#ifdef __NVCC__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

constexpr uint32_t n_threads_linear = 128;

template <typename T>
HOST_DEVICE T div_round_up(T val, T divisor) {
	return (val + divisor - 1) / divisor;
}

template <typename T>
constexpr uint32_t n_blocks_linear(T n_elements) {
	return (uint32_t)div_round_up(n_elements, (T)n_threads_linear);
}

#ifdef __NVCC__
template <typename K, typename T, typename ... Types>
inline void linear_kernel(K kernel, uint32_t shmem_size, cudaStream_t stream, T n_elements, Types ... args) {
	if (n_elements <= 0) {
		return;
	}
	kernel<<<n_blocks_linear(n_elements), n_threads_linear, shmem_size, stream>>>(n_elements, args...);
}
#endif
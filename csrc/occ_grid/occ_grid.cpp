/** @file   occ_grid.cpp
 *  @author Jianfei Guo, Shanghai AI Lab
 *  @brief  Occupancy grid marching operations.
 *  Modified from https://github.com/KAIR-BAIR/nerfacc
 * 	Copyright (c) 2022 Ruilong Li, UC Berkeley.
 */

#ifdef _MSC_VER
#pragma warning(push, 0)
#include <torch/extension.h>
#pragma warning(pop)
#else
#include <torch/extension.h>
#endif

#ifdef snprintf
#undef snprintf
#endif

#include <occ_grid/cpp_api.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	py::enum_<ContractionType>(m, "ContractionType", py::module_local(true))
		.value("AABB", ContractionType::AABB)
		.value("UN_BOUNDED_TANH", ContractionType::UN_BOUNDED_TANH)
		.value("UN_BOUNDED_SPHERE", ContractionType::UN_BOUNDED_SPHERE)
		.export_values()
		;

    m.def("ray_marching", &ray_marching, "ray_marching on a single block");
    m.def("batched_ray_marching", &batched_ray_marching, "ray_marching on batched blocks");
    // m.def("forest_ray_marching", &forest_ray_marching, "ray_marching on a forest of blocks");
}

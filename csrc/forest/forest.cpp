/** @file   forest.cpp
 *  @author Jianfei Guo, Shanghai AI Lab
 *  @brief  Forest of blocks API bindings.
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

#include "forest_cpp_api.h"
#include "kaolin_spc_raytrace_fixed/raytrace.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<ForestMeta>(m, "ForestMeta")
        .def(py::init<>())
        .def_readwrite("octree", &ForestMeta::octree)
        .def_readwrite("exsum", &ForestMeta::exsum)
        .def_readwrite("block_ks", &ForestMeta::block_ks)
        .def_readwrite("world_block_size", &ForestMeta::world_block_size)
        .def_readwrite("world_origin", &ForestMeta::world_origin)
        .def_readwrite("resolution", &ForestMeta::resolution)
        .def_readwrite("n_trees", &ForestMeta::n_trees)
        .def_readwrite("level", &ForestMeta::level)
        .def_readwrite("level_poffset", &ForestMeta::level_poffset)
        .def_readwrite("continuity_enabled", &ForestMeta::continuity_enabled)
        ;
    m.def("raytrace_cuda_fixed", &kaolin::raytrace_cuda_fixed);
}
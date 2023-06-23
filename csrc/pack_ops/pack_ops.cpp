/** @file   pack_ops.cpp
 *  @author Jianfei Guo, Shanghai AI Lab
 *  @brief  Pack ops Pytorch bindings.
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

#include "pack_ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("interleave_arange", py::overload_cast<at::Tensor, bool>(&interleave_arange));
    m.def("interleave_linstep", py::overload_cast<at::Tensor, at::Tensor, at::Tensor, bool>(&interleave_linstep));
    m.def("interleave_linstep", py::overload_cast<at::Tensor, at::Tensor, double, bool>(&interleave_linstep));
    m.def("interleave_linstep", py::overload_cast<at::Tensor, at::Tensor, int32_t, bool>(&interleave_linstep));
    m.def("interleave_sample_step_wrt_depth_clamp_deprecated", &interleave_sample_step_wrt_depth_clamp_deprecated);
    m.def("interleave_sample_step_wrt_depth_clamped", &interleave_sample_step_wrt_depth_clamped);
    m.def("interleave_sample_step_wrt_depth_in_packed_segments", &interleave_sample_step_wrt_depth_in_packed_segments);

    m.def("packed_add", &packed_add);
    m.def("packed_sub", &packed_sub);
    m.def("packed_mul", &packed_mul);
    m.def("packed_div", &packed_div);
    m.def("packed_matmul", &packed_matmul);

    m.def("packed_gt", &packed_gt);
    m.def("packed_geq", &packed_geq);
    m.def("packed_lt", &packed_lt);
    m.def("packed_leq", &packed_leq);
    m.def("packed_eq", &packed_eq);
    m.def("packed_neq", &packed_neq);

    m.def("packed_sum", &packed_sum);
    m.def("packed_diff", &packed_diff);
    m.def("packed_backward_diff", &packed_backward_diff);
    m.def("packed_cumsum", &packed_cumsum);
    m.def("packed_cumprod", &packed_cumprod);

    m.def("packed_sort_qsort", &packed_sort_qsort);
    m.def("packed_sort_thrust", &packed_sort_thrust);
    m.def("packed_searchsorted", &packed_searchsorted);
    m.def("packed_searchsorted_packed_vals", &packed_searchsorted_packed_vals);
    m.def("try_merge_two_packs_sorted_aligned", &try_merge_two_packs_sorted_aligned);
    m.def("packed_invert_cdf", &packed_invert_cdf);
    m.def("packed_alpha_to_vw_forward", &packed_alpha_to_vw_forward);
    m.def("packed_alpha_to_vw_backward", &packed_alpha_to_vw_backward);

    m.def("mark_pack_boundaries_cuda", &mark_pack_boundaries_cuda);
    m.def("octree_mark_consecutive_segments", &octree_mark_consecutive_segments);
}
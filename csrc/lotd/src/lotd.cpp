/** @file   lotd.cpp
 *  @author Jianfei Guo, Shanghai AI Lab
 *  @brief  LoTD Pytorch bindings.
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

// #include <json/json.hpp>
// #include <pybind11_json/pybind11_json.hpp>

#include <lotd/lotd_torch_api.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#define LOTD_OPTIONAL_ARGS \
	py::arg("batch_inds")=nullptr, \
    py::arg("batch_offsets")=nullptr, \
	py::arg("batch_data_size")=nullptr, \
    py::arg("max_level")=nullptr

    // For single-block LoTD and batched LoTD
    m.def("lod_fwd", &lotd::torch::lod_fwd, py::arg("lod_meta"), py::arg("input"), py::arg("params"), LOTD_OPTIONAL_ARGS, py::arg("need_input_grad")=nullptr);
    m.def("lod_bwd", &lotd::torch::lod_bwd, py::arg("lod_meta"), py::arg("dL_dy"), py::arg("input"), py::arg("params"), py::arg("dy_dx")=nullptr, LOTD_OPTIONAL_ARGS, py::arg("need_input_grad")=nullptr, py::arg("need_param_grad")=nullptr);
    m.def("lod_bwd_bwd_input", &lotd::torch::lod_bwd_bwd_input, py::arg("lod_meta"), py::arg("dL_ddLdx"), py::arg("dL_dy"), py::arg("input"), py::arg("params"), py::arg("dy_dx")=nullptr, LOTD_OPTIONAL_ARGS, py::arg("need_dLdinput_ddLdoutput")=nullptr, py::arg("need_dLdinput_dparams")=nullptr, py::arg("need_dLdinput_dinput")=nullptr);
    m.def("lod_get_grid_index", &lotd::torch::lod_get_grid_index, py::arg("lod_meta"), py::arg("input"), LOTD_OPTIONAL_ARGS);

#undef LOTD_OPTIONAL_ARGS

	py::enum_<lotd::LoDType>(m, "LoDType", py::module_local(true))
		.value("Dense", lotd::LoDType::Dense)
		.value("VectorMatrix", lotd::LoDType::VectorMatrix)
		.value("CP", lotd::LoDType::CP)
		.value("CPfast", lotd::LoDType::CPfast)
		.value("NPlaneMul", lotd::LoDType::NPlaneMul)
		.value("NPlaneSum", lotd::LoDType::NPlaneSum)
		.value("Hash", lotd::LoDType::Hash)
		.export_values()
		;

    py::enum_<lotd::InterpolationType>(m, "InterpolationType", py::module_local(true))
        .value("Linear", lotd::InterpolationType::Linear)
        .value("Smoothstep", lotd::InterpolationType::Smoothstep)
        .export_values()
        ;

    py::class_<lotd::torch::LoDMeta>(m, "LoDMeta")
        .def(
            py::init<const int32_t, const std::vector<int32_t>&, const std::vector<int32_t>&, const std::vector<std::string>&, at::optional<uint32_t>, at::optional<bool>>(), "Create lod meta", 
            py::arg("n_input_dims"), py::arg("lod_res"), py::arg("lod_n_feats"), py::arg("lod_types"), py::arg("hashmap_size")=nullptr, py::arg("use_smooth_step")=nullptr
        )
        .def(
            py::init<const int32_t, const std::vector<std::vector<int32_t>>&, const std::vector<int32_t>&, const std::vector<std::string>&, at::optional<uint32_t>, at::optional<bool>>(), "Create lod meta", 
            py::arg("n_input_dims"), py::arg("lod_res_multidim"), py::arg("lod_n_feats"), py::arg("lod_types"), py::arg("hashmap_size")=nullptr, py::arg("use_smooth_step")=nullptr
        )
        .def_readonly("level_res", &lotd::torch::LoDMeta::level_res)
        .def_readonly("level_res_multi_dim", &lotd::torch::LoDMeta::level_res_multi_dim)
        .def_readonly("level_n_params", &lotd::torch::LoDMeta::level_n_params)
        .def_readonly("level_n_feats", &lotd::torch::LoDMeta::level_n_feats)
        .def_readonly("level_types", &lotd::torch::LoDMeta::level_types)
        .def_readonly("level_sizes", &lotd::torch::LoDMeta::level_sizes)
        .def_readonly("level_types_str", &lotd::torch::LoDMeta::level_types_str)
        .def_readonly("level_offsets", &lotd::torch::LoDMeta::level_offsets)
        .def_readonly("map_levels", &lotd::torch::LoDMeta::map_levels)
        .def_readonly("map_cnt", &lotd::torch::LoDMeta::map_cnt)

        .def_readonly("n_levels", &lotd::torch::LoDMeta::n_levels)
        .def_readonly("n_pseudo_levels", &lotd::torch::LoDMeta::n_pseudo_levels)
        .def_readonly("n_feat_per_pseudo_lvl", &lotd::torch::LoDMeta::n_feat_per_pseudo_lvl)
        .def_readonly("n_dims_to_encode", &lotd::torch::LoDMeta::n_dims_to_encode)
        .def_readonly("n_encoded_dims", &lotd::torch::LoDMeta::n_encoded_dims)
        .def_readonly("n_params", &lotd::torch::LoDMeta::n_params)

        .def_readonly("interpolation_type", &lotd::torch::LoDMeta::interpolation_type)
        ;
}
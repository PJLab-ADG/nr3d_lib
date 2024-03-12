/** @file   entry.cu
 *  @author Nianchen Deng, Shanghai AI Lab
 *  @brief  The PyTorch entry of sphere trace module.
 */
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <sphere_trace/dense_grid.cuh>
#include <sphere_trace/sphere_tracer.cuh>
#include <sphere_trace/ray_march.cuh>

using namespace pybind11::literals;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::enum_<RayStatus>(m, "RayStatus")
        .value("ALIVE", RayStatus::ALIVE)
        .value("HIT", RayStatus::HIT)
        .value("OUT", RayStatus::OUT)
        .export_values();

    py::class_<DenseGrid>(m, "DenseGrid")
        .def(py::init([](int x, int y, int z, at::Tensor grid_occ) {
            return new DenseGrid({x, y, z}, grid_occ.data_ptr<bool>());
        }))
        .def_property_readonly("res", [](DenseGrid *self) {
            return std::make_tuple(self->res().x, self->res().y, self->res().z);
        });

    py::class_<SphereTracer>(m, "SphereTracer")
        .def(py::init<float, float, float, float>(), "min_step"_a, "distance_scale"_a,
             "zero_offset"_a = 0.0f, "hit_threshold"_a = 0.001f)
        .def("init_rays", &SphereTracer::init_rays, "rays_o"_a, "rays_d"_a, "valid_rays_idx"_a,
             "segs_pack_info"_a, "segs"_a, "segs_endpoint_distances"_a = nullptr)
        .def("compact_rays", &SphereTracer::compact_rays)
        .def("advance_rays", &SphereTracer::advance_rays, "distances"_a)
        .def("get_rays", &SphereTracer::get_rays, "status"_a)
        .def("get_trace_positions", &SphereTracer::get_trace_positions)
        .def("sample_on_segments", &SphereTracer::sample_on_segments, "step_size"_a)
        .def("trace_on_samples", &SphereTracer::trace_on_samples, "rays_samples_offset"_a,
             "rays_n_samples"_a, "rays_sample_depths"_a, "rays_sample_distances"_a)
        .def("trace", &SphereTracer::trace, "rays_o"_a, "rays_d"_a, "distance_function"_a,
             "max_steps_between_compact"_a, "max_march_iters"_a, "valid_rays_idx"_a,
             "segs_pack_info"_a, "segs"_a, "segs_endpoint_distances"_a = nullptr);

    m.def("ray_march", &ray_march, "grid"_a, "rays_o"_a, "rays_d"_a, "rays_near"_a, "rays_far"_a,
          "return_pts"_a = false, "enable_debug"_a = false);
}

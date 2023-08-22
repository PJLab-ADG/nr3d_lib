#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <dense_grid.cuh>
#include <sphere_tracer.cuh>

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
        }));

    py::class_<SphereTracer>(m, "SphereTracer")
        .def(py::init<bool>(), "debug"_a = false)
        .def("init_rays", &SphereTracer::init_rays, "rays_o"_a, "rays_d"_a, "rays_idx"_a, "near"_a,
             "far"_a, "grid"_a)
        .def("compact_rays", &SphereTracer::compact_rays)
        .def("advance_rays", &SphereTracer::advance_rays, "distances"_a, "zero_offset"_a,
             "distance_scale"_a, "hit_threshold"_a, "hit_at_neg"_a, "grid"_a)
        .def("get_rays", &SphereTracer::get_rays, "status"_a)
        .def("get_trace_positions", &SphereTracer::get_trace_positions)
        .def("trace", &SphereTracer::trace, "rays_o"_a, "rays_d"_a, "rays_idx"_a, "near"_a, "far"_a,
             "distance_function"_a, "zero_offset"_a, "distance_scale"_a, "hit_threshold"_a,
             "hit_at_neg"_a, "max_steps_between_compact"_a, "max_march_iters"_a, "grid"_a);
}

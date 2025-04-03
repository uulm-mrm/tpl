#include "tplcpp/python/bindings_dp_common.cuh"

#include "tplcpp/dyn_prog/common.cuh"

namespace py = pybind11;

void loadDynProgCommonBindings(py::module& m) {

    py::class_<RefLinePoint>(m, "RefLinePoint")
        .def(py::init<>())
        .def_readwrite("x", &RefLinePoint::x)
        .def_readwrite("y", &RefLinePoint::y)
        .def_readwrite("heading", &RefLinePoint::heading)
        .def_readwrite("s", &RefLinePoint::s)
        .def_readwrite("k", &RefLinePoint::k)
        .def_readwrite("v_max", &RefLinePoint::v_max)
        .def_readwrite("d_left", &RefLinePoint::d_left)
        .def_readwrite("d_right", &RefLinePoint::d_right)
        .def_readwrite("semantic", &RefLinePoint::semantic);

    py::class_<RefLine>(m, "RefLine")
        .def(py::init<>())
        .def_static("fromarray", [](array_like<double> arr,
                                    float step_size,
                                    bool no_offset) {
            assert_shape(arr, {{-1, 9}});
            return RefLine((RefLinePoint*)arr.data(), arr.shape(0), step_size, no_offset);
        },
        py::arg("arr"),
        py::arg("step_size"),
        py::arg("no_offset") = false)
        .def_readwrite("points", &RefLine::points)
        .def_readwrite("step_size", &RefLine::step_size)
        .def_readwrite("x_offset", &RefLine::x_offset)
        .def_readwrite("y_offset", &RefLine::y_offset)
        .def("lerp", &RefLine::lerp);

    m.def("cuda_get_device_count", []() { 
        int res = -1;
        cudaError_t err = cudaGetDeviceCount(&res);
        if (err) {
            return 0;
        }
        return res;
    });

    m.def("cuda_set_device", [](int idx) {
        cudaSetDevice(idx);
    });
}

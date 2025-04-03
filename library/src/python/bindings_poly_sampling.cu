#include "tplcpp/python/bindings_poly_sampling.cuh"

#include "tplcpp/poly_sampling.hpp"

namespace py = pybind11;

void loadPolySamplingBindings(py::module& m) {

    py::class_<PolySamplingParams>(m, "PolySamplingParams")
        .def(py::init())
        .def("copy", [](PolySamplingParams& tp) { return tp; }, py::return_value_policy::copy)
        .def("__deepcopy__", [](PolySamplingParams& tp, py::object& memo) { return tp; }, py::return_value_policy::copy)
        .def_readwrite("lane_width", &PolySamplingParams::lane_width)
        .def_readwrite("d_step", &PolySamplingParams::d_step)
        .def_readwrite("T_min", &PolySamplingParams::T_min)
        .def_readwrite("T_max", &PolySamplingParams::T_max)
        .def_readwrite("T_step", &PolySamplingParams::T_step)
        .def_readwrite("dt", &PolySamplingParams::dt)
        .def_readwrite("v_samples", &PolySamplingParams::v_samples)
        .def_readwrite("v_step", &PolySamplingParams::v_step)
        .def_readwrite("trg_d", &PolySamplingParams::trg_d)
        .def_readwrite("k_j", &PolySamplingParams::k_j)
        .def_readwrite("k_t", &PolySamplingParams::k_t)
        .def_readwrite("k_d", &PolySamplingParams::k_d)
        .def_readwrite("k_v", &PolySamplingParams::k_v)
        .def_readwrite("k_lat", &PolySamplingParams::k_lat)
        .def_readwrite("k_lon", &PolySamplingParams::k_lon)
        .def_readwrite("k_overtake_right", &PolySamplingParams::k_overtake_right)
        .def_readwrite("a_max", &PolySamplingParams::a_max)
        .def_readwrite("k_max", &PolySamplingParams::k_max)
        .def_readwrite("rear_axis_to_rear", &PolySamplingParams::rear_axis_to_rear)
        .def_readwrite("rear_axis_to_front", &PolySamplingParams::rear_axis_to_front)
        .def_readwrite("width_ego", &PolySamplingParams::width_ego)
        .def_property_readonly("__slots__", [](PolySamplingParams& p){
            return std::vector<std::string>({
                "lane_width",
                "d_step",
                "T_min",
                "T_max",
                "T_step",
                "dt",
                "v_samples",
                "v_step",
                "trg_d",
                "k_j",
                "k_t",
                "k_d",
                "k_v",
                "k_lat",
                "k_lon",
                "k_overtake_right",
                "a_max",
                "k_max",
                "rear_axis_to_rear",
                "rear_axis_to_front",
                "width_ego",
            });
        });

    py::class_<PolySamplingTrajPoint>(m, "PolySamplingTrajPoint")
        .def(py::init())
        .def("copy", [](PolySamplingTrajPoint& tp) { return tp; }, py::return_value_policy::copy)
        .def("__deepcopy__", [](PolySamplingTrajPoint& tp, py::object& memo) { return tp; }, py::return_value_policy::copy)
        .def_readwrite("t", &PolySamplingTrajPoint::t)
        .def_readwrite("d", &PolySamplingTrajPoint::d)
        .def_readwrite("d_d", &PolySamplingTrajPoint::d_d)
        .def_readwrite("d_dd", &PolySamplingTrajPoint::d_dd)
        .def_readwrite("d_ddd", &PolySamplingTrajPoint::d_ddd)
        .def_readwrite("s", &PolySamplingTrajPoint::s)
        .def_readwrite("s_d", &PolySamplingTrajPoint::s_d)
        .def_readwrite("s_dd", &PolySamplingTrajPoint::s_dd)
        .def_readwrite("s_ddd", &PolySamplingTrajPoint::s_ddd)
        .def_readwrite("x", &PolySamplingTrajPoint::x)
        .def_readwrite("y", &PolySamplingTrajPoint::y)
        .def_readwrite("yaw", &PolySamplingTrajPoint::yaw)
        .def_readwrite("ds", &PolySamplingTrajPoint::ds)
        .def_readwrite("c", &PolySamplingTrajPoint::c);

    py::class_<PolySamplingTraj>(m, "PolySamplingTraj")
        .def(py::init())
        .def_readwrite("points", &PolySamplingTraj::points)
        .def_readwrite("cd", &PolySamplingTraj::cd)
        .def_readwrite("cv", &PolySamplingTraj::cv)
        .def_readwrite("cf", &PolySamplingTraj::cf);

    py::class_<PolySamplingPlanner>(m, "PolySamplingPlanner")
        .def(py::init())
        .def("update", [](PolySamplingPlanner& planner,
                          PolySamplingTrajPoint& c, 
                          array_like<double> ref_line_arr,
                          PolySamplingParams& params) {

            assert_shape(ref_line_arr, {{-1, 6}});

            return planner.update(c,
                                  (PolyRefLinePoint*)ref_line_arr.mutable_data(),
                                  ref_line_arr.shape(0),
                                  params);
        })
        .def("add_obstacle", &PolySamplingPlanner::addObstacle);

}

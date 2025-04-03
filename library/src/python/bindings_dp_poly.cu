#include "tplcpp/python/bindings_dp_poly.cuh"

#include "tplcpp/dyn_prog/poly_planner.cuh"

namespace py = pybind11;

void loadDynProgPolyPlannerBindings(py::module& m) {

    py::class_<DynProgPolyNode>(m, "DynProgPolyNode")
        .def(py::init())
        .def_readwrite("point", &DynProgPolyNode::point);

    py::class_<DynProgPolyPoint>(m, "DynProgPolyPoint")
        .def(py::init())
        .def_readwrite("t", &DynProgPolyPoint::t)
        .def_readwrite("s", &DynProgPolyPoint::s)
        .def_readwrite("ds", &DynProgPolyPoint::ds)
        .def_readwrite("dds", &DynProgPolyPoint::dds)
        .def_readwrite("l", &DynProgPolyPoint::l)
        .def_readwrite("dl", &DynProgPolyPoint::dl)
        .def_readwrite("ddl", &DynProgPolyPoint::ddl)
        .def_readwrite("cost", &DynProgPolyPoint::cost)
        .def("as_numpy", [](DynProgPolyPoint& p) {
            py::array_t<float> arr({6});
            p.writeToArray(arr.mutable_data(), 0);
            return arr;
        });

    py::class_<DynProgPolyCartPoint>(m, "DynProgPolyCartPoint")
        .def_readwrite("t", &DynProgPolyCartPoint::t)
        .def_readwrite("distance", &DynProgPolyCartPoint::distance)
        .def_readwrite("x", &DynProgPolyCartPoint::x)
        .def_readwrite("y", &DynProgPolyCartPoint::y)
        .def_readwrite("heading", &DynProgPolyCartPoint::heading)
        .def_readwrite("k", &DynProgPolyCartPoint::k)
        .def_readwrite("v", &DynProgPolyCartPoint::v)
        .def_readwrite("a", &DynProgPolyCartPoint::a)
        .def("as_numpy", [](DynProgPolyCartPoint& p) {
            py::array_t<double> arr({8});
            p.writeToArray(arr.mutable_data(), 0);
            return arr;
        });

    py::class_<DynProgPolyTraj>(m, "DynProgPolyTraj")
        .def(py::init())
        .def("point_at", &DynProgPolyTraj::at)
        .def("point_at", [](DynProgPolyTraj& traj, array_like<float> ts) {
            DynProgPolyTraj res;
            for (ssize_t i = 0; i < ts.shape(0); ++i) {
                res.points.push_back(traj.at(ts.mutable_data()[i]));
            }
            return res;
         })
        .def("__getitem__", [](DynProgPolyTraj& traj, size_t i) {
            return traj.points[i];
         })
        .def("__setitem__", [](DynProgPolyTraj& traj, size_t i, DynProgPolyPoint& p) {
            traj.points[i] = p;
         })
        .def("__call__", [](DynProgPolyTraj& traj, float t) {
            py::array_t<float> arr({6});
            DynProgPolyPoint p = traj.at(t);
            p.writeToArray(arr.mutable_data(), 0);
            return arr;
         })
        .def("__call__", [](DynProgPolyTraj& traj, array_like<float> ts) {

            assert_shape(ts, {{-1}});

            py::array_t<float> arr({(size_t)ts.shape(0), (size_t)6});
            for (ssize_t i = 0; i < ts.shape(0); ++i) {
                DynProgPolyPoint p = traj.at(ts.mutable_data()[i]);
                p.writeToArray(arr.mutable_data(), i * 6);
            }

            return arr;
         })
        .def("as_numpy", [](DynProgPolyTraj& traj) {

            py::array_t<float> arr({traj.points.size(), (size_t)6});
            for (size_t i = 0; i < traj.points.size(); ++i) {
                traj.points[i].writeToArray(arr.mutable_data(), i * 6);
            }

            return arr;
        })
        .def_readwrite("points", &DynProgPolyTraj::points);

    py::class_<DynProgPolyCartTraj>(m, "DynProgPolyCartTraj")
        .def(py::init())
        .def("point_at", &DynProgPolyCartTraj::at)
        .def("point_at", [](DynProgPolyCartTraj& traj, array_like<double> ts) {
            DynProgPolyCartTraj res;
            for (ssize_t i = 0; i < ts.shape(0); ++i) {
                res.points.push_back(traj.at(ts.mutable_data()[i]));
            }
            return res;
         })
        .def("__getitem__", [](DynProgPolyCartTraj& traj, size_t i) {
            return traj.points[i];
         })
        .def("__setitem__", [](DynProgPolyCartTraj& traj, size_t i, DynProgPolyCartPoint& p) {
            traj.points[i] = p;
         })
        .def("__call__", [](DynProgPolyCartTraj& traj, double t) {
            py::array_t<double> arr({8});
            DynProgPolyCartPoint p = traj.at(t);
            p.writeToArray(arr.mutable_data(), 0);
            return arr;
         })
        .def("__call__", [](DynProgPolyCartTraj& traj, array_like<double> ts) {

            assert_shape(ts, {{-1}});

            py::array_t<double> arr({(size_t)ts.shape(0), (size_t)8});
            for (ssize_t i = 0; i < ts.shape(0); ++i) {
                DynProgPolyCartPoint p = traj.at(ts.mutable_data()[i]);
                p.writeToArray(arr.mutable_data(), i * 8);
            }

            return arr;
         })
        .def("as_numpy", [](DynProgPolyCartTraj& traj) {

            py::array_t<double> arr({traj.points.size(), (size_t)8});
            for (size_t i = 0; i < traj.points.size(); ++i) {
                traj.points[i].writeToArray(arr.mutable_data(), i * 8);
            }

            return arr;
        })
        .def_readwrite("points", &DynProgPolyCartTraj::points);

    py::class_<DynProgPolyPlannerParams>(m, "DynProgPolyPlannerParams")
        .def(py::init())
        .def("update_step_sizes", &DynProgPolyPlannerParams::updateStepSizes)
        .def_readwrite("eval_steps", &DynProgPolyPlannerParams::eval_steps)
        .def_readwrite("t_steps", &DynProgPolyPlannerParams::t_steps)
        .def_readwrite("s_steps", &DynProgPolyPlannerParams::s_steps)
        .def_readwrite("ds_steps", &DynProgPolyPlannerParams::ds_steps)
        .def_readwrite("l_steps", &DynProgPolyPlannerParams::l_steps)
        .def_readwrite("s_min", &DynProgPolyPlannerParams::s_min)
        .def_readwrite("s_max", &DynProgPolyPlannerParams::s_max)
        .def_readwrite("ds_min", &DynProgPolyPlannerParams::ds_min)
        .def_readwrite("ds_max", &DynProgPolyPlannerParams::ds_max)
        .def_readwrite("dds_min", &DynProgPolyPlannerParams::dds_min)
        .def_readwrite("dds_max", &DynProgPolyPlannerParams::dds_max)
        .def_readwrite("l_min", &DynProgPolyPlannerParams::l_min)
        .def_readwrite("l_max", &DynProgPolyPlannerParams::l_max)
        .def_readwrite("dl_min", &DynProgPolyPlannerParams::dl_min)
        .def_readwrite("dl_max", &DynProgPolyPlannerParams::dl_max)
        .def_readwrite("dt", &DynProgPolyPlannerParams::dt)
        .def_readwrite("dt_start", &DynProgPolyPlannerParams::dt_start)
        .def_readwrite("dt_cart", &DynProgPolyPlannerParams::dt_cart)
        .def_readwrite("a_total_max", &DynProgPolyPlannerParams::a_total_max)
        .def_readwrite("a_lat_abs_max", &DynProgPolyPlannerParams::a_lat_abs_max)
        .def_readwrite("w_v_diff", &DynProgPolyPlannerParams::w_v_diff)
        .def_readwrite("w_l", &DynProgPolyPlannerParams::w_l)
        .def_readwrite("w_j", &DynProgPolyPlannerParams::w_j)
        .def_readwrite("width_veh", &DynProgPolyPlannerParams::width_veh)
        .def_readwrite("length_veh", &DynProgPolyPlannerParams::length_veh)
        .def_property_readonly("__slots__", [](DynProgPolyPlannerParams& p){
            return std::vector<std::string>({
                "eval_steps",
                "t_steps",
                "ds_steps",
                "s_steps",
                "l_steps",
                "s_min",
                "s_max",
                "ds_min",
                "ds_max",
                "dds_min",
                "dds_max",
                "l_min",
                "l_max",
                "dl_min",
                "dl_max",
                "dt",
                "dt_start",
                "dt_cart",
                "a_total_max",
                "a_lat_abs_max",
                "w_v_diff",
                "w_l",
                "w_j",
                "width_veh",
                "length_veh"
            });
        });

    py::class_<DynProgPolyPlanner>(m, "DynProgPolyPlanner")
        .def(py::init())
        .def("reinit_buffers", &DynProgPolyPlanner::reinitBuffers,
            py::arg("params"),
            py::arg("force") = false)
        .def("update", &DynProgPolyPlanner::update)
        .def("frenet_to_cartesian", &DynProgPolyPlanner::frenetToCartesian);
}

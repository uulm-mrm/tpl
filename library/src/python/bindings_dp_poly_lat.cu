#include "tplcpp/python/bindings_dp_poly_lat.cuh"

#include "tplcpp/dyn_prog/poly_lat_planner.cuh"

namespace py = pybind11;

void loadDynProgPolyLatPlannerBindings(py::module& m) {

    py::class_<PolyLatTrajPoint>(m, "PolyLatTrajPoint")
        .def(py::init())
        .def("copy", [](PolyLatTrajPoint& tp) {
            return tp; 
        }, py::return_value_policy::copy)
        .def("__deepcopy__", [](PolyLatTrajPoint& tp, py::object& memo) {
            return tp;
        }, py::return_value_policy::copy)
        .def_readwrite("t", &PolyLatTrajPoint::t)
        .def_readwrite("l", &PolyLatTrajPoint::l)
        .def_readwrite("dl", &PolyLatTrajPoint::dl)
        .def_readwrite("ddl", &PolyLatTrajPoint::ddl)
        .def_readwrite("dddl", &PolyLatTrajPoint::dddl)
        .def_readwrite("s", &PolyLatTrajPoint::s)
        .def_readwrite("v", &PolyLatTrajPoint::v)
        .def_readwrite("x", &PolyLatTrajPoint::x)
        .def_readwrite("y", &PolyLatTrajPoint::y)
        .def_readwrite("heading", &PolyLatTrajPoint::heading)
        .def_readwrite("distance", &PolyLatTrajPoint::distance)
        .def_readwrite("k", &PolyLatTrajPoint::k)
        .def("as_numpy", [](PolyLatTrajPoint& p) {
            py::array_t<double> arr({12});
            double* data = arr.mutable_data();
            data[0] = p.t;
            data[1] = p.l;
            data[2] = p.dl;
            data[3] = p.ddl;
            data[4] = p.dddl;
            data[5] = p.s;
            data[6] = p.v;
            data[7] = p.x;
            data[8] = p.y;
            data[9] = p.heading;
            data[10] = p.distance;
            data[11] = p.k;
            return arr;
        });

    py::class_<PolyLatTraj>(m, "PolyLatTraj")
        .def(py::init())
        .def_readwrite("points", &PolyLatTraj::points)
        .def_readwrite("poly", &PolyLatTraj::poly)
        .def_readwrite("cost", &PolyLatTraj::cost)
        .def("insert_after_station", &PolyLatTraj::insertAfterStation)
        .def("update_time_dist_curv", &PolyLatTraj::updateTimeDistCurv)
        .def("lerp", &PolyLatTraj::lerp)
        .def("lerp", [](PolyLatTraj& traj, array_like<double> distances) {
            assert_shape(distances, {{-1}});
            ssize_t l = distances.shape(0);
            py::array_t<double> arr({l, 12l});
            double* data = arr.mutable_data();
            for (ssize_t i = 0; i < l; ++i) {
                PolyLatTrajPoint p = traj.lerp(distances.data()[i]);
                data[i*12] = p.t;
                data[i*12+1] = p.l;
                data[i*12+2] = p.dl;
                data[i*12+3] = p.ddl;
                data[i*12+4] = p.dddl;
                data[i*12+5] = p.s;
                data[i*12+6] = p.v;
                data[i*12+7] = p.x;
                data[i*12+8] = p.y;
                data[i*12+9] = p.heading;
                data[i*12+10] = p.distance;
                data[i*12+11] = p.k;
            }
            return arr;
        })
        .def("as_numpy", [](PolyLatTraj& traj) {
            ssize_t l = traj.points.size();
            py::array_t<double> arr({l, 12l});
            double* data = arr.mutable_data();
            for (ssize_t i = 0; i < l; ++i) {
                PolyLatTrajPoint& p = traj.points[i];
                data[i*12] = p.t;
                data[i*12+1] = p.l;
                data[i*12+2] = p.dl;
                data[i*12+3] = p.ddl;
                data[i*12+4] = p.dddl;
                data[i*12+5] = p.s;
                data[i*12+6] = p.v;
                data[i*12+7] = p.x;
                data[i*12+8] = p.y;
                data[i*12+9] = p.heading;
                data[i*12+10] = p.distance;
                data[i*12+11] = p.k;
            }
            return arr;
        })
        .def("copy", [](PolyLatTraj& traj) {
            return traj;
        }, py::return_value_policy::copy);

    py::class_<PolyLatPlannerParams>(m, "PolyLatPlannerParams")
        .def(py::init())
        .def("copy", [](PolyLatPlannerParams& tp) { 
                return tp; 
        }, py::return_value_policy::copy)
        .def("__deepcopy__", [](PolyLatPlannerParams& tp, py::object& memo) {
                return tp; 
        }, py::return_value_policy::copy)
        .def_readwrite("l_min", &PolyLatPlannerParams::l_min)
        .def_readwrite("l_max", &PolyLatPlannerParams::l_max)
        .def_readwrite("s_min", &PolyLatPlannerParams::s_min)
        .def_readwrite("s_max", &PolyLatPlannerParams::s_max)
        .def_readwrite("s_steps", &PolyLatPlannerParams::s_steps)
        .def_readwrite("l_dst_min", &PolyLatPlannerParams::l_dst_min)
        .def_readwrite("l_dst_max", &PolyLatPlannerParams::l_dst_max)
        .def_readwrite("s_dst_min", &PolyLatPlannerParams::s_dst_min)
        .def_readwrite("s_dst_max", &PolyLatPlannerParams::s_dst_max)
        .def_readwrite("l_dst_steps", &PolyLatPlannerParams::l_dst_steps)
        .def_readwrite("s_dst_steps", &PolyLatPlannerParams::s_dst_steps)
        .def_readwrite("l_trg", &PolyLatPlannerParams::l_trg)
        .def_readwrite("w_l", &PolyLatPlannerParams::w_l)
        .def_readwrite("w_k", &PolyLatPlannerParams::w_k)
        .def_readwrite("w_dl", &PolyLatPlannerParams::w_dl)
        .def_readwrite("w_ddl", &PolyLatPlannerParams::w_ddl)
        .def_readwrite("w_dddl", &PolyLatPlannerParams::w_dddl)
        .def_readwrite("w_right", &PolyLatPlannerParams::w_right)
        .def_readwrite("w_len", &PolyLatPlannerParams::w_len)
        .def_readwrite("k_abs_max", &PolyLatPlannerParams::k_abs_max)
        .def_readwrite("a_lat_abs_max", &PolyLatPlannerParams::a_lat_abs_max)
        .def_readwrite("width_veh", &PolyLatPlannerParams::width_veh)
        .def_readwrite("length_veh", &PolyLatPlannerParams::length_veh)
        .def_property_readonly("__slots__", [](PolyLatPlannerParams& p){
            return std::vector<std::string>({
                "l_min",
                "l_max",
                "s_min",
                "s_max",
                "s_steps",
                "l_dst_min",
                "l_dst_max",
                "s_dst_min",
                "s_dst_max",
                "l_dst_steps",
                "s_dst_steps",
                "l_trg",
                "w_l",
                "w_k",
                "w_dl",
                "w_ddl",
                "w_dddl",
                "w_right",
                "w_len",
                "k_abs_max",
                "a_lat_abs_max",
                "width_veh",
                "length_veh"
            });
        });

    py::class_<PolyLatPlanner>(m, "PolyLatPlanner")
        .def(py::init())
        .def("reinit_buffers", &PolyLatPlanner::reinitBuffers)
        .def("update", &PolyLatPlanner::update);
}

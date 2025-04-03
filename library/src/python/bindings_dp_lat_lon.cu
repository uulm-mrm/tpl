#include "tplcpp/python/bindings_dp_lat_lon.cuh"

#include "tplcpp/dyn_prog/lat_lon_planner.cuh"

namespace py = pybind11;

void loadDynProgLatLonPlannerBindings(py::module& m) {

    py::enum_<LatLonConstr>(m, "LatLonConstr", py::arithmetic())
        .value("OCCUPANCY", LAT_LON_CONSTR_OCCUPANCY)
        .value("VELOCITY", LAT_LON_CONSTR_VELOCITY)
        .value("ANGLE", LAT_LON_CONSTR_ANGLE);

    py::class_<LatDistances>(m, "LatDistances")
        .def(py::init<>())
        .def_readwrite("d_left", &LatDistances::d_left)
        .def_readwrite("d_right", &LatDistances::d_right);

    py::class_<LatLonCartState>(m, "LatLonCartState")
        .def(py::init<>())
        .def_readwrite("t", &LatLonCartState::t)
        .def_readwrite("distance", &LatLonCartState::distance)
        .def_readwrite("x", &LatLonCartState::x)
        .def_readwrite("y", &LatLonCartState::y)
        .def_readwrite("v", &LatLonCartState::v)
        .def_readwrite("a", &LatLonCartState::a)
        .def_readwrite("heading", &LatLonCartState::heading)
        .def_readwrite("k", &LatLonCartState::k)
        .def_readwrite("cost", &LatLonCartState::cost)
        .def_readwrite("constr", &LatLonCartState::constr)
        .def_readwrite("flags_constr", &LatLonCartState::flags_constr)
        .def("as_numpy", [](LatLonCartState& state) {
            py::array_t<double> arr({8});
            double* data = arr.mutable_data();
            data[0] = state.t;
            data[1] = state.distance;
            data[2] = state.x;
            data[3] = state.y;
            data[4] = state.v;
            data[5] = state.a;
            data[6] = state.heading;
            data[7] = state.k;
            return arr;
        });

    py::class_<LatLonState>(m, "LatLonState")
        .def(py::init<>())
        .def_readwrite("t", &LatLonState::t)
        .def_readwrite("s", &LatLonState::s)
        .def_readwrite("ds", &LatLonState::ds)
        .def_readwrite("dds", &LatLonState::dds)
        .def_readwrite("ddds", &LatLonState::ddds)
        .def_readwrite("l", &LatLonState::l)
        .def_readwrite("dl", &LatLonState::dl)
        .def_readwrite("ddl", &LatLonState::ddl)
        .def_readwrite("dddl", &LatLonState::dddl)
        .def_readwrite("cost", &LatLonState::cost)
        .def_readwrite("constr", &LatLonState::constr)
        .def_readwrite("flags_constr", &LatLonState::flags_constr)
        .def("as_numpy", [](LatLonState& state) {
            py::array_t<double> arr({7});
            double* data = arr.mutable_data();
            data[0] = state.s;
            data[1] = state.ds;
            data[2] = state.dds;
            data[3] = state.ddds;
            data[4] = state.l;
            data[5] = state.dl;
            data[6] = state.ddl;
            return arr;
        });

    py::class_<LatLonCartTraj>(m, "LatLonCartTraj")
        .def(py::init<>())
        .def_readwrite("states", &LatLonCartTraj::states)
        .def("__getitem__", [](LatLonCartTraj& traj, int idx) {
            return traj.states[idx];
         }, py::return_value_policy::reference)
        .def("__setitem__", [](LatLonCartTraj& traj, int idx, LatLonCartState& val) {
            traj.states[idx] = val;
         })
        .def("lerp", &LatLonCartTraj::lerp)
        .def("lerp", [](LatLonCartTraj& traj, array_like<double> distances) {
            assert_shape(distances, {{-1}});
            ssize_t l = distances.shape(0);
            py::array_t<double> arr({l, 7l});
            for (ssize_t i = 0; i < l; ++i) {
                LatLonCartState state = traj.lerp(distances.data()[i]);
                double* data = arr.mutable_data();
                data[i*7] = state.distance;
                data[i*7+1] = state.x;
                data[i*7+2] = state.y;
                data[i*7+3] = state.v;
                data[i*7+4] = state.a;
                data[i*7+5] = state.heading;
                data[i*7+6] = state.k;
            }
            return arr;
        })
        .def("as_numpy", [](LatLonCartTraj& traj) {
            ssize_t l = traj.states.size();
            py::array_t<double> arr({l, 8l});
            double* data = arr.mutable_data();
            for (ssize_t i = 0; i < l; ++i) {
                LatLonCartState& state = traj.states[i];
                data[i*8] = state.t;
                data[i*8+1] = state.distance;
                data[i*8+2] = state.x;
                data[i*8+3] = state.y;
                data[i*8+4] = state.v;
                data[i*8+5] = state.a;
                data[i*8+6] = state.heading;
                data[i*8+7] = state.k;
            }
            return arr;
        })
        .def("copy", [](LatLonCartTraj& traj) {
            return traj;
        }, py::return_value_policy::copy);

    py::class_<LatLonTraj>(m, "LatLonTraj")
        .def(py::init<>())
        .def_readwrite("states", &LatLonTraj::states)
        .def("state", &LatLonTraj::state)
        .def("lerp", &LatLonTraj::lerp)
        .def("__getitem__", [](LatLonTraj& traj, int idx) -> LatLonState&{
            return traj.states[idx];
         }, py::return_value_policy::reference)
        .def("__setitem__", [](LatLonTraj& traj, int idx, LatLonState& val) {
            traj.states[idx] = val;
         })
        .def("state", [](LatLonTraj& traj, array_like<double> ts) {
            assert_shape(ts, {{-1}});
            ssize_t l = ts.shape(0);
            py::array_t<double> arr({l, 7l});
            for (ssize_t i = 0; i < l; ++i) {
                LatLonState state = traj.state(ts.data()[i]);
                double* data = arr.mutable_data();
                data[i*7] = state.s;
                data[i*7+1] = state.ds;
                data[i*7+2] = state.dds;
                data[i*7+3] = state.ddds;
                data[i*7+4] = state.l;
                data[i*7+5] = state.dl;
                data[i*7+6] = state.ddl;
            }
            return arr;
        })
        .def("lerp", [](LatLonTraj& traj, array_like<double> ts) {
            assert_shape(ts, {{-1}});
            ssize_t l = ts.shape(0);
            py::array_t<double> arr({l, 7l});
            for (ssize_t i = 0; i < l; ++i) {
                LatLonState state = traj.lerp(ts.data()[i]);
                double* data = arr.mutable_data();
                data[i*7] = state.s;
                data[i*7+1] = state.ds;
                data[i*7+2] = state.dds;
                data[i*7+3] = state.ddds;
                data[i*7+4] = state.l;
                data[i*7+5] = state.dl;
                data[i*7+6] = state.ddl;
            }
            return arr;
        })
        .def("as_numpy", [](LatLonTraj& traj) {
            ssize_t l = traj.states.size();
            py::array_t<double> arr({l, 9l});
            double* data = arr.mutable_data();
            for (ssize_t i = 0; i < l; ++i) {
                LatLonState& state = traj.states[i];
                data[i*9] = state.t;
                data[i*9+1] = state.s;
                data[i*9+2] = state.ds;
                data[i*9+3] = state.dds;
                data[i*9+4] = state.ddds;
                data[i*9+5] = state.l;
                data[i*9+6] = state.dl;
                data[i*9+7] = state.ddl;
                data[i*9+8] = state.dddl;
            }
            return arr;
        }).
        def("copy", [](LatLonTraj& traj) {
            return traj;
        }, py::return_value_policy::copy);

    py::class_<DynProgLatLonPlannerParams>(m, "DynProgLatLonPlannerParams")
        .def(py::init())
        .def("__deepcopy__", [](DynProgLatLonPlannerParams& p, py::object& memo) {
                return p;
        }, py::return_value_policy::copy)
        .def_readwrite("s_min", &DynProgLatLonPlannerParams::s_min)
        .def_readwrite("s_max", &DynProgLatLonPlannerParams::s_max)
        .def_readwrite("ds_min", &DynProgLatLonPlannerParams::ds_min)
        .def_readwrite("ds_max", &DynProgLatLonPlannerParams::ds_max)
        .def_readwrite("l_min", &DynProgLatLonPlannerParams::l_min)
        .def_readwrite("l_max", &DynProgLatLonPlannerParams::l_max)
        .def_readwrite("dds_min", &DynProgLatLonPlannerParams::dds_min)
        .def_readwrite("dds_max", &DynProgLatLonPlannerParams::dds_max)
        .def_readwrite("dl_min", &DynProgLatLonPlannerParams::dl_min)
        .def_readwrite("dl_max", &DynProgLatLonPlannerParams::dl_max)
        .def_readwrite("t_steps", &DynProgLatLonPlannerParams::t_steps)
        .def_readwrite("s_steps", &DynProgLatLonPlannerParams::s_steps)
        .def_readwrite("ds_steps", &DynProgLatLonPlannerParams::ds_steps)
        .def_readwrite("l_steps", &DynProgLatLonPlannerParams::l_steps)
        .def_readwrite("dt", &DynProgLatLonPlannerParams::dt)
        .def_readwrite("dt_smooth_traj", &DynProgLatLonPlannerParams::dt_smooth_traj)
        .def_readwrite("dt_start", &DynProgLatLonPlannerParams::dt_start)
        .def_readwrite("dds_start", &DynProgLatLonPlannerParams::dds_start)
        .def_readwrite("w_dds_start", &DynProgLatLonPlannerParams::w_dds_start)
        .def_readwrite("angle_start", &DynProgLatLonPlannerParams::angle_start)
        .def_readwrite("w_angle_start", &DynProgLatLonPlannerParams::w_angle_start)
        .def_readwrite("l_trg", &DynProgLatLonPlannerParams::l_trg)
        .def_readwrite("w_progress", &DynProgLatLonPlannerParams::w_progress)
        .def_readwrite("w_dds", &DynProgLatLonPlannerParams::w_dds)
        .def_readwrite("w_ddds", &DynProgLatLonPlannerParams::w_ddds)
        .def_readwrite("w_l", &DynProgLatLonPlannerParams::w_l)
        .def_readwrite("w_dl", &DynProgLatLonPlannerParams::w_dl)
        .def_readwrite("w_ddl", &DynProgLatLonPlannerParams::w_ddl)
        .def_readwrite("w_xing_slow", &DynProgLatLonPlannerParams::w_xing_slow)
        .def_readwrite("w_safety_dist", &DynProgLatLonPlannerParams::w_safety_dist)
        .def_readwrite("w_lat_dist", &DynProgLatLonPlannerParams::w_lat_dist)
        .def_readwrite("d_lat_comf", &DynProgLatLonPlannerParams::d_lat_comf)
        .def_readwrite("slope_abs_max", &DynProgLatLonPlannerParams::slope_abs_max)
        .def_readwrite("time_gap", &DynProgLatLonPlannerParams::time_gap)
        .def_readwrite("gap_min", &DynProgLatLonPlannerParams::gap_min)
        .def_readwrite("t_st_min", &DynProgLatLonPlannerParams::t_st_min)
        .def_readwrite("t_st_max", &DynProgLatLonPlannerParams::t_st_max)
        .def_readwrite("s_st", &DynProgLatLonPlannerParams::s_st)
        .def_readwrite("w_spatio_temporal", &DynProgLatLonPlannerParams::w_spatio_temporal)
        .def_readwrite("width_veh", &DynProgLatLonPlannerParams::width_veh)
        .def_readwrite("length_veh", &DynProgLatLonPlannerParams::length_veh)
        .def_property_readonly("__slots__", [](DynProgLatLonPlannerParams& p){
            return std::vector<std::string>({
                "s_min",
                "s_max",
                "ds_min",
                "ds_max",
                "l_min",
                "l_max",
                "dds_min",
                "dds_max",
                "dl_min",
                "dl_max",
                "t_steps",
                "s_steps",
                "ds_steps",
                "l_steps",
                "dt",
                "dt_smooth_traj",
                "dt_start",
                "dds_start",
                "w_dds_start",
                "angle_start",
                "w_angle_start",
                "l_trg",
                "w_progress",
                "w_dds",
                "w_ddds",
                "w_l",
                "w_dl",
                "w_ddl",
                "w_xing_slow",
                "w_safety_dist",
                "w_lat_dist",
                "d_lat_comf",
                "slope_abs_max",
                "time_gap",
                "gap_min",
                "t_st_min",
                "t_st_max",
                "s_st",
                "w_spatio_temporal",
                "width_veh",
                "length_veh"
            });
        });

    py::class_<DynProgLatLonPlanner>(m, "DynProgLatLonPlanner")
        .def(py::init())
        .def_readwrite("traj_dp", &DynProgLatLonPlanner::trajDp)
        .def_readwrite("traj_smooth", &DynProgLatLonPlanner::trajSmooth)
        .def_readwrite("traj_dp_cart", &DynProgLatLonPlanner::trajDpCart)
        .def_readwrite("traj_smooth_cart", &DynProgLatLonPlanner::trajSmoothCart)
        .def_readwrite("lat_dists", &DynProgLatLonPlanner::latDists)
        .def("reinit_buffers", [](DynProgLatLonPlanner& planner,
                                  DynProgLatLonPlannerParams& params,
                                  bool force) {
            planner.reinitBuffers(params, force);
        },
        py::arg("params"),
        py::arg("force") = false)
        .def("update_traj_dp", &DynProgLatLonPlanner::updateTrajDp)
        .def("update_traj_smooth", &DynProgLatLonPlanner::updateTrajSmooth)
        .def("update_traj_cart", &DynProgLatLonPlanner::updateTrajCart)
        .def("frenet_to_cartesian", [](DynProgLatLonPlanner& p, LatLonTraj& traj, RefLine& refLine) {
            LatLonCartTraj result;
            p.frenetToCartesian(traj, refLine, result);
            return result;
        })
        .def("reeval_traj", &DynProgLatLonPlanner::reevalTraj)
        .def("query_backward_pass_tex", [](DynProgLatLonPlanner& p, int idx_t, float s, float ds, float l){
            py::array_t<float, py::array::c_style> val({4});
            p.queryBackwardPassTex(val.mutable_data(), idx_t, s, ds, l);
            return val;
        })
        .def("copy_backward_pass_tex", [](DynProgLatLonPlanner& p, int idx_t){
            py::array_t<float, py::array::c_style> tex({p.params.l_steps,
                                                        p.params.ds_steps,
                                                        p.params.s_steps,
                                                        4});
            p.copyBackwardPassTex(tex.mutable_data(), idx_t);
            return tex;
        });
}

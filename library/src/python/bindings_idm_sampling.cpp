#include "tplcpp/python/bindings_idm_sampling.hpp"

#include "tplcpp/idm_sampling.hpp"

namespace py = pybind11;

void loadIdmSamplingBindings(py::module& m) {

    py::class_<IdmSamplingState>(m, "IdmSamplingState")
        .def(py::init<>())
        .def_readwrite("t", &IdmSamplingState::t)
        .def_readwrite("x", &IdmSamplingState::x)
        .def_readwrite("y", &IdmSamplingState::y)
        .def_readwrite("heading", &IdmSamplingState::heading)
        .def_readwrite("steer_angle", &IdmSamplingState::steer_angle)
        .def_readwrite("v", &IdmSamplingState::v)
        .def_readwrite("a", &IdmSamplingState::a)
        .def_readwrite("s", &IdmSamplingState::s)
        .def_readwrite("l", &IdmSamplingState::l)
        .def("as_numpy", [](IdmSamplingState& state) {
            py::array_t<double> arr({9});
            double* data = arr.mutable_data();
            data[0] = state.t;
            data[1] = state.x;
            data[2] = state.y;
            data[3] = state.heading;
            data[4] = state.steer_angle;
            data[5] = state.v;
            data[6] = state.a;
            data[7] = state.s;
            data[8] = state.l;
            return arr;
        });

    py::class_<IdmSamplingRefState>(m, "IdmSamplingRefState")
        .def(py::init<>())
        .def_readwrite("t", &IdmSamplingRefState::t)
        .def_readwrite("x", &IdmSamplingRefState::x)
        .def_readwrite("y", &IdmSamplingRefState::y)
        .def_readwrite("heading", &IdmSamplingRefState::heading)
        .def_readwrite("v", &IdmSamplingRefState::v)
        .def_readwrite("a", &IdmSamplingRefState::a)
        .def_readwrite("s", &IdmSamplingRefState::s)
        .def_readwrite("l", &IdmSamplingRefState::l)
        .def("as_numpy", [](IdmSamplingRefState& state) {
            py::array_t<double> arr({8});
            double* data = arr.mutable_data();
            data[0] = state.t;
            data[1] = state.x;
            data[2] = state.y;
            data[3] = state.heading;
            data[4] = state.v;
            data[5] = state.a;
            data[6] = state.s;
            data[7] = state.l;
            return arr;
        });

    py::class_<IdmSamplingTraj>(m, "IdmSamplingTraj")
        .def(py::init<>())
        .def_readwrite("states", &IdmSamplingTraj::states)
        .def_readwrite("ref_states", &IdmSamplingTraj::refStates)
        .def_readwrite("invalid", &IdmSamplingTraj::invalid)
        .def("__getitem__", [](IdmSamplingTraj& traj, int idx) {
            return traj.states[idx];
         }, py::return_value_policy::reference)
        .def("__setitem__", [](IdmSamplingTraj& traj, int idx, IdmSamplingState& val) {
            traj.states[idx] = val;
         })
        .def("lerp", &IdmSamplingTraj::lerp)
        .def("lerp_ref", &IdmSamplingTraj::lerpRef)
        .def("lerp", [](IdmSamplingTraj& traj, array_like<double> ts) {
            /*
            assert_shape(ts, {{-1}});
            ssize_t l = ts.shape(0);
            py::array_t<double> arr({l, 7l});
            for (ssize_t i = 0; i < l; ++i) {
                IdmSamplingTraj state = traj.lerp(ts.data()[i]);
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
            */
        })
        .def("as_numpy", [](IdmSamplingTraj& traj) {
            ssize_t l = traj.states.size();
            py::array_t<double> arr({l, 9l});
            double* data = arr.mutable_data();
            for (ssize_t i = 0; i < l; ++i) {
                IdmSamplingState& state = traj.states[i];
                data[i*9] = state.t;
                data[i*9+1] = state.x;
                data[i*9+2] = state.y;
                data[i*9+3] = state.heading;
                data[i*9+4] = state.steer_angle;
                data[i*9+5] = state.v;
                data[i*9+6] = state.a;
                data[i*9+7] = state.s;
                data[i*9+8] = state.l;
            }
            return arr;
        })
        .def("ref_as_numpy", [](IdmSamplingTraj& traj) {
            ssize_t l = traj.refStates.size();
            py::array_t<double> arr({l, 8l});
            double* data = arr.mutable_data();
            for (ssize_t i = 0; i < l; ++i) {
                IdmSamplingRefState& state = traj.refStates[i];
                data[i*8] = state.t;
                data[i*8+1] = state.x;
                data[i*8+2] = state.y;
                data[i*8+3] = state.heading;
                data[i*8+4] = state.v;
                data[i*8+5] = state.a;
                data[i*8+6] = state.s;
                data[i*8+7] = state.l;
            }
            return arr;
        })
        .def("copy", [](IdmSamplingTraj& traj) {
            return traj;
        }, py::return_value_policy::copy);

    py::class_<IdmSamplingParams>(m, "IdmSamplingParams")
        .def(py::init())
        .def("__deepcopy__", [](IdmSamplingParams& p, py::object& memo) {
                return p;
        }, py::return_value_policy::copy)
        .def_readwrite("steps_t", &IdmSamplingParams::steps_t)
        .def_readwrite("dt", &IdmSamplingParams::dt)
        .def_readwrite("dead_time", &IdmSamplingParams::dead_time)
        .def_readwrite("lat_steps", &IdmSamplingParams::lat_steps)
        .def_readwrite("d_safe_lat", &IdmSamplingParams::d_safe_lat)
        .def_readwrite("d_safe_lat_path", &IdmSamplingParams::d_safe_lat_path)
        .def_readwrite("d_comf_lat", &IdmSamplingParams::d_comf_lat)
        .def_readwrite("k_stanley", &IdmSamplingParams::k_stanley)
        .def_readwrite("v_offset_stanley", &IdmSamplingParams::v_offset_stanley)
        .def_readwrite("steer_angle_max", &IdmSamplingParams::steer_angle_max)
        .def_readwrite("steer_rate_max", &IdmSamplingParams::steer_rate_max)
        .def_readwrite("t_vel_lookahead", &IdmSamplingParams::t_vel_lookahead)
        .def_readwrite("d_safe_min", &IdmSamplingParams::d_safe_min)
        .def_readwrite("t_headway_desired", &IdmSamplingParams::t_headway_desired)
        .def_readwrite("a_break_comf", &IdmSamplingParams::a_break_comf)
        .def_readwrite("idm_exp_dcc", &IdmSamplingParams::idm_exp_dcc)
        .def_readwrite("idm_exp_acc", &IdmSamplingParams::idm_exp_acc)
        .def_readwrite("k_p_s", &IdmSamplingParams::k_p_s)
        .def_readwrite("k_p_v", &IdmSamplingParams::k_p_v)
        .def_readwrite("a_min", &IdmSamplingParams::a_min)
        .def_readwrite("a_max", &IdmSamplingParams::a_max)
        .def_readwrite("j_min", &IdmSamplingParams::j_min)
        .def_readwrite("j_max", &IdmSamplingParams::j_max)
        .def_readwrite("d_next_inters_point", &IdmSamplingParams::d_next_inters_point)
        .def_readwrite("width_veh", &IdmSamplingParams::width_veh)
        .def_readwrite("length_veh", &IdmSamplingParams::length_veh)
        .def_readwrite("radius_veh", &IdmSamplingParams::radius_veh)
        .def_readwrite("dist_front_veh", &IdmSamplingParams::dist_front_veh)
        .def_readwrite("dist_back_veh", &IdmSamplingParams::dist_back_veh)
        .def_readwrite("wheel_base", &IdmSamplingParams::wheel_base)
        .def_readwrite("l_trg", &IdmSamplingParams::l_trg)
        .def_readwrite("w_l", &IdmSamplingParams::w_l)
        .def_readwrite("w_a", &IdmSamplingParams::w_a)
        .def_readwrite("w_lat_dist", &IdmSamplingParams::w_lat_dist)
        .def_readwrite("dt_decision", &IdmSamplingParams::dt_decision)
        .def_readwrite("enable_reverse", &IdmSamplingParams::enable_reverse)
        .def_property_readonly("__slots__", [](IdmSamplingParams& p){
            return std::vector<std::string>({
                "steps_t",
                "dt",
                "dead_time",
                "lat_steps",
                "d_safe_lat",
                "d_safe_lat_path",
                "d_comf_lat",
                "k_stanley",
                "v_offset_stanley",
                "steer_angle_max",
                "steer_rate_max",
                "t_vel_lookahead",
                "d_safe_min",
                "t_headway_desired",
                "a_break_comf",
                "idm_exp_dcc",
                "idm_exp_acc",
                "k_p_s",
                "k_p_v",
                "a_min",
                "a_max",
                "j_min",
                "j_max",
                "d_next_inters_point",
                "width_veh",
                "length_veh",
                "radius_veh",
                "dist_front_veh",
                "dist_back_veh",
                "wheel_base",
                "l_trg",
                "w_l",
                "w_a",
                "w_lat_dist",
                "dt_decision",
                "enable_reverse"
            });
        });

    py::class_<IdmSamplingPlanner>(m, "IdmSamplingPlanner")
        .def(py::init())
        .def("insert_dyn_obj", [](IdmSamplingPlanner& p,
                                  array_like<double> prediction,
                                  array_like<double> hull,
                                  bool crossing) {
            assert_shape(prediction, {{-1, 5}});
            assert_shape(hull, {{-1, 2}});
            p.insertDynObj(std::vector<PredictionPoint>(
                                (PredictionPoint*)prediction.data(), 
                                (PredictionPoint*)prediction.data() + prediction.shape(0)),
                           std::vector<vec<2>>(
                                (vec<2>*)hull.data(),
                                (vec<2>*)hull.data() + hull.shape(0)),
                           crossing);
        })
        .def_readonly("trajs", &IdmSamplingPlanner::trajs)
        .def("reset", &IdmSamplingPlanner::reset)
        .def("update", &IdmSamplingPlanner::update);
}

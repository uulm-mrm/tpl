#include "tplcpp/python/bindings_dp_lon.cuh"

#include "tplcpp/dyn_prog/lon_planner.cuh"

namespace py = pybind11;

void loadDynProgLonPlannerBindings(py::module& m) {

    py::class_<DynProgLonPlannerParams>(m, "DynProgLonPlannerParams")
        .def(py::init())
        .def_readwrite("s_min", &DynProgLonPlannerParams::s_min)
        .def_readwrite("s_max", &DynProgLonPlannerParams::s_max)
        .def_readwrite("v_min", &DynProgLonPlannerParams::v_min)
        .def_readwrite("v_max", &DynProgLonPlannerParams::v_max)
        .def_readwrite("a_min", &DynProgLonPlannerParams::a_min)
        .def_readwrite("a_max", &DynProgLonPlannerParams::a_max)
        .def_readwrite("j_min", &DynProgLonPlannerParams::j_min)
        .def_readwrite("j_max", &DynProgLonPlannerParams::j_max)
        .def_readwrite("t_steps", &DynProgLonPlannerParams::t_steps)
        .def_readwrite("s_steps", &DynProgLonPlannerParams::s_steps)
        .def_readwrite("v_steps", &DynProgLonPlannerParams::v_steps)
        .def_readwrite("a_steps", &DynProgLonPlannerParams::a_steps)
        .def_readwrite("dt", &DynProgLonPlannerParams::dt)
        .def_readwrite("dt_start", &DynProgLonPlannerParams::dt_start)
        .def_readwrite("time_gap", &DynProgLonPlannerParams::time_gap)
        .def_readwrite("gap_min", &DynProgLonPlannerParams::gap_min)
        .def_readwrite("w_progress", &DynProgLonPlannerParams::w_progress)
        .def_readwrite("w_a", &DynProgLonPlannerParams::w_a)
        .def_readwrite("w_j", &DynProgLonPlannerParams::w_j)
        .def_readwrite("w_snap", &DynProgLonPlannerParams::w_snap)
        .def_readwrite("w_safety_dist", &DynProgLonPlannerParams::w_safety_dist)
        .def_readwrite("path_step_size", &DynProgLonPlannerParams::path_step_size)
        .def_readwrite("path_steps", &DynProgLonPlannerParams::path_steps)
        .def_readwrite("width_veh", &DynProgLonPlannerParams::width_veh)
        .def_readwrite("length_veh", &DynProgLonPlannerParams::length_veh)
        .def_property_readonly("__slots__", [](DynProgLonPlannerParams& p){
            return std::vector<std::string>({
                "s_min",
                "s_max",
                "v_min",
                "v_max",
                "a_min",
                "a_max",
                "j_min",
                "j_max",
                "t_steps",
                "s_steps",
                "v_steps",
                "a_steps",
                "dt",
                "dt_start",
                "time_gap",
                "gap_min",
                "w_progress",
                "w_a",
                "w_j",
                "w_snap",
                "w_safety_dist",
                "path_step_size",
                "path_steps",
                "width_veh",
                "length_veh"
            });
        });

    py::class_<LonState>(m, "LonState")
        .def(py::init<>())
        .def_readwrite("t", &LonState::t)
        .def_readwrite("s", &LonState::s)
        .def_readwrite("v", &LonState::v)
        .def_readwrite("a", &LonState::a)
        .def_readwrite("j", &LonState::j)
        .def_readwrite("cost", &LonState::cost)
        .def_readwrite("constr", &LonState::constr)
        .def("as_numpy", [](LonState& state) {
            py::array_t<double> arr({4});
            double* data = arr.mutable_data();
            data[0] = state.s;
            data[1] = state.v;
            data[2] = state.a;
            data[3] = state.j;
            return arr;
        });

    m.def("lon_dynamics", &lonDynamics);

    py::class_<LonTraj>(m, "LonTraj")
        .def(py::init<>())
        .def("copy", [](LonTraj& s){ return s; }, py::return_value_policy::copy)
        .def_readwrite("states", &LonTraj::states)
        .def("state", &LonTraj::state)
        .def("state", [](LonTraj& traj, array_like<double> ts) {
            assert_shape(ts, {{-1}});
            ssize_t l = ts.shape(0);
            py::array_t<double> arr({l, 4l});
            double* data = arr.mutable_data();
            for (ssize_t i = 0; i < l; ++i) {
                LonState state = traj.state(ts.data()[i]);
                data[i*4] = state.s;
                data[i*4+1] = state.v;
                data[i*4+2] = state.a;
                data[i*4+3] = state.j;
            }
            return arr;
        })
        .def("as_numpy", [](LonTraj& traj) {
            ssize_t l = traj.states.size();
            py::array_t<double> arr({l, 4l});
            double* data = arr.mutable_data();
            for (ssize_t i = 0; i < l; ++i) {
                LonState& state = traj.states[i];
                data[i*4] = state.s;
                data[i*4+1] = state.v;
                data[i*4+2] = state.a;
                data[i*4+3] = state.j;
            }
            return arr;
        });

    py::class_<DynProgLonPlanner>(m, "DynProgLonPlanner")
        .def(py::init())
        .def("reinit_buffers", &DynProgLonPlanner::reinitBuffers)
        .def("update", [](DynProgLonPlanner& dp,
                          LonState initialState,
                          array_like<float> path,
                          DynProgEnvironment& env) {
            assert_shape(path, {{dp.params.path_steps, 7}});
            return dp.update(initialState, (PathState*)path.mutable_data(), env);
        })
        .def("reeval_traj", [](DynProgLonPlanner& dp,
                               const LonTraj& traj,
                               array_like<float> path,
                               DynProgEnvironment& env) {
            assert_shape(path, {{dp.params.path_steps, 7}});
            return dp.reevalTraj(traj, (PathState*)path.mutable_data(), env);
        });
}

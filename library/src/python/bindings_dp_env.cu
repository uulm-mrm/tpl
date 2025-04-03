#include "tplcpp/python/bindings_dp_env.cuh"

#include "tplcpp/dyn_prog/env.cuh"

namespace py = pybind11;

void loadDynProgEnvironmentBindings(py::module& m) {

    py::class_<DynProgEnvParams>(m, "DynProgEnvParams")
        .def(py::init())
        .def("__deepcopy__", [](DynProgEnvParams& p, py::object& memo) {
                return p;
        }, py::return_value_policy::copy)
        .def_readwrite("t_min", &DynProgEnvParams::t_min)
        .def_readwrite("t_max", &DynProgEnvParams::t_max)
        .def_readwrite("l_min", &DynProgEnvParams::l_min)
        .def_readwrite("l_max", &DynProgEnvParams::l_max)
        .def_readwrite("s_min", &DynProgEnvParams::s_min)
        .def_readwrite("s_max", &DynProgEnvParams::s_max)
        .def_readwrite("ds_max", &DynProgEnvParams::ds_max)
        .def_readwrite("dir_min", &DynProgEnvParams::dir_min)
        .def_readwrite("dir_max", &DynProgEnvParams::dir_max)
        .def_readwrite("scale_objects", &DynProgEnvParams::scale_objects)
        .def_readwrite("dilation", &DynProgEnvParams::dilation)
        .def_readwrite("dt", &DynProgEnvParams::dt)
        .def_readwrite("dt_start", &DynProgEnvParams::dt_start)
        .def_readwrite("t_steps", &DynProgEnvParams::t_steps)
        .def_readwrite("s_steps", &DynProgEnvParams::s_steps)
        .def_readwrite("l_steps", &DynProgEnvParams::l_steps)
        .def_readwrite("dir_steps", &DynProgEnvParams::dir_steps)
        .def_property_readonly("__slots__", [](DynProgEnvParams& p){
            return std::vector<std::string>({
                "t_min",
                "t_max",
                "l_min",
                "l_max",
                "s_min",
                "s_max",
                "ds_max",
                "dir_min",
                "dir_max",
                "scale_objects",
                "dilation",
                "dt",
                "dt_start",
                "t_steps",
                "l_steps",
                "s_steps",
                "dir_steps",
            });
        });

    py::class_<DynProgEnvironment>(m, "DynProgEnvironment")
        .def(py::init())
        .def_readwrite("ref_line", &DynProgEnvironment::refLine)
        .def("reinit_buffers", [](DynProgEnvironment& env,
                                  DynProgEnvParams& params,
                                  bool force) {
            env.reinitBuffers(params, force);
        },
        py::arg("params"),
        py::arg("force") = false)
        .def("insert_geometry", [](DynProgEnvironment& p,
                                   array_like<double> geom,
                                   bool stationary) {
            assert_shape(geom, {{-1, 3}});
            p.insertGeometry(std::span<vec<3>>((vec<3>*)geom.mutable_data(), geom.shape(0)),
                             stationary);
        })
        .def("set_ref_line", [](DynProgEnvironment& p,
                                array_like<double> refLine,
                                float step_size) {
            assert_shape(refLine, {{-1, 9}});
            p.setRefLine((RefLinePoint*)refLine.data(), refLine.shape(0), step_size);
        }, py::return_value_policy::copy)
        .def("update", &DynProgEnvironment::update)
        .def("get_occ_map_cartesian", [](DynProgEnvironment& p){
            py::array_t<uint32_t, py::array::c_style> occ({
                p.occRenderer.fb->_width,
                p.occRenderer.fb->_height});
            p.occRenderer.readCartTex(occ.mutable_data());
            return occ;
        })
        .def("get_occ_map", [](DynProgEnvironment& p){
            py::array_t<float, py::array::c_style> occ({p.params.l_steps,
                                                        p.params.s_steps,
                                                        p.params.t_steps});
            p.getOccMap(occ.mutable_data());
            return occ;
        })
        .def("get_dist_map_lon", [](DynProgEnvironment& p){
            py::array_t<float, py::array::c_style> dist_map_lon({p.params.l_steps,
                                                                 p.params.s_steps,
                                                                 p.params.t_steps,
                                                                 2});
            p.getDistMapLon(dist_map_lon.mutable_data());
            return dist_map_lon;
        })
        .def("get_dist_map_dir", [](DynProgEnvironment& p, int idx_t){
            py::array_t<float, py::array::c_style> ddm({p.params.dir_steps,
                                                        p.params.l_steps,
                                                        p.params.s_steps});
            p.getDistMapDir(idx_t, ddm.mutable_data());
            return ddm;
        });
}

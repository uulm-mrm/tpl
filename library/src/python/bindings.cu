#include "tplcpp/utils.hpp"

#include "tplcpp/python/bindings_dp_common.cuh"
#include "tplcpp/python/bindings_dp_env.cuh"
#include "tplcpp/python/bindings_dp_lat_lon.cuh"
#include "tplcpp/python/bindings_dp_lon.cuh"
#include "tplcpp/python/bindings_dp_poly.cuh"
#include "tplcpp/python/bindings_dp_poly_lat.cuh"
#include "tplcpp/python/bindings_poly_sampling.cuh"
#include "tplcpp/python/bindings_idm_sampling.hpp"

PYBIND11_MODULE(tplcpp, m) {

    loadUtilsBindings(m);

    loadDynProgCommonBindings(m);
    loadDynProgEnvironmentBindings(m);

    loadDynProgLatLonPlannerBindings(m);
    loadDynProgLonPlannerBindings(m);
    loadDynProgPolyPlannerBindings(m);
    loadDynProgPolyLatPlannerBindings(m);

    loadPolySamplingBindings(m);
    loadIdmSamplingBindings(m);
}

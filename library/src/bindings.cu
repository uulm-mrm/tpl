#include "tplcpp/utils.hpp"
#include "tplcpp/poly_sampling.hpp"

PYBIND11_MODULE(tplcpp, m) {

    loadUtilsBindings(m);
    loadPolySamplingPlannerBindings(m);
}

#pragma once

#include "tplcpp/eigen_helpers.hpp"

#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

/**
 * Pybind11 helpers
 */

namespace py = pybind11;

/**
 * Some helpers to make handling arrays easier.
 */

template <typename T>
using array_like = py::array_t<T, py::array::c_style | py::array::forcecast>;

std::string shapeToStr(py::array& array);

void assertArrayShape(std::string name,
                      py::array& array,
                      std::vector<std::vector<int>> shapes);

#define assert_shape(array, ...) assertArrayShape(#array, array, __VA_ARGS__)

/**
 * The pre-made typecasters are not cool enough,
 * we write our own (which have NO bugs, for sure).
 */

namespace pybind11 {
    namespace detail {

        /**
         * Vector of eigen matrices (and vectors)
         */

        template <int R, int C>
        struct type_caster<std::vector<mat<R, C>>> {

        public:
            using vecType = std::vector<mat<R, C>>;

            PYBIND11_TYPE_CASTER(vecType, _("MatrixArray"));

            bool load(handle src, bool) {

                /*
                 * This following statement ensures that,
                 *  - the numpy array is contiguous in memory
                 *  - the unit datatype is double.
                 *
                 * If the pre-requisits are not fullfilled, the "forcecast"
                 * flag ensures that a corresponding conversion is made.
                 *
                 * This has the added benefit that we can feed this
                 * type caster anything that is convertible to a
                 * numpy array (e.g. lists, tuples, buffers, ...).
                 */

                auto array = pybind11::array_t<double,
                     pybind11::array::c_style
                         | pybind11::array::forcecast>::ensure(src);

                if constexpr (R == 1 && C == 1) {
                    if (array.ndim() != 1) {
                        throw pybind11::value_error(
                                "expected array of shape N");
                    }
                } else if constexpr (C == 1) {
                    if (array.ndim() != 2
                            || array.shape(1) != R) {
                        throw pybind11::value_error(
                                "expected array of shape Nx"
                                + std::to_string(R));
                    }
                } else {
                    if (array.ndim() != 3
                            || array.shape(1) != R
                            || array.shape(2) != C) {
                        throw pybind11::value_error(
                                "expected array of shape Nx"
                                + std::to_string(R)
                                + "x"
                                + std::to_string(C));
                    }
                }

                value.resize(array.shape(0));

                // this is definitely, really, very safe
                // because we checked the dimensions above

                std::memcpy((void*)value.data(),
                            (void*)array.data(),
                            sizeof(double) * array.size());

                return true;
            }

            static handle cast(
                    const std::vector<mat<R, C>>& src,
                    return_value_policy policy,
                    handle parent) {

                /**
                 * This returns the vector of matrices as a numpy array,
                 * which uses the vector memory (but does not own it).
                 *
                 * BUG: If the underlying vector is modified
                 * (or reallocates) the array becomes invalid.
                 */

                pybind11::detail::any_container<ssize_t> arrayShape;

                if constexpr (R == 1 && C == 1) {
                    arrayShape = {src.size()};
                } else if constexpr (C == 1) {
                    arrayShape = {src.size(), (size_t)R};
                } else {
                    arrayShape = {src.size(), (size_t)R, (size_t)C};
                }

                pybind11::array_t<double> array(
                        arrayShape,
                        (double*)src.data(),
                        parent);

                return array.release();
            }
        };
    } // namespace detail
} // namespace pybind11

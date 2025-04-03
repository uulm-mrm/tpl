#pragma once

#define EIGEN_NO_DEBUG

#include <Eigen/Dense>

/*
 * Convenience definitions
 */

template<int M, int N>
using mat = Eigen::Matrix<double, M, N>;
template<int N>
using vec = Eigen::Matrix<double, N, 1>;

template<int M, int N>
using matf = Eigen::Matrix<float, M, N>;
template<int N>
using vecf = Eigen::Matrix<float, N, 1>;

template<typename T, size_t N>
constexpr auto make_array(T value) -> std::array<T, N>
{
    std::array<T, N> a{};
    for (auto& x : a) {
        x = value;
    }
    return a;
}

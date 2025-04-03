#pragma once

#include <cmath>
#include <span>
#include <iostream>
#include <vector>

#include "tplcpp/python/binding_helpers.hpp"

#ifdef __CUDACC__
    #define __interop__ __host__ __device__
#else
    #define __interop__ 
    #include <algorithm>
    using std::max;
    using std::min;
#endif

#define sq(x) ((x)*(x))

template <typename T>
__interop__ __inline__ T normalizeAngle(T x) {

    x = std::fmod(x, 2*M_PI);
    x = std::fmod(x + 2*M_PI, 2*M_PI);
    if (x > M_PI) {
        x -= 2*M_PI;
    }

    return x;
}

template<typename T>
__interop__ __inline__ T shortAngleDist(T x, T y) {

    x = normalizeAngle(x);
    y = normalizeAngle(y);

    T a0 = y-x;
    T a1 = y-x + 2*M_PI;
    T a2 = y-x - 2*M_PI;

    if (abs(a1) < abs(a0)) {
        a0 = a1;
    }
    if (abs(a2) < abs(a0)) {
        a0 = a2;
    }

    return a0;
}

struct InterpVars {

    int i_prev = 0;
    int i_next = 0;
    double a = 0.0;

    template<typename A, typename T, typename F>
    InterpVars(A vals, F T::* ptr, double v, bool clipAlpha=true) {

        if (v < vals.front().*ptr) {
            i_prev = 0;
            i_next = 1;
        } else if (v >= vals.back().*ptr) {
            i_prev = vals.size() - 2;
            i_next = vals.size() - 1;
        } else { 
            for (int i = 1; i < (int)vals.size(); ++i) {
                const double& t_prev = vals[i-1].*ptr;
                const double& t_next = vals[i].*ptr;
                if (t_prev <= v && v < t_next) {
                    i_prev = i-1;
                    i_next = i;
                    break;
                }
            }
        }

        const double& t_prev = vals[i_prev].*ptr;
        const double& t_next = vals[i_next].*ptr;
        a = (v - t_prev) / (t_next - t_prev);
        if (clipAlpha) {
            a = std::max(0.0, std::min(1.0, a));
        }
    }

    template<typename A>
    InterpVars(A vals, double v, bool clipAlpha=true) {

        if (v < vals.front()) {
            i_prev = 0;
            i_next = 1;
        } else if (v >= vals.back()) {
            i_prev = vals.size() - 2;
            i_next = vals.size() - 1;
        } else { 
            if (vals.size() < 100) {
                // linear search for small arrays
                for (int i = 1; i < (int)vals.size(); ++i) {
                    const double& t_prev = vals[i-1];
                    const double& t_next = vals[i];
                    if (t_prev <= v && v < t_next) {
                        i_prev = i-1;
                        i_next = i;
                        break;
                    }
                }
            } else {
                // binary search for bigger arrays
                i_prev = 0;
                i_next = vals.size()-1;
                while (i_next - i_prev >= 2) {
                    int m = (i_next + i_prev) / 2;
                    if (vals[m] <= v) {
                        i_prev = m;
                    } else if (vals[m] > v) {
                        i_next = m;
                    } else {
                        break;
                    }
                }
            }
        }

        const double& t_prev = vals[i_prev];
        const double& t_next = vals[i_next];
        a = (v - t_prev) / (t_next - t_prev);
        if (clipAlpha) {
            a = std::max(0.0, std::min(1.0, a));
        }
    }
};

struct RefLinePoint {

    double x;
    double y;
    double heading;
    double s;
    double k;
    double v_max;
    double d_left;
    double d_right;
    double semantic;
};

__interop__ __inline__ RefLinePoint interpRefLine(
        const RefLinePoint* __restrict__ ref_line,
        const int len,
        const float step_size,
        const float s) {

    float a = s / step_size;
    int i0 = max(0, min(len-1, (int)floor(a)));
    int i1 = max(0, min(len-1, (int)ceil(a)));
    a -= i0;
    float ia = 1.0 - a;

    RefLinePoint p{};

    p.x = ref_line[i0].x * ia + ref_line[i1].x * a;
    p.y = ref_line[i0].y * ia + ref_line[i1].y * a;
    p.heading = ref_line[i0].heading + shortAngleDist(ref_line[i0].heading, ref_line[i1].heading) * a;
    p.s = ref_line[i0].s * ia + ref_line[i1].s * a;
    p.k = ref_line[i0].k * ia + ref_line[i1].k * a;
    p.v_max = ref_line[i0].v_max * ia + ref_line[i1].v_max * a;
    p.d_left = ref_line[i0].d_left * ia + ref_line[i1].d_left * a;
    p.d_right = ref_line[i0].d_right * ia + ref_line[i1].d_right * a;
    p.semantic = (ia < 0.5) ? ref_line[i0].semantic : ref_line[i1].semantic;

    return p;
}

struct RefLine {

    std::vector<RefLinePoint> points;
    float step_size = 0.0;
    double x_offset = 0.0;
    double y_offset = 0.0;

    RefLine() = default;

    RefLine(RefLinePoint* data, int len, float step_size, bool no_offset = false) : step_size{step_size} {

        points.resize(len);
        std::memcpy(points.data(),
                    data,
                    sizeof(RefLinePoint) * len);

        if (!no_offset) {
            double x_min = INFINITY;
            double x_max = -INFINITY;
            double y_min = INFINITY;
            double y_max = -INFINITY;

            for (RefLinePoint& p : points) {
                x_min = min(x_min, p.x);
                x_max = max(x_max, p.x);
                y_min = min(y_min, p.y);
                y_max = max(y_max, p.y);
            }

            x_offset = x_min + (x_max - x_min) / 2.0;
            y_offset = y_min + (y_max - y_min) / 2.0;
            for (RefLinePoint& p : points) {
                p.x -= x_offset;
                p.y -= y_offset;
            }
        }
    }

    const RefLinePoint& operator [](const int i) const {
        return points[i];
    }

    RefLinePoint lerp(const float s) const {
        return interpRefLine(points.data(), points.size(), step_size, s);
    }
};

bool pointInPolygon(double x,
                    double y,
                    array_like<double>& argPolygon);

bool intersectPolygons(array_like<double>& polyArray0,
                       array_like<double>& polyArray1);

bool intersectPolygons(std::vector<vec<2>>& poly0,
                       std::vector<vec<2>>& poly1);

bool intersectPolygons(std::span<vec<2>>& poly0,
                       std::span<vec<2>>& poly1);

std::vector<vec<2>> convexHull(std::vector<vec<2>> ps);

/**
 * Represents a projection on a line strip with additional info.
 */
struct Projection {

    int64_t start = 0;
    int64_t end = 0;
    double alpha = 0;

    int64_t index = 0;
    vec<2> point = vec<2>::Zero();

    double distance = 0.0;
    double arc_len = 0.0;

    double angle = 0.0;
    vec<2> tangent = vec<2>::Zero();

    bool in_bounds = false;
};

Projection project(std::span<vec<2>> points,
                   vec<2> position,
                   bool closed);

std::vector<vec<5>> resample(array_like<double>& argPoints,
                             double samplingDist,
                             size_t steps,
                             size_t startIndex);

struct PredictionPoint {

    double t = 0.0;
    double x = 0.0;
    double y = 0.0;
    double heading = 0.0;
    double v = 0.0;
};

PredictionPoint interpPrediction(std::vector<PredictionPoint>& pred, double t);

template<int X, int U>
void lqrSmoother(vec<X> x0,
                 std::vector<vec<X>>& x_ref,
                 mat<X, X> fx,
                 mat<X, U> fu,
                 std::vector<mat<X, X>>& lxx,
                 std::vector<mat<U, U>>& luu,
                 std::vector<vec<X>>& xs,
                 std::vector<vec<U>>& us) {

    const int H = x_ref.size();

    std::vector<vec<U>> k(H);
    std::vector<mat<U, X>> K(H);

    mat<X, X> Vxx = lxx.back();
    vec<X> Vx = -lxx.back() * x_ref.back();

    for (int i = H-2; i >= 0; --i) {
        vec<X> lx = -lxx[i] * x_ref[i];

        vec<X> Qx = lx + fx.transpose() * Vx;
        vec<U> Qu = fu.transpose() * Vx;
        mat<X, X> Qxx = lxx[i] + fx.transpose() * Vxx * fx;
        mat<U, U> Quu = luu[i] + fu.transpose() * Vxx * fu;
        mat<U, X> Qux = fu.transpose() * Vxx * fx;

        mat<U, U> W = -Quu.inverse();
        k[i] = W * Qu;
        K[i] = W * Qux;

        Vx = Qx + K[i].transpose() * Quu * k[i]
                + K[i].transpose() * Qu
                + Qux.transpose() * k[i];

        Vxx = Qxx + K[i].transpose() * Quu * K[i];
        mat<X, X> tmpxx = K[i].transpose() * Qux;
        Vxx += tmpxx + tmpxx.transpose();
    }

    xs.resize(H);
    us.resize(H);

    xs[0] = x0;
    for (int i = 0; i < H-1; ++i) {
        us[i] = K[i] * xs[i] + k[i];
        xs[i+1] = fx * xs[i] + fu * us[i];
    }
}

template<int X, int U>
void lqrSmoother(vec<X> x0,
                 std::vector<vec<X>>& x_ref,
                 mat<X, X> fx,
                 mat<X, U> fu,
                 mat<X, X> lxx,
                 mat<U, U> luu,
                 std::vector<vec<X>>& xs,
                 std::vector<vec<U>>& us) {

    const int H = x_ref.size();

    std::vector<mat<X, X>> lxxs(H);
    std::vector<mat<U, U>> luus(H);
    for (int i = 0; i < H; ++i) {
        lxxs[i] = lxx;
        luus[i] = luu;
    }

    lqrSmoother(x0, x_ref, fx, fu, lxxs, luus, xs, us);
}

std::vector<vec<2>> smoothPath(std::span<vec<2>> path,
                               double ds,
                               double w_v,
                               double w_a,
                               double w_j,
                               bool closed);

void loadUtilsBindings(pybind11::module& m);

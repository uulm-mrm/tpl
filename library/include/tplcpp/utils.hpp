#pragma once

#include "tplcpp/binding_helpers.hpp"

bool pointInPolygon(double x,
                    double y,
                    array_like<double>& argPolygon);

bool intersectPolygons(array_like<double>& polyArray0,
                       array_like<double>& polyArray1);

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

Projection project(array_like<double>& argPoints,
                   vec<2> position,
                   double loop_closing_dist);

std::vector<vec<5>> resample(array_like<double>& argPoints,
                             double samplingDist,
                             size_t steps,
                             size_t startIndex);

void loadUtilsBindings(pybind11::module& m);

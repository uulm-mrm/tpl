#pragma once

#include <vector>
#include <string>

#include "tplcpp/binding_helpers.hpp"
#include "tplcpp/poly_sampling.hpp"
#include "tplcpp/poly_interp.cuh"
#include "tplcpp/dyn_prog/env.cuh"
#include "tplcpp/utils.hpp"

namespace py = pybind11;

struct PolyLatSamplingParams {

    double ds = 1.0;

    double s_end_min = 10.0;
    double s_end_max = 70.0;
    double s_end_step = 10.0;

    double length = 200.0;

    double lane_width = 5.0;
    double d_step = 1.0;

    double k_j = 0.1;
    double k_t = 0.1;
    double trg_d = 0.0;
    double k_d = 1.0;
    double k_d_d = 1.0;
    double k_v = 1.0;
    double k_lat = 1.0;
    double k_lon = 1.0;

    double k_overtake_right = 1.0;

    double a_max = 2.0;
    double k_max = 1.0;

    double a_lat_max = 2.5;

    double rear_axis_to_rear = 0.0;
    double rear_axis_to_front = 0.0;
    double width_ego = 0.0;
};

struct PolyLatSamplingTrajPoint {

    double t = 0.0;
    double d = 0.0;
    double d_d = 0.0;
    double d_dd = 0.0;
    double d_ddd = 0.0;
    double s = 0.0;
    double v = 0.0;

    double x = 0.0;
    double y = 0.0;
    double yaw = 0.0;
    double dist = 0.0;
    double k = 0.0;
};

struct PolyLatSamplingTraj {

    std::vector<PolyLatSamplingTrajPoint> points;

    PolyQuintic poly;

    double cost = 0.0;

    bool collides = false;
    double dist_closest_collision = 0.0;
};

struct PolyLatSamplingPlanner {

    std::vector<Obstacle> obstacles;
    std::vector<double> dists;

    void addObstacle(double x, double y, double yaw, double v, std::string evade, array_like<double>& hull);

    PolyLatSamplingTraj update(
            PolyLatSamplingTrajPoint& c, 
            PolyRefLinePoint* ref_line,
            size_t ref_line_len,
            DynProgEnvironment& env,
            PolyLatSamplingParams& params);
};

void loadPolyLatSamplingPlannerBindings(py::module m);

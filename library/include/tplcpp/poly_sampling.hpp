#pragma once

#include <vector>
#include <string>

#include "tplcpp/binding_helpers.hpp"
#include "tplcpp/utils.hpp"

namespace py = pybind11;

struct PolyRefLinePoint {

    double x = 0.0;
    double y = 0.0;
    double heading = 0.0;
    double s = 0.0;
    double k = 0.0;
    double v_max = 0.0;
};

PolyRefLinePoint interpPolyRefLine(PolyRefLinePoint* ref_line,
                                   int len,
                                   double step_size,
                                   double s);

struct Obstacle {

    double x = 0.0;
    double y = 0.0;
    double yaw = 0.0;
    double v = 0.0;

    std::string evade;

    array_like<double> hull;
};

struct PolySamplingParams {

    double dt = 0.2;

    double T_min = 4.0;
    double T_max = 5.0;
    double T_step = 1.0;

    double lane_width = 1.0;
    double d_step = 1.0;

    int v_samples = 1;
    double v_step = 1.0;

    double k_j = 0.1;
    double k_t = 0.1;
    double trg_d = 0.0;
    double k_d = 1.0;
    double k_v = 1.0;
    double k_lat = 1.0;
    double k_lon = 1.0;

    double k_overtake_right = 1.0;

    double a_max = 2.0;
    double k_max = 1.0;

    double rear_axis_to_rear = 0.0;
    double rear_axis_to_front = 0.0;
    double width_ego = 0.0;
};

struct PolySamplingTrajPoint {

    double t = 0.0;
    double d = 0.0;
    double d_d = 0.0;
    double d_dd = 0.0;
    double d_ddd = 0.0;
    double s = 0.0;
    double s_d = 0.0;
    double s_dd = 0.0;
    double s_ddd = 0.0;

    double x = 0.0;
    double y = 0.0;
    double yaw = 0.0;
    double ds = 0.0;
    double c = 0.0;
};

struct PolySamplingTraj {

    std::vector<PolySamplingTrajPoint> points;

    double cd = 0.0;
    double cv = 0.0;
    double cf = 0.0;
};

struct PolySamplingPlanner {

    std::vector<Obstacle> obstacles;

    void addObstacle(double x, double y, double yaw, double v, std::string evade, array_like<double>& hull);

    PolySamplingTraj update(
            PolySamplingTrajPoint& c, 
            PolyRefLinePoint* ref_line,
            size_t ref_line_len,
            PolySamplingParams& params);
};

void loadPolySamplingPlannerBindings(py::module m);

#include "tplcpp/poly_interp.cuh"
#include "tplcpp/poly_sampling.hpp"

#include "pybind11/stl.h"

#include <iostream>

namespace py = pybind11;

PolyRefLinePoint interpPolyRefLine(PolyRefLinePoint* ref_line,
                                   int len,
                                   double step_size,
                                   double s) {

    auto shortAngleDist = [](double x, double y) { 

        double a0 = y-x;
        double a1 = y-x + 2*M_PI;
        double a2 = y-x - 2*M_PI;

        if (abs(a1) < abs(a0)) {
            a0 = a1;
        }
        if (abs(a2) < abs(a0)) {
            a0 = a2;
        }

        return a0;
    };

    double alpha = s / step_size;
    int i0 = std::max(0, std::min(len-1, (int)std::floor(alpha)));
    int i1 = std::max(0, std::min(len-1, (int)std::ceil(alpha)));
    alpha -= i0;

    PolyRefLinePoint p{};

    // linear interpolation as an approximation
    p.x = ref_line[i0].x * (1.0 - alpha) + ref_line[i1].x * alpha;
    p.y = ref_line[i0].y * (1.0 - alpha) + ref_line[i1].y * alpha;
    p.s = ref_line[i0].s * (1.0 - alpha) + ref_line[i1].s * alpha;
    p.v_max = ref_line[i0].v_max * (1.0 - alpha) + ref_line[i1].v_max * alpha;

    p.heading = ref_line[i0].heading + shortAngleDist(ref_line[i0].heading, ref_line[i1].heading) * alpha;

    // important: constant curvature
    p.k = ref_line[i0].k;

    return p;
}

void PolySamplingPlanner::addObstacle(double x, double y, double yaw, double v, std::string evade, array_like<double>& hull) {

    Obstacle& o = obstacles.emplace_back();

    o.x = x;
    o.y = y;
    o.yaw = yaw;
    o.v = v;
    o.hull = hull;
    o.evade = evade;
}

PolySamplingTraj PolySamplingPlanner::update(
        PolySamplingTrajPoint& c, 
        PolyRefLinePoint* ref_line,
        size_t ref_line_len,
        PolySamplingParams& params) {

    std::vector<PolySamplingTraj> trajs;

    for (double di = -params.lane_width; di < params.lane_width; di += params.d_step) {
        for (double Ti = params.T_min; Ti < params.T_max; Ti += params.T_step) {

            PolySamplingTraj traj_lat;
            PolyQuintic lat_qp(0.0, c.d, c.d_d, c.d_dd, Ti, di, 0.0, 0.0);

            for (double t = 0; t < params.T_max; t += params.dt) {
                PolySamplingTrajPoint& p = traj_lat.points.emplace_back();
                p.t = t;
                p.d = lat_qp.f(t);
                p.d_d = lat_qp.df(t);
                p.d_dd = lat_qp.ddf(t);
                p.d_ddd = lat_qp.dddf(t);
            }
            
            double v_start = int(std::round(c.s_d / params.v_step)) * params.v_step;
            double v_min = v_start - params.v_step * params.v_samples;
            double v_max = v_start + params.v_step * params.v_samples;

            for (double tv = v_min; tv <= v_max; tv += params.v_step) {
                PolySamplingTraj& traj = trajs.emplace_back();
                traj = traj_lat;

                PolyQuartic lon_qp(0.0, c.s, c.s_d, c.s_dd, Ti, tv, 0.0);

                double Jp = 0.0;
                double Js = 0.0;

                for (size_t i = 0; i < traj.points.size(); ++i) {

                    double t = params.dt * i;
                    PolySamplingTrajPoint& p = traj.points[i];

                    p.s = lon_qp.f(t);
                    p.s_d = lon_qp.df(t);
                    p.s_dd = lon_qp.ddf(t);
                    p.s_ddd = lon_qp.dddf(t);

                    Jp += p.d_ddd*p.d_ddd;
                    Js += p.s_ddd*p.s_ddd;
                }

                // encourage the planner to overtake on the left side
                double Jright = 0.0;
                for (size_t i = 0; i < traj.points.size(); ++i) {
                    PolySamplingTrajPoint& p = traj.points[i];
                    if (p.d < 0.0) {
                        Jright += -p.d;
                    }
                }

                // drive as fast as possible, while respecting velocity limits
                double final_v_diff = 100.0 - traj.points.back().s_d;
                double final_d = params.trg_d - traj.points.back().d;

                traj.cd = params.k_j * Jp + params.k_t * Ti + params.k_d * final_d*final_d + params.k_overtake_right * Jright;
                traj.cv = params.k_j * Js + params.k_t * Ti + params.k_v * final_v_diff*final_v_diff;
                traj.cf = params.k_lat * traj.cd + params.k_lon * traj.cv;
            }
        }
    }

    auto shortAngleDist = [](double x, double y) { 

        double a0 = y-x;
        double a1 = y-x + 2*M_PI;
        double a2 = y-x - 2*M_PI;

        if (abs(a1) < abs(a0)) {
            a0 = a1;
        }
        if (abs(a2) < abs(a0)) {
            a0 = a2;
        }

        return a0;
    };

    // convert to cartesian coordinates

    for (PolySamplingTraj& traj : trajs) {

        size_t len_traj = traj.points.size();

        for (int i = 0; i < len_traj; ++i) {
            PolySamplingTrajPoint& p = traj.points[i];
            double heading_frenet = atan(p.d_d / p.s_d);

            PolyRefLinePoint ref = interpPolyRefLine(ref_line,
                                                     ref_line_len,
                                                     0.5,
                                                     p.s);

            double n_x = -sin(ref.heading);
            double n_y = cos(ref.heading);

            p.x = ref.x + n_x * p.d;
            p.y = ref.y + n_y * p.d;
            p.yaw = heading_frenet + ref.heading;
        }

        // recover curvature and acceleration with finite difference approximation

        for (int i = 1; i < len_traj; ++i) {
            PolySamplingTrajPoint& cp0 = traj.points[i-1];
            PolySamplingTrajPoint& cp1 = traj.points[i];

            double dx = cp1.x - cp0.x;
            double dy = cp1.y - cp0.y;
            double ds = sqrt(dx*dx + dy*dy);

            cp0.ds = ds;
            cp0.c = shortAngleDist(cp0.yaw, cp1.yaw) / ds;
        }

        if (len_traj > 1) {
            traj.points[len_traj-1].c = traj.points[len_traj-2].c;
        }
    }

    // check constraints
    
    PolySamplingTraj bestTraj;
    double bestTrajCost = INFINITY;

    double constrPenalty = 10.0e6;

    mat<5, 2> hull_ego;
    hull_ego(0, 0) = -params.rear_axis_to_rear;
    hull_ego(0, 1) = -params.width_ego / 2.0;
    hull_ego(1, 0) = params.rear_axis_to_front;
    hull_ego(1, 1) = -params.width_ego / 2.0;
    hull_ego(2, 0) = params.rear_axis_to_front;
    hull_ego(2, 1) = params.width_ego / 2.0;
    hull_ego(3, 0) = -params.rear_axis_to_rear;
    hull_ego(3, 1) = params.width_ego / 2.0;
    hull_ego(4, 0) = -params.rear_axis_to_rear;
    hull_ego(4, 1) = -params.width_ego / 2.0;

    for (PolySamplingTraj& traj : trajs) {

        double cost = traj.cf;

        for (int i = 0; i < traj.points.size(); ++i) {
            PolySamplingTrajPoint& p = traj.points[i];

            PolyRefLinePoint ref = interpPolyRefLine(ref_line,
                                                     ref_line_len,
                                                     0.5,
                                                     p.s);
            if (std::abs(p.s_d) > ref.v_max) {
                cost += constrPenalty * (std::abs(p.s_d) - ref.v_max);
            }
            if (std::abs(p.c) > params.k_max) {
                cost += constrPenalty * (std::abs(p.c) - params.k_max);
            }
            if (std::abs(p.s_dd) > params.a_max) {
                cost += constrPenalty * (std::abs(p.s_dd) - params.a_max);
            }
            if (std::abs(p.d) > 4.0) {
                cost += constrPenalty * (std::abs(p.d) - 4.0);
            }

            mat<2, 2> rot;
            rot(0, 0) = std::cos(p.yaw);
            rot(0, 1) = -std::sin(p.yaw);
            rot(1, 0) = std::sin(p.yaw);
            rot(1, 1) = std::cos(p.yaw);

            mat<5, 2> transHull = hull_ego * rot.transpose();
            for (int k = 0; k < 5; ++k) {
                transHull(k, 0) += p.x;
                transHull(k, 1) += p.y;
            }

            array_like<double> transHullPy = py::cast(transHull);

            // collision checking
            for (Obstacle& o : obstacles) {
                if (intersectPolygons(transHullPy, o.hull)) {
                    cost += constrPenalty;
                }
            }
        }

        if (cost < bestTrajCost) {
            bestTraj = traj;
            bestTrajCost = cost;
        }
    }

    obstacles.clear();

    return bestTraj;
}

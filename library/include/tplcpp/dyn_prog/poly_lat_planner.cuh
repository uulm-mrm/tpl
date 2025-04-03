#pragma once

#include <vector>
#include <string>

#include "tplcpp/utils.hpp"
#include "tplcpp/poly_interp.cuh"
#include "tplcpp/dyn_prog/env.cuh"
#include "tplcpp/dyn_prog/arr_tex_surf.cuh"

struct PolyLatPlannerParams {

    float l_min = -5.0;
    float l_max = 5.0;
    float s_min = 0.0;
    float s_max = 200.0;

    int s_steps = 201;

    float l_dst_min = -5.0;
    float l_dst_max = 5.0;
    float s_dst_min = 10.0;
    float s_dst_max = 70.0;

    int l_dst_steps = 21;
    int s_dst_steps = 13;

    float l_trg = 0.0;

    float w_l = 1.0;
    float w_k = 0.1;
    float w_dl = 0.0;
    float w_ddl = 0.0;
    float w_dddl = 1.0;
    float w_right = 0.0;
    float w_len = 0.0001;

    float k_abs_max = 1.0;
    float a_lat_abs_max = 2.5;

    float width_veh = 2.0;
    float length_veh = 2.0;

    float _s_step_size = 0.0;
    float _l_dst_step_size = 0.0;
    float _s_dst_step_size = 0.0;

    void updateStepSizes() {

        if (s_steps > 1) {
            _s_step_size = (s_max - s_min) / (float)(s_steps - 1);
        }
        if (l_dst_steps > 1) {
            _l_dst_step_size = (l_dst_max - l_dst_min) / (float)(l_dst_steps - 1);
        }
        if (s_dst_steps > 1) {
            _s_dst_step_size = (s_dst_max - s_dst_min) / (float)(s_dst_steps - 1);
        }
    }

    bool operator==(const PolyLatPlannerParams&) const = default;
};

struct PolyLatTrajPoint {

    float t = 0.0;
    float l = 0.0;
    float dl = 0.0;
    float ddl = 0.0;
    float dddl = 0.0;
    float s = 0.0;
    float v = 0.0;

    double x = 0.0;
    double y = 0.0;
    float heading = 0.0;
    double distance = 0.0;
    float k = 0.0;
};

struct PolyLatTraj {

    std::vector<PolyLatTrajPoint> points;
    PolyQuintic poly;
    double cost = 0.0;

    PolyLatTrajPoint lerp(double distance);

    void insertAfterStation(double s, PolyLatTraj& o);

    void updateTimeDistCurv();
};

struct PolyLatPlanner {

    ArrTexSurf<float4> path_nodes;

    PolyLatPlannerParams params;

    ~PolyLatPlanner();

    void clearBuffers();
    void reinitBuffers(PolyLatPlannerParams& ps);

    PolyLatTraj update(
            PolyLatTrajPoint& c, 
            DynProgEnvironment& env);
};

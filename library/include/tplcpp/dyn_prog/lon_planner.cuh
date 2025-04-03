#pragma once

#include "tplcpp/dyn_prog/common.cuh"
#include "tplcpp/dyn_prog/env.cuh"
#include "tplcpp/dyn_prog/arr_tex_surf.cuh"

struct DynProgLonPlannerParams {

    float s_min = 0.0;
    float s_max = 200.0;
    float v_min = 0.0;
    float v_max = 36.0;
    float a_min = -2.0;
    float a_max = 2.0;
    float j_min = -2.0;
    float j_max = 2.0;
    
    int t_steps = 10;
    int s_steps = 201;
    int v_steps = 37;
    int a_steps = 7;

    float dt_start = 1.0;
    float dt = 1.0;

    float _s_step_size = 0.0;
    float _v_step_size = 0.0;
    float _a_step_size = 0.0;

    float time_gap = 1.5;
    float gap_min = 1.0;

    float w_progress = 1.0;
    float w_a = 0.5;
    float w_j = 0.5;
    float w_snap = 0.5;
    float w_safety_dist = 10.0;

    float path_step_size = 0.5;
    int path_steps = 200;

    float width_veh = 2.0;
    float length_veh = 6.0;

    void updateStepSizes() {

        _s_step_size = (s_max - s_min) / (float)(s_steps - 1);
        _v_step_size = (v_max - v_min) / (float)(v_steps - 1);
        _a_step_size = (a_max - a_min) / (float)(a_steps - 1);
    }

    bool operator==(const DynProgLonPlannerParams&) const = default;
};

struct LonState {

    float t = 0.0;

    // station, velocity, acceleration, jerk
    float s = 0.0;
    float v = 0.0;
    float a = 0.0;
    float j = 0.0;

    float cost = 0.0;
    float constr = 0.0;
};

__interop__ LonState lonDynamics(LonState x0, float acc_trg, float dt);

struct LonTraj {
    std::vector<LonState> states;
    LonState state(float t);
};

struct DynProgLonPlanner {

    // node arrays, textures and surfaces
    std::vector<ArrTexSurf<float4>> backwardPassNodes;
    // cuda pointer to forward node array
    LonState* forwardPassNodes = nullptr;
    // cuda pointer to path computed in lat planner
    PathState* path = nullptr;

    ~DynProgLonPlanner();

    DynProgLonPlannerParams params;

    void clearBuffers();
    void reinitBuffers(DynProgLonPlannerParams& ps);

    LonTraj update(LonState initialState,
                   PathState* cpuPath,
                   DynProgEnvironment& env);

    LonTraj reevalTraj(const LonTraj& traj,
                       PathState* cpuPath,
                       DynProgEnvironment& env);
};

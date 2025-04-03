#pragma once

#include "tplcpp/dyn_prog/common.cuh"
#include "tplcpp/dyn_prog/env.cuh"
#include "tplcpp/dyn_prog/arr_tex_surf.cuh"
#include "tplcpp/eigen_helpers.hpp"


struct DynProgLatLonPlannerParams {

    float s_min = 0.0;
    float s_max = 200.0;
    float ds_min = 0.0;
    float ds_max = 36.0;
    float l_min = -5.0;
    float l_max = 5.0;

    float dds_min = -2.0;
    float dds_max = 2.0;
    float dl_min = -2.0;
    float dl_max = 2.0;
    
    int t_steps = 10;
    int s_steps = 201;
    int ds_steps = 37;
    int l_steps = 21;

    float dt = 1.0;
    float dt_start = 1.0;
    float dt_smooth_traj = 0.1;

    float dds_start = 0.0;
    float w_dds_start = 10.0;
    float angle_start = 0.0;
    float w_angle_start = 10.0;

    float l_trg = 0.0;

    float w_progress = 1.0;
    float w_dds = 1.0;
    float w_ddds = 1.0;
    float w_l = 1.0;
    float w_dl = 1.0;
    float w_ddl = 1.0;
    float w_safety_dist = 10.0;
    float w_xing_slow = 1.0;

    float slope_abs_max = 0.8;

    float w_lat_dist = 0.0;
    float d_lat_comf = 2.0;

    float time_gap = 2.0;
    float gap_min = 2.0;

    float t_st_min = -1.0;
    float t_st_max = -1.0;
    float s_st = 0.0;
    float w_spatio_temporal = 10.0;

    float width_veh = 2.0;
    float length_veh = 6.0;
};

struct LatLonCartState {

    double t = 0.0;

    double distance = 0.0;
    double x = 0.0;
    double y = 0.0;
    double v = 0.0;
    double a = 0.0;
    double heading = 0.0;
    double k = 0.0;

    double cost = 0.0;
    double constr = true;

    uint8_t flags_constr = 0;
};

struct LatLonState {

    float t = 0.0;

    // longitudinal position, velocity, acceleration, jerk
    float s = 0.0;
    float ds = 0.0;
    float dds = 0.0;
    float ddds = 0.0;

    // lateral position, velocity, acceleration, jerk
    float l = 0.0;
    float dl = 0.0;
    float ddl = 0.0;
    float dddl = 0.0;

    float cost = 0.0;
    float constr = 0.0;

    // flags indicate which constraint was violated
    uint8_t flags_constr = 0;
};

enum LatLonConstr {
    LAT_LON_CONSTR_OCCUPANCY = 1,
    LAT_LON_CONSTR_VELOCITY = 2,
    LAT_LON_CONSTR_ANGLE = 4,
};

struct LatLonCartTraj {
    std::vector<LatLonCartState> states{1};
    LatLonCartState lerp(float t);
};

struct LatLonTraj {
    std::vector<LatLonState> states{1};
    LatLonState state(float t);
    LatLonState lerp(float t);
};

struct LatDistances {
    float d_left = 0.0;
    float d_right = 0.0;
};

namespace dp_lat_lon_planner_cu {

    __interop__ LatLonState dynamics(const LatLonState& x0, float a, float dl, float t);
};

struct DynProgLatLonPlanner {

    // node arrays, textures and surfaces
    std::vector<ArrTexSurf<float4>> backwardPassNodes;
    // cuda pointer to forward node array
    LatLonState* forwardPassNodes = nullptr;
    // cuda pointer to trajectory buffer for reeval
    LatLonState* reevalNodes = nullptr;
    // cuda pointer to smoothed trajectory 
    LatLonState* smoothNodes = nullptr;
    // cuda pointer to distances along smoothed 
    LatDistances* distNodes = nullptr;

    std::vector<LatDistances> latDists;

    LatLonTraj trajDp;
    LatLonTraj trajDpResampled;
    LatLonTraj trajSmooth;
    LatLonCartTraj trajDpCart;
    LatLonCartTraj trajSmoothCart;

    DynProgLatLonPlanner();
    ~DynProgLatLonPlanner();

    DynProgLatLonPlannerParams params;

    void clearBuffers();
    void reinitBuffers(DynProgLatLonPlannerParams& ps, bool force);

    void updateTrajDp(DynProgEnvironment& env);
    void updateTrajSmooth(DynProgEnvironment& env);
    void updateTrajCart(const RefLine& refLine);

    void frenetToCartesian(const LatLonTraj& traj,
                           const RefLine& refLine,
                           LatLonCartTraj& result);

    LatLonTraj reevalTraj(const LatLonTraj& traj,
                          DynProgEnvironment& env);

    void copyBackwardPassTex(float* dst, int idx_t);
    void queryBackwardPassTex(float* dst, int idx_t, float s, float ds, float l);
};

#include "tplcpp/utils.hpp"
#include "tplcpp/poly_interp.cuh"

struct IdmSamplingParams {

    int steps_t = 100;
    double dt = 0.1;

    double dead_time = 0.0;

    int lat_steps = 2;
    double d_safe_lat = 0.25;
    double d_safe_lat_path = 0.5;
    double d_comf_lat = 1.0;

    double k_stanley = 1.0;
    double v_offset_stanley = 1.0;

    double steer_angle_max = 0.7;
    double steer_rate_max = 0.6;

    double t_vel_lookahead = 2.0;
    double d_safe_min = 1.0;
    double t_headway_desired = 1.0;
    double a_break_comf = 1.5;

    double idm_exp_dcc = 4.0;
    double idm_exp_acc = 4.0;

    double k_p_s = 1.0;
    double k_p_v = 1.0;

    double a_max = 2.0;
    double a_min = -3.0;
    double j_max = 1.5;
    double j_min = -1.5;

    double d_next_inters_point = 1.0e6;

    double width_veh = 0.0;
    double length_veh = 0.0;
    double radius_veh = 0.0;
    double dist_front_veh = 0.0;
    double dist_back_veh = 0.0;
    double wheel_base = 4.0;

    double l_trg = 0.0;
    double w_l = 1.0;
    double w_a = 1.0;
    double w_lat_dist = 1.0;

    double dt_decision = 0.2;

    bool enable_reverse = false;
};

struct IdmSamplingRefState {

    double t = 0.0;
    double x = 0.0;
    double y = 0.0;
    double heading = 0.0;
    double v = 0.0;
    double a = 0.0;
    double s = 0.0;
    double l = 0.0;
    double d_right = 0.0;
    double d_left = 0.0;
};

struct IdmSamplingState {

    double t = 0.0;
    double x = 0.0;
    double y = 0.0;
    double heading = 0.0;
    double steer_angle = 0.0;
    double v = 0.0;
    double a = 0.0;
    double s = 0.0;
    double l = 0.0;
};

struct DynObj {

    std::string id;

    std::vector<PredictionPoint> predictions;

    std::vector<vec<2>> path;
    std::vector<double> dists;

    double radiusHull = 0.0;
    std::vector<vec<2>> hull;
    std::vector<vec<4>> hullProjs;
    std::vector<std::vector<vec<2>>> hullPreds;

    bool on_local_map = false;

    void updatePredGeometry(std::vector<vec<2>>& linestrip);

    PredictionPoint lerp(double t);
    PredictionPoint lerpStation(double s);

    vec<4> lerpHullProj(double t);
    std::vector<vec<2>> lerpHullPred(double t);
};

struct IdmSamplingTraj {

    double l_trg = 0.0;
    double d_stop = 1.0e6;

    double cost = 0.0;
    double cost_distance = 0.0;
    double cost_interaction = 0.0;
    double cost_collision = 0.0;
    bool reverse = false;
    bool invalid = false;

    std::vector<DynObj> dynObjs;

    std::vector<IdmSamplingState> states;
    std::vector<IdmSamplingRefState> refStates;

    IdmSamplingState lerp(double t);
    IdmSamplingRefState lerpRef(double t);
};

struct IdmSamplingPlanner {

    IdmSamplingPlanner();

    std::vector<DynObj> dynObjs;

    IdmSamplingTraj deadTimeTraj;
    IdmSamplingTraj prevTraj;

    std::vector<IdmSamplingTraj> trajs;

    int i_prev = -1;
    int i_min_prev = 0.0;
    double t_decision = 0;

    void reset();

    double getNextStopPoint(const IdmSamplingRefState& state,
                            const RefLine& refLine,
                            double l_trg,
                            IdmSamplingParams& params);

    vec<4> getLeader(const IdmSamplingRefState& state,
                     IdmSamplingTraj& traj,
                     double l_trg,
                     IdmSamplingParams& params);

    void insertDynObj(std::vector<PredictionPoint> predictions,
                      std::vector<vec<2>> hull,
                      bool on_local_map);

    void rolloutTrajectory(IdmSamplingTraj& traj,
                           const RefLine& refLine,
                           double dt_replan,
                           std::vector<vec<2>>& linestrip,
                           IdmSamplingParams& params);

    void evalTrajectory(IdmSamplingTraj& traj,
                        const RefLine& refLine,
                        IdmSamplingParams& params);

    std::vector<vec<2>> getVehicleHull(IdmSamplingState& s,
                                       IdmSamplingParams& params);

    IdmSamplingTraj update(IdmSamplingState initState, 
                           IdmSamplingRefState initRefState,
                           double dt_replan,
                           const RefLine& refLine,
                           IdmSamplingParams& params);
};

#include "tplcpp/dyn_prog/lat_lon_planner.cuh"

#include <cstring>
#include <stdexcept>

namespace dp_lat_lon_planner_cu {

__device__ __constant__ DynProgLatLonPlannerParams params;

__interop__ LatLonState dynamics(const LatLonState& x0, float dds, float dl, float dt) {
    
    LatLonState res;
    res.t = x0.t + dt;
    res.s = fmaxf(x0.s, x0.s + x0.ds*dt + 0.5*dds*dt*dt);
    res.ds = fmaxf(0.0, x0.ds + dds * dt);
    res.dds = dds;
    res.l = x0.l + dl * dt;
    res.dl = dl;

    return res;
}

struct EvalResult {

    float cost;
    float constr;
    uint8_t flags_constr;

    __inline__ __device__ EvalResult()
        : cost{0.0}, constr{0.0}, flags_constr{0} { }

    __inline__ __device__ EvalResult(float cost, float constr) 
        : cost{cost}, constr{constr}, flags_constr{0} { }

    __inline__ __device__ bool operator< (EvalResult const& rhs) const {

        if (constr == rhs.constr) {
            return cost < rhs.cost;
        }

        return constr < rhs.constr;
    }
};

struct Evaluator {

    LatLonState tp;
    EnvironmentGpu& env;
    float dt;
    float d_safety;
    float d_fwd;
    bool on_intersection;
    float4 rp;

    __inline__ __device__ Evaluator(const LatLonState _tp,
                                    EnvironmentGpu& env,
                                    float dt) : 
                env{env},
                dt{dt} {

        tp = _tp;

        rp = env.ref_line.texLerp(tp.s);
        on_intersection = roundf(rp.w) == 1.0f;
    }

    __inline__ __device__ EvalResult finalState() {

        EvalResult res;

        if (on_intersection) {
            res.cost += params.w_xing_slow;
        }

        float3 mid = getMid();
        res.cost += params.w_l * sq(mid.x - tp.l);

        return res;
    }

    __inline__ __device__ float3 getMid() {

        const float l_step_size = (params.l_max - params.l_min) / (float)(params.l_steps-1);
        const float mean_dist = fmaxf(params.length_veh * 0.5, tp.ds * dt);

        float l_left = 0.0;
        float l_right = 0.0;

        for (int i = 0; i < params.l_steps; ++i) {
            const float l = tp.l + i * l_step_size;
            const float d = env.interpDistMapLon(tp.t, tp.s, l).x;
            if (d < mean_dist) {
                l_left = l;
                break;
            }
        }
        for (int i = 0; i < params.l_steps; ++i) {
            const float l = tp.l - i * l_step_size;
            const float d = env.interpDistMapLon(tp.t, tp.s, l).x;
            if (d < mean_dist) {
                l_right = l;
                break;
            }
        }

        float3 res;
        float mid = l_right + (l_left - l_right) * 0.5;
        res.y = fminf(l_right + params.d_lat_comf, mid);
        res.z = fmaxf(l_left - params.d_lat_comf, mid);

        if (params.l_trg < l_right) {
            res.x = res.y;
        } else if (params.l_trg > l_left) {
            res.x = res.z;
        } else {
            res.x = params.l_trg;
        }

        return res;
    }

    __inline__ __device__ EvalResult evalState() {

        EvalResult res;

        const float v_max_ref = rp.x;

        const float d_left_ref = (rp.y - params.width_veh * 0.5);
        const float d_right_ref = -(rp.z - params.width_veh * 0.5);

        res.cost += 1000.0 * fmaxf(0.0, tp.l - d_left_ref);
        res.cost += 1000.0 * fmaxf(0.0, d_right_ref - tp.l);

        float3 mid = getMid();
        res.cost += params.w_l * sq(mid.x - tp.l);
        if (tp.l < mid.y) {
            res.cost += params.w_lat_dist * sq(mid.y - tp.l);
        }
        if (tp.l > mid.z) {
            res.cost += params.w_lat_dist * sq(mid.z - tp.l);
        }

        res.cost += params.w_progress * (1000.0 - tp.s);

        if (tp.ds > v_max_ref) {
            res.constr += tp.ds - v_max_ref;
            res.flags_constr |= LAT_LON_CONSTR_VELOCITY;
        }

        if (tp.t < params.t_st_min) {
            res.cost += params.w_spatio_temporal * fmaxf(0.0, tp.s - params.s_st);
        }
        if (tp.t > params.t_st_max) {
            res.cost += params.w_spatio_temporal * fmaxf(0.0, params.s_st - tp.s);
        }

        return res;
    }

    __inline__ __device__ EvalResult evalAction(float dds,
                                                float dl,
                                                const LatLonState& tn) {

        EvalResult res;

        const float s_change = tn.s - tp.s;
        const float l_change = tn.l - tp.l;

        const float slope = fabsf(l_change / s_change);

        if (slope > params.slope_abs_max) {
            res.constr += fabsf(slope - params.slope_abs_max) * 1000.0;
            res.flags_constr |= LAT_LON_CONSTR_ANGLE;
        }

        if (s_change > d_fwd) { 
            res.constr += s_change - d_fwd;
            res.flags_constr |= LAT_LON_CONSTR_OCCUPANCY;
        }

        res.cost += params.w_safety_dist * fmaxf(0.0, s_change - d_safety);

        const float ddds = tn.dds - dds;
        const float ddl = tn.dl - dl;

        res.cost += params.w_dds * sq(dds * dt);
        res.cost += params.w_ddds * sq(ddds);
        res.cost += params.w_dl * sq(dl * dt);
        res.cost += params.w_ddl * sq(ddl);

        return res;
    }

    template <int SAMPLES_DDS, int SAMPLES_DL, bool interp = false>
    __inline__ __device__ EvalResult findBestAction(float& dds_best,
                                                    float& dl_best,
                                                    EvalResult& minRes,
                                                    const ArrTexSurf<float4>& nodesNext) {

        EvalResult totalMinRes(INFINITY, INFINITY);

        const float dl_min = params.dl_min; //fmaxf(params.dl_min, params.dl_min / sqrtf(tp.ds));
        const float dl_max = params.dl_max; //fminf(params.dl_max, params.dl_max / sqrtf(tp.ds));

        const float step_size_dl = (dl_max - dl_min) / (float)(SAMPLES_DL-1);

        constexpr const int SAMPLES_DL_2 = SAMPLES_DL / 2;

        float d_fwd_left = 0.0;
        float d_fwd_right = 0.0;

        for (int dl_idx = 0; dl_idx < SAMPLES_DL; ++dl_idx) {

            float dl = 0.0;

            if (dl_idx == 0) {
                // center
                const float d = env.interpDistMapLon(tp.t, tp.s, tp.l).x;
                d_fwd_left = d;
                d_fwd_right = d;
                d_fwd = d;
                dl = 0.0;
            } else if (dl_idx <= SAMPLES_DL_2) {
                // left 
                dl = step_size_dl * dl_idx;
                const float d = env.interpDistMapLon(tp.t, tp.s, tp.l + dl * dt).x;
                d_fwd_left = fminf(d, d_fwd_left);
                d_fwd = d_fwd_left;
            } else {
                // right 
                dl = -step_size_dl * (dl_idx - SAMPLES_DL_2);
                const float d = env.interpDistMapLon(tp.t, tp.s, tp.l + dl * dt).x;
                d_fwd_right = fminf(d, d_fwd_right);
                d_fwd = d_fwd_right;
            }

            d_fwd -= params.length_veh * 0.5;
            d_safety = d_fwd
                     - params.gap_min
                     - tp.ds * params.time_gap;

            for (int dds_idx = 0; dds_idx < SAMPLES_DDS; ++dds_idx) {

                const float dds = params.dds_min + (params.dds_max - params.dds_min)
                               * ((float)dds_idx / (float)(SAMPLES_DDS-1));

                LatLonState tn = dynamics(tp, dds, dl, dt);

                // evaluate value function via texture lookup
                float4 nn;
                if constexpr (interp) {
                    nn = nodesNext.lerp_cached_scaled(tn.s, tn.ds, tn.l);
                } else {
                    nn = nodesNext.get_cached_scaled(tn.s, tn.ds, tn.l);
                }

                const float cost_next = nn.x;
                const float constr_next = nn.y;
                tn.dds = nn.z;
                tn.dl = nn.w;

                EvalResult res = evalAction(dds, dl, tn);
                EvalResult totalRes(res.cost + cost_next,
                                    res.constr + constr_next);

                if (totalRes < totalMinRes) {
                    totalMinRes = totalRes;
                    // store the costs without added value function cost/constr
                    // this way we can later reevaluate and compare the costs
                    // without recomputing the value function
                    minRes = res;
                    dds_best = dds;
                    dl_best = dl;
                }
            }
        }

        return totalMinRes;
    }
};

__global__ void backwardStep(size_t idx_t,
                             ArrTexSurf<float4> nodes,
                             const ArrTexSurf<float4> nodesNext,
                             EnvironmentGpu env,
                             const LatLonState* traj) {

    size_t idx_s = threadIdx.x;
    size_t idx_ds = blockIdx.x;
    size_t idx_l = blockIdx.y;

    if (idx_s >= params.s_steps || idx_ds >= params.ds_steps || idx_l >= params.l_steps) {
        return;
    }

    LatLonState tp;
    tp.t = params.dt_start + (idx_t-1) * params.dt;
    tp.s = nodes.x_min + idx_s * nodes.step_size_x;
    tp.ds = nodes.y_min + idx_ds * nodes.step_size_y;
    tp.l = nodes.z_min + idx_l * nodes.step_size_z;

    Evaluator evaluator(tp, env, params.dt);
    EvalResult resState = evaluator.evalState();
    tp.cost = resState.cost;
    tp.constr = resState.constr;

    EvalResult totalResAction;

    if (idx_t < params.t_steps-1) {
        EvalResult resAction;
        totalResAction = evaluator.findBestAction<7, 7>(
                tp.dds, tp.dl, resAction, nodesNext);
        tp.cost += resAction.cost;
        tp.constr += resAction.constr;
    } else {
        totalResAction = evaluator.finalState();
        tp.cost += totalResAction.cost;
        tp.constr += totalResAction.constr;
    }

    float4 node;
    node.x = resState.cost + totalResAction.cost;
    node.y = resState.constr + totalResAction.constr;
    node.z = tp.dds;
    node.w = tp.dl;
    nodes.set(idx_s, idx_ds, idx_l, node);
}

__global__ void forwardStep(size_t idx_t,
                            const ArrTexSurf<float4> nodesNext,
                            EnvironmentGpu env,
                            LatLonState* traj,
                            float dt) {

    if (threadIdx.x != 0) {
        return;
    }

    LatLonState& tp = traj[idx_t];

    Evaluator evaluator(tp, env, dt);
    EvalResult resState = evaluator.evalState();
    tp.cost = resState.cost;
    tp.constr = resState.constr;
    tp.flags_constr = resState.flags_constr;

    if (idx_t < params.t_steps-1) {
        EvalResult resAction;
        EvalResult totalResAction = evaluator.findBestAction<21, 21, true>(
                tp.dds, tp.dl, resAction, nodesNext);
        tp.cost += resAction.cost;
        tp.constr += resAction.constr;
        tp.flags_constr |= resAction.flags_constr;
        traj[idx_t+1] = dynamics(tp, tp.dds, tp.dl, dt);
    }
}

__global__ void reevalTraj(EnvironmentGpu env, LatLonState* traj, int idx, int len) {

    LatLonState& tp = traj[idx];

    float dt;
    if (idx == len - 1) {
        dt = 0.0;
    } else {
        dt = traj[idx+1].t - tp.t;
    }

    Evaluator evaluator(tp, env, dt);
    EvalResult resState = evaluator.evalState();
    tp.cost = resState.cost;
    tp.constr = resState.constr;
    tp.flags_constr = resState.flags_constr;

    const float l_next = tp.l + dt * tp.dl;
    const float l_step_size = (params.l_max - params.l_min) / (float)(params.l_steps-1);
    const float l_dist = l_next - tp.l;

    // rounded up steps to force step_size to be smaller than l_step_size
    const int steps = ceilf(fabsf(l_dist) / l_step_size);
    const float step_size = l_dist / (float)(steps);

    float d_fwd = env.interpDistMapLon(tp.t, tp.s, tp.l).x;
    for (int i = 0; i < steps; ++i) {
        const float l = tp.l + i * step_size;
        const float d = env.interpDistMapLon(tp.t, tp.s, l).x;
        d_fwd = fminf(d, d_fwd);
    }

    evaluator.d_fwd = d_fwd;
    evaluator.d_safety = d_fwd
                       - params.gap_min
                       - tp.ds * params.time_gap;

    if (idx < len - 1) {
        const LatLonState& tn = traj[idx+1];
        EvalResult resAction = evaluator.evalAction(tp.dds, tp.dl, tn);
        tp.cost += resAction.cost;
        tp.constr += resAction.constr;
        tp.flags_constr |= resAction.flags_constr;
    }
}

__global__ void queryBackwardPassTex(
        float4* cudaRes,
        float s,
        float ds,
        float l,
        ArrTexSurf<float4> nodes) {

    *cudaRes = nodes.get_cached_scaled(s, ds, l);
}

__global__ void getLatDistances(
        EnvironmentGpu env,
        LatLonState* traj,
        LatDistances* dists) {

    int idx = threadIdx.x;
    int len = blockDim.x;

    if (idx >= len) {
        return;
    }
}

};

LatLonState LatLonTraj::state(float t) {

    InterpVars i(states, &LatLonState::t, t);

    float t_rel = t - states[i.i_prev].t;
    float dds = states[i.i_prev].dds;
    float dl = states[i.i_prev].dl;

    return dp_lat_lon_planner_cu::dynamics(states[i.i_prev], dds, dl, t_rel);
}

LatLonState LatLonTraj::lerp(float t) {

    InterpVars i(states, &LatLonState::t, t);

    float ai = 1.0f - i.a;

    LatLonState res;
    res.t = states[i.i_prev].t * ai + states[i.i_next].t * i.a;
    res.s = states[i.i_prev].s * ai + states[i.i_next].s * i.a;
    res.ds = states[i.i_prev].ds * ai + states[i.i_next].ds * i.a;
    res.dds = states[i.i_prev].dds * ai + states[i.i_next].dds * i.a;
    res.ddds = states[i.i_prev].ddds * ai + states[i.i_next].ddds * i.a;
    res.l = states[i.i_prev].l * ai + states[i.i_next].l * i.a;
    res.dl = states[i.i_prev].dl * ai + states[i.i_next].dl * i.a;
    res.ddl = states[i.i_prev].ddl * ai + states[i.i_next].ddl * i.a;
    res.dddl = states[i.i_prev].dddl * ai + states[i.i_next].dddl * i.a;
    res.cost = states[i.i_prev].cost * ai + states[i.i_next].cost * i.a;
    res.constr = states[i.i_prev].constr * ai + states[i.i_next].constr * i.a;

    return res;
}

LatLonCartState LatLonCartTraj::lerp(float t) {

    InterpVars i(states, &LatLonCartState::t, t);

    float ai = 1.0f - i.a;

    LatLonCartState res;
    res.t = states[i.i_prev].t * ai + states[i.i_next].t * i.a;
    res.distance = states[i.i_prev].distance * ai + states[i.i_next].distance * i.a;
    res.x = states[i.i_prev].x * ai + states[i.i_next].x * i.a;
    res.y = states[i.i_prev].y * ai + states[i.i_next].y * i.a;
    res.v = states[i.i_prev].v * ai + states[i.i_next].v * i.a;
    res.a = states[i.i_prev].a * ai + states[i.i_next].a * i.a;
    res.heading = states[i.i_prev].heading + shortAngleDist(states[i.i_prev].heading, states[i.i_next].heading) * i.a;
    res.k = states[i.i_prev].k * ai + states[i.i_next].k * i.a;
    res.cost = states[i.i_prev].cost * ai + states[i.i_next].cost * i.a;
    res.constr = states[i.i_prev].constr * ai + states[i.i_next].constr * i.a;

    return res;
}

DynProgLatLonPlanner::DynProgLatLonPlanner() {

    reinitBuffers(params, true);
}

DynProgLatLonPlanner::~DynProgLatLonPlanner() {

    clearBuffers();
}

void DynProgLatLonPlanner::clearBuffers() {

    if (forwardPassNodes != nullptr) {
        cudaFree(forwardPassNodes);
    }
    forwardPassNodes = nullptr;

    if (reevalNodes != nullptr) {
        cudaFree(reevalNodes);
    }
    reevalNodes = nullptr;

    for (ArrTexSurf<float4>& nodes : backwardPassNodes) {
        nodes.release();
    }
    backwardPassNodes.clear();
}

void DynProgLatLonPlanner::reinitBuffers(DynProgLatLonPlannerParams& ps, bool force = false) {

    bool fullReinitRequired = force
        || ps.t_steps != params.t_steps
        || ps.s_steps != params.s_steps
        || ps.ds_steps != params.ds_steps
        || ps.l_steps != params.l_steps;

    params = ps;

    // copy params to gpu
    {
        cudaError_t err = cudaMemcpyToSymbol(dp_lat_lon_planner_cu::params,
                                             &params,
                                             sizeof(DynProgLatLonPlannerParams));
        checkCudaError(err, "copying env params failed");
    }

    if (!fullReinitRequired) {
        return;
    }

    // need to reallocate

    clearBuffers();

    // trajectory points for forward pass

    trajDp.states.resize(params.t_steps);
    trajSmooth.states.resize(1);
    trajSmoothCart.states.resize(1);

    size_t fwdArrSize = sizeof(LatLonState) * params.t_steps;
    cudaError_t err = cudaMalloc(&forwardPassNodes, fwdArrSize);
    checkCudaError(err, "forward nodes allocation failed");

    size_t smoothArrSize = sizeof(LatLonState)
                           * (params.t_steps - 1)
                           * params.dt / params.dt_smooth_traj;
    err = cudaMalloc(&reevalNodes, smoothArrSize);
    checkCudaError(err, "reeval nodes allocation failed");
    err = cudaMalloc(&smoothNodes, smoothArrSize);
    checkCudaError(err, "smooth nodes allocation failed");

    size_t distArrSize = sizeof(LatDistances)
                         * (params.t_steps - 1)
                         * params.dt / params.dt_smooth_traj;
    err = cudaMalloc(&distNodes, distArrSize);
    checkCudaError(err, "dist nodes allocation failed");

    // arrays, texture, surfaces for backward pass

    backwardPassNodes.resize(params.t_steps);

    for (ArrTexSurf<float4>& nodes : backwardPassNodes) {
        nodes.reinit(params.s_steps,
                     params.ds_steps,
                     params.l_steps,
                     cudaFilterModePoint);
    }
}

void DynProgLatLonPlanner::updateTrajDp(DynProgEnvironment& env) {

    trajDp.states.resize(params.t_steps);

    cudaMemcpy(forwardPassNodes,
               trajDp.states.data(),
               sizeof(LatLonState) * params.t_steps,
               cudaMemcpyHostToDevice);

    checkCudaError(cudaGetLastError(), "copying trajectory to gpu failed");

    // rescale cost nodes

    for (int i = params.t_steps-1; i > 0; --i) {
        ArrTexSurf<float4>& nodes = backwardPassNodes[i];
        //float t = params.dt_start + params.dt * (i-1);
        //float ds_min = (float)(trajDp.states[0].ds + t * params.dds_min);
        //float ds_max = (float)(trajDp.states[0].ds + t * params.dds_max);
        //float s_min = (float)(trajDp.states[0].s + t * trajDp.states[0].ds + 0.5 * t*t * params.dds_min);
        //float s_max = (float)(trajDp.states[0].s + t * trajDp.states[0].ds + 0.5 * t*t * params.dds_max);
        nodes.rescale(params.s_min, params.s_max,
                      params.ds_min, params.ds_max,
                      params.l_min, params.l_max);
    }

    // backward pass

    for (int i = params.t_steps-1; i > 0; --i) {
        dim3 gridSize(params.ds_steps, params.l_steps, 1);
        dim3 blockSize(params.s_steps, 1, 1);
        dp_lat_lon_planner_cu::backwardStep<<<gridSize, blockSize>>>(
                i,
                backwardPassNodes[i],
                backwardPassNodes[min((int)params.t_steps-1, i+1)],
                env.envGpu,
                forwardPassNodes);
    }

    cudaDeviceSynchronize();

    checkCudaError(cudaGetLastError(), "lon planner backward pass failed");

    // forward pass

    for (int i = 0; i < params.t_steps; ++i) {
        dim3 gridSize(1, 1, 1);
        dim3 blockSize(1, 1, 1);

        float dt = params.dt;
        if (i == 0) {
            dt = params.dt_start;
        }

        dp_lat_lon_planner_cu::forwardStep<<<gridSize, blockSize>>>(
                i,
                backwardPassNodes[min((int)params.t_steps-1, i+1)],
                env.envGpu,
                forwardPassNodes,
                dt);
    }

    cudaDeviceSynchronize();
    checkCudaError(cudaGetLastError(), "lon planner forward pass failed");

    cudaMemcpy(trajDp.states.data(),
               forwardPassNodes,
               sizeof(LatLonState) * params.t_steps,
               cudaMemcpyDeviceToHost);

    checkCudaError(cudaGetLastError(), "copying trajectory to host failed");
}

void DynProgLatLonPlanner::updateTrajSmooth(DynProgEnvironment& env) {

    // resample trajectory

    int resample_steps = trajDp.states.back().t / params.dt_smooth_traj;
    trajDpResampled.states.resize(resample_steps);
    for (int i = 0; i < resample_steps; ++i) {
        const float t = i * params.dt_smooth_traj;
        trajDpResampled.states[i] = trajDp.state(t);
    }

    if (trajDp.states.size() == 0) {
        return;
    }

    std::vector<vec<4>> x_ref_s(trajDpResampled.states.size());
    for (size_t i = 0; i < trajDpResampled.states.size(); ++i) {
        LatLonState& tp = trajDpResampled.states[i];
        x_ref_s[i][0] = tp.s;
        x_ref_s[i][1] = tp.ds;
        x_ref_s[i][2] = 0.0; // tp.dds;
        x_ref_s[i][3] = 0.0; // tp.ddds;
    }
    std::vector<vec<4>> x_ref_l(trajDpResampled.states.size());
    for (size_t i = 0; i < trajDpResampled.states.size(); ++i) {
        LatLonState& tp = trajDpResampled.states[i];
        x_ref_l[i][0] = tp.l;
        x_ref_l[i][1] = 0.0; // tp.dl;
        x_ref_l[i][2] = 0.0; // tp.ddl;
        x_ref_l[i][3] = 0.0; // tp.dddl;
    }

    vec<4> x0_s;
    x0_s[0] = trajSmooth.states[0].s;
    x0_s[1] = trajSmooth.states[0].ds;
    x0_s[2] = trajSmooth.states[0].dds;
    x0_s[3] = trajSmooth.states[0].ddds;

    vec<4> x0_l;
    x0_l[0] = trajSmooth.states[0].l;
    x0_l[1] = trajSmooth.states[0].dl;
    x0_l[2] = trajSmooth.states[0].ddl;
    x0_l[3] = trajSmooth.states[0].dddl;

    mat<4, 4> A = mat<4, 4>::Identity();
    A(0, 1) = params.dt_smooth_traj;
    A(1, 2) = params.dt_smooth_traj;
    A(2, 3) = params.dt_smooth_traj;

    mat<4, 1> B = mat<4, 1>::Zero();
    B(3, 0) = params.dt_smooth_traj;

    mat<4, 4> Q_s = mat<4, 4>::Identity();
    Q_s(0, 0) *= 10.0;
    Q_s(1, 1) *= 10.0;
    Q_s(2, 2) *= 10.0;
    Q_s(3, 3) *= 10.0;

    mat<4, 4> Q_l = mat<4, 4>::Identity();
    Q_l(0, 0) *= 1000.0;
    Q_l(1, 1) *= 10.0;
    Q_l(2, 2) *= 0.0;
    Q_l(3, 3) *= 0.0;

    mat<1, 1> R_s = mat<1, 1>::Identity();
    R_s(0, 0) *= 1.0;

    mat<1, 1> R_l = mat<1, 1>::Identity();
    R_l(0, 0) *= 0.1;

    std::vector<vec<4>> xs_s;
    std::vector<vec<1>> us_s;
    lqrSmoother<4, 1>(x0_s, x_ref_s, A, B, Q_s, R_s, xs_s, us_s);

    trajSmooth.states.resize(trajDpResampled.states.size());
    for (size_t i = 0; i < trajDpResampled.states.size(); ++i) {
        LatLonState& tp = trajSmooth.states[i];
        tp.t = i * params.dt_smooth_traj;
        tp.s = xs_s[i][0];
        tp.ds = xs_s[i][1];
        tp.dds = xs_s[i][2];
        tp.ddds = xs_s[i][3];
    }

    // get lateral distances for smoothing corrections
    {
        cudaMemcpy(smoothNodes,
                   trajSmooth.states.data(),
                   sizeof(LatLonState) * trajSmooth.states.size(),
                   cudaMemcpyHostToDevice);

        dp_lat_lon_planner_cu::getLatDistances<<<1, trajSmooth.states.size()>>>(
                env.envGpu, smoothNodes, distNodes);

        latDists.resize(trajSmooth.states.size());

        cudaMemcpy(latDists.data(),
                   distNodes,
                   sizeof(distNodes) * trajSmooth.states.size(),
                   cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();
        checkCudaError(cudaGetLastError(), "getting lateral distances failed");
    }

    std::vector<vec<4>> xs_l;
    std::vector<vec<1>> us_l;
    lqrSmoother<4, 1>(x0_l, x_ref_l, A, B, Q_l, R_l, xs_l, us_l);

    trajSmooth.states.resize(trajDpResampled.states.size());
    for (size_t i = 0; i < trajDpResampled.states.size(); ++i) {
        LatLonState& tp = trajSmooth.states[i];
        tp.t = i * params.dt_smooth_traj;
        tp.s = xs_s[i][0];
        tp.ds = xs_s[i][1];
        tp.dds = xs_s[i][2];
        tp.ddds = xs_s[i][3];
        tp.l = xs_l[i][0];
        tp.dl = xs_l[i][1];
        tp.ddl = xs_l[i][2];
        tp.dddl = xs_l[i][3];
        tp.cost = 0.0;
        tp.constr = 0.0;
    }
}

void DynProgLatLonPlanner::updateTrajCart(const RefLine& refLine) {

    frenetToCartesian(trajDp, refLine, trajDpCart);
    frenetToCartesian(trajSmooth, refLine, trajSmoothCart);
}

void DynProgLatLonPlanner::frenetToCartesian(const LatLonTraj& traj,
                                             const RefLine& refLine,
                                             LatLonCartTraj& result) {
    
    const int len_traj = traj.states.size();

    result.states.resize(len_traj);

    for (int i = 0; i < len_traj; ++i) {
        const LatLonState& tp = traj.states[i];
        RefLinePoint rp = refLine.lerp(tp.s);

        LatLonCartState& cp = result.states[i];
        cp.t = tp.t;
        cp.x = refLine.x_offset + rp.x - (double)tp.l * sin(rp.heading);
        cp.y = refLine.y_offset + rp.y + (double)tp.l * cos(rp.heading);
        if (tp.ds < 1e-3) {
            cp.heading = rp.heading;
        } else {
            cp.heading = atan((double)tp.dl / (double)tp.ds) + rp.heading;
        }
        cp.v = sqrt(sq((1.0 - rp.k * tp.l)*tp.ds) + sq(tp.dl));

        cp.cost = tp.cost;
        cp.constr = tp.constr;
        cp.flags_constr = tp.flags_constr;
    }

    // recover curvature and acceleration with finite difference approximation

    for (int i = 1; i < len_traj; ++i) {
        LatLonCartState& cp0 = result.states[i-1];
        LatLonCartState& cp1 = result.states[i];

        const float dx = cp1.x - cp0.x;
        const float dy = cp1.y - cp0.y;
        const float ds = sqrt(dx*dx + dy*dy);

        cp1.distance = cp0.distance + ds;
        cp0.a = (cp1.v - cp0.v) / (cp1.t - cp0.t);
        if (ds >= 1e-3) { 
            cp0.k = shortAngleDist(cp0.heading, cp1.heading) / ds;
        } else {
            cp1.k = 0.0;
        }
        cp1.k = cp0.k;
        cp1.a = cp0.a;
    }
}

LatLonTraj DynProgLatLonPlanner::reevalTraj(const LatLonTraj& traj,
                                            DynProgEnvironment& env) {

    const int trajLen = traj.states.size();
    const int trajSize = sizeof(LatLonState) * trajLen;

    LatLonTraj evalTraj = traj;

    // upload traj to gpu
    {
        cudaError_t err = cudaMemcpy(reevalNodes,
                                     evalTraj.states.data(),
                                     trajSize,
                                     cudaMemcpyHostToDevice);

        checkCudaError(err, "copying reeval trajectory to gpu failed");
    }

    for (int i = 0; i < trajLen; ++i) {
        dim3 gridSize(1, 1, 1);
        dim3 blockSize(1, 1, 1);
        dp_lat_lon_planner_cu::reevalTraj<<<gridSize, blockSize>>>(
                env.envGpu, reevalNodes, i, trajLen);
    }

    cudaDeviceSynchronize();

    checkCudaError(cudaGetLastError(), "reeval trajectory failed");

    {
        cudaError_t err = cudaMemcpy(evalTraj.states.data(),
                                     reevalNodes,
                                     trajSize,
                                     cudaMemcpyDeviceToHost);

        checkCudaError(err, "copying reeval trajectory to cpu failed");
    }

    return evalTraj;
}

void DynProgLatLonPlanner::copyBackwardPassTex(float* dst, int idx_t) {

    cudaMemcpy3DParms cpyParam;
    memset(&cpyParam, 0, sizeof(cudaMemcpy3DParms));
    cpyParam.dstPtr = make_cudaPitchedPtr((void*)dst,
                                          params.s_steps*sizeof(float4),
                                          params.s_steps,
                                          params.ds_steps);
    cpyParam.srcArray = backwardPassNodes[idx_t].arr;
    cpyParam.extent = make_cudaExtent(params.s_steps,
                                      params.ds_steps,
                                      params.l_steps);
    cpyParam.kind = cudaMemcpyDeviceToHost;
    cudaError_t err = cudaMemcpy3D(&cpyParam);
    checkCudaError(err, "copying backward pass texture gpu to cpu failed");
}

void DynProgLatLonPlanner::queryBackwardPassTex(float* dst, int idx_t, float s, float ds, float l) {

    float4* cudaRes = nullptr;
    cudaMalloc(&cudaRes, sizeof(float4));

    dp_lat_lon_planner_cu::queryBackwardPassTex<<<1,1>>>(
            cudaRes,
            s,
            ds,
            l,
            backwardPassNodes[idx_t]);

    cudaError_t err = cudaMemcpy(dst, cudaRes, sizeof(float4), cudaMemcpyDeviceToHost);
    checkCudaError(err, "querying backward pass texture failed");

    cudaFree(cudaRes);
}

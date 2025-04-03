#include "tplcpp/dyn_prog/lon_planner.cuh"

#include "tplcpp/poly_interp.cuh"

#include <cstring>
#include <stdexcept>

// A few critical parameters are defined at compile time.
// Moving these to memory seems to increase the runtime.
// (Maybe because nvcc cannot do loop unrolling then?)

namespace dp_lon_planner_cu {

__device__ __constant__ DynProgLonPlannerParams params;

__inline__ __device__ void writeNode(cudaSurfaceObject_t& dst,
                                     float4& src,
                                     size_t idx_s,
                                     size_t idx_v,
                                     size_t idx_a) {

    // note: the x component needs to be multiplied with the element size
    surf3Dwrite<float4>(src, dst, idx_s * sizeof(float4), idx_v, idx_a);
}

__inline__ __device__ float4 readNode(cudaTextureObject_t& src,
                                      float s,
                                      float v,
                                      float a) {

    // performs a fast linear interpolation lookup using hardware acceleration
    // note: tex3D uses 9 bit fixed point precision (sufficient for our case)
    // note: textures are voxel centered, therefore an offset of 0.5 is required
    return tex3D<float4>(src,
                         (s - params.s_min) / (params.s_max - params.s_min) * (params.s_steps-1) + 0.5,
                         (v - params.v_min) / (params.v_max - params.v_min) * (params.v_steps-1) + 0.5,
                         (a - params.a_min) / (params.a_max - params.a_min) * (params.a_steps-1) + 0.5);
}

__inline__ __device__ float stateCost(float s,
                                      float v,
                                      float a,
                                      float v_trg,
                                      float s_dist) {

    float cost = 0.0;

    cost += params.w_a * sq(a);
    cost += params.w_progress * abs(1000.0 - s);

    cost += params.w_safety_dist * max(0.0, 
            v * params.time_gap + params.gap_min - s_dist);

    return cost;
}

__inline__ __device__ float stateActionCost(float v, float v_max, float j, float dt) {

    float cost = 0.0;

    cost += params.w_j * sq(j * dt);

    return cost;
}

struct ActionResult {

    float cost = 0.0;
    float constr = 0.0;
};

template <int N>
__inline__ __device__ void evalNode(LonState& tp,
                                    cudaTextureObject_t nextNodeTex,
                                    PathState* path,
                                    EnvironmentGpu env,
                                    float dt,
                                    bool chooseAction=true) {

    PathState cps = interpPath(path, tp.s, params.path_step_size, params.path_steps);

    const float v_max = cps.v_max;

    const float s_dist = env.interpDistMapPath(tp.t, cps.s) - params.length_veh * 0.6;

    const float state_cost = stateCost(tp.s, tp.v, tp.a, v_max, s_dist);
    float state_constr = 0.0;

    if (tp.v > v_max) {
        state_constr += tp.v - v_max;
    }

    if (roundf(tp.t / params.dt) == params.t_steps-1) {
        // last time step, no connection evaluation possible
        tp.cost = state_cost;
        return;
    }

    auto evalAction = [&](float j) -> ActionResult {

        LonState tn = lonDynamics(tp, j, dt);

        float cost = state_cost;
        float constr = state_constr;

        float4 next_node = readNode(nextNodeTex, tn.s, tn.v, tn.a);
        cost += next_node.x;
        constr += next_node.y;

        cost += params.w_snap * sq(next_node.z - j);

        cost += stateActionCost(tp.v, v_max, j, dt);

        double v_max = interpPath(path, tn.s, params.path_step_size, params.path_steps).v_max;

        if (tn.v > v_max) {
            constr += tn.v - v_max;
        }

        const float s_change = tn.s - tp.s;
        if (s_change > s_dist) {
            constr += s_change - s_dist;
        }

        if (tn.a < params.a_min) {
            constr += params.a_min - tn.a;
        }
        if (tn.a > params.a_max) {
            constr += tn.a - params.a_max;
        }

        ActionResult res;
        res.cost = cost;
        res.constr = constr;

        return res;
    };

    if (!chooseAction) { 
        ActionResult res = evalAction(tp.j);
        tp.cost = res.cost;
        tp.constr = res.constr;
        return;
    }

    ActionResult results[N];
    for (int j_idx = 0; j_idx < N; ++j_idx) {
        const float j = params.j_min + (params.j_max - params.j_min)
                        * ((float)j_idx / (float)(N-1));
        results[j_idx] = evalAction(j);
    }

    float constr_min = INFINITY;
    for (ActionResult& res : results) {
        if (res.constr < constr_min) { 
            constr_min = res.constr;
        }
    }

    float cost_min = INFINITY;
    int idx_min = 0.0;
    for (int j_idx = 0; j_idx < N; ++j_idx) {
        if (results[j_idx].constr > constr_min) {
            continue;
        }
        if(results[j_idx].cost < cost_min) {
            cost_min = results[j_idx].cost;
            idx_min = j_idx;
        }
    }

    const float j_min = params.j_min + (params.j_max - params.j_min)
                        * ((float)idx_min / (float)(N-1));

    tp.j = j_min;
    tp.cost = cost_min;
    tp.constr = constr_min;
}

__global__ void lonBackwardStep(cudaSurfaceObject_t nodeSurf,
                                cudaTextureObject_t nodeTex,
                                PathState* path,
                                EnvironmentGpu env,
                                size_t idx_t,
                                float dt) {

    size_t idx_s = threadIdx.x;
    size_t idx_v = blockIdx.x;
    size_t idx_a = blockIdx.y;

    if (idx_s >= params.s_steps || idx_v >= params.v_steps || idx_a >= params.a_steps) {
        return;
    }

    LonState tp;
    tp.t = params.dt_start + (idx_t-1) * params.dt;
    tp.s = params.s_min + idx_s * params._s_step_size;
    tp.v = params.v_min + idx_v * params._v_step_size;
    tp.a = params.a_min + idx_a * params._a_step_size;

    evalNode<9>(tp, nodeTex, path, env, params.dt);

    float4 node;
    node.x = tp.cost;
    node.y = tp.constr;
    node.z = tp.j;
    node.w = 0.0f;
    writeNode(nodeSurf, node, idx_s, idx_v, idx_a);
}

__global__ void lonForwardStep(size_t idx_t,
                               LonState* traj,
                               cudaTextureObject_t nodeTex,
                               PathState* path,
                               EnvironmentGpu env,
                               float dt) {

    LonState& tp = traj[idx_t];

    evalNode<21>(tp, nodeTex, path, env, dt);

    if (idx_t < params.t_steps-1) {
        traj[idx_t+1] = lonDynamics(tp, tp.j, dt);
    }
}

__global__ void lonReevalNode(size_t idx_t,
                              LonState* traj,
                              cudaTextureObject_t nodeTex,
                              PathState* path,
                              EnvironmentGpu env,
                              float dt) {

    LonState& tp = traj[idx_t];
    evalNode<21>(tp, nodeTex, path, env, dt, false);
}

};

__interop__ LonState lonDynamics(LonState x0, float j, float dt) {

    LonState res;
    res.t = x0.t + dt;
    res.s = fmaxf(x0.s, x0.s + x0.v*dt + 0.5*x0.a*dt*dt + 1.0/6.0*j*dt*dt*dt);
    res.v = fmaxf(0.0, x0.v + x0.a*dt + 0.5*j*dt*dt);
    res.a = x0.a + j * dt;
    res.j = j;

    return res;
}

LonState LonTraj::state(float t) {

    InterpVars i(states, &LonState::t, t);

    float t_rel = t - states[i.i_prev].t;
    float j = states[i.i_prev].j;

    return lonDynamics(states[i.i_prev], j, t_rel);
}

DynProgLonPlanner::~DynProgLonPlanner() {
    clearBuffers();
}

void DynProgLonPlanner::clearBuffers() {

    if (forwardPassNodes != nullptr) {
        cudaFree(forwardPassNodes);
    }
    forwardPassNodes = nullptr;

    for (ArrTexSurf<float4>& nodes : backwardPassNodes) {
        nodes.release();
    }
    backwardPassNodes.clear();

    if (path != nullptr) {
        cudaFree(path);
        path = nullptr;
    }
}

void DynProgLonPlanner::reinitBuffers(DynProgLonPlannerParams& ps) {

    ps.updateStepSizes();

    if (ps == params) { 
        return;
    }

    params = ps;

    // copy params to gpu
    {
        cudaError_t err = cudaMemcpyToSymbol(dp_lon_planner_cu::params,
                                             &params,
                                             sizeof(DynProgLonPlannerParams));
        checkCudaError(err, "copying env params failed");
    }

    // need to reallocate

    clearBuffers();

    // trajectory points for forward pass

    size_t fwdArrSize = sizeof(LonState) * params.t_steps;
    cudaError_t err = cudaMalloc(&forwardPassNodes, fwdArrSize);
    checkCudaError(err, "trajectory allocation failed");

    // arrays, texture, surfaces for backward pass

    backwardPassNodes.resize(params.t_steps);

    for (ArrTexSurf<float4>& nodes : backwardPassNodes) {
        nodes.reinit(params.s_steps, params.v_steps, params.a_steps, cudaFilterModeLinear);
    }

    // path array
    {
        cudaError_t err = cudaMalloc(&path,
                                     sizeof(PathState) * params.path_steps);
        checkCudaError(err, "path array allocation failed");
    }
}

LonTraj DynProgLonPlanner::update(LonState initialState,
                                  PathState* cpuPath,
                                  DynProgEnvironment& env) {

    // upload path to gpu
    {
        cudaError_t err = cudaMemcpy(path,
                                     cpuPath,
                                     sizeof(PathState) * params.path_steps,
                                     cudaMemcpyHostToDevice);
        checkCudaError(err, "copying path failed");
    }

    // compute path distance map

    env.updateDistMapPath(path, params.path_steps, params.path_step_size);

    // backward pass

    for (int i = params.t_steps-1; i > 0; --i) {
        dim3 gridSize(params.v_steps, params.a_steps, 1);
        dim3 blockSize(params.s_steps, 1, 1);
        dp_lon_planner_cu::lonBackwardStep<<<gridSize, blockSize>>>(
                backwardPassNodes[i].surf,
                backwardPassNodes[min((int)params.t_steps-1, i+1)].tex,
                path,
                env.envGpu,
                i,
                params.dt);
        
        cudaDeviceSynchronize();

        checkCudaError(cudaGetLastError(), "lon planner backward pass failed " + std::to_string(i));
    }

    // forward pass

    cudaMemcpy(forwardPassNodes,
               &initialState,
               sizeof(LonState),
               cudaMemcpyHostToDevice);

    for (int i = 0; i < params.t_steps; ++i) {
        dim3 gridSize(1, 1, 1);
        dim3 blockSize(1, 1, 1);

        float dt = params.dt;
        if (i == 0) {
            dt = params.dt_start;
        }

        dp_lon_planner_cu::lonForwardStep<<<gridSize, blockSize>>>(
                i,
                forwardPassNodes,
                backwardPassNodes[min((int)params.t_steps-1, i+1)].tex,
                path,
                env.envGpu,
                dt);
    }

    cudaDeviceSynchronize();

    checkCudaError(cudaGetLastError(), "lon planner forward pass failed");

    LonTraj traj;
    traj.states.resize(params.t_steps);

    cudaMemcpy(traj.states.data(),
               forwardPassNodes,
               sizeof(LonState) * params.t_steps,
               cudaMemcpyDeviceToHost);

    return traj;
}

LonTraj DynProgLonPlanner::reevalTraj(const LonTraj& traj,
                                      PathState* cpuPath,
                                      DynProgEnvironment& env) {

    // upload path to gpu
    {
        cudaError_t err = cudaMemcpy(path,
                                     cpuPath,
                                     sizeof(PathState) * params.path_steps,
                                     cudaMemcpyHostToDevice);
        checkCudaError(err, "copying path failed");
    }

    // compute path distance map

    env.updateDistMapPath(path, params.path_steps, params.path_step_size);

    const int trajLen = traj.states.size();
    const int trajSize = sizeof(LonState) * trajLen;

    if (trajLen != params.t_steps) {
        throw std::runtime_error("trajectory size does not match t_steps");
    }

    LonTraj evalTraj = traj;

    // upload traj to gpu
    {
        cudaError_t err = cudaMemcpy(forwardPassNodes,
                                     evalTraj.states.data(),
                                     trajSize,
                                     cudaMemcpyHostToDevice);

        checkCudaError(err, "copying trajectory to gpu failed");
    }

    for (int i = 0; i < trajLen; ++i) {
        dim3 gridSize(1, 1, 1);
        dim3 blockSize(1, 1, 1);

        float dt = params.dt;
        if (i == 0) {
            dt = params.dt_start;
        }

        dp_lon_planner_cu::lonReevalNode<<<gridSize, blockSize>>>(
                i,
                forwardPassNodes,
                backwardPassNodes[min((int)params.t_steps-1, i+1)].tex,
                path,
                env.envGpu,
                dt);
    }

    cudaDeviceSynchronize();

    checkCudaError(cudaGetLastError(), "reeval pass failed");

    {
        cudaError_t err = cudaMemcpy(evalTraj.states.data(),
                                     forwardPassNodes,
                                     trajSize,
                                     cudaMemcpyDeviceToHost);

        checkCudaError(err, "copying reevaluted trajectory to cpu failed");
    }

    return evalTraj;
}

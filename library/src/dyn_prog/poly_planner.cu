#include "tplcpp/dyn_prog/poly_planner.cuh"
#include "tplcpp/poly_interp.cuh"

#include <cstring>
#include <stdexcept>

namespace dp_poly_planner {

__device__ __constant__ DynProgPolyPlannerParams params;

__global__ void evalEdge(size_t eval_step,
                         size_t idx_max,
                         const DynProgPolyNode* __restrict__ startNodes,
                         const DynProgPolyEdge* __restrict__ edges,
                         DynProgPolyNode* endNodes,
                         EnvironmentGpu env) {

    const size_t idx_edge = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx_edge >= idx_max) {
        return;
    }

    const DynProgPolyEdge edge = edges[idx_edge];
    const DynProgPolyNode start = startNodes[edge.startNodeIdx];
    DynProgPolyNode end = endNodes[edge.endNodeIdx];
    DynProgPolyPoint& pe = end.point;

    PolyQuartic poly_lon(start.point.t,
                         start.point.s,
                         start.point.ds,
                         start.point.dds,
                         pe.t,
                         pe.ds,
                         0.0);

    PolyQuintic poly_lat(start.point.t,
                         start.point.l,
                         start.point.dl,
                         start.point.ddl,
                         pe.t,
                         pe.l,
                         0.0,
                         0.0);

    float cost = 0.0;

    float jerk_lon = 0.0;
    for (float t = start.point.t; t <= pe.t; t += 0.25) {
        jerk_lon += sq(poly_lon.dddf(t));
    }
    float jerk_lat = 0.0;
    for (float t = start.point.t; t <= pe.t; t += 0.25) {
        jerk_lat += sq(poly_lat.dddf(t));
    }
    cost += params.w_j * jerk_lon;
    cost += params.w_j * jerk_lat;

    cost += params.w_l * fabsf(0.0 - pe.l);

    float t_end; 
    if (eval_step == params.eval_steps - 1) {
        // in the last eval step evaluated up to the horizon
        t_end = (params.t_steps - 1) * params.dt;
    } else {
        t_end = pe.t;
    }

    float ds;
    float s;
    float l;

    float dt_step = 0.25;
    for (float t = start.point.t; t <= t_end; t += dt_step) {
        if (t <= pe.t) {
            ds = poly_lon.df(t);
            s = poly_lon.f(t);
            l = poly_lat.f(t);
        } else {
            ds = poly_lon.df(pe.t);
            s = poly_lon.f(pe.t) + (t - pe.t) * ds;
            l = poly_lat.f(pe.t);
        }

        RefLinePoint rp = env.ref_line.lerp(s);

        cost += params.w_v_diff * fabsf(100.0 - ds);
        cost += 100.0 * fmaxf(0.0f, ds - rp.v_max);

        //float dir = atanf(dl / ds);
        //float len = sqrtf(sq(dl) + sq(ds)) * dt_step;

        const float d_max_front = env.interpDirDistMap(t, s, l, 0.0);
        const float d_safety = d_max_front
                 - params.length_veh * 0.5
                 - 1.0
                 - ds * 1.0;

        if (ds*dt_step > d_safety) {
            cost += 100.0f * (ds*dt_step - d_safety);
        }
    }

    pe.s = poly_lon.f(pe.t);
    pe.cost = cost;

    endNodes[edge.endNodeIdx].point = pe;
}

__global__ void propagateCost(size_t idx_max,
                              DynProgPolyNode* startNodes,
                              const DynProgPolyEdge* __restrict__ edges,
                              const DynProgPolyNode* __restrict__ endNodes) {

    const size_t idx_start = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx_start >= idx_max) {
        return;
    }

    float cost_min = INFINITY;
    uint32_t idx_min = 0;

    DynProgPolyNode start = startNodes[idx_start];
    for (uint32_t i = start.edgeStartIdx; i < start.edgeEndIdx; ++i) {
        const DynProgPolyEdge edge = edges[i];
        const DynProgPolyNode end = endNodes[edge.endNodeIdx];
        if (end.point.cost < cost_min) {
            cost_min = end.point.cost;
            idx_min = i;
        }
    }

    DynProgPolyPoint& ps = startNodes[idx_start].point;
    ps.cost += cost_min;
    ps.idx_next = idx_min;
}

__global__ void copyTrajectory(int idx,
                               const DynProgPolyNode* __restrict__ nodes,
                               DynProgPolyPoint* trajectoryPoints) {

    if (idx == 0) {
        trajectoryPoints[0] = nodes[0].point;
    } else {
        uint32_t idx_next = trajectoryPoints[idx-1].idx_next;
        const DynProgPolyNode& n = nodes[idx_next];
        trajectoryPoints[idx] = n.point;
    }
}

};

DynProgPolyPoint DynProgPolyTraj::at(float t) {

    InterpVars i(points, &DynProgPolyPoint::t, t);

    DynProgPolyPoint& start = points[i.i_prev];
    DynProgPolyPoint& end = points[i.i_next];

    PolyQuintic poly_lon(start.t, start.s, start.ds, start.dds,
                         end.t, end.s, end.ds, end.dds);
    PolyQuintic poly_lat(start.t, start.l, start.dl, start.ddl,
                         end.t, end.l, end.dl, end.ddl);

    DynProgPolyPoint res;
    float t_end = points[points.size() - 1].t;
    if (t > t_end) {
        res.t = t;
        res.ds = poly_lon.df(t_end);
        res.s = poly_lon.f(t_end) + res.ds * (t - t_end);
        res.dds = 0.0;
        res.l = poly_lat.f(t_end);
        res.dl = 0.0;
        res.ddl = 0.0;
    } else {
        res.t = t;
        res.s = poly_lon.f(t);
        res.ds = poly_lon.df(t);
        res.dds = poly_lon.ddf(t);
        res.l = poly_lat.f(t);
        res.dl = poly_lat.df(t);
        res.ddl = poly_lat.ddf(t);
    }

    return res;
}

DynProgPolyCartPoint DynProgPolyCartTraj::at(double t) {

    InterpVars i(points, &DynProgPolyCartPoint::t, t);

    double ai = 1.0f - i.a;

    DynProgPolyCartPoint res;
    res.t = points[i.i_prev].t * ai + points[i.i_next].t * i.a;
    res.distance = points[i.i_prev].distance * ai + points[i.i_next].distance * i.a;
    res.x = points[i.i_prev].x * ai + points[i.i_next].x * i.a;
    res.y = points[i.i_prev].y * ai + points[i.i_next].y * i.a;
    res.v = points[i.i_prev].v * ai + points[i.i_next].v * i.a;
    res.a = points[i.i_prev].a * ai + points[i.i_next].a * i.a;
    res.heading = points[i.i_prev].heading + shortAngleDist(points[i.i_prev].heading, points[i.i_next].heading) * i.a;
    res.k = points[i.i_prev].k * ai + points[i.i_next].k * i.a;

    return res;
}

DynProgPolyPlanner::DynProgPolyPlanner() {
    reinitBuffers(params, true);
}

DynProgPolyPlanner::~DynProgPolyPlanner() {
    clearBuffers();
}

void DynProgPolyPlanner::clearBuffers() {

    if (trajectoryPoints != nullptr) {
        cudaFree(trajectoryPoints);
    }
    trajectoryPoints = nullptr;
    
    // clear evaluation graphs

    for (GpuEvalGraph& g : evalGraphs) {
        for (CudaEvalList<DynProgPolyNode> l : g.evalNodeList) {
            cudaFree(l.cudaPtr);
        }
        for (CudaEvalList<DynProgPolyEdge> l : g.evalEdgeList) {
            cudaFree(l.cudaPtr);
        }
    }
    evalGraphs.clear();
}

GpuEvalGraph DynProgPolyPlanner::buildEvalGraph(int idx_l_start, int idx_ds_start) {

    // for each evaluation step ...
    // ... a set of nodes and ...
    std::vector<std::vector<DynProgPolyNode>> evalNodes(params.eval_steps + 1);
    // ... a list of edges to be evaluated
    std::vector<std::vector<DynProgPolyEdge>> evalEdges(params.eval_steps);

    // create initial node

    DynProgPolyNode node_start;
    node_start.point.t = 0.0;
    node_start.point.s = params.s_min;
    node_start.point.ds = params.ds_min + idx_ds_start * params._ds_step_size;
    node_start.point.l = params.l_min + idx_l_start * params._l_step_size;

    evalNodes[0].push_back(node_start);

    // build the graph, already prune infeasible connections

    for (size_t idx_eval = 0; idx_eval < evalNodes.size() - 1; ++idx_eval) {
        for (size_t idx_start_node = 0; idx_start_node < evalNodes[idx_eval].size(); ++idx_start_node) {
        
            DynProgPolyNode& node_start = evalNodes[idx_eval][idx_start_node];
            node_start.edgeStartIdx = evalEdges[idx_eval].size();

            int idx_t_start = round(node_start.point.t / params.dt);

            for (int idx_t_end = idx_t_start + 1; idx_t_end < params.t_steps; ++idx_t_end) {

                float t_end = idx_t_end * params.dt;
                float t_change = t_end - node_start.point.t;

                float ds_change_min = params.dds_min * t_change;
                float ds_change_max = params.dds_max * t_change;
                float l_change_min = params.dl_min * t_change;
                float l_change_max = params.dl_max * t_change;

                for (int idx_ds_end = 0; idx_ds_end < params.ds_steps; ++idx_ds_end) {

                    float ds_end = params.ds_min + idx_ds_end * params._ds_step_size;

                    float ds_change = ds_end - node_start.point.ds;
                    if (ds_change < ds_change_min || ds_change > ds_change_max) {
                        continue;
                    }
                    
                    for (int idx_l_end = 0; idx_l_end < params.l_steps; ++idx_l_end) {

                        float l_end = params.l_min + idx_l_end * params._l_step_size;

                        float l_change = l_end - node_start.point.l;
                        if (l_change < l_change_min || l_change > l_change_max) {
                            continue;
                        }

                        DynProgPolyNode& node_end = evalNodes[idx_eval+1].emplace_back();
                        node_end.point.t = t_end;
                        node_end.point.ds = ds_end;
                        node_end.point.l = l_end;

                        DynProgPolyEdge& e = evalEdges[idx_eval].emplace_back();
                        e.startNodeIdx = idx_start_node;
                        e.endNodeIdx = evalNodes[idx_eval+1].size() - 1;
                    }
                }
            }

            node_start.edgeEndIdx = evalEdges[idx_eval].size();
        }
    }

    // copy to gpu

    GpuEvalGraph graph;

    for (std::vector<DynProgPolyNode>& nodes : evalNodes) {
        CudaEvalList<DynProgPolyNode>& l = graph.evalNodeList.emplace_back();
        l.size = nodes.size();
        const size_t len = nodes.size() * sizeof(DynProgPolyNode);
        cudaMalloc(&l.cudaPtr, len);
        cudaMemcpy(l.cudaPtr, nodes.data(), len, cudaMemcpyHostToDevice);
    }

    for (std::vector<DynProgPolyEdge>& edges : evalEdges) {
        CudaEvalList<DynProgPolyEdge>& l = graph.evalEdgeList.emplace_back();
        l.size = edges.size();
        const size_t len = edges.size() * sizeof(DynProgPolyEdge);
        cudaMalloc(&l.cudaPtr, len);
        cudaMemcpy(l.cudaPtr, edges.data(), len, cudaMemcpyHostToDevice);
    }

    return graph;
}

void DynProgPolyPlanner::reinitBuffers(DynProgPolyPlannerParams& ps, bool force = false) {

    ps.updateStepSizes();

    bool fullReinitRequired = force
        || ps.eval_steps != params.eval_steps
        || ps.t_steps != params.t_steps
        || ps.s_steps != params.s_steps
        || ps.ds_steps != params.ds_steps
        || ps.l_steps != params.l_steps;

    params = ps;

    // copy params to gpu
    {
        cudaError_t err = cudaMemcpyToSymbol(dp_poly_planner::params,
                                             &params,
                                             sizeof(DynProgPolyPlannerParams));
        checkCudaError(err, "copying poly planner params failed");
    }

    if (!fullReinitRequired) {
        return;
    }

    // need to reallocate

    clearBuffers();

    // rebuild evaluation graphs

    evalGraphs.resize(params.ds_steps * params.l_steps);

    #pragma omp parallel for collapse(2)
    for (int idx_ds_start = 0; idx_ds_start < params.ds_steps; ++idx_ds_start) {
        for (int idx_l_start = 0; idx_l_start < params.l_steps; ++idx_l_start) {
            int idx_graph = idx_ds_start * params.l_steps + idx_l_start;
            evalGraphs[idx_graph] = buildEvalGraph(idx_l_start, idx_ds_start);
        }
    }
    checkCudaError(cudaGetLastError(), "creating eval graphs failed");

    // trajectory points for forward pass

    size_t trajArrSize = sizeof(DynProgPolyPoint) * (params.eval_steps + 1);
    cudaError_t err = cudaMalloc(&trajectoryPoints, trajArrSize);
    checkCudaError(err, "trajectory allocation failed");
}

DynProgPolyTraj DynProgPolyPlanner::update(DynProgPolyPoint initialState,
                                           DynProgEnvironment& env) {

    int idx_ds = roundf((initialState.ds - params.ds_min) / params._ds_step_size);
    int idx_l = roundf((initialState.l - params.l_min) / params._l_step_size);
    int idx_graph = idx_ds * params.l_steps + idx_l;

    const int blockSize = 100;
    
    GpuEvalGraph graph = evalGraphs[idx_graph];

    cudaMemcpy(graph.evalNodeList[0].cudaPtr
                    + offsetof(DynProgPolyNode, point),
               &initialState,
               sizeof(DynProgPolyPoint),
               cudaMemcpyHostToDevice);

    for (size_t i = 0; i < graph.evalEdgeList.size(); ++i) {
        const int gridSize = graph.evalEdgeList[i].size / blockSize + 1;
        dim3 gridDims(gridSize, 1, 1);
        dim3 blockDims(blockSize, 1, 1);
        dp_poly_planner::evalEdge<<<gridDims, blockDims>>>(
                i,
                graph.evalEdgeList[i].size,
                graph.evalNodeList[i].cudaPtr,
                graph.evalEdgeList[i].cudaPtr,
                graph.evalNodeList[i+1].cudaPtr,
                env.envGpu);
    }
    cudaDeviceSynchronize();
    checkCudaError(cudaGetLastError(), "edge evaluation failed");

    for (ssize_t i = graph.evalEdgeList.size() - 1; i >= 0; --i) {
        const int gridSize = graph.evalNodeList[i].size / blockSize + 1;
        dim3 gridDims(gridSize, 1, 1);
        dim3 blockDims(blockSize, 1, 1);
        dp_poly_planner::propagateCost<<<gridDims, blockDims>>>(
                graph.evalNodeList[i].size,
                graph.evalNodeList[i].cudaPtr,
                graph.evalEdgeList[i].cudaPtr,
                graph.evalNodeList[i+1].cudaPtr);
    }
    cudaDeviceSynchronize();
    checkCudaError(cudaGetLastError(), "cost propagation failed");

    for (size_t i = 0; i < graph.evalNodeList.size(); ++i) {
        dp_poly_planner::copyTrajectory<<<1, 1>>>(
                i,
                graph.evalNodeList[i].cudaPtr,
                trajectoryPoints);
    }
    DynProgPolyTraj traj;
    traj.points.resize(params.eval_steps + 1);

    cudaMemcpy(traj.points.data(),
               trajectoryPoints,
               sizeof(DynProgPolyPoint) * (params.eval_steps + 1),
               cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    checkCudaError(cudaGetLastError(), "copying trajectory failed");

    return traj;
}

DynProgPolyCartTraj DynProgPolyPlanner::frenetToCartesian(
        const DynProgPolyTraj& traj, const RefLine& refLine) {

    DynProgPolyCartTraj result;

    const int len_traj = traj.points.size();

    result.points.resize(len_traj);

    for (int i = 0; i < len_traj; ++i) {
        const DynProgPolyPoint& tp = traj.points[i];
        RefLinePoint rp = refLine.lerp(tp.s);

        DynProgPolyCartPoint& cp = result.points[i];
        cp.t = tp.t;
        cp.x = refLine.x_offset + rp.x - (double)tp.l * sin(rp.heading);
        cp.y = refLine.y_offset + rp.y + (double)tp.l * cos(rp.heading);
        if (tp.ds < 1e-3) {
            cp.heading = rp.heading;
        } else {
            cp.heading = atan((double)tp.dl / (double)tp.ds) + rp.heading;
        }
        cp.v = sqrt(sq((1.0 - rp.k * tp.l)*tp.ds) + sq(tp.dl));
    }

    // recover curvature and acceleration with finite difference approximation

    for (int i = 1; i < len_traj; ++i) {
        DynProgPolyCartPoint& cp0 = result.points[i-1];
        DynProgPolyCartPoint& cp1 = result.points[i];

        const double dx = cp1.x - cp0.x;
        const double dy = cp1.y - cp0.y;
        const double ds = sqrt(dx*dx + dy*dy);

        cp1.distance = cp0.distance + ds;
        cp0.a = (cp1.v - cp0.v) / (cp1.t - cp0.t);
        if (cp0.v >= 1e-3) {
            cp0.k = shortAngleDist(cp0.heading, cp1.heading) / cp0.v;
        } else {
            cp1.k = 0.0;
        }
        cp1.k = cp0.k;
        cp1.a = cp0.a;
    }

    return result;
}

/*
LatTraj DynProgPolyPlanner::reevalTraj(const LatTraj& traj, DynProgEnvironment& env) {

    // TODO: implement again

    return traj;
}
*/

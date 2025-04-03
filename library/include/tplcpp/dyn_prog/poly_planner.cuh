#pragma once

#include "tplcpp/dyn_prog/common.cuh"
#include "tplcpp/dyn_prog/env.cuh"

#include <unordered_set>

struct DynProgPolyPlannerParams {

    int eval_steps = 2;

    int t_steps = 10;
    int s_steps = 201;
    int ds_steps = 15;
    int l_steps = 21;

    float s_min = 0.0;
    float s_max = 200.0;
    float ds_min = 0.0;
    float ds_max = 36.0;
    float dds_min = -3.0;
    float dds_max = 3.0;
    float l_min = -5.0;
    float l_max = 5.0;
    float dl_min = -2.0;
    float dl_max = 2.0;
    float dt = 1.0;
    float dt_start = 1.0;
    float dt_cart = 0.1;

    float a_total_max = 3.0;
    float a_lat_abs_max = 3.0;

    float w_v_diff = 1.0;
    float w_l = 1.0;
    float w_j = 1.0;

    float _s_step_size = 0.0;
    float _ds_step_size = 0.0;
    float _l_step_size = 0.0;

    float width_veh = 0.0;
    float length_veh = 0.0;

    void updateStepSizes() {

        _s_step_size = (s_max - s_min) / (float)(s_steps - 1);
        _ds_step_size = (ds_max - ds_min) / (float)(ds_steps - 1);
        _l_step_size = (l_max - l_min) / (float)(l_steps - 1);
    }
};

struct DynProgPolyPoint {

    float t = 0.0;
    float s = 0.0;
    float ds = 0.0;
    float dds = 0.0;
    float l = 0.0;
    float dl = 0.0;
    float ddl = 0.0;
    
    float cost = 0.0;
    uint32_t idx_next = 0;

    void writeToArray(float* data, size_t offset) {

        data[offset + 0] = s;
        data[offset + 1] = ds;
        data[offset + 2] = dds;
        data[offset + 3] = l;
        data[offset + 4] = dl;
        data[offset + 5] = ddl;
    }
};

struct DynProgPolyCartPoint {

    double t = 0.0;
    double distance = 0.0;
    double x = 0.0;
    double y = 0.0;
    double v = 0.0;
    double a = 0.0;
    double heading = 0.0;
    double k = 0.0;

    void writeToArray(double* data, size_t offset) {

        data[offset + 0] = t;
        data[offset + 1] = distance;
        data[offset + 2] = x;
        data[offset + 3] = y;
        data[offset + 4] = v;
        data[offset + 5] = a;
        data[offset + 6] = heading;
        data[offset + 7] = k;
    }
};

struct DynProgPolyNode {

    DynProgPolyPoint point = {};

    // indices into the corresponding evalEdges array
    // edges in [edgeStartIdx, ..., edgeEndIdx] start at this node
    uint32_t edgeStartIdx = 0;
    uint32_t edgeEndIdx = 0;
};

struct DynProgPolyEdge {

    // these are indices into the evalNodes vector
    uint32_t startNodeIdx = 0;
    uint32_t endNodeIdx = 0;
};

template<typename T>
struct CudaEvalList {
    size_t size = 0;
    T* cudaPtr = nullptr;
};

struct GpuEvalGraph {
    std::vector<CudaEvalList<DynProgPolyNode>> evalNodeList;
    std::vector<CudaEvalList<DynProgPolyEdge>> evalEdgeList;
};

struct DynProgPolyTraj {

    std::vector<DynProgPolyPoint> points;
    DynProgPolyPoint at(float t);
};

struct DynProgPolyCartTraj {

    std::vector<DynProgPolyCartPoint> points;
    DynProgPolyCartPoint at(double t);
};

struct DynProgPolyPlanner {

    DynProgPolyPlannerParams params;

    // contains an evaluation graph for each (velocity, lateral) index
    std::vector<GpuEvalGraph> evalGraphs;

    // cuda pointer to forward node array
    DynProgPolyPoint* trajectoryPoints = nullptr;

    DynProgPolyPlanner();
    ~DynProgPolyPlanner();

    void clearBuffers();
    void reinitBuffers(DynProgPolyPlannerParams& ps, bool force);

    GpuEvalGraph buildEvalGraph(int idx_v_start, int idx_l_start);

    DynProgPolyTraj update(DynProgPolyPoint initialState,
                           DynProgEnvironment& env);

    DynProgPolyCartTraj frenetToCartesian(
            const DynProgPolyTraj& traj, const RefLine& refLine);
                        
    /*
    LatTraj reevalTraj(const LatTraj& traj,
                       DynProgEnvironment& env); */
};

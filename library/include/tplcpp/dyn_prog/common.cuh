#pragma once

#include <vector>
#include <string>
#include <cstring>
#include <iostream>

#define CONSTR_PENALTY 10.0e7

#include "tplcpp/utils.hpp"

void checkCudaError(cudaError_t err, std::string msg);

#include "tplcpp/dyn_prog/arr_tex_surf.cuh"

struct RefLineGpu {

    int steps = 0;
    float step_size = 0.0;
    RefLinePoint* points = nullptr;

    ArrTexSurf<float4> ats;

    void release() {

        if (points != nullptr) {
            cudaFree(points);
        }
        points = nullptr;
        steps = 0;
        step_size = 0.0;

        ats.release();
    }

    void upload(RefLine& refLine) {

        if (steps != refLine.points.size()) {
            release();
            cudaError_t err = cudaMalloc(&points,
                                         sizeof(RefLinePoint) * refLine.points.size());
            checkCudaError(err, "ref line gpu array allocation failed");

            ats.reinit(refLine.points.size(), 1, 1);

            steps = refLine.points.size();
        }

        // copy to global memory

        cudaError_t err = cudaMemcpy(points,
                                     refLine.points.data(),
                                     sizeof(RefLinePoint) * steps,
                                     cudaMemcpyHostToDevice);
        checkCudaError(err, "copying ref line to gpu failed");

        // copy to texture memory

        std::vector<float4> texturedRefLine(steps);
        for (size_t i = 0; i < refLine.points.size(); ++i) {
            texturedRefLine[i].x = refLine.points[i].v_max;
            texturedRefLine[i].y = refLine.points[i].d_left;
            texturedRefLine[i].z = refLine.points[i].d_right;
            texturedRefLine[i].w = refLine.points[i].semantic;
        }

        cudaMemcpy3DParms cpyParam;
        memset(&cpyParam, 0, sizeof(cudaMemcpy3DParms));
        cpyParam.srcPtr = make_cudaPitchedPtr((void*)texturedRefLine.data(),
                                              steps*sizeof(float4),
                                              steps,
                                              1);
        cpyParam.dstArray = ats.arr;
        cpyParam.extent = make_cudaExtent(steps, 1, 1);
        cpyParam.kind = cudaMemcpyHostToDevice;

        err = cudaMemcpy3D(&cpyParam);
        checkCudaError(err, "copying ref line to texture failed");

        // do not forget the step size! (don't ask me how i know that)

        step_size = refLine.step_size;
    }

    __device__ const RefLinePoint& operator [](const int i) const {
        return points[i];
    }

    __device__ RefLinePoint lerp(const float s) const {
        return interpRefLine(points, steps, step_size, s);
    }

    __device__ float4 texLerp(const float s) const {
        return ats.get_cached(s / step_size, 0.0, 0.0);
    }
};

struct PathState {

    float x = 0.0;
    float y = 0.0;
    // frenet coordinates wrt. ref_line
    float s = 0.0;
    float l = 0.0;

    float k = 0.0;
    float v_max = 0.0;

    float distance = 0.0;
};

__inline__ __device__ PathState interpPath(
        PathState* path,
        float distance,
        float path_step_size,
        int path_steps) {

    PathState res;

    if (path == nullptr) {
        return res;
    }

    float alpha = distance / path_step_size;
    int i0 = max(0, min(path_steps-1, (int)floor(alpha)));
    int i1 = max(0, min(path_steps-1, (int)ceil(alpha)));
    alpha -= i0;
    float alpha_inv = 1.0 - alpha;

    res.x = path[i0].x * alpha_inv + path[i1].x * alpha;
    res.y = path[i0].y * alpha_inv + path[i1].y * alpha;
    res.s = path[i0].s * alpha_inv + path[i1].s * alpha;
    res.l = path[i0].l * alpha_inv + path[i1].l * alpha;
    res.k = path[i0].k * alpha_inv + path[i1].k * alpha;
    res.v_max = path[i0].v_max * alpha_inv + path[i1].v_max * alpha;
    res.distance = path[i0].distance * alpha_inv + path[i1].distance * alpha;

    return res;
}

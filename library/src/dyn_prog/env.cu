#include "tplcpp/dyn_prog/env.cuh"
#include "tplcpp/utils.hpp"

#include <cstring>
#include <sstream>

namespace dp_env_cu {

__constant__ DynProgEnvParams params = {};

__global__ void occupancyToFrenet(EnvironmentGpu env) {

    const size_t idx_t = blockIdx.x;
    const size_t idx_s = blockIdx.y;
    const size_t idx_l = threadIdx.x;

    if (idx_t >= params.t_steps
            || idx_s >= params.s_steps
            || idx_l >= params.l_steps) {
        return;
    }

    const float s = params.s_min + idx_s * params._s_step_size;

    const int dil_steps = int(ceilf(params.dilation * env.occupancy.ppm));

    const float l = params.l_min + idx_l * params._l_step_size;
    const RefLinePoint rp = env.ref_line.lerp(s);

    const float x = rp.x - l * sin(rp.heading);
    const float y = rp.y + l * cos(rp.heading);

    const float x_pixels = x*env.occupancy.ppm + (float)env.occupancy.size_pixels/2.0;
    const float y_pixels = y*env.occupancy.ppm + (float)env.occupancy.size_pixels/2.0;

    unsigned int occ = 0;
    for (int i = -dil_steps; i < dil_steps; ++i) {
        for (int j = -dil_steps; j < dil_steps; ++j) {
            if (sqrtf(i*i + j*j) > dil_steps) {
                continue;
            }
            occ |= tex2D<unsigned int>(env.occupancy.tex, x_pixels + i, y_pixels + j);
        }
    }

    const unsigned int mask = 1;
    float val = (occ & (mask << idx_t)) != 0;

    bool stationary = (occ & (mask << 31)) != 0;
    if (stationary) {
        val = 3.0;
    }

    if (l > rp.d_left || l < -rp.d_right) {
        val = 2.0;
    } else if (idx_l == 0 || idx_l == params.l_steps - 1) {
        val = 2.0;
    } else if (rp.v_max < 0.1) {
        val = 2.0;
    }

    env.occ_map.set(idx_t, idx_s, idx_l, val);
}

__global__ void closeIntersections(EnvironmentGpu env) {

    const size_t idx_t = blockIdx.x;
    const size_t idx_s = blockIdx.y;

    if (idx_t >= params.t_steps || idx_s >= params.s_steps) {
        return;
    }

    const float s = params.s_min + idx_s * params._s_step_size;
    const float semantic = env.ref_line.texLerp(s).w;

    if (semantic < 0.5) {
        return;
    }

    bool close = false;
    for (int idx_l = 1; idx_l < params.l_steps-1; ++idx_l) {
        if (env.occ_map.get_cached(idx_t, idx_s, idx_l) == 1.0f) {
            close = true;
        }
    }
    if (close) {
        float val = 1.0;
        for (int idx_l = 1; idx_l < params.l_steps-1; ++idx_l) {
            env.occ_map.set(idx_t, idx_s, idx_l, val);
        }
    }
}

__global__ void updateDistMapLon(EnvironmentGpu env) {

    const size_t idx_t = blockIdx.x;
    const size_t idx_l = blockIdx.y;

    if (idx_t >= params.t_steps 
            || idx_l >= params.l_steps) {
        return;
    }

    float d = 10000.0;
    for (int idx_s = params.s_steps-1; idx_s >= 0; idx_s -= 1) {
        float occ = env.occ_map.get_cached(idx_t, idx_s, idx_l);
        if (occ > 0) {
            d = 0.0;
        } else { 
            d += params._s_step_size;
        }
        float2 dists(d, 0.0);
        env.dist_map_lon.set(idx_t, idx_s, idx_l, dists);
    }

    d = 10000.0;
    for (int idx_s = 0; idx_s < params.s_steps; idx_s += 1) {
        float occ = env.occ_map.get_cached(idx_t, idx_s, idx_l);
        if (occ > 0) {
            d = 0.0;
        } else {
            d += params._s_step_size;
        }
        float2 dists = env.dist_map_lon.get(idx_t, idx_s, idx_l);
        dists.y = d;
        env.dist_map_lon.set(idx_t, idx_s, idx_l, dists);
    }
}

__global__ void updateDistMapPath(EnvironmentGpu env,
                                  PathState* path,
                                  int lenPath,
                                  float stepSizePath) {

    const size_t idx_t = blockIdx.x;

    if (idx_t >= params.t_steps) {
        return;
    }

    float d = 10000.0;
    for (int idx_s = params.s_steps-1; idx_s >= 0; idx_s -= 1) {
        const float s = params.s_min + idx_s * params._s_step_size;
        const PathState cps = interpPath(path, s, stepSizePath, lenPath);

        const int idx_path_s = (cps.s - params.s_min) / params._s_step_size;
        const int idx_path_l = (cps.l - params.l_min) / params._l_step_size;

        const float occ = env.occ_map.get_cached(idx_t, idx_path_s, idx_path_l);
        if (occ > 0) {
            d = 0.0;
        } else {
            d += params._s_step_size;
        }
        env.dist_map_path.set(idx_t, idx_s, 0, d);
    }
}

__global__ void updateDirDistMap(size_t idx_t,
                                 EnvironmentGpu env) {

    const size_t idx_s = blockIdx.x;
    const size_t idx_l = blockIdx.y;
    const size_t idx_dir = threadIdx.x;

    if (idx_s >= params.s_steps 
            || idx_l >= params.l_steps 
            || idx_dir >= params.dir_steps) {
        return;
    }
    
    const float angle = params.dir_min 
        + idx_dir * (params.dir_max - params.dir_min) 
        / (params.dir_steps - 1.0);

    const float s = params.s_min + idx_s * params._s_step_size;
    const float l = params.l_min + idx_l * params._l_step_size;

    const float step_size = fminf(params._s_step_size, params._l_step_size);
    const float ds = step_size * cosf(angle);
    const float dl = step_size * sinf(angle);
    const int steps = params.ds_max / step_size;

    float dist = 0.0;
    bool collision = false;
    for (int i = 0; i < steps; ++i) {

        float is = ((s + ds * i) - params.s_min) / params._s_step_size;
        float il = ((l + dl * i) - params.l_min) / params._l_step_size;

        if (is <= 0.0 || is >= params.s_steps) {
            collision = true;
            break;
        }
        if (il <= 0.0 || il >= params.l_steps) {
            collision = true;
            break;
        }

        const float occ = env.occ_map.get_cached(idx_t, is, il);
        if (occ > 0.0) {
            collision = true;
            break;
        }
        dist += step_size;
    }

    if (!collision) {
        dist = 10000.0;
    }

    env.dir_dist_map[idx_t].set(idx_s, idx_l, idx_dir, dist);
}

__global__ void copyDist(EnvironmentGpu envGpu, float t, float s, float l, float dt, float* res) {

    *res = envGpu.interpDistField(t, s, l);
}

};

// utility functions

__device__ float EnvironmentGpu::interpVMax(float s) {

    return ref_line.lerp(s).v_max;
}

__device__ float EnvironmentGpu::interpDistField(float t, float s, float l) {

    using namespace dp_env_cu;

    float t_idx = t < params.dt_start ? 0 : (roundf((t - params.dt_start) / params.dt) + 1.0);

    return occ_map.get_cached(
            t_idx,
            (s - params.s_min) / (params.s_max - params.s_min) * (params.s_steps-1),
            (l - params.l_min) / (params.l_max - params.l_min) * (params.l_steps-1));
}

__device__ float2 EnvironmentGpu::interpDistMapLon(float t, float s, float l) {

    using namespace dp_env_cu;

    float t_idx = t < params.dt_start ? 0 : (roundf((t - params.dt_start) / params.dt) + 1.0);

    return dist_map_lon.get_cached(
            t_idx,
            (s - params.s_min) / (params.s_max - params.s_min) * (params.s_steps-1),
            roundf((l - params.l_min) / (params.l_max - params.l_min) * (params.l_steps-1)));
}

__device__ float EnvironmentGpu::interpDistMapPath(float t, float s) {

    using namespace dp_env_cu;

    float t_idx = t < params.dt_start ? 0 : (roundf((t - params.dt_start) / params.dt) + 1.0);

    return dist_map_path.get_cached(
            t_idx,
            (s - params.s_min) / (params.s_max - params.s_min) * (params.s_steps-1),
            0);
}

__device__ float EnvironmentGpu::interpDirDistMap(float t, float s, float l, float dir) {

    using namespace dp_env_cu;

    int t_idx = t < params.dt_start ? 0 : (roundf((t - params.dt_start) / params.dt) + 1.0);

    return dir_dist_map[t_idx].get_cached(
            (s - params.s_min) / (params.s_max - params.s_min) * (params.s_steps-1),
            (l - params.l_min) / (params.l_max - params.l_min) * (params.l_steps-1),
            (dir - params.dir_min) / (params.dir_max - params.dir_min) * (params.dir_steps-1));
}

/*
 * Actual class implementation
 */

DynProgEnvironment::DynProgEnvironment() {

    reinitBuffers(params, true);
}

DynProgEnvironment::~DynProgEnvironment() {

    clearBuffers();
}

void DynProgEnvironment::clearBuffers() {

    envGpu.ref_line.release();
    envGpu.occ_map.release();
    envGpu.dist_map_lon.release();
    envGpu.dist_map_path.release();

    for (int i = 0; i < envGpu.SIZE_DIR_DIST_MAX; ++i) {
        envGpu.dir_dist_map[i].release();
    }
}

void DynProgEnvironment::reinitBuffers(DynProgEnvParams& ps, bool force = false) {

    ps.updateStepSizes();

    bool fullReinitRequired = force
        || ps.t_steps != params.t_steps
        || ps.l_steps != params.l_steps
        || ps.s_steps != params.s_steps;

    params = ps;

    // copy params to gpu
    {
        cudaError_t err = cudaMemcpyToSymbol(dp_env_cu::params,
                                             &params,
                                             sizeof(DynProgEnvParams));
        checkCudaError(err, "copying env params failed");
    }

    if (!fullReinitRequired) {
        return;
    }

    clearBuffers();

    envGpu.occ_map.reinit(
            params.t_steps,
            params.s_steps,
            params.l_steps);

    envGpu.dist_map_lon.reinit(
            params.t_steps,
            params.s_steps,
            params.l_steps);

    envGpu.dist_map_path.reinit(
            params.t_steps,
            params.s_steps,
            1);

    for (int i = 0; i < envGpu.SIZE_DIR_DIST_MAX; ++i) {
        envGpu.dir_dist_map[i].reinit(
                params.s_steps,
                params.l_steps,
                params.dir_steps);
    }
}

void DynProgEnvironment::insertGeometry(std::span<vec<3>> geom,
                                        bool stationary) {

    if (refLine.points.size() == 0) {
        throw std::runtime_error(
                "cannot insert dyn object points with ref_line of 0 steps");
    }

    occRenderer.vertices.reserve(occRenderer.vertices.size() + geom.size());

    for (vec<3>& v : geom) {
        int t_idx = v[2] < params.dt_start ? 0 : (roundf((v[2] - params.dt_start) / params.dt) + 1.0);
        uint32_t flags = pow(2, t_idx);
        if (stationary) {
            flags |= ((uint32_t)1) << 31;
        }

        Vertex& v0 = occRenderer.vertices.emplace_back();
        v0.x = v[0] - refLine.x_offset;
        v0.y = v[1] - refLine.y_offset;
        v0.t = flags;
    }
}

void DynProgEnvironment::setRefLine(RefLinePoint* refLineData, size_t len, float step_size) {

    refLine = RefLine(refLineData, len, step_size);
}

void DynProgEnvironment::update() {

    // upload data to gpu

    if (refLine.points.size() * refLine.step_size < params.s_max) {
        throw std::runtime_error(
                "refline length = "
                + std::to_string(refLine.points.size() * refLine.step_size)
                + " < than environment s_max = "
                + std::to_string(params.s_max));
    }

    envGpu.ref_line.upload(refLine);

    // render occupancy via opengl

    OccupancyRendererParams occParams;
    occParams.s_max = params.s_max;
    occParams.l_max = params.l_max;
    occParams.ppm = 4.0;
    occRenderer.render(occParams);

    occRenderer.mapCudaTexture();
    envGpu.occupancy = occRenderer.occTexture;

    // convert occupancy map to frenet coordinates
    {
        dim3 gridSize(params.t_steps, params.s_steps, 1);
        dim3 blockSize(params.l_steps, 1, 1);
        dp_env_cu::occupancyToFrenet<<<gridSize, blockSize>>>(envGpu);

        cudaDeviceSynchronize();
        checkCudaError(cudaGetLastError(), "converting occupancy to frenet failed");
    }

    occRenderer.unmapCudaTexture();

    // cleanup occupancy
    {
        dim3 gridSize(params.t_steps, params.s_steps, 1);
        dim3 blockSize(1, 1, 1);
        dp_env_cu::closeIntersections<<<gridSize, blockSize>>>(envGpu);

        cudaDeviceSynchronize();
        checkCudaError(cudaGetLastError(), "closing occupancy on intersections failed");
    }

    // compute longitudinal distance map
    {
        dim3 gridSize(params.t_steps, params.l_steps, 1);
        dim3 blockSize(1, 1, 1);
        dp_env_cu::updateDistMapLon<<<gridSize, blockSize>>>(envGpu);

        cudaDeviceSynchronize();
        checkCudaError(cudaGetLastError(), "updating longitudinal distance map failed");
    }
}

void DynProgEnvironment::updateDistMapPath(PathState* pathGpu,
                                           int lenPath,
                                           float pathStepSize) {

    dim3 gridSize(params.t_steps, 1, 1);
    dim3 blockSize(1, 1, 1);
    dp_env_cu::updateDistMapPath<<<gridSize, blockSize>>>(
            envGpu, pathGpu, lenPath, pathStepSize);

    cudaDeviceSynchronize();
    checkCudaError(cudaGetLastError(), "updating path distance map failed");
}

float DynProgEnvironment::getDist(float t, float s, float l, float dt) {

    float res = 0.0;
    float* cudaRes = nullptr;
    cudaMalloc(&cudaRes, sizeof(float));
    dp_env_cu::copyDist<<<1,1>>>(envGpu, t, s, l, dt, cudaRes);
    cudaMemcpy(&res, cudaRes, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(cudaRes);

    return res;
}

void DynProgEnvironment::getOccMap(float* dst) {

    cudaMemcpy3DParms cpyParam;
    memset(&cpyParam, 0, sizeof(cudaMemcpy3DParms));
    cpyParam.dstPtr = make_cudaPitchedPtr((void*)dst,
                                          params.t_steps*sizeof(float),
                                          params.t_steps,
                                          params.s_steps);
    cpyParam.srcArray = envGpu.occ_map.arr;
    cpyParam.extent = make_cudaExtent(params.t_steps,
                                      params.s_steps,
                                      params.l_steps);
    cpyParam.kind = cudaMemcpyDeviceToHost;
    cudaError_t err = cudaMemcpy3D(&cpyParam);
    checkCudaError(err, "copying occupancy map from gpu to cpu failed");
}

void DynProgEnvironment::getDistMapLon(float* dst) {

    cudaMemcpy3DParms cpyParam;
    memset(&cpyParam, 0, sizeof(cudaMemcpy3DParms));
    cpyParam.dstPtr = make_cudaPitchedPtr((void*)dst,
                                          params.t_steps*sizeof(float2),
                                          params.t_steps,
                                          params.s_steps);
    cpyParam.srcArray = envGpu.dist_map_lon.arr;
    cpyParam.extent = make_cudaExtent(params.t_steps,
                                      params.s_steps,
                                      params.l_steps);
    cpyParam.kind = cudaMemcpyDeviceToHost;
    cudaError_t err = cudaMemcpy3D(&cpyParam);
    checkCudaError(err, "copying longitudinal distance map from gpu to cpu failed");
}

void DynProgEnvironment::getDistMapDir(int idx_t, float* dst) {

    cudaMemcpy3DParms cpyParam;
    memset(&cpyParam, 0, sizeof(cudaMemcpy3DParms));
    cpyParam.dstPtr = make_cudaPitchedPtr((void*)dst,
                                          params.s_steps*sizeof(float),
                                          params.s_steps,
                                          params.l_steps);
    cpyParam.srcArray = envGpu.dir_dist_map[idx_t].arr;
    cpyParam.extent = make_cudaExtent(params.s_steps,
                                      params.l_steps,
                                      params.dir_steps);
    cpyParam.kind = cudaMemcpyDeviceToHost;
    cudaError_t err = cudaMemcpy3D(&cpyParam);
    checkCudaError(err, "copying directional distance map from gpu to cpu failed");
}

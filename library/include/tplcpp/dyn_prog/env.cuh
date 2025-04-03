#pragma once

#include "tplcpp/eigen_helpers.hpp"

#include "tplcpp/dyn_prog/common.cuh"
#include "tplcpp/dyn_prog/arr_tex_surf.cuh"
#include "tplcpp/dyn_prog/occupancy_renderer.cuh"

namespace dp_env_cu { };

struct DynProgEnvParams { 

    float t_min = 0.0;
    float t_max = 10.0;
    float l_min = -5.0;
    float l_max = 5.0;
    float s_min = 0.0;
    float s_max = 200.0;

    float ds_max = 36.0;

    float dir_min = -M_PI_2;
    float dir_max = M_PI_2;

    float dt = 1.0;
    float dt_start = 1.0;

    int t_steps = 10;
    int l_steps = 21;
    int s_steps = 201;
    int dir_steps = 21;

    float scale_objects = 1.0;
    float dilation = 2.0;

    float _l_step_size = 0.0;
    float _s_step_size = 0.0;

    void updateStepSizes() {

        _l_step_size = (l_max - l_min) / (float)(l_steps - 1);
        _s_step_size = (s_max - s_min) / (float)(s_steps - 1);
    }
};

struct EnvironmentGpu {

    RefLineGpu ref_line;

    OccupancyTexture occupancy;

    ArrTexSurf<float> occ_map;
    ArrTexSurf<float2> dist_map_lon;
    ArrTexSurf<float> dist_map_path;

    inline static const int SIZE_DIR_DIST_MAX = 20;
    ArrTexSurf<float> dir_dist_map[SIZE_DIR_DIST_MAX];

    __device__ float interpVMax(float s);
    __device__ float interpDistField(float t, float s, float l);
    __device__ float2 interpDistMapLon(float t, float s, float l);
    __device__ float interpDistMapPath(float t, float s);
    __device__ float interpDirDistMap(float t, float s, float l, float dt);
};

struct DynProgEnvironment {

    RefLine refLine;

    DynProgEnvParams params;
    EnvironmentGpu envGpu;

    OccupancyRenderer occRenderer;

    DynProgEnvironment();
    ~DynProgEnvironment();

    DynProgEnvironment(DynProgEnvironment&) = delete;
    DynProgEnvironment(const DynProgEnvironment&) = delete;

    void clearBuffers();
    void reinitBuffers(DynProgEnvParams& ps, bool force);

    void insertGeometry(std::span<vec<3>> geom, bool stationary);

    void setRefLine(RefLinePoint* refLineData, size_t len, float step_size);

    void update();
    void updateDistMapPath(PathState* pathCpu, int lenPath, float pathStepSize);

    float getDist(float t, float s, float l, float dt);
    void getOccMap(float* dst);
    void getDistMapLon(float* dst);
    void getDistMapDir(int idx_t, float* dst);
};

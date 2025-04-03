#pragma once

#include <string>
#include <memory>
#include <vector>

#include "tplcpp/dyn_prog/frame_buffer.hpp"
#include "tplcpp/dyn_prog/common.cuh"

using GLenum = unsigned int;

struct OccupancyRendererParams {

    float s_max = 0.0;
    float l_max = 0.0;
    int ppm = 4.0;
};

struct Vertex {

    float x = 0.0;
    float y = 0.0;
    uint32_t t = 0;
};

struct OccupancyTexture {

    cudaTextureObject_t tex;
    float size = 0.0;
    int ppm = 0.0;
    int size_pixels = 0.0;
};

struct OccupancyRenderer { 

    static inline bool eglInitialized = false;

    OccupancyRenderer();
    ~OccupancyRenderer();

    OccupancyRenderer(OccupancyRenderer&) = delete;
    OccupancyRenderer(const OccupancyRenderer&) = delete;

    unsigned int idFragmentShader;
    unsigned int idVertexShader;
    unsigned int idShaderProgram;

    unsigned int idVertexArray;
    unsigned int idVertexBuffer;

    std::unique_ptr<FrameBuffer> fb;

    std::vector<Vertex> vertices;

    cudaGraphicsResource_t occRes;
    OccupancyTexture occTexture;

    void initEgl();
    unsigned int initGlShader(GLenum type, std::string source);
    unsigned int initGlShaderProgram();
    void initVertexBuffer();

    void checkGlError(const char* msg);

    void render(OccupancyRendererParams& params);

    void mapCudaTexture();
    void unmapCudaTexture();

    void readCartTex(unsigned int* data);
};

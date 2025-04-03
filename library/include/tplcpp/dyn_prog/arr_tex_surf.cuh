#pragma once

#include "tplcpp/dyn_prog/common.cuh"

template <class T>
struct ArrTexSurf {

    size_t s_x = 0;
    size_t s_y = 0;
    size_t s_z = 0;

    float x_min = 0.0;
    float x_max = 1.0;
    float y_min = 0.0;
    float y_max = 1.0;
    float z_min = 0.0;
    float z_max = 1.0;

    float step_size_x = 0.0;
    float step_size_y = 0.0;
    float step_size_z = 0.0;

    cudaArray* arr = nullptr;
    cudaTextureObject_t tex;
    cudaTextureObject_t tex_linear;
    cudaTextureObject_t surf;

    ArrTexSurf() = default;

    ArrTexSurf(size_t ns_x,
               size_t ns_y, 
               size_t ns_z,
               cudaTextureFilterMode filterMode = cudaFilterModeLinear,
               cudaTextureReadMode readMode = cudaReadModeElementType) {

        reinit(ns_x, ns_y, ns_y, filterMode, readMode);
    }

    void release() {

        if (nullptr != arr) {
            cudaFreeArray(arr);
            cudaDestroyTextureObject(tex);
            cudaDestroyTextureObject(tex_linear);
            cudaDestroySurfaceObject(surf);
        }

        arr = nullptr;
    }

    void reinit(size_t ns_x,
                size_t ns_y,
                size_t ns_z, 
                cudaTextureFilterMode filterMode = cudaFilterModeLinear,
                cudaTextureReadMode readMode = cudaReadModeElementType) {

        if (arr != nullptr && ns_x == s_x && ns_y == s_y && ns_z == s_z) {
            return;
        }

        release();

        s_x = ns_x;
        s_y = ns_y;
        s_z = ns_z;

        cudaChannelFormatDesc channelDescr = cudaCreateChannelDesc<T>();
        cudaExtent volumeSize = make_cudaExtent(s_x, s_y, s_z);

        cudaError_t err = cudaMalloc3DArray(&arr, &channelDescr, volumeSize);
        checkCudaError(err, "array allocation failed");

        // texture

        cudaResourceDesc texRes;
        memset(&texRes, 0, sizeof(cudaResourceDesc));
        texRes.resType = cudaResourceTypeArray;
        texRes.res.array.array = arr;

        cudaTextureDesc texDescr;
        memset(&texDescr, 0, sizeof(cudaTextureDesc));
        texDescr.normalizedCoords = 0;
        texDescr.filterMode = filterMode;
        texDescr.addressMode[0] = cudaAddressModeClamp;
        texDescr.addressMode[1] = cudaAddressModeClamp;
        texDescr.addressMode[2] = cudaAddressModeClamp;
        texDescr.readMode = readMode;

        err = cudaCreateTextureObject(&tex, &texRes, &texDescr, nullptr);
        checkCudaError(err, "texture creation failed");

        // texture linear interpolation

        memset(&texRes, 0, sizeof(cudaResourceDesc));
        texRes.resType = cudaResourceTypeArray;
        texRes.res.array.array = arr;

        memset(&texDescr, 0, sizeof(cudaTextureDesc));
        texDescr.normalizedCoords = 0;
        texDescr.filterMode = cudaFilterModeLinear;
        texDescr.addressMode[0] = cudaAddressModeClamp;
        texDescr.addressMode[1] = cudaAddressModeClamp;
        texDescr.addressMode[2] = cudaAddressModeClamp;
        texDescr.readMode = readMode;

        err = cudaCreateTextureObject(&tex_linear, &texRes, &texDescr, nullptr);
        checkCudaError(err, "texture linear creation failed");

        // surface

        cudaResourceDesc surfDescr;
        memset(&surfDescr, 0, sizeof(cudaResourceDesc));
        surfDescr.resType = cudaResourceTypeArray;
        surfDescr.res.array.array = arr;

        err = cudaCreateSurfaceObject(&surf, &surfDescr);
        checkCudaError(err, "surface creation failed");
    }

    void rescale(float x_min, float x_max,
                 float y_min, float y_max,
                 float z_min, float z_max) {

        this->x_min = x_min;
        this->x_max = x_max;
        this->y_min = y_min;
        this->y_max = y_max;
        this->z_min = z_min;
        this->z_max = z_max;

        step_size_x = (x_max - x_min) / (float)(s_x - 1);
        step_size_y = (y_max - y_min) / (float)(s_y - 1);
        step_size_z = (z_max - z_min) / (float)(s_z - 1);
    }

    /*
     * Lookup a value in a 3d texture using the texture cache.
     * Warning: Since textures are assumed to be read-only, writes to the texture
     * in the same kernel invocation may not be visible to this function!
     * Note: texture caches are cleared between kernel calls.
     */
    __inline__ __device__ T get_cached(float x, float y, float z) const {

        // performs a fast linear interpolation lookup using hardware acceleration
        // note: tex3D uses 9 bit fixed point precision (256 interpolation steps)
        // note: textures are voxel centered, therefore an offset of 0.5 is required

        return tex3D<T>(tex, x + 0.5f, y + 0.5f, z + 0.5f);
    }

    __inline__ __device__ T get_cached_scaled(float x, float y, float z) const {

        x = (x - x_min) / step_size_x;
        y = (y - y_min) / step_size_y;
        z = (z - z_min) / step_size_z;

        return tex3D<T>(tex, x + 0.5f, y + 0.5f, z + 0.5f);
    }

    __inline__ __device__ T lerp_cached_scaled(float x, float y, float z) const {

        x = (x - x_min) / step_size_x;
        y = (y - y_min) / step_size_y;
        z = (z - z_min) / step_size_z;

        return tex3D<T>(tex_linear, x + 0.5f, y + 0.5f, z + 0.5f);
    }

    __inline__ __device__ T get(size_t x, size_t y, size_t z) const {
        
        T value{};
        // note: the x component needs to be multiplied with the element size
        surf3Dread<T>(&value, surf, x * sizeof(T), y, z);
        return value;
    }

    __inline__ __device__ void set(size_t x, size_t y, size_t z, T value) {

        // note: the x component needs to be multiplied with the element size
        surf3Dwrite<T>(value, surf, x * sizeof(T), y, z);
    }
};

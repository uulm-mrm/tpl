#include "tplcpp/dyn_prog/common.cuh"

void checkCudaError(cudaError_t err, std::string msg) {

    if (cudaSuccess != err) {
        throw std::runtime_error(
                std::string(cudaGetErrorString(err))
                + ": " + msg);
    }
}

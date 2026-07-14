#pragma once
#include "cuda_runtime.h"
#include <string>
#include <stdexcept>


namespace CUDAUtils {
    inline void check_cuda_error(const cudaError_t cuda_error, const std::string& operation) {
        if (cuda_error != cudaSuccess) {
            throw std::runtime_error("CUDA failure: " + std::string(cudaGetErrorString(cuda_error)) + ". Operation: " + operation); // Credit to jefflarkin for this error check string
        }
    }
}


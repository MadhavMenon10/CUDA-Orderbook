#include "backtester.cuh"
#include "cuda_utils.hpp"

BacktestResults::BacktestResults(size_t num_configs) {
    CUDAUtils::check_cuda_error(cudaMalloc(reinterpret_cast<void**>(&pnl_results_), sizeof(std::int32_t) * num_configs),  "Allocate memory for pnl_results");
    num_configs_ = num_configs;
}

BacktestResults::~BacktestResults() {
    if (pnl_results_ != nullptr) {
        CUDAUtils::check_cuda_error(cudaFree(pnl_results_), "Free pnl_results");
        pnl_results_ = nullptr;
    }
}
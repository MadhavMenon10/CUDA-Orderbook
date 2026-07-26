#include "backtester.cuh"
#include "cuda_utils.hpp"

BacktestResults::BacktestResults(const StrategyConfig* configs, size_t num_configs) {
    CUDAUtils::check_cuda_error(cudaMalloc(reinterpret_cast<void**>(&pnl_results_), sizeof(std::int32_t) * num_configs),  "Allocate memory for pnl_results");
    CUDAUtils::check_cuda_error(cudaMalloc(reinterpret_cast<void**>(&configs_), sizeof(StrategyConfig) * num_configs),  "Allocate memory for configs");
    CUDAUtils::check_cuda_error(cudaMemcpy(configs_, configs, sizeof(StrategyConfig) * num_configs, cudaMemcpyHostToDevice), "Transfer configs into class");
    num_configs_ = num_configs;
}

BacktestResults::~BacktestResults() {
    if (pnl_results_ != nullptr) {
        CUDAUtils::check_cuda_error(cudaFree(pnl_results_), "Free pnl_results");
        pnl_results_ = nullptr;
    }
    if (configs_ != nullptr) {
        CUDAUtils::check_cuda_error(cudaFree(configs_), "Free configs_");
        configs_ = nullptr;
    }
}
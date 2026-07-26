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


__global__ void run_strategy(const OrderBookTick* ticks, size_t num_ticks, const StrategyConfig* configs, std::int32_t* pnl_results) {
    size_t idx = blockIdx.x;
    StrategyConfig config = configs[idx];
    bool holding_shares = false;
    std::int32_t pnl = 0;
    std::uint32_t price = 0;
    if (threadIdx.x == 0) {
        for (size_t i = 0; i < num_ticks; ++i) {
            std::uint32_t spread = ticks[i].asks[0].price - ticks[i].bids[0].price;
            std::uint32_t mid_price = (ticks[i].asks[0].price + ticks[i].bids[0].price) / 2;
            std::uint32_t spread_bps = (spread * 10000) / mid_price; // spread in basis points
            if (spread_bps < config.threshold && !holding_shares) {
                price = ticks[i].asks[0].price;
                holding_shares = true;
            } else if (spread_bps >= config.threshold && holding_shares) {
                std::int32_t sell_difference = static_cast<std::int32_t>(ticks[i].bids[0].price) - static_cast<std::int32_t>(price);
                pnl += (sell_difference * config.position_sizing);
                holding_shares = false;
            }
        }
        pnl_results[idx] = pnl;
    }
}
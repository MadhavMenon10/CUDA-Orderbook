#include "backtester.cuh"
#include "cuda_utils.hpp"

BacktestResults::BacktestResults(const StrategyConfig *configs, size_t num_configs) {
  CUDAUtils::check_cuda_error(cudaMalloc(reinterpret_cast<void**>(&pnl_results_), sizeof(std::int32_t) * num_configs), "Allocate memory for pnl_results");
  CUDAUtils::check_cuda_error(cudaMalloc(reinterpret_cast<void**>(&configs_), sizeof(StrategyConfig) * num_configs), "Allocate memory for configs");
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


/*
 * The specific threshold/lookback/position-size values generated here are arbitrary
 * This project's actual engineering weight is in the order book reconstruction and the
 * GPU-parallel infrastructure (the hash table, compaction, warp-sharded book state),
 * not in the trading strategy's own logic, which is intentionally a minimal,
 * and deliberately simple placeholder. A real strategy's parameter ranges would come from
 * domain research or backtesting against known market behavior, neither of which
 * this project is attempting to demonstrate.
 */
std::vector<StrategyConfig> generate_configs(size_t num_thresholds, size_t num_lookbacks, size_t num_position_sizes) {
  size_t total_configs = num_thresholds * num_lookbacks * num_position_sizes;
  std::vector<StrategyConfig> configs;
  configs.reserve(total_configs);
  for (size_t i = 0; i < num_thresholds; ++i) {
      for (size_t j = 0; j < num_lookbacks; ++j) {
          for (size_t k = 0; k < num_position_sizes; ++k) {
              StrategyConfig config;
              config.threshold = static_cast<std::uint16_t>(i + 1);
              config.lookback_window = (j + 1) * 5;
              config.position_size = static_cast<std::uint32_t>(k + 1) * 100;
              configs.push_back(config);
          }
      }
  }
  return configs;
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
      std::uint32_t mid_price =
          (ticks[i].asks[0].price + ticks[i].bids[0].price) / 2;
      std::uint32_t spread_bps =
          (spread * 10000) / mid_price; // spread in basis points
      if (spread_bps < config.threshold && !holding_shares) {
        price = ticks[i].asks[0].price;
        holding_shares = true;
      } else if (spread_bps >= config.threshold && holding_shares) {
        std::int32_t sell_difference =
            static_cast<std::int32_t>(ticks[i].bids[0].price) -
            static_cast<std::int32_t>(price);
        pnl += (sell_difference * config.position_size);
        holding_shares = false;
      }
    }
    pnl_results[idx] = pnl;
  }
}



void launch_run_strategy_kernel(const OrderBookSnapshot& order_book_snapshot, BacktestResults& backtest_results) {
    OrderBookTick* ticks = order_book_snapshot.get_ticks();
    size_t total_tick_count = order_book_snapshot.get_total_tick_count();
    const StrategyConfig* configs = backtest_results.get_configs();
    std::int32_t* pnl_results = backtest_results.get_pnl_results();
    size_t num_configs = backtest_results.get_num_configs();
    run_strategy<<<num_configs, 1>>>(ticks, total_tick_count, configs, pnl_results);
}

#pragma once
#include "reconstructor.cuh"
#include "types.hpp"
#include <cstdint>



class BacktestResults {
public:
  BacktestResults(const StrategyConfig* configs, size_t num_configs);
  ~BacktestResults();
  BacktestResults(const BacktestResults&) = delete;
  BacktestResults &operator=(const BacktestResults&) = delete;
  BacktestResults(BacktestResults&&) = delete;
  BacktestResults &operator=(BacktestResults&&) = delete;
  inline std::int32_t* get_pnl_results() const { return pnl_results_;};
  inline const StrategyConfig* get_configs() const { return configs_;};
  inline size_t get_num_configs() const {return num_configs_;};

private:
  std::int32_t* pnl_results_; // Signed so negative number represents a loss
  StrategyConfig* configs_;
  size_t num_configs_;
};

std::vector<StrategyConfig> generate_configs();

void launch_run_strategy_kernel(const OrderBookSnapshot& order_book_snapshot, BacktestResults& backtest_results);

__global__ void run_strategy(const OrderBookTick* ticks, size_t num_ticks, const StrategyConfig* configs, std::int32_t* pnl_results);

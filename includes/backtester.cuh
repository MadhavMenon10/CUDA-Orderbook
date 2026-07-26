#pragma once 
#include <cstdint>

class BacktestResults {
  public:
      BacktestResults(size_t num_configs);
      ~BacktestResults();
      BacktestResults(const BacktestResults&) = delete;
      BacktestResults& operator=(const BacktestResults&) = delete;
      BacktestResults(BacktestResults&&) = delete;
      BacktestResults& operator=(BacktestResults&&) = delete;
      inline const std::int32_t* get_pnl_results() const {return pnl_results_;};
      inline size_t get_num_configs() const {return num_configs_};
  private:
      std::int32_t* pnl_results_; // Signed so negative number represents a loss
      size_t num_configs_;
};
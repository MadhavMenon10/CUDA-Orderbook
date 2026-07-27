#include <iostream>
#include <algorithm>
#include "results_analysis.hpp"
#include "backtester.cuh"

void BackTestResultsAnalysis::find_best_pnl(const std::vector<std::int32_t>& pnl_results) {
    auto highest_pnl = std::max_element(pnl_results.begin(), pnl_results.end());
    std::cout << "The best PnL value was: " << *highest_pnl << "\n";
}

void BackTestResultsAnalysis::find_worst_pnl(const std::vector<std::int32_t>& pnl_results) {
    auto lowest_pnl = std::min_element(pnl_results.begin(), pnl_results.end());
    std::cout << "The worst PnL value was: " << *lowest_pnl << "\n";
}

void BackTestResultsAnalysis::report_best_strategy(const std::vector<std::int32_t>& pnl_results, const std::vector<StrategyConfig>& configs) {
    auto highest_pnl = std::max_element(pnl_results.begin(), pnl_results.end());
    size_t highest_pnl_idx = std::distance(pnl_results.begin(), highest_pnl);
    StrategyConfig best_config = configs[highest_pnl_idx];
    std::cout << "Threshold: " << best_config.threshold << "\n" << "Lookback window: " << best_config.lookback_window << "\n" << "Position size: " << best_config.position_size << "\n";
} 

#pragma once
#include <cstdint>
#include <vector>
#include "types.hpp"

namespace BackTestResultsAnalysis {
    void find_best_pnl(const std::vector<std::int32_t>& pnl_results);
    void find_worst_pnl(const std::vector<std::int32_t>& pnl_results);
    void report_best_strategy(const std::vector<std::int32_t>& pnl_results, const std::vector<StrategyConfig>& configs);
}

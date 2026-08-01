#include <iostream>
#include <vector>
#include "types.hpp"
#include "reader.hpp"
#include "decoder.hpp"
#include "baseline.hpp"
#include "results_analysis.hpp"
#include <chrono>

/*
 * Mirrors main.cu stage for stage so the two sets of timings line up. Parsing is left outside the
 * timers on both sides because it is the same host-side work either way and is not what the GPU
 * is being credited with, and the tick regrouping sits between the timers exactly where
 * TickCompactor sits in main.cu.
*/

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "ITCH File Path is missing\n";
        return 1;
    }
    try {
        std::string file_path(argv[1]);
        size_t one_mb = 1048576;
        size_t length_prefix = 2; // Test NASDAQ sample file wraps every message in a 2-byte big-endian length prefix
        ItchReader itch_reader(file_path, one_mb, length_prefix);
        std::array<uint8_t, 64> scratch_memory;
        size_t message_size = 0;
        size_t message_count = 0;
        while (itch_reader.get_next_message(scratch_memory.data(), scratch_memory.size(), message_size)) {
            ++message_count;
        }
        std::vector<DecodedOrder> messages;
        messages.reserve(message_count);
        itch_reader.reset_state();
        while (itch_reader.get_next_message(scratch_memory.data(), scratch_memory.size(), message_size)) {
            std::optional<DecodedOrder> decoded_order = ItchDecoder::decode_order(scratch_memory.data());
            if (decoded_order) {
                messages.push_back(*decoded_order); // We need to dereference the std::optional
            }
        }
        std::cout << "Total messages: " << messages.size() << "\n";
        size_t max_active_orders = 10000000; // Same estimate main.cu hands GPUHashTable
        CPUBaseline::Reconstructor reconstructor(max_active_orders);
        auto reconstruct_start = std::chrono::high_resolution_clock::now();
        reconstructor.run(messages);
        auto reconstruct_end = std::chrono::high_resolution_clock::now();
        double reconstruct_seconds = std::chrono::duration<double>(reconstruct_end - reconstruct_start).count();
        std::cout << "Reconstruction: " << messages.size() << " messages in " << reconstruct_seconds << "s (" << (messages.size() / reconstruct_seconds) << " messages/sec)\n";
        std::cout << "Ticks written: " << reconstructor.get_tick_count() << ", reconstruction failures: " << reconstructor.get_failure_count() << "\n";
        std::vector<std::vector<CPUBaseline::OrderBookTick>> symbol_ticks = CPUBaseline::group_ticks_by_symbol(reconstructor.get_ticks());
        std::vector<StrategyConfig> configs = CPUBaseline::generate_configs();
        CPUBaseline::Backtester backtester(configs);
        auto backtest_start = std::chrono::high_resolution_clock::now();
        backtester.run(symbol_ticks);
        auto backtest_end = std::chrono::high_resolution_clock::now();
        double backtest_seconds = std::chrono::duration<double>(backtest_end - backtest_start).count();
        std::cout << "Backtest: " << configs.size() << " configs in " << backtest_seconds << "s\n";
        BackTestResultsAnalysis::report_best_strategy(backtester.get_pnl_results(), configs);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}

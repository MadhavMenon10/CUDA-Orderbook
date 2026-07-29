#include <iostream>
#include <vector>
#include "types.hpp"
#include "reader.hpp"
#include "decoder.hpp"
#include "soa.hpp"
#include "cuda_utils.hpp"
#include "hash_table.cuh"
#include "symbol_compactor.cuh"
#include "reconstructor.cuh"
#include "backtester.cuh"
#include "tick_compactor.cuh"
#include "results_analysis.hpp"
#include <chrono>


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
        SoaArrays soa_arrays;
        soa_arrays.reserve(message_count);
        itch_reader.reset_state();
        // size_t messages_to_keep = 5000000; // temporary cap, diagnosing whether the redesign fails even at small scale
        // size_t i = 0;
        while (itch_reader.get_next_message(scratch_memory.data(), scratch_memory.size(), message_size)) { 
            std::optional<DecodedOrder> decoded_order = ItchDecoder::decode_order(scratch_memory.data());
            if (decoded_order) {
                soa_arrays.append(*decoded_order); // We need to dereference the optional
            }
            // ++i;
        }
        std::cout << "Total messages: " << soa_arrays.size() << "\n";
        size_t max_active_orders = 1000000; // Estimate
        GPUHashTable hash_table(max_active_orders);
        SymbolCompactor compacted_symbols(soa_arrays);
        OrderBookSnapshot order_book_snapshot(compacted_symbols, soa_arrays.size());
        auto reconstruct_start = std::chrono::high_resolution_clock::now();
        launch_reconstruction_kernel(compacted_symbols, order_book_snapshot, hash_table);
        CUDAUtils::check_cuda_error(cudaDeviceSynchronize(), "Wait for reconstruction to finish before backtesting");
        auto reconstruct_end = std::chrono::high_resolution_clock::now();
        double reconstruct_seconds = std::chrono::duration<double>(reconstruct_end - reconstruct_start).count();
        std::cout << "Reconstruction: " << soa_arrays.size() << " messages in " << reconstruct_seconds << "s (" << (soa_arrays.size() / reconstruct_seconds) << " messages/sec)\n";
        std::vector<StrategyConfig> configs = generate_configs();
        BacktestResults backtest_results(configs.data(), configs.size());
        TickCompactor tick_compactor(order_book_snapshot);
        auto backtest_start = std::chrono::high_resolution_clock::now();
        launch_run_strategy_kernel(tick_compactor, backtest_results);
        CUDAUtils::check_cuda_error(cudaDeviceSynchronize(), "Wait for backtesting to finish");
        auto backtest_end = std::chrono::high_resolution_clock::now();
        double backtest_seconds = std::chrono::duration<double>(backtest_end - backtest_start).count();
        std::cout << "Backtest: " << configs.size() << " configs in " << backtest_seconds << "s\n";
        std::vector<std::int32_t> pnl_results(backtest_results.get_num_configs());
        CUDAUtils::check_cuda_error(cudaMemcpy(pnl_results.data(), backtest_results.get_pnl_results(), sizeof(std::int32_t) * backtest_results.get_num_configs(),  cudaMemcpyDeviceToHost), "Copy PNL results to host");
        BackTestResultsAnalysis::report_best_strategy(pnl_results, configs);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
    return 0;
}

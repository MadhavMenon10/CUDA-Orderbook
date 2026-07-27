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
#include "results_analysis.hpp"


int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "ITCH File Path is missing\n";
        return 1;
    }
    try {
        std::string file_path(argv[1]);
        size_t one_mb = 1048576;
        ItchReader itch_reader(file_path, one_mb);
        std::array<uint8_t, 64> scratch_memory;
        size_t message_size = 0;
        size_t message_count = 0;
        while (itch_reader.get_next_message(scratch_memory.data(), scratch_memory.size(), message_size)) {
            ++message_count;
        }
        SoaArrays soa_arrays;
        soa_arrays.reserve(message_count);
        itch_reader.reset_state();
        while (itch_reader.get_next_message(scratch_memory.data(), scratch_memory.size(), message_size)) {
            DecodedOrder decoded_order = ItchDecoder::decode_order(scratch_memory.data());
            soa_arrays.append(decoded_order);
        }
        size_t max_active_orders = 1000000; // Estimate
        GPUHashTable hash_table(max_active_orders);
        SymbolCompactor compacted_symbols(soa_arrays);
        OrderBookSnapshot order_book_snapshot(compacted_symbols, message_count);
        launch_reconstruction_kernel(compacted_symbols, order_book_snapshot, hash_table);
        CUDAUtils::check_cuda_error(cudaDeviceSynchronize(), "Wait for reconstruction to finish before backtesting");
        std::vector<StrategyConfig> configs = generate_configs();
        BacktestResults backtest_results(configs.data(), configs.size());
        launch_run_strategy_kernel(order_book_snapshot, backtest_results);
        CUDAUtils::check_cuda_error(cudaDeviceSynchronize(), "Wait for backtesting to finish"); 
        std::vector<std::int32_t> pnl_results(backtest_results.get_num_configs());
        CUDAUtils::check_cuda_error(cudaMemcpy(pnl_results.data(), backtest_results.get_pnl_results(), sizeof(std::int32_t) * backtest_results.get_num_configs(),  cudaMemcpyDeviceToHost), "Copy PNL results to host");
        BackTestResultsAnalysis::report_best_strategy(pnl_results, configs);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
    return 0;
}

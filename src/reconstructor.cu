#include "reconstructor.cuh"

OrderBookSnapshot::OrderBookSnapshot(size_t total_message_count, size_t num_unique_symbols) {
    CUDAUtils::check_cuda_error(cudaMalloc(reinterpret_cast<void**>(&ticks_), sizeof(OrderBookTick) * total_message_count), "Allocate memory for ticks_");
    CUDAUtils::check_cuda_error(cudaMalloc(reinterpret_cast<void**>(&tick_start_offsets_), sizeof(size_t) * num_unique_symbols), "Allocate memory for tick_start_offsets_");
    CUDAUtils::check_cuda_error(cudaMalloc(reinterpret_cast<void**>(&tick_counts_), sizeof(size_t) * num_unique_symbols), "Allocate memory for tick_counts_");
    num_unique_symbols_ = num_unique_symbols;
}

OrderBookSnapshot::~OrderBookSnapshot() {
    if (ticks_ != nullptr) {
        CUDAUtils::check_cuda_error(cudaFree(ticks_), "Free ticks_");
        ticks_ = nullptr;
    }
    if (tick_start_offsets_ != nullptr) {
        CUDAUtils::check_cuda_error(cudaFree(tick_start_offsets_), "Free tick_start_offsets_");
        tick_start_offsets_ = nullptr;
    }
    if (tick_counts_ != nullptr) {
        CUDAUtils::check_cuda_error(cudaFree(tick_counts_), "Free tick_counts_");
        tick_counts_ = nullptr;
    }
    num_unique_symbols_ = 0;
}


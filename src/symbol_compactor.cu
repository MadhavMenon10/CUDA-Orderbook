#include "compactor.cuh"
#include <thrust/device_vector.h>
#include <thrust/sequence.h>

SymbolCompactor::SymbolCompactor(const SoaArrays& soa_array) {
    size_t total_message_count = soa_array.size();
    CUDAUtils::check_cuda_error(cudaMalloc(reinterpret_cast<void**>(&order_ids_), sizeof(std::uint64_t) * total_message_count), "Allocate memory for order_ids_ for symbol compaction");
    CUDAUtils::check_cuda_error(cudaMalloc(reinterpret_cast<void**>(&old_order_ids_), sizeof(std::uint64_t) * total_message_count), "Allocate memory for old_order_ids_ for symbol compaction");
    CUDAUtils::check_cuda_error(cudaMalloc(reinterpret_cast<void**>(&timestamps_), sizeof(std::uint64_t) * total_message_count), "Allocate memory for timestamps_ for symbol compaction");
    CUDAUtils::check_cuda_error(cudaMalloc(reinterpret_cast<void**>(&prices_), sizeof(std::uint32_t) * total_message_count), "Allocate memory for prices_ for symbol compaction");
    CUDAUtils::check_cuda_error(cudaMalloc(reinterpret_cast<void**>(&quantities_), sizeof(std::uint32_t) * total_message_count), "Allocate memory for quantities_ for symbol compaction");
    CUDAUtils::check_cuda_error(cudaMalloc(reinterpret_cast<void**>(&symbol_ids_), sizeof(std::uint16_t) * total_message_count), "Allocate memory for symbol_ids_ for symbol compaction");
    CUDAUtils::check_cuda_error(cudaMalloc(reinterpret_cast<void**>(&order_types_), sizeof(MessageType) * total_message_count), "Allocate memory for order_types_ for symbol compaction");
    CUDAUtils::check_cuda_error(cudaMalloc(reinterpret_cast<void**>(&sides_), sizeof(char) * total_message_count), "Allocate memory for sides_ for symbol compaction");
    thrust::device_vector<size_t> temp_sort_array(total_message_count);
    thrust::sequence(temp_sort_array.begin(), temp_sort_array.end());
    
    size_t* temp_sort_array_ptr = thrust::raw_pointer_cast(temp_sort_array.data());


    
}

SymbolCompactor::~SymbolCompactor() {
    if (order_ids_ != nullptr) {
        cudaFree(order_ids_);
    }
    if (old_order_ids_ != nullptr) {
        cudaFree(old_order_ids_);
    }
    if (timestamps_ != nullptr) {
        cudaFree(timestamps_);
    }
    if (prices_ != nullptr) {
        cudaFree(prices_);
    }
    if (quantities_ != nullptr) {
        cudaFree(quantities_);
    }
    if (symbol_ids_ != nullptr) {
        cudaFree(symbol_ids_);
    }
    if (order_types_ != nullptr) {
        cudaFree(order_types_);
    }
    if (sides_ != nullptr) {
        cudaFree(sides_);
    }
    if (unique_symbol_ids_ != nullptr) {
        cudaFree(unique_symbol_ids_);
    }
    if (symbol_start_offsets_ != nullptr) {
        cudaFree(symbol_start_offsets_);
    }
    if (symbol_counts_ != nullptr) {
        cudaFree(symbol_counts_);
    }
    num_distinct_symbols_ = 0;
}

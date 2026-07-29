#include "tick_compactor.cuh"
#include "cuda_utils.hpp"
#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/gather.h>
#include <thrust/sequence.h>


__global__ void extract_symbol_ids(const OrderBookTick* ticks, std::uint16_t* symbol_ids_out, size_t count) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < count) {
        symbol_ids_out[idx] = ticks[idx].symbol_id;
    }
}


TickCompactor::TickCompactor(const OrderBookSnapshot& order_book_snapshot) {
    size_t real_tick_count;
    CUDAUtils::check_cuda_error(cudaMemcpy(&real_tick_count, order_book_snapshot.get_tick_write_count(), sizeof(size_t), cudaMemcpyDeviceToHost), "Copy tick_write_count in constructor of TickCompactor");
    thrust::device_vector<std::uint16_t> symbol_ids_only(real_tick_count);
    std::uint16_t* symbol_ids_ptr = thrust::raw_pointer_cast(symbol_ids_only.data());
    int threads_per_block = 256;
    const OrderBookTick* uncompacted_ticks = order_book_snapshot.get_ticks();
    extract_symbol_ids<<<(real_tick_count + threads_per_block - 1) / threads_per_block, threads_per_block>>>(uncompacted_ticks, symbol_ids_ptr, real_tick_count);
    thrust::device_vector<size_t> sort_indices(real_tick_count);
    thrust::sequence(sort_indices.begin(), sort_indices.end());
    thrust::device_vector<std::uint16_t> sorted_symbol_ids(real_tick_count);
    thrust::device_vector<size_t> sorted_indices(real_tick_count);
    std::uint16_t* sorted_symbol_ids_ptr = thrust::raw_pointer_cast(sorted_symbol_ids.data());
    size_t* sorted_indices_ptr = thrust::raw_pointer_cast(sorted_indices.data());
    size_t* sort_indices_ptr = thrust::raw_pointer_cast(sort_indices.data());
    void* temp = nullptr;
    size_t temp_storage_bytes = 0;
    CUDAUtils::check_cuda_error(cub::DeviceRadixSort::SortPairs(temp, temp_storage_bytes, symbol_ids_ptr, sorted_symbol_ids_ptr, sort_indices_ptr, sorted_indices_ptr, real_tick_count), "First SortPairs call for TickCompactor");
    CUDAUtils::check_cuda_error(cudaMalloc(reinterpret_cast<void**>(&temp), temp_storage_bytes), "cudaMalloc for temp before second SortPairs call");
    CUDAUtils::check_cuda_error(cub::DeviceRadixSort::SortPairs(temp, temp_storage_bytes,symbol_ids_ptr, sorted_symbol_ids_ptr, sort_indices_ptr, sorted_indices_ptr, real_tick_count), "Second SortPairs call for TickCompactor");
    CUDAUtils::check_cuda_error(cudaFree(temp), "Free temp after TickCompactor sort");
    CUDAUtils::check_cuda_error( cudaMalloc(reinterpret_cast<void**>(&sorted_ticks_), sizeof(OrderBookTick) * real_tick_count), "Allocate memory for sorted_ticks_ in TickCompactor");
    gather_field<OrderBookTick>(uncompacted_ticks, sorted_ticks_, sorted_indices);
    CUDAUtils::check_cuda_error(cudaMalloc(reinterpret_cast<void**>(&unique_symbol_ids_), sizeof(std::uint16_t) * real_tick_count), "Allocate unique_symbol_ids_");
    CUDAUtils::check_cuda_error(cudaMalloc(reinterpret_cast<void**>(&tick_counts_), sizeof(size_t) * real_tick_count), "Allocate tick_counts_");
    CUDAUtils::check_cuda_error(cudaMalloc(reinterpret_cast<void**>(&tick_start_offsets_), sizeof(size_t) * real_tick_count), "Allocate tick_start_offsets_");
    thrust::device_vector<size_t> num_symbols_device(1);
    void* rle_temp = nullptr;
    size_t rle_temp_bytes = 0;
    CUDAUtils::check_cuda_error(cub::DeviceRunLengthEncode::Encode(rle_temp, rle_temp_bytes, sorted_symbol_ids_ptr, unique_symbol_ids_, tick_counts_, thrust::raw_pointer_cast(num_symbols_device.data()), real_tick_count), "First RunLengthEncode for TickCompactor");
    CUDAUtils::check_cuda_error(cudaMalloc(reinterpret_cast<void**>(&rle_temp), rle_temp_bytes), "cudaMalloc rle_temp");
    CUDAUtils::check_cuda_error(cub::DeviceRunLengthEncode::Encode(rle_temp, rle_temp_bytes, sorted_symbol_ids_ptr, unique_symbol_ids_, tick_counts_, thrust::raw_pointer_cast(num_symbols_device.data()), real_tick_count), "Second RunLengthEncode for TickCompactor");
    CUDAUtils::check_cuda_error(cudaMemcpy(&num_unique_symbols_, thrust::raw_pointer_cast(num_symbols_device.data()), sizeof(size_t), cudaMemcpyDeviceToHost), "Copy num_unique_symbols_ back to host");
    CUDAUtils::check_cuda_error(cudaFree(rle_temp), "Free rle_temp");
    thrust::device_ptr<size_t> counts_ptr(tick_counts_);
    thrust::device_ptr<size_t> offsets_ptr(tick_start_offsets_);
    thrust::exclusive_scan(counts_ptr, counts_ptr + num_unique_symbols_, offsets_ptr);
}


TickCompactor::~TickCompactor() {
    if (sorted_ticks_ != nullptr) {
        CUDAUtils::check_cuda_error(cudaFree(sorted_ticks_), "Free sorted_ticks_");
        sorted_ticks_ = nullptr;
    }
    if (tick_start_offsets_ != nullptr) {
        CUDAUtils::check_cuda_error(cudaFree(tick_start_offsets_), "Free tick_start_offsets_");
        tick_start_offsets_ = nullptr;
    }
    if (tick_counts_ != nullptr) {
        CUDAUtils::check_cuda_error(cudaFree(tick_counts_), "Free tick_counts_");
        tick_counts_ = nullptr;
    }
    if (unique_symbol_ids_ != nullptr) {
        CUDAUtils::check_cuda_error(cudaFree(unique_symbol_ids_), "Free unique_symbol_ids_");
        unique_symbol_ids_ = nullptr;
    }
    num_unique_symbols_ = 0;
}

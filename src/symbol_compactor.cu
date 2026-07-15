#include "compactor.cuh"
#include <thrust/device_vector.h>
#include <thrust/sequence.h>

// Compacts the symbols so that one reconstruction warp can be assigned to each symbol



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
    thrust::device_vector<size_t> sort_array_in(total_message_count);
    thrust::device_vector<size_t> sort_array_out(total_message_count);
    thrust::sequence(sort_array_in.begin(), sort_array_in.end());
    size_t* sort_array_in_ptr = thrust::raw_pointer_cast(sort_array_in.data());
    size_t* sort_array_out_ptr = thrust::raw_pointer_cast(sort_array_out.data());
    void* temp = nullptr;
    size_t temp_storage_bytes = 0;
    thrust::device_vector<std::uint16_t> unsorted_symbol_ids = copy_field_to_device(soa_array.get_symbol_ids(), total_message_count);
    const std::uint16_t* unsorted_symbol_ids_ptr = thrust::raw_pointer_cast(unsorted_symbol_ids.data());
    CUDAUtils::check_cuda_error(cub::DeviceRadixSort::SortPairs(temp, temp_storage_bytes, unsorted_symbol_ids_ptr, symbol_ids_, sort_array_in_ptr, sort_array_out_ptr, total_message_count), "First SortPair call");
    CUDAUtils::check_cuda_error(cudaMalloc(reinterpret_cast<void**>(&temp), temp_storage_bytes), "cudaMalloc for temp array before second SortPair call");
    CUDAUtils::check_cuda_error(cub::DeviceRadixSort::SortPairs(temp, temp_storage_bytes, unsorted_symbol_ids_ptr, symbol_ids_, sort_array_in_ptr, sort_array_out_ptr, total_message_count), "Second SortPair call");
    CUDAUtils::check_cuda_error(cudaFree(temp), "Free temp");
    sort_field(soa_array.get_order_ids(), order_ids_, total_message_count, sort_array_out);
    sort_field(soa_array.get_old_order_ids(), old_order_ids_, total_message_count, sort_array_out);
    sort_field(soa_array.get_timestamps(), timestamps_, total_message_count, sort_array_out);
    sort_field(soa_array.get_prices(), prices_, total_message_count, sort_array_out);
    sort_field(soa_array.get_quantities(), quantities_, total_message_count, sort_array_out);
    sort_field(soa_array.get_order_types(), order_types_, total_message_count, sort_array_out);
    sort_field(soa_array.get_sides(), sides_, total_message_count, sort_array_out);
    CUDAUtils::check_cuda_error(cudaMalloc(reinterpret_cast<void**>(&unique_symbol_ids_), sizeof(std::uint16_t) * total_message_count), "Allocate memory for unique_symbol_ids_");
    CUDAUtils::check_cuda_error(cudaMalloc(reinterpret_cast<void**>(&symbol_counts_), sizeof(size_t) * total_message_count), "Allocate memory for symbol_counts");
    thrust::device_vector<size_t> num_symbols_device(1); 
    temp = nullptr;
    temp_storage_bytes = 0;
    CUDAUtils::check_cuda_error(cub::DeviceRunLengthEncode::Encode(temp, temp_storage_bytes, symbol_ids_, unique_symbol_ids_, symbol_counts_, thrust::raw_pointer_cast(num_symbols_device.data()), total_message_count), "First RunLengthEncode call");
    CUDAUtils::check_cuda_error(cudaMalloc(reinterpret_cast<void**>(&temp), temp_storage_bytes), "cudaMalloc for temp array before second encode call");
    CUDAUtils::check_cuda_error(cub::DeviceRunLengthEncode::Encode(temp, temp_storage_bytes, symbol_ids_, unique_symbol_ids_, symbol_counts_, thrust::raw_pointer_cast(num_symbols_device.data()), total_message_count), "Second RunLengthEncode call");
    CUDAUtils::check_cuda_error(cudaMemcpy(&num_unique_symbols_, thrust::raw_pointer_cast(num_symbols_device.data()), sizeof(size_t), cudaMemcpyDeviceToHost), "Copy num_unique_symbols back to host");
    CUDAUtils::check_cuda_error(cudaFree(temp), "Free temp after second encode run");
    thrust::device_ptr<size_t> counts_ptr(symbol_counts_);
    thrust::device_ptr<size_t> offsets_ptr(symbol_start_offsets_);
    thrust::exclusive_scan(counts_ptr, counts_ptr + num_unique_symbols_, offsets_ptr);
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
    num_unique_symbols_ = 0;
}

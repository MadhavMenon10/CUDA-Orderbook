#include "hash_table.cuh"

GPUHashTable::GPUHashTable(size_t max_active_orders) {
    capacity_ = max_active_orders * 2;
    CUDAUtils::check_cuda_error(cudaMalloc(reinterpret_cast<void**>(&order_ids_), sizeof(std::uint64_t) * capacity_), "GPU hash table memory allocation for order_ids_");
    CUDAUtils::check_cuda_error(cudaMemset(order_ids_, 0xFF, capacity_ * sizeof(std::uint64_t)), "Fill every byte of order_ids_ with UINT64_MAX");
    CUDAUtils::check_cuda_error(cudaMalloc(reinterpret_cast<void**>(&quantities_), sizeof(std::uint32_t) * capacity_), "GPU hash table memory allocation for quantities_");
    CUDAUtils::check_cuda_error(cudaMalloc(reinterpret_cast<void**>(&prices_), sizeof(std::uint32_t) * capacity_), "GPU Hash table memory allocation for prices_");
    CUDAUtils::check_cuda_error(cudaMalloc(reinterpret_cast<void**>(&symbol_ids_), sizeof(std::uint16_t) * capacity_), "GPU Hash table memory allocation for symbol_ids_");
    CUDAUtils::check_cuda_error(cudaMalloc(reinterpret_cast<void**>(&sides_), sizeof(char) * capacity_), "GPU Hash table memory allocation for sides_");
}


GPUHashTable::~GPUHashTable() {
    if (order_ids_ != nullptr) {
        cudaFree(order_ids_);
    }
    if (quantities_ != nullptr) {
        cudaFree(quantities_);
    }
    if (prices_ != nullptr) {
        cudaFree(prices_);
    }
    if (symbol_ids_ != nullptr) {
        cudaFree(symbol_ids_);
    }
    if (sides_ != nullptr) {
        cudaFree(sides_);
    }
    capacity_ = 0;
}

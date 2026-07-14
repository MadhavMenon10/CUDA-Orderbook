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

__device__ std::uint64_t hash(std::uint64_t order_id) {
    // Used part of the splitmix64 algorithm as a hashing function
    std::uint64_t hashed_order_id = (order_id ^ (order_id >> 30)) * 0xbf58476d1ce4e5b9;  /* xor the variable with the variable right bit shifted 30 then multiply by a constant */
    hashed_order_id = (hashed_order_id ^ (hashed_order_id >> 27)) * 0x94d049bb133111eb;  /* xor the variable with the variable right bit shifted 27 then multiply by a constant */
    return hashed_order_id ^ (hashed_order_id >> 31);                             /* return the variable xored with itself right bit shifted 31 */
}

__global__ void insert(std::uint64_t order_id, std::uint32_t order_quantity, std::uint32_t order_price, std::uint16_t order_symbol_id, char order_side, std::uint64_t* order_ids, std::uint32_t* quantities, std::uint32_t* prices, std::uint16_t* symbol_ids, char* sides, size_t hash_table_capacity) {
    std::uint64_t hash_index = hash(order_id) % hash_table_capacity;
    
}

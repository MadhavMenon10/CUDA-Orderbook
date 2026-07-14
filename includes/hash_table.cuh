#pragma once
#include <cstdint>
#include <cstddef>
#include "cuda_utils.hpp"

/* 
 * Hash table kept on the GPU so that during reconstruction, no round trip has to be made back to the CPU
 * If the hash table lives on the GPU, the latency involced in handing data to the CPU and back can be removed
*/

class GPUHashTable {
    public:
        GPUHashTable(size_t max_active_orders);
        ~GPUHashTable();
        GPUHashTable(const GPUHashTable&) = delete;
        GPUHashTable& operator=(const GPUHashTable&) = delete;
        GPUHashTable(GPUHashTable&&) = delete;
        GPUHashTable& operator=(GPUHashTable&&) = delete;
        inline std::uint64_t* get_order_ids() const {return order_ids_;};
        inline std::uint32_t* get_quantities() const {return quantities_;};
        inline std::uint32_t* get_prices() const {return prices_;};
        inline std::uint16_t* get_symbol_ids() const {return symbol_ids_;};
        inline char* get_sides() const {return sides_;};
        inline size_t capacity() const {return capacity_;};
    private:
        std::uint64_t* order_ids_; // key for the hash table, remaining member variables take SoA approach
        std::uint32_t* quantities_;
        std::uint32_t* prices_;
        std::uint16_t* symbol_ids_;
        char* sides_;
        size_t capacity_;  
};

__device__ std::uint64_t hash(std::uint64_t order_id);

__global__ void insert(std::uint64_t order_id, std::uint32_t order_quantity, std::uint32_t order_price, std::uint16_t order_symbol_id, char order_side, std::uint64_t* order_ids, std::uint32_t* quantities, std::uint32_t* prices, std::uint16_t* symbol_ids, char* sides, size_t hash_table_capacity);


#pragma once
#include <cstdint>
#include <cstddef>
#include "cuda_utils.hpp"

class GPUHashTable {
    public:
        GPUHashTable(size_t max_active_orders);
        ~GPUHashTable();
        GPUHashTable(const GPUHashTable&) = delete;
        GPUHashTable& operator=(const GPUHashTable&) = delete;
        GPUHashTable(GPUHashTable&&) = delete;
        GPUHashTable& operator=(GPUHashTable&&) = delete;
    private:
        std::uint64_t* order_ids_; // key for the hash table, remaining member variables take SoA approach
        std::uint32_t* quantities_;
        std::uint32_t* prices_;
        std::uint16_t* symbol_ids_;
        char* sides_;
        size_t capacity_;  
};

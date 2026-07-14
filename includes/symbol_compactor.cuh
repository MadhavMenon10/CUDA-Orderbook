#pragma once
#include "soa.hpp"
#include "cuda_utils.hpp"
#include <cstdint>
#include <cub/cub.h>

class SymbolCompactor {
    public:
        SymbolCompactor(const SoaArrays& soa_array);
        ~SymbolCompactor();
        SymbolCompactor(const SymbolCompactor&) = delete;
        SymbolCompactor& operator=(const SymbolCompactor&) = delete;
        SymbolCompactor(SymbolCompactor&&) = delete;
        SymbolCompactor& operator=(SymbolCompactor&&) = delete;
    private:
        std::uint64_t* order_ids_;
        std::uint64_t* old_order_ids_;
        std::uint64_t* timestamps_;
        std::uint32_t* prices_;
        std::uint32_t* quantities_;
        std::uint16_t* symbol_ids_;
        MessageType* order_types_;
        char* sides_;
        std::uint16_t* unique_symbol_ids_;
        size_t* symbol_start_offsets_;
        size_t* symbol_counts_;
        size_t num_distinct_symbols_;
};

// template <typename T>
// void gather_field(const T* source_ptr, const T* dest_ptr, const thrust::device_vector<T>& map_vector, size_t count) {
//     thrust::device_vector<const T> src_ptr(source_ptr);
//     thrust::device_vector<const T> dst_ptr(dest_ptr);
//     thrust::gather(map.begin(), map.end(), src_ptr, dst_ptr);
// }

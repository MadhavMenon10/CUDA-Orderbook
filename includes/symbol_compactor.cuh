#pragma once
#include "soa.hpp"
#include "cuda_utils.hpp"
#include <cstdint>
#include <cub/cub.cuh>

class SymbolCompactor {
    public:
        SymbolCompactor(const SoaArrays& soa_array);
        ~SymbolCompactor();
        SymbolCompactor(const SymbolCompactor&) = delete;
        SymbolCompactor& operator=(const SymbolCompactor&) = delete;
        SymbolCompactor(SymbolCompactor&&) = delete;
        SymbolCompactor& operator=(SymbolCompactor&&) = delete;
        inline std::uint64_t* get_order_ids() const {return order_ids_;};
        inline std::uint64_t* get_old_order_ids() const {return old_order_ids_;};
        inline std::uint64_t* get_timestamps() const {return timestamps_;};
        inline std::uint32_t* get_prices() const {return prices_;};
        inline std::uint32_t* get_quantities() const {return quantities_;};
        inline std::uint16_t* get_symbol_ids() const {return symbol_ids_;};
        inline MessageType* get_order_types() const {return order_types_;};
        inline char* get_sides() const {return sides_;};
        inline std::uint16_t* get_unique_symbol_ids() const {return unique_symbol_ids_;};
        inline size_t* get_symbol_start_offsets() const {return symbol_start_offsets_;};
        inline size_t* get_symbol_counts() const {return symbol_counts_;};
        inline size_t get_num_unique_symbols() const {return num_unique_symbols_;};
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
        size_t num_unique_symbols_;
};

template <typename T>
void gather_field(const T* source_ptr, T* dest_ptr, const thrust::device_vector<size_t>& map_vector) {
    thrust::device_ptr<const T> src_ptr(source_ptr);
    thrust::device_ptr<T> dst_ptr(dest_ptr);
    thrust::gather(map_vector.begin(), map_vector.end(), src_ptr, dst_ptr);
}

template <typename T>
thrust::device_vector<T> copy_field_to_device(const T* host_ptr, size_t num_elements) {
    thrust::device_vector<T> device_vector(host_ptr, host_ptr + num_elements);
    return device_vector;
}

template <typename T>
void sort_field(const T* host_ptr, T* dest_ptr, size_t num_elements, const thrust::device_vector<size_t>& map_vector) {
    thrust::device_vector<T> device_vector = copy_field_to_device(host_ptr, num_elements);
    T* device_ptr = thrust::raw_pointer_cast(device_vector.data());
    gather_field(device_ptr, dest_ptr, map_vector);
}

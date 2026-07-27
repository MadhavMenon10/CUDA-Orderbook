#pragma once
#include "types.hpp"
#include "cuda_utils.hpp"

class SoaArrays {
    public:
        SoaArrays();
        ~SoaArrays();
        SoaArrays(const SoaArrays &) = delete;
        SoaArrays &operator=(const SoaArrays &) = delete;
        SoaArrays(SoaArrays &&) = delete;
        SoaArrays &operator=(SoaArrays &&) = delete;
        void append(const DecodedOrder &decoded_order);
        void reserve(const size_t message_count);
        inline const std::uint64_t* get_order_ids() const { return order_ids_; };
        inline const std::uint64_t* get_old_order_ids() const { return old_order_ids_; };
        inline const std::uint64_t* get_timestamps() const { return timestamps_;};
        inline const std::uint32_t* get_quantities() const { return quantities_; };
        inline const std::uint32_t* get_prices() const { return prices_; };
        inline const std::uint16_t* get_symbol_ids() const { return symbol_ids_; };
        inline const MessageType* get_order_types() const {return order_types_;};
        inline const char* get_sides() const { return sides_; };
        inline size_t size() const {return size_;};
        inline size_t capacity() const { return capacity_; };

    private:
        std::uint64_t* order_ids_;
        std::uint64_t* old_order_ids_;
        std::uint64_t* timestamps_;
        std::uint32_t* prices_;
        std::uint32_t* quantities_;
        std::uint16_t* symbol_ids_;
        MessageType* order_types_;
        char* sides_;
        size_t size_;
        size_t capacity_;
};

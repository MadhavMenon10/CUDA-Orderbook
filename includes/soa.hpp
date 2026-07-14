#pragma once
#include "types.hpp"
#include "cuda_utils.hpp"

class SoaArrays {
    public:
        SoaArrays();
        ~SoaArrays();
        SoaArrays(const SoaArrays&) = delete;
        SoaArrays& operator=(const SoaArrays&) = delete;
        SoaArrays(SoaArrays&&) = delete;
        SoaArrays& operator=(SoaArrays&&) = delete;
        void append(const DecodedOrder& decoded_order);
        void reserve(const size_t message_count);

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

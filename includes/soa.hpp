#pragma once
#include "types.hpp"
#include "cuda_runtime.h"

class SoaArrays {
    public:
        SoaArrays();
        ~SoaArrays();
        void append(const DecodedOrder& decoded_order);
        reserve();


    private:
        std::uint64_t* order_ids_;
        std::uint64_t* old_order_ids_;
        std::uint64_t* timestamps_;
        std::uint32_t* prices_;
        std::uint32_t* quantities_;
        std::uint16_t* symbol_ids_;
        MessageType* order_types_;
        char* sides;
        size_t size_;
        size_t capacity_;
        
};

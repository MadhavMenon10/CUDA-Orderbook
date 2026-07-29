#pragma once
#include "reconstructor.cuh"
#include <cstdint>
#include <cstddef>

class TickCompactor {
    public:
        TickCompactor(const OrderBookSnapshot& order_book_snapshot);
        ~TickCompactor();
        TickCompactor(const TickCompactor&) = delete;
        TickCompactor& operator=(const TickCompactor&) = delete;
        TickCompactor(TickCompactor&&) = delete;
        TickCompactor& operator=(TickCompactor&&) = delete;
        inline const OrderBookTick* get_sorted_ticks() const {return sorted_ticks_;};
        inline const size_t* get_tick_start_offsets() const {return tick_start_offsets_;};
        inline const size_t* get_tick_counts() const {return tick_counts_;};
        inline const std::uint16_t* get_unique_symbol_ids() const {return unique_symbol_ids_;};
        inline size_t get_num_unique_symbols() const {return num_unique_symbols_;};
    private:
        OrderBookTick* sorted_ticks_;
        size_t* tick_start_offsets_;
        size_t* tick_counts_;
        std::uint16_t* unique_symbol_ids_;
        size_t num_unique_symbols_;
};

__global__ void extract_symbol_ids(const OrderBookTick* ticks, std::uint16_t* symbol_ids_out, size_t count);

#pragma once
#include <cstdint>
#include <cstddef>

struct PriceLevel {
    std::uint32_t price;
    std::uint32_t quantity;
};

struct OrderBookTick {
    PriceLevel bids[5];
    PriceLevel asks[5]; // Avoids std::array since these are consumed inside a CUDA kernel
    std::uint64_t timestamp;
};

class OrderBookSnapshot {
    public:
        OrderBookSnapshot(size_t total_message_count, size_t num_unique_symbols);
        ~OrderBookSnapshot();
        OrderBookSnapshot(const OrderBookSnapshot&) = delete;
        OrderBookSnapshot& operator=(const OrderBookSnapshot&) = delete;
        OrderBookSnapshot(OrderBookSnapshot&&) = delete;
        OrderBookSnapshot& operator=(OrderBookSnapshot&&) = delete;
    private:
        OrderBookTick* ticks_; 
        size_t* tick_start_offsets_;
        size_t* tick_counts_;
        size_t num_unique_symbols_;
};

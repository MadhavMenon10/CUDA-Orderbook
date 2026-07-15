#pragma once
#include <cstdint>

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
        OrderBookSnapshot();
        ~OrderBookSnapshot();
        OrderBookSnapshot(const OrderBookSnapshot&) = delete;
        OrderBookSnapshot& operator=(const OrderBookSnapshot&) = delete;
        OrderBookSnapshot(OrderBookSnapshot&&) = delete;
        OrderBookSnapshot& operator=(OrderBookSnapshot&&) = delete;
    private:
        

};
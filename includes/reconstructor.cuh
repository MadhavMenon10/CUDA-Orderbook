#pragma once
#include <cstdint>
#include <cstddef>
#include "types.hpp"

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


// Following structs are created so that our reconstruction kernel doesn't take like 20 parameters
struct CompactedMessage {
    std::uint64_t* order_ids;
    std::uint64_t* timestamps;
    std::uint32_t* prices;
    std::uint32_t* quantities;
    std::uint16_t* symbol_ids;
    MessageType* order_types;
    char* sides;
    size_t* symbol_start_offsets;
    size_t* symbol_counts;
};

struct HashTableData {
    std::uint64_t* order_ids; 
    std::uint32_t* quantities;
    std::uint32_t* prices;
    std::uint16_t* symbol_ids;
    char* sides;
    size_t capacity;  
};

struct OrderBookSnapshotData {
    OrderBookTick* ticks; 
    size_t* tick_start_offsets;
    size_t* tick_counts;
};


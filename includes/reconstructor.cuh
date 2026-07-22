#pragma once
#include <cstdint>
#include <cstddef>
#include "types.hpp"
#include "hash_table.cuh"

constexpr std::uint32_t PRICE_EMPTY_SLOT_SENTINEL = UINT32_MAX;

/*
 * Each of the warp's 32 lanes shards a subset of its symbol's active price levels,
 * decided by hash(price) % 32. Since more than 32 distinct prices can exist for a
 * busy symbol, a single lane may end up owning several unrelated prices at once.
 * Shared memory has no dynamic growth once the kernel launches, so this bounds
 * each lane's local price-level array to a fixed, generously-provisioned size.
 * 16 is a reasoned starting guess, not yet validated against real ITCH price
 * distributions -- worth revisiting once this can be profiled on real data.
*/

constexpr int MAX_LEVELS_PER_LANE = 16; 
constexpr int WARP_SIZE = 32;

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
    std::uint64_t* old_order_ids;
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

__device__ bool reconstruct_add(CompactedMessage message_params, HashTableData hash_table_data, PriceLevel shared_levels[][MAX_LEVELS_PER_LANE], int idx);

__device__ bool reconstruct_cancel(CompactedMessage message_params, HashTableData hash_table_data, PriceLevel shared_levels[][MAX_LEVELS_PER_LANE], int idx);

__device__ bool reconstruct_execute(CompactedMessage message_params, HashTableData hash_table_data, PriceLevel shared_levels[][MAX_LEVELS_PER_LANE], int idx);

__device__ bool reconstruct_execute_with_price(CompactedMessage message_params, HashTableData hash_table_data, PriceLevel shared_levels[][MAX_LEVELS_PER_LANE], int idx);

__device__ bool reconstruct_delete(CompactedMessage message_params, HashTableData hash_table_data, PriceLevel shared_levels[][MAX_LEVELS_PER_LANE], int idx);

__device__ bool reconstruct_replace(CompactedMessage message_params, HashTableData hash_table_data, PriceLevel shared_levels[][MAX_LEVELS_PER_LANE], int idx);

__global__ void reconstruct(CompactedMessage messages, HashTableData hash_table_data, OrderBookSnapshotData output);

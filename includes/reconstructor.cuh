#pragma once
#include <cstdint>
#include <cstddef>
#include "types.hpp"
#include "hash_table.cuh"
#include "symbol_compactor.cuh"

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

constexpr int MAX_LEVELS_PER_LANE = 64; // Number of distinct price levels a single one of the warp's 32 lanes is allowed to hold in its own small local array at once
constexpr int WARP_SIZE = 32;

struct PriceLevel {
    std::uint32_t price;
    std::uint32_t quantity;
};

struct OrderBookTick {
    PriceLevel bids[5];
    PriceLevel asks[5]; // Avoids std::array since these are consumed inside a CUDA kernel
    std::uint64_t timestamp;
    std::uint16_t symbol_id;
};

class OrderBookSnapshot {
    public:
        OrderBookSnapshot(const SymbolCompactor& compacted_symbols, size_t total_message_count);
        ~OrderBookSnapshot();
        OrderBookSnapshot(const OrderBookSnapshot&) = delete;
        OrderBookSnapshot& operator=(const OrderBookSnapshot&) = delete;
        OrderBookSnapshot(OrderBookSnapshot&&) = delete;
        OrderBookSnapshot& operator=(OrderBookSnapshot&&) = delete;
        inline OrderBookTick* get_ticks() const {return ticks_;};
        inline size_t get_num_unique_symbols() const {return num_unique_symbols_;};
        inline size_t get_total_tick_count() const {return total_tick_count_;};
        inline size_t* get_tick_write_count() const {return tick_write_count_;};
        inline size_t get_ticks_capacity() const {return ticks_capacity_;};
    private:
        OrderBookTick* ticks_; 
        size_t num_unique_symbols_;
        size_t total_tick_count_;
        size_t* tick_write_count_;
        size_t ticks_capacity_;
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
    size_t* tick_write_count;
    size_t ticks_capacity;
};


__device__ bool reconstruct_add(CompactedMessage message_params, HashTableData hash_table_data, PriceLevel bid_levels[][MAX_LEVELS_PER_LANE], PriceLevel ask_levels[][MAX_LEVELS_PER_LANE], int idx);

__device__ bool reconstruct_cancel(CompactedMessage message_params, HashTableData hash_table_data, PriceLevel bid_levels[][MAX_LEVELS_PER_LANE], PriceLevel ask_levels[][MAX_LEVELS_PER_LANE], int idx);

__device__ bool reconstruct_execute(CompactedMessage message_params, HashTableData hash_table_data, PriceLevel bid_levels[][MAX_LEVELS_PER_LANE], PriceLevel ask_levels[][MAX_LEVELS_PER_LANE], int idx);

__device__ bool reconstruct_execute_with_price(CompactedMessage message_params, HashTableData hash_table_data, PriceLevel bid_levels[][MAX_LEVELS_PER_LANE], PriceLevel ask_levels[][MAX_LEVELS_PER_LANE], int idx);

__device__ bool reconstruct_delete(CompactedMessage message_params, HashTableData hash_table_data, PriceLevel bid_levels[][MAX_LEVELS_PER_LANE], PriceLevel ask_levels[][MAX_LEVELS_PER_LANE], int idx);

__device__ bool reconstruct_replace(CompactedMessage message_params, HashTableData hash_table_data, PriceLevel bid_levels[][MAX_LEVELS_PER_LANE], PriceLevel ask_levels[][MAX_LEVELS_PER_LANE], int idx);

__device__ PriceLevel find_best_level(PriceLevel levels[][MAX_LEVELS_PER_LANE], bool find_max, const PriceLevel* top_5);

__global__ void reconstruct(CompactedMessage messages, HashTableData hash_table_data, OrderBookSnapshotData output);

void launch_reconstruction_kernel(const SymbolCompactor& compacted_symbols, OrderBookSnapshot& order_book_snapshot, GPUHashTable& hash_table);

#include "reconstructor.cuh"

OrderBookSnapshot::OrderBookSnapshot(size_t total_message_count, size_t num_unique_symbols) {
    CUDAUtils::check_cuda_error(cudaMalloc(reinterpret_cast<void**>(&ticks_), sizeof(OrderBookTick) * total_message_count), "Allocate memory for ticks_");
    CUDAUtils::check_cuda_error(cudaMalloc(reinterpret_cast<void**>(&tick_start_offsets_), sizeof(size_t) * num_unique_symbols), "Allocate memory for tick_start_offsets_");
    CUDAUtils::check_cuda_error(cudaMalloc(reinterpret_cast<void**>(&tick_counts_), sizeof(size_t) * num_unique_symbols), "Allocate memory for tick_counts_");
    num_unique_symbols_ = num_unique_symbols;
}

OrderBookSnapshot::~OrderBookSnapshot() {
    if (ticks_ != nullptr) {
        CUDAUtils::check_cuda_error(cudaFree(ticks_), "Free ticks_");
        ticks_ = nullptr;
    }
    if (tick_start_offsets_ != nullptr) {
        CUDAUtils::check_cuda_error(cudaFree(tick_start_offsets_), "Free tick_start_offsets_");
        tick_start_offsets_ = nullptr;
    }
    if (tick_counts_ != nullptr) {
        CUDAUtils::check_cuda_error(cudaFree(tick_counts_), "Free tick_counts_");
        tick_counts_ = nullptr;
    }
    num_unique_symbols_ = 0;
}

__device__ bool reconstruct_add(CompactedMessage message_params, HashTableData hash_table_data, PriceLevel* local_levels, int idx) {
    std::uint64_t order_id = message_params.order_ids[idx];
    std::uint32_t price = message_params.prices[idx];
    std::uint32_t quantity = message_params.quantities[idx];
    std::uint16_t symbol_id = message_params.symbol_ids[idx];
    char side = message_params.sides[idx];
    bool insert_local_level = false;
    bool hash_insert = hash_table_insert(order_id, quantity, price, symbol_id, side, hash_table_data.order_ids, hash_table_data.quantities, hash_table_data.prices, hash_table_data.symbol_ids, hash_table_data.sides, hash_table_data.capacity);
    for (int i = 0; i < MAX_LEVELS_PER_LANE; ++i) {
        PriceLevel& price_level = local_levels[i];
        if (price_level.price == PRICE_EMPTY_SLOT_SENTINEL) {
            price_level.price = price;
            price_level.quantity = quantity;
            insert_local_level = true;
            break;
        } else if (price_level.price == price) {
            price_level.quantity += quantity;
            insert_local_level = true;
            break;
        }
    }
    return hash_insert && insert_local_level;
}

__device__ bool reconstruct_cancel(CompactedMessage message_params, HashTableData hash_table_data, PriceLevel* local_levels, int idx) {
    std::uint64_t order_id = message_params.order_ids[idx];
    std::uint32_t cancel_quantity = message_params.quantities[idx];
    HashTableEntry lookup_value;
    bool hash_lookup = hash_table_lookup(order_id, hash_table_data.order_ids, hash_table_data.quantities, hash_table_data.prices, hash_table_data.symbol_ids, hash_table_data.sides, hash_table_data.capacity, lookup_value);
    if (!hash_lookup) {
        return false;
    }
    std::uint32_t price = lookup_value.price;
    bool hash_subtracted = hash_table_subtract(order_id, cancel_quantity, hash_table_data.order_ids, hash_table_data.quantities, hash_table_data.capacity);
    bool update_local_level = false;
    for (int i = 0; i < MAX_LEVELS_PER_LANE; ++i) {
        PriceLevel& price_level = local_levels[i];
        if (price_level.price == price) {
            price_level.quantity -= cancel_quantity;
            update_local_level = true;
            break;
        }
    }
    return hash_subtracted && update_local_level;
}



__global__ void reconstruct(CompactedMessage messages, HashTableData hash_table_data, OrderBookSnapshotData output) {
    size_t symbol_idx = blockIdx.x; // One warp per symbol so one block gets assigned exactly one symbol
    size_t symbol_offset = messages.symbol_start_offsets[symbol_idx];
    size_t symbol_count = messages.symbol_counts[symbol_idx];
    for (size_t i = symbol_offset; i < symbol_offset + symbol_count; ++i) {
        MessageType message = messages[i];

    }
}



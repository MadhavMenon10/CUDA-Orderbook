#include "reconstructor.cuh"

OrderBookSnapshot::OrderBookSnapshot(const SymbolCompactor& compacted_symbols, size_t total_message_count):  num_unique_symbols_(compacted_symbols.get_num_unique_symbols()), total_tick_count_(total_message_count), tick_write_count_(0) {
    CUDAUtils::check_cuda_error(cudaMalloc(reinterpret_cast<void**>(&ticks_), sizeof(OrderBookTick) * total_message_count), "Allocate memory for ticks_");
    CUDAUtils::check_cuda_error(cudaMalloc(reinterpret_cast<void**>(&tick_write_count_), sizeof(size_t)), "Allocate tick_write_count_");
    CUDAUtils::check_cuda_error(cudaMemset(tick_write_count_, 0, sizeof(size_t)), "Initialise tick_write_count_ with 0");
    ticks_capacity_ = total_message_count / 4;
}

OrderBookSnapshot::~OrderBookSnapshot() {
    if (ticks_ != nullptr) {
        CUDAUtils::check_cuda_error(cudaFree(ticks_), "Free ticks_");
        ticks_ = nullptr;
    }
    if (tick_write_count_ != nullptr) {
        CUDAUtils::check_cuda_error(cudaFree(tick_write_count_), "Free tick_write_count_");
        tick_write_count_ = nullptr;
    }
    num_unique_symbols_ = 0;
    ticks_capacity_ = 0;
}

__device__ size_t failure_print_count = 0;

__device__ bool reconstruct_add(CompactedMessage message_params, HashTableData hash_table_data, PriceLevel bid_levels[][MAX_LEVELS_PER_LANE], PriceLevel ask_levels[][MAX_LEVELS_PER_LANE], int idx) {
    std::uint64_t order_id = message_params.order_ids[idx];
    std::uint32_t price = message_params.prices[idx];
    std::uint32_t quantity = message_params.quantities[idx];
    std::uint16_t symbol_id = message_params.symbol_ids[idx];
    char side = message_params.sides[idx];
    bool insert_local_level = false;
    bool hash_insert = hash_table_insert(order_id, quantity, price, symbol_id, side, hash_table_data.order_ids, hash_table_data.quantities, hash_table_data.prices, hash_table_data.symbol_ids, hash_table_data.sides, hash_table_data.capacity);
    std::uint64_t lane = hash(price) % WARP_SIZE;
    PriceLevel (*levels)[MAX_LEVELS_PER_LANE] = (side == 'B') ? bid_levels : ask_levels;
    for (int i = 0; i < MAX_LEVELS_PER_LANE; ++i) {
        PriceLevel& price_level = levels[lane][i];
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

__device__ bool reconstruct_cancel(CompactedMessage message_params, HashTableData hash_table_data, PriceLevel bid_levels[][MAX_LEVELS_PER_LANE], PriceLevel ask_levels[][MAX_LEVELS_PER_LANE], int idx) {
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
    PriceLevel (*levels)[MAX_LEVELS_PER_LANE] = (lookup_value.side == 'B') ? bid_levels : ask_levels;
    std::uint64_t lane = hash(price) % WARP_SIZE;
    for (int i = 0; i < MAX_LEVELS_PER_LANE; ++i) {
        PriceLevel& price_level = levels[lane][i];
        if (price_level.price == price) {
            if (price_level.quantity >= cancel_quantity) {
                price_level.quantity -= cancel_quantity;
            } else {
                price_level.quantity = 0;
            }
            if (price_level.quantity == 0) {
                price_level.price = PRICE_EMPTY_SLOT_SENTINEL;
            }
            update_local_level = true;
            break;
        }
    }
    return hash_subtracted && update_local_level;
}

__device__ bool reconstruct_execute(CompactedMessage message_params, HashTableData hash_table_data, PriceLevel bid_levels[][MAX_LEVELS_PER_LANE], PriceLevel ask_levels[][MAX_LEVELS_PER_LANE], int idx) {
    // Same as reconstruct_cancel since from the book's perspective, an existing order goes down by same amount
    // Since these are two different market events, they are kept separated
    std::uint64_t order_id = message_params.order_ids[idx];
    std::uint32_t execute_quantity = message_params.quantities[idx];
    HashTableEntry lookup_value;
    bool hash_lookup = hash_table_lookup(order_id, hash_table_data.order_ids, hash_table_data.quantities, hash_table_data.prices, hash_table_data.symbol_ids, hash_table_data.sides, hash_table_data.capacity, lookup_value);
    if (!hash_lookup) {
        return false;
    }
    std::uint32_t price = lookup_value.price;
    bool hash_subtracted = hash_table_subtract(order_id, execute_quantity, hash_table_data.order_ids, hash_table_data.quantities, hash_table_data.capacity);
    bool update_local_level = false;
    PriceLevel (*levels)[MAX_LEVELS_PER_LANE] = (lookup_value.side == 'B') ? bid_levels : ask_levels;
    std::uint64_t lane = hash(price) % WARP_SIZE;
    for (int i = 0; i < MAX_LEVELS_PER_LANE; ++i) {
        PriceLevel& price_level = levels[lane][i];
        if (price_level.price == price) {
            if (price_level.quantity >= execute_quantity) {
                price_level.quantity -= execute_quantity;
            } else {
                price_level.quantity = 0;
            }
            if (price_level.quantity == 0) {
                price_level.price = PRICE_EMPTY_SLOT_SENTINEL;
            }
            update_local_level = true;
            break;
        }
    }
    return hash_subtracted && update_local_level;
}


__device__ bool reconstruct_execute_with_price(CompactedMessage message_params, HashTableData hash_table_data,  PriceLevel bid_levels[][MAX_LEVELS_PER_LANE], PriceLevel ask_levels[][MAX_LEVELS_PER_LANE], int idx) {
    std::uint64_t order_id = message_params.order_ids[idx];
    std::uint32_t execute_quantity = message_params.quantities[idx];
    HashTableEntry lookup_value;
    bool hash_lookup = hash_table_lookup(order_id, hash_table_data.order_ids, hash_table_data.quantities, hash_table_data.prices, hash_table_data.symbol_ids, hash_table_data.sides, hash_table_data.capacity, lookup_value);
    if (!hash_lookup) {
        return false;
    }
    std::uint32_t price = lookup_value.price; 
    /*
     * A Cancel/Execute message never carries the order's price — it only gives an order ID
     * and an amount. We look the order up in the hash table to find where its liquidity
     * actually rests (e.g. an order quoted at $50.00 stays a $50.00-level update even if
     * a later execution prints at $50.15 due to price improvement) — the book's price
     * levels must shrink at the price where the liquidity was resting, not at whatever
     * price a specific event happens to report.
    */

    bool hash_subtracted = hash_table_subtract(order_id, execute_quantity, hash_table_data.order_ids, hash_table_data.quantities, hash_table_data.capacity);
    bool update_local_level = false;
    PriceLevel (*levels)[MAX_LEVELS_PER_LANE] = (lookup_value.side == 'B') ? bid_levels : ask_levels;
    std::uint64_t lane = hash(price) % WARP_SIZE;
    for (int i = 0; i < MAX_LEVELS_PER_LANE; ++i) {
        PriceLevel& price_level = levels[lane][i];
        if (price_level.price == price) {
            if (price_level.quantity >= execute_quantity) {
                price_level.quantity -= execute_quantity;
            } else {
                price_level.quantity = 0;
            }
            if (price_level.quantity == 0) {
                price_level.price = PRICE_EMPTY_SLOT_SENTINEL;
            }
            update_local_level = true;
            break;
        }
    }
    return hash_subtracted && update_local_level;
}


__device__ bool reconstruct_delete(CompactedMessage message_params, HashTableData hash_table_data, PriceLevel bid_levels[][MAX_LEVELS_PER_LANE], PriceLevel ask_levels[][MAX_LEVELS_PER_LANE], int idx) {
    std::uint64_t order_id = message_params.order_ids[idx];
    HashTableEntry lookup_value;
    bool hash_lookup = hash_table_lookup(order_id, hash_table_data.order_ids, hash_table_data.quantities, hash_table_data.prices, hash_table_data.symbol_ids, hash_table_data.sides, hash_table_data.capacity, lookup_value);
    if (!hash_lookup) {
        return false;
    }
    std::uint32_t price = lookup_value.price;
    bool hash_deleted = hash_table_delete(order_id, hash_table_data.order_ids, hash_table_data.capacity);
    bool lookup_deleted = false;
    PriceLevel (*levels)[MAX_LEVELS_PER_LANE] = (lookup_value.side == 'B') ? bid_levels : ask_levels;
    std::uint64_t lane = hash(price) % WARP_SIZE;
    for (int i = 0; i < MAX_LEVELS_PER_LANE; ++i) {
        PriceLevel& price_level = levels[lane][i];
        if (price_level.price == price) {
            if (price_level.quantity >= lookup_value.quantity) {
                price_level.quantity -= lookup_value.quantity;
            } else {
                price_level.quantity = 0;
            }
            if (price_level.quantity == 0) {
                price_level.price = PRICE_EMPTY_SLOT_SENTINEL;
            }
            lookup_deleted = true;
            break;
        }
    }
    return hash_deleted && lookup_deleted;
}


__device__ bool reconstruct_replace(CompactedMessage message_params, HashTableData hash_table_data, PriceLevel bid_levels[][MAX_LEVELS_PER_LANE], PriceLevel ask_levels[][MAX_LEVELS_PER_LANE], int idx) {
    std::uint64_t new_order_id = message_params.order_ids[idx];
    std::uint64_t old_order_id = message_params.old_order_ids[idx];
    std::uint32_t new_price = message_params.prices[idx];
    std::uint32_t new_quantity = message_params.quantities[idx];
    HashTableEntry old_lookup_value; 
    bool old_lookup = hash_table_lookup(old_order_id, hash_table_data.order_ids, hash_table_data.quantities, hash_table_data.prices, hash_table_data.symbol_ids, hash_table_data.sides, hash_table_data.capacity, old_lookup_value);
    if (!old_lookup) {
        return false;
    }
    bool old_entry_deleted = hash_table_delete(old_order_id, hash_table_data.order_ids, hash_table_data.capacity);
    if (!old_entry_deleted) {
        return false;
    }
    PriceLevel (*levels)[MAX_LEVELS_PER_LANE] = (old_lookup_value.side == 'B') ? bid_levels : ask_levels;
    bool new_entry_inserted = hash_table_insert(new_order_id, new_quantity, new_price, old_lookup_value.symbol_id, old_lookup_value.side, hash_table_data.order_ids, hash_table_data.quantities, hash_table_data.prices, hash_table_data.symbol_ids, hash_table_data.sides, hash_table_data.capacity); // A replace order never crosses symbols (one order book per instrument) or side so we used the old values
    std::uint64_t old_lane = hash(old_lookup_value.price) % WARP_SIZE;
    for (int i = 0; i < MAX_LEVELS_PER_LANE; ++i) {
        PriceLevel& price_level = levels[old_lane][i];
        if (price_level.price == old_lookup_value.price) {
            if (price_level.quantity >= old_lookup_value.quantity) {
                price_level.quantity -= old_lookup_value.quantity;
            } else {
                price_level.quantity = 0;
            }
            if (price_level.quantity == 0) {
                price_level.price = PRICE_EMPTY_SLOT_SENTINEL;
            }
            break;
        }
    }
    bool insert_local_level = false;
    std::uint64_t new_lane = hash(new_price) % WARP_SIZE;
    for (int i = 0; i < MAX_LEVELS_PER_LANE; ++i) {
        PriceLevel& price_level = levels[new_lane][i];
        if (price_level.price == PRICE_EMPTY_SLOT_SENTINEL) {
            price_level.price = new_price;
            price_level.quantity = new_quantity;
            insert_local_level = true;
            break;
        } else if (price_level.price == new_price) {
            price_level.quantity += new_quantity;
            insert_local_level = true;
            break;
        }
    }
    return new_entry_inserted && insert_local_level;
}

__device__ PriceLevel find_best_level(PriceLevel levels[][MAX_LEVELS_PER_LANE], bool find_max, const PriceLevel* top_5) {
    std::uint32_t best_price = PRICE_EMPTY_SLOT_SENTINEL;
    std::uint32_t best_quantity = 0;
    for (int i = 0; i < MAX_LEVELS_PER_LANE; ++i) {
        std::uint32_t price = levels[threadIdx.x][i].price;
        std::uint32_t quantity = levels[threadIdx.x][i].quantity;
        bool already_excluded = false;
        for (int j = 0; j < 5; ++j) {
            if (top_5[j].price == price) {
                already_excluded = true;
                break;
            }
        }
        if ((price != PRICE_EMPTY_SLOT_SENTINEL) && ((best_price == PRICE_EMPTY_SLOT_SENTINEL) || (find_max && price > best_price) || (!find_max && price < best_price)) && !already_excluded) {
            best_price = price;
            best_quantity = quantity;
        }
    }
    // To find the best price and quantity across the entire warp
    for (int offset = 16; offset != 0; offset /= 2) {
        std::uint32_t other_price = __shfl_down_sync(0xFFFFFFFF, best_price, offset);
        std::uint32_t other_quantity = __shfl_down_sync(0xFFFFFFFF, best_quantity, offset);
        if ((other_price != PRICE_EMPTY_SLOT_SENTINEL) && ((best_price == PRICE_EMPTY_SLOT_SENTINEL) || (find_max && other_price > best_price) || (!find_max && other_price < best_price))) {
            best_price = other_price;
            best_quantity = other_quantity;
        }
    }
    return PriceLevel{best_price, best_quantity};
}


__global__ void reconstruct(CompactedMessage messages, HashTableData hash_table_data, OrderBookSnapshotData output) {
    __shared__ PriceLevel bid_levels[WARP_SIZE][MAX_LEVELS_PER_LANE];
    __shared__ PriceLevel ask_levels[WARP_SIZE][MAX_LEVELS_PER_LANE];
    __shared__ PriceLevel bid_top5[5];
    __shared__ PriceLevel ask_top5[5];
    __shared__ PriceLevel last_bid_top5[5];
    __shared__ PriceLevel last_ask_top5[5];
    for (int i = 0; i < MAX_LEVELS_PER_LANE; ++i) {
        bid_levels[threadIdx.x][i].price = PRICE_EMPTY_SLOT_SENTINEL;
        bid_levels[threadIdx.x][i].quantity = 0;
        ask_levels[threadIdx.x][i].price = PRICE_EMPTY_SLOT_SENTINEL;
        ask_levels[threadIdx.x][i].quantity = 0;
    }
    __syncthreads(); // So all threads finishing writing the sentinel value before price_levels is read from
    size_t symbol_idx = blockIdx.x; // One warp per symbol so one block gets assigned exactly one symbol
    size_t symbol_offset = messages.symbol_start_offsets[symbol_idx];
    size_t symbol_count = messages.symbol_counts[symbol_idx];
    for (size_t i = symbol_offset; i < symbol_offset + symbol_count; ++i) {
        MessageType message = messages.order_types[i];
        bool message_carried = false;
        if (threadIdx.x == 0) {
            switch (message) {
                case MessageType::Add:
                    message_carried = reconstruct_add(messages, hash_table_data, bid_levels, ask_levels, i);
                    break;
                case MessageType::Cancel:
                    message_carried = reconstruct_cancel(messages, hash_table_data, bid_levels, ask_levels, i);
                    break;
                case MessageType::Execute:
                    message_carried = reconstruct_execute(messages, hash_table_data, bid_levels, ask_levels, i);
                    break;
                case MessageType::ExecuteWithPrice:
                    message_carried = reconstruct_execute_with_price(messages, hash_table_data, bid_levels, ask_levels, i);
                    break;
                case MessageType::Delete:
                    message_carried = reconstruct_delete(messages, hash_table_data, bid_levels, ask_levels, i);
                    break;
                case MessageType::Replace:
                    message_carried = reconstruct_replace(messages, hash_table_data, bid_levels, ask_levels, i);
                    break;
                default:
                    break;
            }
            if (!message_carried) {
                size_t print_index = atomicAdd(reinterpret_cast<unsigned long long*>(&failure_print_count), static_cast<unsigned long long>(1));
                if (print_index < 100) {  // only the first 100 failures across the whole run
                    printf("Reconstruction failed at message %llu (type %d) in symbol block %d\n", static_cast<unsigned long long>(i), static_cast<int>(message), blockIdx.x);
                }
            }
        } 
        /*
         * bid_top5 / ask_top5 hold the current message's top-5 price levels, discovered
         * fresh via a warp-wide reduction over bid_levels/ask_levels. That reduction has
         * to start from a clean slate every message since the last message's top-5 values are
         * stale the moment this message's dispatch (the switch above) has potentially
         * changed the book, and leftover values from a prior message must never leak
         * into this message's tick.
         */
        if (threadIdx.x < 5) {
            bid_top5[threadIdx.x].price = PRICE_EMPTY_SLOT_SENTINEL;
            bid_top5[threadIdx.x].quantity = 0;
            ask_top5[threadIdx.x].price = PRICE_EMPTY_SLOT_SENTINEL;
            ask_top5[threadIdx.x].quantity = 0;
        }
        __syncthreads(); 
        bool find_max = true;
        for (int round = 0; round < 5; ++round) {
            PriceLevel best_bid = find_best_level(bid_levels, find_max, bid_top5);
            if (threadIdx.x == 0) { // All threads will have the best price level so we only use the first thread to avoid redundancy
                bid_top5[round] = best_bid;
            }
            __syncthreads();
        }
        find_max = false;
        for (int round = 0; round < 5; ++round) {
            PriceLevel best_ask = find_best_level(ask_levels, find_max, ask_top5);
            if (threadIdx.x == 0) { 
                ask_top5[round] = best_ask;
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) { // We don't need every thread to create an output_tick
            bool changed = false;
            for (int j = 0; j < 5; ++j) {
                if (bid_top5[j].price != last_bid_top5[j].price || bid_top5[j].quantity != last_bid_top5[j].quantity) {
                    changed = true;
                    break;
                }
                if (ask_top5[j].price != last_ask_top5[j].price || ask_top5[j].quantity != last_ask_top5[j].quantity) {
                    changed = true;
                    break;
                }
            }
            if (changed) {
                size_t slot = atomicAdd(reinterpret_cast<unsigned long long*>(output.tick_write_count), static_cast<unsigned long long>(1)); // Same as atomicCAS
                if (slot < output.ticks_capacity) {
                    OrderBookTick output_tick;
                    output_tick.timestamp = messages.timestamps[i];
                    output_tick.symbol_id = messages.symbol_ids[i];
                    for (int j = 0; j < 5; ++j) {
                        output_tick.bids[j] = bid_top5[j];
                        output_tick.asks[j] = ask_top5[j];
                        last_bid_top5[j] = bid_top5[j];
                        last_ask_top5[j] = ask_top5[j];
                    }
                    output.ticks[slot] = output_tick;
                } else {
                    printf("Tick capacity exceeded for symbol block %d, dropping tick at message %llu\n", blockIdx.x, static_cast<unsigned long long>(i)); 
                }
            }
        }   
    }
}


void launch_reconstruction_kernel(const SymbolCompactor& compacted_symbols, OrderBookSnapshot& order_book_snapshot, GPUHashTable& hash_table) {
    CompactedMessage compacted_message;
    compacted_message.order_ids = compacted_symbols.get_order_ids();
    compacted_message.old_order_ids = compacted_symbols.get_old_order_ids();
    compacted_message.timestamps = compacted_symbols.get_timestamps();
    compacted_message.prices = compacted_symbols.get_prices();
    compacted_message.quantities = compacted_symbols.get_quantities();
    compacted_message.symbol_ids = compacted_symbols.get_symbol_ids();
    compacted_message.order_types = compacted_symbols.get_order_types();
    compacted_message.sides = compacted_symbols.get_sides();
    compacted_message.symbol_start_offsets = compacted_symbols.get_symbol_start_offsets();
    compacted_message.symbol_counts = compacted_symbols.get_symbol_counts();
    HashTableData hash_table_data;
    hash_table_data.order_ids = hash_table.get_order_ids();
    hash_table_data.quantities = hash_table.get_quantities();
    hash_table_data.prices = hash_table.get_prices();
    hash_table_data.symbol_ids = hash_table.get_symbol_ids();
    hash_table_data.sides = hash_table.get_sides();
    hash_table_data.capacity = hash_table.capacity();
    OrderBookSnapshotData order_book_snapshot_data;
    order_book_snapshot_data.ticks = order_book_snapshot.get_ticks();
    order_book_snapshot_data.tick_write_count = order_book_snapshot.get_tick_write_count();
    order_book_snapshot_data.ticks_capacity = order_book_snapshot.get_ticks_capacity();
    reconstruct<<<compacted_symbols.get_num_unique_symbols(), WARP_SIZE>>>(compacted_message, hash_table_data, order_book_snapshot_data);
}

#include "hash_table.cuh"

GPUHashTable::GPUHashTable(size_t max_active_orders) {
    capacity_ = max_active_orders * 2;
    CUDAUtils::check_cuda_error(cudaMalloc(reinterpret_cast<void**>(&order_ids_), sizeof(std::uint64_t) * capacity_), "GPU hash table memory allocation for order_ids_");
    CUDAUtils::check_cuda_error(cudaMemset(order_ids_, 0xFF, capacity_ * sizeof(std::uint64_t)), "Fill every byte of order_ids_ with UINT64_MAX");
    CUDAUtils::check_cuda_error(cudaMalloc(reinterpret_cast<void**>(&quantities_), sizeof(std::uint32_t) * capacity_), "GPU hash table memory allocation for quantities_");
    CUDAUtils::check_cuda_error(cudaMalloc(reinterpret_cast<void**>(&prices_), sizeof(std::uint32_t) * capacity_), "GPU Hash table memory allocation for prices_");
    CUDAUtils::check_cuda_error(cudaMalloc(reinterpret_cast<void**>(&symbol_ids_), sizeof(std::uint16_t) * capacity_), "GPU Hash table memory allocation for symbol_ids_");
    CUDAUtils::check_cuda_error(cudaMalloc(reinterpret_cast<void**>(&sides_), sizeof(char) * capacity_), "GPU Hash table memory allocation for sides_");
}


GPUHashTable::~GPUHashTable() {
    if (order_ids_ != nullptr) {
        cudaFree(order_ids_);
    }
    if (quantities_ != nullptr) {
        cudaFree(quantities_);
    }
    if (prices_ != nullptr) {
        cudaFree(prices_);
    }
    if (symbol_ids_ != nullptr) {
        cudaFree(symbol_ids_);
    }
    if (sides_ != nullptr) {
        cudaFree(sides_);
    }
    capacity_ = 0;
}

__device__ std::uint64_t hash(std::uint64_t order_id) {
    // Used part of the splitmix64 algorithm as a hashing function
    std::uint64_t hashed_order_id = (order_id ^ (order_id >> 30)) * 0xbf58476d1ce4e5b9;  /* xor the variable with the variable right bit shifted 30 then multiply by a constant */
    hashed_order_id = (hashed_order_id ^ (hashed_order_id >> 27)) * 0x94d049bb133111eb;  /* xor the variable with the variable right bit shifted 27 then multiply by a constant */
    return hashed_order_id ^ (hashed_order_id >> 31);                             /* return the variable xored with itself right bit shifted 31 */
}

__device__ bool hash_table_insert(std::uint64_t order_id, std::uint32_t order_quantity, std::uint32_t order_price, std::uint16_t order_symbol_id, char order_side, std::uint64_t* order_ids, std::uint32_t* quantities, std::uint32_t* prices, std::uint16_t* symbol_ids, char* sides, size_t hash_table_capacity) {
    std::uint64_t hash_index = hash(order_id) % hash_table_capacity;
    size_t slots_checked = 0;
    bool written = false;
    while (!written) {
        if (slots_checked >= hash_table_capacity) {
            break;
        }
        if (atomicCAS(&order_ids[hash_index], EMPTY_SLOT_SENTINEL, order_id) == EMPTY_SLOT_SENTINEL) {
            insert_into_hash_index(order_quantity, order_price, order_symbol_id, order_side, quantities, prices, symbol_ids, sides, hash_index);
            written = true;
        } else if (atomicCAS(&order_ids[hash_index], TOMBSTONE_SENTINEL, order_id) == TOMBSTONE_SENTINEL) {
            insert_into_hash_index(order_quantity, order_price, order_symbol_id, order_side, quantities, prices, symbol_ids, sides, hash_index);
            written = true;
        } else {
            hash_index = (hash_index + 1) % hash_table_capacity;
            ++slots_checked;
        }
    }
    return written;
}

__device__ bool hash_table_lookup(std::uint64_t order_id, const std::uint64_t* order_ids, const std::uint32_t* quantities, const std::uint32_t* prices, const std::uint16_t* symbol_ids, const char* sides, size_t hash_table_capacity, HashTableEntry& lookup_value) {
    std::uint64_t hash_index = hash(order_id) % hash_table_capacity;
    size_t slots_checked = 0; // Hash table not empty and desired key is not present 
    bool found = false;
    while (!found) {
        if (slots_checked >= hash_table_capacity) {
            break;
        }
        if (order_ids[hash_index] == order_id) {
            lookup_value.quantity = quantities[hash_index];
            lookup_value.price = prices[hash_index];
            lookup_value.symbol_id = symbol_ids[hash_index];
            lookup_value.side = sides[hash_index];
            found = true;
            break;
        } else if (order_ids[hash_index] == EMPTY_SLOT_SENTINEL) {
            break;
        } else {
            hash_index = (hash_index + 1) % hash_table_capacity;
            ++slots_checked;
        }
    }
    return found;
}

__device__ bool hash_table_delete(std::uint64_t order_id, std::uint64_t* order_ids, size_t hash_table_capacity) {
    std::uint64_t hash_index = hash(order_id) % hash_table_capacity;
    size_t slots_checked = 0;
    bool deleted = false;
    while (!deleted) {
        if (slots_checked >= hash_table_capacity) {
            break;
        }
        if (order_ids[hash_index] == order_id) {
            if (atomicCAS(&order_ids[hash_index], order_id, TOMBSTONE_SENTINEL) == order_id) {
                deleted = true;
            }
            break;
        } else if (order_ids[hash_index] == EMPTY_SLOT_SENTINEL) {
            break;
        } else {
            hash_index = (hash_index + 1) % hash_table_capacity;
            ++slots_checked;
        }
    }
    return deleted;
}
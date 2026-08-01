#pragma once
#include "types.hpp"
#include <cstdint>
#include <cstddef>
#include <map>
#include <unordered_map>
#include <vector>

/*
 * Single-threaded CPU baseline for the reconstruction and backtesting kernels, so the GPU
 * pipeline has something to be measured against. This is deliberately not a line-for-line port
 * of the kernels. The GPU shards price levels across 32 lanes by hash(price) and caps each lane
 * at MAX_LEVELS_PER_LANE, and it keeps orders in an open-addressed table sized up front because
 * shared and device memory cannot grow once a kernel launches. A CPU has none of those
 * constraints, so handing it those data structures would measure the hardware against an
 * algorithm that was only ever shaped by the hardware. The baseline instead does what a CPU
 * implementation would actually do: an ordered map of price levels per symbol and a
 * std::unordered_map keyed by order ID.
 *
 * The consequence is that the two tick streams will not match exactly. The GPU drops levels once
 * a lane fills up, and the baseline never does, so the baseline is the more correct of the two
 * and its failure count should come out at or below the GPU's.
 *
 * Nothing in here touches CUDA, so the baseline builds and runs on a machine with no GPU in it.
*/

namespace CPUBaseline {
    constexpr std::uint32_t EMPTY_PRICE_SENTINEL = UINT32_MAX; // Same value as PRICE_EMPTY_SLOT_SENTINEL in reconstructor.cuh so the two tick streams stay directly comparable

    struct PriceLevel {
        std::uint32_t price;
        std::uint32_t quantity;
    };

    struct OrderBookTick {
        PriceLevel bids[5];
        PriceLevel asks[5];
        std::uint64_t timestamp;
        std::uint16_t symbol_id;
    };

    struct OrderEntry { // The CPU equivalent of a HashTableEntry: what has to be remembered about an order because a Cancel/Execute/Delete message only carries its ID
        std::uint32_t quantity;
        std::uint32_t price;
        std::uint16_t symbol_id;
        char side;
    };

    struct Book {
        Book();
        std::map<std::uint32_t, std::uint32_t> bids; // Ascending, so the best bid is the last entry
        std::map<std::uint32_t, std::uint32_t> asks; // Ascending, so the best ask is the first entry
        PriceLevel last_bids[5];
        PriceLevel last_asks[5];
    };

    class Reconstructor {
        public:
            Reconstructor(size_t max_active_orders);
            ~Reconstructor() = default;
            Reconstructor(const Reconstructor&) = delete;
            Reconstructor& operator=(const Reconstructor&) = delete;
            Reconstructor(Reconstructor&&) = delete;
            Reconstructor& operator=(Reconstructor&&) = delete;
            void run(const std::vector<DecodedOrder>& messages);
            inline std::unordered_map<std::uint16_t, std::vector<OrderBookTick>>& get_ticks() {return ticks_;}; // Non-const so the ticks can be moved out rather than copied, since a full day's stream runs to several GB
            inline size_t get_tick_count() const {return tick_count_;};
            inline size_t get_failure_count() const {return failure_count_;};
        private:
            bool reconstruct_add(const DecodedOrder& message, Book& book);
            bool reconstruct_cancel(const DecodedOrder& message, Book& book);
            bool reconstruct_execute(const DecodedOrder& message, Book& book);
            bool reconstruct_execute_with_price(const DecodedOrder& message, Book& book);
            bool reconstruct_delete(const DecodedOrder& message, Book& book);
            bool reconstruct_replace(const DecodedOrder& message, Book& book);
            void write_tick(const DecodedOrder& message, Book& book);
            std::unordered_map<std::uint64_t, OrderEntry> orders_; // Stands in for GPUHashTable, and like it is shared across every symbol and never cleared
            std::unordered_map<std::uint16_t, Book> books_;
            std::unordered_map<std::uint16_t, std::vector<OrderBookTick>> ticks_;
            size_t tick_count_;
            size_t failure_count_;
    };

    class Backtester {
        public:
            Backtester(const std::vector<StrategyConfig>& configs);
            ~Backtester() = default;
            Backtester(const Backtester&) = delete;
            Backtester& operator=(const Backtester&) = delete;
            Backtester(Backtester&&) = delete;
            Backtester& operator=(Backtester&&) = delete;
            void run(const std::vector<std::vector<OrderBookTick>>& symbol_ticks);
            inline const std::vector<std::int32_t>& get_pnl_results() const {return pnl_results_;};
        private:
            std::vector<StrategyConfig> configs_;
            std::vector<std::int32_t> pnl_results_;
    };

    bool remove_liquidity(std::map<std::uint32_t, std::uint32_t>& levels, std::uint32_t price, std::uint32_t quantity);

    void find_top_5(const std::map<std::uint32_t, std::uint32_t>& levels, bool find_max, PriceLevel* top_5);

    std::vector<std::vector<OrderBookTick>> group_ticks_by_symbol(std::unordered_map<std::uint16_t, std::vector<OrderBookTick>>& ticks);

    std::vector<StrategyConfig> generate_configs(size_t num_thresholds = 500, size_t num_lookbacks = 10, size_t num_position_sizes = 2); // Mirrors the generator in backtester.cuh, duplicated so the baseline does not have to link against a CUDA translation unit
}

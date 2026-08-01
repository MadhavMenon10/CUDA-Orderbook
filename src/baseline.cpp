#include "baseline.hpp"
#include <algorithm>
#include <iostream>

CPUBaseline::Book::Book() {
    for (int i = 0; i < 5; ++i) {
        last_bids[i] = PriceLevel{EMPTY_PRICE_SENTINEL, 0};
        last_asks[i] = PriceLevel{EMPTY_PRICE_SENTINEL, 0};
    }
}

CPUBaseline::Reconstructor::Reconstructor(size_t max_active_orders) : tick_count_(0), failure_count_(0) {
    orders_.reserve(max_active_orders); // GPUHashTable is sized up front from the same estimate, so the baseline is not made to pay for rehashing inside the timed loop
}

bool CPUBaseline::remove_liquidity(std::map<std::uint32_t, std::uint32_t>& levels, std::uint32_t price, std::uint32_t quantity) {
    auto level = levels.find(price);
    if (level == levels.end()) {
        return false;
    }
    if (level->second > quantity) {
        level->second -= quantity;
    } else {
        levels.erase(level); // The kernel clamps at zero and then writes the empty sentinel back over the slot, which is an erase here
    }
    return true;
}

void CPUBaseline::find_top_5(const std::map<std::uint32_t, std::uint32_t>& levels, bool find_max, PriceLevel* top_5) {
    for (int i = 0; i < 5; ++i) {
        top_5[i] = PriceLevel{EMPTY_PRICE_SENTINEL, 0};
    }
    int filled = 0;
    if (find_max) { // Bids, so walk the map backwards from the highest price
        for (auto level = levels.rbegin(); level != levels.rend() && filled < 5; ++level) {
            top_5[filled] = PriceLevel{level->first, level->second};
            ++filled;
        }
    } else {
        for (auto level = levels.begin(); level != levels.end() && filled < 5; ++level) {
            top_5[filled] = PriceLevel{level->first, level->second};
            ++filled;
        }
    }
}

bool CPUBaseline::Reconstructor::reconstruct_add(const DecodedOrder& message, Book& book) {
    OrderEntry entry;
    entry.quantity = message.quantity;
    entry.price = message.price;
    entry.symbol_id = message.symbol_id;
    entry.side = message.side;
    bool inserted = orders_.emplace(message.order_id, entry).second; // False when the feed hands us an order ID that is already live. The kernel would instead park the duplicate in a second slot and call it a success, so this is one place the baseline is the stricter of the two
    std::map<std::uint32_t, std::uint32_t>& levels = (message.side == 'B') ? book.bids : book.asks;
    levels[message.price] += message.quantity;
    return inserted;
}

bool CPUBaseline::Reconstructor::reconstruct_cancel(const DecodedOrder& message, Book& book) {
    auto order = orders_.find(message.order_id);
    if (order == orders_.end()) {
        return false;
    }
    OrderEntry& entry = order->second;
    std::uint32_t cancel_quantity = message.quantity;
    entry.quantity = (entry.quantity >= cancel_quantity) ? entry.quantity - cancel_quantity : 0;
    std::map<std::uint32_t, std::uint32_t>& levels = (entry.side == 'B') ? book.bids : book.asks;
    return remove_liquidity(levels, entry.price, cancel_quantity); // A Cancel never carries a price, so the level to shrink is the one the order was resting at
}

bool CPUBaseline::Reconstructor::reconstruct_execute(const DecodedOrder& message, Book& book) {
    // Same as reconstruct_cancel since from the book's perspective, an existing order goes down by same amount
    // Kept separate for the same reason the kernel keeps them separate: they are two different market events
    auto order = orders_.find(message.order_id);
    if (order == orders_.end()) {
        return false;
    }
    OrderEntry& entry = order->second;
    std::uint32_t execute_quantity = message.quantity;
    entry.quantity = (entry.quantity >= execute_quantity) ? entry.quantity - execute_quantity : 0;
    std::map<std::uint32_t, std::uint32_t>& levels = (entry.side == 'B') ? book.bids : book.asks;
    return remove_liquidity(levels, entry.price, execute_quantity);
}

bool CPUBaseline::Reconstructor::reconstruct_execute_with_price(const DecodedOrder& message, Book& book) {
    auto order = orders_.find(message.order_id);
    if (order == orders_.end()) {
        return false;
    }
    OrderEntry& entry = order->second;
    std::uint32_t execute_quantity = message.quantity;
    entry.quantity = (entry.quantity >= execute_quantity) ? entry.quantity - execute_quantity : 0;
    std::map<std::uint32_t, std::uint32_t>& levels = (entry.side == 'B') ? book.bids : book.asks;
    return remove_liquidity(levels, entry.price, execute_quantity); // entry.price and not message.price: the liquidity has to shrink where it was resting, not at whatever price the execution happened to print at
}

bool CPUBaseline::Reconstructor::reconstruct_delete(const DecodedOrder& message, Book& book) {
    auto order = orders_.find(message.order_id);
    if (order == orders_.end()) {
        return false;
    }
    OrderEntry entry = order->second;
    orders_.erase(order);
    std::map<std::uint32_t, std::uint32_t>& levels = (entry.side == 'B') ? book.bids : book.asks;
    return remove_liquidity(levels, entry.price, entry.quantity); // A Delete pulls whatever is left of the order, so the whole remaining quantity comes off the level
}

bool CPUBaseline::Reconstructor::reconstruct_replace(const DecodedOrder& message, Book& book) {
    auto old_order = orders_.find(message.old_order_id);
    if (old_order == orders_.end()) {
        return false;
    }
    OrderEntry old_entry = old_order->second;
    orders_.erase(old_order);
    OrderEntry new_entry;
    new_entry.quantity = message.quantity;
    new_entry.price = message.price;
    new_entry.symbol_id = old_entry.symbol_id;
    new_entry.side = old_entry.side; // A replace never crosses symbols (one order book per instrument) or side so we reuse the old values
    bool inserted = orders_.emplace(message.order_id, new_entry).second;
    std::map<std::uint32_t, std::uint32_t>& levels = (old_entry.side == 'B') ? book.bids : book.asks;
    remove_liquidity(levels, old_entry.price, old_entry.quantity);
    levels[message.price] += message.quantity;
    return inserted;
}

void CPUBaseline::Reconstructor::write_tick(const DecodedOrder& message, Book& book) {
    PriceLevel bid_top5[5];
    PriceLevel ask_top5[5];
    find_top_5(book.bids, true, bid_top5);
    find_top_5(book.asks, false, ask_top5);
    bool changed = false;
    for (int i = 0; i < 5; ++i) {
        if (bid_top5[i].price != book.last_bids[i].price || bid_top5[i].quantity != book.last_bids[i].quantity) {
            changed = true;
            break;
        }
        if (ask_top5[i].price != book.last_asks[i].price || ask_top5[i].quantity != book.last_asks[i].quantity) {
            changed = true;
            break;
        }
    }
    if (!changed) {
        return;
    }
    OrderBookTick tick;
    tick.timestamp = message.timestamp;
    tick.symbol_id = message.symbol_id;
    for (int i = 0; i < 5; ++i) {
        tick.bids[i] = bid_top5[i];
        tick.asks[i] = ask_top5[i];
        book.last_bids[i] = bid_top5[i];
        book.last_asks[i] = ask_top5[i];
    }
    ticks_[message.symbol_id].push_back(tick); // Appending per symbol as we go means there is no CPU counterpart to TickCompactor: the stream comes out already grouped and in order
    ++tick_count_;
}

/*
 * The kernel gets its per-symbol ordering from SymbolCompactor and then walks one symbol per warp.
 * There is nothing to compact here. A single thread walking the file in arrival order already
 * respects causality within every symbol at once, so the baseline just keeps a book per symbol
 * and looks the right one up per message.
*/
void CPUBaseline::Reconstructor::run(const std::vector<DecodedOrder>& messages) {
    for (const DecodedOrder& message : messages) {
        Book& book = books_[message.symbol_id];
        bool message_carried = false;
        switch (message.order_type) {
            case MessageType::Add:
                message_carried = reconstruct_add(message, book);
                break;
            case MessageType::Cancel:
                message_carried = reconstruct_cancel(message, book);
                break;
            case MessageType::Execute:
                message_carried = reconstruct_execute(message, book);
                break;
            case MessageType::ExecuteWithPrice:
                message_carried = reconstruct_execute_with_price(message, book);
                break;
            case MessageType::Delete:
                message_carried = reconstruct_delete(message, book);
                break;
            case MessageType::Replace:
                message_carried = reconstruct_replace(message, book);
                break;
            default:
                break;
        }
        if (!message_carried) {
            ++failure_count_;
            if (failure_count_ <= 100) { // Only the first 100 failures across the whole run, same cap the kernel uses
                std::cout << "Reconstruction failed for symbol " << message.symbol_id << " (type " << static_cast<int>(message.order_type) << ")\n";
            }
        }
        write_tick(message, book); // Runs whether or not the message carried, since the kernel also recomputes its top-5 every message
    }
}

std::vector<std::vector<CPUBaseline::OrderBookTick>> CPUBaseline::group_ticks_by_symbol(std::unordered_map<std::uint16_t, std::vector<OrderBookTick>>& ticks) {
    std::vector<std::uint16_t> symbol_ids;
    symbol_ids.reserve(ticks.size());
    for (const auto& symbol : ticks) {
        symbol_ids.push_back(symbol.first);
    }
    std::sort(symbol_ids.begin(), symbol_ids.end()); // Sorted so the walk order matches the kernel's, which reads symbols in sorted order out of TickCompactor
    std::vector<std::vector<OrderBookTick>> symbol_ticks;
    symbol_ticks.reserve(symbol_ids.size());
    for (std::uint16_t symbol_id : symbol_ids) {
        symbol_ticks.push_back(std::move(ticks[symbol_id]));
    }
    return symbol_ticks;
}

std::vector<StrategyConfig> CPUBaseline::generate_configs(size_t num_thresholds, size_t num_lookbacks, size_t num_position_sizes) {
    size_t total_configs = num_thresholds * num_lookbacks * num_position_sizes;
    std::vector<StrategyConfig> configs;
    configs.reserve(total_configs);
    for (size_t i = 0; i < num_thresholds; ++i) {
        for (size_t j = 0; j < num_lookbacks; ++j) {
            for (size_t k = 0; k < num_position_sizes; ++k) {
                StrategyConfig config;
                config.threshold = static_cast<std::uint16_t>(i + 1);
                config.lookback_window = (j + 1) * 5;
                config.position_size = static_cast<std::uint32_t>(k + 1) * 100;
                configs.push_back(config);
            }
        }
    }
    return configs;
}

CPUBaseline::Backtester::Backtester(const std::vector<StrategyConfig>& configs) : configs_(configs), pnl_results_(configs.size(), 0) {}

/*
 * One config per iteration where the kernel gives one config per block, and the arithmetic is
 * kept identical to run_strategy so the PnL tables can be compared entry for entry. The one
 * deliberate difference is the mid_price guard: a tick with an empty side carries the price
 * sentinel, and the resulting sum can wrap to zero. The GPU shrugs that off, whereas an integer
 * divide by zero on the CPU is a hardware trap, so those ticks are skipped instead.
*/
void CPUBaseline::Backtester::run(const std::vector<std::vector<OrderBookTick>>& symbol_ticks) {
    for (size_t i = 0; i < configs_.size(); ++i) {
        const StrategyConfig& config = configs_[i];
        std::int32_t pnl = 0;
        for (const std::vector<OrderBookTick>& ticks : symbol_ticks) {
            bool holding_shares = false;
            std::uint32_t price = 0;
            for (const OrderBookTick& tick : ticks) {
                std::uint32_t spread = tick.asks[0].price - tick.bids[0].price;
                std::uint32_t mid_price = (tick.asks[0].price + tick.bids[0].price) / 2;
                if (mid_price == 0) {
                    continue;
                }
                std::uint32_t spread_bps = (spread * 10000) / mid_price; // spread in basis points
                if (spread_bps < config.threshold && !holding_shares) {
                    price = tick.asks[0].price;
                    holding_shares = true;
                } else if (spread_bps >= config.threshold && holding_shares) {
                    std::int32_t sell_difference = static_cast<std::int32_t>(tick.bids[0].price) - static_cast<std::int32_t>(price);
                    pnl += (sell_difference * config.position_size);
                    holding_shares = false;
                }
            }
        }
        pnl_results_[i] = pnl;
    }
}

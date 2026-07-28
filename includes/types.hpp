#pragma once
#include <array>
#include <climits>
#include <cstddef>
#include <cstdint>

// The different order types that can be called in the order book
enum class MessageType : char {
    Add = 'A',
    Execute = 'E',
    ExecuteWithPrice = 'C',
    Cancel = 'X',
    Delete = 'D',
    Replace = 'U'
};

struct DecodedOrder {
    std::uint64_t order_id;
    std::uint64_t old_order_id; // Used only for the Replace order
    std::uint64_t timestamp;
    std::uint32_t price;
    std::uint32_t quantity;
    std::uint16_t symbol_id;
    MessageType order_type; // 1B
    char side;
};

// Defined here and not backtester.cuh because results_analysis.hpp uses this
struct StrategyConfig {
  std::uint16_t threshold; // Signal threshold in basis points (1/100th of a percent)
  size_t lookback_window; // How many past ticks to consider when evaluating the signal
  std::uint32_t position_size; // Fixed number of shares to buy or sell when the signal fires (stays fixed for simplicity)
};

inline std::array<size_t, 256> build_message_lengths() {
    std::array<size_t, 256> message_lengths;
    message_lengths.fill(SIZE_MAX);
    // Initialises the lengths of each ITCH message indexed by the message type
    // Lengths obtained from https://www.nasdaqtrader.com/content/technicalsupport/specifications/dataproducts/NQTVITCHSpecification.pdf
    message_lengths[static_cast<unsigned char>('A')] = 36;
    message_lengths[static_cast<unsigned char>('B')] = 19;
    message_lengths[static_cast<unsigned char>('C')] = 36;
    message_lengths[static_cast<unsigned char>('D')] = 19;
    message_lengths[static_cast<unsigned char>('E')] = 31;
    message_lengths[static_cast<unsigned char>('F')] = 40; // With attribution, 36 without
    // No 'G'
    message_lengths[static_cast<unsigned char>('H')] = 25;
    message_lengths[static_cast<unsigned char>('h')] = 21;
    message_lengths[static_cast<unsigned char>('I')] = 50;
    message_lengths[static_cast<unsigned char>('J')] = 35;
    message_lengths[static_cast<unsigned char>('K')] = 28;
    message_lengths[static_cast<unsigned char>('L')] = 26;
    // No 'M'
    message_lengths[static_cast<unsigned char>('N')] = 20;
    message_lengths[static_cast<unsigned char>('O')] = 48;
    message_lengths[static_cast<unsigned char>('P')] = 44;
    message_lengths[static_cast<unsigned char>('Q')] = 40;
    message_lengths[static_cast<unsigned char>('R')] = 39;
    message_lengths[static_cast<unsigned char>('S')] = 12;
    // No 'T'
    message_lengths[static_cast<unsigned char>('U')] = 35;
    message_lengths[static_cast<unsigned char>('V')] = 35;
    message_lengths[static_cast<unsigned char>('W')] = 12;
    message_lengths[static_cast<unsigned char>('X')] = 23;
    message_lengths[static_cast<unsigned char>('Y')] = 20;
    // No 'Z'
    return message_lengths;
}

inline const std::array<size_t, 256> itch_message_lengths = build_message_lengths();

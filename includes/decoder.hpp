#pragma once
#include "types.hpp"
#include <cstring>
#include <limits>
#include <optional>

namespace ItchDecoder {
    std::uint16_t read_be_u16(const std::uint8_t* buffer, size_t offset);
    std::uint32_t read_be_u32(const std::uint8_t* buffer, size_t offset);
    std::uint64_t read_be_u64(const std::uint8_t* buffer, size_t offset);
    std::uint64_t read_be_u48_as_u64(const std::uint8_t* buffer, size_t offset); 
    DecodedOrder decode_add_order(const std::uint8_t* buffer);
    DecodedOrder decode_execute_order(const std::uint8_t* buffer);
    DecodedOrder decode_execute_with_price_order(const std::uint8_t* buffer);
    DecodedOrder decode_cancel_order(const std::uint8_t* buffer);
    DecodedOrder decode_delete_order(const std::uint8_t* buffer);
    DecodedOrder decode_replace_order(const std::uint8_t* buffer);
    std::optional<DecodedOrder> decode_order(const uint8_t* buffer); // We use std::optional as an ITCH file will likely include order types we do not support yet
}



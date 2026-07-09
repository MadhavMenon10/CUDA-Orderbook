#include "../includes/decoder.hpp"

std::uint16_t ItchDecoder::read_be_u16(const std::uint8_t* buffer, size_t offset) {
    // Swaps endianness since machine stores information using little endian while ITCH data is stored using big endian
    std::uint16_t byte_value;
    std::memcpy(&byte_value, buffer + offset, sizeof(std::uint16_t)); // Always reads 2 bytes
    if (std::endian::native == std::endian::little) {
        byte_value = std::byteswap(byte_value);
    }
    return byte_value;
}

std::uint32_t ItchDecoder::read_be_u32(const std::uint8_t* buffer, size_t offset) {
    // Swaps endianness since machine stores information using little endian while ITCH data is stored using big endian
    std::uint32_t byte_value;
    std::memcpy(&byte_value, buffer + offset, sizeof(std::uint32_t)); // Always reads 4 bytes
    if (std::endian::native == std::endian::little) {
        byte_value = std::byteswap(byte_value);
    }
    return byte_value;
}

std::uint64_t ItchDecoder::read_be_u64(const std::uint8_t* buffer, size_t offset) {
    // Swaps endianness since machine stores information using little endian while ITCH data is stored using big endian
    std::uint64_t byte_value;
    std::memcpy(&byte_value, buffer + offset, sizeof(std::uint64_t)); // Always reads 8 bytes
    if (std::endian::native == std::endian::little) {
        byte_value = std::byteswap(byte_value);
    }
    return byte_value;
}

DecodedOrder decode_add_order(const std::uint8_t* buffer) {
    DecodedOrder decoded_add_order;
    decoded_add_order.order_type = MessageType::Add;
    decoded_add_order.order_id = ItchDecoder::read_be_u64(buffer, 11);
    decoded_add_order.price = ItchDecoder::read_be_u32(buffer, 32);
    decoded_add_order.quantity = ItchDecoder::read_be_u32(buffer, 20);
    decoded_add_order.timestamp = ItchDecoder::read_be_u64(buffer, 5);
    decoded_add_order.side = buffer[19];
    decoded_add_order.symbol_id = ItchDecoder::read_be_u16(buffer, 1);
    return decoded_add_order;
}
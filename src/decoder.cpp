#include "../includes/decoder.hpp"

std::uint16_t ItchDecoder::read_be_u16(const std::uint8_t *buffer, size_t offset)
{
    // Swaps endianness since machine stores information using little endian while ITCH data is stored using big endian
    std::uint16_t byte_value;
    std::memcpy(&byte_value, buffer + offset, sizeof(std::uint16_t)); // Always reads 2 bytes
    if (std::endian::native == std::endian::little)
    {
        byte_value = std::byteswap(byte_value);
    }
    return byte_value;
}

std::uint32_t ItchDecoder::read_be_u32(const std::uint8_t *buffer, size_t offset)
{
    // Swaps endianness since machine stores information using little endian while ITCH data is stored using big endian
    std::uint32_t byte_value;
    std::memcpy(&byte_value, buffer + offset, sizeof(std::uint32_t)); // Always reads 4 bytes
    if (std::endian::native == std::endian::little)
    {
        byte_value = std::byteswap(byte_value);
    }
    return byte_value;
}

std::uint64_t ItchDecoder::read_be_u64(const std::uint8_t *buffer, size_t offset)
{
    // Swaps endianness since machine stores information using little endian while ITCH data is stored using big endian
    std::uint64_t byte_value;
    std::memcpy(&byte_value, buffer + offset, sizeof(std::uint64_t)); // Always reads 8 bytes
    if (std::endian::native == std::endian::little)
    {
        byte_value = std::byteswap(byte_value);
    }
    return byte_value;
}

std::uint64_t ItchDecoder::read_be_u48_as_u64(const std::uint8_t *buffer, size_t offset)
{
    // Used to read the timestamps correctly
    std::array<uint8_t, 8> local_buffer{0}; // Timestamp is only 6B but read_be_u64 always reads 8B. So we create a local buffer by zero-padding the top 2B (which has lower-significance in big endian) so the 6B land in the correct order position when read_be_u64 is called
    std::memcpy(local_buffer.data() + 2, buffer + offset, 6);
    return ItchDecoder::read_be_u64(local_buffer.data(), 0);
}

DecodedOrder ItchDecoder::decode_add_order(const std::uint8_t *buffer)
{
    DecodedOrder decoded_add_order;
    decoded_add_order.order_id = ItchDecoder::read_be_u64(buffer, 11);
    decoded_add_order.old_order_id = std::numeric_limits<uint64_t>::max();
    decoded_add_order.timestamp = ItchDecoder::read_be_u48_as_u64(buffer, 5);
    decoded_add_order.price = ItchDecoder::read_be_u32(buffer, 32);
    decoded_add_order.quantity = ItchDecoder::read_be_u32(buffer, 20);
    decoded_add_order.symbol_id = ItchDecoder::read_be_u16(buffer, 1);
    decoded_add_order.order_type = MessageType::Add;
    decoded_add_order.side = buffer[19];
    return decoded_add_order;
}

DecodedOrder ItchDecoder::decode_execute_order(const std::uint8_t *buffer)
{
    DecodedOrder decoded_execute_order;
    decoded_execute_order.order_id = ItchDecoder::read_be_u64(buffer, 11);
    decoded_execute_order.old_order_id = std::numeric_limits<uint64_t>::max();
    decoded_execute_order.timestamp = ItchDecoder::read_be_u48_as_u64(buffer, 5);
    decoded_execute_order.price = 0; // Execute order does not have a price field so this is a sentinel vlaue
    decoded_execute_order.quantity = ItchDecoder::read_be_u32(buffer, 19);
    decoded_execute_order.symbol_id = ItchDecoder::read_be_u16(buffer, 1);
    decoded_execute_order.order_type = MessageType::Execute;
    decoded_execute_order.side = 0; // Execute orders do not have a side field so this is a sentinel value
    return decoded_execute_order;
}

DecodedOrder ItchDecoder::decode_execute_with_price_order(const std::uint8_t *buffer)
{
    DecodedOrder decoded_execute_with_price_order;
    decoded_execute_with_price_order.order_id = ItchDecoder::read_be_u64(buffer, 11);
    decoded_execute_with_price_order.old_order_id = std::numeric_limits<uint64_t>::max();
    decoded_execute_with_price_order.timestamp = ItchDecoder::read_be_u48_as_u64(buffer, 5);
    decoded_execute_with_price_order.price = ItchDecoder::read_be_u32(buffer, 32);
    decoded_execute_with_price_order.quantity = ItchDecoder::read_be_u32(buffer, 19);
    decoded_execute_with_price_order.symbol_id = ItchDecoder::read_be_u16(buffer, 1);
    decoded_execute_with_price_order.order_type = MessageType::ExecuteWithPrice;
    decoded_execute_with_price_order.side = 0; // Execute orders do not have a side field so this is a sentinel value
    return decoded_execute_with_price_order;
}

DecodedOrder ItchDecoder::decode_cancel_order(const std::uint8_t *buffer)
{
    DecodedOrder decoded_cancel_order;
    decoded_cancel_order.order_id = ItchDecoder::read_be_u64(buffer, 11);
    decoded_cancel_order.old_order_id = std::numeric_limits<uint64_t>::max();
    decoded_cancel_order.timestamp = ItchDecoder::read_be_u48_as_u64(buffer, 5);
    decoded_cancel_order.price = 0;
    decoded_cancel_order.quantity = ItchDecoder::read_be_u32(buffer, 19);
    decoded_cancel_order.symbol_id = ItchDecoder::read_be_u16(buffer, 1);
    decoded_cancel_order.order_type = MessageType::Cancel;
    decoded_cancel_order.side = 0;
    return decoded_cancel_order;
}

DecodedOrder ItchDecoder::decode_delete_order(const std::uint8_t *buffer)
{
    DecodedOrder decoded_delete_order;
    decoded_delete_order.order_id = ItchDecoder::read_be_u64(buffer, 11);
    decoded_delete_order.old_order_id = std::numeric_limits<uint64_t>::max();
    decoded_delete_order.timestamp = ItchDecoder::read_be_u48_as_u64(buffer, 5);
    decoded_delete_order.price = 0;
    decoded_delete_order.quantity = 0;
    decoded_delete_order.symbol_id = ItchDecoder::read_be_u16(buffer, 1);
    decoded_delete_order.order_type = MessageType::Delete;
    decoded_delete_order.side = 0;
    return decoded_delete_order;
}

DecodedOrder ItchDecoder::decode_replace_order(const std::uint8_t *buffer)
{
    DecodedOrder decoded_replace_order;
    decoded_replace_order.order_id = ItchDecoder::read_be_u64(buffer, 19); //
    decoded_replace_order.old_order_id = ItchDecoder::read_be_u64(buffer, 11);
    decoded_replace_order.timestamp = ItchDecoder::read_be_u48_as_u64(buffer, 5);
    decoded_replace_order.price = ItchDecoder::read_be_u32(buffer, 31);
    decoded_replace_order.quantity = ItchDecoder::read_be_u32(buffer, 27);
    decoded_replace_order.symbol_id = ItchDecoder::read_be_u16(buffer, 1);
    decoded_replace_order.order_type = MessageType::Replace;
    decoded_replace_order.side = 0;
    return decoded_replace_order;
}

DecodedOrder ItchDecoder::decode_order(const std::uint8_t *buffer)
{
    DecodedOrder decoded_order;
    char order_message_type = buffer[0];
    if (order_message_type == 'F')
    {
        // Add with MPID treated as an Add Order
        order_message_type = 'A';
    }
    MessageType message_type = static_cast<MessageType>(order_message_type);
    switch (message_type)
    {
    case MessageType::Add:
        decoded_order = decode_add_order(buffer);
        break;
    case MessageType::Execute:
        decoded_order = decode_execute_order(buffer);
        break;
    case MessageType::ExecuteWithPrice:
        decoded_order = decode_execute_with_price_order(buffer);
        break;
    case MessageType::Cancel:
        decoded_order = decode_cancel_order(buffer);
        break;
    case MessageType::Delete:
        decoded_order = decode_delete_order(buffer);
        break;
    case MessageType::Replace:
        decoded_order = decode_replace_order(buffer);
        break;
    default:
        throw std::runtime_error("Order Type not supported.");
    }
    return decoded_order;
}

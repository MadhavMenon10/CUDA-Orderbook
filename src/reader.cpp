#include "reader.hpp"
#include "decoder.hpp"
#include <cstring>

ItchReader::ItchReader(const std::string& file_path, size_t buffer_size, size_t length_prefix_size) : buffer_(buffer_size), file_(file_path), buffer_pos_(0), buffer_valid_(0), length_prefix_size_(length_prefix_size) {
    if (!file_.is_open()) {
        throw std::runtime_error(file_path + " is an invalid path.");
    }
}

size_t ItchReader::num_bytes_available() const {
    return buffer_valid_ - buffer_pos_;
}

void ItchReader::refill() {
    size_t num_bytes_to_move = num_bytes_available();
    memmove(buffer_.data(), buffer_.data() + buffer_pos_, num_bytes_to_move);
    size_t num_fresh_bytes = buffer_.size() - num_bytes_to_move;
    file_.read(reinterpret_cast<char*>(buffer_.data() + num_bytes_to_move), num_fresh_bytes);
    size_t bytes_read = static_cast<size_t>(file_.gcount());
    buffer_pos_ = 0;
    buffer_valid_ = num_bytes_to_move + bytes_read; 
}

bool ItchReader::get_next_message(uint8_t* memory_space, size_t memory_size, size_t& message_size) {
    if (num_bytes_available() < length_prefix_size_) {
        refill();
    }
    if (num_bytes_available() < length_prefix_size_) {
        return false; // or throw, if less-than-expected-but-nonzero is itself a sign of a truncated/corrupt file
    }
    std::uint64_t prefix_length = 0;
    switch (length_prefix_size_) {
        case 2:
            prefix_length = ItchDecoder::read_be_u16(buffer_.data(), buffer_pos_);
            break;
        case 4:
            prefix_length = ItchDecoder::read_be_u32(buffer_.data(), buffer_pos_);
            break;
        case 8:
            prefix_length = ItchDecoder::read_be_u64(buffer_.data(), buffer_pos_);
            break;
        default:
            throw std::runtime_error("Prefix length not supported. Please try another ITCH file");
    }
    buffer_pos_ += length_prefix_size_;
    size_t message_length = itch_message_lengths[static_cast<unsigned char>(buffer_[buffer_pos_])];
    if (prefix_length != message_length) {
        throw std::runtime_error("Prefix length " + std::to_string(prefix_length) + " does not match expected message length " + std::to_string(message_length) + " for type " + std::string(1, buffer_[buffer_pos_]) + " at position " + std::to_string(buffer_pos_));
    }
    if (message_length == SIZE_MAX) {
        throw std::runtime_error("Message type " + std::string(1, buffer_[buffer_pos_]) + " not supported. Error occured at " + std::to_string(buffer_pos_));
    }
    if (num_bytes_available() == 0) {
        refill();
    }
    if (num_bytes_available() == 0) {
        return false;
    }
    if (buffer_pos_ + message_length > buffer_valid_) {
        refill();
    }
    if (buffer_pos_ + message_length > buffer_valid_) {
        throw std::runtime_error(std::to_string(buffer_pos_ + message_length) + " exceeds read limit of " + std::to_string(buffer_valid_));
    }
    if (message_length > memory_size) {
        throw std::runtime_error("The message length of " + std::to_string(message_length) + " exceeds the memory size of " + std::to_string(memory_size) + "B");
    }
    std::memcpy(memory_space, buffer_.data() + buffer_pos_, message_length);
    buffer_pos_ += message_length;
    message_size = message_length;
    return true;
}

void ItchReader::reset_state() {
    file_.clear();
    file_.seekg(0, std::ios::beg);
    buffer_pos_ = 0;
    buffer_valid_ = 0;
}



#include "reader.hpp"

ItchReader::ItchReader(const std::string& file_path, size_t buffer_size) : buffer_(buffer_size), file_(file_path), buffer_pos_(0), buffer_valid_(0) {
    if (!file_.is_open()) {
        throw std::runtime_error(file_path + " is an invalid path.");
    }

}

size_t ItchReader::num_bytes_available() const {
    return buffer_valid_ - buffer_pos_;
}

void ItchReader::refill() {
    
    size_t num_bytes_to_move = this->num_bytes_available();
    memmove(buffer_.data(), buffer_[buffer_pos_], num_bytes_to_move);
    buffer_pos_ = 0;

}

bool ItchReader::get_next_message(uint8_t* memory_space, size_t memory_size, size_t& message_size) {

}



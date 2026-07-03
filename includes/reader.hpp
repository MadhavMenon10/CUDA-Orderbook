#pragma once
#include <cstddef>
#include <vector>
#include <cstdint>
#include <fstream>
#include <string>
#include <stdexcept>


class ItchReader {
    public:
        ItchReader(const std::string& file_path, size_t buffer_size);
        bool get_next_message(uint8_t* memory_space, size_t memory_size, size_t& message_size);
    private:
        size_t num_bytes_available() const;
        void refill();
        std::vector<std::uint8_t> buffer_;
        std::ifstream file_;
        size_t buffer_pos_;
        size_t buffer_valid_;
};


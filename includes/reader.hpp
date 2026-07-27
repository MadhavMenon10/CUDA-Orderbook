#pragma once
#include "types.hpp"
#include <cstddef>
#include <vector>
#include <cstdint>
#include <fstream>
#include <string>
#include <stdexcept>

class ItchReader {
    public:
        ItchReader(const std::string &file_path, size_t buffer_size);
        ItchReader(const ItchReader &) = delete;
        ItchReader &operator=(const ItchReader &) = delete;
        ItchReader(ItchReader &&) = delete;
        ItchReader &operator=(ItchReader &&) = delete;
        bool get_next_message(uint8_t *memory_space, size_t memory_size, size_t &message_size);
        void reset_state();

    private:
        size_t num_bytes_available() const;
        void refill();
        std::vector<std::uint8_t> buffer_;
        std::ifstream file_;
        size_t buffer_pos_;
        size_t buffer_valid_;
};

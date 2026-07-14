#include <iostream>
#include "types.hpp"
#include "reader.hpp"
#include "decoder.hpp"
#include "soa.hpp"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "ITCH File Path is missing\n";
        return 1;
    }
    try {
        std::string file_path(argv[1]);
        size_t one_mb = 1048576;
        ItchReader itch_reader(file_path, one_mb);
        std::array<uint8_t, 64> scratch_memory;
        size_t message_size = 0;
        size_t message_count = 0;
        while (itch_reader.get_next_message(scratch_memory.data(), scratch_memory.size(), message_size)) {
            ++message_count;
        }
        SoaArrays soa_arrays;
        soa_arrays.reserve(message_count);
        itch_reader.reset_state();
        while (itch_reader.get_next_message(scratch_memory.data(), scratch_memory.size(), message_size)) {
            DecodedOrder decoded_order = ItchDecoder::decode_order(scratch_memory.data());
            soa_arrays.append(decoded_order);
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
    return 0;
}

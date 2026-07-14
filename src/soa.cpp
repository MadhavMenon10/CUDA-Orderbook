#include "soa.hpp"

SoaArrays::SoaArrays() : order_ids_(nullptr), old_order_ids_(nullptr), timestamps_(nullptr), prices_(nullptr), quantities_(nullptr), symbol_ids_(nullptr), order_types_(nullptr), sides(nullptr), size_(0), capacity_(0) {}

SoaArrays::~SoaArrays()
{
    if (order_ids != nullptr)
    {
        cudaFreeHost(order_ids_);
    }
    if (old_order_ids != nullptr)
    {
        cudaFreeHost(old_order_ids_);
    }
    if (timestamps != nullptr)
    {
        cudaFreeHost(timestamps_);
    }
    if (prices != nullptr)
    {
        cudaFreeHost(prices_);
    }
    if (quantities != nullptr)
    {
        cudaFreeHost(quantities_);
    }
    if (symbol_ids != nullptr)
    {
        cudaFreeHost(symbol_ids_);
    }
    if (order_types != nullptr)
    {
        cudaFreeHost(order_types_);
    }
    if (sides != nullptr)
    {
        cudaFreeHost(sides_);
    }
    size_ = 0;
    capacity_ = 0;
}


void SoaArrays::reserve(const size_t message_count) {
    if (order_ids_ != nullptr) { // Only need to check one array as all arrays are assigned together
        throw std::runtime_error("reserve() has already been called before"); 
    }
    check_cuda_memalloc_error(cudaMallocHost(reinterpret_cast<void**>(&order_ids_), sizeof(std::uint64_t) * message_count), "Memory allocation for order_ids_");
    check_cuda_memalloc_error(cudaMallocHost(reinterpret_cast<void**>(&old_order_ids_), sizeof(std::uint64_t) * message_count), "Memory allocation for old_order_ids_");
    check_cuda_memalloc_error(cudaMallocHost(reinterpret_cast<void**>(&timestamps_), sizeof(std::uint64_t) * message_count), "Memory allocation for timestamps_");
    check_cuda_memalloc_error(cudaMallocHost(reinterpret_cast<void**>(&prices_), sizeof(std::uint32_t) * message_count), "Memory allocation for prices_");
    check_cuda_memalloc_error(cudaMallocHost(reinterpret_cast<void**>(&quantities_), sizeof(std::uint32_t) * message_count), "Memory allocation for quantities_");
    check_cuda_memalloc_error(cudaMallocHost(reinterpret_cast<void**>(&symbol_ids_), sizeof(std::uint16_t) * message_count), "Memory allocation for symbol_ids_");
    check_cuda_memalloc_error(cudaMallocHost(reinterpret_cast<void**>(&order_types_), sizeof(MessageType) * message_count), "Memory allocation for order_types_");
    check_cuda_memalloc_error(cudaMallocHost(reinterpret_cast<void**>(&sides_), sizeof(char) * message_count), "Memory allocation for sides_");
    size_ = 0;
    capacity_ = message_count;
}

void SoaArrays::append(const DecodedOrder &decoded_order)
{
    if (size_ >= capacity_)
    {
        throw std::runtime_error("The size of the SoaArrays exceed its capacity");
    }
    order_ids_[size_] = decoded_order.order_id;
    old_order_ids_[size_] = decoded_order.old_order_id;
    timestamps_[size_] = decoded_order.timestamp;
    prices_[size_] = decoded_order.price;
    quantities_[size_] = decoded_order.quantity;
    symbol_ids_[size_] = decoded_order.symbol_id;
    order_types_[size_] = decoded_order.order_type;
    sides_[size_] = decoded_order.side;
    ++size_;
}


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

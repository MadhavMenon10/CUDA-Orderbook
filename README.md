# CUDA Orderbook
This README was written by a human.

## CUDA Orderbook Reconstructor and Backtester

A GPU pipeline that parses raw NASDAQ TotalView ITCH 5.0 market data, reconstructs a live limit order book for every traded symbol, and sweeps 10,000 trading strategy configurations against the reconstructed book in parallel (using the GPU).

### Benchmarks

The following values come from testing on the 30/08/2019 NASDAQ TotalView-ITCH  5.0 file available at the [NASDAQ Public Archive](https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/)

- **305,105,310 messages** reconstructed in **69.5s** (**~4.4M msgs/sec**)
- **10,000 configs** backtested in **83.8s**



## Results

## Build

## Languages and Tooling

## Directory Layout

## Pipeline

### Parsing the ITCH Feed

### GPU Hash Table

### Compacting by Symbols

### Reconstruction

### Backtesting

## Design Choices


## Challenges

## Limitations

## Future Work


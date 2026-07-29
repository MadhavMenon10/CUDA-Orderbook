# CUDA Orderbook
This README was written by a human.

## CUDA Orderbook Reconstructor and Backtester

A GPU pipeline that parses raw NASDAQ TotalView ITCH 5.0 market data, reconstructs a live limit order book for every traded symbol, and sweeps 10,000 trading strategy configurations against the reconstructed book in parallel (using the GPU).

## Results

### Measurements

The following values come from testing on the 30/08/2019 NASDAQ TotalView-ITCH  5.0 file available at the [NASDAQ Public Archive](https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/). The entire project was run on an NVIDIA H100 SXM cloud instance on [Runpod.io](https://www.runpod.io/)

- **305,105,310 messages** reconstructed in **69.5s** (**~4.4M msgs/sec**)
- **10,000 configs** backtested in **83.8s**

#### Nsight Systems Profile

- OS Runtime Summary

![OS Runtime Summary](docs/pics/osrt_summary.png)

- CUDA GPU Kernel Summary

![CUDA GPU Kernel Summary](docs/pics/kernel_summary.png)

- CUDA GPU MemOps Summary (by Time)

![CUDA GPU MemOps Summary (by Time)](docs/pics/memops_summary.png)

- CUDA GPU MemOps Summary (by Size)

![CUDA GPU MemOps Summary (by Size)](docs/pics/memops_summary_size.png)

Unfortunately, the GPU cloud instance does not support NVIDIA Nsight Compute `ncu`. This is common on most rented GPU containers due to driver security restrictions; an `ncu` analysis was thus ommitted.

## Build

Clone the repo and compile with
```bash
make bin/itch_parser
```
If you have a NASDAQ Itch file on disc, execute the binary with

```bash
./bin/itch_parser path_to_file
```
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


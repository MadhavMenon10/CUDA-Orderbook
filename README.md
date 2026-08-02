# CUDA Orderbook
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

### CPU Baseline

A GPU measurement on its own says very little, so `bin/cpu_baseline` runs the same two stages single-threaded on the host. The GPU shards price levels across 32 lanes and caps each lane at `MAX_LEVELS_PER_LANE` because shared memory cannot dynamically grow once a kernel launches. The CPU implementation on the other hand uses an ordered map of price levels per symbol, and a `std::unordered_map` keyed by order ID in place of the open-addressed table. Both halves of the comparison were measured on the same H100 instance, so what follows is the GPU against that machine's host CPU.

- Reconstruction: **453.3s** (**~673K msgs/sec**), so the GPU is **6.5x** faster
- Backtesting: **~6.5 hours**, so the GPU is **~279x** faster


## Build

Clone the repo and compile with
```bash
make bin/itch_parser
```
If you have a NASDAQ Itch file on disc, execute the binary with

```bash
./bin/itch_parser path_to_file
```

The CPU baseline is a separate binary that links against no CUDA at all, so it builds and runs on a machine with no GPU in it

```bash
make bin/cpu_baseline
./bin/cpu_baseline path_to_file
```

The CUDA paths in the Makefile are specific to the GPU cloud instance. You may have to change it to match your system.

## Languages and Tooling
C++ was used for code on the host (CPU) while CUDA C++ was used for everything on the device (GPU).  CUB was used for radix sort and run-length-encoding used for compaction. Thrust was used to facilitate this.

## Directory Layout

## Pipeline

### Parsing the ITCH Feed
An average ITCH file can be >= 10 GB. Thus loading the entire file at once into memory, before starting any of the reconstructing and backtesting, can cause a severe bottleneck. Thus, A custom `ItchReader` class and `ItchDecoder` namespace was created to parse ITCH files. `ItchReader` keeps a fixed-size pinned buffer (set to 1 MB in `main.cu` to read chunks from the file, `ItchDecoder` decodes messages out of this window, and the buffer gets refilled as it is consumed. `ItchDecoder` decodes the message and stores its data into a struct of arrays (SoA) format, in order to keep data contiguous for the GPU. 

Parsing then runs two full passes over the file. The first counts the number of messages so that we can allocate the right amount of memory for our `SoA` once, as opposed to growing and reallocating pinned memory repeatedly as messages come in. The second pass then decodes each message, adds its data into our `SoA`. For simplicity, only the following message types were handled:
- Add
- Execute
- Execute With Price
- Cancel
- Delete
- Replace

We return a `std::optional<DecodedOrder>` for message types not handled.

NASDAQ's sample files prefix every message with a 2-byte big-endian length field, which is not a part of the ITCH spec as this field is a property of how the sample archive packages messages for download, and not the ITCH protocol itself. Thus, `ItchReader` takes a `length_prefix_size` constructor argument for this, reads and skips that many bytes before the type byte, and cross-checks the resulting length against what the ITCH spec. A mismatch throws immediately with the exact byte position.

### GPU Hash Table

Not all messages contain the same data. For example, an `Add` order would include everything from the order ID to whether it was on the buy or sell side, while a `Delete` order would only include the order ID. Thus, a hash table is needed to remember all the parameters associated with an order. Reconstruction involves inserting, looking up, and deleting orders potentially hundreds of millions of times, from inside a kernel running thousands of warps in parallel. Thus, the hash table was implemented entirely on the GPU to avoid round-trip memory accesses which would bottleneck the program. The downside of this is concurrency, as many warps can use the table at the same instant. To mitigate this, we used atomic operations to ensure that all warps agree on the state of the hash table at any given time.

The hash table is sized at twice the number of `max_active_orders` in the orderbook. This leads to a load factor of about 50% which allows linear probe chains to be relatively short. We use two sentinel values to represent the states a given entry in the hash table can be

- `UINT64_MAX`: A slot is empty
- `UINT64_MAX - 1`: An order was used at this slot and then deleted

An insert treats either sentinel as a slot available to claim. A lookup, however, has to treat the two differently: it can only stop early on UINT64_MAX, since a tombstoned slot (UINT64_MAX - 1) might sit in the middle of a probe chain where some other key's insert collided and continued further along. Stopping at a tombstone the way we stop at a true empty slot would incorrectly report an order as not found when it's actually a few slots further down the chain.

The hash function itself is a splitmix64-style mix rather than a plain modulo.

### Compacting by Symbols

The raw parsed ITCH stream interleaves every symbol together in timestamp order, since that's the order they actually arrived in the file. `SymbolCompactor` groups it into contiguous per-symbol blocks, in device memory, without disturbing the relative order of messages within a symbol, so a warp assigned one symbol can read a sequential slice instead of scanning past every other symbol's data. 

To do this, we use `cub::DeviceRadixSort::SortPairs` to sort two arrays: a proxy index array and `symbol_id`. We do this in two phases: the first phase sizes the required scratch memory and the second phase does the sort. The sorted index array becomes a permutation: position i in the output came from index `permutation[i]` in the input. This permutation drives a `thrust::gather` pass for each SoA field by pulling  `field[permutation[i]]` into position `i`. Then, `cub::DeviceRunLengthEncode::Encode` walks the sorted `symbol_id` array and finds each distinct symbol along with how many consecutive messages belong to it. A `thrust::exclusive_scan` over those counts finds the offset of each symbol, which is used by a warp during reconstruction to find its own slice.

### Reconstruction

For reconstruction, we settled on assigning one warp to each symbol, and one block holding one warp. We used one warp per symbol because of the inherent causality inside a single symbol's message stream. For example, a `Cancel` message cannot be executed before the `Add` message it references exists. Thus, for a symbol, messages have to be processed strictly in the order that they come through. However, across symbols, there is no such constraint. For example, AAPL's messages have nothing to do with MSFT's messages and vice-versa. Thus, a warp was the natural execution unit for a symbol. We used one warp per block because reconstruction requires several states to live in shared memory like the best bid and the best ask. In CUDA, shared memory is scoped per block. Thus, if we had multiple warps in a block, it would be possible for one symbol to mutate the data associated with another symbol. Thus, it was natural to give one warp its own block.

Within a given symbol, the work we do is naturally single-threaded and sequential. The parallelism behind this project comes from running potentially thousands of warps at once across symbols, not from spreading one message's update across 32 threads. All 32 threads collaborate when the orderbook is stored and queried. Bid and ask prices live in two separate arrays. Each thread owns a owns a small fixed-size local array of price levels, and which thread owns a given price is decided by a custom hashing function. Since a symbol routinely has more than 32 live prices and several hash to the thread, a lane's array is small enough that finding or claiming a slot in it is just a linear scan.

After a message is processed, all 32 threads work to find the top 5 bid and ask levels. This is done via a warp shuffle in five rounds that halves the number of active threads each round (32 -> 16 -> 8 -> 4 -> 2 -> 1).

### Backtesting

We settled on 10,000 different trading strategy configurations, one per block, with `blockIdx.x` indexing directly into an array storing trading configurations. Each block runs a single thread through the whole tick stream in order, since a strategy's decision at tick `N` depends on what it already decided at tick `N-1`. This is the same causality constraint from reconstruction showing up again at a different scale, so it can't be parallelised within one config either. The parallelism is entirely in running 10,000 of these independent walks at once.

Reconstruction doesn't hand the backtester a clean tick stream to begin with, though. Ticks get claimed through a single global `atomicAdd` on a shared counter, so the write order across the whole file reflects whichever warp happened to finish a given message first, not chronological order within any one symbol. `TickCompactor` re-sorts the claimed ticks by symbol, using the same sort/gather/run-length-encode/scan sequence as `SymbolCompactor`,

The trading strategy is itself deliberately simple as that was not the main focus of the project. The strategy was to buy a fixed size when the spread narrows below a config's threshold and nothing is currently held, sell once it widens back past that threshold. The spread is measured in basis points rather than raw cents, since the same cent spread means something different on a $5 stock than a $500 one.

## Challenges

The main challenge was a hardware ceiling on shared memory. The per-thread price level array size started at 16 (an educated guess), which failed for a handful of heavily traded symbols once tested against real data. Raising it to 64 fixed most of that but was still insufficient. Pushing further ran directly into the cloud GPU's shared memory limit of 49,152 bytes per block. At 96, the kernel failed to link. The real ceiling on this hardware works out to around 94 price levels per thread.

## Limitations

One symbol out of roughly 9,000 in the full test file still produces a small number of reconstruction failures (99 out of 305,105,310 messages, all in what appears to be a single, very high liquidity name), even after raising the per-lane capacity as far as the shared memory ceiling allows. 

There are also some inefficiencies present within the project: `SymbolCompactor`'s offset table arrays are sized at the full message count when they only need `num_unique_symbols` which can cost several GB on a full scale run. However, this is done because `num_unique_symbol` is not known until after `cub::DeviceRunLengthEncode::Encode` runs. We still need to `cudaMalloc` space since CUB needs pointers to write output into. We also overallocate space for `ticks_`. Before, we used to allocate one tick per message. This design consumed too much space. At real, full-day scale, `OrderBookTick` is a large struct, and multiplying that by 310 million (one tick per every single message) worked out to roughly 27GB just for that one array. Thus, we used an atomic-claim redesign: instead of guaranteeing a slot for every message regardless of whether anything changed, we only claim a slot when the tick differs from the last one written, via an atomic counter. That's a real reduction in how many ticks actually get written. However, we didn't know how many ticks were written until after reconstruction. Thus, we had to overallocate for `ticks_` too.

## Future Work

Currently, `ItchReader` reading the file, the host-to-device transfer of `SoaArrays` into device memory, and the GPU running reconstruction all happen one after another. `main.cu` reads through the whole file twice with `ItchReader` (once to count how many messages we have and once to decode) before anything ever touches the GPU, so the GPU sits completely idle while that's happening, and `ItchReader` sits idle again once the GPU takes over. The async pipeline is about overlapping those three stages instead of running them in sequence by breaking everything into chunks. For example, while reconstruct is working through chunk `N`, a second CUDA stream is transferring chunk `N+1'`s decoded messages from host to device, and a separate CPU thread is running ItchReader on chunk `N+2 `at the same time. 

Further, tombstones are never reclaimed right now in the hash table. A deleted slot stays deleted forever instead of eventually going back to a true empty state, so a long enough run degrades every probe toward a full linear scan of the whole table once no true empty slot remains anywhere. The current workaround is oversizing the table. We may have to work on periodic compaction of the hash table to enable reclamation of slots. 

We would also like to get occupancy numbers out of Nsight Compute at some point, but that needs a host that actually allows performance counter access, which a rented cloud GPU container generally will not. This will have to wait until I get my own NVIDIA GPU.

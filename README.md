Page 1: GPU Hardware & Memory Hierarchy

Covers: Tensor Cores, Streaming Multiprocessors, NVLink/NVSwitch, memory tiers (HBM → L2 → shared → registers). Central dimension: where data lives and how fast it moves.

Page 2: CUDA Execution Model

Covers: Threads/Warps/Blocks/Grids, occupancy, warp efficiency, instruction-level parallelism. Central dimension: how work is mapped onto GPU compute resources.

Page 3: GPU Memory Access Patterns

Covers: Coalesced vs uncoalesced access, vectorized loads, shared memory tiling, warp shuffle, async prefetch. Central dimension: memory access strategy and bandwidth utilization.

Page 4: Kernel Optimization Techniques

Covers: Kernel fusion, micro-tiling, mixed precision, CUTLASS, PTX tuning. Central dimension: arithmetic intensity and compute efficiency.

Page 5: Pipelining & Concurrency

Covers: Intra-kernel pipelining, warp-specialized producer-consumer, CUDA streams, CUDA Graphs, persistent kernels. Central dimension: overlapping compute, memory, and communication.

Page 6: Distributed Communication Primitives

Covers: All-reduce, reduce-scatter, all-gather, all-to-all, P2P — when each is used, bandwidth cost, NCCL topology awareness, SHARP. Central dimension: collective operation type and cost. (This is a natural companion to your existing parallelism cheatsheet.)

Page 7: Inference Serving & Batching

Covers: Dynamic batching, continuous batching, prefill vs decode phases, disaggregated prefill/decode, KV cache management. Central dimension: request lifecycle and compute phase.

Page 8: KV Cache & Attention Kernels

Covers: FlashAttention, FlashMLA, paged KV cache, KV cache quantization, GQA/MQA. Central dimension: memory footprint and recompute tradeoffs during inference.

Page 9: Quantization & Precision

Covers: FP32/BF16/FP8/INT8/INT4, weight-only vs activation quantization, dynamic precision changes. Central dimension: precision level and accuracy/performance tradeoff.

Page 10: Inference Parallelism for MoE & Speculative Decoding

Covers: EP routing strategies, speculative decoding, parallel decoding, disaggregated routing. Central dimension: decode throughput strategies. (Extends your existing EP card.)

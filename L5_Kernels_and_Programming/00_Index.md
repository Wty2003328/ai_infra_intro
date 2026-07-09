# L5 — Kernels & Programming

> This layer covers GPU kernel programming — from CUDA fundamentals through advanced optimization, Triton DSL, FlashAttention algorithm, and cutting-edge kernel libraries. L5 is where hardware (L0-L3) becomes software: every roofline bottleneck identified in L3 gets a kernel-level solution here.

---

## Scope of this layer

L5 covers the **kernel-level programming** of AI accelerators.

1. **CUDA fundamentals** — thread/block/warp/grid hierarchy, memory model, synchronization, streams, events.
2. **CUDA optimization** — coalescing, bank conflicts, occupancy, tensor cores, async copy, TMA, warp specialization, profiling.
3. **Triton and kernel DSLs** — Triton programming model, autotuning, matmul/softmax/attention kernels, CUTLASS/CuTe overview.
4. **FlashAttention deep dive** — v1/v2/v3 algorithms, online softmax derivation, IO complexity proof, Hopper-specific optimizations.
5. **Cutting-edge kernels** — CUTLASS 3.x, FlashInfer, DeepEP, TileLang, ThunderKittens, Liger Kernel.

L5 explicitly does **not** cover chip architecture (→ L3), network topology (→ L4), model algorithms (→ L6), or distributed training (→ L7).

---

## Pages (read in order)

1. [CUDA_Programming](01_CUDA_Programming.md) — the CUDA programming model from scratch. Thread hierarchy, memory spaces, synchronization, streams, complete matmul example.
2. [CUDA_Optimization](02_CUDA_Optimization.md) — the optimization hierarchy: coalescing, shared memory tiling, bank conflicts, occupancy, tensor cores, async copy, TMA, warp specialization, profiling with Nsight.
3. [Triton_and_Kernels](04_Triton_and_Kernels.md) — Triton DSL, JIT compilation, autotuning, block-pointer model, matmul/softmax/attention in Triton, CUTLASS/CuTe overview.
4. [FlashAttention_Deep_Dive](05_FlashAttention_Deep_Dive.md) — full algorithm walkthrough v1→v2→v3, online softmax derivation, IO complexity proof, tile-sizing math, backward pass, FlashDecoding.
5. [Cutting_Edge_Kernels](06_Cutting_Edge_Kernels.md) — CUTLASS 3.x, CuTe layout algebra, FlashInfer, DeepEP, TileLang, ThunderKittens, Liger Kernel, quantized matmul kernels.
6. [ML_Compilers](07_ML_Compilers.md) — torch.compile (Dynamo/AOTAutograd/Inductor), guards and graph breaks, CUDA Graphs, XLA/GSPMD/Pallas, MLIR ecosystem, autotuning, compilers inside vLLM/SGLang, what compilers still can't do.

---

## What L5 hands off to L6 and above

| L5 fact | Higher-layer consequence |
|---|---|
| FA-v3 hits 75% of Hopper FP16 peak | Training attention is no longer the bottleneck (L7) |
| FlashDecoding parallelizes decode at batch=1 | Single-request latency is hardware-limited, not software-limited (L8) |
| Triton achieves 90-95% of cuBLAS | Custom kernels have low engineering cost (L6, L7, L8) |
| CUTLASS 3.x + CuTe unlocks TMA + wgmma | Hopper/Blackwell kernel tuning is a structured problem, not black art |
| DeepEP achieves 5-10× over NCCL for MoE all-to-all | MoE training throughput no longer limited by communication (L7) |
| Liger Kernel eliminates (B,S,V) logits tensor | Cross-entropy memory drops 50% at large vocab (L7) |
| PagedAttention kernel handles scattered KV | vLLM-style serving has negligible memory waste (L8) |

---

## Reading order

Top-down for orientation: skim this Index → FlashAttention_Deep_Dive for the canonical algorithm → back to CUDA_Programming for fundamentals.

Strict bottom-up for depth: CUDA_Programming → CUDA_Optimization → Triton_and_Kernels → FlashAttention_Deep_Dive → Cutting_Edge_Kernels → ML_Compilers.

After this layer, ascend to [L6 — Algorithms & Models](../L6_Algorithms_and_Models/00_Index.md) where kernels become model architectures.


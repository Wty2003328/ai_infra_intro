# GPU Architecture — Senior Engineer Level

This page walks through the silicon that runs LLMs: how an SM is built, why tensor cores dominate FLOPS, how a warp actually executes, and what changes between Ampere, Hopper, and Blackwell generations. If you understand this page, the rest of inference system design becomes mechanical.

---

## 1. Why GPUs for LLMs

### 1.1 The workload shape

Transformer inference is a sequence of large dense matrix multiplications (GEMMs) plus a handful of element-wise, reduction, and softmax operations. The GEMMs dominate both FLOPS and memory traffic:

- MLP up-projection: `[batch·seq, hidden] × [hidden, 4·hidden]`
- MLP down-projection: `[batch·seq, 4·hidden] × [4·hidden, hidden]`
- QKV projection: `[batch·seq, hidden] × [hidden, 3·hidden]`
- Output projection: `[batch·seq, hidden] × [hidden, hidden]`

For a 70B model, a single forward pass over a 4K-token prompt is roughly 2·70e9·4096 ≈ 5.7·10¹⁴ FLOPs — 570 teraFLOPs. CPUs top out at a few TFLOPS; GPUs do tens of petaFLOPS. That's the gap.

### 1.2 Architectural philosophy: latency vs throughput

| Dimension | CPU | GPU |
|---|---|---|
| Optimization target | Latency of a single thread | Throughput across thousands of threads |
| Control logic per core | Large (branch predictor, OOO, speculative execution) | Tiny (SIMT lane dispatch) |
| Cache per core | Large (L1 32-48 KB, private L2 1 MB, shared L3 30-100 MB) | Tiny per thread; shared SRAM per SM |
| Memory bandwidth | ~100 GB/s (DDR5) | 3.3–12 TB/s (HBM) |
| Concurrent threads | 8–128 cores × 2 hyperthreads | 100,000+ resident threads |

The core insight: a GPU does not avoid memory latency — it **hides** it. When a warp stalls on a load, the scheduler instantly switches to another ready warp at zero cost. An SM keeps up to 64 warps (2048 threads) resident, so there is almost always some warp ready to execute while others wait on memory.

This is why "occupancy" (active warps / max warps) matters on a GPU and why "ILP" (instruction-level parallelism per thread) matters on a CPU — different mechanisms for different problems.

---

## 2. The Chip Hierarchy

Modern NVIDIA GPUs have a strict nested hierarchy:

```
GPU (full chip: H100 = 132 SMs, B200 = 144 SMs per die × 2 dies)
├── GPC (Graphics Processing Cluster) × 7–8
│   ├── TPC (Texture Processing Cluster) × 9
│   │   └── SM (Streaming Multiprocessor) × 2
│   │       ├── CUDA Cores (FP32) × 128
│   │       ├── Tensor Cores × 4
│   │       ├── Load/Store Units × 32
│   │       ├── SFU (Special Function Units — exp, log, rsqrt) × 16
│   │       ├── Warp Schedulers × 4 (one per "processing block")
│   │       ├── Register File: 256 KB
│   │       ├── L1 Cache / Shared Memory: 256 KB (configurable split)
│   │       └── Hopper+: Tensor Memory Accelerator (TMA)
├── L2 Cache (shared across all SMs): 50 MB (H100), 36–72 MB (B200 per die)
├── Memory Controllers → HBM3 / HBM3e stacks
├── NVLink ports (18 × 50 GB/s on Blackwell)
├── PCIe Gen5 x16 interface (128 GB/s bidirectional)
├── GigaThread Engine (global block scheduler)
└── Copy Engines (DMA for host ↔ device)
```

### 2.1 Key chip specs to memorize

| Metric | A100 (80 GB) | H100 SXM5 | B200 (SXM) | B300 / Blackwell Ultra |
|---|---|---|---|---|
| SM count | 108 | 132 | 144 × 2 dies = 288 exposed | 288 |
| Transistors | 54 B | 80 B | 208 B | 208 B |
| Process | TSMC 7N | TSMC 4N | TSMC 4NP | TSMC 4NP |
| FP16 tensor TFLOPS | 312 | 989 | 2250 (dense) | 2250 |
| FP8 tensor TFLOPS | — | 1979 | 4500 | 4500 |
| FP4 tensor TFLOPS | — | — | 9000 | 15000 |
| HBM capacity | 80 GB | 80 GB HBM3 | 192 GB HBM3e | 288 GB HBM3e |
| HBM bandwidth | 2.0 TB/s | 3.35 TB/s | 8.0 TB/s | 12 TB/s |
| NVLink per GPU | 600 GB/s | 900 GB/s | 1800 GB/s | 1800 GB/s |
| TDP | 400 W | 700 W | 1000 W | 1200 W |

**Numbers to burn in:** H100 has 3.35 TB/s HBM and ~990 FP16 TFLOPS; the ridge point is ~295 ops/byte. B200 doubles HBM bandwidth to 8 TB/s. Blackwell Ultra goes to 12 TB/s and 288 GB.

---

## 3. Inside a Streaming Multiprocessor

An SM is the atomic unit of GPU compute. It has four **processing blocks** (a.k.a. sub-partitions, quads), each with:

- 1 warp scheduler + dispatch unit
- 16 FP32 CUDA cores + 16 INT32 cores (fused on Hopper as 32 FP32 lanes)
- 1 Tensor Core
- 8 Load/Store units
- 4 Special Function Units (sin, cos, exp, log, rsqrt)
- 16 KB register file slice (total 64 KB × 4 = 256 KB per SM)

The four processing blocks share:
- 256 KB L1 cache / shared memory (programmer configures the split)
- L1 instruction cache
- Texture units
- Tensor Memory Accelerator (Hopper+) for bulk async memory copy

### 3.1 Warp execution model (SIMT)

A **warp** is 32 threads that execute **one instruction at a time** in lockstep. This is Single Instruction Multiple Thread (SIMT) — like SIMD, but with per-thread control flow handled via masking.

```c
if (threadIdx.x < 16) {
    // Threads 0-15 execute this branch; 16-31 are masked off
    A();
} else {
    // Threads 16-31 execute this branch; 0-15 are masked off
    B();
}
```

When threads in a warp take divergent paths, the warp executes both paths serially, with each thread active only on its own path. This is **warp divergence**, and it halves or further reduces effective throughput. Modern compilers and the Independent Thread Scheduling (Volta+) mitigate this, but it's still a primary optimization concern.

### 3.2 Instruction issue

Each warp scheduler can issue one instruction per clock. With 4 schedulers per SM and 132 SMs on H100, peak issue rate is:

    4 × 132 × 1.98 GHz ≈ 1045 Giga-instructions/sec

This is the "instruction roof" — separate from the FLOPS roof. For sparse, control-heavy code, you hit this first.

**Latency hiding.** If an instruction has latency L (e.g., 20 cycles for an HBM load after L2 miss ~ 400 cycles), you need at least L warps ready to keep the scheduler fed. This is why occupancy matters: too few warps and the scheduler runs out of ready work during long memory operations.

### 3.3 Register file

Each SM has a 256 KB register file (64K × 32-bit registers). Registers are partitioned among resident threads. With max 2048 threads per SM, each thread gets up to 128 registers. Using more registers per thread reduces the number of warps that can be resident — the classic **register pressure vs occupancy** tradeoff.

---

## 4. Tensor Cores

Tensor Cores are the reason a GPU does 10 PFLOPS while its CUDA cores do ~60 TFLOPS. They're fixed-function matrix multiply-accumulate units.

### 4.1 The MMA operation

A Tensor Core performs `D = A·B + C` where A, B, C, D are small matrices. The exact tile size varies by generation and precision:

| Generation | Instruction | Shape (M×N×K) | Dtype |
|---|---|---|---|
| Volta | `HMMA.884` | 8×8×4 | FP16→FP32 |
| Turing | `HMMA.1688` | 16×8×8 | FP16→FP32 |
| Ampere | `HMMA.16816` | 16×8×16 | FP16/BF16→FP32 |
| Hopper | `wgmma` | 64×N×16 (N variable) | FP16/BF16/FP8 |
| Blackwell | `mma5` | larger still | FP16/BF16/FP8/FP6/FP4 |

On Hopper, a single **warpgroup matrix multiply accumulate** (`wgmma`) instruction involves 4 warps cooperating on a 64×N×16 tile, and it runs asynchronously — the warp can issue other work while the MMA executes.

### 4.2 Why tensor cores are ~8× denser than CUDA cores

A tensor core does a 16×16×16 MMA = 16·16·16 = **4096 fused multiply-adds in one instruction**, or 8192 FLOPs. A CUDA core does 2 FLOPs per cycle (FMA). Even with 128 CUDA cores per SM at 2 FLOPs each, that's 256 FLOPs/cycle vs 4 tensor cores × 8192/8 cycles ≈ 4096 FLOPs/cycle. Hence the ratio.

### 4.3 Using tensor cores in practice

You rarely write `wgmma` yourself. You use:

- **cuBLAS/cuBLASLt**: `cublasGemmEx`, `cublasLtMatmul` with `CUBLAS_COMPUTE_16F`
- **cuDNN**: for conv / attention
- **CUTLASS**: template library with composable tiles, epilogues, swizzles
- **Triton**: `tl.dot(a, b)` lowers to tensor core
- **PyTorch**: `torch.matmul` + AMP autocast

The key constraint for activating tensor cores is **dtype + alignment + shape multiples**:
- Dimensions must be multiples of 8 or 16 (precision-dependent)
- Strides must be aligned (128-byte alignment is ideal)
- Data must be in a supported dtype (FP16/BF16/FP8/TF32)

A mis-shaped tensor (e.g., hidden=4097) silently falls back to CUDA cores and loses ~10× throughput. This is a common bug.

---

## 5. Memory Subsystem

### 5.1 Hierarchy

```
Registers (per-thread, 256 KB/SM) — ~0 cycle, dedicated, not shared
   │
Shared memory / L1 (per-SM, 256 KB) — ~30 cycle, programmer-managed
   │
L2 cache (chip-wide, 50 MB on H100) — ~200 cycle, auto-managed
   │
HBM3/3e (device, 80–288 GB) — ~400 cycle, 3.3–12 TB/s
   │
PCIe Gen5 → Host DDR (~512 GB – 2 TB) — ~10,000 cycle, 64–128 GB/s
   │
NVMe / network — ms latency, GB/s bandwidth
```

### 5.2 Shared memory — the programmer's scratchpad

Shared memory is the most important resource in kernel programming. It's SRAM on the SM, physically identical to L1 cache, but visible to all threads in a block and manually managed.

- **Latency:** ~30 cycles — 10× faster than L2, 100× faster than HBM
- **Bandwidth:** ~12 TB/s aggregate per SM (thousands of TB/s chip-wide)
- **Size:** Configurable up to 228 KB per SM on Hopper (rest goes to L1)

Shared memory is organized into **32 banks** (on Hopper: 32 banks × 32 bits each = 128-byte line). Accesses to different banks by threads in a warp are served in parallel; accesses to the same bank serialize (a "bank conflict") and halve throughput per collision.

See [CUDA_Optimization](CUDA_Optimization.md) for bank conflict avoidance patterns (padding, XOR swizzle).

### 5.3 L2 cache

L2 is chip-wide, shared across all SMs. On H100 it's 50 MB (large for a GPU), split into two partitions (25 MB each, cross-connected). Hopper added **L2 residency control** — you can hint to the hardware that certain ranges should persist in L2, useful for small working sets that are reused across kernels (e.g., a frequently accessed lookup table).

### 5.4 HBM — the primary bottleneck for LLM inference

HBM (High Bandwidth Memory) is stacked DRAM on the GPU package, connected via a silicon interposer. On Blackwell Ultra, 8 HBM3e stacks at 8-hi give 288 GB total.

The bandwidth comes from **width**, not **clock speed**:
- 8 stacks × 1024 bits × 2 transfers/clock × 1.5 GHz ≈ 12 TB/s

HBM is the dominant constraint for decode. During decode, every token reads all model weights once. For a 70B FP16 model on an H100, that's 140 GB read per token — at 3.35 TB/s that's ~42 ms just for weight reads, ignoring everything else. This is the hard floor.

---

## 6. Execution Pipeline Deep Dive

### 6.1 Kernel launch

```c
kernel<<<grid, block, shmem, stream>>>(args...);
```

The launch:
1. Host posts a kernel-launch command packet to the command queue (via `cudaStream_t`)
2. The GPU's front-end fetches the packet
3. Blocks are dispatched to SMs by the **GigaThread Engine**, one block at a time, as SMs have capacity
4. Each block stays on its assigned SM until completion — **no migration**
5. Warps within the block are scheduled by the SM's warp schedulers

### 6.2 Block-to-SM scheduling

The GigaThread Engine assigns a block to any SM with enough free resources (registers, shared memory, warp slots). Once assigned, the block's shared memory allocation is reserved on that SM and cannot be used by other blocks until this block completes.

If your block requests 64 KB of shared memory, only 3 blocks fit on an SM with 228 KB (the remaining 36 KB is wasted). If your block uses 96 registers per thread with 256 threads, that's 24,576 registers — only 2.6 blocks fit (rounded to 2) even if shared memory is plentiful.

### 6.3 Warp scheduling within an SM

Each warp scheduler picks, per clock, one of its resident warps that has a ready instruction (operands available, functional unit free) and issues it. If no warp is ready, the scheduler stalls that cycle.

**Why occupancy matters.** If your SM has only 4 warps resident and all 4 are waiting on an HBM load (~400 cycles), the SM is idle for those 400 cycles. With 64 resident warps, there are usually several ready at any moment.

**When occupancy doesn't matter.** Kernels with very high ILP (e.g., FlashAttention v3) can hide latency with instruction-level parallelism at low occupancy. Empirically, once you exceed ~40% occupancy, further gains are minimal for compute-bound kernels.

---

## 7. Hopper and Blackwell Generational Additions

### 7.1 Hopper (H100) features

**FP8 tensor cores.** Native `fp8_e4m3` and `fp8_e5m2` support, 2× the throughput of FP16. E4M3 has 4 exponent bits, 3 mantissa bits; used for forward activations and weights. E5M2 has 5 exponent, 2 mantissa; used for gradients.

**Transformer Engine.** NVIDIA's Python library that auto-scales FP8 per-tensor based on runtime histograms, inserting the right scale factors to keep numerics in range.

**TMA (Tensor Memory Accelerator).** Bulk async memory copy engine. You say "copy this 128×128 tile from HBM to shared memory" and it handles addressing, swizzling, and bulk transfer, asynchronously — no wastegoing of warp cycles on address arithmetic.

**Thread Block Clusters.** Groups of up to 16 blocks that share distributed shared memory and can synchronize. Enables cooperative algorithms across SMs.

**DSMEM (Distributed Shared Memory).** Within a cluster, blocks can access each other's shared memory via the **SM-to-SM network**, avoiding a round-trip through HBM.

**Asynchronous transactions.** `cp.async.bulk`, `wgmma`, and TMA all execute asynchronously, overlapping with warp-level compute via barriers (`mbarrier`).

### 7.2 Blackwell (B200) features

**Dual-die package.** Two GPU dies connected by a 10 TB/s chip-to-chip (C2C) link, presented as one device with unified memory. The C2C bandwidth is comparable to an NVLink domain — die-to-die feels local.

**FP4 tensor cores.** Native `fp4_e2m1` support, 2× FP8 throughput. Works well for inference; less for training.

**Second-gen Transformer Engine.** Supports microscaling (per-block-of-32 scale factors) for FP8 and FP4.

**Confidential Compute.** Hardware-enforced isolation (attestation, memory encryption) for multi-tenant deployments.

**NVLink 5 and NVL72.** NVLink per-GPU doubled to 1.8 TB/s; NVSwitch domain extends to 72 GPUs in a single rack. Enables TP=72 — critical for trillion-parameter MoE.

### 7.3 Blackwell Ultra (B300)

Spec bump over B200: 288 GB HBM3e at 12 TB/s. FP4 throughput increased. Primary enabler of long-context (1M+ token) inference and large MoE serving on a single node.

---

## 8. Alternative AI Accelerators

The interview-worth summary of the competitive landscape:

| Accelerator | Architecture | Strengths | Weaknesses |
|---|---|---|---|
| **Google TPU v5p/Trillium** | Systolic array, bfloat16/int8/FP8 native, 3D torus interconnect | Excellent for transformer GEMMs, tight integration with JAX/XLA, lower $/FLOP | Limited to GCP, XLA-only, no CUDA ecosystem |
| **AMD MI300X / MI325X** | CDNA3, 192–288 GB HBM3e | More HBM than H100, ROCm/PyTorch native, price competitive | Software ecosystem still maturing (vLLM, SGLang supported, but fewer kernels) |
| **Intel Gaudi 3** | 128 GB HBM2e, deterministic RDMA fabric | Training + inference, good price/perf | Small software ecosystem, requires rewrite for custom kernels |
| **AWS Trainium 2 / Inferentia 2** | Custom NeuronCore, systolic | Cost-effective on AWS, Neuron SDK with PyTorch/JAX | Locked to AWS, fewer supported models |
| **Cerebras WSE-3** | Wafer-scale, 4 trillion transistors, 900 K cores, 44 GB on-chip SRAM | Eliminates memory wall for models that fit on-chip; extremely fast decode | Limited capacity, expensive, unusual programming model |
| **Groq LPU** | SRAM-only (no HBM), deterministic scheduling | Ultra-low TTFT and ITL, predictable latency | Small per-chip memory, needs many chips for large models |
| **SambaNova SN40L** | Reconfigurable dataflow, 3-tier memory | Flexible, handles sparse ops | Niche ecosystem |

The practical implication for infra engineers: production deployments standardize on NVIDIA (80%+ market share in AI training as of 2026), with AMD MI300 growing as the "dual-vendor" hedge. Specialty accelerators like Groq and Cerebras are used for specific latency SLAs.

---

## 9. Power, Cooling, and Data-Center Implications

**Power.** An H100 SXM draws 700 W TDP. A DGX H100 (8 GPUs) draws ~10.2 kW. A rack of 4 DGX systems = ~41 kW. Blackwell B200 draws 1000 W; B300 draws 1200 W; an NVL72 rack can draw 120+ kW.

For comparison, a normal server rack runs 5–15 kW. AI infra requires:
- **Liquid cooling** (direct-to-chip or rear-door heat exchangers) above ~30 kW/rack
- **Higher-voltage power distribution** (48 VDC busbars becoming standard)
- **Dense electrical feeds** (redundant 3-phase 415/480 V)
- **New facility designs** (AI factories like OCI Gen2, Crusoe, CoreWeave)

**Why this matters for you as an engineer.** Hardware selection isn't just FLOPS/$ — it's FLOPS/watt and FLOPS/rack-U. A site with 20 MW of power capacity can deploy ~130 B200 racks. Capacity planning is constrained by grid, not by cash.

---

## 10. GPU Lifecycle: Acquisition, Utilization, Failure

**Supply.** NVIDIA allocates GPUs via NVIDIA-certified OEMs (Dell, HPE, Supermicro) and cloud providers (AWS, Azure, GCP, Oracle, CoreWeave). Lead times in 2024-2025 were 6+ months; by 2026 this has normalized to ~2 months for Blackwell.

**Utilization.** In production training clusters, sustained GPU utilization is 30–50% (due to dataloading, synchronization, checkpointing, and failures). Inference clusters can sustain 60–80%. Anything below 20% signals a problem.

**Failures.** GPU MTBF at cluster scale is ~1 failure per 1000 GPU-days. For a 10K-GPU cluster, that's ~10 failures per day. Production systems must handle:
- Uncorrectable ECC errors (GPU drops offline)
- XID errors (driver-level GPU hang)
- NVLink flaps (degraded bandwidth mid-run)
- SMBus / fan failures (thermal throttling)
- Silent data corruption (SDC) — rare but devastating; detected by bit-wise replay on known batches

See [Observability_and_Debugging](Observability_and_Debugging.md) for monitoring patterns.

---

## 11. Worked Example: Capacity Math for Llama-3-70B on an H100 Node

Here's a worked example that appears in interviews constantly.

**Setup:** 8× H100 SXM (80 GB each = 640 GB HBM total), NVLink, Llama-3-70B.

**Model memory:**
- FP16 weights: 140 GB → fits on 2 GPUs but need multi-GPU for performance
- With TP=8: each GPU holds ~17.5 GB of weights
- Optimizer/ops state (inference only): ~4 GB of activations, workspace, NCCL buffers per GPU

**Per-GPU budget:**
- 80 GB HBM
- − 17.5 GB weights
- − 4 GB workspace
- − 2 GB NCCL + CUDA runtime
- = **56.5 GB remaining for KV cache**

**KV cache math for Llama-3-70B** (80 layers, 8 KV heads, 128 head dim, GQA):
- Per-token per-GPU (TP=8 shards KV heads): 2 × 80 × (8/8) × 128 × 2 bytes = **40 KB/token**
- Wait — TP in Llama shards query heads but not KV heads (since there are only 8 KV heads and 8 GPUs). Each GPU holds one KV head.
- 56.5 GB / 40 KB = ~1.4 M tokens total per-GPU → **1.4 M total tokens** across the batch

So an 8×H100 node at TP=8 can hold ~1.4 M tokens of KV cache for Llama-3-70B. At 4K context per request, that's ~350 concurrent requests. At 128K context, ~11 concurrent requests.

**Decode throughput:**
- Memory-bound: each decode step reads 140 GB of weights / 8 GPUs = 17.5 GB per GPU per step
- At 3.35 TB/s: 17.5 / 3350 ≈ **5.2 ms per step**
- But: KV cache reads add proportionally to context length. At 4K context: ~2.5 ms extra. Total ~8 ms/step → ~125 tokens/sec per request
- With batching (128 concurrent decodes at same step): ~16,000 tokens/sec aggregate

These are back-of-the-envelope numbers that interviewers expect you to recite.

---

## 12. Common Interview Questions on GPU Architecture

**Q: Why is HBM bandwidth, not FLOPS, the bottleneck for decode?**
A: Decode is a matrix-vector multiply per layer — arithmetic intensity of 2 ops/byte (read weight, compute 2 FLOPs: multiply + add). Even with batch=64, intensity is ~128 ops/byte, below the H100 ridge point of ~295. The operation is fundamentally memory-bound, so performance scales with bandwidth, not FLOPS.

**Q: What is SIMT and how is it different from SIMD?**
A: SIMT (Single Instruction Multiple Thread) executes one instruction across 32 threads in a warp, but each thread has its own program counter and registers. SIMD (Single Instruction Multiple Data) requires the programmer to pack multiple data elements into wide registers (e.g., AVX-512). SIMT masks off inactive threads to handle divergent control flow, while SIMD cannot diverge. The tradeoff: SIMT loses throughput on divergence; SIMD forces the programmer to handle it manually.

**Q: What happens when a warp diverges?**
A: The warp executes each path serially, with threads active only on their own path (via a mask). Latency is the sum of both branch latencies, throughput is halved. With nested divergence, it compounds. Modern GPUs (Volta+) support independent thread scheduling, but the throughput cost remains.

**Q: What is a tensor core's MMA tile on Hopper?**
A: `wgmma` — a warpgroup (4 warps, 128 threads) cooperating on a 64×N×16 MMA, where N is 8/16/32/64/128/256. It's asynchronous, so the warpgroup can issue other work while the MMA runs.

**Q: Why does Hopper's TMA matter?**
A: It moves bulk async copies out of the warp — the warp issues a `cp.async.bulk` and waits on an `mbarrier`, freeing all 128 threads to compute. On Ampere, the same copy used ~128 threads' worth of address arithmetic per tile. TMA is why FlashAttention v3 gets close to peak FLOPS.

**Q: How big is L2 cache on H100, and how much does it help for LLM inference?**
A: 50 MB. For Llama-70B with 140 GB weights, you cannot fit the model in L2. But L2 helps with KV cache locality across consecutive decode steps — recently accessed K/V tiles can still be in L2 and avoid HBM round-trips.

**Q: Why do GPUs have shared memory but CPUs don't?**
A: Shared memory is a programmer-managed scratchpad. CPUs rely on hardware-managed caches that adapt automatically. On a GPU, you know exactly what data is reused across threads in a block (e.g., a K-tile in attention is used by all query rows), and manually staging it in shared memory gives you deterministic ~30-cycle access vs 200-cycle L2 access. The tradeoff: more programming complexity for 5-10× performance.

**Q: What's the difference between L1 and shared memory?**
A: On Hopper they are physically the same 256 KB SRAM, configurable via `cudaFuncSetAttribute` into (0 KB shared, 228 KB L1) or (228 KB shared, 28 KB L1) and points in between. L1 is a hardware-managed cache; shared memory is programmer-managed. Tensor kernels typically max out shared memory; memory-indirect kernels prefer L1.

**Q: If I want to write a fused attention kernel, what's the SM-level architecture I need to exploit?**
A: (1) Load Q, K, V tiles into shared memory via TMA / cp.async. (2) Use warpgroup `wgmma` to compute attention scores `S = Q·K^T` inside SRAM. (3) Do online softmax using warp-level reductions (shuffle intrinsics). (4) Use `wgmma` again for `O = P·V`. (5) Write O back to HBM via TMA store. All of this overlaps using `mbarrier` barriers — the "producer" warps load data while "consumer" warps compute. See [FlashAttention_Deep_Dive](FlashAttention_Deep_Dive.md) for the full algorithm.

---

## 13. Further Reading

- NVIDIA Hopper Architecture Whitepaper
- NVIDIA Blackwell Architecture Whitepaper
- "Dissecting the NVIDIA Hopper Architecture through Microbenchmarking" (Luo et al., arXiv 2402.13499)
- NVIDIA CUDA C Programming Guide (current version)
- NVIDIA PTX ISA Reference
- "Demystifying Tensor Cores to Optimize Half-Precision Matrix Multiply" (Raihan et al., HPCA 2019)

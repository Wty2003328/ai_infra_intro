# L5 — Kernels and Programming: Interview Questions

Consolidated interview Q&A and worked problems from every page in `L5_Kernels_and_Programming/`, moved here to keep topic pages focused on concepts. Each section links back to its source page for full context.

---

## CUDA Optimization — From Working Kernel to Peak Throughput

*From [CUDA_Optimization.md](../L5_Kernels_and_Programming/02_CUDA_Optimization.md)*

### Problem 1: Coalescing diagnosis

**Question.** A kernel has threads reading `data[tid * 3]` where `data` is `float*` and `tid` is `threadIdx.x`. The kernel achieves 200 GB/s on an A100. Diagnose and fix.

**Solution.**

Stride-3 access means thread 0 reads byte 0, thread 1 reads byte 12, thread 2 reads byte 24, etc. A warp's 32 threads span bytes 0 through $31 \times 12 = 372$. This requires $\lceil 372 / 128 \rceil = 3$ cache-line transactions per warp instead of the ideal 1. Efficiency = 33%.

Fix: convert from AoS to SoA. Instead of `struct { float x, y, z; } data[N]`, use three arrays `float data_x[N], data_y[N], data_z[N]`. Now `data_x[tid]` is stride-1, fully coalesced (1 transaction per warp).

Expected improvement: from 200 GB/s to ~1 800 GB/s (9x).

### Problem 2: Occupancy calculation

**Question.** A kernel uses 128 registers per thread, 48 KB of shared memory per block, and 256 threads per block. How many blocks can run concurrently on a Hopper SM? What is the occupancy?

**Solution.**

Three limiters:

1. **Registers.** $W_{\text{reg}} = \lfloor 65\,536 / (128 \cdot 32) \rfloor = \lfloor 65\,536 / 4\,096 \rfloor = 16$ warps per SM. At 256 threads/block = 8 warps/block: $B_{\text{reg}} = \lfloor 16 / 8 \rfloor = 2$ blocks.

2. **Shared memory.** $W_{\text{smem}} = \lfloor 228\,608 / 49\,152 \rfloor = \lfloor 4.65 \rfloor = 4$ blocks (48 KB = 49 152 bytes, SM budget 228 KB usable). $B_{\text{smem}} = 4$ blocks.

3. **Threads.** Max 2 048 threads / SM. At 256/block: $B_{\text{threads}} = 8$ blocks.

The register limiter binds: $B = \min(2, 4, 8) = 2$ blocks. Occupancy = $2 \cdot 8 / 64 = 16 / 64 = 25\%$.

To improve: reduce registers to 64 with `-maxrregcount=64`, which gives $W_{\text{reg}} = 32$ warps, $B_{\text{reg}} = 4$ blocks. New occupancy = $4 \cdot 8 / 64 = 50\%$. The cost is potential register spills; profile to confirm net gain.

### Problem 3: Tile sizing for matmul

**Question.** Design the SMEM tile sizes for an FP16 matmul kernel targeting Hopper's wgmma. The kernel should use 2 SMEM buffers (double buffering) for both A and B tiles. Constraints: stay under 200 KB SMEM per SM, target 4 blocks/SM.

**Solution.**

Available SMEM per block: $200\,000 / 4 = 50\,000$ bytes $\approx 48$ KB.

Four buffers total (2 for A, 2 for B): each buffer gets $\approx 12$ KB.

For wgmma with $M = 64$, $K = 16$ (FP16): tile A is $64 \times 16 = 1\,024$ elements = 2 KB (FP16). Tile B is $16 \times N$ elements.

Buffer B size = 12 KB = 12 288 bytes = 6 144 FP16 elements. So $N = 6\,144 / 16 = 384$. But wgmma max $N = 256$. So tile B = $16 \times 256 = 4$ KB per buffer.

Total per block: $2 \times 2\text{KB} + 2 \times 4\text{KB} = 12$ KB per block. Four blocks = 48 KB. Well under the 200 KB limit.

This is the CUTLASS 3.x default tiling for Hopper FP16: $64 \times 256 \times 16$ with double buffering.

### Problem 4: Bank conflict analysis

**Question.** A kernel has `__shared__ float tile[32][32];`. Each thread in a warp reads `tile[threadIdx.x][col]` for a fixed `col`. Analyze the bank conflicts.

**Solution.**

SMEM bank assignment: word at address $a$ is in bank $a \mod 32$. For `tile[row][col]`, the address of element `tile[r][c]` is $(r \cdot 32 + c) \cdot 4$ bytes. Bank = $(r \cdot 32 + c) \mod 32 = c \mod 32 = c$ (since $c < 32$).

Wait — the question says each thread reads `tile[threadIdx.x][col]`, i.e., thread $t$ reads row $t$, column `col`. Since column is fixed, all 32 threads access the same column index, meaning they all access different rows but the same column. The address for thread $t$ is $(t \cdot 32 + col) \cdot 4$, so bank = $(t \cdot 32 + col) \mod 32 = col \mod 32 = col$.

All 32 threads hit the **same bank** (bank $= col$). This is a **32-way bank conflict**, serializing into 32 separate accesses. The single read value is broadcast, so only 1 cycle is actually needed (broadcast exception applies — all threads read the same address). Wait, no: each thread reads a *different* address (`tile[0][col]`, `tile[1][col]`, ...) but they all land in the same bank. The broadcast exception only applies when multiple threads read the *same address*. Here they read different addresses in the same bank, so it is a genuine 32-way conflict.

Fix: pad the row dimension. `__shared__ float tile[32][33];` changes stride to 33. Now thread $t$ accesses bank $(t \cdot 33 + col) \mod 32 = (t + col) \mod 32$, which is distinct for each $t$. Zero-way conflict.

### Problem 5: Double-buffering speedup estimate

**Question.** A tiled matmul kernel has compute time $T_c = 500$ ns per tile and load time $T_l = 600$ ns per tile (HBM round-trip). Without double buffering, total time for $K$ tiles is $K \cdot (T_c + T_l)$. With double buffering, estimate the speedup.

**Solution.**

Without double buffering: $T_{\text{total}} = K \cdot (500 + 600) = 1100K$ ns.

With double buffering: the first tile loads in $T_l = 600$ ns. Subsequently, load and compute overlap. Since $T_l > T_c$ (memory-bound regime), the load dominates: $T_{\text{total}} = T_l + K \cdot \max(T_c, T_l) = 600 + K \cdot 600 = 600(K + 1)$ ns.

Speedup $= \frac{1100K}{600(K+1)}$. For large $K$: speedup $\to 1100 / 600 = 1.83\times$.

If the kernel is compute-bound ($T_c > T_l$), the speedup converges to $(T_c + T_l) / T_c$. In the balanced case ($T_c = T_l$), speedup = 2x (ideal overlap).

---

## CUDA Programming — The GPU Kernel Model

*From [CUDA_Programming.md](../L5_Kernels_and_Programming/01_CUDA_Programming.md)*

### Problem 1: Occupancy calculation

**Question.** A kernel uses 64 registers per thread and launches blocks of 256 threads. What is the occupancy on Hopper? Is it sufficient for a memory-bound kernel?

**Derivation.** Per-SM register file: 65 536 registers. Threads per block: 256. Registers per thread: 64. Registers needed per block: $256 \times 64 = 16\,384$. Blocks per SM limited by registers: $\lfloor 65\,536 / 16\,384 \rfloor = 4$. Blocks per SM limited by threads: $\lfloor 2048 / 256 \rfloor = 8$. Register limit is binding: 4 blocks. Active warps: $4 \times 256 / 32 = 32$. Occupancy: $32 / 64 = 50\%$.

For a memory-bound kernel, 50% occupancy provides 32 active warps. The rule of thumb for hiding ~400-cycle HBM latency is $W_{\min} \approx 80$ warps (latency / issue interval). At 32 warps, some memory latency stalls will occur. Either reduce register pressure (recompile with `-maxrregcount 48` to get 5 blocks/SM = 40 warps) or accept the stalls if the kernel is actually compute-bound with sufficient ILP.

### Problem 2: Shared memory capacity and occupancy

**Question.** A kernel allocates 64 KB of shared memory per block. How many blocks fit per SM on Hopper? What is the maximum occupancy?

**Derivation.** Total SMEM per SM: 228 KB. Must call `cudaFuncSetAttribute` to raise the per-block limit above 48 KB. Blocks per SM limited by SMEM: $\lfloor 228 / 64 \rfloor = 3$ (with some overhead for the SMEM carveout). At 3 blocks with 256 threads each: $3 \times 256 = 768$ threads = 24 warps. Occupancy: $24 / 64 = 37.5\%$.

This is the classic SMEM-occupancy trade-off. If the kernel is compute-bound with heavy SMEM reuse (e.g., tiled matmul), 37.5% occupancy may be acceptable because the SMEM tiling raises arithmetic intensity well above the ridge point. If memory-bound, the low occupancy hurts latency hiding.

### Problem 3: Coalescing analysis

**Question.** A warp of 32 threads executes `float x = data[threadIdx.x * 33]` where `data` is a `float*` in global memory. How many memory transactions does this require? What is the effective bandwidth utilization?

**Derivation.** Thread $i$ reads address $\text{base} + 4 \times (33i)$ bytes. The 32 addresses span from offset 0 to $4 \times 33 \times 31 = 4092$ bytes. CUDA memory transactions are 128-byte aligned segments. The span of 4092 bytes covers $4092 / 128 = 31.97$ segments, so **32 segments** are touched. However, because stride-33 means threads hit addresses $0, 132, 264, \ldots$ and these are not aligned to the same 128-byte segment boundaries, each thread likely requires its own transaction.

More precisely: thread $i$ reads byte address $132i$. Segment containing byte $132i$ is $\lfloor 132i / 128 \rfloor$. For $i = 0, 1, 2, \ldots$: segments $0, 1, 2, \ldots$. Each thread touches a unique segment. Result: **32 transactions**, meaning effective bandwidth is $1/32$ of peak. This is the worst case for non-overlapping addresses.

Fix: restructure to stride-1 access, or use shared memory as a transpose buffer.

### Problem 4: Warp-level reduction

**Question.** Implement a complete block-level reduction that uses warp shuffles for the final stage. Assume block size is 1024 threads.

**Derivation.** Strategy: (1) reduce in shared memory from 1024 to 32 values, (2) use warp shuffle to reduce 32 to 1.

```c
__global__ void block_reduce(const float *input, float *output, int n) {
    __shared__ float smem[1024];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    smem[tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();

    // Shared memory tree: 1024 -> 512 -> 256 -> 128 -> 64 -> 32
    for (int s = 512; s > 32; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    // Final 32 elements reduced via warp shuffle (no SMEM sync needed)
    float val = (tid < 32) ? smem[tid] : 0.0f;
    if (tid < 32) {
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
        if (tid == 0) atomicAdd(output, val);
    }
}
```

The warp-shuffle stage eliminates 5 `__syncthreads()` calls (for $s = 32, 16, 8, 4, 2$), each costing ~20–100 cycles. Net savings: ~100–500 cycles per block.

### Problem 5: Kernel timing and roofline analysis

**Question.** A kernel processes a $4096 \times 4096$ matrix transpose using shared memory tiling. The kernel takes 1.2 ms on H100 (HBM BW = 3.35 TB/s). Is it bandwidth-bound? What is the achieved bandwidth?

**Derivation.** Matrix size: $4096 \times 4096 \times 4$ bytes (FP32) = 64 MB. Transpose reads 64 MB and writes 64 MB: total data movement $Q = 128$ MB = $1.28 \times 10^8$ bytes. With shared memory tiling, the kernel avoids the naive 32x coalescing penalty on the write path.

Achieved bandwidth: $Q / t = 1.28 \times 10^8 / (1.2 \times 10^{-3}) = 106.7$ GB/s. Peak HBM: 3350 GB/s. Utilization: $106.7 / 3350 = 3.2\%$.

Is this bandwidth-bound? The transpose does $\sim 16.8$M FLOPs (index arithmetic only — a transpose is not a FLOP-heavy operation). Arithmetic intensity: $\sim 0.13$ FLOP/B — well below the H100 ridge of 295 FLOP/B. Yes, firmly **memory-bound**. The question is why utilization is only 3.2%.

Likely causes: (a) the transpose has no data reuse beyond one tile, so it is fundamentally limited by how fast the kernel can issue loads and stores; (b) shared memory bank conflicts on the transpose path; (c) suboptimal tile size. Profiling with Nsight Compute would distinguish these.

---

## Cutting-Edge Kernels — The 2025-2026 Kernel Programming Frontier

*From [Cutting_Edge_Kernels.md](../L5_Kernels_and_Programming/06_Cutting_Edge_Kernels.md)*

### Problem 1: Arithmetic Intensity and Kernel Choice

**Question:** An attention kernel has sequence length $S = 8192$, head dimension $D = 128$, batch $B = 1$, $H = 32$. The attention score matrix is $B \times H \times S \times S$. What is the arithmetic intensity of this attention layer, and is it compute-bound or memory-bound on H100?

**Solution:**

FLOPs for attention (QK + softmax + PV):

$$\text{FLOPs} = 2 \cdot B \cdot H \cdot S^2 \cdot D + 3 \cdot B \cdot H \cdot S^2 = 2 \cdot 1 \cdot 32 \cdot 8192^2 \cdot 128 + 3 \cdot 1 \cdot 32 \cdot 8192^2$$

$$= 2 \cdot 32 \cdot 67108864 \cdot 128 + 3 \cdot 32 \cdot 67108864 \approx 549.8 \times 10^9 + 6.4 \times 10^9 \approx 556 \text{ GFLOPS}$$

Bytes transferred (read Q, K, V + write O):

$$\text{Bytes} = 4 \cdot B \cdot H \cdot S \cdot D \cdot 2 + B \cdot H \cdot S \cdot D \cdot 2 = 10 \cdot B \cdot H \cdot S \cdot D \cdot 2$$

$$= 10 \cdot 1 \cdot 32 \cdot 8192 \cdot 128 \cdot 2 = 671 \text{ MB}$$

Arithmetic intensity:

$$\text{AI} = \frac{556 \times 10^9}{671 \times 10^6} \approx 829 \text{ FLOPS/byte}$$

H100 ridge point (FP16): $\frac{990 \text{ TFLOPS}}{3.35 \text{ TB/s}} \approx 295 \text{ FLOPS/byte}$

Since $829 > 295$, this attention layer is **compute-bound**. FlashAttention tiling is the correct approach (online softmax avoids materializing the $S \times S$ matrix). On H100 with FA-v3, expect ~75% of 990 TFLOPS = ~742 TFLOPS.

---

### Problem 2: Shared Memory Budget for a GEMM Tile

**Question:** A GEMM kernel on Hopper uses BM = BN = 128, BK = 64 with FP16 data. The kernel double-buffers A and B tiles. How much shared memory does each threadblock use? Can 3 threadblocks coexist on one SM?

**Solution:**

Each A tile: $BM \times BK \times 2 \text{ bytes} = 128 \times 64 \times 2 = 16{,}384 \text{ bytes} = 16 \text{ KB}$

Each B tile: $BK \times BN \times 2 \text{ bytes} = 64 \times 128 \times 2 = 16{,}384 \text{ bytes} = 16 \text{ KB}$

Double-buffered: $2 \times (16 + 16) = 64 \text{ KB}$ per threadblock for data tiles.

Add overhead: mbarrier state (~256 bytes), accumulator spill area (~4 KB), miscellaneous (~2 KB). Total per block: $\approx 70 \text{ KB}$.

Hopper shared memory per SM: 228 KB (configurable, max shared).

$$\text{Blocks per SM} = \left\lfloor \frac{228}{70} \right\rfloor = 3$$

Three threadblocks fit in 210 KB, leaving 18 KB for the rest. This is tight but feasible. With 132 SMs on H100, that gives $3 \times 132 = 396$ concurrent threadblocks, sufficient for large matrices. Register pressure from wgmma (64 accumulator registers per warpgroup) is the other constraint; verify with `--ptxas-options=-v`.

---

### Problem 3: Fused Linear + Cross-Entropy Memory Savings

**Question:** A Llama-3-70B training run uses batch size $B = 4$, sequence length $S = 8192$, vocabulary $V = 128256$. Compute the memory saved by Liger's fused linear + cross-entropy kernel vs the standard unfused path.

**Solution:**

Standard path materializes the logits tensor in FP32 (needed for stable softmax):

$$\text{Logits size} = B \times S \times V \times 4 \text{ bytes} = 4 \times 8192 \times 128256 \times 4$$

$$= 4 \times 8192 \times 128256 \times 4 = 16{,}826{,}705{,}920 \text{ bytes} \approx 15.7 \text{ GB}$$

The backward pass additionally materializes the log-probabilities tensor:

$$\text{Log-probs size} = B \times S \times V \times 4 \text{ bytes} \approx 15.7 \text{ GB}$$

Total peak: $\approx 31.4 \text{ GB}$ of intermediate tensors that exist simultaneously during the backward pass.

Fused path: the kernel streams over the vocabulary dimension in shared memory tiles. Peak intermediate storage is one tile of size $(B \times S \times \text{BLOCK\_V})$ where $\text{BLOCK\_V} \approx 128$. This is:

$$\text{Tile size} = 4 \times 8192 \times 128 \times 4 = 16{,}777{,}216 \text{ bytes} \approx 16 \text{ MB}$$

Memory saved: $31.4 \text{ GB} - 16 \text{ MB} \approx 31.4 \text{ GB}$.

This is the difference between fitting and not fitting on a single H100 (80 GB) at these batch sizes, especially when combined with model weights, gradients, and optimizer states.

---

### Problem 4: Choosing Between Kernel Abstraction Levels

**Question:** A team needs a fused kernel that computes: (1) a GEMM with FP8 inputs and FP32 accumulation, (2) applies RMSNorm to the output rows, (3) quantizes the result back to FP8 with per-row scaling. This runs on Hopper in the training critical path. Which kernel authoring tool should they use and why?

**Solution:**

Analysis of each level:

**Triton**: Can express all three operations but has limited FP8 support (matmul works, but fine-grained FP8 scaling control is incomplete). Cannot fuse the epilogue (RMSNorm + quant) into the GEMM mainloop efficiently. Would require 2-3 separate kernel launches or a complex custom implementation. Estimated 80-85% of peak.

**CUTLASS-Python / CuteDSL**: Provides CUTLASS epilogue fusion with custom operations. FP8 GEMM with per-row scaling is a built-in epilogue variant. RMSNorm can be added as a custom epilogue functor. The entire operation fuses into a single kernel: TMA load → FP8 wgmma → RMSNorm epilogue → FP8 quant + store. Estimated 95-98% of peak.

**CUTLASS C++ + CuTe**: Same as CUTLASS-Python but with full control over every detail. Only needed if CuteDSL cannot express the specific epilogue. Same estimated performance but 3-5x more development time.

**TileLang**: Could express this with TMA + wgmma + custom post-processing, but the library is newer and has fewer pre-built epilogue templates for FP8 quant.

**Recommendation**: CUTLASS-Python (CuteDSL). It provides the FP8 GEMM with custom epilogue fusion needed, avoids the complexity of C++ template metaprogramming, and targets 95%+ of peak. Triton would work but leaves 10-15% on the table due to incomplete FP8 epilogue fusion.

---

### Problem 5: DeepEP vs NCCL All-to-All Analysis

**Question:** An MoE model with 64 experts uses expert parallelism across 8 GPUs on a single H100 node (NVLink 4, 900 GB/s bidirectional). Each GPU sends approximately 2 MB of tokens to each of the other 7 GPUs (after top-2 routing). Compare NCCL all-to-all vs DeepEP.

**Solution:**

**Total data movement per GPU:** $7 \times 2 \text{ MB} = 14 \text{ MB}$ in, $14 \text{ MB}$ out.

**NCCL all-to-all**: Per-message overhead dominates. NCCL launches a channel-based algorithm with multiple steps per peer. For 7 peers x 2 MB each, the startup overhead per message (~5-10 $\mu$s on NVLink) is significant relative to transfer time:

$$\text{Transfer time per message} = \frac{2 \text{ MB}}{900 \text{ GB/s}} \approx 2.2 \mu s$$

$$\text{Startup overhead} \approx 8 \mu s$$

Total: $7 \times (2.2 + 8) \approx 71 \mu s$ per all-to-all. Effective bandwidth: $\frac{14 \text{ MB}}{71 \mu s} \approx 197 \text{ GB/s}$, which is $\frac{197}{900} \approx 22\%$ of NVLink capacity.

**DeepEP**: Persistent kernel eliminates per-message startup. Fused gating + dispatch packs and sends in one operation. Direct NVLink memcpy bypasses NCCL protocol overhead.

$$\text{Transfer time} = \frac{14 \text{ MB}}{900 \text{ GB/s}} \approx 15.6 \mu s$$

$$\text{Overhead} \approx 3-5 \mu s \text{ (persistent kernel dispatch)}$$

Total: $\approx 20 \mu s$. Effective bandwidth: $\frac{14 \text{ MB}}{20 \mu s} \approx 700 \text{ GB/s}$, which is $\approx 78\%$ of NVLink capacity.

**Speedup**: $\frac{71}{20} \approx 3.5\times$. For larger expert counts (256 experts, more peers), the speedup grows to 5-10x because NCCL overhead scales with peer count while DeepEP overhead is nearly constant.

---

## FlashAttention Deep Dive — IO-Aware Attention from First Principles

*From [FlashAttention_Deep_Dive.md](../L5_Kernels_and_Programming/05_FlashAttention_Deep_Dive.md)*

### Problem 1: Arithmetic Intensity Comparison

**Q:** Training with $N=8192$, $d=256$, FP16 on H100 SXM. Is naive attention compute- or memory-bound? What about FlashAttention with 228 KB SRAM?

**A:** Naive: $AI \approx d/2 = 128 < 295$ (ridge) $\Rightarrow$ memory-bound. FlashAttention: $AI \approx M/d = 228 \times 1024 / 256 = 912 \gg 295 \Rightarrow$ compute-bound. FA achieves $912/295 \approx 3.1\times$ the throughput ceiling of naive.

### Problem 2: Tile Budget Verification

**Q:** $d=64$, FP16, 164 KB SRAM/block (A100, 2 blocks/SM). Maximum $B_r, B_c$ (powers of 2)?

**A:** Try $B_r=128, B_c=256$: $Q(16\text{K}) + K(32\text{K}) + V(32\text{K}) + S(65.5\text{K}) + O(16\text{K}) + m{+}\ell(512) = 164{,}352$ bytes $= 160.5$ KB $\leq 164$ KB. Fits. $B_r=256, B_c=256$ fails ($S$ alone = 131 KB + 98 KB > 229 KB). Answer: $B_r=128, B_c=256$.

### Problem 3: HBM Traffic Savings

**Q:** Derive the FA-to-naive traffic ratio for $N=2048$, $d=128$, $B_r=64$, $B_c=128$, $M=164$ KB.

**A:** Naive: $\text{HBM} = 8N^2 + 8Nd = 33.55\text{M} + 2.10\text{M} = 35.65$ MB. FA concrete traffic ($T_r = N/B_r = 32$): $Q$ reads + $O$ writes $= 2 \times 2Nd = 2.10$ MB; $K{+}V$ re-reads $= 2 \times T_r \times 2Nd = 32.0$ MB; total $\approx 34.1$ MB. Raw ratio: $35.65/34.1 \approx 1.05$x (comparable when $B_r \approx d$). With L2 caching of $K, V$ ($2Nd = 0.5$ MB, fits in L2): FA effective traffic $\approx 4Nd = 2.10$ MB, giving $35.65/2.10 \approx$ **17x less traffic**.

### Problem 4: FlashDecoding Split Count

**Q:** Decode with $N_{kv}=65536$, $d=128$, batch=1, H100 (132 SMs). How many KV splits?

**A:** $\text{Splits} = \min(\lfloor N_{kv}/B_c \rfloor, 4 \times \text{SMs}) = \min(\lfloor 65536/128 \rfloor, 528) = \min(512, 528) = 512$. Each SM processes ~4 blocks, high occupancy. Final reduction over 512 partials is lightweight (single warp).

### Problem 5: FP8 Throughput

**Q:** FA v3 on H100 uses FP8 E4M3. Theoretical max and achieved TFLOPS?

**A:** H100 FP8 dense peak = 1,979 TFLOPS. v3 achieves 75%: $0.75 \times 1{,}979 = 1{,}484$ TFLOPS — 1.5x the entire FP16 peak of the chip.

---

## ML Compilers — torch.compile, XLA, MLIR, and the Codegen Stack

*From [ML_Compilers.md](../L5_Kernels_and_Programming/07_ML_Compilers.md)*

- **"What does torch.compile do?"** → Dynamo captures bytecode into FX graphs with guards; AOTAutograd traces joint fwd/bwd and decomposes to a core op set; Inductor fuses and emits Triton/C++; optional CUDA-graph wrapping. Falls back per-fragment via graph breaks.
- **"Why is XLA fine without hand kernels but PyTorch isn't?"** → TPU: one vendor, regular systolic hardware, whole-program AOT with layout assignment; XLA fusion ≈ the kernel. GPU: irregular SKU zoo + peak tensor-core kernels (attention/GEMM) still human-won; compiler handles the glue.
- **"Compiler vs FlashAttention?"** → fusion can't change memory complexity; online-softmax tiling is an algorithmic identity a compiler won't discover. Compile the glue, hand-write the hot loop.
- **"Why does vLLM pad batches?"** → CUDA graphs need static shapes; padding to captured buckets trades ≤ one bucket of wasted compute for ms-scale launch savings.

---

## Triton and Custom Kernels — High-Performance GPU Programming in Python

*From [Triton_and_Kernels.md](../L5_Kernels_and_Programming/04_Triton_and_Kernels.md)*

**Q1.** *A Triton matmul kernel uses BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, num_stages=3 with FP16 inputs. Calculate the shared memory per block. Is this feasible on SM90 (227 KB)?*

Per stage: $S_{stage} = (128 \times 64 + 64 \times 128) \times 2 = 32{,}768$ bytes. Total: $3 \times 32{,}768 = 98{,}304$ bytes $\approx 96$ KB. The FP32 accumulator ($128 \times 128 \times 4 = 64$ KB) lives in registers, not shared memory. 96 KB is well within 227 KB; remaining budget allows up to $\lfloor 131{,}072 / 32{,}768 \rfloor = 4$ more stages. In practice, 4 stages total is optimal — beyond that, register pressure reduces occupancy.

**Q2.** *A softmax kernel processes rows of length N=4096 with BLOCK_SIZE=4096. How many program instances are needed for a batch of B=128 sequences, each with H=32 attention heads?*

Each row is handled by one program instance (BLOCK_SIZE = N). Total rows: $B \times H = 128 \times 32 = 4096$. Total program instances: 4096. Each instance performs 1 `tl.load` (the row) and 1 `tl.store` (the output), plus internal reductions (`tl.max`, `tl.sum`) that operate in registers/shared memory without additional global memory traffic.

If BLOCK_SIZE = 2048 instead, each row requires $\lceil 4096/2048 \rceil = 2$ blocks with a cross-block reduction, doubling the number of instances to 8192 and requiring additional synchronization — demonstrating why setting BLOCK_SIZE = N is the idiomatic approach.

**Q3.** *A Triton FlashAttention kernel processes Q=[B, H, M, D] and K,V=[B, H, N, D] with BLOCK_M=64, BLOCK_N=64, D=128. For B=1, H=1, M=N=8192, how many tl.dot calls does each program instance make?*

FlashAttention uses a 1D grid over query tiles: grid = $(\lceil 8192 / 64 \rceil,) = (128,)$. Each program instance iterates over $\lceil 8192 / 64 \rceil = 128$ key/value tiles. Per iteration: one `tl.dot(Q, K^T)` for attention scores and one `tl.dot(scores, V)` for value accumulation. Total per instance: $128 \times 2 = 256$ `tl.dot` calls. Total across all instances: $128 \times 256 = 32{,}768$.

Verification: each `tl.dot` performs $2 \times 64 \times 64 \times 128 = 1{,}048{,}576$ FLOPs (multiply-add). Total FLOPs: $32{,}768 \times 1{,}048{,}576 = 34{,}359{,}738{,}368 \approx 34.4$ GFLOPs. Theoretical attention compute: $4 \times N^2 \times D = 4 \times 8192^2 \times 128 = 34.4$ GFLOPs. Matches.

**Q4.** *A 65,536-GPU cluster trains an MoE model with 256 experts and 58 MoE layers. Each router produces a [B, 256] logits tensor. With B=4096 tokens per GPU, compare HBM traffic for: (a) materializing full logits then top-K, vs (b) fused router.*

(a) Full logits: $4096 \times 256 \times 4 = 4$ MB write + 4 MB read = 8 MB/layer. Total: $58 \times 8 = 464$ MB per GPU.

(b) Fused: only top-K=8 indices+weights written: $4096 \times 8 \times 8 = 0.26$ MB/layer. Total: $58 \times 0.26 = 15$ MB per GPU.

Savings: $(464 - 15)/464 = 96.8\%$. Across 65,536 GPUs: $\approx 29.4$ TB per forward pass.

**Q5.** *Design an autotune config space for a Triton FP8 GEMM kernel targeting SM90 (227 KB shared memory). BLOCK_M, BLOCK_N, BLOCK_K must be multiples of 64. Enumerate feasible tuples.*

FP8 is 1 byte per element. $S_{stage} = (BM \times BK + BK \times BN) \times 1$. Constraint: $num\_stages \times S_{stage} \le 232{,}448$.

| BLOCK_M | BLOCK_N | BLOCK_K | S_stage | Max stages | Good configs |
|---|---|---|---|---|---|
| 128 | 128 | 64 | 16,384 | 14 | (w4,s3), (w8,s4) |
| 128 | 128 | 128 | 32,768 | 7 | (w4,s3), (w4,s4) |
| 128 | 64 | 64 | 12,288 | 18 | (w4,s4), (w8,s5) |
| 256 | 128 | 64 | 24,576 | 9 | (w8,s3) |
| 128 | 256 | 64 | 24,576 | 9 | (w8,s3) |

Recommended 8 configs with `num_warps` in {4, 8} cover the most promising combinations. Autotune time: ~10-30 seconds (one-time cost, cached per M/N/K triple).


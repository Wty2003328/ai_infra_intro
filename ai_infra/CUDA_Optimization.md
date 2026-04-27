# CUDA Optimization

This page is where "I wrote a working kernel" meets "I wrote a kernel that hits 80% of peak." It covers the optimizations that matter in practice: coalescing, shared-memory tiling, bank conflicts, occupancy, async copies, warp specialization, and Hopper-specific features like TMA and wgmma.

---

## 1. The Optimization Checklist

Before writing a line of tuning code, profile. Nsight Compute tells you which roof you're hitting. But across kernels, the optimizations roughly follow this hierarchy — address them in this order:

1. **Correctness.** Are writes/reads in-bounds? Is synchronization correct?
2. **Coalescing.** Are global loads/stores contiguous in warp-index order?
3. **Shared memory tiling.** Is data reused across threads staged through shared memory?
4. **Bank conflicts.** Are shared-memory accesses conflict-free?
5. **Occupancy.** Are there enough resident warps to hide memory latency?
6. **Tensor cores.** Is the matmul dispatched to tensor cores?
7. **Async copies.** Is data movement overlapped with compute?
8. **Vector types.** Are you using `float4` / `int4` loads for bandwidth?
9. **Instruction mix.** Any expensive divisions, modulos, or sin/cos?
10. **CUDA graphs.** Can you amortize launch overhead?

---

## 2. Coalescing (Revisited)

From [CUDA_Programming](CUDA_Programming.md): a warp issues one load; 32 consecutive 4-byte addresses = one transaction.

### 2.1 Diagnosing with Nsight Compute

In Nsight Compute's "Memory Workload Analysis":
- **L1/TEX Hit Rate, L2 Hit Rate** — cache behavior
- **Global Load Transactions Per Request** — ideal is 1.0 (coalesced). Values > 1 indicate uncoalesced.

### 2.2 Layout transforms

If your natural data layout is uncoalesced, transform at the kernel boundary:

```c
// Transpose K^T at load time using shared memory
__shared__ float K_sm[BK][BN];
// Coalesced global load:
K_sm[threadIdx.y][threadIdx.x] = K[(block_k + threadIdx.y) * N + (block_n + threadIdx.x)];
__syncthreads();
// In compute:  K_sm[kk][...] is naturally transposed for GEMM
```

The transpose happens in shared memory where access pattern doesn't matter (assuming no bank conflicts).

### 2.3 Common coalescing traps

- **`threadIdx.y` outer, `threadIdx.x` inner:** if your innermost threadIdx.x maps to the contiguous memory dimension, you get coalescing. Swap x/y and you lose it silently.
- **Strided KV writes:** writing K cache with layout `[heads, tokens, dim]` has threads within a warp writing to scattered heads. Fix: pack `[tokens, heads, dim]` or `[tokens, heads * dim]`.

---

## 3. Shared Memory Tiling

### 3.1 Why tile

The key insight: global memory has high latency (400 cyc) and limited bandwidth, but shared memory has both ~10× lower latency and ~4× higher effective bandwidth per SM. If data is read multiple times by threads in a block, staging it in shared memory is essentially free after the first access.

For matmul `C[M][N] = A[M][K] × B[K][N]`, each element of A is read N times across output elements and each element of B is read M times. Tiling bounds this reuse to the tile dimensions.

### 3.2 Basic tile structure

```
Block computes a BM × BN output tile of C.
Iterates over K in chunks of BK.
At each iteration, loads a BM × BK tile of A and a BK × BN tile of B into shared memory.
Each thread in the block computes a small slice of the output tile.
```

Typical tile sizes on H100:
- BM = BN = 128, BK = 32 (FP16)
- 128 × 32 × 2 bytes = 8 KB per tile; 2 tiles per block = 16 KB — fits easily

### 3.3 Thread-to-output mapping

Naive: each thread owns one output element. Problem: threads must each do K MACs before writing, and inner-loop bandwidth is limited by load/store unit throughput.

Better: **register tiling** — each thread owns a TM × TN slice of the output tile in registers.

```c
float acc[TM][TN] = {0};
for (k block):
    load A_tile, B_tile to shared mem
    for (kk = 0; kk < BK; kk++):
        float Areg[TM];
        float Breg[TN];
        // Load from shared to registers
        for (int m = 0; m < TM; m++) Areg[m] = sA[thread_row*TM + m][kk];
        for (int n = 0; n < TN; n++) Breg[n] = sB[kk][thread_col*TN + n];
        // Compute outer product in registers
        for (int m = 0; m < TM; m++)
            for (int n = 0; n < TN; n++)
                acc[m][n] += Areg[m] * Breg[n];
```

With TM = TN = 8, each thread has 64 accumulator registers plus 16 for Areg/Breg — 80 registers per thread. At 256 threads/block, that's 20 K registers (well under the 64 K per SM), leaving room for 2-3 resident blocks.

This is the pattern CUTLASS scales up: block tile (e.g., 128×256) → warp tile (64×64) → thread tile (8×8) → instruction tile (wmma/wgmma).

---

## 4. Bank Conflicts

Shared memory is split into 32 banks. Accesses to the same bank (different addresses) by different threads in a warp serialize.

### 4.1 The basic conflict

```c
__shared__ float smem[32][32];
// Threads 0..31 read smem[threadIdx.x][0]
// All hit bank 0 → 32-way conflict, 32× slower
```

### 4.2 Padding fix

Add a column:
```c
__shared__ float smem[32][33];  // 33 not 32
// Threads 0..31 read smem[threadIdx.x][0] → banks 0, 33, 66, ... mod 32 = 0, 1, 2, 3, ... → conflict-free
```

Cost: 3% extra shared memory. Benefit: 32× throughput on that access.

### 4.3 XOR swizzle for tile transposes

For GEMM load patterns, padding is wasteful. XOR swizzling reorders shared memory addresses so loads and stores are both conflict-free:

```c
// Write: linearly
smem[row][col] = load_from_global(...);

// Read: XOR swizzle the row index
val = smem[row][col ^ row];
```

This is what CUTLASS's `Swizzle<3, 3>` etc. does under the hood.

### 4.4 Diagnosing conflicts

Nsight Compute: "Shared Memory — Bank Conflicts" metric. If >0, find the conflict and pad/swizzle.

---

## 5. Occupancy

### 5.1 Definition

Occupancy = active warps per SM / maximum warps per SM.

Max warps per SM on H100: 64 (2048 threads). Occupancy is bounded by three resources:

1. **Threads per block:** e.g., 1024 threads / block = 2 blocks max = 64 warps → 100% occupancy if no other constraint.
2. **Registers per thread:** 64 K regs / SM / (threads × regs/thread) blocks. E.g., 128 regs/thread × 1024 threads = 128 K regs > 64 K → fails to fit; only half the block resides at 100% register usage.
3. **Shared memory per block:** 228 KB / smem_per_block blocks.

The `--ptxas-options=-v` compiler flag prints registers-per-thread and shared-memory-per-block.

### 5.2 The register-pressure tradeoff

Using more registers per thread → fewer resident warps → less latency hiding.
Using fewer registers per thread → more register spills to local memory (which is HBM) → slower.

Sweet spot: 64-128 registers/thread for most kernels. Use `__launch_bounds__(threads_per_block, min_blocks_per_sm)` to tell the compiler your target:

```c
__global__
__launch_bounds__(256, 4)  // 256 threads/block, at least 4 blocks/SM
void my_kernel(...) { ... }
```

The compiler will spill more aggressively to meet this occupancy goal.

### 5.3 When occupancy doesn't matter

If your kernel has high instruction-level parallelism (lots of independent ops per thread), you can hide memory latency at ~30-40% occupancy. This is why FA3 (highly tuned, ILP-heavy) runs at ~25% occupancy but near peak FLOPS.

Rule: optimize for occupancy first on memory-bound kernels, for ILP on compute-bound.

---

## 6. Tensor Core Utilization

### 6.1 The API hierarchy

From lowest to highest:
- **`mma` PTX / CUDA warp MMA intrinsics (`wmma::mma_sync`)** — direct tensor core instructions
- **CUTLASS** — templated C++ library composing MMAs into tiles, epilogues
- **cuBLAS / cuBLASLt** — pre-tuned kernels for standard GEMM shapes
- **PyTorch / Triton / JAX** — high-level dispatch

### 6.2 Constraints for activation

For H100 FP16 tensor cores:
- Matrix dimensions must be multiples of 16
- Data must be contiguous (strides aligned to 16 bytes)
- Dtype FP16/BF16/FP8/TF32

Miss any constraint and cuBLAS falls back to CUDA cores (~60 TFLOPS vs ~990 TFLOPS).

### 6.3 The warp MMA instruction

```cpp
#include <mma.h>
using namespace nvcuda::wmma;

// 16×16×16 MMA, FP16 in, FP32 accumulate
fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
fragment<accumulator, 16, 16, 16, float> c_frag;

fill_fragment(c_frag, 0.0f);

// Iterate over K
for (int k = 0; k < K; k += 16) {
    load_matrix_sync(a_frag, A_ptr + row*lda + k, lda);
    load_matrix_sync(b_frag, B_ptr + k*ldb + col, ldb);
    mma_sync(c_frag, a_frag, b_frag, c_frag);
}

store_matrix_sync(C_ptr + row*ldc + col, c_frag, ldc, mem_row_major);
```

### 6.4 Hopper's wgmma

On Hopper, the asynchronous warpgroup MMA is preferred:

```cpp
// PTX pseudo-syntax (real usage is via inline PTX or CUTLASS)
wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16
    {accum0, accum1, ..., accum63},
    descA, descB, scaleD;
wgmma.commit_group;
wgmma.wait_group 0;
```

`wgmma` operates on a full warpgroup (128 threads = 4 warps) and runs asynchronously — the warp continues executing while the MMA progresses.

---

## 7. Async Copy (cp.async, TMA)

### 7.1 Ampere's cp.async

Before Ampere, to stage data to shared memory, you had to load into registers then write to shared memory — synchronous. `cp.async` does global→shared directly and asynchronously.

```cpp
// cp.async from global pointer gmem_ptr to shared memory smem_ptr, 16 bytes
asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(smem_ptr), "l"(gmem_ptr));

asm volatile("cp.async.commit_group;\n");
asm volatile("cp.async.wait_group 0;\n");   // wait for all pending groups
```

In C++, CUDA 11+ exposes this via `cuda::memcpy_async`.

### 7.2 Hopper's TMA (Tensor Memory Accelerator)

TMA generalizes cp.async with:
- Multi-dimensional address calculation in hardware
- Automatic swizzling / transposition
- Cross-block cluster copies
- A handful of threads kick off a copy that streams hundreds of KB

```cpp
// Example: load a 128×128 FP16 tile using TMA
__shared__ __align__(16) __nv_fp16 tile[128][128];

// Setup descriptor (one-time, on host or device)
TmaDescriptor desc = make_tma_desc(global_ptr, gmem_dims, ...);

// In kernel, one thread initiates the copy:
if (threadIdx.x == 0) {
    __builtin_nvptx_tma_bulk_copy_tensor(&tile[0][0], &desc, coord_x, coord_y);
}
// Other threads do work; sync via mbarrier later
```

CUTLASS 3.x and hand-rolled FlashAttention kernels use TMA extensively.

### 7.3 Double buffering

```
Iteration k:
    stage load_global_to_shared(k+1)  [async]
    compute_on_shared_tile(k)          [overlap]
    sync
    swap buffers
```

With 2 buffers, the next tile's load overlaps with the current tile's compute. Effective HBM bandwidth during compute approaches the full 3.35 TB/s.

---

## 8. Warp Specialization

On Hopper, with TMA and wgmma, it pays to **specialize warps** within a block into roles:

- **Producer warps:** issue TMA loads to bring tiles into shared memory, signal consumers via mbarrier
- **Consumer warps:** wait on mbarrier, execute wgmma, produce output tiles

This is the pattern FlashAttention v3 uses. See [FlashAttention_Deep_Dive](FlashAttention_Deep_Dive.md) for the full walkthrough.

Benefits:
- Producers and consumers run independently; no coupling between load latency and compute throughput
- Better overlap than single-warp-per-role double-buffering
- Exploits Hopper's async instructions fully

Cost: more complex code, more warps needed (so occupancy must be planned carefully).

---

## 9. Vector Types and Instruction-Level Parallelism

### 9.1 Vector loads

```c
// Scalar: 4 bytes per thread, 128 bytes/warp
float x = a[tid];

// Vec2: 8 bytes per thread, 256 bytes/warp
float2 x = reinterpret_cast<float2*>(a)[tid];  // halve # threads needed

// Vec4: 16 bytes per thread, 512 bytes/warp
float4 x = reinterpret_cast<float4*>(a)[tid];  // quarter # threads needed
```

For memory-bound kernels, using `float4` often gives 15-30% speedup because:
- Fewer address calculations
- Better use of 128-byte cache lines
- Higher ILP (more loads in flight per thread)

### 9.2 ILP via loop unrolling

```c
// Low ILP: each iteration depends on previous
for (int i = 0; i < 16; i++) acc += a[i] * b[i];

// Higher ILP: independent accumulators
float acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
for (int i = 0; i < 16; i += 4) {
    acc0 += a[i]   * b[i];
    acc1 += a[i+1] * b[i+1];
    acc2 += a[i+2] * b[i+2];
    acc3 += a[i+3] * b[i+3];
}
float acc = acc0 + acc1 + acc2 + acc3;
```

Each independent accumulator is an independent dependency chain for the pipeline. 4× ILP gives ~4× throughput on FMA-bound kernels.

### 9.3 `#pragma unroll`

```c
#pragma unroll 8
for (int i = 0; i < 128; i++) { ... }
```

Tells the compiler to unroll. Useful when the loop body is small and fits in the instruction cache.

Beware: excessive unrolling bloats code, increases register pressure, hurts ICache residency.

---

## 10. Kernel Launch Overhead

Each `<<<...>>>` launch has ~5-10 μs overhead (host-side launch + device-side block distribution). For small kernels with milliseconds of compute, launch overhead is negligible. For tiny kernels (microseconds), it dominates.

### 10.1 Solutions

**Kernel fusion:** combine multiple small kernels into one larger kernel with fewer launches. Cost: more complex code. Benefit: eliminates the launch overhead and the HBM round-trips between kernels.

**CUDA graphs:** build a DAG of kernels once, launch repeatedly with one call.

**Persistent kernels:** a long-running kernel that reads work items from a queue. No repeated launch. Used in some inference frameworks for the decode loop.

### 10.2 When to fuse

Fuse if: kernels are small, share data, and aren't already in a CUDA graph.

Don't fuse if: kernels are each already compute-bound (no benefit), or they're from different parts of the model (modularity loss).

---

## 11. Measuring and Profiling

### 11.1 Nsight Compute

```bash
ncu --set full ./my_app  # collects all metrics
ncu --section "MemoryWorkloadAnalysis,ComputeWorkloadAnalysis" ./my_app
```

Key sections:
- **GPU Speed Of Light** — overall throughput
- **Memory Workload Analysis** — coalescing, L1/L2/HBM traffic
- **Compute Workload Analysis** — SM throughput, pipeline utilization
- **Scheduler Statistics** — warp issue, eligibility
- **Warp State Statistics** — where warps stall (memory, execution dependencies)

### 11.2 Nsight Systems

System-wide timeline. Use for:
- Seeing CUDA kernel execution overlap with copies
- Identifying CPU-side gaps
- Debugging stream synchronization issues

### 11.3 CUDA events for timing

```c
cudaEvent_t start, stop;
cudaEventCreate(&start); cudaEventCreate(&stop);
cudaEventRecord(start);
kernel<<<g, b>>>();
cudaEventRecord(stop);
cudaEventSynchronize(stop);
float ms;
cudaEventElapsedTime(&ms, start, stop);
```

More accurate than wall-clock because it times exactly the kernel portion.

### 11.4 Benchmarking considerations

- Warmup: run 5-10 times before measuring to prime caches, JIT compile
- Measure the median of many runs, not the mean (hit by OS noise, GPU power states)
- Lock clocks during benchmarks: `nvidia-smi -lgc 1530` (H100 base clock)
- Validate via roofline: does achieved performance match what the model says is possible?

---

## 12. Optimization Case Study: A Reduction Kernel

### 12.1 Naive

```c
__global__ void reduce_v1(float *in, float *out, int n) {
    __shared__ float smem[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    smem[tid] = (i < n) ? in[i] : 0;
    __syncthreads();

    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2*s) == 0) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = smem[0];
}
```

Problems: warp divergence (only even-indexed threads work on odd iterations; the % operation is slow); bank conflicts.

### 12.2 Interleaved, no divergence

```c
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) smem[tid] += smem[tid + s];
    __syncthreads();
}
```

Now the first `s` threads do the work contiguously — no divergence in the first half.

### 12.3 Warp-level shuffle finish

```c
if (tid < 32) {
    float v = smem[tid];
    for (int offset = 16; offset > 0; offset /= 2)
        v += __shfl_down_sync(0xffffffff, v, offset);
    if (tid == 0) out[blockIdx.x] = v;
}
```

Last 32 elements reduced in registers via shuffles; no more shared memory traffic or `__syncthreads()`.

### 12.4 Vector loads

```c
float4 val = reinterpret_cast<float4*>(in)[blockIdx.x * blockDim.x + tid];
float local = val.x + val.y + val.z + val.w;  // pre-reduce in register
smem[tid] = local;
```

Each thread does 4 elements, halving block count and increasing arithmetic per HBM access.

### 12.5 Multi-block kernel with atomic finalize

Replace the two-pass (per-block output + host reduction) with a single kernel:

```c
if (tid == 0) atomicAdd(out, smem[0]);
```

Tradeoff: atomic contention on a single address. Modern GPUs handle this well for small # blocks; for many blocks, use block-level partial sums and a second reduction kernel.

This progression is typical: each step gains 1.5-2×, totaling 5-10× over the naive kernel.

---

## 13. Common Optimization Pitfalls

- **Unused tensor cores:** dim mismatch or wrong dtype → silently falls to CUDA cores.
- **Excessive register spills:** compiler spills to "local memory" which is HBM — catastrophic.
- **Over-syncthread:** `__syncthreads()` everywhere when warp-level sync would suffice.
- **Small problem size:** launching with too few blocks leaves SMs idle. Need ≥ 2× SM count in blocks for good utilization.
- **Debug prints:** `printf` from device kernel adds synchronization and is very slow; strip in production builds.
- **Copy-on-kernel-edge:** unnecessary H2D/D2H around small kernels dominate time.

---

## 14. Common Interview Questions

**Q: How do you diagnose a memory-bound kernel?**
A: Profile with Nsight Compute. If "Memory Throughput" is near 80-100% of peak HBM and "Compute Throughput" is much lower, you're memory-bound. Compute arithmetic intensity for the kernel; if it's below the ridge point (~295 for H100 FP16), you're indeed memory-bound. Fixes: coalesce, tile into shared memory, fuse with adjacent kernels, lower precision, vector loads.

**Q: What's warp specialization and why does it help on Hopper?**
A: Dedicating different warps in a block to distinct roles (producers load via TMA; consumers compute via wgmma). Because TMA and wgmma are asynchronous, producers and consumers run independently, maximizing overlap between data movement and compute. FlashAttention v3 uses this and reaches 70-80% of peak FLOPS on Hopper.

**Q: Why does `float4` improve performance over `float` for memory-bound kernels?**
A: Each `float4` load issues one instruction for 16 bytes instead of four instructions for 4 bytes each. Fewer address calculations, better utilization of the L/S unit, higher ILP (more outstanding loads per thread). Gains 15-30% on bandwidth-bound kernels.

**Q: You have a kernel at 25% occupancy. Should you optimize for higher occupancy?**
A: Depends on the kernel. For memory-bound kernels, yes — more warps means better latency hiding. For compute-bound kernels with high ILP (per-thread accumulators, independent dependency chains), 25% is often fine. Check Nsight's "Warp State Statistics" — if warps are stalling on memory, increase occupancy; if they're stalling on execution dependencies, look at ILP/instruction mix.

**Q: How do you avoid shared memory bank conflicts in a matrix transpose?**
A: Pad the tile with +1 column: `__shared__ float tile[32][33]`. This rotates the bank assignment for successive rows — reading columns hits different banks. Alternatively, use XOR swizzling: `tile[row][col ^ row]` ensures both row-major writes and column-major reads are conflict-free without extra memory.

**Q: What's `cp.async` and when would you use it?**
A: `cp.async` (Ampere+) copies from global to shared memory asynchronously. Issue the copy, continue computing, then wait when you need the data. This enables double-buffered pipelines where the next tile's load overlaps with current tile's compute. On Hopper, TMA generalizes this with multi-dim address handling.

**Q: How do you implement a reduction across a block?**
A: Stage each thread's value into shared memory. Use a log-step reduction halving active thread count per step: `for s = blockDim/2; s > 0; s /= 2: if (tid<s) smem[tid] += smem[tid+s]; __syncthreads();`. The last 32 elements can be reduced with warp shuffles (`__shfl_down_sync`) — no shared memory or sync needed. If cross-block reduction is needed, atomicAdd the block result to a global total or launch a second kernel.

**Q: Your kernel registers spill and performance drops. What can you do?**
A: (1) Reduce the working set — compute on smaller tiles, or reduce the thread tile size so fewer accumulators per thread. (2) Use `__launch_bounds__(threads, min_blocks_per_sm)` to tell the compiler to compile for a specific occupancy target; it will rearrange register allocation. (3) Decompose into two smaller kernels. (4) Accept the spill — if it's to L1 (not HBM), it may be tolerable.

**Q: Difference between Nsight Systems and Nsight Compute?**
A: Nsight Systems is a system-wide timeline (CPU threads, CUDA streams, NVTX ranges, API calls). Good for finding gaps, synchronization issues, overlap opportunities. Nsight Compute is per-kernel deep profiling (metrics, roofline, warp stalls). Use Systems for "where's the time going", Compute for "why is this kernel slow".

---

## 15. Further Reading

- Mark Harris, "Optimizing Parallel Reduction in CUDA" (classic whitepaper)
- NVIDIA CUTLASS and CUTe documentation
- "Optimizing CUDA" session recordings from GTC (every year)
- PTX ISA Reference — for `cp.async`, `mbarrier`, `wgmma`, TMA
- "How to optimize a CUDA matmul kernel for cuBLAS-like performance" by Simon Boehm (blog series)

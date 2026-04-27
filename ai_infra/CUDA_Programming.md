# CUDA Programming

This page is the reference for writing CUDA kernels from scratch. It assumes you understand [GPU_Architecture](GPU_Architecture.md) — if you don't know what a warp is or what an SM contains, read that first. Here we focus on the programming model: how to express parallel work, manage memory, and synchronize threads.

---

## 1. The Programming Model

### 1.1 Thread hierarchy

```
Grid (entire kernel launch)
├── Block (0,0)          Block (1,0)    ...
│   ├── Warp 0 (threads 0-31)
│   ├── Warp 1 (threads 32-63)
│   └── ... up to 32 warps (1024 threads)
├── Block (0,1)          Block (1,1)    ...
└── ...
```

- **Grid** — all blocks from a single kernel launch. Up to 3 dimensions. Max size per dim: 2^31-1 in x, 65535 in y/z.
- **Block** — up to 1024 threads, mapped entirely to one SM. Threads in a block share shared memory and can synchronize.
- **Warp** — 32 threads. The actual execution unit. Always multiples of 32 for efficiency.

### 1.2 Built-in indexing

Every CUDA kernel has these implicit variables:

```c
threadIdx.{x,y,z}    // thread index within block (0..blockDim-1)
blockIdx.{x,y,z}     // block index within grid (0..gridDim-1)
blockDim.{x,y,z}     // block shape
gridDim.{x,y,z}      // grid shape
warpSize             // always 32

// Linearized global thread ID (1D case)
int tid = blockIdx.x * blockDim.x + threadIdx.x;
```

### 1.3 The classic first kernel

```c
// Vector addition: c[i] = a[i] + b[i]
__global__ void vecAdd(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];   // boundary check avoids out-of-bounds writes
}

int main() {
    int n = 1 << 20;                                 // 1M elements
    size_t bytes = n * sizeof(float);
    float *dA, *dB, *dC;
    cudaMalloc(&dA, bytes);  cudaMalloc(&dB, bytes);  cudaMalloc(&dC, bytes);
    // ... initialize dA, dB ...
    int tpb = 256;                                  // threads per block
    int blocks = (n + tpb - 1) / tpb;               // ceil division
    vecAdd<<<blocks, tpb>>>(dA, dB, dC, n);
    cudaDeviceSynchronize();                        // wait for kernel
    // ... copy dC back, free ...
}
```

Notes:
- `__global__` — callable from host, runs on device. `__device__` — callable from device only. `__host__` — callable from host only.
- `<<<grid, block>>>` — CUDA's extension syntax for kernel launch, translated by nvcc into a `cudaLaunchKernel` call.
- The grid size is computed to cover all N elements; the boundary check handles the last partial block.

---

## 2. Memory Spaces

CUDA exposes multiple distinct memory spaces. Knowing which one a pointer lives in is critical.

### 2.1 Table of spaces

| Space | Scope | Lifetime | Latency | Typical size |
|---|---|---|---|---|
| Register | Thread | Kernel | 1 cyc | 256 KB / SM |
| Local | Thread | Kernel | 400 cyc | Spills to HBM |
| Shared | Block | Kernel | 30 cyc | Up to 228 KB / block on Hopper |
| Constant | Grid | Kernel | 1 cyc (cached) | 64 KB total |
| Texture / Surface | Grid | Kernel | 200 cyc | HBM, cached |
| Global | Grid (and cross-kernel) | Explicit free | 400 cyc | HBM |
| Unified (UVM) | Host+Device | Explicit free | Migration-dependent | HBM + host RAM |

### 2.2 Declaration syntax

```c
__global__ void kernel(float *global_mem /* __device__ implicit */) {
    float reg_var;                       // register
    __shared__ float smem[256];          // shared memory, per-block
    __constant__ float c_const[128];     // constant memory (declared at file scope)
    extern __shared__ float dsmem[];     // dynamic shared memory, size set at launch
}
```

### 2.3 Host-side allocation

```c
// 1. Standard device memory
float *d_ptr;
cudaMalloc(&d_ptr, size);
// Explicit transfer needed:
cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice);

// 2. Pinned (page-locked) host memory — enables DMA, faster transfers
float *h_pinned;
cudaMallocHost(&h_pinned, size);   // or cudaHostAlloc with flags
// Pinned memory enables: async copies, higher BW on PCIe, but reduces system RAM

// 3. Managed memory — unified, auto-migrating
float *um_ptr;
cudaMallocManaged(&um_ptr, size);
// CPU and GPU both access via the same pointer; page faults migrate

// 4. Free corresponding functions
cudaFree(d_ptr);
cudaFreeHost(h_pinned);
cudaFree(um_ptr);
```

### 2.4 Pinned vs pageable transfer bandwidth

Pageable (malloc'd) host → device transfers:
- Can't DMA directly; runtime stages via an internal pinned buffer
- Peak BW ~6 GB/s on PCIe Gen4, ~13 GB/s on PCIe Gen5

Pinned host → device transfers:
- Direct DMA
- Peak BW ~27 GB/s on PCIe Gen4, ~55 GB/s on PCIe Gen5

Rule: for anything >1 MB, use pinned. For training dataloaders, always pin.

---

## 3. Launch Configuration

### 3.1 Choosing block size

```c
int tpb = 256;        // common: 128, 256, 512
int blocks = (n + tpb - 1) / tpb;
```

Guidelines:
- Must be multiple of warp size (32) — else wasted threads in the last warp
- Typical sweet spot: 128-512
- Small blocks (32-64) reduce parallelism per SM
- Large blocks (1024) limit how many blocks fit per SM (each SM has 1536-2048 thread capacity)

### 3.2 Grid dimensions for multi-dimensional work

```c
dim3 block(32, 16);                                // 512 threads, 16 warps
dim3 grid((W + 31) / 32, (H + 15) / 16);
matrix_kernel<<<grid, block>>>(input, output, W, H);

// In kernel:
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
```

### 3.3 Dynamic shared memory

```c
// Host side: pass shared memory size at launch
kernel<<<blocks, threads, sharedBytes>>>(args);

// Device side: declare with extern
__global__ void kernel(...) {
    extern __shared__ float dsmem[];
    // Use dsmem[0 .. sharedBytes/sizeof(float) - 1]
}
```

Useful when shared memory size depends on runtime parameters (e.g., tile size determined by problem dimension).

### 3.4 Maximum shared memory

Default max is 48 KB per block. To access up to 228 KB on Hopper (the maximum), you must opt in:

```c
cudaFuncSetAttribute(kernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize, 228 * 1024);
```

---

## 4. Synchronization

### 4.1 Within a block

```c
__syncthreads();   // all threads in block wait here, no thread proceeds until all arrive
```

Critical for shared memory correctness: if thread 0 writes smem and thread 31 reads it, you need `__syncthreads()` between the write and the read (or the CUDA memory model doesn't guarantee visibility).

`__syncthreads()` is free for warps (warp-synchronous programming) but has real cost across warps in the block.

### 4.2 Within a warp

Before Volta (2017), threads in a warp were implicitly synchronized (always in lockstep). Volta introduced **independent thread scheduling** — threads in a warp can diverge and rejoin arbitrarily.

Explicit primitives:
```c
__syncwarp(0xffffffff);                         // mask of active threads
__shfl_sync(mask, val, srcLane);                // broadcast val from srcLane to all threads in mask
__shfl_down_sync(mask, val, delta);             // shift down by delta
__ballot_sync(mask, predicate);                 // return 32-bit mask of which threads have predicate=true
__any_sync(mask, predicate);                    // any thread has true?
__all_sync(mask, predicate);                    // all threads have true?
```

Classic warp-level reduction:
```c
__device__ float warp_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;  // thread 0 has the sum
}
```

### 4.3 Across blocks

CUDA has no direct primitive to synchronize across blocks within a single kernel. Use:
- **Kernel launches as a sync point** — launch kernel A, then kernel B, B sees A's global writes.
- **Cooperative groups** — since CUDA 9, you can launch a "cooperative" kernel (`cudaLaunchCooperativeKernel`) with `cg::grid_group` and call `.sync()`. Limited to kernels that fit on-chip (all blocks resident simultaneously).
- **Atomic operations** on a counter for a barrier pattern (careful — easy to deadlock).

### 4.4 Streams and events

```c
cudaStream_t stream;
cudaStreamCreate(&stream);
// Kernels on the same stream are ordered; different streams can overlap
kernelA<<<g, b, 0, stream>>>();
kernelB<<<g, b, 0, stream>>>();     // starts after kernelA completes
cudaEvent_t event;
cudaEventCreate(&event);
cudaEventRecord(event, stream);     // records when prior work completes
cudaStreamWaitEvent(otherStream, event);  // otherStream waits for event
cudaStreamDestroy(stream);
```

Streams are the primary mechanism for overlapping computation and memory transfer.

---

## 5. Memory Access Patterns

### 5.1 Coalescing

A warp (32 threads) issues one load instruction. If the 32 threads reference 32 consecutive 4-byte addresses (128 bytes aligned), the hardware satisfies the load with **one memory transaction**.

```c
float x = arr[threadIdx.x];   // coalesced — threads 0..31 read words 0..31
float x = arr[threadIdx.x * 2];  // strided — 2 transactions
float x = arr[threadIdx.x * 32]; // worst — 32 transactions (bandwidth / 32)
```

Array-of-structures vs structure-of-arrays:

```c
struct Particle { float x, y, z, vx, vy, vz; };
Particle particles[N];

// Reading `x` for all threads: strided, poor coalescing
__global__ void bad(Particle *p) { float x = p[threadIdx.x].x; }

// SoA layout:
struct Particles {
    float *x, *y, *z, *vx, *vy, *vz;
} particles;

__global__ void good(Particles p) { float x = p.x[threadIdx.x]; }  // coalesced
```

### 5.2 Shared memory banks

Shared memory is organized into 32 banks of 4 bytes each. Bank b holds addresses `b, b+128, b+256, …`.

- If 32 threads in a warp each access a different bank → parallel, one cycle
- If multiple threads access the same bank (different addresses) → **bank conflict**, serialized

```c
__shared__ float smem[32][32];

// Threads read smem[threadIdx.x][0..31] — all same column, same bank → 32-way conflict
// Threads read smem[0..31][threadIdx.x] — each thread hits different bank → no conflict
```

Padding trick to avoid column-access conflicts:
```c
__shared__ float smem[32][33];  // pad by 1 column; rotates bank assignments, no conflicts
```

More on this in [CUDA_Optimization](CUDA_Optimization.md).

### 5.3 Alignment

Pointers for coalesced access must be 128-byte aligned. `cudaMalloc` returns 256-byte-aligned pointers by default. Vector types (`float4`, `int4`) enforce alignment at the type level:

```c
float4 *data;  // 16-byte aligned; one load moves 16 bytes per thread
float4 v = data[threadIdx.x];  // 128 bytes per warp = one transaction
```

### 5.4 Vector loads

`float4` / `int4` / `double2` loads are beneficial because:
- Fewer address-arithmetic instructions
- Better utilization of the Load/Store unit
- Improves ILP

For bandwidth-bound kernels, using `float4` often gives 20-30% speedup over `float`.

---

## 6. Atomic Operations

```c
atomicAdd(&counter, 1);               // int, float, double (CC ≥ 6.0 for double)
atomicMin(&min_val, x);
atomicCAS(&lock, 0, 1);               // compare-and-swap
atomicExch(&ptr, new_value);
```

Performance:
- Atomics on global memory: ~HBM latency (~400 cycles)
- Atomics on shared memory: ~30 cycles
- Heavily contended atomics (many threads writing same address): throughput drops ~32×

Pattern for reduction: each block reduces its slice in shared memory, one thread atomically adds the block's result to a global total.

```c
__global__ void reduce_sum(float *input, float *output, int n) {
    __shared__ float partial[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    partial[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    // Intra-block reduction (simplified)
    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (tid < s) partial[tid] += partial[tid + s];
        __syncthreads();
    }
    // One thread per block adds to the global total
    if (tid == 0) atomicAdd(output, partial[0]);
}
```

---

## 7. Error Handling

CUDA errors are returned by every runtime API call but not by kernel launches directly. Use this pattern:

```c
#define CUDA_CHECK(err) do { \
    cudaError_t _e = (err); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d: %s\n", \
            cudaGetErrorName(_e), __FILE__, __LINE__, cudaGetErrorString(_e)); \
        abort(); \
    } \
} while (0)

CUDA_CHECK(cudaMalloc(&d_ptr, size));
kernel<<<grid, block>>>(args);
CUDA_CHECK(cudaGetLastError());        // checks launch errors
CUDA_CHECK(cudaDeviceSynchronize());   // catches any async errors from the kernel
```

Common errors:
- `cudaErrorInvalidConfiguration` — block too big, or shared mem exceeds limit
- `cudaErrorIllegalAddress` — kernel read/wrote out of bounds; can corrupt subsequent kernels
- `cudaErrorLaunchTimeout` — kernel too long (watchdog timeout, only on display GPUs)
- `cudaErrorECCUncorrectable` — hardware ECC error; GPU goes offline

In production, `cudaErrorIllegalAddress` is very serious — it indicates a bug (probably OOB), and the GPU context is corrupted. You must restart the process.

---

## 8. Streams and Concurrency

### 8.1 Default stream

All kernel launches and memory ops without a specified stream use the "default" (stream 0), which is synchronizing — it waits for all prior work on the device.

Use non-default streams for concurrency:

```c
cudaStream_t s1, s2;
cudaStreamCreate(&s1);
cudaStreamCreate(&s2);

// Concurrent data transfer + compute
cudaMemcpyAsync(d_next_input, h_next_input, size, cudaMemcpyHostToDevice, s1);
kernel<<<g, b, 0, s2>>>(d_current_input, d_output);
```

### 8.2 Events

```c
cudaEvent_t start, stop;
cudaEventCreate(&start); cudaEventCreate(&stop);

cudaEventRecord(start, stream);
kernel<<<g, b, 0, stream>>>();
cudaEventRecord(stop, stream);

cudaEventSynchronize(stop);   // wait
float ms;
cudaEventElapsedTime(&ms, start, stop);
printf("kernel time: %f ms\n", ms);
```

### 8.3 Overlapping patterns

**Compute + H2D transfer (double buffering):**
```c
for (int chunk = 0; chunk < nchunks; chunk++) {
    int buf = chunk % 2;
    cudaMemcpyAsync(d_in[buf], h_in[chunk], size, cudaMemcpyHostToDevice, copy_stream);
    process<<<g, b, 0, compute_stream>>>(d_in[buf], d_out[buf]);
}
```

**Compute + D2H (writing results back while next batch computes):**
```c
for (...) {
    cudaMemcpyAsync(h_out[prev], d_out[prev], size, cudaMemcpyDeviceToHost, copy_stream);
    process<<<g, b, 0, compute_stream>>>(d_in[cur]);
}
```

To benefit from overlap, both streams must use pinned host memory and the GPU must have at least 2 copy engines (most datacenter GPUs do).

### 8.4 CUDA Graphs

Build a graph of dependent operations once, launch many times. Eliminates per-launch overhead (~5-10 μs) which matters for small kernels.

```c
cudaGraph_t graph;
cudaGraphExec_t exec;

cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
for (auto &k : kernels) k.launch(stream);
cudaStreamEndCapture(stream, &graph);

cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);

// Launch repeatedly:
for (int step = 0; step < N; step++) cudaGraphLaunch(exec, stream);
```

vLLM and TRT-LLM both use CUDA graphs for the decode loop to amortize launch overhead.

---

## 9. Template: Complete Matmul Kernel

Here's a simple tiled matmul illustrating shared memory, block-level synchronization, and coalesced access. It's not optimal (not using tensor cores), but it's pedagogically complete.

```c
#define BM 128
#define BN 128
#define BK 16

__global__ void matmul(const float *A, const float *B, float *C,
                       int M, int N, int K) {
    __shared__ float sA[BM][BK];
    __shared__ float sB[BK][BN];

    int row = blockIdx.y * BM + threadIdx.y;
    int col = blockIdx.x * BN + threadIdx.x;

    float acc = 0.0f;

    for (int k = 0; k < K; k += BK) {
        // Each thread loads one element into shared mem
        if (row < M && (k + threadIdx.x) < K)
            sA[threadIdx.y][threadIdx.x] = A[row * K + k + threadIdx.x];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && (k + threadIdx.y) < K)
            sB[threadIdx.y][threadIdx.x] = B[(k + threadIdx.y) * N + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int kk = 0; kk < BK; kk++) {
            acc += sA[threadIdx.y][kk] * sB[kk][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// Launch:
// dim3 block(BN, BM / ??);  // need careful blocking; see optimization page
```

Real matmul (CUTLASS/cuBLAS) uses:
- Warp-level tiles (each warp computes a 64×64 output tile)
- Tensor cores (`wmma::mma_sync`)
- Double buffering (prefetch next K-tile while computing current)
- Asynchronous copies (cp.async on Ampere, TMA on Hopper)

This pushes the kernel from ~2 TFLOPS (naive) to ~900 TFLOPS (cuBLAS) on H100 — 450× speedup from optimization.

---

## 10. Host-Side Python Integration

### 10.1 PyTorch extension with setuptools

```python
# setup.py
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from setuptools import setup

setup(
    name="myops",
    ext_modules=[CUDAExtension("myops", ["ops.cu"])],
    cmdclass={"build_ext": BuildExtension},
)
```

```cpp
// ops.cu
#include <torch/extension.h>
__global__ void my_kernel(float *, float *, int);

torch::Tensor my_add(torch::Tensor a, torch::Tensor b) {
    auto out = torch::empty_like(a);
    int n = a.numel();
    my_kernel<<<(n+255)/256, 256>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), n);
    return out;
}

PYBIND11_MODULE(myops, m) {
    m.def("my_add", &my_add, "My custom add");
}
```

### 10.2 PyTorch's at::cuda and c10 APIs

For production kernels, use:
- `at::cuda::CUDAStream` for stream management integrated with autograd
- `at::cuda::CUDAGuard` to set the right device
- `AT_DISPATCH_*` macros for dtype dispatch

### 10.3 Triton for rapid kernel development

Triton lets you write GPU kernels in Python-like DSL. See [Triton_and_Kernels](Triton_and_Kernels.md).

---

## 11. Common Interview Questions on CUDA

**Q: What's the difference between `__syncthreads()` and `__syncwarp()`?**
A: `__syncthreads()` synchronizes all threads in a block (up to 1024), including cross-warp. `__syncwarp()` synchronizes a set of threads within a single warp (the mask argument says which lanes). On Volta+, warps can diverge, so `__syncwarp()` is sometimes required to rejoin. `__syncthreads()` is more expensive.

**Q: What is warp divergence and how do you avoid it?**
A: Warp divergence occurs when threads in a warp take different branches of a conditional. The hardware executes each path serially with the other threads masked off. To avoid: align your conditionals with warp boundaries (all threads in a warp take the same branch) or restructure data so threads access similar regions.

**Q: Explain memory coalescing.**
A: When a warp issues a load, if the 32 threads reference 32 consecutive aligned addresses, the hardware serves the load in one transaction. Strided or scattered accesses require multiple transactions, reducing effective bandwidth proportionally. AoS-vs-SoA decisions in data layout often come down to preserving coalescing.

**Q: What's a bank conflict?**
A: Shared memory is split into 32 banks (4 bytes each). If multiple threads in a warp access the same bank (different addresses), the accesses serialize. Conflict-free patterns: threads accessing different banks, or all threads accessing the same address (broadcast, no conflict).

**Q: When should I use `cudaMallocManaged` vs `cudaMalloc`?**
A: `cudaMallocManaged` is convenient but has unpredictable performance due to page migrations. Use it for prototyping or when you don't care about peak performance. For production, `cudaMalloc` + explicit transfers give you control. On GH200 Grace Hopper with NVLink-C2C unified memory, `cudaMallocManaged` performs close to native due to fast page migration, but this is a special case.

**Q: What's the difference between pinned and pageable memory?**
A: Pinned (page-locked) memory cannot be swapped out by the OS. This lets the NIC/DMA engines transfer to/from it directly, doubling H2D bandwidth. Pageable memory (from regular `malloc`) triggers a bounce through an internal pinned buffer. Use pinned for any transfer >1 MB or in hot paths.

**Q: What does occupancy mean and when should you optimize for it?**
A: Occupancy is active warps per SM / max resident warps. Higher occupancy helps hide latency for memory-bound kernels. For compute-bound kernels with high ILP (e.g., matmul with tensor cores), low occupancy (20-30%) is often fine. Use `cudaOccupancyMaxPotentialBlockSize` to find a reasonable block size automatically.

**Q: Write a kernel that computes the sum of an array of N floats.**
A: See Section 6 (reduction sketch). Key points: each block reduces its slice in shared memory using a tree pattern, then thread 0 atomically adds the block result to the global output. Better: use warp-level shuffles in the final stage to eliminate shared memory traffic.

**Q: What is a CUDA graph and why use it?**
A: A CUDA graph is a captured DAG of operations that can be launched as a single unit. Benefits: amortizes per-launch overhead (~5-10 μs), reduces CPU-GPU synchronization, enables driver-level optimization. Critical for small kernels in tight loops (decode inference, training with small batches). vLLM uses graphs for the decode step.

**Q: I have a kernel that uses 64 KB of shared memory per block. What's the issue?**
A: (1) Default max is 48 KB — you need `cudaFuncSetAttribute` to raise the limit up to 228 KB on Hopper. (2) At 64 KB per block, only 3 blocks fit per SM (228 KB / 64 KB = 3.6 → 3). This limits occupancy. If you need more concurrency, either reduce shared memory or accept that this kernel is shared-memory-limited.

**Q: What's `cp.async` and when do you use it?**
A: `cp.async` (Ampere+) is an asynchronous global-to-shared memory copy. Issue the copy, proceed to compute, then synchronize with `cp.async.wait_all` or `cp.async.wait_group N`. Lets you overlap data loading with computation, critical for high-performance matmul kernels. Hopper's TMA generalizes this further.

---

## 12. Further Reading

- NVIDIA CUDA C Programming Guide
- "Professional CUDA C Programming" (Cheng, Grossman, McKercher)
- NVIDIA CUDA Samples repository
- "Learn CUDA Programming" by Jaegeun Han et al.
- PTX ISA Reference for lower-level details
- Jupyter notebook series: "GPU Mode" (YouTube + code)

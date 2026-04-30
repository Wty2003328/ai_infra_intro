# Triton and Custom Kernels

Triton is the DSL of choice for writing high-performance GPU kernels in 2024-2026. It gives you ~90% of CUDA/CUTLASS performance with ~10% of the code. This page covers Triton's programming model, how it compiles to PTX, common patterns (matmul, softmax, attention), and when you should still reach for CUDA or CUTLASS.

---

## 1. What Triton Is

Triton is a Python-embedded DSL from OpenAI (started 2020, mature by 2022). It compiles to LLVM → PTX, skipping NVCC entirely. The compiler handles:

- Block-level tiling and scheduling
- Shared memory management (autoscheduling)
- Tensor core dispatch
- Async copy insertion
- Warp-level parallelism

What Triton does **not** do:
- Warp-specialization tricks (as of 2024; coming)
- Full TMA support (partial; improving)
- Arbitrary fine-grained synchronization

The sweet spot: matmul, softmax, layernorm, attention, and custom ops that fit the "block computes a tile" pattern. For 90% of deep learning kernels, Triton is enough.

---

## 2. Programming Model

### 2.1 The Triton kernel

```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)          # block id (1D grid)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)

# Launch
def add(x, y):
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = 1024
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    add_kernel[grid](x, y, out, n, BLOCK=BLOCK)
    return out
```

Key concepts:
- `@triton.jit` — compiles the function to PTX on first call
- `tl.program_id(axis)` — analogous to CUDA's `blockIdx`
- `tl.arange(0, BLOCK)` — creates a vector of indices (like SIMD lanes)
- `tl.load` / `tl.store` — with optional `mask` for boundary handling
- `BLOCK: tl.constexpr` — a compile-time constant (each unique value triggers a recompile)

### 2.2 Triton is "block-SPMD"

In Triton, each "program" (what CUDA calls a block) is a unit of work, and within a program you think in **tensor** operations — not individual thread operations. Triton auto-assigns threads within a warp to tensor elements. You get CUDA's performance without thinking about warps and threads.

```python
# CUDA thinking: thread i does one element
# Triton thinking: program computes a tile of BLOCK elements
```

### 2.3 Tensor vs scalar operations

Every variable in Triton is a vector of size BLOCK (or a multidim tensor with BLOCK × BLOCK). Operations are element-wise automatically:

```python
x = tl.load(x_ptr + offsets)   # shape: [BLOCK]
y = x * 2.0 + 1.0               # shape: [BLOCK]
z = tl.exp(y) - 1.0             # shape: [BLOCK]
```

There's no implicit "for each thread" loop — the compiler handles it.

---

## 3. Matmul in Triton

Here's a classic fused matmul, simplified from the Triton tutorial:

```python
@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k  = tl.arange(0, BLOCK_K)

    a_ptrs = A + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (offs_k[None, :] < K - k))
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k) & (offs_bn[None, :] < N))
        acc += tl.dot(a, b)                    # ← tensor core op
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = C + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16),
             mask=(offs_am[:, None] < M) & (offs_bn[None, :] < N))
```

Points:
- `tl.dot(a, b)` lowers to `mma` / `wmma` / `wgmma` depending on hardware
- `acc` is FP32 accumulator; `a`, `b` are FP16 — standard mixed-precision matmul
- Masks handle the ragged edges when dimensions aren't multiples of BLOCK_*

### 3.1 Autotuning

Different hardware and shapes like different block sizes. Triton's autotuner sweeps configurations:

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(...): ...
```

On first call with a given (M,N,K), Triton runs each config and caches the best. Subsequent calls with the same key use the cached config. Compile and tune costs ~seconds per shape.

### 3.2 Performance

A well-tuned Triton matmul on H100 achieves 90-95% of cuBLAS throughput for common shapes. For uncommon shapes (M=N=4097) where cuBLAS lacks a specialized kernel, Triton often wins.

---

## 4. Softmax in Triton

Softmax is memory-bound (bandwidth-limited), so fusion matters enormously:

```python
@triton.jit
def softmax_kernel(
    y_ptr, x_ptr,
    stride_xr, stride_yr,
    n_cols,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col_offs = tl.arange(0, BLOCK)
    mask = col_offs < n_cols

    # Load a row
    x_row = tl.load(x_ptr + row * stride_xr + col_offs, mask=mask, other=-float('inf'))

    # Stable softmax
    x_max = tl.max(x_row, axis=0)
    x_shift = x_row - x_max
    numerator = tl.exp(x_shift)
    denom = tl.sum(numerator, axis=0)
    out = numerator / denom

    tl.store(y_ptr + row * stride_yr + col_offs, out, mask=mask)
```

In one kernel:
- Read row once
- Reduce max, subtract, exp, reduce sum, divide, write
- Eliminates the naive 4-pass approach (4 HBM round-trips) to 1 pass

Triton handles the intra-block reduction (`tl.max`, `tl.sum`) via warp shuffles automatically.

---

## 5. Attention in Triton (Simplified)

FlashAttention in Triton (used in production by HuggingFace, PyTorch) is about 200 lines. Here's the structure:

```python
@triton.jit
def flash_attn_fwd(
    Q, K, V, Out, L,
    stride_qb, stride_qh, stride_qm, stride_qk,
    # ... more strides
    B, H, M, N, D,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)        # query block
    pid_bh = tl.program_id(1)       # batch * head

    # Pointer setup
    q_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    k_offs = tl.arange(0, BLOCK_N)
    d_offs = tl.arange(0, BLOCK_D)

    q = tl.load(Q_ptr + q_offs[:, None] * stride_qm + d_offs[None, :] * stride_qk,
                mask=q_offs[:, None] < M)

    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)   # running max
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)                 # running sum

    for start_n in range(0, N, BLOCK_N):
        # Load a K, V tile
        k = tl.load(K_ptr + start_n * stride_kn + k_offs[None, :] * stride_kn
                    + d_offs[:, None] * stride_kk)  # [D, BLOCK_N]
        v = tl.load(V_ptr + start_n * stride_vn + k_offs[:, None] * stride_vn
                    + d_offs[None, :] * stride_vk)  # [BLOCK_N, D]

        # Compute scores: [BLOCK_M, BLOCK_N]
        s = tl.dot(q, k) * inv_sqrt_d

        # Online softmax
        m_new = tl.maximum(m_i, tl.max(s, axis=1))   # new max
        p = tl.exp(s - m_new[:, None])               # shift & exp
        alpha = tl.exp(m_i - m_new)                  # rescale factor
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p, v)
        m_i = m_new

    out = acc / l_i[:, None]
    tl.store(Out_ptr + ..., out, mask=q_offs[:, None] < M)
```

Key algorithmic pieces (covered fully in [FlashAttention_Deep_Dive](FlashAttention_Deep_Dive.md)):
- Tile Q rows into BLOCK_M, iterate over K/V columns in BLOCK_N tiles
- Track running max (m_i) and running sum (l_i) for incremental softmax
- Rescale previous output when a larger max appears

This runs at 500-700 TFLOPS on H100 (vs ~200 TFLOPS for the naive standard attention using cuBLAS).

---

## 6. Triton Autotuning Deep Dive

The autotuner has specific knobs:

### 6.1 num_warps

Number of warps per program (4 default, up to 16). More warps = more parallelism per block, larger shared memory use.

### 6.2 num_stages

Software pipelining depth. `num_stages=3` means 3 iterations of the loop are inflight simultaneously: one loading data, one computing, one writing. Higher stages = more overlap but more shared memory use.

### 6.3 Pruning

```python
configs = [...]
def prune(config, named_args):
    if config.kwargs['BLOCK_M'] * config.kwargs['BLOCK_K'] * 2 > 228*1024: 
        return False   # too much shared memory
    return True

@triton.autotune(configs=configs, key=['M','N','K'], prune_configs_by={'early_config_prune': prune})
```

Prevents Triton from trying configurations that wouldn't fit.

---

## 7. CUTLASS — When You Need More

CUTLASS is NVIDIA's C++ template library for building high-performance GEMM and related kernels. It's significantly more verbose than Triton but gives you full control over:

- Tile hierarchy (thread block tile → warp tile → MMA)
- Swizzle patterns for shared memory
- Epilogue fusion (adding bias, activation, quantization inside the kernel)
- TMA (Tensor Memory Accelerator) usage on Hopper
- Warp specialization

### 7.1 When to reach for CUTLASS

- You need the absolute last 5-10% of performance
- You're fusing an unusual epilogue (e.g., quantized GEMM with per-channel scales)
- You're doing something Triton doesn't support well (warp specialization, complex async pipelines)
- You're implementing a novel architecture (e.g., MLA attention with compressed KV)

### 7.2 CUTLASS 3.x and CuTe

The newest CUTLASS (3.x) uses **CuTe** (CUTLASS Tensor) — a tensor-algebra library for expressing layouts and tile hierarchies. It's very elegant but has a steep learning curve.

```cpp
// Approximate CuTe layout example
using LayoutA = Layout<Shape<_128, _16>, Stride<_16, _1>>;   // row-major tile
using TiledMma = TiledMMA<MMA_Atom<SM90_64x128x16_F32F16F16F32_TN>,
                          Layout<Shape<_1, _2>>,
                          Tile<_128, _256, _16>>;
```

Production inference engines (TRT-LLM, FlashAttention v3) are written in CUTLASS/CuTe for the critical path.

---

## 8. Comparing Triton / CUTLASS / CUDA

| Dimension | CUDA | CUTLASS | Triton |
|---|---|---|---|
| Language | C++ | C++ templates | Python (JIT) |
| Lines of code (matmul) | 500-1000 | 200-400 | 50-100 |
| Peak performance achievable | 100% | 100% | 90-95% |
| Learning curve | Medium | Very steep | Gentle |
| Development speed | Slow | Slowest | Fastest |
| Debuggability | Good (cuda-gdb, Nsight) | OK | Limited (still improving) |
| Autotuning | Manual | Manual + sweep | Built-in |
| Use case | Unique ops, research | Production GEMM/attention | 90% of ML kernels |

Engineering rule of thumb: **start in Triton. If you're leaving 20%+ on the table, port to CUTLASS. Write raw CUDA only for cases that neither supports (unusual sync, warp specialization without CuTe primitives).**

---

## 9. Real Production Kernels

### 9.1 FlashAttention (v1 Triton, v2 Triton/CUDA, v3 CUTLASS/CuTe)

Started as a Triton kernel. By v3, migrated to CUTLASS for Hopper-specific async + warp-specialized pipelines. The reference Triton implementation is still great for learning.

### 9.2 PagedAttention (vLLM)

Written in CUDA. Complex because of the paged memory access pattern (each block has a block table, and threads gather from non-contiguous blocks). Hard to express in Triton due to the indirect addressing.

### 9.3 FP8 GEMM (TransformerEngine)

CUTLASS / CuTe. Uses Hopper's native FP8 tensor cores plus per-tensor scaling in the epilogue.

### 9.4 Fused MoE router (DeepEP, DeepSpeed-MoE)

Mix: top-K selection in a Triton kernel, scatter/gather in CUDA, inter-node all-to-all in NCCL. Triton is used for the dense per-token computation; custom CUDA for communication.

### 9.5 Custom op examples in PyTorch

```python
@torch.library.custom_op("my_lib::fused_rope", mutates_args=())
def fused_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    # Triton kernel
    out = torch.empty_like(x)
    grid = lambda meta: (x.shape[0], x.shape[1])
    rope_triton_kernel[grid](x, cos, sin, out, ...)
    return out
```

PyTorch 2.x integrates Triton kernels as first-class custom ops, making them composable with autograd, `torch.compile`, and export.

---

## 10. Debugging Triton Kernels

### 10.1 Common failure modes

- **NaN outputs:** uninitialized accumulator, or `exp` overflow (use stable softmax).
- **Wrong results at boundaries:** mask mismatch — `mask=` must be on every load and store for partial blocks.
- **Compile errors on shapes:** Triton wants static shapes at compile time (hence `BLOCK: tl.constexpr`). Can't have dynamic tile sizes in a single compile.

### 10.2 Debug mode

```python
import triton
triton.runtime.driver.active.utils.TRITON_DEBUG = 1
```

Enables more informative error messages and optionally insert asserts.

### 10.3 Interpreter mode

```python
@triton.jit(interpret=True)
def kernel(...): ...
```

Runs the kernel on CPU in Python, single-threaded. Useful for debugging logic; performance is irrelevant.

### 10.4 IR inspection

`triton.compiler.inspect_ir(kernel, example_args)` returns the TTGIR (Triton GPU IR) and PTX. Useful to verify tensor-core instructions are present (`mma.sync`, `wmma.mma`, `wgmma`).

---

## 11. When Triton is Not the Right Tool

- **Irregular memory access patterns** — Triton expects tile-structured loads. Random or indirect addressing is painful.
- **Heavy control flow** — lots of dynamic branching doesn't vectorize well in Triton's block model.
- **Warp-specialized pipelines** — Triton's programming model doesn't expose "these warps do X, those warps do Y" directly. Growing support but not mature.
- **Custom synchronization** (mbarrier, cluster sync) — write CUDA/CUTLASS.
- **Multi-GPU communication kernels** — NCCL, not Triton.

---

## 12. Common Interview Questions

**Q: What is Triton and when should I use it over CUDA?**
A: Triton is a Python DSL for writing GPU kernels that compiles to PTX. Use Triton when you want CUDA-like performance (90-95% of peak) with Python-level productivity. Use CUDA directly when you need warp-level control, custom synchronization, or peak-of-peak performance for a specific kernel.

**Q: How does Triton handle shared memory?**
A: The programmer doesn't manage shared memory explicitly. Triton's compiler decides what to stage in shared memory based on reuse patterns and tile sizes. This is a key productivity win — and a potential performance ceiling, since Triton's scheduling isn't always optimal for every kernel.

**Q: What does `tl.dot(a, b)` lower to?**
A: On Hopper, `wgmma.mma_async` (asynchronous warpgroup MMA). On Ampere, `mma.sync.m16n8k16`. On Volta, `hmma.884`. The Triton compiler picks the best instruction for the available hardware and data types.

**Q: What's autotuning in Triton?**
A: A decorator that registers multiple configurations (block sizes, num_warps, num_stages). On first invocation with a given `key`, Triton runs each config and caches the winner. Subsequent calls with the same key reuse the cached selection.

**Q: Compare Triton and CUTLASS.**
A: Both target high-performance GEMM-like kernels. Triton is Python, higher productivity, 90-95% performance. CUTLASS is C++ templates, full control (swizzles, pipelines, warp specialization), 100% performance. For production library code where every percentage point matters (cuBLAS alternatives, FA3), CUTLASS. For application-specific kernels where engineering velocity matters more, Triton.

**Q: Why is FlashAttention v3 written in CUTLASS rather than Triton?**
A: FA3 uses Hopper-specific features Triton didn't support at the time: warp specialization (producer/consumer warps), full TMA integration, and hand-crafted mbarrier synchronization. These are low-level enough that CUTLASS/CuTe expose them but Triton abstracts them away. As Triton gains these features, some of the gap is closing.

**Q: You wrote a Triton kernel that's 30% slower than cuBLAS. What would you check?**
A: (1) Is it using tensor cores? Inspect IR for `mma.*` instructions. (2) Are block sizes autotuned for the shape? (3) Is shared memory occupancy OK? (4) Is `num_stages` > 1 for pipelining? (5) Are strides aligned for vector loads? (6) Is the shape degenerate (e.g., K=100 which is awkward for BLOCK_K=32)? For edge-case shapes cuBLAS may be slow too; compare to theoretical peak, not just cuBLAS.

**Q: What does `num_stages` do in Triton?**
A: It enables software pipelining of the loop body. With `num_stages=N`, N iterations of the loop are inflight simultaneously: loads for iteration k+2 happen while compute for iteration k runs. More stages = better overlap but more shared memory use. Typical values 2-4.

**Q: Can you write a custom autograd op with Triton?**
A: Yes. Define forward and backward Triton kernels, wrap them in `torch.autograd.Function` or `torch.library.custom_op`. PyTorch 2.x integrates smoothly: the op composes with `torch.compile`, has C++ bindings, and can be exported via TorchScript or ONNX.

---

## 13. Further Reading

- Triton official tutorials: https://triton-lang.org/main/getting-started/tutorials/
- "Introducing Triton" (OpenAI blog, 2021)
- PyTorch 2 custom ops tutorial
- CUTLASS 3 / CuTe documentation
- "Hopper-specific optimizations in CUTLASS 3.x" (GTC talks)
- FlashAttention v2 Triton reference implementation (Tri Dao's repo)

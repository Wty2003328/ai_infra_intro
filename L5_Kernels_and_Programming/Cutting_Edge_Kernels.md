# Cutting-Edge Kernels — The 2025-2026 Kernel Programming Frontier

> **Layer:** L5.
> **Prerequisites:** [Triton_and_Kernels](Triton_and_Kernels.md), [CUDA_Optimization](CUDA_Optimization.md), [Blackwell_Architecture](../L3_Microarchitecture/Blackwell_Architecture.md).
> **Hands off to:** [Modern_Quantization_Frontier](../L6_Algorithms_and_Models/Modern_Quantization_Frontier.md), [vLLM_Internals](../L8_Inference_and_Serving/vLLM_Internals.md), [Inference_Frameworks](../L8_Inference_and_Serving/Inference_Frameworks.md).

---

## 0. The kernel frontier in 2025-2026

The gap between "what the hardware can do" and "what software extracts" has narrowed dramatically on NVIDIA Hopper and Blackwell, but only for engineers who can navigate the modern kernel stack: Triton for productivity, CUTLASS 3.x with CuTe for peak hardware utilization, and a growing ecosystem of specialized libraries (FlashInfer, DeepEP, TileLang, ThunderKittens, Liger Kernel) that encode domain-specific expertise into reusable, high-performance primitives. This page maps the entire hierarchy, explains the key technologies, and provides the mental models needed to choose the right tool and approach for any kernel-level problem.

---

## 1. The Modern Kernel-Author Stack

Every GPU kernel written in 2025-2026 lives somewhere on the abstraction hierarchy below. The choice determines development time, portability, and how close the result gets to theoretical peak.

```mermaid
%%{init: {"flowchart": {"defaultRenderer": "elk", "nodeSpacing": 60, "rankSpacing": 60, "htmlLabels": false}}}%%
flowchart TB
    subgraph STACK["Kernel Abstraction Hierarchy"]
        direction TB
        PY["PyTorch eager / compile<br/>torch.compile, TorchInductor"]
        TI["TorchInductor<br/>auto-codegen from FX graph"]
        TR["Triton 3.x<br/>Python DSL → LLVM → PTX"]
        TL["TileLang / ThunderKittens<br/>tile-level DSLs"]
        CP["CUTLASS-Python / CuteDSL<br/>Python frontend to CuTe"]
        CC["CUTLASS C++ 3.x + CuTe<br/>template metaprogramming"]
        PT["PTX / inline assembly<br/>direct hardware instructions"]
    end

    PY --> TI --> TR --> TL
    TL --> CP --> CC --> PT

    style PY fill:#4a9eff,color:#fff
    style TI fill:#3d8be0,color:#fff
    style TR fill:#2d6fbf,color:#fff
    style TL fill:#1f5a9e,color:#fff
    style CP fill:#164785,color:#fff
    style CC fill:#0e346c,color:#fff
    style PT fill:#062352,color:#fff
```

### 1.1 When to use each level

| Level | Lines of code (matmul) | Typical perf vs peak | Development time | Use when |
|---|---|---|---|---|
| PyTorch ops | 5-10 | cuBLAS/cuDNN speed | Minutes | Standard ops, no custom kernels |
| TorchInductor | 0 (auto) | 85-95% | Zero (auto) | `torch.compile` covers it |
| Triton | 50-100 | 85-95% | Hours-days | Custom ops, autotuning, moderate complexity |
| TileLang | 40-80 | 90-98% | Hours-days | Hopper/Blackwell tile pipelines |
| CUTLASS-Python | 60-120 | 95-100% | Days | CuTe expressiveness without C++ |
| CUTLASS C++ | 200-600 | 98-100% | Days-weeks | Absolute peak, production GEMM kernels |
| PTX inline asm | 500+ | 100% | Weeks | Instructions not exposed by compilers |

### 1.2 Decision flow

The practical rule: start at the highest level that could work. Escalate only when measurement shows a gap. A Triton kernel at 90% of peak that ships in a day beats a CUTLASS kernel at 98% that takes two weeks, unless the kernel is on the training or inference critical path for a multi-million-dollar cluster.

---

## 2. Triton 3.x — Updated Capabilities

> **Moved:** the full Triton 3.x feature survey now lives in [Triton_and_Kernels](Triton_and_Kernels.md) §2, next to the Triton programming model it extends.
>
> One-paragraph summary: the 3.x line (through **Triton 3.6.0**, 2026) added Hopper TMA, async `wgmma` dispatch, block pointers, FP8/FP4, `num_stages` pipelining, and — in 3.6.0 — production warp specialization, Blackwell TMEM and ragged-TMA support, AMD GFX950/GFX1250 backends, the Gluon IR framework, and MXFP scaled-dot support. Remaining gaps vs. CUTLASS: custom `mbarrier` patterns, fine-grained CuTe layout control, and full NVFP4 epilogue fusion.

---

## 3. CUTLASS 3.x and CuTe

NVIDIA's CUTLASS (CUDA Templates for Linear Algebra Subroutines) is the de facto standard for production GEMM, attention, and quantized-matmul kernels. CUTLASS 3.x (2023-2026) is a ground-up redesign built on **CuTe** (CUTLASS Tensor Expressions). The kernel dataflow and warp-group roles at a glance:

```mermaid
%%{init: {"flowchart": {"defaultRenderer": "elk", "nodeSpacing": 60, "rankSpacing": 60, "htmlLabels": false}}}%%
flowchart TD
    subgraph CUTLASS3["CUTLASS 3.x Kernel"]
        TMA["TMA Descriptor (async load)"] --> SMEM["Shared Memory (swizzled)"]
        SMEM --> WGMMA["wgmma.mma_async"]
        WGMMA --> EPI["Epilogue (fused bias/act/quant)"]
        EPI --> GMEM["Global Memory Store"]
    end
    subgraph WG["Warp Groups"]
        WG0["WG0: DMA (TMA loads)"]
        WG1["WG1-2: MMA (compute)"]
    end
    WG0 --> TMA
    WG1 --> WGMMA

    classDef hw fill:#fde68a,stroke:#b45309,color:#000
    classDef wg fill:#bfdbfe,stroke:#1d4ed8,color:#000
    class TMA,SMEM,WGMMA,EPI,GMEM hw
    class WG0,WG1 wg
```

### 3.1 CuTe layout algebra

CuTe represents every tensor as a pair `(shape, stride)` called a **layout**. Layouts compose: a tile in shared memory is a layout; a warp's slice of that tile is a layout partition; a swizzle is a layout transformation.

```cpp
#include <cute/layout.hpp>
using namespace cute;

// A 128x64 row-major tile in shared memory
auto smem_layout = make_layout(make_shape(Int<128>{}, Int<64>{}),
                               make_stride(Int<64>{}, Int<1>{}));

// A swizzled layout to avoid bank conflicts
auto swizzled = composition(Swizzle<3,3,3>{}, smem_layout);

// Partition the tile for warp 0 out of 4 warps
auto warps = make_layout(make_shape(Int<4>{}));
auto warp_tile = logical_divide(smem_layout, warps);
// warp_tile is the slice that warp_idx owns
```

Key CuTe operations:

| Operation | Meaning |
|---|---|
| `make_layout(shape, stride)` | Define a tensor layout |
| `composition(A, B)` | Compose two layouts (nested tiling) |
| `logical_divide(L, divider)` | Partition layout into groups |
| `zipped_divide(L, tile)` | Separate into inner-tile and outer-tile |
| `coalesce(L)` | Merge dimensions with stride-1 |
| `composition(Swizzle{}, L)` | Apply XOR swizzle to shared memory |

The CuTe type system is entirely compile-time (C++ template metaprogramming). The layout algebra runs at compile time, producing zero-overhead runtime indexing. Steep learning curve, but once internalized, any Hopper/Blackwell tile layout can be expressed in under 20 lines.

Two further core abstractions build on `Layout`: a **Tensor** is a Layout plus a pointer — it can be sliced, transposed, and composed; a **Tile** is a sub-layout extracted via `local_tile(tensor, tile_shape, tile_coord)`. CuTe makes explicit what Triton abstracts away, providing precise control when the autotuner cannot find a good configuration.

### 3.2 Templated GEMM structure

A CUTLASS 3.x GEMM kernel assembles from five templated components:

```mermaid
%%{init: {"flowchart": {"defaultRenderer": "elk", "nodeSpacing": 60, "rankSpacing": 60, "htmlLabels": false}}}%%
flowchart TD
    subgraph CUTLASS_GEMM["CUTLASS 3.x GEMM Pipeline"]
        SCHED["Scheduler<br/>assigns CTA tiles"]
        MAIN["Mainloop<br/>TMA load → wgmma"]
        EPI["Epilogue<br/>bias/act/quant/store"]
        SMA["Shared Memory<br/>Allocation"]
        WGR["Warp Group<br/>Partitioning"]
    end

    SCHED --> MAIN --> EPI
    SMA -. feed .-> MAIN
    WGR -. configure .-> MAIN
```

The kernel template signature (simplified):

```cpp
template <
    class ProblemShape,           // (M, N, K)
    class CollectiveMainloop,     // TMA + wgmma schedule
    class CollectiveEpilogue,     // output transformation
    class TileScheduler           // CTA-to-tile mapping
>
class Sm90GemmKernel;
```

Each template parameter is itself assembled from CuTe layout primitives. The mainloop encapsulates the entire producer-consumer pipeline: TMA descriptors for A and B tiles, shared memory staging, wgmma issue, and barrier synchronization.

### 3.3 TMA-aware mainloop

On Hopper, the mainloop uses TMA (Tensor Memory Accelerator) for asynchronous bulk tile transfers from HBM to shared memory:

```cpp
// Simplified Sm90 TMA mainloop pseudocode
template <class... Args>
struct Sm90TmaGemmMainloop {
    static __device__ void run(
        Params const& params,
        Tensor const& tCrA, Tensor const& tCrB,   // register fragments
        Tensor const& tCsA, Tensor const& tCsB,    // shared memory tiles
        TmaDescriptor const& tma_a, TmaDescriptor const& tma_b
    ) {
        // Producer warpgroup: issue TMA prefetch
        if (elect_one_sync()) {
            copy(tma_a, tCsA(_, _, k_block_next));  // async TMA copy
            copy(tma_b, tCsB(_, _, k_block_next));
        }

        // Consumer warpgroups: wgmma on current stage
        warpgroup_fence();
        gemm(tCrA, tCrB, tCc);                     // wgmma_async
        warpgroup_arrive();

        // Barrier: wait for next stage load to complete
        barrier_wait(shared_barrier, k_stage_next);
    }
};
```

The pipeline has 2-4 stages. Stage $k+1$ loads while stage $k$ computes. With TMA, only 1-2 threads issue the load (the TMA hardware does the heavy lifting), freeing the remaining threads for compute.

### 3.4 Warp specialization

CUTLASS 3.x on Hopper uses warp specialization: within a threadblock, one warp group (4 warps, 128 threads) acts as the **producer** (issues TMA loads), and one or more warp groups act as **consumers** (issue wgmma, accumulate, run epilogue).

```mermaid
%%{init: {"flowchart": {"defaultRenderer": "elk", "nodeSpacing": 60, "rankSpacing": 60, "htmlLabels": false}}}%%
sequenceDiagram
    participant P as Producer Warp Group
    participant B as Shared Memory Barrier
    participant C as Consumer Warp Group

    loop For each K tile
        P->>B: TMA copy A[k], B[k] to SMEM
        P->>B: arrive (signal ready)
        B-->>C: wait (block until data ready)
        C->>C: wgmma_async(A[k], B[k], C_acc)
        C->>B: arrive (signal done)
    end
    C->>C: Epilogue: bias + activation + store
```

This decouples load latency from compute throughput. The producer can prefetch stages ahead while the consumer saturates the tensor cores. FlashAttention v3 uses an identical pattern.

### 3.5 Epilogue fusion

The epilogue is where post-matmul operations get fused into the kernel, avoiding extra HBM round-trips:

```text
// Built-in epilogue operations in CUTLASS 3.x
using Epilogue = cutlass::epilogue::collective::Sm90TmaWarpSpecialized<
    // Output tile: accumulate then apply
    cutlass::epilogue::threadblock::EpilogueWithBroadcast<
        float,                              // accumulator type
        half_t,                             // output type
        BiasOp,                             // bias addition
        cutlass::epilogue::thread::ReLU,    // activation
        cutlass::epilogue::thread::ScaleOp  // per-channel quant scale
    >
>;
```

Common fused epilogues: bias + activation (ReLU, GELU, SwiGLU), per-row/per-column scaling (for quantization), dequantization, and direct TMA store back to HBM.

### 3.6 CUTLASS vs Triton decision flowchart

```mermaid
%%{init: {"flowchart": {"defaultRenderer": "elk", "nodeSpacing": 60, "rankSpacing": 60, "htmlLabels": false}}}%%
flowchart TD
    START["Custom kernel needed"] --> Q1{"Standard or fused GEMM?"}
    Q1 -->|Yes| TRITON["Triton + autotune"]
    Q1 -->|No| Q2{"Need TMA / warp spec?"}
    Q2 -->|Yes| Q3{"C++ templates OK?"}
    Q3 -->|Yes| CUTLASS["CUTLASS 3.x + CuTe"]
    Q3 -->|No| TRITON2["Triton (extra tuning)"]
    Q2 -->|No| Q4{"Complex reduction/scan?"}
    Q4 -->|Yes| CUDACPP["CUDA C++"]
    Q4 -->|No| TRITON

    classDef t fill:#bbf7d0,stroke:#15803d,color:#000
    classDef c fill:#bfdbfe,stroke:#1d4ed8,color:#000
    classDef n fill:#fecaca,stroke:#991b1b,color:#000
    class TRITON,TRITON2 t
    class CUTLASS c
    class CUDACPP n
```

Rule of thumb: GEMM-shaped with a standard epilogue → Triton first; TMA/warp-specialization/sparse-MMA control → CUTLASS; awkward reductions and scans → CUDA C++.

---

## 4. CUTLASS-Python / CuteDSL

NVIDIA has released a Python frontend for CUTLASS that compiles CuTe primitives into kernels without manual C++ template programming.

### 4.1 CuteDSL kernel example

```python
import cutlass
from cutlass import *

# Define a GEMM with FP16 inputs, FP32 accumulation
plan = cutlass.op.Gemm(
    element=cutlass.float16,
    layout=cutlass.LayoutType.RowMajor,
    element_accumulator=cutlass.float32,
    arch=90,  # Hopper SM90
)

# Configure the mainloop with TMA + wgmma
plan.mainloop = cutlass.gemm.collective.MainloopSm90(
    tile_shape=(128, 256, 64),
    stage_count=3,
    scheduler=cutlass.gemm.kernel.TileScheduler.StreamK,
)

# Fuse bias + activation into epilogue
plan.epilogue = cutlass.epilogue.collective.EpilogueSm90(
    bias=True,
    activation=cutlass.epilogue.thread.ReLU,
)

# Compile and run
plan.compile()
plan.run(A, B, C, bias=bias_tensor)
```

CuteDSL provides Python-level ergonomics while generating CUTLASS-quality PTX. Adoption is growing in teams that need CUTLASS performance without C++ template expertise.

### 4.2 CUTLASS-Python vs Triton

| Dimension | CUTLASS-Python | Triton |
|---|---|---|
| Backend | CUTLASS C++ templates | LLVM → PTX |
| TMA / wgmma control | Full | Partial |
| Epilogue fusion | Built-in | Manual |
| Autotuning | Manual or external | Built-in |
| Learning curve | Medium (need CuTe mental model) | Low |
| Typical perf | 95-100% peak | 85-95% peak |

CUTLASS-Python excels when the kernel structure is GEMM-like (tile load, matmul, epilogue). Triton excels for non-GEMM patterns (reductions, sorting, element-wise operations with complex control flow).

---

## 5. FlashInfer — Attention Kernels for Inference

FlashInfer (Yu et al., 2024-2025, CMU) is the dominant attention-kernel library for LLM inference. It complements FlashAttention (training-focused) with serving-specific features that no other library provides in a unified interface.

### 5.1 Architecture

```mermaid
%%{init: {"flowchart": {"defaultRenderer": "elk", "nodeSpacing": 60, "rankSpacing": 60, "htmlLabels": false}}}%%
flowchart TB
    subgraph FlashInfer["FlashInfer Architecture"]
        PREFILL["Prefill Kernels<br/>chunked, batched"]
        DECODE["Decode Kernels<br/>single-query, batched"]
        PAGE["Paged KV Manager<br/>block table dereferencing"]
        VARIANT["Attention Variants<br/>GQA / MLA / MQA"]
        SPEC["Speculative Decoding<br/>tree attention"]
        SAMPLE["Sampling Kernels<br/>top-k, top-p, min-p"]
    end

    PREFILL --> PAGE
    DECODE --> PAGE
    VARIANT --> PAGE
    SPEC --> DECODE
```

### 5.2 Key features

| Feature | Description | Why it matters |
|---|---|---|
| Paged KV cache | Takes block tables directly; no contiguous KV needed | vLLM/SGLang paged memory model |
| Variable (ragged) batch | Different sequence lengths in one batch | Avoids padding waste in serving |
| GQA / MLA / MQA | Specialized kernels for grouped/multi-head/compressed attention | Llama-3, DeepSeek-V3, Mixtral |
| FP8 KV cache | FP8 (E4M3) key-value storage | 2x KV cache capacity |
| Chunked prefill | Splits long prefill into chunks to overlap with decode | Hybrid prefill-decode serving |
| Tree attention | Verifies multiple speculative tokens in one pass | Speculative decoding backends |
| Custom attention mask | Supports block-sparse, sliding window, hybrid masks | DeepSeek-NSA, MiniMax-MoBA |

### 5.3 Usage pattern

```python
import flashinfer

# Decode: single-token attention against paged KV cache
wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
    workspace_buffer,
    kv_layout="NHD",  # [num_pages, page_size, num_heads, head_dim]
)
wrapper.plan(
    qo_indptr,         # query offsets
    kv_indptr,         # KV page offsets
    kv_last_page_len,  # valid length in last page
    num_qo_heads=32,
    num_kv_heads=8,    # GQA: 32/8 = 4 query heads per KV head
    head_dim=128,
    causal=True,
)
output = wrapper.run(q, paged_kv_cache)
```

### 5.4 FlashInfer vs FlashAttention

FlashAttention (Tri Dao lab) is the canonical training implementation. FA-v3 (Hopper) reaches ~75% of FP16 peak for bulk prefill; FA-v4 (beta, 2026) adds Blackwell support, paged KV, MLA, and ROCm. FlashInfer remains specialized for the serving regime: variable batch sizes, complex GQA/MLA decoding, and speculative verification trees. Production stacks increasingly use FA-v4 for both training and bulk prefill (now that it supports paged KV), while FlashInfer remains the choice for the decode path with complex batching and speculative verification.

| Dimension | FlashAttention (FA-v3/v4) | FlashInfer |
|---|---|---|
| Primary use case | Training, bulk prefill | Inference decode, paged KV |
| Peak perf (training) | ~75% FP16 peak on Hopper | ~65-70% (paged overhead) |
| Paged KV | No (FA3) / Yes (FA4) | Yes |
| Variable batch | Limited | Native |
| Spec decoding | No | Tree attention |
| FP8 KV | Partial (FA3) / Full (FA4) | Native |
| MLA support | Yes (FA4) | Yes |
| ROCm / AMD | Yes (FA4) | Limited |

---

## 6. FlashAttention 4 (Beta) — Blackwell and Beyond

> **Moved:** the FA4 deep dive now lives in [FlashAttention_Deep_Dive](FlashAttention_Deep_Dive.md) §6, following the FA-1 → FA-3 lineage.
>
> One-paragraph summary: FA4 (Tri Dao lab, beta 2026) adds Blackwell SM100/SM120 and AMD ROCm support, native head_dim=256, full FP8, paged KV, MLA, block sparsity, a 2CTA cross-CTA cooperation scheme, and a CuTe-DSL codebase — blurring the traditional training/inference kernel split (FlashInfer remains stronger for pure decode serving, §5.4).

---

## 7. DeepEP — MoE All-to-All

DeepSeek's open-source Expert Parallelism communication library (2025). DeepEP provides hand-tuned Hopper/Blackwell kernels for the all-to-all dispatch and combine operations fundamental to Mixture-of-Experts models.

### 7.1 Why not NCCL

Standard NCCL all-to-all is optimized for uniform message sizes. MoE routing produces **non-uniform, small messages** to many peers: each token goes to 1-8 of 256+ experts. This pattern underperforms on NCCL by 5-10x due to:

- Per-message startup overhead dominating transfer time.
- Suboptimal NVLink/IB path selection for small payloads.
- Lack of fusion between gating computation and dispatch.

### 7.2 DeepEP design

```mermaid
%%{init: {"flowchart": {"defaultRenderer": "elk", "nodeSpacing": 60, "rankSpacing": 60, "htmlLabels": false}}}%%
flowchart TD
    subgraph FWD["MoE Forward Pass"]
        GATE["Gating Network<br/>top-K expert selection"]
        DISPATCH["DeepEP Dispatch<br/>persistent all-to-all"]
        EXPERT["Expert Compute<br/>GEMM on received tokens"]
        COMBINE["DeepEP Combine<br/>reverse all-to-all"]
    end

    GATE --> DISPATCH --> EXPERT --> COMBINE

    subgraph DEEP_INTERNALS["DeepEP Internals"]
        PERSIST["Persistent kernels<br/>hold NVLink connections open"]
        FUSED["Fused gating + dispatch<br/>top-K + pack + send"]
        ASYM["Asymmetric paths<br/>NVLink intra-node / IB inter-node"]
    end

    DISPATCH -. uses .-> PERSIST
    DISPATCH -. uses .-> FUSED
    DISPATCH -. uses .-> ASYM
```

Key techniques:

- **Persistent kernels**: a long-running kernel maintains NVLink connections across dispatch calls, eliminating per-call setup overhead.
- **Fused gating + dispatch**: the top-K expert selection and token packing happen in the same kernel that initiates the all-to-all, avoiding intermediate materialization.
- **Asymmetric paths**: intra-node uses direct NVLink memcpy (bypassing NCCL); inter-node uses IB with pre-posted receives.

### 7.3 Performance

DeepSeek reports 5-10x speedup over NCCL all-to-all for typical MoE shapes (e.g., DeepSeek-V3 with 256 experts, top-8 routing, 7K tokens per microbatch). This makes MoE communication no longer the bottleneck; expert compute GEMM dominates.

### 7.4 Integration

DeepEP plugs into Megatron-LM, vLLM, and SGLang as a selectable all-to-all backend:

```yaml
# vLLM config
moe_all_to_all_backend: deepep   # or nccl, mscclpp
```

Alternatives: MSCCL++ (Microsoft, collective algorithm synthesis), NVSHMEM (GPU-native shared-memory primitives), and NCCL native all-to-all (still the default fallback).

---

## 8. TileLang — Tile-Level DSL

TileLang (BitMagic / OSU collaboration, 2024-2025) is a Python-embedded DSL that targets Hopper/Blackwell tile-level programming. Its abstraction is fundamentally different from Triton: instead of "a block of threads doing per-element work over tiles," TileLang models "tiles flowing through producers and consumers."

### 8.1 Programming model

```python
import tilelang as TL

@TL.prim_func
def matmul_kernel(
    A: TL.Buffer((M, K), "float16"),
    B: TL.Buffer((K, N), "float16"),
    C: TL.Buffer((M, N), "float16"),
):
    with TL.Kernel(M_BLOCKS, N_BLOCKS, threads=128) as (bx, by):
        # Allocate shared memory tiles
        A_shared = TL.alloc_shared((BM, BK), "float16")
        B_shared = TL.alloc_shared((BK, BN), "float16")
        # Allocate register accumulator
        C_reg = TL.alloc_fragment((BM, BN), "float32")

        TL.clear(C_reg)

        # Software-pipelined K loop
        for k in TL.Pipelined(K_BLOCKS, num_stages=3):
            # TMA-backed async copies (producer)
            TL.copy(A[by*BM:(by+1)*BM, k*BK:(k+1)*BK], A_shared)
            TL.copy(B[k*BK:(k+1)*BK, bx*BN:(bx+1)*BN], B_shared)
            # wgmma matmul on shared memory (consumer)
            TL.gemm(A_shared, B_shared, C_reg)

        # Store result
        TL.copy(C_reg, C[by*BM:(by+1)*BM, bx*BN:(bx+1)*BN])
```

TileLang compiles this directly to PTX with TMA loads and wgmma operations. The `TL.Pipelined` construct generates multi-stage software pipelines automatically.

### 8.2 Producer-consumer semantics

TileLang explicitly models the Hopper pipeline:

| Construct | Hopper mapping |
|---|---|
| `TL.alloc_shared()` | Shared memory with swizzle layout |
| `TL.alloc_fragment()` | Register tile for accumulators |
| `TL.copy()` (global → shared) | TMA async bulk copy |
| `TL.gemm()` | wgmma async warpgroup MMA |
| `TL.Pipelined()` | Multi-stage software pipeline with barriers |
| `TL.copy()` (fragment → global) | TMA store or STG |

### 8.3 Performance and adoption

TileLang generates PTX competitive with CUTLASS for common GEMM shapes (within 2-5% of peak on H100). Adoption is concentrated in research kernel labs (Tsinghua, OSU, BitMagic) but growing in production settings. The key advantage over Triton: native producer-consumer modeling means zero abstraction-layer mismatch with Hopper hardware.

---

## 9. ThunderKittens — Stanford Tile Library

ThunderKittens (Hazy Research, Stanford, 2024) is a header-only C++ library of Hopper tile primitives. It aims for CUTLASS-class performance with Triton-class code complexity.

### 9.1 Core abstractions

```cpp
#include "kittens.cuh"
using namespace kittens;

__global__ void flash_attention_kernel(
    const half* Q, const half* K, const half* V, half* O,
    int seq_len, int dim
) {
    // Shared memory tiles with built-in swizzle
    auto q_smem = SharedTile<half, 64, 64>::create();
    auto k_smem = SharedTile<half, 64, 64>::create();
    auto v_smem = SharedTile<half, 64, 64>::create();

    // Register accumulator
    auto o_reg = RegTile<float, 64, 64>::create();
    clear(o_reg);

    // Running softmax state
    auto m_i = RegTile<float, 64, 1>::create();
    auto l_i = RegTile<float, 64, 1>::create();
    fill(m_i, -INFINITY);
    clear(l_i);

    for (int kv_block = 0; kv_block < seq_len; kv_block += 64) {
        load_async(k_smem, K + kv_block * dim);   // TMA
        load_async(v_smem, V + kv_block * dim);   // TMA
        barrier();

        // QK^T in registers
        auto scores = RegTile<float, 64, 64>::create();
        mma_AB(scores, q_smem, k_smem);           // wgmma

        // Online softmax update
        auto m_new = max_each_row(scores, m_i);
        auto alpha = exp(m_i - m_new);
        mul(o_reg, o_reg, alpha);

        auto p = exp(scores - m_new);
        auto l_new = l_i * alpha + sum_each_row(p);

        // PV accumulation
        mma_AB(o_reg, p, v_smem);                  // wgmma

        copy(m_i, m_new);
        copy(l_i, l_new);
    }

    // Normalize and store
    div(o_reg, o_reg, l_i);
    store(O, o_reg);
}
```

ThunderKittens handles swizzle layouts, bank-conflict avoidance, TMA descriptor setup, and wgmma dispatch internally. The resulting FlashAttention kernel is reportedly under 200 lines and achieves >90% of Hopper peak FP16.

### 9.2 Comparison

| Dimension | ThunderKittens | CUTLASS 3.x | Triton |
|---|---|---|---|
| Language | C++ (header-only) | C++ templates | Python |
| Learning curve | Low-medium | Very steep | Low |
| Tile abstraction | First-class | Via CuTe | Via blocks |
| Code size (FA) | ~180 lines | ~500-1000 lines | ~200 lines |
| Peak perf | >90% | 98-100% | 75-85% (attention) |
| Warp specialization | Built-in | Explicit | Not supported |

Adoption is primarily educational and research-focused, with gradual production uptake.

---

## 9b. NKI (Neuron Kernel Interface) — AWS Trainium

### 9b.1 Overview

**NKI (Neuron Kernel Interface)** is Amazon's Python-based kernel DSL for programming AWS Trainium's NeuronCores. It is the only publicly available kernel DSL for a non-NVIDIA AI accelerator, making it significant both practically (for teams using Trainium instances) and as a reference point for cross-vendor kernel programming models.

### 9b.2 Programming model

NKI exposes Trainium's NeuronCore architecture through Python, providing explicit control over the accelerator's memory hierarchy and compute units:

```python
import neuronxcc.nki as nki
import neuronxcc.nki.language as nk

@nki.kernel
def matmul_kernel(
    A: nki.ndarray,  # shape (M, K), in HBM
    B: nki.ndarray,  # shape (K, N), in HBM
    C: nki.ndarray,  # shape (M, N), in HBM
):
    # Tile over output blocks
    for m_block in nki.range(0, M, tile_M):
        for n_block in nki.range(0, N, tile_N):
            # Allocate on-chip buffers
            a_sbuf = nki.dram_to_sbuf(A[m_block:m_block+tile_M, :])  # SBUF tile
            c_psum = nki.zeros((tile_M, tile_N), dtype=nki.float32)   # PSUM accumulator

            for k_block in nki.range(0, K, tile_K):
                b_sbuf = nki.dram_to_sbuf(B[k_block:k_block+tile_K, n_block:n_block+tile_N])
                a_tile = a_sbuf[:, k_block:k_block+tile_K]
                # Matrix multiply on NeuronCore's tensor engine
                c_psum += nki.matmul(a_tile, b_sbuf)

            # Write result back to HBM
            nki.sbuf_to_dram(c_psum, C[m_block:m_block+tile_M, n_block:n_block+tile_N])
```

### 9b.3 NeuronCore memory hierarchy

NKI provides explicit access to Trainium's memory tiers:

| Memory tier | API name | Description |
|---|---|---|
| **HBM** | `dram` | High Bandwidth Memory, ~1.6 TB/s per NeuronCore |
| **SBUF** (Shared Buffer) | `sbuf` | On-chip SRAM for staging tiles, similar to GPU SMEM |
| **PSUM** (Partial Sum) | `psum` | Accumulator buffer for matmul results, attached to tensor engine |

The explicit SBUF/PSUM management is similar to GPU kernel tiling, but the programmer has direct control over which buffer class each tile resides in, rather than relying on compiler heuristics for SMEM vs. register allocation.

### 9b.4 Relevance

NKI is relevant to the broader AI kernel ecosystem for several reasons:

1. **Cross-vendor perspective**: studying NKI alongside Triton and CUTLASS reveals which kernel programming concepts are hardware-specific and which are universal (tiling, double buffering, producer-consumer pipelines).
2. **AWS Trainium adoption**: teams running on Trn1/Trn2 instances use NKI to write custom operators not covered by the Neuron compiler's graph-level optimizations.
3. **NeuronCore ISA**: NKI compiles to the NeuronCore instruction set architecture, providing the same level of hardware access that PTX provides for NVIDIA GPUs.

NKI is limited to AWS Trainium hardware and does not port to NVIDIA or AMD GPUs. Its API surface is smaller than Triton's but sufficient for the matmul, attention, and element-wise kernels that dominate LLM workloads.

---

## 9c. Pallas — Google JAX Kernel DSL for TPU

### 9c.1 Overview

**Pallas** is Google's JAX-native kernel programming framework for TPU (and experimentally GPU). It extends JAX with `jax.pallas_call`, allowing developers to write custom TPU kernels that have direct access to the TPU's memory hierarchy and compute units. Pallas is the TPU counterpart to Triton for GPU — a high-level DSL that compiles down to hardware-specific instructions.

### 9c.2 Programming model

Pallas kernels are written as JAX functions with explicit memory management:

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def matmul_kernel(
    x_ref,     # Ref to input tile in VMEM
    y_ref,     # Ref to input tile in VMEM
    o_ref,     # Ref to output tile in VMEM
    acc_ref,   # Ref to accumulator in registers
):
    @pl.when(pl.program_id(2) == 0)
    def _():
        acc_ref[:] = jnp.zeros_like(acc_ref)

    # Load tiles from VMEM, compute partial matmul
    acc_ref[:] += jnp.dot(x_ref[...], y_ref[...])

    @pl.when(pl.program_id(2) == pl.num_programs(2) - 1)
    def _():
        o_ref[:] = acc_ref[:]

# Launch kernel via pallas_call
result = pl.pallas_call(
    matmul_kernel,
    out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32),
    in_specs=[
        pl.BlockSpec((BM, BK), lambda i, j, k: (i, k)),   # A tile
        pl.BlockSpec((BK, BN), lambda i, j, k: (k, j)),   # B tile
    ],
    out_specs=pl.BlockSpec((BM, BN), lambda i, j, k: (i, j)),
    grid=(M // BM, N // BN, K // BK),
)(A, B)
```

### 9c.3 TPU memory hierarchy access

Pallas provides access to the TPU's distinct memory tiers:

| TPU memory | Pallas access | Description |
|---|---|---|
| **HBM** | Implicit (via `BlockSpec`) | High-bandwidth memory, ~2.7 TB/s on TPU v5p |
| **VMEM** | `Ref` objects in kernel | Vector memory, on-chip SRAM for tile staging (~128 MB per core) |
| **CMEM** | `Ref` with CMEM annotation | Circular memory for streaming data, smaller than VMEM |
| **MXU** | `jnp.dot` | Matrix Multiply Unit — the TPU's systolic array, analogous to tensor cores |

The VMEM-to-MXU pipeline is analogous to the SMEM-to-tensor-core pipeline on NVIDIA GPUs. Pallas gives the programmer explicit control over tile residency in VMEM, similar to how CUTLASS manages SMEM tiles.

### 9c.4 Comparison with GPU kernel DSLs

| Dimension | Pallas (TPU) | Triton (GPU) | NKI (Trainium) |
|---|---|---|---|
| Language | Python (JAX) | Python | Python |
| Host framework | JAX | PyTorch / standalone | Neuron SDK |
| Tile abstraction | `BlockSpec` + `Ref` | `tl.load` / `tl.store` | `nki.dram_to_sbuf` |
| Matmul primitive | `jnp.dot` | `tl.dot` | `nki.matmul` |
| Memory model | VMEM / CMEM / MXU | SMEM / RF / TC | SBUF / PSUM / TE |
| Hardware target | Google TPU | NVIDIA GPU | AWS Trainium |
| Maturity | Experimental (JAX) | Stable, production | Stable |

Pallas is the only kernel DSL that integrates natively with a host ML framework (JAX) rather than being a standalone compilation target. This means Pallas kernels compose seamlessly with JAX's autodiff, `jax.jit`, and `jax.vmap` transformations — a significant ergonomic advantage for research use cases. However, it is currently limited to TPU and is experimental for GPU backends.

---

## 10. Liger Kernel — Production Training Kernels

Liger Kernel (LinkedIn, 2024+) is an open-source library of fused Triton kernels targeting the LLM training critical path. It provides drop-in replacements for standard transformer operations. The latest release, **v0.7.0** (2026), expands hardware support and adds new loss types and attention kernels.

### 10.1 Kernels and impact

| Kernel | Fused operations | Memory saved | Speedup |
|---|---|---|---|
| Fused RMSNorm | norm + scale + shift | 1 HBM round-trip | 5-10% |
| Fused RoPE | cos/sin + complex multiply | 2 intermediate tensors | 5-10% |
| Fused SwiGLU / GeGLU | gate + activation + multiply | 1 intermediate tensor | 10-15% |
| Fused Linear + Cross-Entropy | matmul + log-softmax + NLL loss | **entire (B,S,V) logits tensor** | 10-20% |
| Fused JSD / KL | log-softmax + divergence | intermediate tensors | 5-10% |
| CISPO / SAPO loss | truncated loss variants for RLHF | intermediate tensors | 5-10% |
| GRPO loss (Triton) | group relative policy optimization | intermediate tensors | 10-15% |
| Fused Neighborhood Attention | local window attention in Triton | KV intermediates | 5-10% |
| Sparsemax | sparse attention activation | intermediate tensors | 5-10% |

### 10.2 New in v0.7.0

| Feature | Description |
|---|---|
| Transformers v5 support | Compatible with HuggingFace Transformers v5 API |
| CISPO / SAPO loss types | Clipped importance sampling and self-adaptive policy optimization losses for RLHF/PPO training |
| GRPO loss in Triton | Group Relative Policy Optimization loss implemented as a fused Triton kernel |
| NPU (Huawei Ascend) support | Runs on Huawei Ascend NPUs via TorchNPU backend |
| XPU (Intel) support | Runs on Intel GPUs via Intel Extension for PyTorch |
| Fused Neighborhood Attention | Triton kernel for local/sliding-window attention patterns |
| Sparsemax kernel | Differentiable sparse alternative to softmax for attention sparsification |

### 10.3 The linear + cross-entropy fusion

This is the highest-impact kernel. The standard path materializes a $(B, S, V)$ logits tensor in HBM before computing loss:

$$\text{Standard: } X \in \mathbb{R}^{B \times S \times H} \xrightarrow{\text{matmul}} \text{logits} \in \mathbb{R}^{B \times S \times V} \xrightarrow{\text{cross-entropy}} \text{loss} \in \mathbb{R}$$

At $B=4, S=8192, V=128256$ (Llama-3-70B), the logits tensor is $4 \times 8192 \times 128256 \times 2 \text{ bytes} \approx 8.4 \text{ GB}$ in FP16, or $\approx 16.8 \text{ GB}$ in FP32. This is materialized and immediately consumed by the loss function.

Liger's fused kernel streams over the vocabulary dimension in shared memory, computing the matmul tile, log-softmax, and NLL loss in a single kernel pass. The $(B, S, V)$ tensor is never allocated:

$$\text{Fused: } X \in \mathbb{R}^{B \times S \times H} \xrightarrow{\text{fused kernel}} \text{loss} \in \mathbb{R}, \quad \nabla X \in \mathbb{R}^{B \times S \times H}$$

The backward pass similarly avoids materializing logits. Combined with fused RMSNorm, RoPE, and SwiGLU, Liger achieves 50% memory reduction and 10-20% training throughput improvement on large-vocabulary models.

### 10.4 Integration

```python
from liger_kernel.transformers import apply_liger_kernel_to_model

# One-line integration into HuggingFace trainer (v5 compatible)
apply_liger_kernel_to_model(model)

# RLHF training with GRPO loss
from liger_kernel.ops.grpo_loss import LigerGRPOLoss
grpo_loss = LigerGRPOLoss()
```

Liger is de facto required in production pretraining stacks for models with vocabulary >64K. It supports Llama, Mistral, Mixtral, Qwen, and Gemma architectures. As of v0.7.0, it runs on NVIDIA GPUs, Huawei Ascend NPUs, and Intel XPUs, making it the only fused-kernel library with cross-vendor GPU/NPU coverage.

---

## 11. Quantized Matmul Kernels

Quantized inference depends on hand-tuned kernels that decompress low-precision weights and compute in higher-precision activations. The kernel landscape in 2025-2026:

| Kernel | Weight format | Activation format | Hardware | Key technique |
|---|---|---|---|---|
| **Marlin** | INT4 (packed) | FP16 | Ampere+ | Tile-wide dequant, register tiling |
| **Machete** | INT4, INT8 | FP16, BF16 | Hopper | TMA + wgmma, updated Marlin |
| **CUTLASS FP8 GEMM** | FP8 E4M3 / E5M2 | FP8 / FP16 | Hopper+ | Native FP8 tensor cores |
| **CUTLASS NVFP4 GEMM** | NVFP4 (block-FP4) | FP16 / BF16 | Blackwell | 5th-gen tensor cores, TMEM |
| **GPTQ-Triton** | INT4 | FP16 | Various | Triton dequant + matmul |
| **AWQ kernels** | INT4 (activation-aware) | FP16 | Various | Per-group scaling |
| **TransformerEngine FP8** | FP8 | FP8 / BF16 / FP16 | Hopper+ | Delayed scaling, 2-queue approach |
| **BitBLAS** | Mixed (1-8 bit) | FP16 / BF16 | Various | Auto-tuned low-bit kernels |

Each format requires a custom kernel because the dequantization schedule differs: INT4 needs bitwise unpacking, FP8 uses direct tensor-core dispatch, NVFP4 requires block-level shared-exponent handling. Production inference engines (vLLM, TRT-LLM, SGLang) maintain kernel tables and select per-layer based on profiling.

### 11.1 Marlin internals

Marlin (2024) achieves near-FP16 matmul throughput with INT4 weights on Ampere/Hopper:

1. Weights stored as packed INT4 (2 values per byte).
2. Kernel unpacks 4-bit values into FP16 in registers.
3. Per-group scale applied during unpack (dequant).
4. FP16 tensor-core matmul on dequantized values.
5. Output accumulated in FP32, stored as FP16.

The dequant overhead is hidden by overlapping unpack with matmul in a software pipeline. Effective throughput is 80-90% of an equivalent FP16 GEMM while using 4x less weight memory bandwidth.

---

## 12. BitNet / 1-bit LLM Kernels

BitNet (Microsoft, 2024-2025) represents the extreme end of quantization: weights compressed to ternary (-1, 0, +1) or binary representations. Running these models efficiently requires entirely new kernel designs that replace multiplication with addition or lookup tables.

### 12.1 bitnet.cpp — Official Inference Framework

Microsoft's **bitnet.cpp** is the reference inference framework for 1-bit LLMs (BitNet b1.58 and BitNet b1.0). It provides optimized CPU and GPU kernels that exploit the extreme weight sparsity to achieve throughput gains over conventional FP16 inference.

**Kernel types across architectures:**

| Kernel type | Weight format | Compute method | Target hardware |
|---|---|---|---|
| **I2_S** | 1.58-bit ternary (2-bit packed) | Integer addition (no multiplication) | x86 (AVX2/AVX-512), ARM (NEON/SVE) |
| **TL1** | Ternary lookup (1-bit) | Lookup table-based dot product | x86, ARM |
| **TL2** | Ternary lookup (2-bit) | Lookup table with 2-bit packing | x86, ARM |

### 12.2 T-MAC lookup table methodology

The core insight of bitnet.cpp is the **T-MAC (Ternary Multiply-Accumulate via lookup)** approach: since weights are ternary (-1, 0, +1), the matrix multiply reduces to addition/subtraction. For packed representations, T-MAC uses precomputed lookup tables that map packed weight nibbles to partial sums, replacing the multiply-accumulate entirely:

$$\text{Standard: } y_i = \sum_j W_{ij} \cdot x_j \quad \text{(multiplication)}$$
$$\text{BitNet: } y_i = \sum_j \text{sign}(W_{ij}) \cdot x_j \quad \text{(addition only, where } W_{ij} \in \{-1, 0, +1\}\text{)}$$

This eliminates all weight-related multiplications, reducing the arithmetic intensity profile from compute-bound to memory-bandwidth-bound at a much lower bandwidth requirement (weights are 1-2 bits vs 16 bits).

### 12.3 GPU inference kernels

GPU kernels for BitNet were released in May 2025, extending the lookup-table approach to CUDA. The GPU kernels pack ternary weights into 2-bit containers and use warp-level shuffle instructions for efficient reduction, achieving high throughput on tensor cores by unpacking into FP16/BF16 for the activation-weighted accumulation.

### 12.4 Performance

| Metric | ARM (NEON/SVE) | x86 (AVX2/AVX-512) | GPU (CUDA) |
|---|---|---|---|
| Speedup vs FP16 | 1.37x - 5.07x | 2.37x - 6.17x | Comparable or faster than FP16 |
| Energy reduction vs FP16 | 55-82% | 55-82% | Significant (fewer ALU operations) |
| 100B model on single CPU | 5-7 tokens/sec | 5-7 tokens/sec | N/A |

The headline result: a **100B-parameter BitNet model can run on a single CPU at 5-7 tokens/sec**, making models that previously required multi-GPU clusters runnable on commodity hardware. Energy consumption drops by 55-82% compared to FP16, making 1-bit models attractive for edge and data-center inference where power is a constraint.

### 12.5 Comparison with other quantized kernels

| Dimension | BitNet / bitnet.cpp | Marlin (INT4) | FP8 GEMM |
|---|---|---|---|
| Weight bits | 1-2 bit (ternary) | 4 bit | 8 bit |
| Core operation | Addition / lookup table | Dequant + FP16 matmul | FP8 tensor core matmul |
| Hardware requirement | Any CPU or GPU | Ampere+ GPU | Hopper+ GPU |
| Quality retention | Requires 1-bit-trained model | Good for post-training quant | Good for post-training quant |
| Memory reduction vs FP16 | 8-16x | 4x | 2x |

BitNet kernels are not a drop-in replacement for standard quantized kernels: they require models specifically trained with 1-bit weight constraints. However, for models trained in the BitNet paradigm, these kernels provide an order-of-magnitude efficiency advantage.

---

## 13. Attention Kernel Variants in 2025-2026

The attention kernel landscape has fragmented into specialized variants for different model architectures and serving scenarios:

```mermaid
%%{init: {"flowchart": {"defaultRenderer": "elk", "nodeSpacing": 60, "rankSpacing": 60, "htmlLabels": false}}}%%
flowchart TB
    subgraph ATTN["Attention Kernel Landscape 2025-2026"]
        TRAIN["Training Kernels"]
        INFER["Inference Kernels"]
        SPARSE["Sparse / Structured"]
        SPEC["Speculative Decoding"]
    end

    TRAIN --> FA4["FlashAttention-v4<br/>Blackwell + ROCm, paged KV, MLA"]
    TRAIN --> FA3["FlashAttention-v3<br/>75% peak FP16, Hopper CUTLASS"]
    TRAIN --> RING["RingFlashAttn<br/>sequence-parallel training"]

    INFER --> FI["FlashInfer<br/>paged KV, GQA/MLA, FP8 KV"]
    INFER --> PA["PagedAttention<br/>vLLM block-table kernel"]
    INFER --> TREE["Tree Attention<br/>speculative decoding"]
    INFER --> CHUNK["Chunked Prefill<br/>overlaps prefill + decode"]

    SPARSE --> NSA["NSA / MoBA<br/>block-sparse attention"]
    SPARSE --> LIGHT["Lightning Attention<br/>linear attention kernels"]
    SPARSE --> SWA["Sliding Window<br/>local attention kernels"]

    SPEC --> DFLASH["DFLASH<br/>speculative decoding attention"]
    SPEC --> SUFFIX["Suffix Automaton<br/>TRT-LLM speculative"]
    SPEC --> EAGLE3["EAGLE-3 Dynamic Tree<br/>vLLM/SGLang/TRT-LLM"]
```

| Kernel variant | Architecture | Producer | Notes |
|---|---|---|---|
| FlashAttention-v4 | Blackwell SM100/SM120, Hopper, ROCm | Tri Dao | Paged KV, MLA, FP8, block sparsity, 2CTA |
| FlashAttention-v3 | Hopper CUTLASS/CuTe | Tri Dao | ~75% FP16 peak, FP8 mode |
| FlashInfer batched attn | All inference GPU | CMU team | Paged KV, ragged batch, GQA |
| PagedAttention (vLLM) | CUDA (not Triton) | vLLM team | Block-table indirect addressing |
| RingFlashAttn | NCCL + FA | Multiple | Ulysses + ring for long context |
| Tree attention | CUDA/CUTLASS | Multiple | Medusa/EAGLE speculative decoding |
| NSA kernels | CUTLASS custom | DeepSeek | Native sparse attention blocks |
| MoBA kernels | Triton custom | Moonshot | Mixture-of-block attention |
| Lightning Attention | Triton | MiniMax | Linear attention with cumulative sum |
| Sliding window attn | Triton/CUTLASS | Multiple | Local window, ignore distant tokens |
| MLA attention | CUTLASS custom | DeepSeek | Compressed KV with low-rank projection |
| FlexAttention | PyTorch native | Meta | Flexible score-mod/mask-mod, compile-based |
| DFLASH | CUDA | Research | Speculative decoding with draft-then-verify |
| Suffix automaton spec | CUDA | TRT-LLM | Suffix-automaton-based draft token generation |
| EAGLE-3 dynamic tree | CUDA/Triton | EAGLE team | Dynamic tree speculative decoding, production in vLLM/SGLang/TRT-LLM |

### 13.1 Speculative decoding kernels

Speculative decoding has emerged as a critical inference acceleration technique, and the kernel landscape has specialized accordingly:

| Technique | Mechanism | Production status | Speedup |
|---|---|---|---|
| **DFLASH** | Speculative decoding with draft-then-verify attention; verifies multiple draft tokens in a single merged attention pass | Research / early adoption | 1.5-2.5x latency reduction |
| **Suffix automaton (TRT-LLM)** | Builds a suffix automaton from the prompt to generate high-quality draft tokens without a separate draft model | Production in TensorRT-LLM | 1.3-2.0x latency reduction |
| **EAGLE-3 dynamic tree** | Learns a dynamic tree structure for draft tokens, verifying multiple tree paths in one attention pass | Production in vLLM, SGLang, TRT-LLM | 2.0-3.5x latency reduction |

The key kernel challenge in speculative decoding is **tree attention**: verifying a tree-structured batch of draft tokens against the KV cache in a single kernel launch, rather than verifying each token sequentially. FlashInfer's tree attention API and FA4's paged KV support both address this, but EAGLE-3's dynamic trees require even more flexible tree-topology handling within the attention kernel.

The key trend: attention kernels are no longer one-size-fits-all. Each model architecture and serving scenario demands a kernel tuned for its specific access pattern (paged, sparse, compressed, linear, tree-structured, speculative).

---

## 14. End-to-End: From Model Forward Pass to Hardware

```mermaid
%%{init: {"flowchart": {"defaultRenderer": "elk", "nodeSpacing": 60, "rankSpacing": 60, "htmlLabels": false}}}%%
flowchart TD
    INPUT["Input tokens<br/>shape (B, S)"]
    EMBD["Embedding lookup<br/>gather from weight matrix"]

    subgraph TRANSFORMER_BLOCK["Transformer Block (repeated N times)"]
        direction TB
        ROPE["Fused RoPE<br/>(Liger Triton kernel)"]
        QKV["QKV Projection GEMM<br/>(CUTLASS FP8 or Triton)"]
        ATTN["Attention<br/>(FA-v4 or FlashInfer)"]
        PROJ["Output Projection GEMM"]
        RMS1["Fused RMSNorm<br/>(Liger Triton kernel)"]
        FFN_GLU["SwiGLU FFN<br/>gate + up + down GEMMs<br/>(Liger fused)"]
        RMS2["Fused RMSNorm"]
    end

    subgraph FINAL["Final Layer"]
        LNORM["LayerNorm"]
        LCE["Fused Linear + Cross-Entropy<br/>(Liger: no logits tensor)"]
    end

    INPUT --> EMBD --> ROPE --> QKV --> ATTN --> PROJ --> RMS1 --> FFN_GLU --> RMS2
    RMS2 -->|repeat N-1| ROPE
    RMS2 -->|final block| LNORM --> LCE

    subgraph KERNEL_MAPPING["Kernel Selection per Op"]
        GEMM_SEL["GEMM: CUTLASS / Triton<br/>FP8 on Hopper, NVFP4 on Blackwell"]
        ATTN_SEL["Attention: FA-v4 (training, Blackwell/Hopper)<br/>FlashInfer (inference)"]
        FUSE_SEL["Fused norms/activations:<br/>Liger Triton kernels"]
        QUANT_SEL["Quantized matmul:<br/>Marlin/Machete (INT4)<br/>CUTLASS FP8 (FP8)<br/>bitnet.cpp (1-bit)"]
    end

    QKV -. kernel .-> GEMM_SEL
    ATTN -. kernel .-> ATTN_SEL
    ROPE -. kernel .-> FUSE_SEL
    FFN_GLU -. kernel .-> FUSE_SEL

    style INPUT fill:#2d6fbf,color:#fff
    style LCE fill:#1f5a9e,color:#fff
    style KERNEL_MAPPING fill:#f0f0f0,color:#333
```

The data flows through GEMM kernels (CUTLASS or Triton), attention kernels (FA-v4 or FlashInfer), and fused element-wise kernels (Liger). At each stage, the kernel selection determines the precision format, memory traffic, and achieved throughput. A production training stack for Llama-3-70B uses all three kernel categories simultaneously across the 80 transformer blocks. For 1-bit models (BitNet), the GEMM path is replaced by bitnet.cpp's lookup-table kernels, and for inference serving, EAGLE-3 speculative decoding adds a draft-then-verify attention pass using tree attention kernels.

---

## 15. Numbers to Memorize

| Quantity | Value | Context |
|---|---|---|
| H100 FP16 tensor-core peak | 990 TFLOPS (dense) | SM90, with wgmma |
| H100 FP8 tensor-core peak | 1,979 TFLOPS | FP8 E4M3 |
| B200 FP4 tensor-core peak | ~9,000 TFLOPS (dense) | NVFP4 on 5th-gen cores; ~4,500 is FP8 peak |
| H100 HBM bandwidth | 3.35 TB/s | HBM3, 80 GB |
| B200 HBM bandwidth | 8 TB/s | HBM3e, 192 GB |
| Triton matmul vs cuBLAS | 85-95% of peak | Autotuned |
| CUTLASS GEMM vs peak | 95-100% | Hand-tuned, TMA + wgmma |
| FA-v3 on H100 | ~75% of FP16 peak | Training attention |
| FA-v4 on Hopper/Blackwell | Improved over FA-v3 | 2CTA, paged KV, MLA |
| FlashInfer decode throughput | ~65-70% of peak | Paged KV overhead |
| DeepEP vs NCCL all-to-all | 5-10x faster | MoE dispatch shapes |
| Liger memory savings (linear+CE) | 50% at V=128K | Eliminates logits tensor |
| Marlin INT4 vs FP16 matmul | 80-90% throughput | 4x less weight BW |
| ThunderKittens FA lines of code | <200 lines | >90% peak |
| BitNet CPU speedup vs FP16 | 1.37-6.17x | ARM/x86, 1-bit weights |
| BitNet 100B model on single CPU | 5-7 tokens/sec | Ternary weight model |
| BitNet energy reduction vs FP16 | 55-82% | Addition replaces multiplication |
| EAGLE-3 speculative decoding | 2.0-3.5x latency reduction | Production in vLLM/SGLang/TRT-LLM |
| TMA vs cp.async bandwidth | ~20-30% higher on Hopper | Bulk tile transfer |
| wgmma vs mma.sync throughput | ~2x per instruction | 128 threads vs 32 |
| Shared memory per SM (Hopper) | 228 KB | Limits tile sizing |
| Registers per SM (Hopper) | 64K (256 KB) | Limits occupancy |
| L2 cache (Hopper) | 50 MB | Tile reuse |
| NVLink 4 bandwidth | 900 GB/s bidirectional (450 GB/s per direction) | Intra-node MoE dispatch |
| NCCL all-to-all efficiency | 10-20% of NVLink BW | For small MoE messages |

---

## 16. Reference Code Sources

| Source | URL / Location | Best for |
|---|---|---|
| FlashAttention (Tri Dao) | `github.com/Dao-AILab/flash-attention` | Reference FA-v1/v2/v3/v4, training-grade attention |
| FlashInfer | `github.com/flashinfer-ai/flashinfer` | Inference attention: paged KV, GQA, spec decode |
| CUTLASS (NVIDIA) | `github.com/NVIDIA/cutlass` | Reference GEMM, CuTe layout, all variants |
| Triton (OpenAI) | `github.com/openai/triton` | Triton 3.6.0 compiler + reference kernels |
| vLLM kernel source | `github.com/vllm-project/vllm/csrc/` | Paged attention, sampling, fused ops |
| Liger Kernel (LinkedIn) | `github.com/linkedin/Liger-Kernel` | Fused training kernels v0.7.0 (RMSNorm, RoPE, CE, GRPO) |
| ThunderKittens (Hazy) | `github.com/HazyResearch/ThunderKittens` | Educational tile DSL kernels |
| TileLang | `github.com/tile-ai/tilelang` | Tile DSL examples and compiler |
| DeepEP (DeepSeek) | `github.com/deepseek-ai/DeepEP` | MoE all-to-all kernels |
| Marlin | `github.com/IST-DASLab/marlin` | INT4 quantized matmul |
| TransformerEngine (NVIDIA) | `github.com/NVIDIA/TransformerEngine` | FP8 training, delayed scaling |
| BitBLAS | `github.com/microsoft/BitBLAS` | Auto-tuned low-bit kernels |
| bitnet.cpp (Microsoft) | `github.com/microsoft/BitNet` | 1-bit LLM inference, I2_S/TL1/TL2 kernels |
| EAGLE | `github.com/SafeAILab/EAGLE` | Speculative decoding with dynamic tree attention |

For interview preparation: read FlashAttention-v4 source (CuTe DSL patterns, 2CTA), one CUTLASS GEMM example (mainloop + epilogue structure), and Liger Kernel's fused linear-CE (training optimization).

---

## 17. Common Pitfalls

- **Reaching for CUTLASS prematurely**: the complexity tax is real. Start with Triton; escalate only with measured evidence of a gap. Triton 3.6.0's warp specialization and TMEM support cover more ground than before.
- **Re-implementing what FlashInfer ships**: before writing a new attention kernel, check FlashInfer's growing API.
- **NVCC version mismatch**: CUTLASS 3.x requires CUDA 12.4+. Older toolchains produce compile errors or silent miscompilation.
- **Using FA-v2 on Hopper instead of FA-v3/v4**: leaves 30%+ performance on the table. FA-v4's 2CTA optimization and Blackwell support make it the right choice on new hardware.
- **Skipping TMA on Hopper**: porting Ampere-style kernels without TMA wastes register slots and instruction bandwidth on address computation.
- **Ignoring NCCL fallback**: DeepEP wins for MoE shapes but is not universal. Always have NCCL as a correct fallback path.
- **Benchmarking without a reference**: assuming a kernel is fast without comparing to cuBLAS/CUTLASS. Always benchmark against the reference implementation.
- **Custom kernels without numerical tests**: silent correctness regressions in production are catastrophic. PyTorch reference + numerical tolerance check ($\text{atol} = 10^{-3}$ for FP16, $10^{-2}$ for FP8) is mandatory.
- **Underestimating register pressure**: wgmma accumulators consume 64 registers per warpgroup. Combined with shared-memory pointers and loop variables, this can push past the 255-register-per-thread limit, causing spills.
- **Using BitNet kernels on non-1-bit models**: bitnet.cpp only works with models trained from scratch with ternary weight constraints. Post-training quantization to 1-bit does not produce good results.
- **Ignoring speculative decoding kernels**: with EAGLE-3 in production across vLLM/SGLang/TRT-LLM, not using speculative decoding leaves 2-3x latency on the table for autoregressive inference.

---

## 18. References

1. NVIDIA, "CUTLASS 3.x Documentation and Examples," 2024-2026.
2. NVIDIA, "CuTe Layout Algebra," included in CUTLASS repository.
3. NVIDIA, "CUTLASS-Python / CuteDSL," 2025 release.
4. Tri Dao et al., "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Hardware-Awareness," 2024.
5. Tri Dao et al., "FlashAttention-4 (Beta): Blackwell, ROCm, MLA, and Paged KV Support," 2026.
6. FlashInfer team (Yu et al.), "FlashInfer: Efficient and Customizable Attention for Inference," 2024-2025.
7. DeepSeek-AI, "DeepEP: Efficient Expert Parallelism for Mixture-of-Experts," 2025.
8. TileLang team (BitMagic / OSU), "TileLang: A Tile-Level DSL for Hopper/Blackwell," 2024-2025.
9. Hazy Research (Stanford), "ThunderKittens: Header-Only Hopper Tile Library," 2024.
10. LinkedIn, "Liger Kernel v0.7.0: Fused Triton Kernels for LLM Training," 2024-2026.
11. IST-DASLab, "Marlin: Fast INT4 GEMM for LLM Inference," 2024.
12. OpenAI, "Triton 3.6.0: A Language and Compiler for Efficient Deep Learning," 2022-2026.
13. NVIDIA, "Hopper Tuning Guide: wgmma, TMA, Warp Specialization," 2024.
14. NVIDIA, "Blackwell Tuning Guide: NVFP4, TMEM, 5th-Gen Tensor Cores," 2025.
15. NVIDIA, "PTX ISA Reference," version 8.5+, 2024-2026.
16. Microsoft, "bitnet.cpp: Official Inference Framework for 1-bit LLMs," 2024-2025.
17. Microsoft, "T-MAC: Ternary Multiply-Accumulate via Lookup Tables for 1-bit LLMs," 2024.
18. EAGLE team, "EAGLE-3: Dynamic Tree Speculative Decoding for LLM Inference," 2025-2026.
19. NVIDIA, "TensorRT-LLM: Suffix Automaton Speculative Decoding," 2025.

---

**Up:** [Index](Index.md) | [Triton_and_Kernels](Triton_and_Kernels.md) | [CUDA_Optimization](CUDA_Optimization.md) | [FlashAttention_Deep_Dive](FlashAttention_Deep_Dive.md)

**Down:** [Modern_Quantization_Frontier](../L6_Algorithms_and_Models/Modern_Quantization_Frontier.md) | [vLLM_Internals](../L8_Inference_and_Serving/vLLM_Internals.md) | [Inference_Frameworks](../L8_Inference_and_Serving/Inference_Frameworks.md)

**Cross:** [Blackwell_Architecture](../L3_Microarchitecture/Blackwell_Architecture.md) | [GPU_Architecture](../L3_Microarchitecture/GPU_Architecture.md) | [Memory_Hierarchy_and_Roofline](../L3_Microarchitecture/Memory_Hierarchy_and_Roofline.md)

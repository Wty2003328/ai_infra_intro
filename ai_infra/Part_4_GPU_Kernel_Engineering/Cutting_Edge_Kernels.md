# Cutting-Edge Kernels — CUTLASS, CuTe, FlashInfer, DeepEP, TileLang

The 2024–2026 kernel programming stack for Hopper / Blackwell. Where Triton stops being enough, where CUTLASS 3.x and CuTe shine, what FlashInfer does for inference attention, how DeepEP makes MoE all-to-all fast, and how TileLang and ThunderKittens are reshaping kernel-author productivity.

**Prerequisites**: [CUDA_Programming](CUDA_Programming.md), [CUDA_Optimization](CUDA_Optimization.md), [Triton_and_Kernels](Triton_and_Kernels.md), [Blackwell_Architecture](../Part_2_Microarchitecture/Blackwell_Architecture.md).

---

## 1. The Modern Kernel-Author Stack

There's a hierarchy of abstraction levels, each with different productivity / control trade-offs:

```
Highest abstraction
       ▲
       │  PyTorch ops (eager / compile)
       │  TorchInductor (auto-codegen kernels)
       │  Triton (Python DSL, compiles to PTX)
       │  TileLang / ThunderKittens (tile DSLs, modern)
       │  CUTLASS-Python / CUTLASS-DSL (templated tile programming)
       │  CUTLASS C++ + CuTe (deep C++ template programming)
       │  PTX / inline assembly
       ▼
Lowest abstraction
```

The 2025 reality: you reach for whichever level matches the task. Triton or TileLang for moderate-complexity new kernels. CUTLASS / CuTe when chasing the last 10-20% on Hopper/Blackwell (TMA, wgmma, warp specialization, FP8/FP4). Production LLM stacks ship a mix.

---

## 2. Triton 3.x in 2025–2026

OpenAI's Triton has matured significantly:
- Hopper TMA support.
- wgmma async tensor-core ops.
- FP8 / FP4 (Blackwell) support emerging.
- Better autotuner / cache.
- `tl.async_copy`, `tl.barrier` async primitives.
- Block pointers and `tl.make_block_ptr` for ergonomic tile programming.

Use Triton when:
- Kernel logic is moderate complexity.
- Autotuning helps.
- You need quick iteration.

Don't use Triton when:
- You need TMA + cluster + DSMEM combos that Triton hasn't surfaced yet.
- The kernel is one of a few in your hottest path and final 10-20% matters.

Production examples: vLLM ships many Triton kernels; FlashInfer mixes Triton and CUTLASS; SGLang uses Triton heavily.

---

## 3. CUTLASS 3.x and CuTe

NVIDIA's C++ template library for high-performance dense linear algebra. CUTLASS 3 (2023–2025) is built on **CuTe**: a layout-algebra abstraction that captures arbitrary tensor shapes, strides, partitions, and tile mappings.

### 3.1 CuTe Mental Model

CuTe represents tensors as `(shape, stride)` layouts. A tile in shared memory is a layout. A warp's slice is a layout. Composing layouts gives you nested tilings, swizzles, and partitions all in one type system.

```cpp
// Simplified flavor; actual API much richer
auto smem_layout = tile_to_shape(SmemLayoutAtomA{}, Shape<_128, _64>{});
auto smem_tile = make_tensor(smem_ptr, smem_layout);
auto warp_tile = local_partition(smem_tile, warp_idx, ...);
```

Once you internalize CuTe, you can express any tile / warp / TMA / wgmma layout cleanly in C++ templates. Steep learning curve but unlocks Hopper/Blackwell performance.

### 3.2 What CUTLASS 3 Gives You

- **Templated GEMM**: `cutlass::gemm::kernel::Sm90SchedulerKernel<...>` lets you assemble custom matmul kernels by specifying tile shape, mainloop schedule, epilogue, scheduler.
- **TMA-aware mainloop**: producer-consumer style with named barriers.
- **wgmma async**: matmul issued asynchronously, software pipelined.
- **Warp specialization**: producer warp groups load with TMA, consumer warps run wgmma.
- **Epilogues**: arbitrary post-matmul ops (bias, activation, scale, quant) baked in.

Most production matmul / attention / MoE kernels in TRT-LLM and Marlin use CUTLASS 3 internally.

### 3.3 CUTLASS-Python / DSL

CUTLASS now has a Python frontend (CuteDSL) for assembling kernels from CuTe primitives without writing raw C++ templates. Much friendlier; gradually replacing Triton for performance-critical work in some shops.

---

## 4. FlashInfer — Attention Kernels for Inference

FlashInfer (Yu et al., 2024–2025) is a standalone library focused on **attention kernels for LLM inference**. It complements FlashAttention (which is pretrained / training-focused) with serving-specific features:

- **Paged KV** support: takes block tables, dereferences during attention.
- **Variable batch shape**: ragged batching with continuous KV.
- **GQA / MQA / MLA** specialized kernels.
- **FP8 KV** native.
- **Chunked prefill** efficient kernels.
- **Speculative decoding** (tree attention, draft-verifier) primitives.
- **Sampling** kernels (top-k, top-p, repetition penalty).

Used by SGLang, vLLM (V1), MLC, and others. Often the actual kernel underneath an "FA-v3 in vLLM" reference.

```python
import flashinfer
out = flashinfer.single_prefill_with_kv_cache(
    q, k_cache, v_cache,
    qo_indptr, kv_indptr, kv_last_page_len,
    causal=True, ...
)
```

### 4.1 FlashAttention vs FlashInfer

- **FlashAttention** (Dao Lab): the canonical implementation. v3 hits ~75% of Hopper peak FP16. Best for training, bulk prefill.
- **FlashInfer** (CMU + others): specialized for inference variants. Slightly different optimizations for per-step decode at small batch.

Production stacks ship both; choose per kernel call.

---

## 5. DeepEP — MoE All-to-All Done Right

DeepSeek's open-source EP communication library (2025). Hand-tuned Hopper kernels for the all-to-all dispatch / combine in MoE forward / backward.

### 5.1 Why a Specialized Library?

Generic NCCL all-to-all on top of NVLink achieves a fraction of theoretical bandwidth for the small-message-many-peer pattern that MoE produces. DeepEP rewrites this for the Hopper / Blackwell era:

- Persistent kernels that hold connections open across many calls.
- Direct NVLink memcpy primitives bypassing some NCCL overhead.
- Dispatch logic fused with gating computation.
- Asymmetric node-internal vs node-external paths.

Reported: 5-10× speedup over NCCL all-to-all for typical MoE shapes. Critical for DeepSeek-V3-class models hitting their target throughput.

### 5.2 Integration

DeepEP plugs into Megatron-LM, vLLM, SGLang as an all-to-all backend. Selected via config:
```
moe_communicator: deepep | nccl | mscclpp
```

### 5.3 Alternatives

- **MSCCL++** (Microsoft): collective compiler that synthesizes algorithms tailored to topology. Sometimes beats NCCL on niche shapes.
- **NVSHMEM**: GPU-native shared-memory primitives for arbitrary collective patterns.
- **NCCL native all-to-all**: still the default in many stacks.

---

## 6. TileLang — Tile-Level DSL

TileLang (BitMagic / OSU collab, 2024–2025) is a Python DSL targeting Hopper/Blackwell tile-level programming.

### 6.1 Mental Model

Hopper-style kernels are about **tiles** (rectangular slabs of tensors), **producers** (TMA loaders), **consumers** (wgmma operators), and **pipelines** between them. TileLang exposes these concepts directly:

```python
@T.prim_func
def matmul_kernel(A, B, C):
    with T.Kernel(M_BLOCKS, N_BLOCKS, threads=128) as (bx, by):
        A_shared = T.alloc_shared((BM, BK), "float16")
        B_shared = T.alloc_shared((BK, BN), "float16")
        C_local  = T.alloc_fragment((BM, BN), "float32")

        T.clear(C_local)
        for k in T.Pipelined(K_BLOCKS, num_stages=3):
            T.copy(A[by*BM:(by+1)*BM, k*BK:(k+1)*BK], A_shared)  # TMA
            T.copy(B[k*BK:(k+1)*BK, bx*BN:(bx+1)*BN], B_shared)  # TMA
            T.gemm(A_shared, B_shared, C_local)                   # wgmma

        T.copy(C_local, C[by*BM:(by+1)*BM, bx*BN:(bx+1)*BN])
```

Compiles to high-quality PTX competitive with CUTLASS for many shapes.

### 6.2 Why It Matters

Triton's abstraction is "block of threads doing per-element work." TileLang's is "tiles flowing through producers / consumers." Closer to how Hopper actually works → fewer abstraction-layer mismatches.

Adoption: research kernel labs (Tsinghua, OSU, BitMagic) use it heavily; production adoption growing.

---

## 7. ThunderKittens — Stanford's Tile Library

Hazy Research's ThunderKittens (2024) is a **header-only C++ library** of Hopper tile primitives — TMA loads, wgmma matmul, tile registers, swizzled SMEM layouts.

```cpp
using namespace kittens;

__global__ void matmul(const half* A, const half* B, float* C, int M, int N, int K) {
    auto a_tile = SharedTile<half, 64, 64>::create();
    auto b_tile = SharedTile<half, 64, 64>::create();
    auto c_reg  = RegTile<float, 64, 64>::create();

    // Async loads, wgmma, etc. via tile primitives
    load_async(a_tile, A_ptr);
    load_async(b_tile, B_ptr);
    barrier();
    mma_async(c_reg, a_tile, b_tile);
    ...
}
```

Goal: kernel-author productivity comparable to Triton, performance comparable to hand-tuned CUTLASS. Authors report sub-200-line FlashAttention kernels at >90% peak.

Adoption: research and educational; gradual production use.

---

## 8. Liger-Kernel — Production Triton Library

LinkedIn's open-source library of optimized training kernels (2024+):

- **Fused RMSNorm, LayerNorm, RoPE**.
- **Fused SwiGLU, GeGLU** (gate + activation + mul fused).
- **Fused linear + cross-entropy** (the big one — eliminates `(B, S, V)` logits in HBM).
- **Fused JSD, KL divergence** for RLHF.

All Triton with light CUDA. Drop-in via `from liger_kernel.transformers import apply_liger_kernel_to_llama`.

Reported: 10-20% training speedup, 50% memory reduction at large vocab. De facto required in production pretraining stacks.

---

## 9. Quantized Matmul Kernels

| Kernel | Format | Hardware | Notes |
|--------|--------|----------|-------|
| **Marlin** | INT4 weights, FP16 acts | Ampere/Hopper | vLLM-shipped; very fast |
| **Machete** | INT4, INT8 weights, FP16/BF16 acts | Hopper | TMA + wgmma based; updated Marlin |
| **GPTQ-Triton** | INT4 weights | Various | Reference impl |
| **CUTLASS FP8 GEMM** | FP8 (E4M3, E5M2) | Hopper+ | Production FP8 |
| **CUTLASS NVFP4 GEMM** | FP4 | Blackwell | Bleeding edge |
| **AWQ kernels** | INT4 acts-aware | Various | Used in TRT-LLM |
| **TilePack / Pack4bit** | INT4 packed | Various | LLM-int4 utilities |

The pattern: each format gets a hand-tuned kernel for its hardware target. Production engines select per layer based on profile.

---

## 10. Attention Kernel Variants in 2025–2026

Beyond classic FA-v3:

- **FlashAttention-3** (Hopper): 75%+ peak FP16, FP8 mode.
- **FlashInfer attention**: paged, ragged, GQA / MLA, FP8 KV.
- **xFormers**: pre-FA lineage, less common now.
- **Triton FA / FA-Triton**: reference Triton implementations.
- **PagedAttention** (vLLM csrc): handles paged-tables.
- **NSA kernels** (DeepSeek native sparse attention).
- **MoBA kernels** (Moonshot block-attention).
- **Lightning Attention** (MiniMax linear attention).
- **RingFlashAttention** for long-context training.
- **Tree attention** for Medusa / EAGLE spec decode.
- **Hopper-specific paged-attention with TMA descriptors**: hand-tuned for paged + Hopper.

If a model needs an attention variant, a kernel exists or someone is writing it.

---

## 11. Where to Look for Reference Code

| Source | Best for |
|--------|----------|
| FlashAttention GitHub (Tri Dao) | Reference FA-v1/2/3, training-grade |
| FlashInfer GitHub (CMU + others) | Inference attention |
| vLLM csrc/ | Production paged attention, sampling, fused ops |
| TensorRT-LLM examples | Hopper-tuned kernels (closed-source binaries, but configs exposed) |
| CUTLASS GitHub (NVIDIA) | Reference matmul, all the variants |
| Liger-Kernel GitHub (LinkedIn) | Production training Triton |
| Triton GitHub (OpenAI) | Triton compiler + reference kernels |
| NVIDIA DeepSeek-V3-DeepEP | EP all-to-all hand-tuned |
| Marlin / Machete (vLLM) | INT4 quant kernels |
| ThunderKittens (Hazy Research) | Educational tile-DSL kernels |
| TileLang GitHub | Tile DSL examples |

For interview prep: read FlashAttention-3 source + a CUTLASS GEMM example + Liger-Kernel's fused linear-CE. Covers the spectrum.

---

## 12. Common Pitfalls

- **Reaching for CUTLASS prematurely**: huge complexity tax. Start with Triton; only escalate if you can quantify the gap.
- **Re-implementing what FlashInfer ships**: check before writing new attention kernels.
- **Forgetting NVCC version constraints**: CUTLASS 3.x needs CUDA 12.4+; older toolchains won't compile.
- **Using FA-v2 on Hopper instead of FA-v3**: leaves 30%+ on the table.
- **Skipping TMA**: pre-Hopper kernels ported as-is to Hopper without TMA waste registers and BW.
- **Ignoring NCCL fallback for AllReduce in MoE**: DeepEP wins big but NCCL is correct fallback; have it.
- **Not benchmarking against a reference**: write a kernel; assume it's fast; ship it; learn it's 50% of CUTLASS. Benchmark always.
- **Custom kernels without tests**: silent correctness regressions kill production. PyTorch reference + numerical tolerance check is mandatory.

---

## 13. Common Interview Questions

**Q: When would you use Triton vs CUTLASS?**
A: Triton for moderate-complexity kernels, autotuning, fast iteration, and ~80-90% of peak performance. CUTLASS / CuTe for the absolute peak (TMA + wgmma + warp specialization on Hopper/Blackwell), or for production fused matmuls in TRT-LLM-class engines. CUTLASS has a steep learning curve but offers complete control.

**Q: What's CuTe?**
A: Layout algebra at the heart of CUTLASS 3. Tensors are `(shape, stride)` layouts; tiles, partitions, swizzles compose as nested layouts. Once internalized, it lets you express arbitrary tile / warp / TMA / wgmma mappings cleanly in C++ templates. Steep learning curve, very powerful.

**Q: What is FlashInfer and how does it relate to FlashAttention?**
A: FlashInfer is an attention-kernel library specialized for inference: paged KV, variable batch shape, GQA/MLA/MQA variants, FP8 KV, chunked prefill, spec decoding. FlashAttention (Dao Lab) is the canonical training/prefill implementation. Production stacks ship both — FA for training, FlashInfer for serving.

**Q: What does DeepEP do?**
A: DeepSeek's open-source EP all-to-all library. Hand-tuned Hopper kernels for MoE dispatch/combine that achieve 5-10× over NCCL all-to-all on typical MoE shapes. Critical for DeepSeek-V3-class models. Plugs into Megatron-LM, vLLM, SGLang as an all-to-all backend.

**Q: Why does Liger-Kernel give a 50% memory reduction?**
A: Its fused linear + cross-entropy kernel never materializes the `(B, S, V)` logits tensor in HBM — streams over vocab in shared memory while computing loss + gradient. At V=128K and S=8K, B=4: the logits tensor would be ~32 GB; eliminated. Plus fused norms / RoPE / SwiGLU save additional memory and compute.

**Q: What's the difference between TileLang and Triton?**
A: Triton's abstraction is "thread-block doing element-level work over tiles." TileLang's is "tiles flowing through producers (TMA) and consumers (wgmma) on Hopper-style hardware." TileLang is closer to how Hopper actually works → less abstraction mismatch for advanced kernels. Performance is competitive with CUTLASS for many shapes.

**Q: How does ThunderKittens compare to CUTLASS?**
A: Header-only C++ library of Hopper tile primitives — much shorter learning curve than CUTLASS (no template metaprogramming wizardry). Educators and research labs use it; production adoption growing. Reportedly hits >90% peak with sub-200-line FlashAttention kernels.

**Q: Walk me through the structure of a Hopper-tuned matmul kernel.**
A: Producer warps issue TMA loads of A and B tiles into SMEM. Consumer warps wait on barriers, then issue wgmma async ops accumulating into register fragments. Pipelined across N stages (typically 3-4) so producer fills stage K+1 while consumer computes on stage K. Epilogue applies bias / activation / scaling / quant. Output written via TMA store.

**Q: What's `wgmma`?**
A: Warp-Group Matrix Multiply Accumulate (Hopper). Issues an async tensor-core matmul on shared-memory operands, with the result accumulated into register fragments. 4 warps cooperating per wgmma instruction. Higher throughput than per-warp `mma.sync` (Ampere style); enables true producer-consumer pipelines via async semantics.

**Q: What's TMA and why does it matter?**
A: Tensor Memory Accelerator (Hopper). Async bulk tile transfer from HBM to SMEM with one descriptor. Frees the registers and instruction slots that pre-Hopper kernels used for address computation in `LDGSTS` chains. Critical for getting close to peak BW on Hopper — kernels without TMA leave 20-30% on the table.

**Q: When would you NOT use FlashAttention?**
A: Very small sequence length (tile / launch overhead dominates); head_dim > 256 (tile doesn't fit in SMEM); attention with custom scoring / non-softmax normalization where FlexAttention is easier; on hardware that doesn't have FA kernels for the precision you want.

**Q: How would you optimize a custom MoE all-to-all?**
A: Start with NCCL and benchmark. If 5-10× under theoretical NVLink BW, look at message size distribution (MoE produces small messages → NCCL overhead dominates). Move to DeepEP or MSCCL++. Persistent kernels, fused gating + dispatch, asymmetric paths inside-NVL vs across-IB. Verify with rail-aligned topology.

**Q: What's the right level of abstraction to write a new fused kernel today?**
A: Start in Triton — quickest path to a working kernel. If perf is within 10-20% of theoretical, ship it. If you need the last bit, escalate to CUTLASS-Python (CuteDSL) or TileLang. Drop to raw CUTLASS C++ + CuTe only when you need TMA + wgmma + cluster + DSMEM combos that higher levels don't expose yet.

---

## 14. Further Reading

- CUTLASS GitHub (NVIDIA) — examples, especially `examples/` and CuTe docs.
- "FlashAttention-3" paper (2024).
- FlashInfer GitHub and docs.
- DeepEP GitHub (DeepSeek).
- TileLang GitHub and recent papers.
- ThunderKittens GitHub (Hazy Research).
- Liger-Kernel GitHub (LinkedIn).
- Triton tutorials and blog posts.
- CUTLASS-Python / CuteDSL release notes (NVIDIA).
- "Sm90 Mainloop and Epilogue Schedules" — CUTLASS docs deep-dive.

---

**Next:** [Index](../Index.md) — updated with the full frontier section.
**See also:** [CUDA_Optimization](CUDA_Optimization.md), [Triton_and_Kernels](Triton_and_Kernels.md), [Blackwell_Architecture](../Part_2_Microarchitecture/Blackwell_Architecture.md), [Modern_Quantization_Frontier](../Part_5_Algorithms_and_Quantization/Modern_Quantization_Frontier.md).

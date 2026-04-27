# Memory Hierarchy and the Roofline Model

The single most important mental model in AI infrastructure is the **roofline**: every operation lives somewhere on a curve, and that position decides whether you're bound by FLOPS or by bandwidth. Once you can compute arithmetic intensity for an operation in your head, interviews become pattern-matching.

---

## 1. The Memory Wall

The memory wall is the observation that compute throughput (FLOPS) has grown faster than memory bandwidth (bytes/s) for decades.

| Year | Flagship GPU | Peak FP16 TFLOPS | HBM BW (TB/s) | Ridge Point (FLOPS/BW) |
|---|---|---|---|---|
| 2017 | V100 | 125 | 0.9 | 139 |
| 2020 | A100 | 312 | 2.0 | 156 |
| 2022 | H100 | 989 | 3.35 | 295 |
| 2024 | B200 | 2250 | 8.0 | 281 |
| 2025 | B300 | 2500 | 12.0 | 208 |

The "ridge point" is FLOPS / bandwidth — the arithmetic intensity at which the roofline transitions from memory-bound to compute-bound. Note how it **climbed** through 2022 (making more ops memory-bound) and **dropped** again in 2024-2025 as HBM generations caught up.

**Why the wall exists:** compute scales with transistor count (Moore's law), but memory bandwidth scales with pin count × I/O frequency. HBM's use of 1024-bit stacks brute-forces this, but each HBM stack costs power (15-20 W) and package area.

---

## 2. Arithmetic Intensity (AI)

**Definition:** Arithmetic intensity is the ratio of floating-point operations to bytes of memory traffic required by an algorithm.

    AI = FLOPs / bytes

The key word is **required** — the minimum bytes an algorithm must move between levels of the hierarchy (from HBM, typically). You can always make this worse by a bad implementation, but not better than the algorithm's intrinsic lower bound.

### 2.1 Computing AI for common operations

**Dot product of length-N vectors:**
- FLOPs: 2N (N multiplies + N adds)
- Bytes: 2N × dtype_bytes (read both vectors once)
- AI = **1 FLOP/byte** in FP16, **0.5 FLOPS/byte** in FP32

**SAXPY (y ← αx + y):**
- FLOPs: 2N
- Bytes: 2N × dtype_bytes read (x, y) + N × dtype_bytes written (y) = 3N × dtype
- AI = **0.33 FLOPS/byte** in FP16 — deeply memory-bound

**Matrix-vector multiply, (M, K) × (K,):**
- FLOPs: 2MK
- Bytes: MK × dtype (matrix) + K × dtype (vector) + M × dtype (output)
- AI ≈ **1 FLOP/byte** — memory-bound

**Dense matrix multiply, (M, K) × (K, N):**
- FLOPs: 2MKN
- Bytes: MK + KN + MN, times dtype
- AI = 2MKN / (MK + KN + MN) × dtype. For large square matmul (M=N=K=D), AI ≈ 2D/3 per byte. At D=4096 FP16: AI ≈ 2·4096/3·2 ≈ 1365 FLOPS/byte. **Highly compute-bound.**

**Attention:**
- For prefill with sequence length S and head dim d, FLOPs ≈ 4 S²d; bytes ≈ 4Sd × dtype. AI ≈ S, grows with sequence length.
- For decode step (S queries = 1, K cache length L): FLOPs ≈ 4Ld; bytes ≈ 4Ld × dtype; AI ≈ **1 FLOP/byte** — memory-bound

### 2.2 The decode problem stated precisely

Decode is a matrix-vector multiply (single token in, matrix of weights). That puts it at AI ≈ 1, far left of any ridge point. The only way to move it rightward is **batching**: N concurrent decode requests turn matrix-vector into matrix-matrix where the inner dimension is batch size.

    Batched decode: (N, 1, D) × (D, D) → (N, 1, D)
    FLOPs ≈ 2ND²
    Bytes ≈ D² (weights) + 2ND (activations)
    AI ≈ N  (for large D where weights dominate)

So batch size N directly equals the achievable AI for decode. To reach H100's ridge point (295), you'd need **batch ≈ 295**. That's why production inference systems push batch sizes as high as KV-cache capacity allows.

---

## 3. The Roofline Model

### 3.1 The plot

```
     log(GFLOPS)
          ▲
          │
   P_peak ├──────────────────────●━━━━━━━━━━━  ← compute-bound ceiling
          │                    ╱
          │                  ╱
          │                ╱  slope = peak bandwidth (bytes/s)
          │              ╱
          │            ╱
          │          ╱
          │        ╱
          └──────╱──────────────────────────────→ log(AI, FLOP/byte)
               Ridge point = P_peak / BW
```

**Formula:**

    achievable_FLOPS(AI) = min(P_peak, BW × AI)

Where:
- `P_peak` is peak compute (FLOPS/s)
- `BW` is peak bandwidth (bytes/s)
- `AI` is arithmetic intensity for the operation

### 3.2 Regimes

- **Memory-bound (left of ridge):** performance scales linearly with bandwidth × AI. To improve: reduce bytes moved (caching, fusion, compression) or increase AI (batching).
- **Compute-bound (right of ridge):** performance is capped at P_peak regardless of AI. To improve: use tensor cores, lower precision, algorithmic tricks.

### 3.3 Worked example: H100, ridge point 295 ops/byte

| Operation | Typical AI | Regime | Rough achievable FLOPS |
|---|---|---|---|
| Softmax / LayerNorm / RMSNorm | ~1 | memory-bound | ~3 TFLOPS (0.3% of peak) |
| RoPE embedding | ~1 | memory-bound | ~3 TFLOPS |
| Decode matmul, batch=1 | 2 | memory-bound | ~7 TFLOPS |
| Decode matmul, batch=64 | 128 | memory-bound | ~430 TFLOPS (43% of peak) |
| Decode matmul, batch=256 | 256 | memory-bound | ~860 TFLOPS (86% of peak) |
| Prefill matmul (seq=2048) | ~1000 | compute-bound | 989 TFLOPS (100% of peak) |
| Flash attention, prefill | ~seq_len | mostly compute-bound | 700-900 TFLOPS |

This is why "just crank the batch" is the default lever on the decode side. But batch is bounded by KV cache memory.

---

## 4. Hierarchy-Aware Analysis

The roofline above assumes a single memory level (HBM). But a real kernel moves data across a hierarchy, and the bandwidth at each level differs.

### 4.1 Multi-level roofline

You can draw a roofline per level:

| Level | Bandwidth (per SM, approx) | Latency | Capacity |
|---|---|---|---|
| Registers | ~20 TB/s effective | 0 cyc | 256 KB/SM |
| SRAM (L1/shared) | 12 TB/s | 30 cyc | 256 KB/SM |
| L2 | ~6 TB/s | 200 cyc | 50 MB chip |
| HBM3e | 3.35 TB/s | 400 cyc | 80 GB chip |

The rule of thumb: if an operation fits in shared memory with reuse, you can achieve bandwidth ≈ SRAM BW not HBM BW. This is the fundamental insight behind FlashAttention: attention naively has O(N²) HBM traffic, but with tiling you get O(N²) SRAM traffic and only O(N·d) HBM traffic.

### 4.2 Tiling math

If a matrix multiply `(M, K) × (K, N)` is tiled into `Bm × Bk` and `Bk × Bn` subtiles held in shared memory, then:

- Per tile: 2·Bm·Bn·Bk FLOPs
- Per tile: reads Bm·Bk + Bk·Bn bytes from HBM (plus write Bm·Bn)
- AI per tile = 2·Bm·Bn·Bk / (Bm·Bk + Bk·Bn + Bm·Bn) × dtype

For Bm=Bn=Bk=128 FP16: AI = 2·128³ / (3·128²·2) = 128/3 ≈ **43 FLOPS/byte** — still memory-bound on H100 (ridge 295). So you need larger tiles.

At Bm=Bn=128, Bk=64, with a *K-dimension reduction loop* that stages new K-tiles while computing: effective AI can approach ~M·N/(M+N) = the full-matrix AI. This is exactly what CUTLASS and cuBLAS do.

### 4.3 FlashAttention's tile analysis

For a tile of size `B_r × B_c` in SRAM:
- Compute per tile: 4·B_r·B_c·d (QK^T and PV)
- HBM reads: (B_r + B_c)·d × dtype
- AI = 4·B_r·B_c / (B_r + B_c) ≈ 2·min(B_r, B_c) FLOPs/byte

For B_c = 128 FP16: AI ≈ 256 FLOPS/byte — just past H100's ridge. This is why FA is compute-bound on H100 despite attention being "memory-bound" in the naive formulation. See [FlashAttention_Deep_Dive](FlashAttention_Deep_Dive.md) for the full derivation.

---

## 5. Operational Intensity vs Machine Balance

**Machine balance:** `P_peak / BW` in FLOPS/byte. For H100 FP16, this is ~295.

**Operational intensity:** your algorithm's `FLOPs / bytes`.

**Rule:** if operational intensity < machine balance → memory-bound. Else compute-bound.

Different precisions shift machine balance:
- H100 FP16: 295 FLOPS/byte
- H100 FP8: 590 FLOPS/byte (2× compute, same bandwidth)
- B200 FP16: 281 FLOPS/byte
- B200 FP4: 1125 FLOPS/byte (4× compute, same bandwidth)

So moving to lower precision raises the bar for being compute-bound. An attention tile that was compute-bound in FP16 might become memory-bound in FP4. That's why FlashAttention v3 specifically optimizes for FP8 with hardware-aware tile sizes.

---

## 6. Cache and Bandwidth Behavior on GPUs

### 6.1 HBM bandwidth is *achievable* only with coalesced access

A 32-thread warp issuing a single load of 4 bytes each = 128 bytes = one cache line. This is a **coalesced** access — one memory transaction.

If threads access scattered addresses, the hardware issues multiple transactions. Worst case: 32 transactions for 32 threads. Effective bandwidth drops 32×.

```c
// Coalesced: threads access consecutive addresses
float val = buf[threadIdx.x];           // 128 bytes, 1 transaction

// Strided: threads access non-consecutive
float val = buf[threadIdx.x * stride];   // up to 32 transactions

// Random: pointer chasing
float val = buf[indirect[threadIdx.x]]; // up to 32 transactions
```

For LLM inference, the standard contiguous layouts (row-major activations, interleaved KV cache layouts) are coalesced by construction. You mostly worry about coalescing when writing custom kernels.

### 6.2 L2 cache residency

For repeated access to a small working set, L2 stays warm. The 50 MB L2 on H100 is large enough to hold:
- A full 40 MB layer's weights (so the decode of layer L keeps L's weights warm in L2 until layer L comes back next step — but by then 79 other layers have displaced it)
- A moderate prefix cache (e.g., 100 K tokens of KV for a 70B model = 100K × 320 KB / 1024 KB per MB = too much, won't fit)

The L2 doesn't really help cross-step reuse for 70B+ models. It does help intra-step reuse of weights across the batch (all requests in a batch read the same weights).

### 6.3 Hopper's L2 residency control

Hopper added programmer control over L2. You can:
- Mark a memory range as "high residency" so it prefers staying in L2
- Allocate an "access window" for L2 hit priority

This is used by `cudaMemPrefetchAsync` with flags for persistent caching. For LLM inference, mostly used by runtime systems, not by you directly.

---

## 7. Bandwidth Efficiency Metrics

### 7.1 MFU (Model FLOPS Utilization)

**MFU = achieved FLOPS / peak FLOPS.** Applies for compute-bound workloads.

- Good training MFU: 45-55% (large dense models)
- Great training MFU: 60-70% (highly optimized kernels, minimal communication overhead)
- Ceiling (practically): ~75%

Never attainable: 100%, because real kernels always have non-MMA ops (softmax, layernorm, residuals).

### 7.2 MBU (Memory Bandwidth Utilization)

**MBU = achieved bytes/s / peak bytes/s.** Applies for memory-bound workloads.

- Good decode MBU on vLLM: 70-80%
- Great: 85%+
- Ceiling: ~90%

### 7.3 Which to report

If your workload is decode-heavy, MBU is the relevant number. MFU is a vanity metric for decode (will always be small). If training/prefill-heavy, report MFU.

In mixed inference workloads, report both or report effective tokens/sec/GPU.

---

## 8. Bandwidth Calculations for LLM Decode

### 8.1 Weight bandwidth dominates at moderate batch

For model size `W` bytes and batch `B`, decode step traffic is roughly:

    bytes_per_step ≈ W + B × (2 × KV_per_token × seq_len + activations)

Weight reads (W) are shared across the batch. KV cache reads scale with batch and sequence length.

For Llama-70B FP16 (140 GB weights, GQA, 320 KB/token KV, 4K ctx):
- Weights: 140 GB
- KV per request: 320 KB × 4096 = 1.25 GB
- At B=128: total = 140 + 160 = 300 GB/step
- At B=1: 140 + 1.25 ≈ 141 GB/step — weights dominate

On 8×H100 with TP=8 (each GPU does 140/8=17.5 GB of weight reads + its share of KV):
- B=1: 17.5 GB / 3.35 TB/s ≈ **5.2 ms/step → 190 tokens/sec per request**
- B=128: (140/8 + 160/8) / 3.35 TB/s ≈ 11.2 ms/step aggregate; 128 tokens produced per step → **11400 aggregate tokens/sec**

### 8.2 When does KV bandwidth catch up with weight bandwidth?

The crossover happens when B × KV_bytes_per_request > W.

For Llama-70B with 4K context: B × 1.25 GB > 140 GB → B > 112. At batch 112+, KV traffic exceeds weight traffic — further batching no longer amortizes decode cost as well.

For 128K context: B × 40 GB > 140 GB → B > 3.5. KV becomes the dominant memory traffic at batch 4+. This is why long-context serving is expensive.

### 8.3 KV cache quantization as a lever

Moving KV cache to FP8 halves the KV bandwidth. Recomputing the table above for Llama-70B, 128K context, KV in FP8:

- KV per request = 20 GB (half of FP16)
- Crossover batch = 140/20 = 7

You doubled your batch capacity before KV traffic dominates. See [Quantization](Quantization.md) for accuracy implications.

---

## 9. Roofline-Driven Optimization Strategies

### 9.1 Memory-bound operation menu

1. **Fuse kernels** — eliminate intermediate HBM writes/reads.
2. **Operate in lower precision** — fewer bytes per value.
3. **Quantize KV cache** — fewer bytes per token.
4. **Increase batch** — spread weight reads across more requests.
5. **Cache (prefix caching)** — skip bytes you've already read.
6. **Compress (paged KV, sparse attention)** — physically smaller working set.

### 9.2 Compute-bound operation menu

1. **Tensor cores + lower precision** — raises P_peak.
2. **Larger tiles** — better ILP, more reuse.
3. **Async execution** (TMA, wgmma) — overlap data movement with compute.
4. **Better schedules** (Triton autotune, CUTLASS) — get closer to peak.
5. **Algorithmic tricks** — e.g., Strassen-like matmul, though rarely worth it on GPUs.

### 9.3 Worked debugging flow

You have a kernel hitting 40 TFLOPS on H100 (4% of peak). What do you do?

1. **Profile.** Nsight Compute → section "Compute Workload Analysis" and "Memory Workload Analysis."
2. **Read the `SM Throughput` vs `Memory Throughput` numbers.** If SM throughput < 40% and Memory throughput > 80%, you're memory-bound. Else compute-bound.
3. **Compute AI** for your kernel. Compare to machine balance.
4. **If memory-bound:** check coalescing, try fusion with adjacent kernels, consider lower precision.
5. **If compute-bound at ~40 TFLOPS:** you're probably missing tensor cores. Check dtype, alignment, shape multiples.

---

## 10. Common Interview Questions on Roofline

**Q: Why is inference memory-bound but training is compute-bound?**
A: Training has batch size in the tens-to-hundreds and sequence length in thousands; every matmul involves batch×seq FLOPS per weight byte. Inference decode has effective "batch" of 1 token per request — orders of magnitude less compute per byte. Only when batching many concurrent decoders do you approach the compute roof.

**Q: What's arithmetic intensity, and how do I compute it for attention?**
A: AI = FLOPs / bytes. For naive attention, FLOPs = 4S²d (QK^T and PV) and bytes = 4S²d (the full attention matrix round-trips through HBM) plus 4Sd for inputs. AI ≈ 1, deeply memory-bound. For FlashAttention with tile size B_r, B_c: AI = 4·B_r·B_c / (B_r+B_c) ≈ 2·B_c. At B_c=128, AI ≈ 256, compute-bound on H100.

**Q: What is the roofline ridge point and what does it mean?**
A: Ridge point = P_peak / BW. It's the AI at which you transition from memory-bound (linear in BW) to compute-bound (capped at P_peak). On H100 FP16, ridge ≈ 295 FLOPS/byte.

**Q: If I halve my weight precision (FP16 → FP8), do I double my decode throughput?**
A: Yes if you're memory-bound (which decode is) — you halve bytes read per token. You also double the ridge point (P_peak doubles, BW stays) but that's moot when you're far left of it. In practice quantization gives 1.5-1.9× throughput depending on overheads.

**Q: Why does continuous batching help decode but not prefill?**
A: Prefill is already compute-bound with AI in the hundreds; more concurrent prefills just serialize through the same FLOPS. Decode is memory-bound with AI ≈ batch_size; batching moves AI rightward along the roofline, unlocking more of P_peak. Prefill gets <5% improvement from batching; decode can get 10-100×.

**Q: What's MFU and MBU, and when to use each?**
A: MFU = achieved FLOPS / peak FLOPS. MBU = achieved bandwidth / peak bandwidth. Use MFU for compute-bound workloads (training, prefill); use MBU for memory-bound (decode). Reporting MFU for a decode kernel at batch=1 tells you nothing — it's always ~1%.

**Q: How does L2 cache affect the roofline?**
A: The "bandwidth" in the roofline should be the bandwidth of the level your kernel actually hits. If the working set fits in L2, effective BW is L2 BW (~6 TB/s/SM), not HBM BW (3.35 TB/s). The roofline slope for L2 is steeper — more achievable FLOPS at the same AI. Kernels optimized for L2 residency (small matmuls in the weight-update phase of training, for example) can be 2-3× faster than their HBM-bound siblings.

---

## 11. Further Reading

- Williams, Waterman, Patterson. "Roofline: an insightful visual performance model for multicore architectures." CACM 2009.
- NVIDIA Nsight Compute documentation — Memory Workload Analysis and Compute Workload Analysis sections.
- Horace He, "Making deep learning go brrr from first principles" (2022).
- Andrej Karpathy's nanoGPT speed-runs and commentary.
- Tri Dao, FlashAttention papers (v1, v2, v3).

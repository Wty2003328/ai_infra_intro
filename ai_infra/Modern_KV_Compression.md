# Modern KV Compression

A focused tour of the 2024–2026 KV-cache compression toolbox: architectural (MLA, GQA, MQA), quantization (FP8/INT8/INT4 with KIVI), eviction (StreamingLLM, H2O, SnapKV, FastGen), query-aware selection (Quest), block-routing (MoBA), and the system-level integrations.

**Prerequisites**: [KV_Cache](KV_Cache.md), [Attention_Mechanisms](Attention_Mechanisms.md), [Long_Context_Engineering](Long_Context_Engineering.md), [Quantization](Quantization.md).

---

## 1. Why So Many Techniques

The KV cache scales linearly in sequence length and batch size, dominating HBM at long context or large batch. Every percent of compression directly translates to either more concurrent requests or longer contexts. This area moves fast because:

- Architectural changes (MLA) require retraining → expensive but huge wins.
- Quantization is drop-in, modest accuracy cost.
- Eviction policies are training-free, layer on top.
- Query-aware methods recover quality at extreme compression.

In production you usually stack 2-3 of these.

---

## 2. The Compression Stack

```
                    Architectural          (MLA, GQA, MQA — choose at design time)
                          │
                          ▼
                    Quantization           (FP8 / INT8 / INT4 — drop-in, ~2-4×)
                          │
                          ▼
                    Eviction               (StreamingLLM, H2O — ~2-4×, training-free)
                          │
                          ▼
                    Query-aware            (Quest, SnapKV — recovers quality)
                          │
                          ▼
                    Sparse routing         (MoBA, NSA — limit attention to subset)
                          │
                          ▼
                    Tiered storage         (HBM → CPU → NVMe → object store)
```

Each layer roughly multiplies the compression of the layers above.

---

## 3. Architectural Compression Recap

### 3.1 MQA / GQA

- **MQA** (Multi-Query): all Q heads share one K and one V. K, V dim per token: `2·d_head·b`. Crushing compression but quality loss.
- **GQA** (Grouped-Query): groups of Q heads share K, V. Llama-3-70B: H_q=64, H_kv=8 → 8× compression vs MHA.
- **Trade-off**: GQA balances quality and capacity; default in 2024+.

### 3.2 MLA (Multi-head Latent Attention) — DeepSeek

Stores per token, per layer:
- `c_KV`: low-rank latent (e.g., 512 dim).
- `k_R`: small decoupled rotary part (e.g., 64 dim) shared across heads.

Total: `(d_c + d_R)` per token per layer, roughly **30× smaller** than dense MHA, **3-4× smaller** than GQA at typical configs.

At attention time, project up to full K, V via `W_UK · c_KV` and `W_UV · c_KV`. Extra compute, but trivial vs the BW savings.

For DeepSeek-V3 (61 layers): 70 KB per token total KV vs ~600 KB naive MHA.

### 3.3 Comparison Table

| Mechanism | KV bytes/token, 70B-class | Quality cost | Inference compute cost |
|-----------|---------------------------|--------------|-------------------------|
| MHA       | ~2.5 MB (FP16)            | baseline     | baseline               |
| GQA (8×)  | ~320 KB                   | ~0.5pp MMLU  | baseline               |
| MQA       | ~80 KB                    | ~2-5pp drop  | baseline               |
| MLA       | ~70 KB                    | ~0pp         | +5-10% attention compute|

MLA is the new gold standard for new architectures. GQA remains universal in Llama-family.

---

## 4. Quantization-Based Compression

### 4.1 FP8 KV (Hopper Standard)

Cast K, V to FP8 E4M3 with per-tensor (or per-token) scales. Attention kernel reads FP8, accumulates in FP32, writes O in FP16.

- Compression: 2× vs FP16.
- Quality: typically <1% MMLU drop; long-context (Ruler, NIAH) more sensitive but usually <2%.
- Standard in vLLM, TRT-LLM, SGLang on Hopper.

### 4.2 INT8 KV (Pre-Hopper)

Per-token symmetric scales. Accuracy slightly worse than FP8 because INT8's uniform distribution doesn't capture K's exponential tails. Acceptable for short-context workloads.

### 4.3 KIVI: 2-bit KV with Asymmetric Per-Channel/Per-Token Scaling

Liu et al., 2024. Key insight from analysis of KV distributions:

- **K** has channel-wise outliers (some channels vastly larger than others). Quantize K per-channel.
- **V** is more uniform per-token. Quantize V per-token.

Stores K, V at **2 bits per element** with quantization scales. Total compression: 8× vs FP16.

Quality: ~1pp MMLU drop on long context. Better than naive 2-bit by 5-10pp.

### 4.4 GEAR / KIVI variants

Several follow-up papers (GEAR, ZipCache, MiKV) refine the K vs V quantization split, add residual streams for outliers, or use mixed precision. The pattern: 2-4 bit KV is a viable production option for capacity-constrained deployments.

### 4.5 NVFP4 / MXFP4 KV (Blackwell)

Microscaling FP4: blocks of 16 or 32 elements share an exponent. Native tensor-core support on Blackwell. KV at 4 bits with hardware-accelerated dequant. ~4× compression vs FP16, near-lossless.

---

## 5. Eviction-Based Compression

These approaches **drop tokens** from the KV cache to bound memory.

### 5.1 StreamingLLM (Xiao et al., 2023)

Observation: attention scores have a **lot of mass on the first few tokens** ("attention sinks") regardless of relevance. Naive eviction of first tokens collapses model quality.

StreamingLLM keeps:
- First **N** tokens (the sinks, e.g., N=4).
- Last **W** tokens (the recent window, e.g., W=2048).
- Drops everything in the middle.

Properly handled (with position-id rebasing for the sliding window), the model stays coherent indefinitely. KV memory bounded at `(N + W) · per_token_bytes`.

### 5.2 H2O (Heavy-Hitter Oracle)

Zhang et al., 2023. Identifies a small set of "heavy hitter" tokens (those with high cumulative attention scores) and keeps them; evicts low-score ones. Approximate the importance via running attention statistics.

Keeps quality higher than StreamingLLM at similar budgets, especially on tasks needing mid-context information.

### 5.3 SnapKV (Li et al., 2024)

Key insight: **the last few queries reveal which tokens matter for the upcoming generation**. SnapKV uses an "observation window" (last W queries) to score historical tokens by attention; keeps top-K plus a recent sliding window.

Used at prefill end to prune KV before decode. Achieves 4-8× compression with little quality loss on long-context QA.

### 5.4 FastGen / DropKV / Ada-KV / PyramidKV

Various refinements:
- Per-head adaptive eviction (different heads keep different tokens).
- Layer-wise budget allocation (attention is sparser in later layers; budget shifts to earlier).
- Pyramid: more tokens kept in early layers, fewer in late layers (PyramidKV).

These typically achieve another 1.5-3× on top of architectural compression, with some quality cost on long-context reasoning.

---

## 6. Query-Aware KV Selection

Instead of evicting permanently, **select dynamically per query** which historical tokens to attend to.

### 6.1 Quest (Tang et al., 2024)

For each new query:
1. Maintain per-token-block summary stats (max, min) of K.
2. Query computes a fast estimate of which blocks are relevant via the bounds.
3. Only top-K blocks (e.g., 16 of 64) are loaded for full attention.

Keeps **all** KV in HBM but only reads a subset per step → effectively reduces HBM bandwidth, the actual decode bottleneck.

Reported: 7-8× decode speedup at long context with <1% accuracy loss. Increasingly standard for ultra-long-context inference.

### 6.2 InfLLM / RetrievalAttention

Treat the KV cache as a vector database. For each query, retrieve top-k nearest K-vectors and attend only to them. Memory stays O(N), but **compute and bandwidth** are O(top-k).

Works because for long contexts, most attention probability concentrates on a small number of tokens. Effective for retrieval-style queries.

### 6.3 Tradeoff With FlashAttention

Naive top-K attention loses FlashAttention's tile parallelism. Production implementations (e.g., Quest's kernel) restore it via block-level selection that maps to FA tiles.

---

## 7. Sparse Routing / Block Attention

### 7.1 MoBA (Mixture of Block Attention)

Route each query to a small set of K,V **blocks** via a gating mechanism. Like MoE for attention. Blocks are persistent; gate decides which to attend to per query.

Trainable end-to-end; achieves long-context efficiency without the eviction artifact.

### 7.2 Native Sparse Attention (NSA)

Already covered in [Long_Context_Engineering §4](Long_Context_Engineering.md). Compressed + selected + window branches with trainable mixing weights.

### 7.3 Why "Trainable Sparse" Matters

Training-free eviction (StreamingLLM, H2O, SnapKV) drops quality at extreme compression because the model wasn't trained for it. NSA / MoBA train the model to use sparse attention from scratch, so quality matches dense at extreme savings.

---

## 8. Tiered Storage (Mooncake, NIXL, KV Pools)

Beyond per-replica HBM:

- **Cluster-wide KV pool**: KV cache lives in distributed CPU RAM across the fleet, accessible via fast interconnect. Mooncake (Moonshot, 2024) is the canonical example — chat-workload prefix cache hit rates jump to 80%+ with cluster pooling vs 30-50% per-replica.
- **Disaggregated KV transfer**: NIXL moves KV between prefill and decode pools or between active and standby replicas. See [Networking_and_Interconnect](Networking_and_Interconnect.md).
- **Object storage for cold prefixes**: shared system prompts, RAG document caches that don't need to reside hot.

Implementation patterns:
- **DRAM as L2**: each replica has a CPU-RAM extension for warm KV. Spills under pressure.
- **Distributed Hash Table for KV blocks**: cluster-wide consistent hashing on block hashes; lookup before recompute.
- **Locality routing**: frontends route prompts to replicas with cached prefix.

---

## 9. Combining the Stack — Worked Example

Goal: serve a 70B-class model at 1M context on 8×H100.

Per-token KV cost (FP16 baseline GQA): 320 KB.
1M tokens × 1 sequence × 320 KB = 320 GB. Way over 8×80=640 GB minus weights (~140 GB) → KV budget ~400 GB → 1.25M tokens. So one sequence fits, but no concurrency.

Stack compression:
- **GQA**: assume baseline (H_kv=8). 320 KB/tok.
- **FP8 KV**: 160 KB/tok. (× 2 capacity)
- **SnapKV / pruning**: keep effectively ~250K tokens active. (× 4 effective)
- **Quest selection**: 8× decode-time bandwidth reduction, no memory change.

Now 1M tokens at FP8+SnapKV: ~40 GB per sequence. Can fit ~10 concurrent sequences.

Or use an MLA model:
- MLA: 70 KB/tok. 1M = 70 GB. Fits 5+ concurrent sequences directly, no further compression.
- Add FP8: 35 GB per sequence. ~10 concurrent.
- Add Quest: even faster decode. Same memory.

This is roughly the math behind production long-context deployments.

---

## 10. Implementation Notes

### 10.1 In vLLM

- FP8 KV: `--kv-cache-dtype fp8` or `fp8_e4m3`. Standard.
- Prefix caching: on by default.
- Eviction-based compression: not built-in; would require custom blocks.

### 10.2 In SGLang

- RadixAttention native; great for prefix sharing.
- FP8 KV.
- Some experimental support for quantized KV.

### 10.3 In TRT-LLM

- Comprehensive KV quantization (FP8, INT8).
- Plugin architecture for custom attention.
- KV cache reuse / paging.

### 10.4 In Frameworks

- KIVI / Quest: research kernels in Triton, integrating into vLLM/SGLang ad hoc.
- StreamingLLM: trivial implementation in HF Transformers; less common in production engines (paged engines need eviction-compatible block management).
- Mooncake: not open source; concepts adopted by Dynamo and llm-d.

---

## 11. Quality Implications by Method

| Method | Compression | Long-context recall | Reasoning quality |
|--------|-------------|---------------------|-------------------|
| MQA    | 8-16×       | ~no change          | small drop        |
| GQA    | 4-8×        | ~no change          | minimal           |
| MLA    | 30×         | ~no change          | minimal           |
| FP8 KV | 2×          | minor drop          | minimal           |
| INT4 KV (KIVI) | 8× | small drop          | small drop        |
| StreamingLLM | bounded    | poor mid-context | poor reasoning    |
| H2O    | 2-4×        | moderate drop       | moderate drop     |
| SnapKV | 4-8×        | good (esp. QA)      | some drop         |
| Quest  | 1× (BW only) | good                | minimal           |
| NSA / MoBA | 4-16× (trained) | good       | good (when trained) |

Match the method to the workload:
- Chat / RAG: GQA + FP8 KV + prefix cache.
- Long-context QA: GQA/MLA + FP8 + SnapKV or Quest.
- Reasoning: MLA + FP8 + spec decode.
- 1M+ context: MLA + FP8 + Quest or NSA.

---

## 12. Common Pitfalls

- **Aggressive eviction in agentic flows**: agents often look back at early plans; StreamingLLM-style mid-eviction wrecks them.
- **Quantization without recalibration after fine-tune**: distributions shift; old scales fail.
- **Prefix-cache invalidation under FP8 mode change**: cache hashes must include dtype.
- **Quest false positives**: the bounds-based estimator can miss critical tokens; fall back to full attention if score drops.
- **KIVI on outlier-heavy distributions**: needs careful per-channel scales for K; verify on target data.
- **Tiered KV swap latency**: bringing 1 GB back from CPU during decode stalls the step. Prefetch aggressively.

---

## 13. Common Interview Questions

**Q: Why is the KV cache the dominant memory consumer at long context?**
A: It scales linearly with sequence length and batch size, while weights are fixed. At seq=128K, batch=32 for a 70B model, KV is ~80 GB while weights are 140 GB; at 1M context, KV easily exceeds weights by 10×. Compression is mandatory.

**Q: Compare MLA, GQA, and MQA.**
A: MQA shares one K and V across all Q heads — biggest compression but quality cost. GQA shares K, V across groups (e.g., 8× compression). MLA stores low-rank latent K, V (~30× compression) and reconstructs at attention time. MLA is highest compression and roughly preserves quality.

**Q: How does StreamingLLM bound KV memory and where does it fail?**
A: Keeps the first N "attention sink" tokens plus the last W recent tokens; drops the middle. Bounded memory regardless of generation length. Fails when the model needs to retrieve mid-context information (e.g., long-doc QA, multi-step reasoning over a long history).

**Q: What is SnapKV and how does it differ from H2O?**
A: SnapKV: at prefill end, use the last few queries (an observation window) to score historical tokens by attention and keep top-K. H2O: maintain running attention statistics during generation, evict lowest-score tokens periodically. SnapKV is one-shot pruning; H2O is online eviction.

**Q: What is Quest?**
A: Query-aware KV selection. Keeps per-block summary statistics; for each query, fast-estimates which KV blocks are relevant and reads only those for full attention. All KV stays in HBM; only the read bandwidth shrinks. Up to 8× decode speedup at long context with <1% quality loss.

**Q: Why does FP8 KV usually outperform INT8 KV on long context?**
A: K's distribution has wide dynamic range with channel outliers; FP8's exponent handles this naturally. INT8's uniform spacing forces a per-channel scale to absorb outliers, losing precision elsewhere. Hopper's hardware-fused FP8 scaling makes it nearly free.

**Q: Walk me through KIVI.**
A: K is quantized per-channel (channels have outliers, so per-channel is better), V per-token (tokens are more uniform). 2 bits per element with asymmetric scales (zero-point + scale). Implements 8× KV compression with ~1% MMLU drop on long context.

**Q: How do native sparse attention methods (NSA, MoBA) differ from training-free eviction?**
A: NSA / MoBA bake sparsity into the architecture and training — the model learns to use it. Eviction methods (StreamingLLM, H2O, SnapKV) operate at inference on a model trained dense, accepting some off-distribution behavior. Trained-sparse achieves the same compression at higher quality.

**Q: How does Mooncake achieve high prefix-cache hit rates?**
A: Distributes the KV pool across the fleet's CPU RAM. Locality-aware routing pulls cached prefixes to active replicas via fast transport. A request's prefix may be cached on **any** replica's host RAM, so hit rate scales with cluster size, not per-replica RAM.

**Q: When would you not use KV quantization?**
A: Latency-critical workloads where the 2× capacity isn't needed (small batch, short context). Models very sensitive to KV precision (some math-heavy reasoning models). When kernels for the target precision don't exist on your hardware.

**Q: How would you serve a 1M-context Llama-4 model on a single 8-GPU node?**
A: FP8 KV; chunked prefill (32K chunks); aggressive prefix caching for shared documents; potentially Quest for decode-time selection. Llama-4 already uses iRoPE + chunked attention internally. Cap concurrent 1M-prompt sequences explicitly to avoid KV exhaustion.

**Q: How do you stack KV compression methods?**
A: Architectural (MLA/GQA) is multiplicative with quantization (FP8/INT4). On top, eviction (SnapKV) or selection (Quest) is multiplicative again on the sequence dim. So MLA + FP8 + Quest can compound to 100× effective decode-bandwidth reduction vs FP16 MHA.

**Q: What's the failure mode of Quest's bounds-based selector?**
A: It estimates relevance from per-block max/min K. Adversarial or rare cases produce false negatives — relevant tokens whose bounds aren't tight enough to be selected. Mitigations: keep a small "always-attend" recent window; fall back to full attention for low-confidence steps.

---

## 14. Further Reading

- KIVI paper (Liu et al., 2024).
- StreamingLLM paper (Xiao et al., 2023).
- H2O paper (Zhang et al., 2023).
- SnapKV paper (Li et al., 2024).
- Quest paper (Tang et al., 2024).
- FastGen, GEAR, ZipCache, PyramidKV — recent KV compression survey.
- DeepSeek-V2/V3 papers (MLA).
- Mooncake paper (Moonshot, 2024).
- Native Sparse Attention (DeepSeek, 2025).
- MoBA paper (Moonshot, 2025).

---

**Next:** [State_Space_Models_and_Hybrids](State_Space_Models_and_Hybrids.md), [Disaggregated_Serving_2025](Disaggregated_Serving_2025.md).
**See also:** [KV_Cache](KV_Cache.md), [Long_Context_Engineering](Long_Context_Engineering.md), [Quantization](Quantization.md).

# Long Context Engineering

How modern models handle 128K, 1M, and 10M+ context windows. Covers position-encoding extension (YaRN, LongRoPE, NTK), chunked / sliding-window / global hybrid attention, ring attention, native sparse attention (NSA, MoBA), and the inference-time mechanics — KV pressure, prefix caching, hierarchical compression, retrieval interplay.

**Prerequisites**: [Attention_Mechanisms](Attention_Mechanisms.md), [KV_Cache](KV_Cache.md), [Modern_KV_Compression](Modern_KV_Compression.md), [Parallelism_Strategies](Parallelism_Strategies.md).

---

## 1. Why Long Context Is Hard

Two scaling problems:

1. **Quadratic compute** of vanilla attention: `O(N²·d)` per layer. At N=1M, this is intractable per-layer.
2. **Linear KV memory**: O(N·L·H_kv·d) per sequence. At N=1M for Llama-3-70B, that's 320 GB per sequence — bigger than a node's HBM.

Plus a quality problem:

3. **Position generalization**: a model trained on 8K context performs poorly at 1M without specific extension techniques.

The 2024–2026 toolbox attacks all three.

---

## 2. Position-Encoding Extensions

### 2.1 RoPE Recap

Rotary Position Embedding rotates each pair of channels in Q and K by an angle θ_m = m · base^{-2k/d}, where m is position. Dot products `q_m · k_n` depend only on (m − n), giving relative position behavior.

The **base frequency** controls the wavelength of slow-rotating channels. Default for Llama-2: base=10000.

### 2.2 The Extension Problem

A model trained at base=10000, max_position=4096 generalizes poorly to position=32K — the slow channels are operating at frequencies the model has never seen.

### 2.3 NTK-Aware Scaling (PI / NTK)

Position Interpolation: scale positions down by `1/k` so position k·N maps to N (where N is training length). Equivalent to evaluating attention at original positions but with denser inputs.

NTK-aware: don't scale all frequencies uniformly. Scale slow-rotating channels more, fast-rotating less, preserving high-frequency information.

### 2.4 YaRN (Yet another RoPE extensioN)

Refines NTK with **piecewise scaling**:
- Low-frequency channels (long wavelengths): scale by `s = N_target / N_train`.
- High-frequency channels: don't scale.
- Mid-frequency: smooth interpolation.

Plus **attention temperature scaling**: scale the attention logits by `1/sqrt(s)` to compensate for changes in the distribution.

YaRN with brief fine-tuning (a few billion tokens) extends a 4K-trained Llama-2 to 128K with minimal quality loss. Standard for production extension.

### 2.5 LongRoPE (Microsoft, 2024)

Search-based: use evolutionary search to find per-channel scaling factors that minimize perplexity at the target length. Goes beyond YaRN's heuristic, finds non-uniform scaling that works to 2M+ tokens.

### 2.6 Llama-3.1+ "RoPE Theta=500K"

Llama-3.1 onward uses a much larger base (500K) directly in pretraining. This encodes positions natively across long ranges without requiring extension at fine-tune time. 128K context is "free" in pretraining.

### 2.7 Llama-4 iRoPE (interleaved RoPE/NoPE)

Every fourth attention layer uses **no positional encoding** (NoPE — Kazemnejad et al., 2023). The remaining layers use RoPE with extended base. NoPE layers, surprisingly, generalize better to lengths beyond training. The mix achieves 10M context.

Why NoPE works: with appropriate causal masking, the model can learn implicit position from token identity sequences. Specific layers acting "position-agnostic" generalize length-wise better than uniform RoPE.

---

## 3. Sparse / Local / Global Attention Patterns

### 3.1 Sliding-Window Attention (SWA)

Each query attends to the last W tokens only. Cost O(N·W·d) per layer instead of O(N²·d). Used by Mistral (W=4096), Gemma-3 (W=4096 in 5/6 layers), Phi-3-medium.

KV per token still grows linearly with N, but **the kernel cost per step is bounded**. KV memory bounded only if sliding-window also evicts old KV (some implementations do).

Limitation: information from beyond W tokens cannot reach the query directly. Need either layer stacking (information walks W steps per layer) or global layers.

### 3.2 Mixed Local + Global Layers

Gemma-3 pattern: **5 local layers, 1 global layer**, repeated. The global layer attends over the full context; local layers stay bounded.

Effective long-context cost per step:
```
O(L_local · N · W + L_global · N²)
```
At W=4096, L_global=L/6, L=80, N=128K:
- Local: 80·5/6 · 128K · 4K = 22 GFLOPs per layer-block
- Global: 80·1/6 · 128K² ≈ 175 GFLOPs per layer-block

Global layers dominate at very long N. To push further, the global layers themselves need sparse / chunked / NSA-style attention.

### 3.3 BigBird / Longformer Patterns (legacy)

Local + global tokens + random tokens. Less common in 2025 but the pattern survives in some long-document models.

### 3.4 Chunked Attention

Llama-4 long-context layers chunk the sequence into segments. Within a chunk, full attention; across chunks, anchor or summary tokens carry information. Cost O(N·chunk_size·d) per layer.

---

## 4. Native Sparse Attention (NSA)

DeepSeek's NSA paper (early 2025) introduces **trainable sparse attention** that runs as a single kernel competitive with FlashAttention.

### 4.1 NSA Decomposition

For each query position, compute attention via three branches and combine:

1. **Compressed branch**: down-sample K,V across positions (block-pooled) → coarse global view.
2. **Selected branch**: top-K most relevant blocks (by compressed-attention score) → precise critical-block attention.
3. **Sliding-window branch**: recent tokens for local detail.

```
output = w1 · attn_compressed + w2 · attn_selected + w3 · attn_window
```
where weights w1, w2, w3 are learned.

### 4.2 Properties

- **End-to-end trainable**: not just an inference shortcut.
- **Hardware-efficient**: NSA's kernel achieves throughput close to FlashAttention at long context.
- **Drop-in**: replaces full attention in the architecture.
- Reported quality: matches or exceeds full attention at 64K-128K with much lower compute and memory.

### 4.3 MoBA (Mixture of Block Attention)

Concurrent work (Moonshot, 2025): each query routes to a small set of K,V **blocks** (gated like MoE). Reduces effective attention from N to a few blocks per query.

### 4.4 Why These Matter for Systems

NSA / MoBA make 1M+ context tractable at training time, not just inference. Combined with MLA / sliding-window, they shape the next generation of long-context architectures.

---

## 5. Ring Attention (Distributed Long-Context)

For training/inference on contexts that don't fit on one GPU's HBM:

```
Each rank i owns a contiguous chunk of Q, K, V.
Round 1: rank i computes attention of its Q against its own K,V chunk.
Round 2: rank i sends its K,V chunk to rank (i+1), receives from (i-1), computes against received.
…
After P rounds (P = ring size), each Q chunk has attended to all K,V.
```

Each rank maintains running online-softmax stats; combines across rounds.

### 5.1 Bandwidth Requirement

Per layer per round, send `2 · chunk · d · b` bytes (K and V). At total seq N, P ranks, chunk = N/P:
```
bytes_per_round = 2 · N/P · d · b
total per layer  = (P-1) · 2 · N/P · d · b  ≈ 2 · N · d · b
```

Independent of P at large P. The ring overlaps comm with compute almost perfectly.

### 5.2 Production Use

- Training ultra-long-context models (1M+).
- Inference on huge prompts when KV doesn't fit per-GPU. Less common because most production prompts don't exceed HBM after MLA / GQA reductions.
- Combined with FlashAttention as **Ring FlashAttention** (Liu et al., 2023).

### 5.3 Striped / Zigzag Variants

Standard ring is load-imbalanced for **causal** attention: early-position queries attend to fewer keys than late-position queries. Striped ring partitions interleaves Q across ranks so each rank gets a balanced workload.

---

## 6. Chunked Prefill at Long Context

For a 1M-token prompt:
- Naive prefill: one giant forward pass — won't fit; activation memory blows up.
- Chunked prefill: split into chunks of e.g. 8K-32K, prefill chunk by chunk.

Each chunk attends to **all previous KV** (already computed and cached). The kernel runs `(chunk_size, prev_KV_size)` attention. With FlashAttention, this is efficient.

For 1M prompt at chunk=32K: 32 chunks. Each chunk processes 32K queries against a growing KV. Total compute: same as one-shot prefill; spread across steps to interleave with decode of other requests.

Implication: long-context prefill takes seconds-to-minutes wall-clock. TTFT for 1M-prompt requests is fundamentally higher; UX must reflect this (streaming "processing prompt…" indicators).

---

## 7. KV Compression at Long Context

Long context magnifies KV memory pressure → KV compression becomes critical.

Modern approaches (full treatment in [Modern_KV_Compression](Modern_KV_Compression.md)):

- **MLA**: ~30× compression, architectural.
- **GQA / MQA**: 4-8× shared K,V heads.
- **FP8 / INT4 KV**: 2-4× via quantization.
- **StreamingLLM**: keep first N "anchor" tokens + last W tokens, drop middle.
- **H2O / SnapKV**: drop low-attention tokens.
- **Quest**: query-aware KV selection at each step.
- **KIVI**: 2-bit KV with per-channel/per-token scales.

Stack them: MLA + FP8 KV + sliding-window architecture → Llama-4-Scout-style 10M context fits on a single node.

---

## 8. Prefix Caching at Long Context

Long shared prefixes (system prompts, RAG documents, persistent agent context) are exactly where prefix caching shines:

- A 50K-token system prompt cached once serves thousands of requests.
- A 200K-token document context (e.g., a long paper) reused across queries skips its prefill entirely after the first request.

vLLM's hash-based and SGLang's RadixAttention both work. The radix tree handles partial-block matches better at long context.

Production gains:
- Long-doc RAG: 70-95% prefix-cache savings.
- Agentic workflows reusing context: 60-80%.

This effectively makes long-context inference much cheaper than the per-token math suggests.

---

## 9. Hierarchical / Multi-Tier KV

For ultra-long-context production at scale, KV is moved across tiers:

```
HBM       (hot)      ← active sequences' KV
Host RAM  (warm)     ← preempted / paused sequences
NVMe      (cold)     ← long-term agent contexts, RAG warm pool
Object store (frozen) ← multi-tenant prefix cache
```

Mooncake (Moonshot, 2024) introduced a global KV pool spanning a cluster's CPU RAM, with prefix caching at cluster scale (not per-replica). Hit rates on chat workloads jumped to 80%+ because all replicas see all prefixes.

Tradeoff: tier transitions cost bandwidth (swap-in latency). Schedulers route requests to replicas with cached prefixes (locality-aware) to keep the hot tier hit.

---

## 10. Sequence / Context Parallelism (CP / SP)

For training: see [Parallelism_Strategies §6](Parallelism_Strategies.md). Ring attention is CP applied to attention.

For inference: rare. Most long-context inference scenarios are amenable to per-GPU full-sequence attention (after MLA, sliding-window, etc.). When KV truly doesn't fit on one GPU, use TP-aware KV sharding (KV by heads — natural with TP) or split KV by sequence chunks.

---

## 11. Long-Context Inference Engineering Checklist

For serving long-context workloads:

- **Use a long-context model**: sliding-window + global / MLA / NSA. Don't try to extend a 4K model to 1M.
- **Enable FP8 KV**: 2× capacity for free on Hopper.
- **Use chunked prefill**: 8K-32K chunks for prefill smoothness.
- **Aggressive prefix caching**: hash-based for blocks, radix for partial.
- **Tiered KV** if at scale: spillover to host RAM with smart prefetch.
- **Disaggregate prefill and decode**: prefill of 1M is expensive; don't entangle with decodes.
- **Plan for slow TTFT**: educate clients, emit progress signals.
- **Long-tail SLOs**: a 10% tail of 1M-prompt requests destroys p99 unless segregated.

---

## 12. Quality at Long Context — How to Measure

Standard benchmarks:

- **Needle in a Haystack (NIAH)**: insert a fact at random position in a long context, ask the model to retrieve it. Tests literal recall.
- **RULER** (NVIDIA, 2024): comprehensive long-context benchmark with 13 tasks (NIAH variants, multi-hop, aggregation, reasoning).
- **LongBench, ZeroSCROLLS**: realistic long-doc QA, summarization.
- **BABILong** (FAIR): synthetic reasoning at long range.

Quality often degrades sharply past the trained length. Llama-3.1's 128K is solid; pushing further with YaRN/LongRoPE works but accuracy drops gradually. Production deployments often cap at the validated length.

---

## 13. Common Pitfalls

- **Treating 1M context as N=1M attention everywhere**: it's typically MLA or sliding-window. Read the model card.
- **Forgetting chunked prefill**: 1M prompt OOMs without it.
- **Naïve YaRN to extreme lengths**: works to ~10× training length, then quality cliffs. Use LongRoPE for further.
- **Prefix-cache invalidation across model upgrades**: cache lives in token-id hash space; tokenizer changes invalidate everything.
- **NIAH-pass-fail thinking**: passing NIAH at 1M doesn't mean general 1M reasoning. Use multi-task benchmarks.
- **TTFT SLOs designed for chat workloads**: do not apply to long-context. Set separate SLOs for long-prompt requests.

---

## 14. Common Interview Questions

**Q: How do you extend a 4K-trained model to 128K context?**
A: Apply YaRN or LongRoPE position-encoding extension, then briefly fine-tune (a few billion tokens at the new context length). Quality loss is small with proper extension. Recent models (Llama-3.1+) train natively at 128K with elevated RoPE base.

**Q: What's iRoPE in Llama-4?**
A: Interleaved RoPE / NoPE: every fourth attention layer has no positional encoding at all, the rest use RoPE. NoPE layers generalize better to lengths beyond training. The mix supports 10M context.

**Q: Compare sliding-window attention and full attention at long N.**
A: Full: O(N²·d) per layer, all-tokens-to-all. Sliding: O(N·W·d) per layer, capped attention range W. Sliding is much cheaper but loses global mixing — must be combined with global layers (Gemma-3 5:1 ratio) or work-around in subsequent layers.

**Q: How does NSA work and why is it efficient?**
A: Native Sparse Attention (DeepSeek). Three trainable branches per query: compressed (block-pooled), selected (top-K blocks by score), sliding window (recent). Combined with learned weights. End-to-end trainable, drop-in for full attention, hardware-efficient (single fused kernel). Quality matches full attention at long context with much less compute.

**Q: Walk me through ring attention.**
A: Each rank holds a chunk of Q, K, V. Each round, compute attention of local Q against current K,V chunk; rotate K,V to next rank. After P rounds, every Q has seen all K,V. Use online softmax to combine partial stats. Bandwidth per round is independent of P at large P; comm overlaps with compute.

**Q: How would you deploy a 1M-context model on 8×H100?**
A: Use MLA architecture (e.g., DeepSeek-V3) so KV/token is tiny. FP8 KV cache. Chunked prefill (32K chunks). Aggressive prefix caching for shared documents. Disaggregate prefill from decode. Set long-context-specific SLOs. With these, 1M prompts at modest concurrency are feasible per node.

**Q: Why is chunked prefill required at long context?**
A: Naive single-shot prefill of 1M tokens has huge activation memory and one giant compute step. Chunking into 8K-32K pieces (a) bounds activation memory, (b) lets decodes interleave between chunks, (c) makes scheduling smoother. Same total FLOPs as one-shot.

**Q: What's the difference between YaRN and LongRoPE?**
A: YaRN uses heuristic piecewise frequency scaling (low/mid/high) plus attention temperature scaling. LongRoPE uses evolutionary search to find per-channel scaling factors that minimize perplexity at target length. LongRoPE pushes further (2M+) at the cost of search.

**Q: How does prefix caching help long-context inference?**
A: Long shared prefixes (system prompts, document context, agent state) are exactly the scenarios where prefix caching skips the expensive prefill. Chat / RAG / agent workloads commonly see 60-95% prefix-cache hit rates, making the marginal cost of long prompts much lower than per-token math suggests.

**Q: What's Mooncake?**
A: Moonshot's cluster-scale KV pool architecture: KV cache lives in a global tier across all replicas (CPU RAM), with locality-aware routing pulling cached prefixes to active replicas. Achieves 80%+ prefix-cache hit rate at scale where per-replica caches would only see 30-50%. NIXL-style fast transport.

**Q: How do you benchmark long-context quality?**
A: Use a multi-task suite. NIAH for literal recall, RULER for diverse tasks (multi-hop, aggregation, reasoning), LongBench for realistic doc QA, BABILong for long-range reasoning. Don't rely on NIAH alone — passing NIAH ≠ general long-context capability.

**Q: Why does model quality degrade beyond the trained length even with YaRN?**
A: The model has no training signal at those positions; YaRN extrapolates the position encoding but learned attention patterns weren't exposed to those distance scales. Fine-tuning at the target length closes most of the gap. Limit: training compute scales with target length.

---

## 15. Further Reading

- "YaRN: Efficient Context Window Extension of Large Language Models" (Peng et al., 2023).
- "LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens" (Ding et al., 2024).
- "Native Sparse Attention" (DeepSeek, 2025).
- "MoBA: Mixture of Block Attention for Long-Context LLMs" (Moonshot, 2025).
- "Ring Attention with Blockwise Transformers for Near-Infinite Context" (Liu et al., 2023).
- "RULER: What's the Real Context Size of Your Long-Context Language Models?" (NVIDIA, 2024).
- Mooncake paper (Moonshot, 2024).
- "LongLoRA: Efficient Fine-tuning of Long-Context LLMs" (Chen et al., 2023).
- Gemma-3 technical report.
- Llama-4 release notes.

---

**Next:** [Modern_KV_Compression](Modern_KV_Compression.md), [State_Space_Models_and_Hybrids](State_Space_Models_and_Hybrids.md).
**See also:** [Attention_Mechanisms](Attention_Mechanisms.md), [KV_Cache](KV_Cache.md), [Disaggregated_Serving_2025](Disaggregated_Serving_2025.md).

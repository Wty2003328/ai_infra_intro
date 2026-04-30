# Long Context Engineering

How modern models handle 128K, 1M, and 10M+ context windows. Covers position-encoding extension (YaRN, LongRoPE, NTK), chunked / sliding-window / global hybrid attention, ring attention, native sparse attention (NSA, MoBA), and the inference-time mechanics — KV pressure, prefix caching, hierarchical compression, retrieval interplay.

**Prerequisites**: [[Attention_Mechanisms]], [[KV_Cache]], [[Modern_KV_Compression]], [[Parallelism_Strategies]].

---

## 1. Why Long Context Is Hard

Two scaling problems:

1. **Quadratic compute** of vanilla attention: $O(N^2 \cdot d)$ per layer. At $N = 1\text{M}$, this is intractable per-layer.
2. **Linear KV memory**: $O(N \cdot L \cdot H_{kv} \cdot d)$ per sequence. At $N = 1\text{M}$ for Llama-3-70B, that's ~320 GB per sequence — bigger than a node's HBM.

Plus a quality problem:

3. **Position generalization**: a model trained on 8K context performs poorly at 1M without specific extension techniques.

The 2024--2026 toolbox attacks all three.

---

## 2. Position-Encoding Extensions

### 2.1 RoPE Recap

Rotary Position Embedding rotates each pair of channels in Q and K by an angle:

$$\theta_m = m \cdot \text{base}^{-2k/d}$$

where $m$ is position, $k$ is the channel pair index, and $d$ is the head dimension. Dot products $q_m \cdot k_n$ depend only on $(m - n)$, giving relative position behavior.

The **base frequency** controls the wavelength of slow-rotating channels. Default for Llama-2: $\text{base} = 10{,}000$.

### 2.2 The Extension Problem

A model trained at $\text{base} = 10{,}000$, $\text{max\_position} = 4096$ generalizes poorly to $\text{position} = 32\text{K}$ — the slow channels are operating at frequencies the model has never seen.

### 2.3 NTK-Aware Scaling (PI / NTK)

**Position Interpolation (PI):** scale positions down by $\frac{1}{k}$ so position $k \cdot N$ maps to $N$ (where $N$ is training length). Equivalent to evaluating attention at original positions but with denser inputs.

**NTK-aware:** don't scale all frequencies uniformly. Scale slow-rotating channels more, fast-rotating less, preserving high-frequency information. The key insight is that different frequency components of RoPE carry different types of positional information — high-frequency channels encode local token order (which should be preserved), while low-frequency channels encode global position (which must be compressed to fit the longer context). The NTK-aware approach modifies the base frequency:

$$\text{base}' = \text{base} \cdot \alpha^{d/(d-2)}$$

where $\alpha = \frac{N_{\text{target}}}{N_{\text{train}}}$ is the scaling factor, achieving non-uniform scaling across frequency bands automatically.

### 2.4 YaRN (Yet another RoPE extensioN)

Refines NTK with **piecewise scaling**:
- Low-frequency channels (long wavelengths): scale by $s = \frac{N_{\text{target}}}{N_{\text{train}}}$.
- High-frequency channels: don't scale (scale factor = 1).
- Mid-frequency: smooth interpolation between the two extremes via a ramp function.

Plus **attention temperature scaling**: scale the attention logits by $\frac{1}{\sqrt{s}}$ to compensate for changes in the distribution of attention scores after extension. Without this correction, extended models tend to produce overly peaked attention distributions.

YaRN with brief fine-tuning (a few billion tokens) extends a 4K-trained Llama-2 to 128K with minimal quality loss. Standard for production extension.

### 2.5 LongRoPE (Microsoft, 2024)

Search-based: use evolutionary search to find per-channel scaling factors that minimize perplexity at the target length. Goes beyond YaRN's heuristic, finds non-uniform scaling that works to 2M+ tokens. The search operates in a space of $d/2$ scaling factors (one per channel pair), using perplexity on long-context validation data as the fitness function. A two-stage approach first extends to an intermediate length, then progressively extends further, achieving better results than single-stage extension.

### 2.6 Llama-3.1+ "RoPE Theta=500K"

Llama-3.1 onward uses a much larger base ($\theta = 500{,}000$) directly in pretraining. This encodes positions natively across long ranges without requiring extension at fine-tune time. 128K context is "free" in pretraining.

The higher base stretches all wavelengths, meaning even the "fastest" rotating channels have wavelengths spanning thousands of tokens, and the slowest channels have wavelengths spanning billions of tokens — more than enough to encode 128K positions without aliasing.

### 2.7 Llama-4 iRoPE (interleaved RoPE/NoPE)

Every fourth attention layer uses **no positional encoding** (NoPE — Kazemnejad et al., 2023). The remaining layers use RoPE with extended base. NoPE layers, surprisingly, generalize better to lengths beyond training. The mix achieves 10M context.

Why NoPE works: with appropriate causal masking, the model can learn implicit position from token identity sequences. Specific layers acting "position-agnostic" generalize length-wise better than uniform RoPE. The NoPE layers rely entirely on the causal mask structure and content-based attention to infer ordering, which has no hard positional ceiling. Combined with RoPE layers that provide explicit position signals where needed, the interleaved design gets the best of both worlds.

---

## 3. Sparse / Local / Global Attention Patterns

### 3.1 Sliding-Window Attention (SWA)

Each query attends to the last $W$ tokens only. Cost $O(N \cdot W \cdot d)$ per layer instead of $O(N^2 \cdot d)$. Used by Mistral ($W = 4096$), Gemma-3 ($W = 4096$ in 5/6 layers), Phi-3-medium.

KV per token still grows linearly with $N$, but **the kernel cost per step is bounded**. KV memory bounded only if sliding-window also evicts old KV (some implementations do).

Limitation: information from beyond $W$ tokens cannot reach the query directly. Need either layer stacking (information walks $W$ steps per layer) or global layers. With $L$ stacked SWA layers, information can propagate at most $L \cdot W$ tokens — this is the model's effective receptive field.

### 3.2 Mixed Local + Global Layers

Gemma-3 pattern: **5 local layers, 1 global layer**, repeated. The global layer attends over the full context; local layers stay bounded.

Effective long-context cost per step:

$$\text{Cost} = O(L_{\text{local}} \cdot N \cdot W + L_{\text{global}} \cdot N^2)$$

At $W = 4096$, $L_{\text{global}} = L/6$, $L = 80$, $N = 128\text{K}$:
- Local: $80 \times \frac{5}{6} \times 128\text{K} \times 4\text{K} \approx 22$ GFLOPs per layer-block
- Global: $80 \times \frac{1}{6} \times (128\text{K})^2 \approx 175$ GFLOPs per layer-block

Global layers dominate at very long $N$. To push further, the global layers themselves need sparse / chunked / NSA-style attention.

### 3.3 BigBird / Longformer Patterns (legacy)

Local + global tokens + random tokens. Less common in 2025 but the pattern survives in some long-document models. The key contribution was demonstrating that sparse attention patterns can theoretically approximate full attention with $O(N)$ complexity, provided the sparsity pattern includes local, global, and random components — a result from graph connectivity theory.

### 3.4 Chunked Attention

Llama-4 long-context layers chunk the sequence into segments. Within a chunk, full attention; across chunks, anchor or summary tokens carry information. Cost $O(N \cdot C \cdot d)$ per layer, where $C$ is the chunk size. This is similar to the Memorizing Transformer pattern, where cross-chunk information flows through a small set of designated tokens that attend globally while the bulk of computation stays local.

---

## 4. Native Sparse Attention (NSA)

DeepSeek's NSA paper (early 2025) introduces **trainable sparse attention** that runs as a single kernel competitive with FlashAttention.

### 4.1 NSA Decomposition

For each query position, compute attention via three branches and combine:

1. **Compressed branch**: down-sample K,V across positions (block-pooled) to produce a coarse global view. If the sequence has $N$ tokens and blocks are of size $B$, this branch attends to $\frac{N}{B}$ compressed tokens.
2. **Selected branch**: top-$k$ most relevant blocks (by compressed-attention score) are selected for precise critical-block attention. This provides fine-grained access to the most important tokens.
3. **Sliding-window branch**: recent $W$ tokens for local detail.

$$\text{output} = w_1 \cdot \text{attn}_{\text{compressed}} + w_2 \cdot \text{attn}_{\text{selected}} + w_3 \cdot \text{attn}_{\text{window}}$$

where weights $w_1, w_2, w_3$ are learned per-head gating values.

### 4.2 Properties

- **End-to-end trainable**: not just an inference shortcut — the model learns to use sparse attention from pretraining.
- **Hardware-efficient**: NSA's kernel achieves throughput close to FlashAttention at long context by fusing all three branches into a single pass.
- **Drop-in**: replaces full attention in the architecture.
- Reported quality: matches or exceeds full attention at 64K-128K with much lower compute and memory.

### 4.3 MoBA (Mixture of Block Attention)

Concurrent work (Moonshot, 2025): each query routes to a small set of K,V **blocks** (gated like MoE). Reduces effective attention from $N$ to a few blocks per query. The gating mechanism scores each block's relevance to the current query and selects the top blocks, making the attention pattern both dynamic (changes per query) and sparse.

### 4.4 Why These Matter for Systems

NSA / MoBA make 1M+ context tractable at training time, not just inference. Combined with MLA / sliding-window, they shape the next generation of long-context architectures. The key system implication is that memory and compute now scale sub-quadratically with context length while preserving quality — enabling practical training runs at context lengths that were previously prohibitive.

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

Per layer per round, send $2 \cdot \text{chunk} \cdot d \cdot b$ bytes (K and V). At total seq $N$, $P$ ranks, $\text{chunk} = \frac{N}{P}$:

$$\text{bytes\_per\_round} = 2 \cdot \frac{N}{P} \cdot d \cdot b$$

$$\text{total per layer} = (P-1) \cdot 2 \cdot \frac{N}{P} \cdot d \cdot b \approx 2 \cdot N \cdot d \cdot b$$

Independent of $P$ at large $P$. The ring overlaps comm with compute almost perfectly — while rank $i$ computes attention against the current K,V chunk, it simultaneously sends/receives the next chunk. This overlap is possible because the attention computation per chunk is $O\!\left(\frac{N^2}{P^2} \cdot d\right)$ FLOPs while the transfer is $O\!\left(\frac{N}{P} \cdot d\right)$ bytes, and at large enough $N$ the compute dominates.

### 5.2 Production Use

- Training ultra-long-context models (1M+).
- Inference on huge prompts when KV doesn't fit per-GPU. Less common because most production prompts don't exceed HBM after MLA / GQA reductions.
- Combined with FlashAttention as **Ring FlashAttention** (Liu et al., 2023).

### 5.3 Striped / Zigzag Variants

Standard ring is load-imbalanced for **causal** attention: early-position queries attend to fewer keys than late-position queries. Striped ring partitions interleaves Q across ranks so each rank gets a balanced workload. For example, with 4 ranks and 16 positions, rank 0 gets positions {0, 4, 8, 12}, rank 1 gets {1, 5, 9, 13}, etc. — each rank has roughly equal total attention work.

---

## 6. Chunked Prefill at Long Context

For a 1M-token prompt:
- Naive prefill: one giant forward pass — won't fit; activation memory blows up.
- Chunked prefill: split into chunks of e.g. 8K-32K, prefill chunk by chunk.

Each chunk attends to **all previous KV** (already computed and cached). The kernel runs $(\text{chunk\_size}, \text{prev\_KV\_size})$ attention. With FlashAttention, this is efficient.

For 1M prompt at chunk=32K: 32 chunks. Each chunk processes 32K queries against a growing KV. Total compute: same as one-shot prefill; spread across steps to interleave with decode of other requests.

Activation memory per chunk is bounded by $O(\text{chunk\_size} \cdot d)$ rather than $O(N \cdot d)$ for the full sequence, which is the key memory saving. The KV cache itself still grows to $O(N)$, but activation tensors (the intermediate attention maps, projections, and residuals) stay small.

Implication: long-context prefill takes seconds-to-minutes wall-clock. TTFT for 1M-prompt requests is fundamentally higher; UX must reflect this (streaming "processing prompt..." indicators).

---

## 7. KV Compression at Long Context

Long context magnifies KV memory pressure — KV compression becomes critical.

Modern approaches (full treatment in [[Modern_KV_Compression]]):

- **MLA**: ~$30\times$ compression, architectural.
- **GQA / MQA**: $4$-$8\times$ shared K,V heads.
- **FP8 / INT4 KV**: $2$-$4\times$ via quantization.
- **StreamingLLM**: keep first $N$ "anchor" tokens + last $W$ tokens, drop middle.
- **H2O / SnapKV**: drop low-attention tokens.
- **Quest**: query-aware KV selection at each step.
- **KIVI**: 2-bit KV with per-channel/per-token scales.

Stack them: MLA + FP8 KV + sliding-window architecture results in Llama-4-Scout-style 10M context fitting on a single node. The multiplicative nature of these compressions is the key insight:

$$\text{Effective compression} = R_{\text{arch}} \times R_{\text{quant}} \times R_{\text{eviction}}$$

For MLA ($30\times$) + FP8 ($2\times$) + SnapKV ($4\times$) = $240\times$ effective compression over naive FP16 MHA.

---

## 8. Prefix Caching at Long Context

Long shared prefixes (system prompts, RAG documents, persistent agent context) are exactly where prefix caching shines:

- A 50K-token system prompt cached once serves thousands of requests.
- A 200K-token document context (e.g., a long paper) reused across queries skips its prefill entirely after the first request.

vLLM's hash-based and SGLang's RadixAttention both work. The radix tree handles partial-block matches better at long context — it can match the longest common prefix between any two prompts at the granularity of individual KV blocks rather than requiring exact full-prefix matches.

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

Tradeoff: tier transitions cost bandwidth (swap-in latency). Schedulers route requests to replicas with cached prefixes (locality-aware) to keep the hot tier hit. The bandwidth gap between tiers is substantial: HBM at 3-8 TB/s, host RAM at ~200 GB/s via NVLink-C2C (or ~64 GB/s via PCIe), NVMe at ~7 GB/s. Each tier transition adds latency proportional to KV size divided by the tier's bandwidth.

---

## 10. Sequence / Context Parallelism (CP / SP)

For training: see [[Parallelism_Strategies]] section 6. Ring attention is CP applied to attention.

For inference: rare. Most long-context inference scenarios are amenable to per-GPU full-sequence attention (after MLA, sliding-window, etc.). When KV truly doesn't fit on one GPU, use TP-aware KV sharding (KV by heads — natural with TP) or split KV by sequence chunks.

---

## 11. Long-Context Inference Engineering Checklist

For serving long-context workloads:

- **Use a long-context model**: sliding-window + global / MLA / NSA. Don't try to extend a 4K model to 1M.
- **Enable FP8 KV**: $2\times$ capacity for free on Hopper.
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

- **Treating 1M context as $N = 1\text{M}$ attention everywhere**: it's typically MLA or sliding-window. Read the model card.
- **Forgetting chunked prefill**: 1M prompt OOMs without it.
- **Naive YaRN to extreme lengths**: works to ~$10\times$ training length, then quality cliffs. Use LongRoPE for further.
- **Prefix-cache invalidation across model upgrades**: cache lives in token-id hash space; tokenizer changes invalidate everything.
- **NIAH-pass-fail thinking**: passing NIAH at 1M doesn't mean general 1M reasoning. Use multi-task benchmarks.
- **TTFT SLOs designed for chat workloads**: do not apply to long-context. Set separate SLOs for long-prompt requests.

---

## 14. Common Interview Questions

**Q: How do you extend a 4K-trained model to 128K context?**
A: Apply YaRN or LongRoPE position-encoding extension, then briefly fine-tune (a few billion tokens at the new context length). YaRN applies piecewise frequency scaling — low-frequency channels are scaled by $s = N_{\text{target}} / N_{\text{train}}$, high-frequency channels are left unchanged, and mid-frequency channels are smoothly interpolated. An attention temperature correction of $\frac{1}{\sqrt{s}}$ compensates for distribution shifts. Quality loss is small with proper extension. Recent models (Llama-3.1+) train natively at 128K with elevated RoPE base ($\theta = 500\text{K}$), avoiding extension entirely.

**Q: What's iRoPE in Llama-4?**
A: Interleaved RoPE / NoPE: every fourth attention layer has no positional encoding at all, the rest use RoPE. NoPE layers generalize better to lengths beyond training because they rely on causal-mask structure and content-based attention for ordering, which has no hard positional ceiling. The mix supports 10M context while maintaining quality at trained lengths.

**Q: Compare sliding-window attention and full attention at long N.**
A: Full: $O(N^2 \cdot d)$ per layer, all-tokens-to-all. Sliding: $O(N \cdot W \cdot d)$ per layer, capped attention range $W$. Sliding is much cheaper but loses global mixing — must be combined with global layers (Gemma-3 5:1 ratio) or work-around in subsequent layers. With $L$ stacked SWA layers, information propagates at most $L \cdot W$ tokens.

**Q: How does NSA work and why is it efficient?**
A: Native Sparse Attention (DeepSeek). Three trainable branches per query: compressed (block-pooled $K, V$ for coarse global view), selected (top-$k$ blocks by compressed score for precise attention), sliding window (recent $W$ tokens). Combined via learned per-head gating weights: $\text{out} = w_1 \cdot \text{attn}_{\text{comp}} + w_2 \cdot \text{attn}_{\text{sel}} + w_3 \cdot \text{attn}_{\text{win}}$. End-to-end trainable, drop-in for full attention, hardware-efficient (single fused kernel). Quality matches full attention at long context with much less compute.

**Q: Walk me through ring attention.**
A: Each rank holds a chunk of Q, K, V. Each round, compute attention of local Q against current K,V chunk; rotate K,V to next rank. After $P$ rounds, every Q has seen all K,V. Use online softmax to combine partial stats. Total bandwidth per layer is $\approx 2 \cdot N \cdot d \cdot b$ bytes, independent of $P$ at large $P$; comm overlaps with compute. Striped/zigzag variants fix load imbalance from causal masking.

**Q: How would you deploy a 1M-context model on 8xH100?**
A: Use MLA architecture (e.g., DeepSeek-V3) so KV/token is tiny (~70 KB). FP8 KV cache halves that to ~35 KB. Chunked prefill (32K chunks). Aggressive prefix caching for shared documents. Disaggregate prefill from decode. Set long-context-specific SLOs. With these, 1M prompts at modest concurrency are feasible per node — at FP8, 1M tokens = ~35 GB KV per sequence, fitting ~10 concurrent sequences.

**Q: Why is chunked prefill required at long context?**
A: Naive single-shot prefill of 1M tokens has activation memory $O(N \cdot d)$ per layer which blows up HBM. Chunking into 8K-32K pieces (a) bounds activation memory to $O(\text{chunk\_size} \cdot d)$, (b) lets decodes interleave between chunks, (c) makes scheduling smoother. Same total FLOPs as one-shot — only the peak memory and scheduling granularity change.

**Q: What's the difference between YaRN and LongRoPE?**
A: YaRN uses heuristic piecewise frequency scaling (low/mid/high bands) plus attention temperature scaling by $\frac{1}{\sqrt{s}}$. LongRoPE uses evolutionary search to find per-channel scaling factors that minimize perplexity at target length. LongRoPE pushes further (2M+) at the cost of search compute and is more architecture-specific.

**Q: How does prefix caching help long-context inference?**
A: Long shared prefixes (system prompts, document context, agent state) are exactly the scenarios where prefix caching skips the expensive prefill. Chat / RAG / agent workloads commonly see 60-95% prefix-cache hit rates, making the marginal cost of long prompts much lower than per-token math suggests. SGLang's RadixAttention matches at block granularity, enabling partial-prefix reuse.

**Q: What's Mooncake?**
A: Moonshot's cluster-scale KV pool architecture: KV cache lives in a global tier across all replicas (CPU RAM), with locality-aware routing pulling cached prefixes to active replicas. KV blocks are hashed by `(parent_hash, token_ids)` and registered in a distributed hash table. Achieves 80%+ prefix-cache hit rate at scale where per-replica caches would only see 30-50%. NIXL-style fast transport moves KV at network speed.

**Q: How do you benchmark long-context quality?**
A: Use a multi-task suite. NIAH for literal recall, RULER for diverse tasks (multi-hop, aggregation, reasoning across 13 task types), LongBench for realistic doc QA, BABILong for long-range reasoning. Don't rely on NIAH alone — passing NIAH at 1M doesn't imply general long-context capability; the model may still fail at multi-hop reasoning or aggregation at that length.

**Q: Why does model quality degrade beyond the trained length even with YaRN?**
A: The model has no training signal at those positions; YaRN extrapolates the position encoding but learned attention patterns weren't exposed to those distance scales. The model's internal representations for "how to attend at distance $d$" were only trained for $d \leq N_{\text{train}}$. Fine-tuning at the target length closes most of the gap by providing training signal at the new distances. Limit: training compute scales with target length.

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

**Next:** [[Modern_KV_Compression]], [[State_Space_Models_and_Hybrids]].
**See also:** [[Attention_Mechanisms]], [[KV_Cache]], [[Disaggregated_Serving_2025]].

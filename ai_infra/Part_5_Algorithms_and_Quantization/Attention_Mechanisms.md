# Attention Mechanisms

Attention is the heart of the transformer. This page is the complete attention story: the original dot-product formulation, the numerical derivation of "online softmax" that unlocked FlashAttention, and the family of KV-reduction variants (MHA → MQA → GQA → MLA → sliding-window → sparse). The FlashAttention algorithm itself — the tiling, the SRAM dance, the v1/v2/v3 progression — has its own page: [FlashAttention_Deep_Dive](../Part_4_GPU_Kernel_Engineering/FlashAttention_Deep_Dive.md).

---

## 1. Dot-Product Attention from First Principles

### 1.1 The motivating problem

Given a sequence of tokens, each token should be able to pull information from relevant other tokens — a "content-addressable" memory lookup. The query says "I'm looking for X"; each key says "I provide Y"; the value is the payload. Dot-product similarity between Q and K picks the right keys; softmax turns similarities into a weight distribution; weighted sum of values gives the answer.

```
Q (query) [S, d]         "what am I looking for"
K (key)   [S, d]         "what am I (as a key)"
V (value) [S, d_v]       "what information I carry"

attention(Q, K, V) = softmax(Q K^T / sqrt(d)) V
```

### 1.2 Why divide by sqrt(d)

Q and K entries are typically ~N(0, 1). Their dot product (sum of d products) has variance d → std ~sqrt(d). Without scaling, logits grow unbounded in d, softmax saturates (gradients vanish). Dividing by sqrt(d) normalizes variance.

### 1.3 Multi-head

Single attention head captures one type of relation. Multiple heads in parallel, each with their own W_Q/K/V projections, capture multiple relations.

```
For h in 1..H:
    Q_h = X W_Q^h   [S, d_h]
    K_h = X W_K^h   [S, d_h]
    V_h = X W_V^h   [S, d_h]
    head_h = softmax(Q_h K_h^T / sqrt(d_h)) V_h
Out = concat(head_1..head_H) W_O
```

Typically d_h = D/H, so total Q/K/V compute is the same as a single head of dim D but with distinct subspaces.

### 1.4 The causal mask

In a decoder, position i should only attend to positions ≤ i.
```
mask[i, j] = 0 if j <= i else -inf
```
Added to scores before softmax. Adding -inf to a logit makes exp(−inf) = 0 in softmax → weight = 0. Compute is still O(S²) but information flow is causal.

During decoder inference with KV cache, no explicit mask is needed — the query at position T attends only to keys at positions 0..T, all valid.

---

## 2. The Standard (Naive) Attention Algorithm

```python
def attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / sqrt(d_k)   # [B, H, S, S]
    if mask is not None:
        scores += mask
    probs = softmax(scores, dim=-1)                # [B, H, S, S]
    out = probs @ V                                 # [B, H, S, d_v]
    return out
```

Characteristics:
- Materializes the full S×S attention matrix in HBM
- For S=128K: 128K × 128K × 2 bytes = 32 GB per head per layer — impossible
- Even for S=4K: 64 MB per head per layer — lots of HBM round-trips

This is why naive attention is practically unusable for long contexts. Enter FlashAttention.

---

## 3. Online Softmax: The Algebraic Key to FlashAttention

The softmax operation looks like it *requires* seeing the full row to compute correctly. But it doesn't — there's an online algorithm that produces the same result by processing chunks incrementally, with constant auxiliary state.

### 3.1 Standard softmax

```
For a row v ∈ R^N:
    m = max(v)               # numerical stability
    shifted = v - m
    numerator = exp(shifted)
    denom = sum(numerator)
    softmax(v) = numerator / denom
```

### 3.2 Online softmax (a.k.a. streaming softmax)

Process v in chunks v₁, v₂, .... After processing the first k chunks:
```
m_k = max of values seen so far
l_k = Σᵢ exp(vᵢ - m_k)  (sum of exponentials of seen values, shifted by m_k)
```

Transitioning from state (m_k, l_k) to (m_{k+1}, l_{k+1}) after seeing chunk v_{k+1}:
```
m_local = max(v_{k+1})                       # local max of new chunk
m_{k+1} = max(m_k, m_local)                  # new global max

# Rescale previous accumulators to the new max
l_rescaled_old = l_k · exp(m_k - m_{k+1})
l_local = Σ exp(v_{k+1} - m_{k+1})
l_{k+1} = l_rescaled_old + l_local
```

After all chunks seen, final softmax = exp(v - m_N) / l_N.

### 3.3 Why it works (proof sketch)

Initially: `softmax = exp(vᵢ - m) / Σⱼ exp(vⱼ - m)`. This ratio is invariant under uniform shift of the `m` applied to both numerator and denominator — that's the whole point of the shifting (numerical stability).

If we haven't seen all values yet, we compute `exp(vⱼ - m_k)` where m_k is the max so far. When a new max appears, we need to rescale: `exp(vⱼ - m_{k+1}) = exp(vⱼ - m_k) · exp(m_k - m_{k+1}) = old_value · exp(m_k - m_{k+1})`. Since m_{k+1} ≥ m_k, the rescale factor ≤ 1, numerically stable.

### 3.4 Online softmax × V

The critical extension: we want `softmax(QK^T) · V`, not just softmax. Including V in the online computation:

```
# State after k chunks:
# m_k = max of logits seen
# l_k = sum of exp(logit - m_k)  (scalar per query row)
# o_k = Σ exp(logit - m_k) · v   (partial attention output)

# Process chunk k+1: compute new logits s_{k+1} = Q K_{k+1}^T / sqrt(d), get V_{k+1}
m_new = max(m_k, max(s_{k+1}))
alpha = exp(m_k - m_new)             # rescale factor for old state
l_new = alpha · l_k + Σ exp(s_{k+1} - m_new)
o_new = alpha · o_k + (exp(s_{k+1} - m_new)) · V_{k+1}
```

At the end: `output = o_N / l_N`.

This is the algorithmic skeleton of FlashAttention. It processes Q row × K columns chunk by chunk, never materializing the full row in HBM — only running state (m, l, o) in SRAM.

---

## 4. FlashAttention Overview

Given the online softmax, the rest of FA is a matter of scheduling:

- Tile Q rows into blocks of B_r
- Tile K/V columns into blocks of B_c
- For each Q-row tile, iterate over all K/V-column tiles, updating the (m, l, o) running state in SRAM
- Write final (m, l, o) → o/l → output to HBM once per Q row

Memory traffic becomes O(N·d) (read Q and K/V tiles once) instead of O(N²·d). FLOPs are the same (you still compute all QK and PV products), but compute is efficient because tiles fit in SRAM.

Full algorithm details (v1 forward, v2 forward with parallelism over seq, v3 with warp specialization) in [FlashAttention_Deep_Dive](../Part_4_GPU_Kernel_Engineering/FlashAttention_Deep_Dive.md).

---

## 5. The KV-Head Family: MHA, MQA, GQA, MLA

The dominant memory cost during decode is the KV cache. Reducing KV heads directly reduces this cost.

### 5.1 Multi-Head Attention (MHA) — the original

- H query heads, H key heads, H value heads
- Each head is independent
- KV cache per token: 2·H·dₕ values

Used in GPT-2, GPT-3, Llama-2 7B.

### 5.2 Multi-Query Attention (MQA)

- H query heads share a single K head and single V head
- KV cache per token: 2·dₕ values (H× smaller than MHA)
- Mechanism: `Q: [H, dₕ]`, `K: [1, dₕ]`, `V: [1, dₕ]`; broadcast K/V over H query heads

Introduced by Shazeer (2019). Used in PaLM, Falcon, some GPT variants.

**Tradeoff:** KV memory H× smaller, but quality often degrades — the single K/V may not have enough capacity to represent all the context that H query heads want to attend to.

### 5.3 Grouped-Query Attention (GQA)

- H query heads organized into G groups (G divides H)
- Each group shares one K head and one V head
- KV cache per token: 2·G·dₕ values (H/G× smaller than MHA)

Llama-3 uses G=8 for H=64 → 8× KV reduction. Close to MQA's memory savings, close to MHA's quality.

Implementation: repeat K/V tensors across groups before attention:
```
K: [B, G, S, dₕ]  →  repeat  →  [B, H, S, dₕ]
```

Or (more efficient) use gathered attention that knows about groups:
```
for g in 0..G-1:
    for h in g·(H/G) .. (g+1)·(H/G)-1:
        out[h] = attention(Q[h], K[g], V[g])
```

### 5.4 Multi-Head Latent Attention (MLA) — DeepSeek

MLA (from DeepSeek-V2, extended in V3) is a dramatic KV compression technique.

**Idea:** instead of storing K and V directly, store a low-rank compressed latent `c_kv` and reconstruct K/V on the fly.

```
c_kv = X W_DKV          # [S, r]  r << d, e.g., r=512 for D=7168
K = c_kv W_UK + ...     # expand to [S, H, dₕ] on demand
V = c_kv W_UV + ...     # same
```

KV cache per token: just r (e.g., 512 floats per layer), regardless of H or dₕ.

For DeepSeek-V3 (128 heads, d_h = 128, 61 layers):
- MHA would be: 2·128·128·61 = 2 MB per token in FP16 → 2 GB at 1K context
- MLA uses: ~1 KB per token → 1 MB at 1K context, 1000× reduction

This is what makes DeepSeek-V3's 128K context viable on modest hardware.

**Caveats:** more compute at attention time (reconstruct K/V). Quality-preserving requires careful training. As of 2025-2026, MLA is becoming widely adopted in new LLMs.

---

## 6. Sliding-Window Attention

For very long contexts, full attention is prohibitive. Sliding-window restricts attention to a fixed-size window W behind each position:

```
mask[i, j] = 0 if i - W < j <= i else -inf
```

- Each query attends to at most W keys
- FLOPs: O(S·W) instead of O(S²)
- Memory: KV cache for the last W tokens per layer

Mistral uses sliding-window with W=4096 (along with full attention on certain layers).

**Limitation:** information from beyond the window can't be directly attended. Compensated by stacking layers — each layer adds another W window of effective receptive field, so L layers give effective context L·W.

---

## 7. Sparse Attention

Various patterns enable attention over long contexts with sub-quadratic cost.

### 7.1 Local + global

Local (window) attention for most positions + a handful of "global" tokens (CLS, first few tokens) that everyone attends to and that attend to everyone.

Used in Longformer, BigBird.

### 7.2 Dilated / strided

Attend to every k-th token. Multiple heads with different k give a logarithmic-depth connectivity.

### 7.3 Learned sparsity

Routed attention (e.g., Reformer's LSH attention, Routing Transformer) groups similar queries and keys via hashing or clustering, attending only within groups.

### 7.4 Linear attention

`softmax(QK^T) V` can be approximated as `φ(Q) · (φ(K)^T V)` where φ is a feature map. Compute is O(S·d) instead of O(S²).

**Family includes:**
- Performer (random features)
- Linformer (low-rank K/V)
- Mamba / State Space Models (not strictly attention, but replace attention entirely)

Quality has historically trailed full attention, but Mamba-2 and Jamba (hybrid attention + Mamba) are closing the gap for long-context tasks.

### 7.5 When sparse attention is used in production

- Very long contexts (1M+ tokens): sliding-window + global tokens
- Long-context retrieval: block-sparse patterns (e.g., Minference, DuoAttention)
- Streaming: StreamingLLM — keep first K "attention sinks" + a sliding window

As of 2026, full attention (with FlashAttention) is still default for contexts up to 128K-256K. Above that, sparse and hybrid architectures dominate.

---

## 8. Cross-Attention

In encoder-decoder models (T5, BART) and multi-modal (vision-language), cross-attention connects two sequences:

```
Q from sequence A    [S_a, d]
K, V from sequence B [S_b, d]
attn(Q, K, V)  →     [S_a, d]    # sequence A attends to sequence B
```

Decoder-only LLMs typically don't use cross-attention (everything is self-attention on the concatenated context). But for image-to-text models (Llava, Flamingo) or encoder-decoder seq2seq, it's essential.

---

## 9. Attention Variations in Production Models

| Model | Attention | Context | Notes |
|---|---|---|---|
| Llama-3 70B | GQA (8 KV heads, 64 Q heads) | 128K | RoPE with scaling |
| Llama-3 405B | GQA (8 KV heads, 128 Q heads) | 128K | |
| Mistral 7B | GQA + sliding window (4K) | 32K | |
| Mixtral 8x7B | GQA | 32K | MoE |
| DeepSeek-V3 | MLA | 128K | 671B/37B active, MLA + MoE |
| Gemini 2.5 | Undisclosed, likely hybrid | 2M+ | |
| Claude 3.5 | Undisclosed | 200K | |
| GPT-4 / GPT-4o | Undisclosed | 128K | |

Public models trend toward GQA + RoPE + sliding-window variants. Frontier labs are experimenting with hybrid Mamba/attention and MLA.

---

## 10. Attention in Training vs Inference

### 10.1 Training

- All S positions computed in parallel (prefill-like)
- Attention is compute-bound (large S² FLOPs)
- FlashAttention used throughout for memory
- Backward pass recomputes attention weights to avoid saving S² tensor
- Dropout applied to attention weights (not typical at inference)

### 10.2 Prefill (inference)

- Same as training forward
- FlashAttention, compute-bound
- Produces the KV cache for subsequent decode

### 10.3 Decode (inference)

- Query is one token; attention is over the cached K, V
- Memory-bound (reading KV cache)
- Modified FlashAttention (FlashDecoding, FlashAttention-Decode) that parallelizes over KV-sequence rather than Q-sequence
- KV cache layout (PagedAttention, radix caching) matters enormously

---

## 11. Common Interview Questions

**Q: Write the attention formula.**
A: `attention(Q, K, V) = softmax(Q K^T / sqrt(d)) V`. With causal mask for decoders. Multi-head splits into H parallel attention heads, concatenates outputs, and projects with W_O.

**Q: Why scale by sqrt(d)?**
A: To keep the variance of dot products bounded. Entries of Q, K ~ N(0, 1), so `Q·K` has variance d. Without scaling, logits grow with d, pushing softmax into saturation (gradients vanish or dominate). Dividing by sqrt(d) normalizes variance to 1.

**Q: What's the memory cost of naive attention vs FlashAttention?**
A: Naive: O(S²) HBM for the attention matrix. FlashAttention: O(N·d) — the matrix lives in SRAM; only Q, K, V, and output round-trip to HBM. Same FLOPs, vastly less bandwidth.

**Q: Derive online softmax.**
A: Maintain running state (m, l). On seeing new value x: new_m = max(m, x). Rescale old state: l := l · exp(m - new_m). Accumulate: l += exp(x - new_m). At end, softmax(x_i) = exp(x_i - m) / l. Proof: the shifted-log-sum-exp identity is invariant under uniform shift, and the rescale corrects for maxima updates.

**Q: What's the difference between MHA, MQA, GQA, and MLA?**
A: MHA: H independent Q, K, V heads. MQA: H Q heads share 1 K and 1 V. GQA: H Q heads in G groups, each group shares K/V (H/G × reduction vs MHA). MLA: compresses KV into a small latent vector (r dims) and reconstructs K/V on attention. Memory per token: MHA > GQA > MQA > MLA.

**Q: How does GQA reduce KV cache?**
A: Instead of H KV heads, GQA has G KV heads where G < H. Multiple query heads share each KV head (broadcast). KV memory is proportional to G not H. Llama-3's GQA with G=8 and H=64 gives 8× smaller KV cache with minimal quality loss.

**Q: What is MLA and why is it interesting?**
A: Multi-Head Latent Attention stores a compressed latent `c_kv` per token (low rank, e.g., 512 floats) and reconstructs K, V on demand via learned projections. KV cache shrinks to a single vector per token per layer, independent of H and dₕ. DeepSeek-V3 uses it to serve 128K context with far less HBM than GQA.

**Q: When would you use sliding-window attention?**
A: When context lengths greatly exceed practical full-attention budgets. Window W bounds attention cost to O(S·W) per layer. Stacked layers give effective receptive field L·W. Good for streaming, chat with long history where the very oldest info has diminishing importance. Used in Mistral.

**Q: Is attention always memory-bound?**
A: No. Prefill attention (training forward) is compute-bound: Q·K^T and P·V are large matmul with AI ~ S. Decode attention (single Q per step) is memory-bound: AI ~ 1. FlashAttention is compute-bound on H100 because tiling moves the bandwidth roof to SRAM (~12 TB/s per SM), far higher than HBM.

**Q: How does FlashAttention change algorithmically vs standard?**
A: It swaps the inner and outer loops: instead of computing all scores then softmax then PV, it processes one Q-tile × all-K-tiles at a time, maintaining online-softmax state in SRAM. Numerically identical to standard; just reorders operations so the O(S²) intermediate never hits HBM.

---

## 12. Further Reading

- "Attention Is All You Need" (Vaswani et al., 2017)
- "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (Dao et al., NeurIPS 2022)
- "FlashAttention-2" (Dao, 2023)
- "FlashAttention-3" (Shah et al., 2024)
- "GQA" (Ainslie et al., 2023)
- "DeepSeek-V2" (DeepSeek, 2024)  — MLA paper
- Tri Dao's blog posts on online softmax
- "Online normalizer calculation for softmax" (Milakov and Gimelshein, 2018)

# Transformer Internals

This page is the math reference for a modern LLM's forward pass. Every operation, its FLOP and byte count, why it exists, and where it lives on the roofline. If you can reconstruct the forward pass on a whiteboard with correct dimensions, you're 80% done with a system-design interview.

---

## 1. The Top-Level Forward Pass

A decoder-only transformer (GPT/Llama style) processes input token IDs to output logits:

```
token_ids [B, S]
    │ embedding lookup
    ▼
hidden [B, S, D]
    │
    ▼
for each of L transformer layers:
    hidden = layer_norm(hidden)
    q, k, v = project(hidden)
    attn_out = self_attention(q, k, v)
    hidden = hidden + attn_out
    hidden = layer_norm(hidden)
    mlp_out = mlp(hidden)
    hidden = hidden + mlp_out
    ▼
hidden [B, S, D]
    │ final LayerNorm
    ▼
logits = hidden @ W_emb.T   [B, S, V]
```

Typical dimensions for Llama-3 family:

| Model | L | D | H (heads) | K (kv heads) | dₕ | FFN mult | V | Params |
|---|---|---|---|---|---|---|---|---|
| Llama-3-8B | 32 | 4096 | 32 | 8 | 128 | 3.5× | 128K | 8B |
| Llama-3-70B | 80 | 8192 | 64 | 8 | 128 | 3.5× | 128K | 70B |
| Llama-3-405B | 126 | 16384 | 128 | 8 | 128 | 3.5× | 128K | 405B |

FFN mult: the ratio `d_ff / d_model`. For Llama uses SwiGLU which has `d_ff = 8/3 · d_model` rounded up to a multiple.

---

## 2. Embedding

### 2.1 Token embedding

```
E [V, D]  (vocabulary × hidden dim)
hidden[b, s, :] = E[token_ids[b, s], :]
```

- FLOPs: ~0 (just indexing)
- Memory: B·S·D·dtype bytes read from E; rest of E stays hot in L2
- For Llama-3-70B: V=128K, D=8192, E is 1 GB

### 2.2 Positional encoding

Modern LLMs use **rotary position embeddings (RoPE)** applied to Q and K inside each layer, not absolute position embeddings on the input. RoPE is covered in section 4.

Earlier LLMs (GPT-2, GPT-3) used learned absolute positional embeddings added to the token embedding:
```
hidden = token_emb[token_ids] + pos_emb[position_ids]
```

Some models use ALiBi (linear attention bias, added to attention scores) instead.

---

## 3. Normalization

### 3.1 LayerNorm

Classical:
```
mean = hidden.mean(-1, keepdim=True)
var  = ((hidden - mean)**2).mean(-1, keepdim=True)
out  = (hidden - mean) / sqrt(var + eps) * gain + bias
```

Per-element: mean/var over last dim (hidden_dim=D), then scale and shift.

- FLOPs: O(B·S·D)
- Memory-bound (low arithmetic intensity)

### 3.2 RMSNorm (used in Llama, most post-2022 LLMs)

```
rms = sqrt((hidden**2).mean(-1, keepdim=True) + eps)
out = hidden / rms * gain
```

No mean subtraction, no bias. Cheaper (saves a reduction + a subtraction).

Why it works: empirically, the "mean shift" step in LayerNorm isn't important; what matters is the variance normalization. Simpler and ~20% faster.

### 3.3 Pre-norm vs post-norm

Modern (pre-norm):
```
y = x + attention(norm(x))
y = y + mlp(norm(y))
```

Normalization happens **before** the residual-adding submodule. Enables stable training at 100+ layers.

Classical (post-norm, from original transformer paper):
```
y = norm(x + attention(x))
y = norm(y + mlp(y))
```

Less stable at depth. Modern models universally use pre-norm.

---

## 4. Rotary Position Embedding (RoPE)

RoPE encodes absolute position information by rotating query and key vectors in 2D subspaces, with rotation angle proportional to position. Dot-products between rotated vectors then encode **relative** position.

### 4.1 The construction

For each 2D slice (dims 2i, 2i+1) of a d-dimensional Q or K vector at position m:

```
θ_i = 10000^(-2i/d)
[q_m,2i ']    [cos(m·θ_i)  -sin(m·θ_i)] [q_m,2i]
[q_m,2i+1']  =[sin(m·θ_i)   cos(m·θ_i)] [q_m,2i+1]
```

- Low-index dims (i small): high frequency, rotate fast with position
- High-index dims (i large): low frequency, rotate slowly

### 4.2 Why it works

`(Rot_m · q)^T (Rot_n · k) = q^T Rot_{n-m} k` — the dot product only depends on the position difference (n-m), not absolute positions. This is exactly the relative positional bias you want.

### 4.3 Implementation

Instead of building a full rotation matrix:
```
def rope(x, cos, sin):
    # x: [B, S, H, D]; cos, sin: [S, D/2]
    x_even = x[..., 0::2]
    x_odd  = x[..., 1::2]
    return torch.stack([x_even * cos - x_odd * sin,
                        x_even * sin + x_odd * cos], dim=-1).flatten(-2)
```

Fused implementations (Triton, CUDA) avoid the reshape/stack overhead.

### 4.4 NTK-aware and YARN scaling

For context extension beyond the trained length, simply extrapolating RoPE fails — high-frequency dims don't generalize. Techniques:
- **Positional interpolation (PI):** scale down position indices by a factor
- **NTK-aware:** modify base frequency differently for low vs high dims
- **YARN:** rescale theta with attention scaling factor

Llama 3's context extension to 128K used these techniques on top of training with ~8K contexts.

---

## 5. Self-Attention

### 5.1 Projections (Q, K, V)

For input hidden `X` of shape [B, S, D]:

```
Q = X @ W_Q  [B, S, H·dₕ]
K = X @ W_K  [B, S, K·dₕ]
V = X @ W_V  [B, S, K·dₕ]
```

Where H is query heads, K is KV heads. In MHA H=K; in GQA H > K; in MQA K=1.

Reshape:
```
Q: [B, S, H, dₕ] → [B, H, S, dₕ]
K: [B, S, K, dₕ] → [B, K, S, dₕ]
V: [B, S, K, dₕ] → [B, K, S, dₕ]
```

In GQA, each group of H/K query heads shares the same K/V pair — reduce K/V memory by H/K×.

### 5.2 Attention math

```
scores = Q @ K^T / sqrt(dₕ)     [B, H, S, S]
scores += causal_mask            # -inf on (i<j)
probs  = softmax(scores, -1)
out    = probs @ V               [B, H, S, dₕ]
```

Reshape and output project:
```
out: [B, H, S, dₕ] → [B, S, H·dₕ]
out = out @ W_O                  [B, S, D]
```

### 5.3 Causal masking

In a decoder, position i only attends to positions ≤ i:
```
mask[i, j] = 0 if j <= i else -inf
scores = scores + mask
```

At inference with KV cache, you don't need a mask for decode — the query is at position T and the cache has positions 0..T, all valid.

### 5.4 FLOPs and bytes

For prefill:
- `Q @ K^T`: 2·B·H·S²·dₕ FLOPs; reads 2·B·H·S·dₕ (Q+K), writes B·H·S² (scores)
- `scores @ V`: 2·B·H·S²·dₕ FLOPs; reads B·H·S² + B·H·S·dₕ, writes B·H·S·dₕ
- **Total attention:** 4·B·H·S²·dₕ FLOPs, O(S²) memory bytes (quadratic)

For decode step (query is 1 token, KV cache has length T):
- `Q @ K^T`: 2·B·H·T·dₕ FLOPs; reads B·H·(dₕ + T·dₕ) bytes
- `scores @ V`: 2·B·H·T·dₕ FLOPs; reads B·H·(T + T·dₕ) bytes
- Total: 4·B·H·T·dₕ FLOPs, O(T) bytes
- Arithmetic intensity: ~1 FLOP/byte — deeply memory-bound

This is why FlashAttention (which keeps the O(S²) matrix in SRAM, not HBM) is critical.

---

## 6. FFN / MLP Block

### 6.1 Classical (GPT-2 style)

```
h1 = X @ W_up + b_up         [B, S, 4D]
h2 = GeLU(h1)
out = h2 @ W_down + b_down   [B, S, D]
```

Two linear layers with GeLU activation. The "4× hidden" expansion is standard.

### 6.2 SwiGLU (Llama, PaLM, most modern)

```
gate = X @ W_gate            [B, S, 8/3 D]
up   = X @ W_up              [B, S, 8/3 D]
hidden = SiLU(gate) * up      # elementwise gated
out  = hidden @ W_down       [B, S, D]
```

Three matrices (gate, up, down). The gate-up is an **gated linear unit**:
- SiLU(x) = x · sigmoid(x)
- Multiplicative gate lets the model "modulate" the up-projection per-feature

Empirically better than plain GeLU MLP at the same parameter budget.

### 6.3 FLOPs and bytes

For SwiGLU with d_ff = 8/3·D:
- gate, up, down each: 2·B·S·D·d_ff FLOPs
- Total: 6·B·S·D·d_ff = 16·B·S·D² FLOPs

For Llama-3-70B, D=8192: per token prefill = 16·8192² = 1.07 GFLOPs/token just for FFN.

Per layer total = 4·attention + 16·FFN ≈ 20·B·S·D² FLOPs (assuming small S).

For all 80 layers: 1600·B·S·D² = 1600·8192²·B·S ≈ 107·B·S GFLOPs. At B·S=4096, that's 440 TFLOPs per prefill. On 8×H100, ~0.5 s — consistent with real benchmarks.

---

## 7. Output Head

```
logits = final_hidden @ W_emb.T   [B, S, V]
```

Often W_emb is **weight-tied** with the input embedding (saves V·D params). The output projection is a large matmul when V is large (128K for Llama-3) — one of the most expensive single matmul in the model.

FLOPs: 2·B·S·D·V = 2·4096·8192·128000 ≈ 8.6 TFLOPs for prefill.

In training, a follow-up softmax + cross-entropy is fused for memory (Softmax Cross-Entropy would materialize [B,S,V] = huge). Tools like flash-cross-entropy fuse this.

---

## 8. Full FLOP and Memory Accounting (Llama-3-70B)

### 8.1 Per-layer parameter count

- QKV proj: 3·D² = 3·8192² = 201 M
  - In GQA with 8/64 KV ratio: only 1/8 for K,V, so ~1.125·D² = 75 M... (approximate)
- Output proj: D² = 67 M
- FFN: 3 · D · (8/3·D) = 8·D² = 537 M  
- LayerNorm gains: 2·D = 16 K (negligible)
- **Per-layer total:** ~679 M params
- With 80 layers: 54 B params
- Plus embedding: V·D = 128K·8192 = 1 B
- Total: ~55-56 B (short of 70B by about 14 B)

(Real Llama-3-70B has additional subtleties — this is a rough accounting.)

### 8.2 Per-token forward FLOPs (prefill)

Rough rule: **~2·P FLOPs per token per forward pass**, where P is parameter count (in dense ops only, ignoring softmax etc.).

For 70B model at batch=1, one token: 2·70B = 140 GFLOPs.

At S=4096: 4096·140 GFLOPs = 574 TFLOPs for the prefill.

On 8×H100 (990 TFLOPS each = 7920 peak): theoretical 0.07 s if 100% compute-bound; practically 0.3-0.5 s at ~30-50% MFU.

### 8.3 Per-decode-step memory reads

**Weights:** 140 GB (FP16) or 70 GB (FP8), once per batch step.
**KV cache per request:** 2·L·K·dₕ·ctx·dtype bytes. For Llama-3-70B at ctx=4096 FP16: 1.25 GB.
**Activations:** transient; small.

At batch=128 concurrent decodes:
- Weights: 140 GB (amortized)
- KV: 128 × 1.25 GB = 160 GB
- Total step memory: 300 GB
- On 8×H100: 37.5 GB per GPU per step
- At 3.35 TB/s: 11.2 ms

128 tokens generated per 11.2 ms = 11400 tokens/sec aggregate. Typical real numbers.

---

## 9. Mixture of Experts (MoE)

Modern very-large models (DeepSeek-V3, Mixtral, Grok) are sparse MoEs: only a fraction of parameters activate per token.

### 9.1 Architecture

Replace each FFN with:
```
gate_logits = X @ W_router                  [B, S, E]
top_k_idx, top_k_weights = top_k(gate_logits, k)
out = Σ_k weights[k] * FFN_{top_k_idx[k]}(X)
```

Each token is routed to `k` experts out of `E`. Typical: E=256, k=8.

### 9.2 Parameter vs active compute

DeepSeek-V3: 671B total parameters, 37B active per token.

FLOPs per token: 2·37B = 74 GFLOPs (compute is like a 37B dense model).
Memory: 671B·FP8 = 671 GB weights (full). The inactive experts take up HBM but aren't touched during a given step — unless different tokens in the batch activate different experts, in which case you effectively read more.

### 9.3 Implementation challenges

**Load imbalance:** some experts get more tokens than others. Auxiliary loss during training pushes for balance; at inference, expert replication (assigning hot experts to multiple GPUs) helps.

**All-to-all:** tokens scatter across expert-parallel GPUs, then gather back. This is the dominant cost of MoE inference; DeepEP and similar kernels are the state of the art.

**Shared experts (DeepSeek):** a small set of "shared" experts always activate, for common patterns. Reduces the load on routed experts.

See [Parallelism_Strategies](../Part_7_Distributed_Systems_and_Scale/Parallelism_Strategies.md) for expert parallelism details.

---

## 10. KV Cache (Sketch; full detail in separate page)

During decode, at position T:
- Compute new Q, K, V for just position T
- Append new K, V to the cache
- Attend: `scores = Q · K_full^T / sqrt(dₕ)` where K_full includes all T+1 positions
- Softmax, `P·V_full`, done

The cache layout matters for both correctness (kv positions must be in the right order) and performance (contiguous access for coalesced loads). See [KV_Cache](../Part_6_Inference_Engines_and_Serving/KV_Cache.md) for the deep dive on PagedAttention's layout.

---

## 11. Activations and Intermediate Tensors

During forward pass:

Per layer, per token:
- Input: D floats
- QKV output: 3·D floats (before GQA reduction, or D + 2·D·(K/H) with GQA)
- Attention output: D floats
- FFN intermediate (SwiGLU): 2 · (8/3)·D = 16/3 · D floats
- Residual stream: persists, D floats

For Llama-3-70B, D=8192, per token per layer ≈ 8·8192 = 65K activations = 128 KB. Over 80 layers with no checkpointing: 10 MB per token of activation memory. At batch=1 seq=4096: 40 GB of activation memory.

In training, activations balloon because you must save them for backprop. Activation checkpointing (rematerialization) is the standard solution — save only the residual stream; recompute interior activations on the backward pass.

In inference, activations are transient (freed after the layer completes) and take minimal memory compared to weights + KV.

---

## 12. Common Interview Questions

**Q: Walk me through the forward pass of a transformer layer.**
A: Input X [B, S, D]. Pre-norm: X' = norm(X). Compute Q/K/V via linear projections. Reshape for multi-head. Attention: `softmax(QK^T/sqrt(dₕ) + mask) · V`. Reshape and output project. Add residual. Pre-norm again. FFN: `W_down · SiLU(X W_gate) * (X W_up)`. Add residual. Output [B, S, D].

**Q: Why does GQA exist?**
A: KV cache memory dominates at long contexts. MHA with H heads has H KV pairs, using H·seq·dₕ per token. MQA shares a single KV pair across all query heads, cutting KV memory by H×, but can hurt quality. GQA is the compromise: groups of query heads share KV, giving most of the savings with little quality hit. Llama-3 uses 8 KV heads for 64 query heads — 8× KV reduction.

**Q: What's RoPE and why is it used?**
A: Rotary Position Embedding rotates pairs of query and key dimensions by angle proportional to position. Dot products between rotated Q and K encode relative position, so `QK^T` naturally has positional info. Replaces absolute learned embeddings and enables better length extrapolation (with tricks like NTK-aware scaling for context extension).

**Q: What's the difference between pre-norm and post-norm transformers?**
A: Pre-norm applies LayerNorm/RMSNorm **before** the submodule (attention or FFN), with residual added after: `y = x + f(norm(x))`. Post-norm applies norm after the residual: `y = norm(x + f(x))`. Pre-norm trains stably at 100+ layers; post-norm becomes unstable at depth. All modern LLMs use pre-norm.

**Q: Why SwiGLU instead of GeLU MLP?**
A: SwiGLU is a Gated Linear Unit with SiLU activation: `SiLU(XW_gate) * (XW_up) · W_down`. Three matrices instead of two. The multiplicative gate modulates the up-projection per-feature, giving the model more expressive capacity per parameter than GeLU. Empirically cuts loss at equal parameter budget.

**Q: Roughly how many FLOPs per token in a forward pass?**
A: About 2·P, where P is parameter count. Derivation: each weight participates in one multiply + one add = 2 FLOPs, and we touch each weight once per token. Llama-3-70B → ~140 GFLOPs/token.

**Q: What is a mixture-of-experts model?**
A: Each FFN layer is replaced with E parallel FFNs ("experts") and a router. Per token, the router picks the top-k experts (k=2-8) and combines their outputs. Only the picked experts compute. Gives you a large "total" parameter count (knowledge capacity) with the active compute of a much smaller model. Inference challenges: balanced routing and all-to-all communication.

**Q: How large is the KV cache for Llama-3-70B at 128K context?**
A: Per layer per token: 2 (K+V) · 8 (KV heads) · 128 (head dim) · 2 (FP16) = 4 KB. Over 80 layers: 320 KB per token. At 128K context: ~40 GB of KV cache per request.

**Q: Why is the output projection (to logits) sometimes tied with the input embedding?**
A: Both are D×V matrices. Tying them halves the embedding parameters (which can be 10-20% of total for small models with large vocab) and provides a mild regularization. Most LLMs tied them through GPT-NeoX era; some later models untied them (GPT-4-class) for slight quality gains.

**Q: What's the arithmetic intensity of the attention operation during decode?**
A: Decode attention reads (L×dₕ) KV per token of cache plus 1 new Q; computes ~4·L·dₕ FLOPs. AI ≈ 2 FLOPs/byte — deeply memory-bound. Batching helps because multiple Q tokens share the same K/V cache reads for the prefill they share, but in independent requests each has its own KV cache, so batching doesn't directly amortize KV reads. However, weight reads (which dominate decode memory) do amortize across batch.

---

## 13. Further Reading

- "Attention Is All You Need" (Vaswani et al., 2017)
- "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)
- "GQA: Training Generalized Multi-Query Transformer Models" (Ainslie et al., 2023)
- "GLU Variants Improve Transformer" (Shazeer, 2020)
- "Root Mean Square Layer Normalization" (Zhang and Sennrich, 2019)
- The annotated Transformer (Harvard NLP, for line-by-line code)
- Andrej Karpathy's "Let's build GPT" video series

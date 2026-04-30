# State Space Models and Hybrid Architectures

Mamba, Mamba-2, RWKV-7, and hybrid stacks (Jamba, Zamba, Falcon-Mamba, Granite-Mamba, Hymba) are the most credible competitors to pure-transformer architectures. This page covers the SSM math, the kernel patterns (parallel scan, chunked recurrent), the hybrid designs that ship in 2025, and the inference engineering they require.

**Prerequisites**: [Attention_Mechanisms](Attention_Mechanisms.md), [CUDA_Optimization](../Part_4_GPU_Kernel_Engineering/CUDA_Optimization.md), [KV_Cache](../Part_6_Inference_Engines_and_Serving/KV_Cache.md).

---

## 1. Why SSMs Came Back

Transformer attention is O(N²) compute and O(N) memory per layer. SSMs (state-space models) achieve **O(N) compute and O(1) memory** per layer (after the per-step state). At long sequence lengths this is a structural advantage.

Three things converged in 2023–2024 to make SSMs competitive:
1. **Selective state spaces** (Mamba): the SSM's parameters become input-dependent, gaining the expressive selectivity transformers have.
2. **Hardware-aware kernels**: the parallel-scan / chunkwise algorithms map well to GPUs.
3. **Hybrid designs**: combining a few attention layers with many SSM layers gets transformer quality with SSM efficiency.

By 2026, every major lab has released or evaluated hybrid models.

---

## 2. SSM Foundations

A continuous-time SSM:
```
h'(t) = A · h(t) + B · x(t)
y(t)  = C · h(t) + D · x(t)
```
- `h(t)` ∈ R^N: hidden state.
- `x(t)` ∈ R: scalar input (per channel).
- `A`, `B`, `C`, `D`: matrices.

Discretized (zero-order hold) with step size `Δ`:
```
h_t = A_bar · h_{t-1} + B_bar · x_t
y_t = C · h_t
```
where `A_bar = exp(Δ · A)`, `B_bar ≈ Δ · B`.

This is a linear recurrent network. For one channel, applied across N tokens, it's a 1D convolution with a particular kernel structure.

### 2.1 Why Linear RNNs Are Fast

Linear recurrence can be parallelized via **prefix-scan** algorithms in O(log N) depth. So sequence-parallel inference and training are tractable, unlike a generic RNN.

### 2.2 The Stability Trick (HiPPO / S4)

Choosing `A` from the HiPPO (high-order polynomial projection operators) class makes the SSM able to remember long histories well. S4 demonstrated this; later models simplified the parameterization while keeping the spirit.

---

## 3. Mamba (Selective SSM)

Gu & Dao, late 2023. The breakthrough: **make A, B, C input-dependent** so the model can "select" what to keep.

```
For each token x_t:
  Δ_t = softplus(Linear(x_t))         # selective time step
  B_t = Linear(x_t)                    # selective input gate
  C_t = Linear(x_t)                    # selective output gate
  A_t = exp(Δ_t · A)                   # discretization with selective Δ
  h_t = A_t · h_{t-1} + Δ_t · B_t · x_t
  y_t = C_t · h_t + D · x_t
```

Now the recurrence isn't strictly linear in token order (parameters change per token). But for any fixed sequence, the recurrence is a "lin-non-stationary" linear system, still parallelizable.

### 3.1 Selective Scan Algorithm

The hardware-aware selective-scan kernel computes `h_t` for all t in parallel using a tree-reduction over per-token state-update operators. Mamba's CUDA kernel is the reason it's competitive at scale.

### 3.2 Mamba-1 Properties

- O(N) compute, O(1) state memory (per channel).
- No KV cache for SSM layers.
- Quality comparable to dense Transformer at small-mid scale.
- Quality gap appears at frontier scale → motivates hybrids.

---

## 4. Mamba-2 (SSD: State Space Duality)

Dao & Gu, 2024. Reformulates Mamba's selective scan as a structured matrix operation:

```
Y = SSM(X) ≈ M · X
```
where M is a **structured semi-separable matrix** (lower triangular with specific factorization).

### 4.1 Why It Matters

- **Matmul friendliness**: SSD computes the SSM via standard matmuls + small recurrences, exploiting tensor cores. Original Mamba kernels did element-wise work on CUDA cores. Mamba-2 throughput is much higher.
- **Larger heads**: SSD generalizes to multi-head structure analogous to attention; each head has its own SSM.
- **Theoretical bridge**: SSD reveals that selective SSMs and linear attention live in the same family. Helped explain MaaT (Memory-as-a-Transformer-equivalent).

Production status: Mamba-2 powers most 2025 hybrid models.

---

## 5. RWKV (Receptance Weighted Key Value)

Parallel line of work; RWKV-4, -5, -6, -7 across 2023–2025. Uses a different formulation:
- A linear-attention-like time mixing (channel mixing → token mixing).
- Time-decay weights instead of attention scores.
- Pure linear-RNN inference (constant per-step memory).

RWKV-7 (early 2026 release) approaches transformer quality at small-mid scale; remains an open community effort. Mostly outside the major labs' production stacks but widely used in open-source.

---

## 6. Linear Attention and Friends

Other O(N) attention-substitutes:
- **Linear attention** (Katharopoulos et al., 2020): replace softmax(QK^T)V with `φ(Q)·(φ(K)^T·V)` — fully linear. Quality lags transformers but improving.
- **Lightning Attention** (used in MiniMax): linearizable attention with hardware-aware kernel. Reported very long-context efficiency.
- **TransNormer**: replaces softmax with normalization for linearization.
- **GLA / Gated Linear Attention**: adds input-dependent gating (Mamba-flavor) to linear attention.

These all sit in the "subquadratic attention" family alongside SSMs.

---

## 7. Hybrid Architectures (2025 Standard)

Pure SSMs lag pure transformers at large scale. Hybrids combine them.

### 7.1 Jamba (AI21, 2024)

```
Jamba-1.5 layout (52B / 12B active):
  Block = [Mamba-2 × 7, Attention × 1]   ← 7:1 SSM-to-attention ratio
  + MoE in some blocks
  Total layers: 32, with 4 attention layers (every 8th)
```

Most layers are SSM (cheap + linear); a few attention layers anchor global mixing. KV cache exists only in the 4 attention layers → **massive long-context efficiency**.

### 7.2 Zamba-1, Zamba-2 (Zyphra, 2024–2025)

Mamba backbone with **shared attention block** stitched in periodically. Different placement strategy than Jamba.

### 7.3 Falcon-Mamba (TII, 2024)

Pure Mamba-based 7B model. Demonstrates that even non-hybrid SSMs can be competitive at smaller scale.

### 7.4 Granite-Mamba (IBM, 2025)

IBM's hybrid for Granite-3 family. Code-focused.

### 7.5 Hymba (NVIDIA, 2025)

Hybrid where each layer has **parallel** SSM and attention heads (same layer), instead of alternating layers. Tighter coupling, claims higher quality at small scale.

### 7.6 Pattern: How Many Attention Layers?

Empirically, ~1 attention layer per 6-8 SSM layers is enough to match dense transformer quality. The attention layers typically use sliding-window or full attention; their KV cache dominates the memory budget but is only L_attn / L_total of total.

### 7.7 Linear Attention + Standard Attention Hybrids

MiniMax-Text (2025) uses **lightning linear attention** for 7/8 layers and full attention for 1/8. Achieves 4M context with reasonable cost.

---

## 8. Inference Engineering for SSMs

### 8.1 State Management

Per sequence, per SSM layer:
- **Recurrent state**: `(d_state)` per channel, fixed size. For a model with d=4096 channels and d_state=128: 4096 × 128 = 512K floats = 1 MB per token-position (wait, no — per sequence).

Per sequence, per SSM layer: `(d_inner, d_state)` floats ≈ 1 MB FP16. Across 32 layers: ~32 MB per sequence. **Independent of sequence length** — this is the killer feature.

### 8.2 Continuous Batching With Hybrid Models

Each in-flight sequence has:
- KV blocks (for attention layers) — paged as usual.
- SSM state tensors (one per SSM layer) — fixed-size pool.

The serving engine manages both. vLLM and SGLang have added hybrid model support since 2024.

### 8.3 Prefill Patterns

- **Attention layers**: standard chunked prefill, FlashAttention.
- **SSM layers**: parallel selective scan over the prefill chunk. Mamba-2's SSD makes this efficient (matmul-friendly).
- Bookkeeping: state passes layer-by-layer; final per-sequence state stored.

### 8.4 Decode Patterns

- **Attention layers**: standard paged attention.
- **SSM layers**: one-step recurrence per decode token. Trivial compute; just update state.

Decode is dominated by attention layers' KV reads. SSM steps are nearly free.

### 8.5 Parallelism

- **TP**: SSM layers split across heads (Mamba-2 has multi-head structure). Similar pattern to attention TP.
- **PP**: layer-by-layer; works as in transformers.
- **EP**: only relevant if MoE is mixed in (Jamba does).
- **CP**: less critical; SSM is already O(N) per layer. But for ultra-long context training, ring-style scan parallelism helps.

### 8.6 Quantization

FP8 / INT4 for SSM weights: similar story to transformer FFN — works fine with calibration. State quantization is more delicate (state is the model's "memory"; precision matters). Most production deployments quantize weights but keep state in FP16/BF16.

---

## 9. Quality Considerations

### 9.1 Where SSMs Lag

- **In-context retrieval**: pulling a specific fact from a precise position. SSMs compress all history into a fixed-size state, blurring fine details. NIAH on pure Mamba is poor; hybrids fix this with the attention layers.
- **Long copy / repeat tasks**: similar reason.
- **Few-shot ICL**: somewhat weaker without explicit attention heads to "look back."

### 9.2 Where Hybrids Win

- **Long-context efficiency** without the linear-context KV cost.
- **Throughput**: SSM layers are much faster per token at long context than attention.
- **Memory at long generation**: SSM state is constant; transformer KV grows linearly.

### 9.3 Training

Hybrids train with the same general recipe as transformers: tokenizers, datasets, schedules. Some layer-specific care (Mamba init, SSD parameter ranges) but mainstream open-source code supports them now.

---

## 10. Diffusion Language Models

Tangentially related to SSM/hybrid because they're another non-transformer-autoregressive direction:

- **LLaDA, Mercury**: diffusion-based language models that generate tokens via iterative denoising rather than left-to-right autoregression.
- **Trade-off**: parallel decoding (faster generation per step) vs more steps + more compute per token.
- **Status (2026)**: research-grade in production, not yet flagship-tier quality at scale, but active development.

A diffusion LM doesn't have a KV cache in the same sense — it iteratively refines a full token sequence. Inference engineering looks completely different.

---

## 11. Common Pitfalls

- **Treating Mamba like a transformer for serving**: no KV cache, fixed state — different memory math. Serving frameworks must support both state types.
- **Quantizing state aggressively**: precision loss accumulates over the recurrence; state in FP16 is the safe default.
- **Assuming pure SSMs scale like transformers**: they don't, beyond mid-scale. Hybrids are the path to frontier quality.
- **Training without selective-scan kernels**: throughput drops 5-10× without them. Use the official Mamba kernels or Triton equivalents.
- **NIAH testing on Mamba and concluding "long context works"**: NIAH is exactly the task SSMs struggle with; use diverse benchmarks.
- **Forgetting the attention-layer KV** in hybrid memory budgeting: they dominate even at low frequency.

---

## 12. Common Interview Questions

**Q: What is a state-space model and why is it relevant to LLMs?**
A: A linear recurrent network that maintains a fixed-size hidden state. Compute is O(N) and state memory is O(1) per channel — much cheaper than transformers at long context. Selective SSMs (Mamba) make the state update input-dependent, recovering the expressivity transformers have.

**Q: What's the difference between Mamba-1 and Mamba-2?**
A: Mamba-2 reformulates the selective scan as a structured matrix operation (state-space duality). This unlocks tensor-core-friendly matmul-based kernels for much higher throughput, and generalizes the multi-head structure analogous to attention. Mamba-2 powers production hybrids like Jamba-1.5.

**Q: Why don't pure SSM models match transformers at frontier scale?**
A: SSMs compress history into a fixed-size state, losing precise recall of distant tokens. Tasks requiring exact look-back (long-document QA, in-context learning, few-shot copying) suffer. Hybrids restore this capability by mixing in a few attention layers.

**Q: What's a hybrid SSM-transformer model?**
A: Stacks where most layers are SSM (Mamba-2) and a small fraction (1 in 6-8) are attention layers. Provides transformer-level quality with SSM-level long-context efficiency. Examples: Jamba, Zamba, Hymba.

**Q: What's the KV cache story for hybrid models?**
A: Only the attention layers have KV cache; SSM layers have a constant-size recurrent state per sequence. So at long context, KV memory scales as L_attn × N (small fraction of layers × tokens) instead of L × N. Plus a small per-sequence SSM state.

**Q: How does the inference engine handle hybrid models?**
A: Per sequence, manage two state types: paged KV blocks for attention layers + fixed-size SSM state tensors for SSM layers. Continuous batching works the same; the engine just dispatches the right kernel per layer. vLLM and SGLang both support this.

**Q: Why is Mamba's selective-scan kernel important?**
A: Without a hardware-aware kernel, the recurrence can't be parallelized efficiently across GPUs and looks like an O(N) sequential bottleneck. The selective-scan kernel uses a parallel prefix-scan to compute h_t for all t in O(log N) depth, restoring GPU utilization.

**Q: What's the difference between SSMs and linear attention?**
A: Both are O(N) attention substitutes. Linear attention factorizes softmax(QK^T) into kernel(Q) · (kernel(K)^T · V); SSMs use a recurrent linear update on a hidden state. Mamba-2's SSD shows the two families are deeply related; both can be expressed as structured-matrix operations.

**Q: When would you choose a hybrid model over a pure transformer?**
A: When inference cost at long context dominates: long-document workflows, agent contexts, very long generation. Pure transformer remains preferred for short-context interactive chat where attention's expressivity outweighs the cost.

**Q: How do you serve a 10M-context hybrid model?**
A: Hybrid + MLA-equivalent attention compression + FP8 KV + chunked prefill + aggressive prefix caching. SSM layers handle the bulk of long-context cheaply; attention layers (few in number) get the heavy compression treatment.

**Q: What's RWKV?**
A: Parallel line of linear-attention-flavored models with time-decay-based mixing. Pure linear-RNN inference. Open-source community project; quality approaching transformers at small-mid scale (RWKV-7). Outside major labs' production stacks but widely deployed in open-source.

**Q: How would you write a Mamba-2 kernel?**
A: Implement the SSD state-space duality — compute the SSM via blocks: each block is a matmul (efficient on tensor cores), boundary states pass between blocks via small recurrences. The official kernels are CUDA + Triton; reference implementations exist in the Mamba repository.

**Q: How does a diffusion LM differ from autoregressive?**
A: Generates tokens via iterative denoising of a full-sequence representation rather than left-to-right. Allows parallel decoding (multi-token per step) at the cost of more total compute. No KV cache in the standard sense. Production-grade only at small-mid scale as of 2026.

---

## 13. Further Reading

- "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (Gu & Dao, 2023).
- "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality" (Dao & Gu, 2024) — Mamba-2.
- "Jamba-1.5: Hybrid Transformer-Mamba Models at Scale" (AI21, 2024).
- Zamba and Falcon-Mamba technical reports.
- "Hymba: A Hybrid-head Architecture for Small Language Models" (NVIDIA, 2024).
- RWKV-5/6/7 papers (BlinkDL).
- "Lightning Attention" (MiniMax, 2024).
- "LLaDA: Large Language Diffusion with mAsking" (RUC + Microsoft, 2025).
- Mercury / Inception Labs diffusion-LM materials.

---

**Next:** [Blackwell_Architecture](../Part_2_Microarchitecture/Blackwell_Architecture.md), [Modern_Quantization_Frontier](Modern_Quantization_Frontier.md).
**See also:** [Frontier_Models_2025_2026](Frontier_Models_2025_2026.md), [Long_Context_Engineering](../Part_6_Inference_Engines_and_Serving/Long_Context_Engineering.md), [Modern_KV_Compression](../Part_6_Inference_Engines_and_Serving/Modern_KV_Compression.md).

# Frontier Models 2025–2026

A working architectural map of the models you must be able to discuss in any 2026 AI-infra interview. This is **not** an introduction to transformers — it focuses on the specific 2024–2026 design choices: MLA, MoE shaping, multi-token prediction, native sparse attention, hybrid SSM/Transformer stacks, and the systems implications of each.

**Prerequisites**: [Transformer_Internals](Transformer_Internals.md), [Attention_Mechanisms](Attention_Mechanisms.md), [Modern_MoE](Modern_MoE.md), [Long_Context_Engineering](../Part_6_Inference_Engines_and_Serving/Long_Context_Engineering.md).

---

## 1. The 2025–2026 Landscape

By early 2026, the open and closed model landscape has consolidated around a few architectural patterns:

| Family | Active params | Total params | Architecture highlights |
|--------|---------------|--------------|-------------------------|
| Llama-4 (Scout / Maverick / Behemoth) | 17B / 17B / 288B | 109B / 400B / 2T | Native MoE, iRoPE (interleaved RoPE/NoPE), 10M context |
| DeepSeek-V3 | 37B | 671B | MLA, MoE 256/8 experts + 1 shared, MTP, FP8 native |
| DeepSeek-R1 / R1-0528 | 37B | 671B | V3 + RL pipeline, long-CoT, distilled small variants |
| Qwen-3 / Qwen-3-Max | 235B (active 22B) MoE / 480B+ | dense + MoE variants, "thinking" mode, agentic tool use |
| Mistral Large 2 / Magistral | 123B / 24B reasoning | dense, function-calling, multilingual |
| Gemma-3 | 1B / 4B / 12B / 27B | dense + sliding-window attention 5:1 ratio with global |
| Nemotron-Ultra-340B | 340B dense + reasoning | dense, NVIDIA, long-CoT alignment |
| Llama-3.3-70B / 405B | 70B / 405B | dense, GQA, the de-facto "FP8 70B" baseline |
| Claude-3.7 / Claude-4 (Opus/Sonnet/Haiku) | undisclosed | dense + MoE rumored, hybrid reasoning ("extended thinking") |
| GPT-5 / o3 / o4-mini | undisclosed | likely MoE + reasoning; o-series uses search at decode |
| Gemini-2.5 (Flash / Pro) | undisclosed | MoE + native multimodal + long context |
| Mamba-2 / Jamba-1.5 | 1.6B / 52B | SSM-only / hybrid SSM-MoE-Transformer |
| Granite-3 (IBM) | 2B / 8B | dense + Granite-MoE |
| InternLM-3, Yi-2, Doubao-1.5 | varied | varied; competitive Chinese-lab releases |

The benchmark-relevant patterns to know for systems work:

1. **MoE everywhere at scale** with much higher expert counts (256+) and **fine-grained, shared-expert** designs.
2. **Latent attention (MLA)** as a KV-compression alternative to GQA, with 30× KV reduction.
3. **Multi-token prediction (MTP)** as a built-in speculative-decode mechanism.
4. **FP8 native** training and inference, FP4 for inference.
5. **Reasoning-tuned models** with long-chain-of-thought RL (o1-style).
6. **Hybrid SSM-Transformer** stacks for long-context efficiency.
7. **10M+ context** in some families via RoPE + NoPE interleaving and chunked attention.

---

## 2. DeepSeek-V3 (671B / 37B active)

The single most influential open architectural release of 2024–2025. Trained on 14.8T tokens for ~$5.5M.

### 2.1 Architecture Highlights

- **61 layers**, hidden dim 7168.
- **MLA** (Multi-head Latent Attention): KV projected into a 512-dim latent + 64-dim rotary part. KV per token ≈ (512 + 64) × 2 bytes × 61 layers = **70 KB** vs ~600 KB for dense MHA at same head count.
- **DeepSeekMoE**: 256 routed experts + **1 shared expert** per layer. Top-8 routing per token. Active params per token ≈ 37B / 671B.
- **Auxiliary-loss-free load balancing**: per-expert bias term, updated by EMA, replaces the load-balancing auxiliary loss that hurt main-task training in earlier MoE designs.
- **Multi-Token Prediction (MTP)** training objective: extra heads predict tokens t+1, t+2, …; used as built-in speculative decoding at inference.
- **FP8 native training** with per-tile scales (E4M3 forward/backward, FP32 master, custom scaling pipeline).
- **GRPO** for the post-training RL phase.

### 2.2 MLA Math

For each layer, instead of `(K, V) = (W_K, W_V) · x` with K and V each of size H·d (e.g., 128·128 = 16384 dim per token), DeepSeek-V3 stores:

```
c_KV = W_DKV · x              # (d_c = 512,)
k_R  = RoPE(W_KR · x)         # (d_R = 64,) — decoupled rotary part, shared across heads

# At attention time:
K = [W_UK · c_KV  ;  k_R]     # reconstruct full K per head
V = W_UV · c_KV               # reconstruct V per head
```

Storage per token, per layer: **`d_c + d_R = 576` floats** = **1.15 KB** in FP16 (or 70 KB across 61 layers). This is ~30× less than naive MHA and ~3× less than GQA at the same H_kv.

Cost: extra projections at attention time, but they're tiny vs the BW savings during decode.

### 2.3 MoE Layout

```
For each MoE layer:
  - 1 shared expert (always active, no routing)
  - 256 routed experts (each ~smaller than dense FFN)
  - Top-k = 8 active routed experts per token
  - Gating with sigmoid (not softmax) + group-wise top-k restriction
  - Per-expert bias updated by EMA to balance load (no aux loss)
```

The shared expert absorbs general computation; routed experts specialize. Fine-grained: each routed expert is 1/k the size of a dense FFN, so total active size matches a dense FFN of the original size.

### 2.4 Multi-Token Prediction

During training, attach D=1 extra prediction heads. For position i, the model predicts tokens at i+1 and i+2. Loss = standard CE at i+1 + 0.3 × CE at i+2.

At inference, the second head's predictions can be used as speculative draft tokens, verified by the main model in the next forward pass — built-in speculative decoding without a separate draft model. Reported acceptance rate ~85%, ~1.8× decode speedup.

### 2.5 Systems Implications

- **MLA changes serving math** dramatically. KV cache budget per token is small enough that 1M-token contexts and large batches coexist on a single 8×H100 node.
- **EP becomes mandatory**: 256 experts × 61 layers = 15K total experts. Cannot fit all on every GPU. EP=8–32 or higher with all-to-all per layer.
- **MTP fits naturally** with continuous batching: one extra position scored per step, accept if hash matches, otherwise rollback one token.
- **FP8 native** means activations and weights are FP8 throughout. Production serving runs at FP8 by default.

---

## 3. DeepSeek-R1 (Reasoning Pipeline)

R1 takes V3 (or a checkpoint thereof) through a three-stage RL pipeline:

1. **R1-Zero**: Pure RL on V3 with rule-based rewards (math correctness, code passing tests). No SFT. Demonstrates that reasoning can emerge from RL alone but produces unreadable output.
2. **R1-Cold-Start**: Cold-start SFT on a small set of curated CoT examples → seed a more readable policy.
3. **R1-RL + Distillation**: GRPO RL on the cold-started model. Then distill to smaller dense models (1.5B–70B) by training them on R1's CoT outputs.

### 3.1 GRPO (Group Relative Policy Optimization)

The RL algorithm. The fundamental difference from PPO (Proximal Policy Optimization) is that GRPO drops the need for an external value model (Critic) that predicts the expected return of a state.

**PPO Objective**:
$L_{PPO} = \mathbb{E}[\min(\frac{\pi_\theta(a|s)}{\pi_{old}(a|s)} \hat{A}_t, \text{clip}(\dots) \hat{A}_t)]$
where $\hat{A}_t$ is the Advantage estimated by a Critic model ($A = R - V(s)$). This requires keeping the Critic model (often the same size as the policy) in memory.

**GRPO Objective**:
Instead of a Critic, sample a group of $G$ completions $\{y_1, y_2, \dots, y_G\}$ for a prompt $q$.
Score each with a reward model or rule-based verifier to get $\{r_1, r_2, \dots, r_G\}$.
Compute relative advantages by normalizing within the group:
$\hat{A}_i = \frac{r_i - \text{mean}(r)}{\text{std}(r)}$

The GRPO loss is then:
$L_{GRPO}(\theta) = \mathbb{E}_{q, \{y_i\}}\left[\frac{1}{G} \sum_{i=1}^G \min\left(\frac{\pi_\theta(y_i|q)}{\pi_{old}(y_i|q)} \hat{A}_i, \text{clip}(\dots) \hat{A}_i\right) - \beta \mathbb{D}_{KL}(\pi_\theta \| \pi_{ref})\right]$

**Memory savings**: Only 2 model copies (policy + reference, e.g., both 671B) instead of 4 (policy, reference, reward, critic). Critical for RL on huge models.

### 3.2 Systems Implications

- **Online sampling at scale** is the bottleneck — you need a fast inference engine (vLLM/SGLang) inside the training loop generating K samples per prompt.
- **Long generations** (10K-100K tokens) per sample → KV cache pressure during sampling.
- **Reward model**: rule-based for math/code (cheap), or LLM-judge for open-ended (expensive).
- **Distillation pipeline**: huge inference job to generate training data for smaller models.

---

## 4. Llama-4

Released April 2025 as a family of native MoE models.

| Variant   | Active | Total | Experts | Routing | Context |
|-----------|--------|-------|---------|---------|---------|
| Scout     | 17B    | 109B  | 16      | top-1   | 10M     |
| Maverick  | 17B    | 400B  | 128     | top-1   | 10M     |
| Behemoth  | 288B   | ~2T   | 16      | top-1   | unreleased |

### 4.1 Architectural Choices

- **Top-1 routing**: each token to exactly one expert. Simpler and lower-comm than top-k > 1.
- **iRoPE** (interleaved RoPE + NoPE): every fourth attention layer has **no positional encoding** at all (NoPE) — empirically improves long-context generalization. The other layers use RoPE with extended base frequency.
- **Chunked attention** in long-context layers: window-based local attention combined with global attention layers.
- **Native multi-modal**: image and video tokens flow through the same backbone (early-fusion).

### 4.2 Systems Implications

- **Top-1 routing** means each token uses exactly one expert; all-to-all is the simplest possible pattern but still 2× per MoE layer.
- **10M context** is a serving challenge: even with MLA-equivalent compression, memory per long-context conversation is enormous. Chunked attention + KV compression are essential.
- **iRoPE** has zero serving cost — just an architectural detail that the inference engine respects via the per-layer config.

---

## 5. Qwen-3 / Qwen-3-Max

Qwen-3 (June 2025) introduced a "thinking mode" toggle: the model can be asked to reason explicitly (long CoT) or respond directly. Same weights, different inference behavior.

- 235B-parameter MoE variant with 22B active.
- Support for tool-use / agentic tasks integrated at the token level.
- Strong on multilingual + code.

Systems implication: the model exposes a system prompt switch (`/think` vs `/no_think`) that changes generation length distributions dramatically. Capacity planning must account for both modes.

---

## 6. Gemma-3 and the Sliding-Window Renaissance

Gemma-3 uses a **5:1 ratio of local-to-global** attention layers:
- 5 layers with **sliding-window attention** (W=4096) — O(N·W) cost.
- 1 layer with global attention.
- Repeat.

This dramatically reduces KV cache for the local layers (only W tokens) while preserving global mixing. Standard for "small but capable" models in 2025.

**KV math for Gemma-3-27B at 128K context**:
- Naive: 32 layers × 128K × ... = huge.
- 5:1 sliding: 5 layers × 4K + 1 layer × 128K, repeated → ~7× reduction.

---

## 7. Hybrid SSM-Transformer (Jamba, Zamba, Falcon-Mamba)

Replace some transformer layers with **Mamba/Mamba-2** state-space layers. SSM cost is O(N) per layer (no KV cache!), so scaling to long context is cheap.

```
Jamba-1.5 layout (52B / 12B active MoE):
  - SSM (Mamba) layers
  - MoE Transformer layers
  - Ratio ~7:1 SSM:Attention
```

Inference engines must support **two state types**: KV cache for transformer layers + recurrent state for SSM layers. SGLang and vLLM have added Mamba-2 support; full TP/PP for hybrids is still maturing.

See [State_Space_Models_and_Hybrids](State_Space_Models_and_Hybrids.md).

---

## 8. Closed Models — What's Inferable

We don't get architectural details for GPT-5/o3/Claude-4/Gemini-2.5 directly, but signals:

- **Reasoning ("thinking") modes** that consume large amounts of test-time compute, billed as separate "reasoning tokens." Suggests a long-CoT RL pipeline similar to R1/o3.
- **Variable thinking budgets** (Anthropic "extended thinking", Gemini "thinking budget"): the model is asked to reason for N tokens before producing user-visible output.
- **Rumored MoE** for Gemini-2.5 and GPT-5 based on per-token cost asymmetries (input cheap, output more expensive — typical of MoE serving).
- **Multi-modal native**: image and audio input/output, often via separate modality encoders feeding into a unified backbone.

For serving these architectures internally, the systems are mostly the same as open MoE + reasoning models discussed above.

---

## 9. The "Reasoning Tax" on Inference Infrastructure

Test-time compute scaling shifts the load profile dramatically:

| Workload type | Output tokens / request | KV pressure | Compute pressure |
|---------------|--------------------------|--------------|-------------------|
| Classic chat  | 100-300                  | low          | medium            |
| RAG           | 200-1000                 | low (chunks) | medium            |
| Agentic       | 500-2000 with tools      | medium       | medium            |
| Long-CoT (o1/R1) | 5K-100K                | high         | high              |
| Deep Research / Operator | 10K-1M+         | very high    | very high         |

Reasoning workloads:
- Need **giant KV pools** because outputs are 10-100× longer.
- Push **TPOT** harder (every token matters; users wait minutes).
- Spread compute across many decode steps → spec-decode is more valuable.
- Disaggregation pays off more because long decodes interfere with prefills more.

See [Reasoning_Models](Reasoning_Models.md) for the full treatment.

---

## 10. What's Likely Next (Forward-Looking)

Educated guesses based on 2025–2026 trends:

- **NVFP4 inference** standard by mid-2026. FP4 weights + scaled FP8 activations on Blackwell.
- **Bigger NVL domains**: NVL576+ post-Rubin, enabling much larger TP/EP within a fabric.
- **More hybrid stacks**: Linear-attention or SSM intermixed with transformer in flagship models, especially for 1M+ context.
- **Online RL becoming routine**: post-training RL with online sampling is no longer exotic; major infra work is in the inference-engine-inside-RL loop.
- **Native sparse attention** (NSA, MoBA) shipped in production for long context.
- **Diffusion-LM convergence**: large-scale diffusion language models (LLaDA, Mercury) maturing as an alternative to autoregressive.

You should be able to discuss any of these with an interviewer — and acknowledge the uncertainty.

---

## 11. Common Interview Questions

**Q: What is MLA and why does it matter?**
A: Multi-head Latent Attention (DeepSeek-V2/V3). Stores a low-rank latent `c_KV` (e.g., 512 dim) plus a small decoupled rotary part (~64 dim) per token, instead of full K and V per head. ~30× KV compression vs MHA, ~3× vs GQA. Enables huge batch × context products on a fixed HBM budget. Reconstructs full K, V at attention time via cheap projections.

**Q: How does DeepSeek-V3 differ from a standard MoE like Mixtral?**
A: (1) Far more experts (256 vs 8) with finer granularity. (2) Adds a shared expert always active per layer. (3) Uses **auxiliary-loss-free** load balancing via per-expert bias EMA. (4) Top-k = 8 routed expert routing. (5) MTP training objective. (6) FP8 native training. The systems impact: EP becomes mandatory and all-to-all costs are non-trivial.

**Q: What is multi-token prediction in DeepSeek-V3 and how is it used at inference?**
A: An auxiliary head predicts tokens at offsets +1 and +2 during training. At inference, the +2 prediction acts as a speculative draft — compared against the main model's prediction one step later. Achieves ~1.8× decode speedup with ~85% acceptance, no separate draft model.

**Q: Why is "auxiliary-loss-free" load balancing better than the classic auxiliary loss?**
A: Classic aux loss adds a term that pushes routing toward uniform; this competes with the main task and degrades quality at scale. Aux-loss-free instead adjusts a per-expert bias term in the gating logits via EMA, achieving balance without distorting the gradient of the main loss. DeepSeek showed this materially improves quality at large MoE.

**Q: What is GRPO and how does it differ from PPO?**
A: Group Relative Policy Optimization (DeepSeek-Math/R1). Removes the critic; instead samples K completions per prompt and uses (reward − group_mean) / group_std as the advantage. Memory savings (no critic) and simpler implementation. Used for R1's reasoning RL.

**Q: What architectural changes in Llama-4 are different from Llama-3?**
A: Native MoE (Llama-3 was dense); top-1 routing; iRoPE (every 4th attention layer is NoPE — no positional encoding) for long-context generalization; chunked + global attention combined; native multimodal early-fusion; 10M context.

**Q: Why are reasoning models harder to serve than chat models?**
A: Outputs are 10-100× longer (5K-100K tokens vs 100-300), KV cache pressure is much higher, TPOT becomes critical because users actually wait the full latency, and output-token economics dominate (inference cost roughly proportional to output tokens, and users do not control reasoning length).

**Q: What's the difference between sliding-window attention and chunked attention?**
A: Sliding window: each query attends only to the last W tokens. O(N·W). Used in Mistral, Gemma-3 (alternating with global). Chunked attention (Llama-4): the sequence is split into chunks; queries attend within their chunk plus a few global anchors. Different tradeoffs: sliding window is uniform, chunked has explicit "global" mixing points.

**Q: What is iRoPE?**
A: Llama-4's interleaved RoPE/NoPE. Every fourth attention layer uses **no positional encoding at all** (NoPE), the rest use RoPE with extended base frequency. Empirically improves long-context generalization without changing inference cost.

**Q: How does a hybrid SSM-Transformer like Jamba serve at inference time?**
A: SSM layers maintain a constant-size recurrent state per sequence (no growing KV cache). Transformer layers maintain the standard KV cache. Inference engine must support both state types per sequence. Long-context cost grows much more slowly because ~7/8 of layers are SSM, only ~1/8 contribute to KV.

**Q: Why are MoE models "cheap to serve but expensive to host"?**
A: At inference, only top-k experts run per token → active FLOPs scale with active params (e.g., 37B for DeepSeek-V3) not total (671B). But all 671B must reside in HBM somewhere because any token might route to any expert → high HBM cost, high cross-GPU comm in EP. Throughput per dollar can still be very competitive when batch is large enough to amortize EP all-to-all.

**Q: Why does FP8-native training matter?**
A: Halves bytes per matmul vs BF16, doubling tensor-core throughput on Hopper. Cuts HBM and inter-GPU communication. Saves ~20-40% wall-clock training time at scale. DeepSeek-V3 is the canonical demonstration that FP8 training works at frontier scale; everyone is moving this way.

**Q: What's likely to change about model architectures in 2026–2027?**
A: More hybrid SSM/attention stacks for long context; native sparse attention (NSA/MoBA) reaching production; FP4 / NVFP4 native; bigger MoE with even finer experts; reasoning RL becoming standard post-training; diffusion-LM as an emerging alternative.

---

## 12. Further Reading

- DeepSeek-V3 technical report (Dec 2024).
- DeepSeek-R1 paper (Jan 2025).
- DeepSeekMoE paper (2024).
- Llama-4 release notes and blog (Apr 2025).
- Mistral Magistral, Qwen-3 technical reports (2025).
- "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models" — origin of GRPO.
- Jamba and Zamba architecture papers.
- Gemma-3 technical report (Feb 2025).
- "Mixture-of-Experts at Scale" — Shazeer et al. (foundational), DeepSeekMoE for the modern formulation.
- NVIDIA Nemotron-Ultra-340B technical report.

---

**Next:** [Reasoning_Models](Reasoning_Models.md), [Modern_MoE](Modern_MoE.md), [State_Space_Models_and_Hybrids](State_Space_Models_and_Hybrids.md).
**See also:** [Attention_Mechanisms](Attention_Mechanisms.md), [Long_Context_Engineering](../Part_6_Inference_Engines_and_Serving/Long_Context_Engineering.md), [Modern_Quantization_Frontier](Modern_Quantization_Frontier.md).

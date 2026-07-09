# L6 — Algorithms & Model Architectures

> The layer where kernel operations (L5) become **model algorithms**: how attention works, how transformers are structured, how MoE routing distributes computation, how quantization maps continuous weights to discrete codes, and what the frontier models of 2025-2026 actually look like inside. L6 is the algorithmic core — every serving optimization in L8 and every training strategy in L7 derives from the architectural decisions catalogued here.

---

## Scope of this layer

L6 covers the **model-level algorithms and architectures** used in modern large language models and multimodal models.

1. **Transformer internals** — the full forward pass: embedding, multi-block transformer, unembedding. Attention block, FFN block, normalization (LayerNorm, RMSNorm), positional encodings (sinusoidal, RoPE, ALiBi, NoPE).
2. **Attention mechanisms** — scaled dot-product, MHA, MQA, GQA, MLA (DeepSeek V2/V3). Online softmax derivation. KV cache size formulas per variant.
3. **Mixture-of-Experts** — gating, top-k routing, load balance loss, aux-loss-free balancing (DeepSeek V3), shared experts, fine-grained experts, EP at NVL72/Helios scale.
4. **State-space models and hybrids** — Mamba/Mamba-2 SSM derivation, RWKV-7, Jamba/Zamba hybrid stacks, serving tradeoffs vs transformers.
5. **Quantization** — INT8/INT4 quantization, PTQ, GPTQ, AWQ, SmoothQuant. Per-tensor / per-channel / per-group. Quantization error bounds.
6. **Modern quantization frontier** — FP8 (E4M3/E5M2), FP6, FP4 (E2M1), MXFP4, NVFP4, SpinQuant, calibration pipelines, Transformer Engine v2.
7. **Frontier models 2025-2026** — survey: Llama-4, DeepSeek V3/R1, Qwen-3, Gemma-3, GPT-5/o3, Claude-4, Gemini-2.5. Key innovations per model.

L6 explicitly does **not** cover kernel implementations (→ L5), training infrastructure (→ L7), or serving systems (→ L8).

Note: Reasoning_Models has been moved to L7 (it is primarily a post-training story).

---

## Pages (read in order)

1. [Transformer_Internals](01_Transformer_Internals.md) — the complete transformer forward pass, block structure, normalization, positional encodings. Foundational for everything that follows.
2. [Attention_Mechanisms](02_Attention_Mechanisms.md) — scaled dot-product through MLA. Online softmax derivation. KV cache size per variant.
3. [Modern_MoE](03_Modern_MoE.md) — gating, routing, load balancing, shared experts, DeepEP, EP scaling.
4. [State_Space_Models_and_Hybrids](04_State_Space_Models_and_Hybrids.md) — Mamba SSM, RWKV-7, hybrid architectures.
5. [Quantization](05_Quantization.md) — integer quantization methods: PTQ, GPTQ, AWQ, SmoothQuant, error bounds.
6. [Modern_Quantization_Frontier](06_Modern_Quantization_Frontier.md) — FP8, FP4, NVFP4, MX formats, Transformer Engine.
7. [Frontier_Models_2025_2026](07_Frontier_Models_2025_2026.md) — architectural survey of production models, key innovations per model.

---

## What L6 hands off to L7 and above

| L6 fact | Higher-layer consequence |
|---|---|
| MLA cuts KV cache ~30x vs MHA | DeepSeek-V3 decode is less memory-bound (L8) |
| MoE top-k routing with EP | All-to-All communication dominates MoE training (L7) |
| FP8 (E4M3) has 3.6% dynamic range of FP16 | Calibration is critical; bad scales -> 5%+ accuracy loss (L7, L8) |
| RoPE enables length extrapolation via YaRN | Long-context serving becomes feasible (L8) |
| GQA reduces KV heads 8x | Multi-query serving fits more batch in same HBM (L8) |
| Mamba-2 has O(1) inference recurrence | Linear-time decode for SSM layers (L8) |
| Frontier models converge on similar architectures | Standardized serving paths (L8) |

---

## Reading order

Top-down: Index -> Frontier_Models_2025_2026 for the landscape -> back to Transformer_Internals for depth.

Strict bottom-up: Transformer_Internals -> Attention_Mechanisms -> Modern_MoE -> State_Space_Models -> Quantization -> Modern_Quantization_Frontier -> Frontier_Models.

After this layer, ascend to [L7 — Training Stack](../L7_Training_Stack/00_Index.md).


# Quantization

Lower-precision math is the single biggest knob in production inference: it shrinks weights, frees HBM for KV, doubles tensor-core throughput, and reduces interconnect bytes — all simultaneously. This page covers the precision formats, weight quantization (INT8/INT4/FP8/FP4), activation quantization, KV quantization, calibration, accuracy preservation, and the kernel implications.

**Prerequisites**: [GPU_Architecture](../Part_2_Microarchitecture/GPU_Architecture.md), [KV_Cache](../Part_6_Inference_Engines_and_Serving/KV_Cache.md), [CUDA_Optimization](../Part_4_GPU_Kernel_Engineering/CUDA_Optimization.md).

---

## 1. Number Formats

| Format | Bits | Range / encoding | Notes |
|--------|------|------------------|-------|
| FP32   | 32   | sign / 8 exp / 23 mant | Default; rare for inference. |
| TF32   | 19 effective | s/8/10 (in 32-bit slot) | Tensor-core matmul on Ampere+; same range as FP32, less precision. |
| BF16   | 16   | s/8/7 | Same range as FP32, half precision. Default for training & most inference. |
| FP16   | 16   | s/5/10 | Tighter range (6e-5 to 65504), more mantissa precision than BF16. |
| FP8 E4M3 | 8 | s/4/3, finite range ±448 | Hopper. Good for weights & forward activations. |
| FP8 E5M2 | 8 | s/5/2, range similar to FP16 | Better range, less precision. Good for gradients. |
| FP4 E2M1 | 4 | s/2/1, very limited | Blackwell. Aggressive weight quant; needs scale per micro-block. |
| INT8 | 8 | -128..127 | Symmetric or asymmetric. Per-channel scales typical. |
| INT4 | 4 | -8..7 | GPTQ/AWQ standard for weight-only. |
| MX (microscaling) | 4–8 | block of 32 with shared exponent | OCP MX standard, used by NVIDIA Blackwell tensor cores. |

### FP8 vs INT8

FP8 has an exponent → handles wide dynamic range. INT8 is uniform → simpler kernels, but needs careful per-channel scaling. On Hopper, FP8 is the dominant choice. Pre-Hopper, INT8 still common.

---

## 2. Where to Quantize

```
Q,K,V projections          ← weights, activations, KV
Attention compute           ← Q,K,V intermediates
FFN gate/up/down projections ← weights, activations
LayerNorm/RMSNorm           ← usually keep FP16/BF16
Output projection / lm_head  ← weights (often kept higher precision)
```

Independent decisions:
- **Weight precision** (INT8 / INT4 / FP8 / FP4).
- **Activation precision** (FP16 / BF16 / FP8 / INT8).
- **KV cache precision** (FP16 / FP8 / INT8 / INT4).
- **Computation precision** (typically dictated by tensor core support).
- **Accumulator precision** (typically FP32 regardless of inputs).

### Common production stacks

| Strategy            | Weight | Act | KV | Use case |
|---------------------|--------|-----|----|----------|
| W8A16               | INT8/FP8 | FP16 | FP16 | Drop-in 2× weight reduction, easy |
| W4A16               | INT4 (GPTQ/AWQ) | FP16 | FP16 | Aggressive, common in vLLM |
| W8A8 (FP8)          | FP8 E4M3 | FP8 | FP8 | Hopper full-FP8, big perf win |
| W4A4 (FP4 / MXFP4)  | FP4 | FP4/FP8 | FP8 | Blackwell, bleeding edge |
| W4A16 KV4           | INT4 | FP16 | INT4 | Capacity-constrained inference |

---

## 3. Symmetric vs Asymmetric Scaling

For INT-N tensor `x`, store `x_q = round(x / s)`. Recover `x ≈ s · x_q`.

**Symmetric**: scale only, zero-centered. `s = max(|x|) / qmax`. Cheap, ideal for weights (zero-mean post-init).

**Asymmetric**: scale + zero-point. `x_q = round(x / s) + z` so that 0 → z. Better for activations with non-zero means (e.g. GELU outputs). One extra subtraction in the dot product.

Per-tensor / per-channel / per-token / per-group:
- **Per-tensor**: one scalar for the whole tensor. Simple, leaks accuracy.
- **Per-channel**: one scale per output channel. Standard for weights.
- **Per-token / per-row**: one scale per token row of activations. Standard for online activation quant.
- **Per-group / per-block**: one scale per group of 32–128 elements. Best for INT4 weights (GPTQ groupsize=128).

---

## 4. Weight Quantization Algorithms

### 4.1 Post-Training Naive (RTN)

`W_q = round(W / s_per_channel)`. Works fine for INT8. Falls apart for INT4 — bigger quantization error, accuracy drops.

### 4.2 GPTQ

Layer-wise greedy: for each weight column, quantize and propagate the error to remaining columns by solving a least-squares update on the Hessian of the layer's input activations. Calibration set ~128 samples. INT4 results within ~1 perplexity point of FP16 for big models.

### 4.3 AWQ (Activation-Aware Weight Quantization)

Observation: a small fraction of weight channels are "salient" — quantizing them hurts much more than others. AWQ scales these channels up before quantization (and scales activations down to compensate), so quantization noise lands on less-important weights. Cheap, no Hessian needed. Often beats GPTQ on chat models.

### 4.4 SmoothQuant

For W8A8: activations have outliers that hurt quantization. SmoothQuant migrates difficulty from activations to weights via a per-channel scaling: `Y = (X · diag(1/s)) · (diag(s) · W)`. Activations become smoother (easier to quantize), weights become slightly harder (still tractable per-channel).

### 4.5 FP8 Calibration

For FP8 weights: per-channel scales chosen so `max(|w|) / s ≤ 448` (E4M3 max). For FP8 activations: per-tensor or per-token scales calibrated on a small dataset. Hopper provides hardware support for `cuBLAS Lt` FP8 matmul with scale fusion.

### 4.6 QLoRA / NF4

Training-time: 4-bit NormalFloat for frozen weights + LoRA adapters in BF16. Used to fine-tune big models on small GPUs. Inference-time: usually merge LoRA back to W4 weights.

---

## 5. KV Cache Quantization

### 5.1 FP8 KV (Hopper)

Drop-in: cast K, V to E4M3 with a per-token (or per-head) scale. Attention kernel reads FP8 K,V, computes in FP16/FP32, writes O in FP16. Marginal accuracy hit (<1% perplexity in most studies).

### 5.2 INT8 KV

Pre-Hopper. Per-token symmetric scales. Some accuracy hit on long contexts.

### 5.3 INT4 KV (KIVI etc.)

Aggressive. Asymmetric per-token. Accuracy degrades on long contexts and reasoning-heavy benchmarks. Use only when capacity is the dominant constraint.

### 5.4 Channel- vs Token-wise

KIVI shows: K is best quantized per-channel (different head dims have different distributions), V per-token (attention smooths the row). This split improves quality at INT4.

---

## 6. Tensor-Core Implications

Throughput per format (H100 dense, single SM):
| Format | TFLOP/s (full chip) | vs FP16 |
|--------|---------------------|---------|
| FP16   | 989                 | 1×      |
| BF16   | 989                 | 1×      |
| FP8    | 1979                | 2×      |
| INT8   | 1979                | 2×      |

Blackwell B200: roughly 2.5× FP8 over H100, plus FP4/MX support.

Quantization gives you (a) capacity reduction (HBM, NVLink, KV), (b) tensor core speedup if the kernels exist for that precision, (c) better roofline because the same FLOPs move fewer bytes.

---

## 7. Kernel Implementation Patterns

### 7.1 Weight-Only INT4 Kernel (W4A16)

Weights stored as INT4 (packed 2/byte) plus per-group scales (FP16). At runtime:
```
1. Dequantize a tile of W into FP16 in shared memory (cheap, group of 128 weights × FP16 scale).
2. FP16 × FP16 matmul on tensor cores.
```
Memory savings: 4× weight bytes. Compute: same as FP16 matmul (since activations are FP16). Wins because decode is BW-bound.

Production examples: Marlin, Machete, GPTQ-Triton, ExLlama.

### 7.2 W8A8 INT8 Kernel

Tensor cores accept INT8 directly. Per-channel weight scales + per-token activation scales fused into the epilogue. cuBLAS Lt supports this natively.

### 7.3 W8A8 FP8 Kernel (E4M3)

cuBLAS Lt FP8 GEMM. Scales applied in the epilogue. Hopper-specific. Twice the throughput of FP16.

### 7.4 Mixed Precision in Attention

In FA-v3 FP8 mode:
```
Q, K, V: FP8 E4M3 (storage)
S = Q·K^T computed in tensor cores → FP32 accumulator
softmax: FP32 (full precision)
P → cast back to FP8
O = P·V in tensor cores → FP32 → FP16 final
```
Critical: softmax in FP32 to preserve dynamic range; everything else FP8.

---

## 8. Calibration

For PTQ (post-training quantization), need representative inputs:
- 128–512 sequences from the target domain.
- Forward pass collects activation statistics (max abs, percentile, distribution).
- Choose scales (per-tensor or per-channel) to minimize quantization error.

GPTQ / AWQ also use calibration data to drive their iterative procedures. Calibration data quality matters — use real production prompts when possible.

---

## 9. Accuracy Preservation Checks

Standard evaluation suite for any quantization:
- Perplexity on WikiText-2 / C4.
- MMLU, GSM8K, HumanEval, MT-Bench (chat), needle-in-haystack (long context).
- Side-by-side generation diffs vs FP16 baseline on a held-out prompt set.

Typical degradation budgets accepted in production:
- W8A16 / FP8: ≤0.05 perplexity / ≤0.5% MMLU.
- W4A16 GPTQ/AWQ: ≤0.2 perplexity / ≤1.5% MMLU.
- W8A8 SmoothQuant: ≤0.1 perplexity / ≤1% MMLU.
- W4A4 / FP4: 1–3% MMLU drop typical; reserve for capacity-constrained deployments.

Long-context tasks (Ruler, NIAH) are more sensitive to KV quantization than short ones.

---

## 10. Outliers Are the Enemy

Why naive INT8 activation quant fails: a few channels have 10× larger magnitudes (the "outlier channels" found by Dettmers et al.). They dominate the per-tensor scale, so the rest of the values lose resolution.

Solutions:
- Per-channel scaling (works for weights, not activations because each row has different channels).
- Mixed precision: keep outlier channels in FP16, rest in INT8 (LLM.int8()).
- Smooth migration (SmoothQuant): scale activations down, weights up.
- Group-wise scaling.

Modern FP8 is more outlier-tolerant because of its exponent — outliers fit in the wider range without dominating a uniform scale.

---

## 11. Quantization-Aware Training (QAT)

If PTQ accuracy is insufficient, QAT inserts fake-quant ops during training so the model learns weights tolerant to quantization. More expensive (full training run). Used for very low precision (W4A4, FP4) where PTQ struggles.

---

## 12. Common Pitfalls

- **Quantizing the LM head**: small, but quality-critical (final logits). Often kept in FP16.
- **Quantizing layer norm parameters**: tiny tensors, no benefit, pure accuracy risk. Don't.
- **Calibration set too small or off-distribution**: mis-calibrated scales → tail-token quality drops invisible in perplexity but obvious in chat.
- **Forgetting per-channel for weights**: per-tensor scales hide most of the quality.
- **Using same scale for K and V in KV cache**: distributions are different; use separate.
- **FP8 without scaling fused in**: bandwidth gains lost to extra scale ops.

---

## 13. Common Interview Questions

**Q: Why quantize?**
A: Three wins simultaneously: (1) shrink weight bytes → fits more KV / serves bigger models / lower memory pressure. (2) shrink interconnect bytes → faster all-reduce, faster KV transfer. (3) double tensor-core throughput on Hopper FP8 / Blackwell FP4. Decode is bandwidth-bound, so byte reduction is throughput.

**Q: Difference between weight-only and weight+activation quant?**
A: Weight-only (W4A16): weights in INT4, activations and compute in FP16. Easier accuracy, dequantizes weights in-kernel, ideal for BW-bound decode. Weight+activation (W8A8): both in INT8 / FP8, full speedup from tensor cores. Higher accuracy risk, especially activation outliers.

**Q: What's GPTQ?**
A: Greedy layerwise weight quantization that propagates errors via the input-Hessian to remaining columns. Calibration set required. Achieves close-to-FP16 quality at INT4 for large models.

**Q: What's AWQ and how does it differ?**
A: Activation-Aware Weight Quantization. Identifies salient weight channels (those whose activations are large) and scales them up before quantization. Quantization noise then lands on less-important channels. Simpler than GPTQ, often equal or better for chat.

**Q: How does SmoothQuant enable W8A8?**
A: Activation outliers make INT8 activations fail. SmoothQuant migrates difficulty from activations to weights via per-channel scaling: `(X/s) @ (sW)` is mathematically the same product, but X becomes smoother (quantizable) at weights' expense (still per-channel quantizable).

**Q: Why is FP8 better than INT8 for activations?**
A: FP8's exponent handles dynamic range — outlier activations fit naturally. INT8's uniform spacing makes outliers dominate the scale. Hopper FP8 also has hardware-fused scaling.

**Q: What's the quality cost of FP8 KV?**
A: Typically <1% MMLU and <0.05 perplexity. Most production stacks default to FP8 KV on Hopper. Long-context retrieval tasks are more sensitive — verify on Ruler / NIAH.

**Q: Why might INT4 KV cache hurt long-context quality more than short-context?**
A: Attention probabilities are derived from precise dot products; small quantization noise compounds across many tokens. With longer context, more tokens contribute small errors to the softmax denominator, drifting the distribution.

**Q: How does per-channel weight quantization interact with the matmul kernel?**
A: The output of matmul `Y = X @ W^T` uses per-channel scales for W (one per output channel = one per row of W^T). Apply the scale in the GEMM epilogue: `Y[m, n] *= scale[n]`. Cheap.

**Q: What's MXFP4 / MX format?**
A: Open Compute Project Microscaling — block of 32 elements share an 8-bit exponent + per-element 4-bit mantissa. Combines FP4's tiny size with block-level dynamic range. Blackwell tensor cores native support.

**Q: When would you use QAT instead of PTQ?**
A: When PTQ accuracy is insufficient and you can afford training time. Typical for FP4 or W4A4. Run a fine-tune with fake-quant ops; weights learn to be quant-friendly.

**Q: How would you debug a quantized model that's giving worse accuracy than expected?**
A: (1) Sanity-check: skip quantization layer-by-layer to find which layer hurts most. (2) Inspect activation stats (outliers? bimodal?). (3) Try per-channel vs per-tensor. (4) Increase group size or move to FP8. (5) Verify calibration data is on-distribution. (6) Check if any layer (LM head, embeddings) shouldn't be quantized.

**Q: Is mixed-precision matmul accumulated in the input precision?**
A: No — accumulators are typically FP32 even for FP8/INT8/FP4 inputs. This avoids overflow during the inner dot product. Only the output cast to lower precision happens at the epilogue.

**Q: How does Marlin (GPTQ kernel) achieve high INT4 throughput?**
A: Carefully fuses dequant + matmul: a tile of INT4 weights is unpacked into shared memory as FP16 with the per-group scale, then standard tensor-core FP16 matmul runs. Memory savings drive the gain (decode is BW-bound).

---

## 14. Further Reading

- Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" (2022).
- Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration" (2023).
- Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models" (2023).
- Dettmers et al., "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale" (2022).
- Liu et al., "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache" (2024).
- NVIDIA Hopper FP8 Transformer Engine docs.
- Open Compute Project MX specification.

---

**Next:** [Parallelism_Strategies](../Part_7_Distributed_Systems_and_Scale/Parallelism_Strategies.md).
**See also:** [GPU_Architecture](../Part_2_Microarchitecture/GPU_Architecture.md), [KV_Cache](../Part_6_Inference_Engines_and_Serving/KV_Cache.md), [CUDA_Optimization](../Part_4_GPU_Kernel_Engineering/CUDA_Optimization.md), [Inference_Frameworks](../Part_6_Inference_Engines_and_Serving/Inference_Frameworks.md).

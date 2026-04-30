# Modern Quantization Frontier — FP4, MXFP, and Beyond

The 2024–2026 quantization story extends well past FP8. NVFP4, MXFP4/6/8, microscaling formats, FP4 inference and FP4 training, smooth migration from FP8, and the kernels that make them practical. This page is the deep dive complementing [Quantization](Quantization.md), focused entirely on the frontier.

**Prerequisites**: [Quantization](Quantization.md), [Blackwell_Architecture](../Part_2_Microarchitecture/Blackwell_Architecture.md), [GPU_Architecture](../Part_2_Microarchitecture/GPU_Architecture.md).

---

## 1. The Modern Number Formats

Beyond FP8 (covered in [Quantization](Quantization.md)):

| Format | Bits | Encoding | Block size | Notes |
|--------|------|----------|------------|-------|
| FP6 E2M3 | 6 | s/2/3 | per-tensor or per-block | Intermediate; rare in production |
| FP6 E3M2 | 6 | s/3/2 | per-tensor or per-block | Wider exponent variant |
| MXFP4 | 4 | s/2/1 | 32 | OCP MX standard, 4-bit FP with shared 8-bit exp |
| MXFP6 | 6 | E2M3 or E3M2 | 32 | 6-bit version with shared exp |
| MXFP8 | 8 | E4M3 or E5M2 | 32 | 8-bit version with shared exp |
| NVFP4 | 4 | s/2/1 | 16 (typical) | NVIDIA's variant; tighter block |
| NF4 | 4 | non-uniform | per-block | 4-bit NormalFloat; used in QLoRA |
| INT4 (GPTQ/AWQ) | 4 | uniform | per-group (~128) | Per-group symmetric/asymmetric scales |
| INT3 | 3 | uniform | per-group | Aggressive; quality cliff |
| INT2 | 2 | uniform + outlier | per-group + LUT | KIVI for KV; SqueezeLLM for weights |

### 1.1 Mathematical Definitions of Formats

To understand the hardware implications, we must look at the exact dequantization math for a weight $w$ inside a block.

**1. INT4 (Symmetric)**:
A group of $K$ elements shares a floating-point scale $S$.
$w \approx S \times q$
where $q \in \{-8, -7, \dots, 7\}$ is a 4-bit two's complement integer.
- **Hardware**: The MAC computes integer multiplication $q_1 \times q_2$, and the scale $S_1 \times S_2$ is applied to the accumulated FP32 sum. Very cheap MAC, but limited dynamic range per group.

**2. NF4 (NormalFloat 4-bit)**:
Used in QLoRA. An information-theoretically optimal format for normally distributed weights.
- The 16 possible 4-bit values $q \in \{0, 1, \dots, 15\}$ map to predefined non-linear quantiles of the standard normal distribution $\mathcal{N}(0, 1)$.
- $w \approx S \times Q\_map[q]$
- **Hardware**: Requires a lookup table (LUT) to convert $q$ to a standard float before multiplication. Highly accurate for weights, but strictly software/LUT-driven. Not suitable for native Tensor Core MACs.

**3. MXFP4 / NVFP4 (Microscaling)**:
A block of $K=32$ elements shares an 8-bit exponent $E_{shared}$. Each element has a 1-bit sign $s$, 2-bit exponent $e$, and 1-bit mantissa $m$ (E2M1 format).
$w \approx (-1)^s \times (1 + 0.5m) \times 2^{e - bias} \times 2^{E_{shared} - bias_{shared}}$
- **Hardware**: The Tensor Core reads $E_{shared}$ once per block. Inside the MAC, it performs a tiny 2-bit exponent addition and a 2x2 bit mantissa multiplication. It naturally supports floating-point dynamic range within the block (values can vary by powers of 2) without the LUT overhead of NF4. This allows native hardware acceleration at 2x the throughput of FP8.

### 1.2 Datapath Precision vs Model Accuracy

The choice of quantization algorithm is strictly bounded by the physical MAC datapath available on the target accelerator. 

**Integer Accumulation (INT8 / INT4)**
When using INT8 or INT4, the hardware MAC natively performs integer multiplication and accumulation.
- **The Accumulator**: An INT32 accumulator is universally used to prevent overflow. A dot product of $N=4096$ elements bounded by $[-127, 127]$ cannot exceed the $\sim 2 \times 10^9$ range of an INT32 register.
- **Algorithmic Constraint**: Because the entire dot-product sum is trapped in the integer domain, the dequantization scale $S$ must be factored out and applied *after* the INT32 sum is computed. This forces the use of **symmetric quantization** ($w \approx S \times q$) where zero exactly maps to zero. Asymmetric quantization (with a zero-point offset) requires complex crossterm compensation during the matmul, drastically reducing throughput.

**Floating-Point Accumulation (FP8 / FP4 / MXFP4)**
When using FP8 or MXFP4, the multiplier generates a floating-point partial product, but it cannot be accumulated at that same low precision.
- **The Accumulator**: An **FP32 accumulator** is mandatory. If an FP4 product were accumulated into an FP8 register, the cascading rounding errors over 4096 additions would completely destroy the signal (catastrophic cancellation). 
- **Algorithmic Freedom**: Because the accumulator is FP32, the hardware naturally handles floating-point sums with massive dynamic range. This allows the OCP MX block exponent ($E_{shared}$) to be applied directly in the MAC pipeline *before* the accumulation stage. The quantization algorithm no longer needs to force all weights into a rigid, uniform integer grid. Instead, it can leverage the non-uniform spacing of floating-point values to better capture the heavy-tailed, bell-curve distributions of neural network weights.

---

## 2. The Microscaling (MX) Standard

OCP MX (Open Compute Project Microscaling) standardized in 2023. The pattern:

```
A "block" of K=32 elements shares:
  - 8-bit exponent (per-block scale)
  - K · (sign + small mantissa) elements
```

For MXFP4: 32 × 4 bits + 8 bits = 136 bits per block = 4.25 bits per element effective.

The block-shared exponent captures the dynamic range; the per-element bits encode within-block variation. Combines floating-point dynamic range with integer-style compactness.

NVIDIA's NVFP4 differs slightly (often K=16, possibly per-tensor scale on top of per-block) but the same family.

### 2.1 Why It Works

In neural network tensors, nearby values tend to have similar magnitudes — they're the output of a small matmul tile or a layer of similar activations. Sharing an exponent across 32 values loses little in practice. It's much better than per-tensor INT4 (one scale for millions of elements) and competitive with per-channel INT4 at smaller metadata cost.

### 2.2 Hardware Support

Blackwell tensor cores support MXFP natively. Loading involves:
1. Read block of 32 packed 4-bit values + 8-bit exponent.
2. Multiply each by 2^exp internally during the matmul.
3. Accumulate in FP32.

No software dequantization step needed for matmul — the tensor core handles it. This is why FP4 is twice as fast as FP8 on Blackwell.

---

## 3. FP4 Inference (W4A4)

The 2025 production goal: **W4A4** end-to-end FP4 inference for matmul-heavy ops.

### 3.1 What Stays Higher Precision

- **Layer norms**: tiny tensors, no benefit, full precision (FP16/BF16).
- **Softmax**: dynamic-range-sensitive, FP32 accumulator.
- **Embeddings**: lookup tables; FP16/INT8 typical.
- **LM head**: small but accuracy-critical; FP16 or FP8.
- **Residual streams**: FP16 between blocks.
- **Optimizer state during fine-tuning**: FP32 master.

Everything else (Q/K/V/O projections, gate/up/down) → FP4.

### 3.2 Calibration Pipeline

1. **Collect activation statistics** on a representative dataset (a few hundred forward passes).
2. **Choose scales** per block: typically the max absolute value in the block sets the exponent.
3. **Validate** on held-out tasks (perplexity, MMLU, long-context).
4. **Tune** outlier handling: clip extreme values or maintain a separate FP16 outlier path.
5. **Optional QAT**: short fine-tuning with fake-FP4 to recover any quality loss.

### 3.3 Quality Numbers

Reported in 2024–2025 papers / NVIDIA blog posts for Llama-3 / DeepSeek class:

- **W4A8 (weights FP4, acts FP8)**: ~0.1pp MMLU drop, near-FP8 quality.
- **W4A4 (full FP4)**: ~0.5-1pp MMLU drop with calibration; ~0.1pp with brief QAT.
- **W4A4 + per-channel scaling on top**: comparable to FP8 in well-tuned setups.

DeepSeek-V3 has been deployed at FP4 in vendor demos (NVIDIA Dynamo) with negligible quality regression — suggests the technique is mature for MoE-class architectures.

### 3.4 KV Cache at FP4

KV-cache quantization down to FP4 is harder because attention is precision-sensitive on long contexts. Production deployments tend to keep KV at FP8 even when weights are FP4. Research (Quest at lower precision, KIVI-style 2-bit KV) is active.

---

## 4. FP4 Training (Late 2025+)

Training at FP4 is harder than inference because gradients have wider distributions and tail behavior matters.

### 4.1 NVIDIA's TE-FP4

Transformer Engine v2 supports FP4 training as of 2025 with:
- **MXFP6 for activations** (more headroom).
- **MXFP4 for weights**.
- **MXFP4 for backward gradients** (with care).
- **FP32 master weights and optimizer state**.

Reported throughput: 2.5× over BF16 training at frontier scale, with quality matching after a brief warm-up phase.

### 4.2 Mixed-Precision Training Schedules

Common pattern:
1. **Warm-up at FP8** for the first N% of training (more stable).
2. **Switch to FP4** once the model has reached a stable training regime.
3. **Some sensitive layers stay FP8/FP16**: usually the LM head and embeddings.

### 4.3 Numerical Stability Tools

- **Stochastic rounding**: random rounding direction per element to avoid bias accumulation.
- **Block-level outlier detection**: occasionally promote a block to FP8 if extreme values appear.
- **Loss scaling** (legacy from FP16): less critical with FP4 due to per-block scales but still helpful at the start.

---

## 5. Quantization Algorithms for FP4

### 5.1 RTN (Round-to-Nearest)

Trivial: per-block, find max abs, set exponent, round to FP4. Works for FP4 weights with minimal quality loss when blocks are small (16-32 elements).

### 5.2 GPTQ-FP4

Same idea as GPTQ INT4 but operating on the FP4 codebook. Per-column greedy quantization with Hessian-based error compensation. Better than RTN on edge layers (LM head, first/last attention).

### 5.3 AWQ-FP4

Activation-aware: identify salient channels, scale them up before FP4 quantization so quantization noise lands on less-important channels. AWQ-style techniques generalize cleanly to FP4.

### 5.4 SmoothQuant for W4A4

Per-channel migration of difficulty from activations to weights via `(X/s) @ (sW)` rebalancing. Makes activations more block-quantization-friendly. Production deployments often layer SmoothQuant on top of AWQ for W4A4.

### 5.5 GPTAQ / Block-Aware Methods

2024–2025 methods that explicitly model block-quantization error during the quantization pass. Better than naive AWQ when block size is small.

---

## 6. KV Cache Beyond FP8

### 6.1 FP4 KV (Experimental)

Some research (Atom, DuQuant) demonstrates FP4 KV with acceptable quality on chat workloads. Long-context retrieval (NIAH, Ruler) still degrades. Not yet a production default.

### 6.2 INT2 KV (KIVI etc.)

KIVI (covered in [Modern_KV_Compression](../Part_6_Inference_Engines_and_Serving/Modern_KV_Compression.md)): K per-channel, V per-token, 2 bits each with asymmetric scales. ~1pp MMLU drop. Niche capacity-constrained deployments.

### 6.3 Tradeoff Map for KV

| KV format | Compression | Quality (chat) | Quality (long ctx) |
|-----------|-------------|-----------------|---------------------|
| FP16      | 1×          | baseline        | baseline            |
| FP8       | 2×          | ~0pp            | ~0-1pp              |
| INT8      | 2×          | ~0pp            | ~1-2pp              |
| INT4 (KIVI 4bit) | 4× | ~0.5pp          | ~2-3pp              |
| FP4 (NVFP4) | 4×        | ~0.5pp          | ~2pp (improving)    |
| INT2 (KIVI) | 8×        | ~1pp            | ~3-5pp              |

Default for new deployments: **FP8 KV on Hopper, FP4/FP8 mix on Blackwell**.

---

## 7. Outlier Handling (Modern Techniques)

Outliers are the universal challenge of low-bit quantization. The 2024–2026 toolbox:

### 7.1 LLM.int8() (Dettmers, 2022) — Foundation

Decompose: 99.9% of channels in INT8, top 0.1% outlier channels in FP16. Per-row mixed precision.

### 7.2 SmoothQuant — Migration

Move difficulty from activations to weights via per-channel scaling.

### 7.3 GPTQ / AWQ — Minimization

Quantize weights in a way that minimizes the impact on output (Hessian-aware or activation-aware).

### 7.4 OmniQuant, AffineQuant, GPTAQ — Learnable Equivalents

Learn small affine transformations per layer that smooth distributions before quantization. End-to-end optimized with a calibration set.

### 7.5 SpinQuant (Meta, 2024)

Apply rotation matrices to weight tensors before quantization — rotations preserve the matmul output but transform distributions to be more uniform / outlier-free. Hardware-friendly Hadamard-style rotations are applied at zero cost during the matmul prologue.

### 7.6 Quest / QQQ / QEFT (2024–2025)

Various refinements that target specific architectures (MoE, MLA) and add per-layer learned compensation. Rapidly improving the W4A4 quality frontier.

---

## 8. Engineering: How To Deploy FP4 Inference

A production rollout plan:

### Step 1: Reproduce FP8 Baseline

Verify quality and throughput at FP8. Establish a measurement harness (perplexity, MMLU, internal eval suite).

### Step 2: Calibrate FP4 Weights

Run AWQ + SmoothQuant + GPTQ-style calibration on a representative dataset. Save quantized weights.

### Step 3: Validate

Run the full eval suite. Measure perplexity, MMLU, code, math, long-context, internal user-experience evals. Compare against FP8.

### Step 4: A/B and Canary

Deploy FP4 to a small fraction of traffic. Compare quality and latency.

### Step 5: Scale

Migrate more traffic. Maintain FP8 fallback for tasks where FP4 lags.

### Step 6: Optimize

Tune kernel configs, block sizes, scale-handling, outlier strategy. Achieve target throughput.

### Step 7: Iterate on QAT

If quality lag persists, run brief quantization-aware fine-tuning to recover. Often <0.5% of pretraining compute.

---

## 9. Frameworks and Tooling

| Tool | What it does | Status |
|------|---------------|--------|
| NVIDIA Transformer Engine v2 | Wraps Linear, manages scales, runs FP8/FP6/FP4 | Production |
| TensorRT Model Optimizer (NVIDIA) | Calibration, GPTQ, AWQ, SpinQuant, FP4 export | Production |
| llmcompressor (Neural Magic) | OSS quantization toolkit, integrates with vLLM | Production |
| GPTQ-for-LLaMA, AutoGPTQ | GPTQ implementations | Mature |
| AutoAWQ | AWQ implementations | Mature |
| BitsAndBytes | INT8, INT4, NF4; integrates with HF | Mature |
| Marlin / Machete (vLLM) | INT4 weight matmul kernels | Production |
| FlashInfer | Quantized attention kernels | Production |
| TileLang | Tile-level kernel DSL with quant support | Emerging |

The mainstream path: **TensorRT Model Optimizer** for calibration, **TRT-LLM** or **vLLM** for serving. Both support W8A8 FP8, W4A16, W4A8, and W4A4 (FP4) on appropriate hardware.

---

## 10. Common Pitfalls

- **Over-aggressive FP4 on small models**: 1B-7B parameter models tolerate FP4 worse than 70B+ because relative quantization noise hits effective capacity harder.
- **Forgetting to re-validate after fine-tuning**: scales calibrated on the base model don't apply to a fine-tuned variant.
- **Per-tensor scales for FP4**: per-block (16-32) is essential; per-tensor crashes accuracy.
- **Mixing block sizes inconsistently**: kernel must agree with the calibration tool's block size.
- **KV at FP4 without long-context validation**: passes chat benchmarks but fails at 64K+ retrieval.
- **Ignoring outliers from chat templates**: unusual tokens at boundaries can produce activation outliers; calibration set must include them.
- **Naive INT4 vs FP4**: FP4 requires hardware support; pre-Blackwell, "FP4" is software-only and pointless. Stick with INT4 (GPTQ/AWQ) on Hopper.

---

## 11. Common Interview Questions

**Q: What's MXFP4 and how does it differ from INT4?**
A: MXFP4 is a 4-bit floating-point format with a per-block (32-element) shared 8-bit exponent. INT4 is uniform integer with per-group (typically 128) scales. MXFP4 captures dynamic range via the per-block exponent → better quality at extreme low bits. Plus Blackwell has hardware support for MXFP4 in tensor cores.

**Q: Can you deploy FP4 inference on Hopper?**
A: Not natively — Hopper tensor cores don't have FP4 hardware. Software FP4 (dequantize to FP8/FP16 inside the kernel) is possible but loses the throughput benefit. FP4 native is Blackwell-and-newer.

**Q: Walk me through deploying a 70B model in FP4.**
A: (1) Establish FP8 baseline. (2) Calibrate FP4 weights via AWQ + SmoothQuant + per-block scales. (3) Validate quality on a comprehensive eval suite — MMLU, code, math, long-context. (4) A/B at small traffic share. (5) Scale up if quality is acceptable. (6) Optionally, brief QAT to close any remaining gap.

**Q: What stays in higher precision when running W4A4?**
A: Layer norms, softmax (FP32 accumulator), embeddings, LM head, residual streams between blocks. Quantizing these tiny tensors saves nothing and risks quality. Only matmul-heavy ops (Q/K/V/O projections, FFN gate/up/down) are FP4.

**Q: What's the typical quality cost of W4A4 vs FP8?**
A: With careful calibration (AWQ/SmoothQuant + per-block FP4 scales), ~0.3-0.8pp MMLU and ~0.05-0.15 perplexity. With brief QAT, often <0.2pp. Long-context tasks more sensitive — verify on RULER / NIAH.

**Q: How does SpinQuant work?**
A: Apply rotation matrices to weight matrices before quantization. The rotation preserves the matmul output (W = R · W' · R^T type identities) but transforms the value distribution to be more uniform and outlier-free. Implemented as Hadamard-style rotations applied at the matmul prologue with negligible cost.

**Q: Why is FP4 KV cache risky for long contexts?**
A: Attention probabilities are derived from precise dot products; quantization noise compounds as more tokens contribute to the softmax denominator. FP4's limited mantissa precision can destabilize attention scores in long contexts. FP8 KV remains the safer default; FP4 KV is research-grade for chat-only workloads.

**Q: How does Transformer Engine v2 handle FP4?**
A: Automatically per-block scaling, calibration via running statistics, and fallback paths for ops that benefit from higher precision. Wraps PyTorch nn.Linear with TE.Linear; users get FP4 with minimal code changes. Manages the precision selection per op based on the layer's role.

**Q: What's the difference between NVFP4 and MXFP4?**
A: Both are 4-bit FP with shared exponent. MXFP4 (OCP standard) uses block size 32. NVFP4 (NVIDIA variant) typically uses block size 16, possibly with an additional per-tensor scale. Tighter block = higher accuracy, more metadata. Hardware supports both.

**Q: When would you NOT quantize to FP4?**
A: Small models (< 7B) where relative quantization noise hits hard; latency-critical workloads where the throughput gain isn't needed; tasks where quality loss is unacceptable (e.g., regulated/safety-critical); when the kernel stack doesn't support it for your model.

**Q: How does FP4 interact with multi-LoRA serving?**
A: LoRA adapters are typically kept in higher precision (FP16 or FP8). The base model weights are FP4; the LoRA delta is added at higher precision and only the matmul gets quantized at runtime. Memory overhead per LoRA is small; throughput follows the base.

**Q: How does FP4 training differ from FP4 inference?**
A: Training requires storing gradients (with wider distributions than activations) and possibly using FP6 for activations and FP4 for weights/grads. Plus FP32 master weights and optimizer state. NVIDIA TE-FP4 training is shipping in 2025 with reported 2.5× over BF16 throughput.

**Q: What's the future beyond FP4?**
A: FP3 / INT3 / INT2 in research; Blackwell doesn't support these natively. Likely we'll see a transition to even more sophisticated mixed-precision schemes (per-layer different formats, learned mixed precision) before lower-than-FP4 becomes mainstream.

---

## 12. Further Reading

- Open Compute Project MX specification (2023).
- NVIDIA Hopper / Blackwell whitepapers.
- Transformer Engine v2 docs.
- NVIDIA TensorRT Model Optimizer documentation.
- "FP8 Formats for Deep Learning" (NVIDIA, 2022).
- "AFPQ: Asymmetric Floating Point Quantization for LLMs" (2024).
- "DuQuant: Distributing Outliers via Dual Transformation Makes Stronger Quantized LLMs" (2024).
- "SpinQuant: LLM Quantization with Learned Rotations" (Meta, 2024).
- "Atom: Low-bit Quantization for Efficient and Accurate LLM Serving" (2024).
- llmcompressor (Neural Magic) GitHub.

---

**Next:** [Disaggregated_Serving_2025](../Part_6_Inference_Engines_and_Serving/Disaggregated_Serving_2025.md), [Modern_Post_Training](../Part_7_Distributed_Systems_and_Scale/Modern_Post_Training.md).
**See also:** [Quantization](Quantization.md), [Blackwell_Architecture](../Part_2_Microarchitecture/Blackwell_Architecture.md), [Modern_KV_Compression](../Part_6_Inference_Engines_and_Serving/Modern_KV_Compression.md).

# L6 — Algorithms and Models: Interview Questions

Consolidated interview Q&A and worked problems from every page in `L6_Algorithms_and_Models/`, moved here to keep topic pages focused on concepts. Each section links back to its source page for full context.

---

## Transformer Internals — The Complete Forward Pass

*From [Transformer_Internals.md](../L6_Algorithms_and_Models/Transformer_Internals.md)*

### Problem 1: KV Cache Size for a New Model

**Question.** A new model has $L=64$ layers, $D=6144$, $H=48$ query heads, $K=6$ KV heads, $d_h=128$, context length $T = 32768$. Compute the KV cache size per request in FP16 and in FP8.

**Solution.** Per token per layer, the KV cache stores $K$ key vectors and $K$ value vectors, each of dimension $d_h$:

$$\text{bytes}_{\text{per-token-per-layer}} = 2 \times K \times d_h \times \text{dtype\_size}$$

In FP16 (2 bytes):

$$= 2 \times 6 \times 128 \times 2 = 3\,072 \text{ bytes/token/layer}$$

Over all layers and context:

$$\text{KV}_{\text{FP16}} = 3\,072 \times 64 \times 32\,768 = 6.44 \times 10^9 \text{ bytes} \approx 6.0 \text{ GB}$$

In FP8 (1 byte):

$$\text{KV}_{\text{FP8}} = 2 \times 6 \times 128 \times 1 \times 64 \times 32\,768 = 3.22 \times 10^9 \text{ bytes} \approx 3.0 \text{ GB}$$

---

### Problem 2: Derive the Parameter Count for Llama-3-8B

**Question.** Llama-3-8B has $L=32$, $D=4096$, $H=32$, $K=8$, $d_h=128$, $d_{\text{ff}}=14336$, $V=128256$. Derive the total parameter count and verify it is approximately 8B.

**Solution.** Per block:

- Q projection: $D \times H d_h = 4096 \times 4096 = 16.78\text{M}$
- K projection: $D \times K d_h = 4096 \times 1024 = 4.19\text{M}$
- V projection: $D \times K d_h = 4096 \times 1024 = 4.19\text{M}$
- Output projection: $H d_h \times D = 4096 \times 4096 = 16.78\text{M}$
- Attention subtotal: $41.94\text{M}$

- Gate projection: $D \times d_{\text{ff}} = 4096 \times 14336 = 58.72\text{M}$
- Up projection: $D \times d_{\text{ff}} = 4096 \times 14336 = 58.72\text{M}$
- Down projection: $d_{\text{ff}} \times D = 14336 \times 4096 = 58.72\text{M}$
- FFN subtotal: $176.16\text{M}$

- RMSNorm gains: $2 \times D = 2 \times 4096 = 0.008\text{M}$

- Per-block total: $41.94\text{M} + 176.16\text{M} + 0.008\text{M} = 218.1\text{M}$

All 32 blocks: $32 \times 218.1\text{M} = 6.979\text{B}$

Embedding: $V \times D = 128256 \times 4096 = 525.3\text{M}$

Final RMSNorm: $D = 0.004\text{M}$

**Total: $6.979\text{B} + 0.525\text{B} + 0.004\text{B} \approx 7.51\text{B}$**

The actual Llama-3-8B reports 8.03B. The discrepancy ($\sim$520M) comes from vocabulary scaling factors, possible untied output head, and rounding in published $d_{\text{ff}}$. If $W_{\text{out}}$ is not tied, add another 525M, giving $\approx 8.04\text{B}$ — matching the published count. Llama-3 does use an untied output projection.

---

### Problem 3: Decode Throughput on H100

**Question.** Estimate the maximum decode throughput (tokens/s) for Llama-3-70B on a single H100 SXM with FP8 weights, batch size 64, and average context length 4096.

**Solution.** Step 1: Memory-bound decode. Each decode step reads all weights once:

$$Q_{\text{weights}} = 70.6\text{B} \times 1 \text{ byte (FP8)} = 70.6 \text{ GB}$$

Step 2: KV cache read per step. Per request:

$$Q_{\text{KV, per-req}} = 2 \times 80 \times 8 \times 128 \times 4096 \times 1 \text{ byte (FP8)} = 0.64 \text{ GB}$$

For 64 requests: $Q_{\text{KV, batch}} = 64 \times 0.64 = 41.0$ GB.

Step 3: Total bytes per step:

$$Q_{\text{total}} = Q_{\text{weights}} + Q_{\text{KV, batch}} = 70.6 + 41.0 = 111.6 \text{ GB}$$

Step 4: Time per step at $\beta = 3.35$ TB/s:

$$t_{\text{step}} = \frac{Q_{\text{total}}}{\beta} = \frac{111.6 \text{ GB}}{3350 \text{ GB/s}} \approx 33.3 \text{ ms}$$

Step 5: Throughput:

$$\text{throughput} = \frac{64 \text{ tokens}}{33.3 \text{ ms}} \approx 1\,920 \text{ tokens/s aggregate}$$

$$\text{per-request} = \frac{1920}{64} = 30 \text{ tokens/s}$$

Reality check: with kernel overhead, attention compute, and sampling, expect 60–80% efficiency. Practical result: $\approx 1\,150$–$1\,540$ tok/s aggregate, 18–24 tok/s per request. This matches published H100 benchmarks for 70B FP8 decode at batch 64.

---

### Problem 4: RoPE Rotation Dimension Analysis

**Question.** For a model with $d_h = 128$ and RoPE base 10000, derive the rotation angles for positions $m = 0, 100, 10000$ at dimension pair $i = 0$ (highest frequency) and $i = 63$ (lowest frequency). Explain why high-frequency dimensions cause problems for length extrapolation.

**Solution.** Base angles:

$$\theta_0 = 10000^{-0/128} = 10000^{0} = 1.0$$

$$\theta_{63} = 10000^{-126/128} = 10000^{-0.984375} \approx 1.037 \times 10^{-4}$$

Rotation angles in radians:

| Position | $\angle_{i=0} = m \cdot \theta_0$ | $\angle_{i=63} = m \cdot \theta_{63}$ |
|---|---|---|
| $m=0$ | 0 | 0 |
| $m=100$ | 100 rad (15.9 full rotations) | 0.01037 rad (0.59 deg) |
| $m=10000$ | 10000 rad (1591 rotations) | 1.037 rad (59.4 deg) |

**Extrapolation problem.** At $i=0$, the rotation completes a full cycle every $2\pi \approx 6.28$ positions. The model has seen positions 0–8191 during training, meaning dimension pair $i=0$ has completed $\approx 1303$ full rotations. At test position 13000, it produces angle 13000 rad — the same as position $13000 \mod 2\pi$, which the model may have seen but at a different absolute position. The periodic aliasing means the model cannot distinguish positions that are $2\pi$ apart in the highest-frequency dimensions, creating ambiguity for out-of-distribution positions.

At $i=63$, the wavelength is $2\pi / \theta_{63} \approx 60\,560$ positions — well beyond the training context. These low-frequency dimensions generalize smoothly. NTK-aware scaling addresses this by slowing down the high-frequency rotation rate: replacing base 10000 with a larger base reduces $\theta_0$, stretching the wavelength of all dimensions.

---

### Problem 5: FLOP Comparison — Dense vs. MoE

**Question.** DeepSeek-V3 has 671B total parameters, 37B active per token, with 61 layers. For a batch of $B=4$, $S=2048$: (a) compute total prefill FLOPs, (b) compare with a hypothetical 671B dense model, (c) estimate prefill time on 8xH100.

**Solution.** (a) Active FLOPs for MoE:

$$\text{FLOPs}_{\text{MoE}} = 2 \times P_{\text{active}} \times B \times S = 2 \times 37\text{B} \times 4 \times 2048 = 606 \text{ TFLOPs}$$

(b) Hypothetical 671B dense:

$$\text{FLOPs}_{\text{dense}} = 2 \times 671\text{B} \times 4 \times 2048 = 10\,983 \text{ TFLOPs}$$

The MoE uses $606 / 10\,983 = 5.5\%$ of the dense FLOPs — an **18x compute savings** for the same total parameter count.

(c) On 8xH100 at 50% MFU:

$$\text{available} = 8 \times 990 \times 0.5 = 3\,960 \text{ TFLOPS}$$

$$t_{\text{MoE}} = \frac{606}{3\,960} \approx 153 \text{ ms}$$

$$t_{\text{dense}} = \frac{10\,983}{3\,960} \approx 2.77 \text{ s}$$

Note: the MoE calculation ignores the all-to-all communication overhead from expert parallelism, which adds $\sim$10–30% wall time depending on network topology. Even with this overhead, the MoE is $\sim$12–15x faster than the equivalent-sized dense model.


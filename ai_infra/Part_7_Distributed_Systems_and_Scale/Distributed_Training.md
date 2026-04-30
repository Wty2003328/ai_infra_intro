# Distributed Training

This page synthesizes how a frontier-scale LLM training run is actually organized: the parallelism cube, the framework stack (PyTorch / FSDP / DeepSpeed / Megatron-LM), checkpointing, fault tolerance, RLHF/post-training pipelines, and the operational reality of a 10K-GPU run.

**Prerequisites**: [[Parallelism_Strategies]], [[Collectives_and_NCCL]], [[Networking_and_Interconnect]], [[Storage_and_Model_Loading]].

---

## 1. The Optimizer Memory Wall

For a transformer with $P$ parameters trained in mixed precision:

| Tensor               | Bytes/param | Why |
|----------------------|--------------|-----|
| FP16/BF16 params     | 2            | forward |
| FP32 master params   | 4            | optimizer accuracy |
| FP16/BF16 gradients  | 2            | backward |
| Adam first moment $m$  | 4            | optimizer state (FP32) |
| Adam second moment $v$ | 4            | optimizer state (FP32) |
| **Total**            | **16**       | per param |

For 70B: $70 \times 10^9 \cdot 16 = 1.12\;\text{TB}$ of optimizer + param state. Doesn't fit on one 80 GB H100 — by $14\times$. Requires sharding.

Plus activations: $O(L \cdot B \cdot S \cdot d)$ — on the order of tens of GB per step before recomputation.

### 1.1 Detailed Memory Breakdown

The total per-parameter memory footprint under mixed-precision AdamW training is:

$$M_{\text{total}} = P \cdot (2 + 4 + 2 + 4 + 4) = 16P \;\text{bytes}$$

where:
- $2P$ bytes for the BF16/FP16 model parameters (used in forward/backward),
- $4P$ bytes for the FP32 master copy of the parameters,
- $2P$ bytes for the BF16/FP16 gradients,
- $4P$ bytes for Adam's first moment $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$ (stored in FP32),
- $4P$ bytes for Adam's second moment $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$ (stored in FP32).

**Worked example — Llama-3-405B on H100 80GB:**

$$M_{\text{total}} = 405 \times 10^9 \times 16 = 6.48\;\text{TB}$$

Minimum GPUs needed (optimizer state alone): $\lceil 6480 / 80 \rceil = 81$ GPUs, but with activations and overhead the real number is much larger.

**Activation memory** (approximate, no recomputation):

$$M_{\text{act}} \approx L \cdot B \cdot S \cdot d \cdot 2 \;\text{bytes}$$

For Llama-3-70B ($L=80$, $B=1$, $S=8192$, $d=8192$, BF16):

$$M_{\text{act}} \approx 80 \times 1 \times 8192 \times 8192 \times 2 \approx 10.7\;\text{GB per microbatch}$$

Plus the attention score matrices scale as $O(L \cdot B \cdot H \cdot S^2)$ without FlashAttention — this is why ZeRO / FSDP exists.

---

## 2. The Parallelism Cube

A real training run combines axes. Example: **Llama-3-70B on 8K H100s**:

```
DP = 1024       (data axis, sharded via ZeRO-3)
TP = 8          (intra-node NVLink)
PP = 1          (not used — ZeRO-3 enough)
Total = 8192 GPUs
```

For dense models in the 70–400B range, TP=8 + ZeRO-3 + DP often suffices. Bigger (1T+) and MoE both push toward PP and EP.

**Llama-3-405B on 16K H100s** (Meta paper):
```
TP = 8
PP = 16
DP = 128
Total = 16384
```
PP = 16 across the depth (126 layers split into 16 stages of ~8 layers each), TP = 8 within node, DP = 128 across the rest with ZeRO-1.

**DeepSeek-V3 (671B MoE)**:
```
TP=8, PP=16, EP=8, DP=8 → 8192 GPUs
```

The right combination depends on (a) model size vs available memory, (b) network topology, and (c) communication-vs-compute balance.

### 2.1 How to Choose the Parallelism Layout

The total GPU count is:

$$N_{\text{GPU}} = \text{TP} \times \text{PP} \times \text{DP} \times \text{EP}$$

**Rules of thumb:**

1. **TP first**: set TP to the number of GPUs connected via NVLink (typically 8 per node). TP requires the highest bandwidth (all-reduce every layer), so it must stay intra-node.
2. **PP next**: if the model is too large to fit in one node even after TP sharding, add PP stages. Each stage holds $\lfloor L / \text{PP} \rfloor$ layers. Pipeline bubble fraction is approximately $\frac{\text{PP} - 1}{\text{PP} + M - 1}$ where $M$ is the number of microbatches.
3. **DP last**: all remaining GPUs form the data-parallel dimension. With ZeRO-3/FSDP, each DP rank holds only $\frac{1}{\text{DP}}$ of the optimizer state.
4. **EP for MoE**: shard experts across the EP dimension; each EP rank holds $\frac{E}{\text{EP}}$ experts.

---

## 3. Framework Stack

### 3.1 PyTorch DDP (DistributedDataParallel)

Vanilla DP with bucketed gradient all-reduce overlapped with backward. Each rank holds full model. Use only if model fits per rank.

### 3.2 FSDP (Fully Sharded Data Parallel)

PyTorch-native ZeRO-3. Shards params, grads, and optimizer state across DP ranks. Wraps modules; gathers params just-in-time per forward.

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD,
             device_id=torch.cuda.current_device())
```

**Per-rank memory under FSDP (ZeRO-3):**

$$M_{\text{rank}} = \frac{16P}{N_{\text{DP}}} + M_{\text{act}}$$

where $M_{\text{act}}$ is the activation memory for one microbatch. For 70B with 1024 DP ranks: $\frac{1.12\;\text{TB}}{1024} \approx 1.1\;\text{GB}$ of optimizer state per rank — trivially fits.

### 3.3 PyTorch DTensor + 2D Parallel

Newer PyTorch native primitive: tensors carry placement (Shard / Replicate) on a `DeviceMesh`. Combines TP and DP cleanly. Used by Llama-3 reference training.

### 3.4 Megatron-LM (NVIDIA)

The reference for TP + PP + SP + interleaved 1F1B. C++/CUDA kernels for TP-aware ops, fused softmax/cross-entropy. Used by NVIDIA, MosaicML's MPT, many internal labs.

### 3.5 DeepSpeed

ZeRO origin. Adds ZeRO-Inference, ZeRO-Offload (CPU/NVMe), pipeline parallel. Often combined with Megatron as "Megatron-DeepSpeed."

### 3.6 NeMo Framework

NVIDIA's higher-level wrapper around Megatron-LM with config-driven training and inference recipes.

### 3.7 JAX / TPU stacks

Pax, MaxText. Same parallelism axes (`jax.sharding.Mesh`), tied to TPU XLA. Different ecosystem; concepts identical.

---

## 4. Mixed Precision

Standard recipe (BF16):
- Master weights in FP32.
- Forward and backward in BF16 (no loss scaling needed thanks to BF16's wide range).
- Gradient all-reduce in BF16.
- Optimizer step in FP32.

FP16 alternative requires loss scaling (multiply loss by $\sim 2^{16}$, divide gradients before update) to avoid underflow. BF16 has won; FP16 is legacy.

### 4.1 Loss Scaling for FP16

When using FP16, small gradients underflow to zero. Loss scaling compensates:

$$\hat{\mathcal{L}} = s \cdot \mathcal{L}$$

where $s$ is the scale factor (e.g., $s = 2^{16} = 65536$). Gradients are computed as $s \cdot \nabla_\theta \mathcal{L}$, then unscaled before the optimizer step:

$$\nabla_\theta \mathcal{L}_{\text{unscaled}} = \frac{1}{s} \cdot \nabla_\theta \hat{\mathcal{L}}$$

Dynamic loss scaling adjusts $s$ automatically: increase $s$ by $2\times$ every $N$ steps (e.g., 2000), and halve $s$ whenever an overflow/NaN is detected in the gradients.

FP8 training: still emerging. Hopper's Transformer Engine offers automatic FP8 with per-tensor scaling for selected ops (matmuls). FP8 forward/backward, FP32 master weights and optimizer. Reported $1.5$–$2\times$ speedup over BF16 with no quality loss when tuned correctly.

---

## 5. Optimizer Choices

- **AdamW** — default for LLMs. $m$/$v$ momentum, decoupled weight decay. 8 bytes/param of state.
- **Lion** — sign-based; 4 bytes/param state. Sometimes converges faster but less robust.
- **Shampoo** / **Distributed Shampoo** — second-order; better convergence per step but heavy to implement and shard.
- **Adafactor** — factorized second moment; saves memory (~half AdamW state). Used in JAX/T5 ecosystem.
- **CAME** — improved Adafactor with confidence interval, used in some recent runs.

8-bit Adam (Dettmers): store $m$, $v$ in INT8 with per-tensor scales. Cuts optimizer state $\sim 4\times$. Drop-in via `bitsandbytes`.

### 5.1 AdamW Update Rule

The AdamW update at step $t$ is:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

Bias-corrected estimates:

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

Parameter update with decoupled weight decay $\lambda$:

$$\theta_t = \theta_{t-1} - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1} \right)$$

Typical hyperparameters: $\beta_1 = 0.9$, $\beta_2 = 0.95$, $\epsilon = 10^{-8}$, $\lambda = 0.1$.

**Memory per parameter**: $m_t$ (4 bytes FP32) + $v_t$ (4 bytes FP32) = 8 bytes/param of optimizer state, plus 4 bytes for FP32 master weights = 12 bytes/param overhead beyond the BF16 model copy.

---

## 6. Activation Recomputation (Gradient Checkpointing)

Trade compute for memory. Don't save activations during forward; recompute during backward.

- **Full** recomputation: ~33% extra FLOPs in backward; biggest savings.
- **Selective** (Korthikanti et al.): recompute attention only — almost free with FA, ~5% extra compute, large memory drop.

Essential for long-context training. PyTorch FSDP calls it `checkpoint_wrapper`.

### 6.1 Memory Savings Analysis

Without recomputation, activation memory for an $L$-layer transformer is:

$$M_{\text{act}} = L \cdot (2 \cdot B \cdot S \cdot d + 2 \cdot B \cdot H \cdot S^2) \cdot \text{bytes\_per\_element}$$

With **full recomputation** (checkpoint every layer), only the input to each layer is saved:

$$M_{\text{act,recomp}} = L \cdot (2 \cdot B \cdot S \cdot d) \cdot \text{bytes\_per\_element}$$

This eliminates the $O(B \cdot H \cdot S^2)$ attention score storage, but requires rerunning the forward pass during backward — adding approximately $\frac{1}{3}$ extra FLOPs (since backward is $\sim 2\times$ forward, recomputing forward once adds $\sim 33\%$).

With **selective recomputation**, only the attention scores are recomputed (the cheapest part when using FlashAttention), saving the $O(S^2)$ memory at only $\sim 5\%$ extra compute.

---

## 7. Data Loading & Tokenization

Throughput at 10K GPUs is huge: ~50 GB/s of tokens needed if you don't want data-load to bottleneck.

### 7.1 Dataset Format

- **Mosaic Streaming / WebDataset** — shard files (`.tar`, `.mds`) on object store; stream over network.
- **Tokenized binary blobs** — pre-tokenize, store as `uint16` / `uint32` arrays. Mmap-friendly.
- **Memory-mapped** — `numpy.memmap` over big binary arrays; OS handles paging.

### 7.2 Sampling

- Pre-shuffle and pack: concatenate documents to fill `seq_len`, separate with EOS token. Avoids padding waste.
- Curriculum / mix: oracle proportions of code, web, math, books — mixing handled in the dataloader.
- Reproducible RNG for ablations.

### 7.3 Throughput

$$\text{tokens\_per\_step} = B_{\text{global}} \cdot S = (\text{microbatch} \times \text{grad\_accum} \times \text{DP}) \cdot S$$

For Llama-3-405B: $\sim 16\text{M tokens/step}$ at $S = 8192$, batch $\sim 2048$. At 1 step / 5 s $= 3.2\;\text{M tok/s} \approx 12\;\text{GB/s}$ of data. Object store + parallel readers handle this.

---

## 8. Checkpointing

Frequent enough to recover from failures, infrequent enough not to dominate cost.

### 8.1 What to save

- Model weights (full, not sharded — or sharded with metadata).
- Optimizer states.
- LR scheduler state.
- RNG state.
- Dataloader cursor / consumed-token count.
- Training step number.

Total: roughly $P \cdot 16$ bytes/param. For 405B: $\sim 6.5\;\text{TB}$.

### 8.2 Strategies

- **Synchronous, full**: stop training, save, resume. Simple. Costly: $6.5\;\text{TB} / (1\;\text{GB/s/GPU} \cdot \text{DP})$ per save.
- **Asynchronous (DCP)**: PyTorch Distributed Checkpoint flushes shards to a staging buffer; background workers write to storage. Training continues.
- **Tier-based**: optimizer state to fast NVMe; weights to S3 every $N$ saves.
- **In-memory replication** (Argonne / OpenAI talk about): write to neighbor's RAM, recover from there if a node dies.

### 8.3 Cadence

Common cadence: `min(once per hour, every 1000 steps)`. Plus end-of-epoch milestones.

### 8.4 Restore

- Verify all expected files / shards exist.
- Match TP/PP layout (or convert with a re-shard utility).
- Rehydrate dataloader cursor for reproducibility.

### 8.5 Checkpoint Size Estimation

The total checkpoint size is:

$$C_{\text{total}} = P \cdot (2 + 4 + 2 + 4 + 4) = 16P \;\text{bytes}$$

For common model sizes:

| Model | Params | Checkpoint Size |
|-------|--------|-----------------|
| 7B    | $7 \times 10^9$   | $\sim 112\;\text{GB}$ |
| 70B   | $7 \times 10^{10}$ | $\sim 1.12\;\text{TB}$ |
| 405B  | $4.05 \times 10^{11}$ | $\sim 6.48\;\text{TB}$ |

Time to save (synchronous): $T_{\text{save}} = \frac{C_{\text{total}}}{\text{aggregate\_write\_BW}}$. With 128 DP ranks writing in parallel at 1 GB/s each: $T_{\text{save}} = \frac{6.48\;\text{TB}}{128\;\text{GB/s}} \approx 50\;\text{s}$.

---

## 9. Fault Tolerance

At 10K GPUs:
- Hardware failure rate: ~1 GPU/day (industry surveys).
- Network blip rate: orders of magnitude higher.
- Software / OOM crash: depends.

Strategies:

### 9.1 Restart-on-Failure

Detect failure via NCCL timeout or rank death. Stop the job. Restart from last checkpoint. Adequate for runs <1 day; painful for multi-week pretraining.

### 9.2 Spare Capacity & Hot Replacement

Allocate $N + \text{spare}$ GPUs; on failure, kill the affected rank, swap in a spare, restart from a recent in-memory or NVMe checkpoint. Used by big labs.

### 9.3 Health Checks

Per-step assertions: loss not NaN, no rank slow by $2\times$ median (straggler), all NCCL BW above threshold. Trigger checkpoint + investigate on red flag.

### 9.4 Reproducibility

Same seed + same data order + same hardware → same loss curve up to noise from non-determinism (CUDA matmul tile order). For frontier runs, ~1% deviation from a previous run is normal.

### 9.5 Cost of Failures

For a multi-week run on 10K GPUs, the expected wasted compute from failures is significant. If the mean time between failures (MTBF) is $T_f$ and checkpoint interval is $T_c$:

$$\text{wasted\_compute\_per\_failure} = \frac{T_c}{2} \cdot N_{\text{GPU}} \cdot \text{cost\_per\_GPU\_hour}$$

$$\text{failures\_per\_day} \approx \frac{N_{\text{GPU}}}{\text{MTBF\_per\_GPU}}$$

With 10K GPUs and MTBF of 10K GPU-hours ($\approx 1$ failure/day), checkpointing every hour wastes on average 30 minutes of 10K-GPU compute per failure — roughly $\$5{,}000$–$\$15{,}000$/failure depending on cloud rates.

---

## 10. Hyperparameters at Scale

### 10.1 Learning Rate Schedule

- Linear warmup (1–4K steps).
- Cosine decay to 10% of peak.
- Optional final linear cooldown to $\sim 0$ in last few %.

The cosine schedule is:

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{\pi \cdot t}{T}\right)\right)$$

where $\eta_{\max}$ is the peak LR, $\eta_{\min} = 0.1 \cdot \eta_{\max}$, and $T$ is the total training steps.

Peak LR scales roughly with $\sqrt{B_{\text{global}}}$ up to a critical batch size.

### 10.2 Batch Size

- "Critical batch size" (McCandlish et al.) $= B_{\text{crit}}$ where adding more tokens stops helping per-step convergence.

$$B_{\text{crit}} = \frac{B_{\text{noise}}}{1 - B_{\text{noise}} / B_{\text{max}}}$$

- Frontier runs: global batch 2–16M tokens (very large).
- Effective batch $= \text{microbatch} \cdot \text{grad\_accum} \cdot \text{DP}$.

### 10.3 Weight Decay

Typically $\lambda = 0.1$ with decoupled (AdamW) — applied to weights but NOT to bias/norm/embeddings.

### 10.4 Gradient Clipping

Norm clip to $\sim 1.0$:

$$g_{\text{clipped}} = g \cdot \frac{\text{max\_norm}}{\max(\|g\|_2,\; \text{max\_norm})}$$

Prevents loss spikes from rare large gradients.

---

## 11. Post-Training Pipelines

After pretraining, modern LLMs go through:

1. **SFT (Supervised Fine-Tuning)** — instruction-following on curated data. Standard cross-entropy. Single-model setup, much smaller scale (often 1 node).
2. **DPO / IPO / KTO / SimPO** — preference learning on `(prompt, chosen, rejected)` triples. Two model copies in memory (policy + reference) — pushes memory pressure.
3. **RLHF / GRPO** — full RL loop with reward model, value model, policy. **4 models** in memory simultaneously (policy, reference, reward, critic). Uses online sampling — needs **inference engine inside training loop**.
4. **Constitutional / RLAIF** — variation: AI generates the reward signal.

### 11.1 RLHF Engineering Reality

A single iteration:
1. Sample $N$ completions from policy (use vLLM/TRT-LLM under the hood).
2. Score with reward model.
3. Compute advantage with critic.
4. Update policy + critic (PPO-style).

Sampling step is the bottleneck → need fast inference within the training process. Modern stacks (TRL, OpenRLHF, NeMo-Aligner, veRL) hand sampling off to a dedicated inference cluster connected via shared memory or RDMA, then weights sync back periodically.

### 11.2 DPO Loss Function

The DPO (Direct Preference Optimization) loss avoids the RL loop by directly optimizing preferences:

$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \beta \log \frac{\pi_\theta(y_l | x)}{\pi_{\text{ref}}(y_l | x)} \right) \right]$$

where $y_w$ is the chosen response, $y_l$ is the rejected response, $\pi_\theta$ is the policy model, $\pi_{\text{ref}}$ is the frozen reference model, and $\beta$ controls the deviation from the reference.

**Memory**: two full model copies ($\pi_\theta$ and $\pi_{\text{ref}}$) $= 2 \times 16P$ bytes for a 70B model $\approx 2.24\;\text{TB}$.

### 11.3 GRPO (DeepSeek)

Drops critic. Uses $K$ samples per prompt, computes advantages within the group → simpler, less memory.

The GRPO advantage for sample $i$ in a group of $K$ responses to prompt $x$:

$$A_i = \frac{r_i - \text{mean}(r_1, \ldots, r_K)}{\text{std}(r_1, \ldots, r_K)}$$

where $r_i$ is the reward for the $i$-th sample. This normalized within-group advantage replaces the learned critic, eliminating one model from memory.

### 11.4 Reasoning-Train Scale-Up (2025+)

Long-CoT RL (R1, o1-style) with very long generations (10K+ tokens), high sample count, multi-stage curriculum. Throughput of the inference engine becomes the rate-limit.

---

## 12. Common Pitfalls

- **Loss spikes** — usually a bad data shard or rare gradient explosion. Gradient clipping + gradient logging (per-layer norms) catches them.
- **NaN losses** — overflow in FP16 (use BF16) or division by zero in LayerNorm; check epsilon and scaling.
- **Diverging on resume** — stale RNG state, or different number of replicas changed effective batch size.
- **Throughput regression after software upgrade** — NCCL version bump, kernel auto-tuner choosing different paths. Lock versions.
- **Storage I/O bottleneck** — checkpoint loading at startup taking forever; use streaming load.

---

## 13. Common Interview Questions

**Q: For a 70B model on 64 H100s, design the parallelism layout for training.**
A: TP=8 within each node (NVLink), DP=8 across nodes with FSDP/ZeRO-3 (or ZeRO-1 if memory allows — saves comm). PP not needed at this scale. Sequence-parallel + selective activation recomputation. Mixed precision BF16 + FP32 master. Memory check: $\frac{70 \times 10^9 \times 16}{8 \text{ DP ranks}} \approx 140\;\text{GB/rank}$ for optimizer state — with TP=8 sharding params within each node, this fits within 80 GB per GPU after accounting for activations and gradient memory.

**Q: Why does ZeRO-3 not always replace TP?**
A: ZeRO-3's all-gather of params per layer is bandwidth-bound and still expensive at large model size; TP cuts per-GPU compute and memory simultaneously without per-layer all-gather. ZeRO-3 is a DP-axis trick; TP shards inside layers. They compose. Quantitatively: ZeRO-3 all-gather for one layer of a 70B model moves $\frac{P_{\text{layer}}}{N_{\text{DP}}} \times N_{\text{DP}} = P_{\text{layer}}$ parameters — every layer, every step. TP only does $O(B \cdot S \cdot d / \text{TP})$-sized all-reduces, which are typically smaller.

**Q: What is the activation memory of training a transformer and how do you reduce it?**
A: Roughly $O(L \cdot B \cdot S \cdot d)$ plus attention which is $O(L \cdot B \cdot H \cdot S^2)$ if you don't use FA. Reduce via (a) FlashAttention (drops the $S^2$ term), (b) activation recomputation (full or selective), (c) sequence parallel (shards the activations along $S$).

**Q: Walk me through a single training step.**
A: (1) Dataloader yields a microbatch. (2) Forward pass through TP/PP stages, with FSDP gathering params just-in-time. (3) Loss + backward, gradients computed and reduce-scattered. (4) Gradient accumulation across microbatches. (5) Optimizer step on sharded state. (6) (Periodically) checkpoint. (7) (Periodically) eval.

**Q: Why is BF16 preferred over FP16?**
A: Same range as FP32 (8-bit exponent), so no loss scaling needed. FP16's tighter range (5-bit exponent) requires careful loss scaling and overflow handling. BF16 just works on Ampere+. The key: BF16 has exponent range $[2^{-126}, 2^{127}]$ (same as FP32), while FP16 maxes out at $65504$ — large gradient values overflow easily.

**Q: How big are checkpoints and how often do you save?**
A: Roughly $P \times 16$ bytes (mixed precision + master + Adam state). For 70B: $\sim 1.1\;\text{TB}$. Cadence balances time-to-save vs failure recovery loss; common is hourly + per-$N$-steps. Async distributed checkpointing minimizes pause.

**Q: What's the difference between SFT, DPO, and RLHF?**
A: SFT: standard cross-entropy on (prompt, response) pairs — $\mathcal{L} = -\sum_t \log p_\theta(y_t | y_{<t}, x)$. DPO: preference loss on (chosen, rejected) using a reference model — no RL loop. RLHF: full PPO with reward + critic models, online sampling — much heavier engineering. Memory requirement: SFT = 1 model, DPO = 2 models, RLHF = 4 models.

**Q: Why is RLHF training infrastructure significantly harder than SFT?**
A: Four models in memory (policy, ref, reward, critic) and online sampling that requires running an inference engine inside the training loop. Sampling throughput often dominates wall-clock. Plus PPO-specific stability issues (KL divergence penalty tuning, advantage normalization, value function clipping).

**Q: How does FSDP compare to ZeRO-3 in DeepSpeed?**
A: Conceptually identical (sharded params/grads/optstate). FSDP is PyTorch-native, integrates with TorchCompile, better TP composition via DTensor. ZeRO-3 has a longer history, more knobs, slightly different memory schedule.

**Q: What's pipeline-parallel "1F1B" scheduling?**
A: Once the pipeline is filled, each stage alternates between forward of the next microbatch and backward of the oldest microbatch — one-forward-one-backward. Bounds activation memory to $O(P)$ microbatches per stage rather than $O(M)$, where $M$ is the total number of microbatches. The pipeline bubble fraction is:

$$\text{bubble} = \frac{P - 1}{P + M - 1}$$

where $P$ is the number of pipeline stages.

**Q: How do you handle a straggler GPU during training?**
A: Detect via per-rank step time variance. Investigate ECC errors, thermal throttling, NIC link issues. Exclude the node and use a spare; restart from last checkpoint. At scale, automated detection is essential.

**Q: How do you keep activation memory bounded with PP and long sequences?**
A: Sequence parallel + recomputation + restrict in-flight microbatches. Even with these, careful planning of `microbatch * seq_len` per stage is needed.

**Q: What's "context parallelism" and when do you need it?**
A: Sharding the sequence dim across ranks (ring attention). Needed when $S$ is so large ($\geq 100\text{K}$) that one rank's attention activations blow memory. Adds a comm pass per layer to rotate KV; otherwise transparent. Memory per rank scales as $O(S / \text{CP})$ instead of $O(S)$.

**Q: How does gradient accumulation interact with the optimizer step?**
A: Accumulate gradients over $K$ microbatches without updating. Effective batch $= \text{microbatch} \cdot K \cdot \text{DP}$. Grad sync (all-reduce) only happens at the $K$-th microbatch. Useful when memory limits microbatch but you want a large effective batch for stability.

**Q: Derive the FLOPs per training step for a transformer.**
A: For a dense transformer with $P$ parameters and a batch of $T$ tokens, each token passes through approximately $2P$ multiply-add operations in the forward pass (one multiply + one add per parameter). The backward pass is $\sim 2\times$ the forward. Total:

$$\text{FLOPs}_{\text{step}} \approx 6 \cdot P \cdot T$$

where $T = B_{\text{global}} \cdot S$. For Llama-3-70B with $T = 16\text{M}$ tokens:

$$\text{FLOPs}_{\text{step}} \approx 6 \times 70 \times 10^9 \times 16 \times 10^6 = 6.72 \times 10^{18}\;\text{FLOPs}$$

---

## 14. Further Reading

- Llama-3 paper (Meta, 2024) — frontier-scale training details.
- DeepSeek-V3 technical report (2024).
- Smith et al., "Megatron-Turing NLG 530B" (2022).
- "Reducing Activation Recomputation" (Korthikanti et al., 2022).
- DeepSpeed ZeRO papers (-1, -2, -3, ZeRO-Infinity).
- TRL, OpenRLHF, veRL repositories for RL infra.
- Chinchilla scaling laws paper (Hoffmann et al., 2022).

---

**Next:** [[Training_Optimization]].
**See also:** [[Parallelism_Strategies]], [[Collectives_and_NCCL]], [[Storage_and_Model_Loading]], [[Quantization]].

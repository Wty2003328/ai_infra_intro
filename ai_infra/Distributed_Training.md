# Distributed Training

This page synthesizes how a frontier-scale LLM training run is actually organized: the parallelism cube, the framework stack (PyTorch / FSDP / DeepSpeed / Megatron-LM), checkpointing, fault tolerance, RLHF/post-training pipelines, and the operational reality of a 10K-GPU run.

**Prerequisites**: [Parallelism_Strategies](Parallelism_Strategies.md), [Collectives_and_NCCL](Collectives_and_NCCL.md), [Networking_and_Interconnect](Networking_and_Interconnect.md), [Storage_and_Model_Loading](Storage_and_Model_Loading.md).

---

## 1. The Optimizer Memory Wall

For a transformer with `P` parameters trained in mixed precision:

| Tensor               | Bytes/param | Why |
|----------------------|--------------|-----|
| FP16/BF16 params     | 2            | forward |
| FP32 master params   | 4            | optimizer accuracy |
| FP16/BF16 gradients  | 2            | backward |
| Adam first moment m  | 4            | optimizer state (FP32) |
| Adam second moment v | 4            | optimizer state (FP32) |
| **Total**            | **16**       | per param |

For 70B: `70e9 · 16 = 1.12 TB` of optimizer + param state. Doesn't fit on one 80 GB H100 — by 14×. Requires sharding.

Plus activations: O(num_layers · B · S · d) ≈ tens of GB per step before recomputation.

This is why ZeRO / FSDP exists.

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

FP16 alternative requires loss scaling (multiply loss by ~2^16, divide gradients before update) to avoid underflow. BF16 has won; FP16 is legacy.

FP8 training: still emerging. Hopper's Transformer Engine offers automatic FP8 with per-tensor scaling for selected ops (matmuls). FP8 forward/backward, FP32 master weights and optimizer. Reported 1.5–2× speedup over BF16 with no quality loss when tuned correctly.

---

## 5. Optimizer Choices

- **AdamW** — default for LLMs. m/v momentum, decoupled weight decay. 8 bytes/param of state.
- **Lion** — sign-based; 4 bytes/param state. Sometimes converges faster but less robust.
- **Shampoo** / **Distributed Shampoo** — second-order; better convergence per step but heavy to implement and shard.
- **Adafactor** — factorized second moment; saves memory (~half AdamW state). Used in JAX/T5 ecosystem.
- **CAME** — improved Adafactor with confidence interval, used in some recent runs.

8-bit Adam (Dettmers): store m, v in INT8 with per-tensor scales. Cuts optimizer state ~4×. Drop-in via `bitsandbytes`.

---

## 6. Activation Recomputation (Gradient Checkpointing)

Trade compute for memory. Don't save activations during forward; recompute during backward.

- **Full** recomputation: ~33% extra FLOPs in backward; biggest savings.
- **Selective** (Korthikanti et al.): recompute attention only — almost free with FA, ~5% extra compute, large memory drop.

Essential for long-context training. PyTorch FSDP calls it `checkpoint_wrapper`.

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

```
tokens_per_step = batch_size_global · seq_len
                = (microbatch · grad_accum · DP) · seq_len
```

For Llama-3-405B: ~16M tokens/step at SeqLen 8K, batch ~2K. At 1 step / 5 s = 3.2 M tok/s = ~12 GB/s of data. Object store + parallel readers handle this.

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

Total: roughly model_size · 16 bytes/param. For 405B: 6.5 TB.

### 8.2 Strategies

- **Synchronous, full**: stop training, save, resume. Simple. Costly: 6.5 TB / (1 GB/s/GPU · DP) per save.
- **Asynchronous (DCP)**: PyTorch Distributed Checkpoint flushes shards to a staging buffer; background workers write to storage. Training continues.
- **Tier-based**: optimizer state to fast NVMe; weights to S3 every N saves.
- **In-memory replication** (Argonne / OpenAI talk about): write to neighbor's RAM, recover from there if a node dies.

### 8.3 Cadence

Common cadence: `min(once per hour, every 1000 steps)`. Plus end-of-epoch milestones.

### 8.4 Restore

- Verify all expected files / shards exist.
- Match TP/PP layout (or convert with a re-shard utility).
- Rehydrate dataloader cursor for reproducibility.

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

Allocate `N + spare` GPUs; on failure, kill the affected rank, swap in a spare, restart from a recent in-memory or NVMe checkpoint. Used by big labs.

### 9.3 Health Checks

Per-step assertions: loss not NaN, no rank slow by 2× median (straggler), all NCCL BW above threshold. Trigger checkpoint + investigate on red flag.

### 9.4 Reproducibility

Same seed + same data order + same hardware → same loss curve up to noise from non-determinism (CUDA matmul tile order). For frontier runs, ~1% deviation from a previous run is normal.

---

## 10. Hyperparameters at Scale

### 10.1 Learning Rate Schedule

- Linear warmup (1–4K steps).
- Cosine decay to 10% of peak.
- Optional final linear cooldown to ~0 in last few %.

Peak LR scales roughly with sqrt(batch_size) up to a critical batch size.

### 10.2 Batch Size

- "Critical batch size" (McCandlish et al.) = where adding more tokens stops helping per-step convergence.
- Frontier runs: global batch 2–16M tokens (very large).
- Effective batch = microbatch · grad_accum · DP.

### 10.3 Weight Decay

Typically 0.1 with decoupled (AdamW) — applied to weights but NOT to bias/norm/embeddings.

### 10.4 Gradient Clipping

Norm clip to ~1.0. Prevents loss spikes from rare large gradients.

---

## 11. Post-Training Pipelines

After pretraining, modern LLMs go through:

1. **SFT (Supervised Fine-Tuning)** — instruction-following on curated data. Standard cross-entropy. Single-model setup, much smaller scale (often 1 node).
2. **DPO / IPO / KTO / SimPO** — preference learning on `(prompt, chosen, rejected)` triples. Two model copies in memory (policy + reference) — pushes memory pressure.
3. **RLHF / GRPO** — full RL loop with reward model, value model, policy. **4 models** in memory simultaneously (policy, reference, reward, critic). Uses online sampling — needs **inference engine inside training loop**.
4. **Constitutional / RLAIF** — variation: AI generates the reward signal.

### 11.1 RLHF Engineering Reality

A single iteration:
1. Sample N completions from policy (use vLLM/TRT-LLM under the hood).
2. Score with reward model.
3. Compute advantage with critic.
4. Update policy + critic (PPO-style).

Sampling step is the bottleneck → need fast inference within the training process. Modern stacks (TRL, OpenRLHF, NeMo-Aligner, veRL) hand sampling off to a dedicated inference cluster connected via shared memory or RDMA, then weights sync back periodically.

### 11.2 GRPO (DeepSeek)

Drops critic. Uses K samples per prompt, computes advantages within the group → simpler, less memory.

### 11.3 Reasoning-Train Scale-Up (2025+)

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
A: TP=8 within each node (NVLink), DP=8 across nodes with FSDP/ZeRO-3 (or ZeRO-1 if memory allows — saves comm). PP not needed at this scale. Sequence-parallel + selective activation recomputation. Mixed precision BF16 + FP32 master.

**Q: Why does ZeRO-3 not always replace TP?**
A: ZeRO-3's all-gather of params per layer is bandwidth-bound and still expensive at large model size; TP cuts per-GPU compute and memory simultaneously without per-layer all-gather. ZeRO-3 is a DP-axis trick; TP shards inside layers. They compose.

**Q: What is the activation memory of training a transformer and how do you reduce it?**
A: Roughly `O(L · B · S · d)` plus attention which is `O(L · B · num_heads · S²)` if you don't use FA. Reduce via (a) FlashAttention (drops the S² term), (b) activation recomputation (full or selective), (c) sequence parallel (shards the activations along S).

**Q: Walk me through a single training step.**
A: (1) Dataloader yields a microbatch. (2) Forward pass through TP/PP stages, with FSDP gathering params just-in-time. (3) Loss + backward, gradients computed and reduce-scattered. (4) Gradient accumulation across microbatches. (5) Optimizer step on sharded state. (6) (Periodically) checkpoint. (7) (Periodically) eval.

**Q: Why is BF16 preferred over FP16?**
A: Same range as FP32 (8-bit exponent), so no loss scaling needed. FP16's tighter range (5-bit exponent) requires careful loss scaling and overflow handling. BF16 just works on Ampere+.

**Q: How big are checkpoints and how often do you save?**
A: Roughly `params · 16 bytes` (mixed precision + master + Adam state). For 70B: ~1.1 TB. Cadence balances time-to-save vs failure recovery loss; common is hourly + per-N-steps. Async distributed checkpointing minimizes pause.

**Q: What's the difference between SFT, DPO, and RLHF?**
A: SFT: standard cross-entropy on (prompt, response) pairs. DPO: preference loss on (chosen, rejected) using a reference model — no RL loop. RLHF: full PPO with reward + critic models, online sampling — much heavier engineering.

**Q: Why is RLHF training infrastructure significantly harder than SFT?**
A: Four models in memory (policy, ref, reward, critic) and online sampling that requires running an inference engine inside the training loop. Sampling throughput often dominates wall-clock. Plus PPO-specific stability issues.

**Q: How does FSDP compare to ZeRO-3 in DeepSpeed?**
A: Conceptually identical (sharded params/grads/optstate). FSDP is PyTorch-native, integrates with TorchCompile, better TP composition via DTensor. ZeRO-3 has a longer history, more knobs, slightly different memory schedule.

**Q: What's pipeline-parallel "1F1B" scheduling?**
A: Once the pipeline is filled, each stage alternates between forward of the next microbatch and backward of the oldest microbatch — one-forward-one-backward. Bounds activation memory to O(P) microbatches per stage rather than O(M).

**Q: How do you handle a straggler GPU during training?**
A: Detect via per-rank step time variance. Investigate ECC errors, thermal throttling, NIC link issues. Exclude the node and use a spare; restart from last checkpoint. At scale, automated detection is essential.

**Q: How do you keep activation memory bounded with PP and long sequences?**
A: Sequence parallel + recomputation + restrict in-flight microbatches. Even with these, careful planning of `microbatch · seq_len` per stage is needed.

**Q: What's "context parallelism" and when do you need it?**
A: Sharding the sequence dim across ranks (ring attention). Needed when seq_len is so large (≥100K) that one rank's attention activations blow memory. Adds a comm pass per layer to rotate KV; otherwise transparent.

**Q: How does gradient accumulation interact with the optimizer step?**
A: Accumulate gradients over `K` microbatches without updating. Effective batch = `microbatch · K · DP`. Grad sync (all-reduce) only happens at the K-th microbatch. Useful when memory limits microbatch but you want a large effective batch for stability.

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

**Next:** [Training_Optimization](Training_Optimization.md).
**See also:** [Parallelism_Strategies](Parallelism_Strategies.md), [Collectives_and_NCCL](Collectives_and_NCCL.md), [Storage_and_Model_Loading](Storage_and_Model_Loading.md), [Quantization](Quantization.md).

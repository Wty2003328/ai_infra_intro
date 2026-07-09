# L7 — Training Stack

> The layer where model architectures (L6) become **trained weights**. Training a trillion-parameter model across 100,000 GPUs requires solving a distributed systems problem: how to partition the model, communicate gradients, handle failures, and optimize every step from mixed-precision arithmetic to post-training alignment. L7 covers the full training stack from parallelism primitives through post-training RL.

---

## Scope of this layer

L7 covers **distributed training and post-training**.

1. **Parallelism strategies** — DP, TP, PP, EP, CP, SP. Communication volume math per scheme. 3D parallelism. Hybrid strategies for trillion-param models.
2. **Collectives and NCCL** — AllReduce algorithms (ring, tree, recursive halving-doubling). Bandwidth modeling. NCCL internals. Cross-rack collectives.
3. **Distributed training** — FSDP, ZeRO-1/2/3 derivation (memory math). Checkpointing, fault tolerance, elastic training, MoE-specific issues.
4. **Training optimization** — Mixed precision (AMP, BF16, FP8 training). Activation checkpointing, gradient accumulation, gradient compression, Transformer Engine.
5. **Modern post-training** — DPO, IPO, KTO, SimPO, ORPO derivations. PPO vs GRPO. Online RL infra (TRL, OpenRLHF, veRL, NeMo-Aligner). Distillation from reasoning teachers.
6. **Reasoning models** — o1/o3/R1 architecture and post-training. Long-CoT RL. Test-time compute scaling. Thinking budgets. Tool-use during reasoning.

L7 explicitly does **not** cover model architecture (→ L6), serving systems (→ L8), or kernel programming (→ L5).

Note: Reasoning_Models was moved here from L6 (it is primarily a post-training story).

---

## Pages (read in order)

1. [Parallelism_Strategies](01_Parallelism_Strategies.md) — DP, TP, PP, EP, CP, SP with communication volume math. Foundational for all distributed training.
2. [Collectives_and_NCCL](02_Collectives_and_NCCL.md) — AllReduce algorithms, bandwidth modeling, NCCL internals.
3. [Distributed_Training](03_Distributed_Training.md) — FSDP, ZeRO derivation, checkpointing, fault tolerance.
4. [Training_Optimization](04_Training_Optimization.md) — Mixed precision, activation checkpointing, gradient accumulation, Transformer Engine.
5. [Modern_Post_Training](05_Modern_Post_Training.md) — DPO, GRPO, RLHF, distillation.
6. [Reasoning_Models](06_Reasoning_Models.md) — o1/o3/R1, long-CoT RL, test-time compute.

---

## What L7 hands off to L8 and above

| L7 fact | Higher-layer consequence |
|---|---|
| ZeRO-3 partitions all states across ranks | Training memory scales as O(1/rank); same math applies to serving |
| TP communication volume = 2*batch*seq*hidden/tube | TP within NVL72 rack avoids cross-rack traffic |
| FP8 training with delayed scaling | Same quantized weights flow directly to FP8 inference (L8) |
| DPO eliminates the reward model | Simpler alignment pipeline; but PPO still wins on some tasks |
| GRPO group sampling reduces variance | Lower-cost RL for reasoning training |
| Reasoning models use 5-10x more tokens at inference | Serving cost increases proportionally (L8) |
| Checkpoint -> safetensors for serving | Storage format bridges training and serving (L4, L8) |

---

## Reading order

Strict order: Parallelism -> Collectives -> Distributed_Training -> Training_Optimization -> Modern_Post_Training -> Reasoning_Models.

After this layer, ascend to [L8 -- Inference & Serving](../L8_Inference_and_Serving/00_Index.md).

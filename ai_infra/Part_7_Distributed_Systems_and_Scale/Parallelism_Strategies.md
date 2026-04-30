# Parallelism Strategies

When a model or its activations don't fit on one GPU — or when one GPU is too slow — we shard. There are five primary axes (DP, TP, PP, EP, CP/SP) plus several refinements (FSDP, ZeRO, gradient/optimizer sharding). This page covers each axis, when to combine them, the math of communication cost, and the engineering tradeoffs.

**Prerequisites**: [GPU_Architecture](../Part_2_Microarchitecture/GPU_Architecture.md), [Networking_and_Interconnect](../Part_3_Hardware_Systems_and_Interconnects/Networking_and_Interconnect.md), [Memory_Hierarchy_and_Roofline](../Part_2_Microarchitecture/Memory_Hierarchy_and_Roofline.md).

---

## 1. The Five Axes

| Axis | Splits | Comm pattern | Where it lives |
|------|--------|--------------|----------------|
| DP — Data Parallel | batch across replicas | all-reduce of grads | Always (almost) |
| TP — Tensor Parallel | hidden dim within a layer | all-reduce / reduce-scatter / all-gather per layer | Inside fast NVLink domain |
| PP — Pipeline Parallel | layers into stages | point-to-point activations between stages | Across slower interconnect |
| EP — Expert Parallel (MoE) | experts across GPUs | all-to-all per MoE layer | Within high-BW fabric |
| CP / SP — Context / Sequence Parallel | sequence dim within a layer | small all-gather / reduce | For very long context |

You can (and at scale, must) combine multiple axes. A typical 70B training: DP × TP × PP. A typical MoE inference: TP × EP. Long-context training adds CP.

---

## 2. Data Parallelism (DP)

Each replica holds a full copy of the model. The batch is split. Each replica processes its shard, computes loss/grads, and replicas all-reduce gradients.

```
forward: independent
backward: independent
gradient sync: all-reduce(g) across DP group
optimizer step: identical on each replica (so weights stay in sync)
```

### 2.1 Comm cost

All-reduce on parameter gradients each step:
```
bytes_per_step = 2 · params · b      # ring all-reduce moves ~2 · payload
```
For 70B at FP16, that's `2 · 70 GB · 2 = 280 GB` per all-reduce. At 100 GB/s NVLink BW per GPU, ~2.8 s — completely unacceptable.

That's why **FSDP / ZeRO** exist: they shard the optimizer state and parameters, so the all-reduce becomes a smaller reduce-scatter + all-gather overlapped with compute.

### 2.2 Sharded DP (FSDP / ZeRO)

ZeRO has three stages:
- **ZeRO-1**: shard optimizer states across DP. Per-rank memory ≈ params + grads + (opt_state / DP).
- **ZeRO-2**: shard optimizer + gradients. Reduce-scatter at backward.
- **ZeRO-3 / FSDP**: also shard parameters. Gather params just-in-time per layer.

ZeRO-3 communication per step is the same total bytes as DP all-reduce — but split across `forward all-gather + backward all-gather + reduce-scatter` and **overlapped with compute**. Effective communication cost approaches the network BW limit rather than blocking.

---

## 3. Tensor Parallelism (TP)

Split a layer's weight matrix across N GPUs. The classic Megatron-LM layout:

### 3.1 Column-Parallel + Row-Parallel Pair

For an MLP `Y = GeLU(XA)`, split A column-wise (each rank gets `d × d_ffn/N`):
```
forward:  Y_i = GeLU(X · A_i)      # no comm (X is replicated)
```
Then for the down projection `Z = YB`, split B row-wise (each rank gets `d_ffn/N × d`):
```
forward:  Z = sum_i (Y_i · B_i)    # all-reduce across N
```

So an MLP block is **one all-reduce per forward**, **one all-reduce per backward**.

### 3.2 Attention TP

Split heads across ranks: each rank owns `H/N` Q,K,V heads. After attention, an output projection (row-parallel) needs an all-reduce.

So a transformer layer with TP=N has **2 all-reduces per forward** (attention out, MLP out) and **2 per backward**.

### 3.3 Comm Cost

Each all-reduce moves `2 · B · S · d · b` bytes. For Llama-70B at TP=8, B=64, S=4096, d=8192, FP16:
```
2 · 64 · 4096 · 8192 · 2 = 8.6 GB per all-reduce
```
At 900 GB/s NVLink ring BW, ~10 ms each. With ~4 all-reduces (2 fwd + 2 bwd) × 80 layers = 3.2 s/step just for TP comm. **TP must run on intra-NVL-domain links**, never across slow fabric.

### 3.4 Sequence Parallel + TP (Megatron-SP)

LayerNorm and dropout don't need TP comm but their inputs/outputs duplicate `B·S·d` activations across ranks. **Sequence-parallel** splits the sequence dim during norm/dropout, reducing activation memory. Comm: replaces all-reduce with reduce-scatter + all-gather (same bytes, slightly different schedule), allowing overlap with the matmul.

---

## 4. Pipeline Parallelism (PP)

Split layers into stages. Each stage on a different GPU. Microbatches flow through:

```
GPU 0 (layers 0..19)  →  GPU 1 (20..39)  →  GPU 2 (40..59)  →  GPU 3 (60..79)
```

Activations move stage-to-stage; gradients move backward.

### 4.1 Bubble Problem

Naive sequential PP wastes time at the start (filling pipeline) and end (draining):
```
bubble fraction ≈ (P - 1) / (M + P - 1)
```
where P = stages, M = microbatches. With P=4, M=4: bubble = 3/7 ≈ 43%. With M=16: 3/19 ≈ 16%.

### 4.2 Schedules

- **GPipe / 1F1B (one-forward-one-backward)**: minimum bubble, simple.
- **Interleaved 1F1B (Megatron)**: each stage owns multiple non-contiguous "virtual" sub-stages → smaller bubble but more comm.
- **Zero-bubble PP** (research): split backward into B(activation) + B(weight) and reorder; bubble disappears under ideal conditions.

### 4.3 PP Comm

Point-to-point activation send/recv per microbatch, per layer boundary. Typically smaller per-step bytes than TP. Suits slower interconnects (cross-node IB, even Ethernet for big jobs).

### 4.4 Memory

Each stage holds 1/P of the layers but must keep activations for in-flight microbatches. Activation memory grows with M. With activation recomputation, this drops dramatically.

---

## 5. Expert Parallelism (EP) for MoE

Mixture-of-Experts layers have E experts (e.g. 256 for DeepSeek-V3). Each token is routed to top-k=1..2 experts. Storing all experts on every GPU is wasteful → shard experts across GPUs.

### 5.1 Layout

EP=N: each rank holds `E/N` experts. Tokens that route to a remote expert must be **dispatched** across the network.

### 5.2 All-to-All

After routing decisions:
```
all-to-all dispatch: rank_i sends tokens routed to rank_j's experts
                     → rank_j computes its experts on those tokens
all-to-all combine:  send results back to original ranks
```
Two all-to-alls per MoE layer. Bytes: roughly `B·S·d · 2k/N · b` per all-to-all (only top-k tokens per rank cross over).

### 5.3 Imbalance

Some experts get more tokens than others ("hot experts"). Solutions:
- **Auxiliary loss** during training to encourage even routing.
- **Expert capacity factor** — drop excess tokens to a backup expert.
- **Load-balanced placement** — replicate hot experts across ranks.

### 5.4 EP + TP

In large MoE (DeepSeek-V3 has 671B params), each expert is itself sharded with TP. So you stack: `DP × TP × EP × PP`. NVL72 makes this practical because 72 GPUs share NVLink fabric.

---

## 6. Context / Sequence Parallelism (CP / SP)

For training on sequences of 100K+ tokens, attention activations explode (O(N²) for naive attention, O(N·d) even with FA). CP splits the **sequence dimension** of activations across ranks.

### 6.1 Ring Attention

Each rank owns a chunk of Q, K, V along the seq dim. Rounds:
```
round r: each rank sends its KV chunk to next neighbor; computes partial attention against received chunk; rotates
after N rounds: each rank has accumulated full attention for its Q chunk
```
Equivalent to FlashAttention chunking but distributed across GPUs. Ring all-gather of KV.

### 6.2 Comm Cost

Per layer, total bytes ≈ `2·N·d·b` (one full pass of K,V around the ring). For CP=8, S=128K, d=8192, FP16: ~2 GB per layer per rank. Adds up but enables sequences impossible on one GPU.

### 6.3 Use Cases

- Long-context fine-tuning (1M context).
- DPO/RLHF on long conversations.
- Training inference-style chunked-prefill.

---

## 7. Choosing the Right Mix

### 7.1 Inference

Decode is BW-bound: TP boosts BW (each rank reads its shard from its own HBM in parallel) and reduces weight memory. Inference TP ≤ NVL domain (8–72). PP is rare in inference (adds latency); EP is required for MoE.

Typical inference layouts:
- **Single 8-GPU node, 70B**: TP=8 (NVLink). KV pool spread across the 8.
- **Dense 405B across 16 GPUs**: TP=8 × PP=2 (interleaved 1F1B).
- **MoE 671B (DeepSeek-V3)**: TP=8 × EP=8 or higher.

### 7.2 Training

DP for batch scaling. TP within nodes (NVLink). PP across nodes (IB). Mix DP × TP × PP × ZeRO.

Example: **64 nodes × 8 H100 = 512 GPUs training 70B**:
```
TP=8 (intra-node NVLink)
PP=8 (across 8 node-groups)
DP=8 (across the remaining DP groups), with ZeRO-1 or ZeRO-3
```
Check: 8 · 8 · 8 = 512 ✓.

### 7.3 The 3D / 4D / 5D Cube

Each axis has its own communication group. Each GPU is identified by `(dp_rank, tp_rank, pp_rank, ...)`. Megatron-LM, Megatron-DeepSpeed, and PyTorch DTensor implement these directly.

---

## 8. Communication Cost Cheat Sheet

For a transformer block with hidden d, sequence S, batch B, FP16 b=2:

```
TP all-reduce per layer: 2 · B · S · d · b · 2  (forward + backward)
PP send per microbatch:  B_micro · S · d · b
DP all-reduce (vanilla): 2 · params · b
ZeRO-3 reduce-scatter:   params · b
ZeRO-3 all-gather:       params · b   (per layer fetch)
EP all-to-all:           2 · B · S · d · b · k / N  (per MoE layer)
CP ring KV:              2 · S · d · b              (per layer per rank)
```

Compare against `(NVLink BW, IB BW)` to find the bottleneck.

---

## 9. Activation Recomputation

Trade compute for memory: don't save certain layer activations during forward; recompute them during backward.

- **Selective recomputation**: only recompute attention (the biggest activation). Free in compute (already FA). Savings huge.
- **Full recomputation**: recompute every block. Saves the most memory; ~33% extra FLOPs in backward.

Critical for long context and ZeRO-3 because activation memory dominates after sharding params.

---

## 10. Common Pitfalls

- **TP over slow interconnect**: kills throughput. Always intra-NVL-domain.
- **Mixing TP=N with attention head count not divisible by N**: requires padding or different splits.
- **Forgetting that PP serializes time**: latency = sum of stage times even with full pipelining; OK for training (lots of microbatches), bad for low-latency inference.
- **MoE without load balance**: one rank does 5× more work than others, idle waiting elsewhere.
- **EP across nodes**: all-to-all on IB is slow; benefit only when EP fits in NVL fabric.
- **CP without ring overlap**: blocking comm dominates.

---

## 11. Common Interview Questions

**Q: What's the difference between DP and FSDP?**
A: DP keeps a full model copy on each replica and all-reduces gradients. FSDP shards parameters, gradients, and optimizer state across replicas; gathers params just-in-time per layer. FSDP enables training models that don't fit on one GPU and overlaps comm with compute, but adds bookkeeping.

**Q: When do you use TP vs PP?**
A: TP within a fast NVLink domain (intra-node, ≤8 GPUs typically) because every layer requires high-BW all-reduces. PP across slower links (cross-node IB) because it only sends activations between stage boundaries — much less BW pressure.

**Q: Walk me through the all-reduces in a TP=8 transformer layer.**
A: One all-reduce after the attention output projection (concat-by-heads then row-parallel down), one all-reduce after the MLP down-projection (column-parallel up + row-parallel down). Megatron's column→row pair pattern. Two all-reduces forward, two backward per layer.

**Q: What's the pipeline bubble and how do you reduce it?**
A: Bubble = idle time at pipeline start/end where stages have nothing to do. Fraction ≈ (P-1)/(M+P-1). Reduce by increasing microbatch count, using interleaved 1F1B, or zero-bubble PP. Typical target bubble <10%.

**Q: How does ZeRO-3 communication cost compare to vanilla DP?**
A: Same total bytes asymptotically (one all-reduce ≡ reduce-scatter + all-gather). But ZeRO-3 spreads comm across forward (gather params for layer i), backward (gather + reduce-scatter), so it overlaps with compute. Vanilla DP is one big blocking all-reduce at end of step.

**Q: What's all-to-all and when do you need it?**
A: Each rank sends a unique chunk to every other rank, simultaneously. Used in MoE (token dispatch/combine) and CP. Bandwidth-limited by the slowest link in the comm group; sensitive to network topology (rail-optimized helps).

**Q: For a 70B model on 8× H100, single node — what parallelism layout?**
A: TP=8 with NVLink. All weights sharded across 8 GPUs; intra-layer all-reduces over NVLink. KV cache also sharded by heads (since GQA H_kv=8 conveniently splits 1 per rank for Llama-3-70B).

**Q: For the same 70B model across 4 nodes (32 H100s) — training — what layout?**
A: TP=8 within each node (NVLink), then DP=4 across nodes with FSDP/ZeRO-3. PP not needed because 70B fits with TP=8 + ZeRO-3 sharding.

**Q: Why does sequence-parallel reduce activation memory?**
A: Ops like LayerNorm and dropout were duplicated across TP ranks (they don't have weights to shard). SP shards the sequence dim during these ops, splitting activation `B·S·d` by `S/N`. The all-reduce after MLP becomes reduce-scatter + all-gather (same total bytes but with less duplicated activation).

**Q: How does MoE expert parallelism work?**
A: Each rank holds a subset of experts. Routing assigns each token to top-k experts. All-to-all-1 sends each token to the rank holding its assigned experts; experts compute; all-to-all-2 sends results back. Two all-to-alls per MoE layer.

**Q: What's ring attention?**
A: Sequence-parallel attention. Each rank owns a chunk of Q,K,V. KV chunks rotate around the ring; each rank accumulates partial softmax statistics from chunks it has seen, like FlashAttention but distributed. Enables training on huge contexts.

**Q: Why is PP almost never used in inference?**
A: It adds latency proportional to the number of stages even without bubbles, and decode is single-token-per-step (microbatch=1) so no pipelining benefit. TP gives BW directly with no latency penalty — much better for inference.

**Q: How do you decide PP=8 vs PP=16 in a training run?**
A: More stages = smaller per-stage memory and lower per-stage FLOPs but bigger bubble unless microbatch count grows. Pick smallest PP that fits memory and gives microbatch count M ≥ ~4·P for reasonable bubble (<20%).

**Q: What does "5D parallelism" mean?**
A: Combining DP × TP × PP × EP × CP. Each axis is its own comm group; each GPU has 5 ranks identifying its position. Used in frontier-scale training (e.g. DeepSeek-V3, Llama-3-405B).

---

## 12. Further Reading

- Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism" (2019).
- Korthikanti et al., "Reducing Activation Recomputation in Large Transformer Models" (2022).
- Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" (2020).
- Liu et al., "Ring Attention with Blockwise Transformers for Near-Infinite Context" (2023).
- Fedus et al., "Switch Transformer" (2022).
- DeepSpeed and Megatron-LM source repositories.
- PyTorch DTensor and FSDP docs.

---

**Next:** [Collectives_and_NCCL](Collectives_and_NCCL.md).
**See also:** [Networking_and_Interconnect](../Part_3_Hardware_Systems_and_Interconnects/Networking_and_Interconnect.md), [Distributed_Training](Distributed_Training.md), [GPU_Architecture](../Part_2_Microarchitecture/GPU_Architecture.md).

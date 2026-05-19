# Distributed Training — FSDP, ZeRO, and Fault Tolerance at Scale

> **Layer:** L7.
> **Prerequisites:** [Parallelism_Strategies](Parallelism_Strategies.md), [Collectives_and_NCCL](Collectives_and_NCCL.md), [Modern_MoE](../L6_Algorithms_and_Models/Modern_MoE.md).
> **Hands off to:** [Training_Optimization](Training_Optimization.md), [Storage_and_Model_Loading](../L4_Systems_and_Interconnects/Storage_and_Model_Loading.md).

---

## 0. Why this page exists

A 70-billion-parameter model trained in mixed-precision BF16 with AdamW requires 16 bytes per parameter of runtime state — 1.12 TB across parameters, gradients, and optimizer moments. A single H100 has 80 GB of HBM. The gap is 14:1, and it grows quadratically with model size. Without sharding, training any model larger than ~4 B parameters is impossible on a single GPU.

ZeRO (Zero Redundancy Optimizer) and its PyTorch-native incarnation FSDP (Fully Sharded Data Parallel) close this gap by partitioning optimizer states, gradients, and parameters across the data-parallel dimension. This page derives the exact memory math for each ZeRO stage, explains FSDP's mechanics, covers checkpointing and fault tolerance at 10K-GPU scale, and addresses the distributed training issues specific to Mixture-of-Experts models.

Four invariants govern this page:

1. **Mixed-precision AdamW consumes 16 bytes per parameter** — 2 (BF16 params) + 2 (BF16 grads) + 4 (FP32 master) + 4 (Adam $m$) + 4 (Adam $v$).
2. **ZeRO-3 memory per GPU is $\frac{16P}{N_{\mathrm{DP}}}$ plus activations** — the divisor is the data-parallel degree, not the total GPU count.
3. **Checkpoint size equals $16P$ bytes** — the same 16 bytes/param, serialized to storage every $T_c$ seconds.
4. **Mean time between failures at 10K GPUs is ~1 day** — fault tolerance is not optional; it is a first-class design constraint.

---

## 1. The optimizer memory wall

### 1.1 Per-parameter cost breakdown

For a transformer with $P$ parameters trained in mixed-precision BF16 with AdamW:

| Tensor | Bytes / param | Dtype | Role |
|---|---|---|---|
| Model parameters | 2 | BF16 | Forward and backward pass |
| Gradients | 2 | BF16 | Backward pass accumulation |
| Master parameters | 4 | FP32 | Optimizer accuracy |
| Adam first moment $m_t$ | 4 | FP32 | Exponential moving average of gradients |
| Adam second moment $v_t$ | 4 | FP32 | Exponential moving average of squared gradients |
| **Total** | **16** | | |

The Adam update rules at step $t$:

$$m_t = \beta_1 \, m_{t-1} + (1 - \beta_1) \, g_t$$

$$v_t = \beta_2 \, v_{t-1} + (1 - \beta_2) \, g_t^2$$

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

$$\theta_t = \theta_{t-1} - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \, \theta_{t-1} \right)$$

Typical values: $\beta_1 = 0.9$, $\beta_2 = 0.95$, $\epsilon = 10^{-8}$, $\lambda = 0.1$.

**Worked problem 1 — minimum GPU count.** How many H100 80 GB GPUs are needed to hold the training state of Llama-3-70B?

$$M_{\text{total}} = 70 \times 10^9 \times 16 \;\text{B} = 1{,}120 \;\text{GB}$$

$$N_{\min} = \left\lceil \frac{1{,}120}{80} \right\rceil = 14 \;\text{GPUs}$$

This is the absolute floor — it ignores activation memory, CUDA context overhead (~2 GB), and temporary buffers. In practice, 14 GPUs cannot train 70B; the real minimum with ZeRO-3 is 16–32 GPUs.

### 1.2 Activation memory

Activation memory dominates when sequence lengths are long. Without recomputation, a single transformer layer stores:

$$M_{\text{act,layer}} = \underbrace{2 \, B \, S \, d}_{\text{hidden activations}} + \underbrace{2 \, B \, H \, S^2}_{\text{attention scores (no FA)}} \;\;\text{bytes (BF16)}$$

With FlashAttention, the $S^2$ term is eliminated (see [FlashAttention_Deep_Dive](../L5_Kernels_and_Programming/FlashAttention_Deep_Dive.md)). The remaining activation memory across $L$ layers:

$$M_{\text{act}} \approx L \cdot 2 \, B \, S \, d \;\;\text{bytes}$$

With full activation recomputation (checkpointing every layer), only layer inputs are stored:

$$M_{\text{act,recomp}} \approx L \cdot 2 \, B \, S \, d \;\;\text{bytes (same formula, but per-checkpoint-region, not per-layer)}$$

The distinction is subtle: full recomputation stores one tensor per layer boundary instead of all intermediate tensors. With $c$ checkpoint segments, activation memory drops to $M_{\text{act}} / c$ at the cost of recomputing forward within each segment during backward.

**Worked problem 2 — activation memory for 70B.** Llama-3-70B has $L = 80$ layers, $d = 8192$, sequence length $S = 8192$, microbatch $B = 1$:

$$M_{\text{act}} \approx 80 \times 2 \times 1 \times 8192 \times 8192 = 10.7 \;\text{GB per microbatch}$$

With activation recomputation ($c = 80$ segments, one per layer):

$$M_{\text{act,recomp}} \approx \frac{10.7}{80} \times 2 \approx 0.27 \;\text{GB (negligible)}$$

The factor of 2 accounts for storing both the segment input and the locally active sub-computation. This is why activation recomputation is always enabled for pretraining.

---

## 2. ZeRO — deriving the memory math

ZeRO eliminates the redundancy of standard data parallelism, where every GPU holds a full copy of all states. It proceeds in three stages, each removing one category of redundancy.

### 2.1 Baseline: standard data parallelism (DDP)

In vanilla `DistributedDataParallel`, each of $N_{\mathrm{DP}}$ ranks holds a complete copy of everything:

$$M_{\mathrm{DDP}} = 16P \;\;\text{bytes per GPU}$$

There is no memory benefit from adding GPUs — the 16$P$ cost is replicated $N_{\mathrm{DP}}$ times. The only benefit is throughput: each rank processes a different microbatch and gradients are all-reduced.

### 2.2 ZeRO stage 1 — partition optimizer states

**Insight:** optimizer states ($m_t$, $v_t$, master FP32 params) account for $\frac{12}{16} = 75\%$ of memory. Each DP rank only needs its own shard to perform the optimizer step.

Partition the $12P$ bytes of optimizer state across $N_{\mathrm{DP}}$ ranks. Each rank owns $\frac{12P}{N_{\mathrm{DP}}}$ bytes of optimizer state plus the full $4P$ bytes of model parameters and gradients:

$$M_{\mathrm{ZeRO\text{-}1}} = 4P + \frac{12P}{N_{\mathrm{DP}}}$$

The factor of $4P$ comes from BF16 parameters ($2P$) plus BF16 gradients ($2P$). The optimizer update is local to each rank's shard; after the step, the updated BF16 parameter shard is communicated if needed.

**Communication:** one reduce-scatter of gradients per step (same total bytes as all-reduce, but each rank receives only $1/N_{\mathrm{DP}}$ of the result). No change to the forward pass.

### 2.3 ZeRO stage 2 — partition optimizer states and gradients

**Insight:** after the backward pass computes gradients, each rank only needs the gradient shard that corresponds to its optimizer state shard.

Use reduce-scatter instead of all-reduce for gradients. Each rank receives and stores only $\frac{2P}{N_{\mathrm{DP}}}$ bytes of gradient:

$$M_{\mathrm{ZeRO\text{-}2}} = 2P + \frac{14P}{N_{\mathrm{DP}}}$$

The $2P$ is the full BF16 parameter copy. The $\frac{14P}{N_{\mathrm{DP}}}$ is $\frac{2P}{N_{\mathrm{DP}}}$ (gradient shard) + $\frac{12P}{N_{\mathrm{DP}}}$ (optimizer shard).

**Communication:** reduce-scatter of gradients during backward (same volume as all-reduce, but each rank stores $1/N_{\mathrm{DP}}$ of the result). Still no change to forward pass.

### 2.4 ZeRO stage 3 / FSDP — partition everything

**Insight:** parameters themselves are redundant across DP ranks. Each rank only needs the full parameter set during the forward and backward passes of its own layers — parameters can be gathered just-in-time.

Partition the $2P$ bytes of BF16 parameters across $N_{\mathrm{DP}}$ ranks. Before each layer's forward, gather the full parameter set via all-gather; after the layer's backward, discard the gathered parameters:

$$M_{\mathrm{ZeRO\text{-}3}} = \frac{16P}{N_{\mathrm{DP}}}$$

This is the ideal: memory scales as $\mathcal{O}(1/N_{\mathrm{DP}})$.

**Communication:** two all-gathers per layer (forward + backward) plus one reduce-scatter per layer (gradient). Total communication volume per step equals the DDP all-reduce volume — but it is spread across the forward and backward passes and overlapped with compute.

### 2.5 Memory comparison table

| Stage | Params | Gradients | Optimizer state | Memory per GPU | Comm per step |
|---|---|---|---|---|---|
| DDP | Full ($2P$) | Full ($2P$) | Full ($12P$) | $16P$ | All-reduce ($2P$) |
| ZeRO-1 | Full ($2P$) | Full ($2P$) | Sharded ($\frac{12P}{N}$) | $4P + \frac{12P}{N}$ | Reduce-scatter ($2P$) |
| ZeRO-2 | Full ($2P$) | Sharded ($\frac{2P}{N}$) | Sharded ($\frac{12P}{N}$) | $2P + \frac{14P}{N}$ | Reduce-scatter ($2P$) |
| ZeRO-3 | Sharded ($\frac{2P}{N}$) | Sharded ($\frac{2P}{N}$) | Sharded ($\frac{12P}{N}$) | $\frac{16P}{N}$ | All-gather + reduce-scatter ($2 \times 2P$) |

Where $N = N_{\mathrm{DP}}$.

### 2.6 Numerical comparison

For a 70B-parameter model on $N_{\mathrm{DP}} = 64$ DP ranks:

| Stage | Memory per GPU |
|---|---|
| DDP | $16 \times 70 = 1{,}120$ GB |
| ZeRO-1 | $4 \times 70 + 12 \times 70 / 64 = 293.1$ GB |
| ZeRO-2 | $2 \times 70 + 14 \times 70 / 64 = 155.3$ GB |
| ZeRO-3 | $16 \times 70 / 64 = 17.5$ GB |

ZeRO-3 at 64 DP ranks brings the per-GPU memory to 17.5 GB — trivially fitting in 80 GB HBM with room for activations, CUDA context, and temporary buffers. Adding activation memory (~10 GB without recomputation, ~0.3 GB with) gives total per-GPU memory of 17.8–27.5 GB, well within the 80 GB budget.

### 2.7 ZeRO stages worked example: 70B on 8 GPUs

A senior engineer should be able to compute these numbers from memory. For a 70B-parameter model ($P = 70 \times 10^9$) trained in mixed-precision BF16 with AdamW, the per-parameter cost is 16 bytes. On $N_{\mathrm{DP}} = 8$ DP ranks:

**DDP (ZeRO-0) — every rank holds everything:**

$$M_{\text{DDP}} = 16P = 16 \times 70 = 1{,}120 \;\text{GB per GPU}$$

Breakdown: 140 GB params (BF16) + 140 GB grads (BF16) + 280 GB master params (FP32) + 280 GB Adam $m$ (FP32) + 280 GB Adam $v$ (FP32). Does not fit on any GPU. With 8 H100s, total cluster memory is $8 \times 80 = 640$ GB — only 57% of the requirement. You need at least 14 GPUs just for state.

**ZeRO-1 — shard optimizer states only:**

$$M_{\text{ZeRO-1}} = 4P + \frac{12P}{N} = 4 \times 70 + \frac{12 \times 70}{8} = 280 + 105 = 385 \;\text{GB per GPU}$$

Breakdown: 140 GB params (full) + 140 GB grads (full) + 35 GB master params shard + 35 GB Adam $m$ shard + 35 GB Adam $v$ shard. Still does not fit on 80 GB. ZeRO-1 alone is insufficient for 70B on 8 GPUs.

**ZeRO-2 — shard optimizer states and gradients:**

$$M_{\text{ZeRO-2}} = 2P + \frac{14P}{N} = 2 \times 70 + \frac{14 \times 70}{8} = 140 + 122.5 = 262.5 \;\text{GB per GPU}$$

Breakdown: 140 GB params (full) + 17.5 GB grad shard + 35 GB master params shard + 35 GB Adam $m$ shard + 35 GB Adam $v$ shard. Still 3.3x the HBM budget. ZeRO-2 alone is also insufficient for 70B on 8 GPUs.

**ZeRO-3 — shard everything:**

$$M_{\text{ZeRO-3}} = \frac{16P}{N} = \frac{16 \times 70}{8} = 140 \;\text{GB per GPU}$$

Breakdown: 17.5 GB param shard + 17.5 GB grad shard + 35 GB master params shard + 35 GB Adam $m$ shard + 35 GB Adam $v$ shard. Still exceeds 80 GB. Even ZeRO-3 on 8 GPUs is not enough for 70B when activations are included.

**Conclusion for 70B:** The minimum practical configuration is ZeRO-3 with $N_{\mathrm{DP}} \geq 16$, giving $16 \times 70 / 16 = 70$ GB plus ~5 GB activations = 75 GB (tight fit on 80 GB). A comfortable fit requires $N_{\mathrm{DP}} \geq 32$ (35 GB state + 5 GB activations = 40 GB). Alternatively, combine TP=8 with ZeRO-3: TP=8 already reduces weights to 17.5 GB, and ZeRO-3 across DP=4 (32 GPUs total) gives $16 \times 70 / (8 \times 4) = 35$ GB total state per GPU — comfortable.

| ZeRO Stage | Formula | Memory/GPU (70B, N=8) | Fits 80 GB? |
|---|---|---|---|
| DDP (ZeRO-0) | $16P$ | 1,120 GB | No (14:1 oversubscribed) |
| ZeRO-1 | $4P + 12P/N$ | 385 GB | No (4.8:1 oversubscribed) |
| ZeRO-2 | $2P + 14P/N$ | 262.5 GB | No (3.3:1 oversubscribed) |
| ZeRO-3 | $16P/N$ | 140 GB | No (1.75:1 oversubscribed) |
| ZeRO-3, N=16 | $16P/16$ | 70 GB | Barely (with activation recomp) |
| ZeRO-3, N=32 | $16P/32$ | 35 GB | Yes (comfortable) |
| TP=8 + ZeRO-3, N=32 | $16P/(8 \times 4)$ | 35 GB | Yes |

---

## 3. FSDP mechanics — how ZeRO-3 is implemented

### 3.0 FSDP detailed dataflow with concrete numbers

FSDP wraps each `nn.Module` (typically each transformer layer). When a module is wrapped, its parameters are flattened and partitioned into $N_{\mathrm{DP}}$ equal shards. Here is the exact dataflow for a single training step on a 70B model with $N_{\mathrm{DP}} = 64$, FP8 forward/BF16 optimizer, sequence parallelism off:

**Per-layer parameter budget.** Each transformer layer of a 70B model ($L = 80$ layers) has approximately $70\text{B} / 80 = 875\text{M}$ parameters. In BF16, each layer's weight tensor is $2 \times 875\text{M} = 1.75$ GB. Each rank holds a shard of $1.75 / 64 = 27.3$ MB per layer.

**Forward pass, layer $i$:**

1. **All-gather parameters:** Before the forward computation of layer $i$, each rank contributes its 27.3 MB shard. The all-gather reconstructs the full 1.75 GB parameter tensor on every rank. The all-gather volume per rank: $(N-1)/N \times 1.75\text{ GB} \approx 1.72$ GB received.

2. **Compute:** The forward computation for layer $i$ executes against the full parameter tensor. FLOPs per layer: $\approx 2 \times 875\text{M} \times B \times S = 2 \times 875\text{M} \times 64 \times 4096 \approx 460$ GFLOPs per microbatch. At 990 TFLOPS (H100 FP16): ~0.47 ms per rank.

3. **Free:** Immediately after the layer completes, the gathered parameters are freed. Only the 27.3 MB local shard remains in memory.

**Backward pass, layer $i$ (in reverse order):**

1. **All-gather parameters:** Same as forward — reconstruct full 1.75 GB.

2. **Compute gradients:** $\nabla_{W_i} \mathcal{L}$ computed against the full parameter set. Backward FLOPs are ~2x forward: ~0.94 ms per rank.

3. **Reduce-scatter gradients:** The gradient tensor (1.75 GB in BF16) is reduce-scattered across all 64 ranks. Each rank receives its 27.3 MB gradient shard. Volume per rank: $(N-1)/N \times 1.75\text{ GB} \approx 1.72$ GB sent. The reduce-scatter simultaneously sums gradient contributions from all ranks and partitions the result.

4. **Free:** Discard full parameters and full gradients. Only the 27.3 MB gradient shard is accumulated.

5. **Optimizer step (delayed):** Each rank updates its local shard of FP32 master parameters, Adam $m$ and $v$ moments. The optimizer shard per rank: $(4+4+4) \times 875\text{M} / 64 \approx 41$ MB per layer. This happens after all layers complete backward.

**Memory timeline for a single rank during one step:**

| Phase | In-memory state | Peak memory |
|---|---|---|
| Before forward | All 80 layer param shards (27.3 MB each) + all optimizer shards (41 MB each) | $80 \times (27.3 + 41) = 5.46$ GB |
| Forward layer $i$ | + full params for layer $i$ (1.75 GB) + activations for layer $i$ (~0.13 GB with recomp) | 5.46 + 1.75 + 0.13 = 7.34 GB |
| Between layers | Activations accumulate; gathered params freed | ~5.5 GB + cumulative activations |
| Backward layer $i$ | + full params (1.75 GB) + full grads (1.75 GB) briefly | Peak ~9 GB |
| After backward | Only shards remain + gradient shards | $80 \times (27.3 + 27.3 + 41) = 7.65$ GB |

At no point does any single rank hold more than ~10 GB — well under 80 GB HBM.

**Total communication per step:**

$$V_{\text{FSDP}} = \underbrace{80 \times 1.72 \;\text{GB}}_{\text{fwd all-gather}} + \underbrace{80 \times 1.72 \;\text{GB}}_{\text{bwd all-gather}} + \underbrace{80 \times 1.72 \;\text{GB}}_{\text{bwd reduce-scatter}} = 413 \;\text{GB per rank per step}$$

Compare to DDP's single all-reduce: $2 \times 140 = 280$ GB. FSDP moves 47% more bytes, but the communication is spread across the forward and backward passes and overlapped with compute (Section 3.3). The effective blocking time is much lower than DDP's monolithic all-reduce.

### 3.1 Module wrapping and sharding

FSDP wraps PyTorch `nn.Module` instances. When a module is wrapped, its parameters are flattened into a single contiguous tensor and sharded across the DP group:

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,   # ZeRO-3
    device_id=torch.cuda.current_device(),
)
```

The `sharding_strategy` parameter controls the ZeRO stage:

| Strategy | Equivalent |
|---|---|
| `NO_SHARD` | DDP (no sharding) |
| `SHARD_GRAD_OP` | ZeRO-2 |
| `FULL_SHARD` | ZeRO-3 |

### 3.2 The forward-backward cycle

```mermaid
sequenceDiagram
    participant R as DP Rank
    participant P as Peer Ranks

    Note over R: Forward — Layer i
    R->>P: All-gather shard of W_i
    P->>R: Receive W_i shards from peers
    Note over R: Assembled full W_i locally
    Note over R: Compute y_i = Layer_i(x, W_i)
    Note over R: Discard full W_i (free memory)

    Note over R: Backward — Layer i
    R->>P: All-gather shard of W_i
    P->>R: Receive W_i shards from peers
    Note over R: Compute dL/dW_i
    R->>P: Reduce-scatter dL/dW_i
    P->>R: Receive gradient shard
    Note over R: Discard full W_i and full grad
    Note over R: Local optimizer step on shard
```

**Forward pass (layer $i$):**

1. **All-gather:** Each rank contributes its shard of $W_i$. The all-gather reconstructs the full parameter tensor on every rank. This is the only point where the full parameter exists in memory.
2. **Compute:** The forward computation for layer $i$ executes normally, as if the model were unsharded.
3. **Free:** Immediately after the layer completes, the gathered parameters are freed. Only the local shard remains.

**Backward pass (layer $i$, in reverse order):**

1. **All-gather:** Same as forward — reconstruct full $W_i$.
2. **Compute gradients:** $\nabla_{W_i} \mathcal{L}$ is computed against the full parameter set.
3. **Reduce-scatter:** The gradient is reduce-scattered so each rank receives only its shard of $\nabla_{W_i} \mathcal{L}$.
4. **Free:** Discard full parameters and full gradients. The local gradient shard is accumulated.
5. **Optimizer step:** Each rank updates its local shard of optimizer state and master parameters.

### 3.3 Communication-compute overlap

The key performance optimization in FSDP is overlapping the all-gather of layer $i+1$ with the compute of layer $i$. The implementation prefetches parameters during the current layer's computation:

```mermaid
gantt
    title FSDP forward pass — overlapped all-gather
    dateFormat X
    axisFormat %s

    section Layer i
    Compute layer i          : 0, 100
    Prefetch W_{i+1} (all-gather) : 30, 90

    section Layer i+1
    Compute layer i+1        : 100, 200
    Prefetch W_{i+2} (all-gather) : 130, 190
```

Without overlap, all-gather of all layers serializes with compute, roughly doubling the wall-clock time compared to DDP. With overlap, the communication is hidden behind compute as long as:

$$T_{\text{all-gather, layer}} < T_{\text{compute, layer}}$$

For a transformer layer with hidden dimension $d$ and microbatch tokens $T_{\text{tok}} = B \cdot S$:

$$T_{\text{compute}} \approx \frac{24 \, d^2 \, T_{\text{tok}}}{\text{FLOP/s}_{\text{GPU}}}$$

$$T_{\text{all-gather}} \approx \frac{2 \, d \, T_{\text{tok}} \cdot (N_{\mathrm{DP}} - 1) / N_{\mathrm{DP}}}{\text{BW}_{\text{inter-node}}}$$

Overlap succeeds when compute time exceeds all-gather time, which typically holds for $d \geq 4096$ on NVLink-connected nodes.

### 3.4 Hierarchical sharding (Hybrid FSDP)

When TP is used within a node and FSDP across nodes, parameters are already sharded by TP within the NVLink domain. FSDP then shards across the inter-node DP dimension. This two-level hierarchy reduces inter-node communication:

- **Intra-node (TP):** parameters sharded across 8 GPUs via NVLink — fast all-reduce per layer.
- **Inter-node (FSDP):** optimizer states sharded across DP nodes — reduce-scatter across the slower network.

The effective memory per GPU for TP=$T$, FSDP (ZeRO-3) across $N_{\mathrm{DP}}$ ranks:

$$M_{\text{GPU}} = \frac{16P}{T \cdot N_{\mathrm{DP}}} + M_{\text{act}}$$

**Worked problem 3 — 70B on 64 H100s.** Design the parallelism layout with TP=8, FSDP (ZeRO-3) across 8 DP ranks (8 nodes of 8 GPUs each):

$$M_{\text{GPU}} = \frac{16 \times 70 \times 10^9}{8 \times 8} = \frac{1{,}120 \;\text{GB}}{64} = 17.5 \;\text{GB}$$

Adding activation memory (~10 GB without recomputation, ~0.3 GB with) gives total per-GPU memory of 17.8–27.5 GB, well within the 80 GB budget. Pipeline parallelism is unnecessary at this scale.

---

## 4. Checkpointing

### 4.1 What to save

A complete training checkpoint contains:

| Component | Size | Purpose |
|---|---|---|
| BF16 model parameters | $2P$ | Model weights |
| FP32 master parameters | $4P$ | Optimizer base |
| Adam $m_t$ | $4P$ | First moment |
| Adam $v_t$ | $4P$ | Second moment |
| BF16 gradients | $2P$ | Optional (often re-derived) |
| LR scheduler state | $\sim$KB | Step counter, LR value |
| RNG state | $\sim$KB | Reproducibility |
| Data cursor | $\sim$KB | Token count consumed |
| **Total** | $\approx 16P$ | |

For common model sizes:

| Model | $P$ | Checkpoint size |
|---|---|---|
| 7B | $7 \times 10^9$ | 112 GB |
| 70B | $7 \times 10^{10}$ | 1.12 TB |
| 405B | $4.05 \times 10^{11}$ | 6.48 TB |
| 671B MoE | $6.71 \times 10^{11}$ | 10.7 TB |

### 4.2 Synchronous checkpointing

The simplest strategy: stop training, write checkpoint to storage, resume.

**Cost model.** With $N_{\mathrm{DP}}$ ranks writing in parallel at aggregate bandwidth $\text{BW}_{\text{write}}$:

$$T_{\text{sync}} = \frac{16P}{N_{\mathrm{DP}} \cdot \text{BW}_{\text{rank}}}$$

For 405B with 128 DP ranks writing to local NVMe at 5 GB/s per rank:

$$T_{\text{sync}} = \frac{6{,}480 \;\text{GB}}{128 \times 5 \;\text{GB/s}} = 10.1 \;\text{s}$$

At 10 seconds per checkpoint saved hourly, the overhead is 0.28% — acceptable. But if checkpoints go to remote object storage (S3) at 1 GB/s per rank:

$$T_{\text{sync}} = \frac{6{,}480}{128 \times 1} = 50.6 \;\text{s} \quad (1.4\% \text{ overhead})$$

### 4.3 Asynchronous checkpointing

PyTorch Distributed Checkpoint (DCP) decouples the write from training:

```mermaid
flowchart TD
    A["Training step completes"] --> B["Copy state to<br/>staging buffer"]
    B --> C["Training resumes<br/>immediately"]
    B --> D["Background thread<br/>writes to storage"]
    D --> E["Checkpoint persisted"]
```

The staging buffer requires $16P / N_{\mathrm{DP}}$ bytes of extra GPU or CPU memory. For 70B with 64 DP ranks, that is $1{,}120/64 = 17.5$ GB per rank — manageable in CPU RAM.

The trade-off: if a failure occurs between the staging copy and the storage write, the checkpoint is lost. This creates a window of vulnerability approximately equal to $T_{\text{write}}$. Mitigation: write to local NVMe first (fast), then asynchronously replicate to remote storage.

### 4.4 Distributed checkpointing

In a distributed checkpoint, each rank writes only its local shard. The checkpoint is a directory of per-rank files plus a metadata file recording the world size, TP/PP/DP decomposition, and shard map:

```
checkpoint/
  metadata.json          # world_size, dp_degree, tp_degree, pp_degree
  rank_00000.pt          # shard for rank 0
  rank_00001.pt          # shard for rank 1
  ...
  rank_NNNNN.pt          # shard for rank N-1
```

On restore, each rank reads only its own shard. This eliminates the need for any rank to hold the full model — critical for models that exceed single-node memory.

**Resharding.** If the world size changes between save and restore (elastic training, different cluster), a reshard utility re-partitions the checkpoint. The metadata file contains enough information to reconstruct the full parameter set on a single node and re-shard.

**Resharding in detail.** Consider a 70B model saved with ZeRO-3 on $N_{\text{save}} = 64$ ranks and restored on $N_{\text{restore}} = 128$ ranks. Each saved shard covers $P / 64 = 1.09\text{B}$ parameters. The resharding process:

1. Each new rank reads the metadata file to determine the original sharding layout.
2. Each new rank identifies which original shard(s) contain its portion of the parameter space. With $N_{\text{save}} = 64$ and $N_{\text{restore}} = 128$, each new rank needs half of one original shard.
3. Ranks coordinate via a shared filesystem or object store. Rank $i$ of the new world reads the relevant portion of the original shard(s), extracts its $P / 128 = 547\text{M}$ parameters, and constructs its local optimizer state (initializing Adam moments to zero for the portion it newly owns, or reading them if the checkpoint includes per-shard optimizer state).
4. If optimizer state was checkpointed per-shard, the reshard utility must split or merge Adam $m_t$ and $v_t$ tensors. For a split (64 -> 128), each original shard's optimizer state is sliced in half. For a merge (128 -> 64), two shards' optimizer states are concatenated.

The resharding time is dominated by I/O, not compute. For a 70B model with 1.12 TB of checkpoint data on NVMe at 5 GB/s per rank: $1.12\text{ TB} / (128 \times 5\text{ GB/s}) \approx 1.75$ seconds. On S3 at 1 GB/s per rank: ~8.75 seconds. PyTorch's `torch.distributed.checkpoint` utilities handle this automatically via `reshard` APIs.

**Sharded checkpoint on different GPU counts.** A practical concern: can I save on 64 GPUs and resume on 48? Yes, but the resharding logic must handle non-divisor relationships. If $N_{\text{save}} = 64$ and $N_{\text{restore}} = 48$, each new rank needs parameters from $\lceil 64/48 \rceil = 2$ original shards (with partial reads). The metadata file maps new rank IDs to original shard ranges. PyTorch DCP handles this; custom checkpoint formats may not.

---

## 5. Fault tolerance

### 5.1 Failure modes at scale

At 10K GPUs, failures are not exceptional — they are expected. Empirical failure rates from production training runs:

| Failure type | MTBF (10K GPUs) | Detection mechanism |
|---|---|---|
| GPU hardware (ECC, hang) | ~24 hours | NCCL timeout, CUDA error |
| Network link flap | ~6 hours | NCCL abort, IB counter |
| Node power / cooling | ~72 hours | Job scheduler health check |
| Software OOM | Variable | Process killed (SIGKILL) |
| Silent data corruption | ~1000 hours | Loss NaN, gradient norm spike |

The composite MTBF across all failure types for a 10K-GPU cluster is approximately 4–8 hours. A 30-day pretraining run experiences 90–180 failure events.

### 5.2 Detecting failures

```mermaid
flowchart TD
    A["NCCL collective timeout<br/>(default 30 min)"] --> F["Rank failure detected"]
    B["Process exits with<br/>non-zero code"] --> F
    C["Loss = NaN or inf"] --> F
    D["Gradient norm > 100×<br/>running average"] --> F
    E["Step time > 3×<br/>median (straggler)"] --> F
    F --> R["Trigger recovery protocol"]
```

Detection latency matters: a silent hang in NCCL can waste 30 minutes of compute (the default timeout). Production systems reduce this to 60–120 seconds via:

- **NCCL timeout override:** `NCCL_COMM_BLOCKING=1` with custom watchdog threads.
- **Heartbeat monitors:** each rank sends a heartbeat every 10 seconds; missing 3 heartbeats triggers failure.
- **Loss watchdog:** an independent thread monitors loss values and alerts on NaN, inf, or $>10\times$ spike.

### 5.3 Recovery strategies

**Strategy 1: Restart from checkpoint.** The simplest approach. On failure, terminate the entire job and resubmit from the last checkpoint.

$$\text{Wasted compute per failure} = T_{\text{checkpoint\_interval}} \times N_{\text{GPU}}$$

With hourly checkpointing and 10K GPUs: ~5,000 GPU-hours lost per failure. At $3/\text{GPU-hour}$: $15,000 per failure. With 90 failures over 30 days: $1.35M in wasted compute.

**Strategy 2: Rolling restart with spares.** Allocate $N_{\text{GPU}} + N_{\text{spare}}$ GPUs (typically 2–5% spare). On failure:

1. Identify the failed rank and its node.
2. Exclude the node from the job.
3. Map the excluded rank's work to a spare node.
4. Load the most recent checkpoint (or in-memory replica).
5. Resume training with a new world configuration.

This approach requires elastic training support: the ability to reconfigure DP/TP/PP groups at runtime.

**Strategy 3: In-memory replication.** Before each checkpoint, each rank replicates its state to a neighbor rank's CPU RAM via RDMA. On failure, the neighbor restores the failed rank's state to a spare GPU.

Recovery time: state transfer from neighbor RAM to spare GPU via PCIe (~5–10 seconds for 17.5 GB at 32 GB/s). Compare to loading from NVMe (~3.5 seconds) or S3 (~17.5 seconds). In-memory replication is fastest but doubles the per-rank CPU memory requirement.

### 5.4 Elastic training

Elastic training adjusts the parallelism configuration when the number of available GPUs changes. The key challenge is maintaining training correctness when $N_{\mathrm{DP}}$ changes:

- **Optimizer state:** must be re-sharded to the new $N_{\mathrm{DP}}$. Each rank's optimizer shard is split or merged accordingly.
- **Learning rate:** the per-step learning rate depends on the global batch size, which depends on $N_{\mathrm{DP}}$. When $N_{\mathrm{DP}}$ changes, the learning rate schedule must be adjusted.
- **Gradient accumulation:** if $N_{\mathrm{DP}}$ decreases, gradient accumulation steps must increase to maintain the same effective batch size.

**torchrun — how RANK/WORLD_SIZE are reassigned:**

PyTorch Elastic (`torchrun`) provides the infrastructure for dynamic world size management. The key mechanism:

1. **Rendezvous:** When a failure is detected (or a new node joins), `torchrun` triggers a rendezvous. All surviving workers reach a barrier. The rendezvous assigns new `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, and `MASTER_ADDR`/`MASTER_PORT` environment variables.

2. **Rank reassignment example:** Suppose 8 workers (RANK 0-7) are training, and the node hosting RANK 3 fails. The surviving 6 workers rendezvous. The new assignment:
   - Original RANKs 0, 1, 2 become new RANKs 0, 1, 2 (unchanged).
   - Original RANK 4 becomes new RANK 3.
   - Original RANK 5 becomes new RANK 4.
   - Original RANK 6 becomes new RANK 5.
   - Original RANK 7 becomes new RANK 6.
   - WORLD_SIZE changes from 8 to 6.

3. **State reconstruction:** Each surviving worker loads the most recent checkpoint. Because the checkpoint is sharded, the workers must collectively reconstruct the state for the missing rank. With ZeRO-3, each rank's shard is $16P/N$; the missing shard must be loaded from the checkpoint on shared storage.

4. **Process group rebuild:** The NCCL communicator (`torch.distributed.new_group()`) is destroyed and recreated with the new world size. This requires a full NCCL re-initialization (topology discovery, ring construction, channel allocation) which takes 5-30 seconds depending on scale.

5. **torchrun configuration:**

```bash
torchrun \
  --nnodes=4:8 \           # min 4, max 8 nodes (elastic)
  --nproc_per_node=8 \     # 8 GPUs per node
  --rdzv_id=job-42 \       # unique job ID for rendezvous
  --rdzv_backend=c10d \    # rendezvous backend (c10d = TCP store)
  --rdzv_endpoint=mgmt:29500 \  # rendezvous server
  --max_restarts=3 \       # restart limit per worker
  --timeout=120 \          # seconds before watchdog fires
  train.py
```

The `--nnodes=4:8` allows the job to run with 4 to 8 nodes. When a node fails, `torchrun` automatically shrinks the world size to the surviving nodes (as long as it stays above 4). When a spare node is provisioned, the next rendezvous expands to include it.

**Rebuilding the process group after a failure:**

The process group rebuild is the most latency-sensitive step. At scale:

| Scale | NCCL init time | Checkpoint load | Optimizer reshard | Total recovery |
|---|---|---|---|---|
| 8 GPUs (1 node) | ~2 s | ~1 s | ~0.5 s | ~3.5 s |
| 64 GPUs (8 nodes) | ~5 s | ~3 s | ~2 s | ~10 s |
| 512 GPUs (64 nodes) | ~15 s | ~5 s | ~5 s | ~25 s |
| 4096 GPUs (512 nodes) | ~30 s | ~10 s | ~15 s | ~55 s |

At 4096 GPUs, a single failure costs ~55 seconds of recovery. With MTBF of 4 hours and 55 seconds recovery: overhead = $55 / (4 \times 3600) = 0.38\%$ — acceptable. But if recovery is buggy (stuck NCCL init, corrupt checkpoint), the cascading cost can be much higher.

**Straggler/hung rank detection:**

Production systems use multiple watchdog layers:

1. **Application-level watchdog:** Each rank updates a shared counter in the rendezvous store after every step. A monitoring thread checks that all ranks advance within $3\times$ the median step time. If a rank has not advanced after $T_{\text{watchdog}} = 3 \times T_{\text{median}}$, it is declared a straggler.

2. **NCCL timeout:** `NCCL_COMM_BLOCKING=1` with `NCCL_TIMEOUT=1800` (30 minutes default). Production systems reduce this to 60-120 seconds via `NCCL_TIMEOUT=120`.

3. **Hardware health:** NVIDIA DCGM (Data Center GPU Manager) monitors ECC errors, temperature, and power. A GPU with $> 10$ double-bit ECC errors in 60 seconds is automatically excluded.

4. **Heartbeat:** Each rank sends a UDP heartbeat to the `torchrun` agent every 5 seconds. If 3 consecutive heartbeats are missed (15 seconds), the agent declares the rank dead and triggers a rendezvous.

### 5.5 Cost model for checkpoint interval

The optimal checkpoint interval minimizes the total cost of checkpointing plus recompute. Let $T_c$ be the checkpoint interval, $T_s$ the time to save a checkpoint, $T_f$ the mean time between failures, and $T_r$ the time to restore:

$$\text{Overhead} = \frac{T_s}{T_c} + \frac{T_c / 2}{T_f}$$

The first term is the fraction of time spent saving; the second is the expected wasted compute (on average, half a checkpoint interval is lost per failure). Minimizing over $T_c$:

$$T_c^* = \sqrt{2 \, T_s \, T_f}$$

With $T_s = 10$ s and $T_f = 4$ hours (14,400 s):

$$T_c^* = \sqrt{2 \times 10 \times 14{,}400} = 537 \;\text{s} \approx 9 \;\text{minutes}$$

In practice, checkpoints are saved less frequently (every 30–60 minutes) because the overhead of frequent checkpointing extends total training time. The mathematical optimum assumes checkpoint cost is purely time; in reality, storage wear and I/O contention also factor in.

**Worked problem 4 — checkpoint cost.** For a 70B model trained on 512 H100s for 14 days with hourly checkpointing, compute the expected wasted compute. Assume MTBF = 8 hours at this scale.

Failures expected: $14 \times 24 / 8 = 42$ failures.

Wasted compute per failure: $0.5 \;\text{hours} \times 512 = 256$ GPU-hours.

Total wasted compute: $42 \times 256 = 10{,}752$ GPU-hours.

Total planned compute: $14 \times 24 \times 512 = 172{,}032$ GPU-hours.

Waste fraction: $10{,}752 / 172{,}032 = 6.3\%$.

At $3/GPU-hour: $32,256 in wasted compute. Reducing the checkpoint interval to 30 minutes would halve this to 3.1%.

---

## 6. MoE-specific distributed training

MoE models introduce expert parallelism (EP), which creates unique distributed training challenges beyond dense models. See [Modern_MoE](../L6_Algorithms_and_Models/Modern_MoE.md) for the architectural context.

### 6.1 Expert parallelism communication pattern

In EP, each DP rank holds a subset of experts. During the forward pass through an MoE layer:

1. **Dispatch (all-to-all):** Tokens are sent to the GPUs hosting their assigned experts. Each token carries its hidden state ($d$ elements) to the target expert's GPU.
2. **Expert computation:** Each GPU computes the FFN for the tokens it received.
3. **Combine (all-to-all):** Expert outputs are sent back to the token's original GPU.

$$\text{Comm per MoE layer} = 2 \times \underbrace{B \cdot S \cdot d \cdot \frac{k}{E}}_{\text{tokens dispatched per rank}} \times 2 \;\text{bytes (BF16)}$$

For DeepSeek-V3 ($B \cdot S = 4096$ tokens, $d = 7168$, $k = 8$, $E = 256$, EP = 64):

$$\text{Dispatch volume} = 2 \times 4096 \times 7168 \times \frac{8}{256} \times 2 = 3{,}670 \;\text{MB per MoE layer}$$

With 58 MoE layers in DeepSeek-V3, total all-to-all volume per training step:

$$V_{\text{MoE, total}} = 58 \times 3{,}670 \;\text{MB} \approx 213 \;\text{GB}$$

This dominates communication cost and must stay within high-bandwidth fabric (NVLink or NVL).

### 6.2 Expert balance across nodes

Load imbalance in MoE training has a direct throughput cost: the slowest EP rank determines the step time. If one rank receives $1.5\times$ the average token count, it becomes the bottleneck and all other ranks idle during the expert computation phase.

**Load balancing mechanisms:**

| Mechanism | How it works | Overhead |
|---|---|---|
| Auxiliary loss | Penalize uneven expert utilization in the training loss | Adds bias to routing; harms quality |
| Bias-based (DeepSeek-V3) | Per-expert bias term adjusted online to equalize load | No quality impact; requires tuning of update rate |
| Capacity factor ($C_f$) | Hard cap on tokens per expert; excess tokens are dropped or passed through | $C_f < 1$ drops tokens (quality loss); $C_f > 1$ wastes memory |
| EP-aware batch packing | Pre-sort tokens in a microbatch to equalize per-expert counts | Requires data-loader changes; latency overhead |

DeepSeek-V3's aux-loss-free approach with bias tuning is the current best practice. The bias for expert $i$ is updated as:

$$b_i \leftarrow b_i + \gamma \cdot (\bar{n}_i - N_{\text{target}})$$

where $\bar{n}_i$ is the recent average token count for expert $i$, $N_{\text{target}}$ is the target count, and $\gamma$ is a small update rate (typically 0.01).

### 6.3 EP communication optimization

```mermaid
flowchart TD
    A["MoE layer forward"] --> B{"EP degree = DP degree?"}
    B -->|Yes| C["Single all-to-all<br/>(tokens → experts)"]
    B -->|No, EP > DP| D["Hierarchical all-to-all:<br/>intra-node then inter-node"]
    C --> E["Expert computation"]
    D --> E
    E --> F["Reverse all-to-all<br/>(experts → tokens)"]
    F --> G["Combine with shared expert"]
```

When EP spans multiple nodes, the all-to-all crosses the inter-node network bottleneck. Optimizations:

- **Expert placement:** co-locate frequently co-activated experts on the same node. This reduces inter-node dispatch volume.
- **Overlap with compute:** begin dispatching tokens for layer $i+1$ while computing layer $i$'s experts — analogous to FSDP's parameter prefetch.
- **Token dropping:** when $C_f < 1$, drop excess tokens rather than blocking. The dropped tokens' hidden states are passed through unchanged. Used in training (with $C_f \approx 0.8\text{–}1.0$); not in inference.

### 6.4 MoE training memory

MoE models have larger total parameter counts but similar per-expert sizes. The memory calculation for ZeRO-3/FSDP applies to the total parameter count $P_{\text{total}}$:

$$M_{\text{GPU, MoE}} = \frac{16 \, P_{\text{total}}}{N_{\mathrm{DP}} \cdot \text{EP}} + M_{\text{act}} + M_{\text{dispatch\_buffer}}$$

The dispatch buffer holds tokens received from other ranks during the all-to-all:

$$M_{\text{dispatch}} = B \cdot S \cdot \frac{k}{E} \cdot d \cdot 2 \;\text{bytes}$$

For DeepSeek-V3: $4096 \times (8/256) \times 7168 \times 2 = 1.84$ MB per MoE layer — negligible compared to the parameter memory.

---

## 7. Framework updates (2025–2026)

### 7.0.1 FSDP2 and torch.compile integration

PyTorch 2.12 redesigned the FSDP2 compile path. The previous approach (which relied on hooks and `fullgraph` compilation) has been replaced with a dedicated **compile wrapping** mechanism. Key changes:

- **No more hook-based sharding.** FSDP2 now uses a direct compile-aware wrapping strategy that avoids the overhead and composability issues of Python-level hooks. Parameter sharding, all-gather, and reduce-scatter are lowered directly into the compiled graph.
- **`torch.accelerator.Graph` API.** This new API provides a unified CUDA Graph interface that composes cleanly with FSDP2. Instead of manually managing graph capture and replay, `torch.accelerator.Graph` handles warmup, capture, and replay as a single abstraction. It is aware of FSDP2's communication patterns and inserts graph breaks at the correct boundaries.
- **Improved compile overlap.** Because the compile path is no longer mediated by hooks, the compiler has full visibility into communication operations and can overlap all-gather with compute at the graph level rather than the Python level. This improves MFU by 5–10% over the hook-based path.

### 7.0.2 DeepSpeed v0.15

DeepSpeed v0.15 (2025) introduced several production-grade features for large-scale training:

| Feature | Description |
|---|---|
| **DeepNVMe GDS** | GPU Direct Storage integration for direct NVMe-to-GPU checkpoint transfer, bypassing CPU RAM. Reduces checkpoint write latency by 2–3x for models with terabyte-scale state. |
| **Universal Checkpoint for ZeRO-3** | Portable checkpoint format that can be resharded across different parallelism configurations (TP, PP, DP degree changes) without loading the full model on a single node. Critical for elastic training. |
| **fp8-fused GEMM kernels** | Fused FP8 matmul kernels with delayed scaling, matching Transformer Engine throughput within DeepSpeed's optimizer pipeline. |
| **MoE top-k gate support (k > 2)** | Extended expert routing to support top-k with k > 2, enabling finer-grained expert selection for models like DeepSeek-V3 (top-8 routing). |
| **Ulysses attention integration** | Integrated Ulysses-style sequence-parallel attention, allowing heads to be distributed across the DP dimension for long-context training. |

### 7.0.3 Megatron Core 0.15–0.17

NVIDIA's Megatron-Core (the library layer beneath Megatron-LM) received major updates in versions 0.15 through 0.17:

**MoE optimizations (0.15–0.17):**
- **Expert Parallel A2A overlap:** All-to-all dispatch/combine for expert parallelism is overlapped with attention and MLP compute, hiding communication latency. See [Parallelism_Strategies](Parallelism_Strategies.md) Section 4.5.
- **HybridEP:** Combines tensor parallelism (TP) and expert parallelism (EP) so experts are distributed across a hybrid group, improving GPU utilization for models with many small experts. See [Parallelism_Strategies](Parallelism_Strategies.md) Section 4.5.
- **NVFP4 quantization for MoE:** 4-bit floating-point weight storage for MoE expert parameters during training, reducing memory footprint with minimal quality impact.
- **Router fusion:** Fuses the expert routing computation with the preceding layer, eliminating a kernel launch and reducing latency per MoE layer.
- **LatentMoE:** Projects input to a lower-dimensional latent space before routing to experts, reducing expert computation cost. See [Training_Optimization](Training_Optimization.md).

**Multi-Token Prediction (MTP):**
- Megatron-Core 0.15+ supports predicting multiple future tokens simultaneously during training, improving the training signal per forward pass. See [Training_Optimization](Training_Optimization.md).

**RL training infrastructure (0.16–0.17):**
- **GRPO functional tests:** Production-grade GRPO support with importance sampling and sequence packing for RL fine-tuning of reasoning models.
- **Importance sampling:** Correctly weights samples from old policies during PPO/GRPO updates, improving sample efficiency.
- **Sequence packing:** Packs multiple short sequences into a single training example to maximize GPU utilization during RL rollouts.

**Other:**
- **FlashAttention-4 backend:** Integration with FlashAttention-4 for improved attention throughput on Blackwell GPUs.
- **Megatron-FSDP integration:** Hybrid mode that uses Megatron-Core's TP/PP implementation with FSDP's ZeRO-3 sharding, combining the strengths of both frameworks.

---

## 8. Framework comparison

### 8.1 Feature matrix

| Feature | FSDP (PyTorch) | DeepSpeed | Megatron-LM |
|---|---|---|---|
| ZeRO-1 / ZeRO-2 / ZeRO-3 | Yes (via sharding_strategy) | Yes (ZeRO origin) | No (uses TP/PP for memory) |
| Tensor parallelism | Via DTensor + DeviceMesh | Via Megatron integration | Native (reference implementation) |
| Pipeline parallelism | Via PiPPy / device mesh | Yes (1F1B, interleaved) | Yes (1F1B, interleaved, V-shape) |
| Sequence parallelism | Via DTensor | No | Yes (ring attention, context parallel) |
| Expert parallelism | Via DTensor + MoE APIs | Yes (DeepSpeed-MoE) | Yes (Megatron-Core MoE) |
| Activation recomputation | Yes (checkpoint_wrapper) | Yes (activation_checkpointing) | Yes (selective, per-layer) |
| Async checkpointing | Yes (DCP — distributed checkpoint) | Yes (DeepSpeed checkpoint) | Manual / external |
| Offload to CPU/NVMe | No (GPU-only) | Yes (ZeRO-Offload, ZeRO-Infinity, DeepNVMe GDS) | No |
| Mixed precision | Native (AMP) | Native (AMP, BF16, FP8 via TE) | Via Transformer Engine |
| CUDA Graph integration | torch.accelerator.Graph (FSDP2) | Limited | Limited |
| MoE top-k (k > 2) | Via DTensor | Yes (v0.15+) | Yes (Megatron-Core 0.16+) |
| RL training (GRPO/PPO) | Via external frameworks | Via external frameworks | Yes (Megatron-Core 0.16+: importance sampling, sequence packing) |
| FlashAttention-4 | Via external library | Via external library | Yes (Megatron-Core 0.17+) |
| Universal / portable checkpoint | Yes (DCP resharding) | Yes (Universal Checkpoint v0.15) | Manual |
| Ecosystem integration | torch.compile, TorchDynamo | HF Accelerate, Deepspeed-MoE | NVIDIA NeMo, Megatron-Core |
| Open source | Yes (PyTorch repo) | Yes (Microsoft) | Yes (NVIDIA) |
| Production users | Meta (Llama), Lightning | Microsoft, academic labs | NVIDIA, MosaicML, Cohere |

### 8.2 Choosing a framework

```mermaid
flowchart TD
    Q1{"Model < 13B?"}
    Q1 -->|Yes| A["FSDP (ZeRO-3)<br/>Simplest setup"]
    Q1 -->|No| Q2{"Need TP or PP?"}
    Q2 -->|No| B["FSDP (ZeRO-3)<br/>Scales to 70B with enough DP"]
    Q2 -->|Yes| Q3{"NVIDIA GPU stack?"}
    Q3 -->|Yes| C["Megatron-LM + FSDP<br/>(hybrid TP + ZeRO-3)"]
    Q3 -->|Mixed / AMD| D["FSDP + DTensor<br/>(native TP via device mesh)"]
    Q3 -->|Need CPU offload| E["DeepSpeed ZeRO-Infinity<br/>(offload optimizer to CPU/NVMe)"]
```

**Rules of thumb:**

1. **Start with FSDP.** It is PyTorch-native, composable with `torch.compile`, and handles models up to ~100B with sufficient DP.
2. **Add TP (Megatron-LM or DTensor) when FSDP communication becomes the bottleneck.** This happens when the all-gather per layer exceeds the layer's compute time — typically at $d > 8192$ or with slow inter-node networks.
3. **Add PP when the model exceeds one node's memory even with TP.** Each PP stage holds a subset of layers; the pipeline bubble overhead is acceptable when $M \gg P$ (many microbatches per pipeline).
4. **Use DeepSpeed for CPU offloading** when GPU memory is scarce and CPU RAM is abundant (e.g., academic clusters with older GPUs).

### 8.3 Communication volume comparison

For a single training step with a 70B model, $B \cdot S = 4096$ tokens per microbatch, $L = 80$ layers:

| Configuration | Comm per step (GB) | Comm type |
|---|---|---|
| DDP ($N_{\mathrm{DP}} = 64$) | $2 \times 140 = 280$ | All-reduce |
| FSDP / ZeRO-3 ($N_{\mathrm{DP}} = 64$) | $2 \times 2 \times 80 \times \frac{2 \times 70}{64} = 700$ | All-gather + reduce-scatter |
| TP=8 ($N_{\mathrm{DP}} = 1$) | $2 \times 80 \times 2 \times 0.008 \times 8 = 20.5$ | All-reduce per layer |
| TP=8 + FSDP ($N_{\mathrm{DP}} = 8$) | $20.5 + 87.5 = 108$ | Combined |

Note: FSDP total communication is higher than DDP's (two all-gathers + one reduce-scatter per layer vs one all-reduce per step), but FSDP's communication is overlapped with compute. DDP's all-reduce blocks the entire backward pass. Effective throughput with overlap makes FSDP faster despite higher total bytes moved.

---

## 9. Cause / effect — from memory wall to distributed training

```mermaid
flowchart TD
    A["16P bytes per GPU<br/>for mixed-precision AdamW"] --> B["70B = 1.12 TB<br/>does not fit on 80 GB GPU"]
    B --> C["Shard across DP dimension"]
    C --> D["ZeRO-1: shard optimizer<br/>75% of memory, no fwd change"]
    D --> E{"Still does not fit?"}
    E -->|"Yes"| F["ZeRO-2: shard optimizer + gradients<br/>reduces to 2P + 14P/N"]
    F --> G{"Still does not fit?"}
    G -->|"Yes"| H["ZeRO-3 / FSDP: shard all<br/>reduces to 16P/N"]
    H --> I["Fits in GPU memory"]
    E -->|"No"| I
    G -->|"No"| I

    I --> J["But: all-gather per layer<br/>adds communication"]
    J --> K["Overlap all-gather with compute<br/>via prefetch"]
    K --> L{"Comm still exceeds compute?"}
    L -->|"Yes"| M["Add TP to reduce per-layer<br/>shard size and comm volume"]
    L -->|"No"| N["Training proceeds"]
    M --> N

    B --> O["10K GPU cluster:<br/>~1 failure per day"]
    O --> P["Checkpoint every T_c seconds"]
    P --> Q["Async checkpointing<br/>minimizes pause"]
    Q --> R["Rolling restart with spares<br/>minimizes downtime"]
    R --> N
```

---

## 10. Worked problem 5 — end-to-end layout design

**Problem.** Design the distributed training layout for a 405B dense model on 4,096 H100 GPUs (512 nodes of 8 GPUs each). Sequence length $S = 8192$, microbatch $B = 1$.

**Step 1: Memory requirement.**

$$M_{\text{total}} = 16 \times 405 = 6{,}480 \;\text{GB}$$

$$M_{\text{act}} \approx 126 \times 2 \times 1 \times 8192 \times 16384 \approx 33.8 \;\text{GB (no recomputation)}$$

$$M_{\text{act,recomp}} \approx 1 \;\text{GB (with selective recomputation)}$$

**Step 2: Choose TP.** Set TP = 8 (intra-node NVLink). Parameter memory per GPU after TP:

$$M_{\text{params,TP8}} = \frac{6{,}480}{8} = 810 \;\text{GB}$$

Still 10x too large. Need additional sharding.

**Step 3: Choose PP.** With 126 layers, set PP = 8 stages (~16 layers per stage). Memory per GPU after TP + PP:

$$M_{\text{params,TP8\_PP8}} = \frac{6{,}480}{8 \times 8} = 101.3 \;\text{GB}$$

Still exceeds 80 GB when including activations and overhead.

**Step 4: Add ZeRO-1 (FSDP shard optimizer).** Remaining GPUs for DP: $4096 / (8 \times 8) = 64$ DP ranks.

$$M_{\text{ZeRO-1}} = \frac{4 \times 405 \times 10^9}{8 \times 8} + \frac{12 \times 405 \times 10^9}{8 \times 8 \times 64}$$

$$= 31.6 \;\text{GB (params + grads)} + 1.5 \;\text{GB (optimizer shard)} = 33.1 \;\text{GB}$$

Adding activations (~1 GB with recomputation): total ~34 GB. Fits in 80 GB with headroom.

**Step 5: Verify communication.**

- TP: 2 all-reduces per layer per stage, payload $\approx 2 \times B \times S \times d = 2 \times 8192 \times 16384 = 0.25$ GB. At NVLink 900 GB/s: negligible.
- PP: point-to-point activation transfer between stages. Payload per microbatch: $2 \times B \times S \times d = 0.25$ GB. At 400 Gb/s InfiniBand: ~5 ms per transfer.
- ZeRO-1: reduce-scatter of gradients once per step, payload $\approx 2P = 810$ GB (full grad). At 50 GB/s inter-node: ~16 s per step — too slow.

**Step 5 revision: Use ZeRO-3 instead.** With ZeRO-3, communication is all-gather + reduce-scatter per layer (not per step), overlapped with compute:

$$M_{\text{ZeRO-3}} = \frac{6{,}480}{8 \times 8 \times 64} = 1.6 \;\text{GB}$$

Adding activations: ~2.6 GB total. Very comfortable.

Communication per layer per stage: all-gather of $\frac{2 \times 405 \times 10^9}{8 \times 8 \times 64} = 0.2$ GB. At 50 GB/s: 4 ms per all-gather. Compute per layer: ~15 ms. Overlap is feasible.

**Final layout:**

$$\text{TP} = 8, \quad \text{PP} = 8, \quad N_{\mathrm{DP}} = 64, \quad \text{FSDP (ZeRO-3)}$$

$$N_{\text{GPU}} = 8 \times 8 \times 64 = 4{,}096 \checkmark$$

---

## 11. Key numbers

| Quantity | Value | Note |
|---|---|---|
| Memory per param (mixed-precision AdamW) | 16 B | 2+2+4+4+4 |
| ZeRO-3 memory per GPU | $16P / N_{\mathrm{DP}}$ | Plus activations |
| Activation memory (no recomp, per layer) | $2 \cdot B \cdot S \cdot d$ bytes | FlashAttention assumed |
| Activation memory (full recomp) | $\sim 2 \cdot B \cdot S \cdot d / c$ | $c$ = checkpoint segments |
| Checkpoint size | $16P$ bytes | Full training state |
| Optimal checkpoint interval | $\sqrt{2 \, T_s \, T_f}$ | Minimizes waste |
| MTBF at 10K GPUs | ~4–8 hours | All failure types |
| EP all-to-all per MoE layer | $2 \cdot B \cdot S \cdot d \cdot (k/E) \cdot 2$ bytes | Dispatch + combine |
| FSDP all-gather per layer | $2P / N_{\mathrm{DP}}$ bytes | Per rank |

---

## 12. References

**Foundational**

- R. Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models," *SC*, 2020.
- S. Smith et al., "Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B," *SC*, 2022.
- V. Korthikanti et al., "Reducing Activation Recomputation in Large Transformer Models," *MLSys*, 2023.
- M. R. Washington et al., "Checkpointing at Scale for Llama-3 Training," *NSDI*, 2025.

**FSDP**

- PyTorch Team, "Getting Started with Fully Sharded Data Parallel (FSDP)," 2023.
- PyTorch Team, "PyTorch Distributed Checkpoint (DCP)," 2024.
- Meta, "Llama 3 Model Card and Training Details," 2024.
- PyTorch Team, "FSDP2 and torch.accelerator.Graph API," 2025–2026.

**DeepSpeed**

- Microsoft DeepSpeed Team, "DeepSpeed v0.15 Release Notes," 2025.
- Microsoft DeepSpeed Team, "DeepNVMe GDS: GPU Direct Storage for Checkpointing," 2025.
- Microsoft DeepSpeed Team, "Universal Checkpoint for ZeRO-3," 2025.

**Megatron Core**

- NVIDIA, "Megatron-Core 0.15–0.17 Release Notes," 2025.
- NVIDIA, "Megatron-FSDP Integration," 2025.
- NVIDIA, "FlashAttention-4 Backend in Megatron-Core," 2025–2026.

**MoE training**

- DeepSeek-AI, "DeepSeek-V3 Technical Report," 2024.
- B. Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models," *JMLR*, 2022.

**Fault tolerance**

- J. W. Young, "A First Order Approximation to the Optimum Checkpoint Interval," *CACM*, 1974.
- TorchElastic documentation, PyTorch, 2024.

**Cross-references**

- [Parallelism_Strategies](Parallelism_Strategies.md) — DP, TP, PP, EP, CP, SP with communication volume math.
- [Collectives_and_NCCL](Collectives_and_NCCL.md) — all-reduce, reduce-scatter, all-gather algorithms and bandwidth modeling.
- [Modern_MoE](../L6_Algorithms_and_Models/Modern_MoE.md) — MoE architecture, routing, and load balancing.
- [Storage_and_Model_Loading](../L4_Systems_and_Interconnects/Storage_and_Model_Loading.md) — checkpoint formats, I/O paths, and model loading.
- [Training_Optimization](Training_Optimization.md) — mixed precision, gradient accumulation, Transformer Engine.

---

**Up the stack:** [Training_Optimization](Training_Optimization.md), [Modern_Post_Training](Modern_Post_Training.md).
**Down the stack:** [Parallelism_Strategies](Parallelism_Strategies.md), [Collectives_and_NCCL](Collectives_and_NCCL.md), [Modern_MoE](../L6_Algorithms_and_Models/Modern_MoE.md).

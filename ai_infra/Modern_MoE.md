# Modern MoE — Fine-Grained, Shared-Expert, and EP at Scale

The 2024–2026 MoE designs (DeepSeekMoE, Llama-4 MoE, Qwen-3-MoE, Granite-MoE, Mixtral-8x22B) look very different from the 2022 Switch Transformer / GShard era. Higher expert counts, finer granularity, shared experts, auxiliary-loss-free balancing, top-1 vs top-k routing, and the EP infrastructure that makes them servable. This page covers the algorithmic and systems details.

**Prerequisites**: [Frontier_Models_2025_2026](Frontier_Models_2025_2026.md), [Parallelism_Strategies](Parallelism_Strategies.md), [Collectives_and_NCCL](Collectives_and_NCCL.md).

---

## 1. The Modern MoE Recipe

```
For each MoE layer:
  1. Apply attention (standard).
  2. Apply MoE FFN block:
       - 1 shared expert (always active)        ← DeepSeekMoE innovation
       - N routed experts (e.g., 256)           ← much larger N than 2022
       - Each expert is FFN of dim d_expert / k ← fine-grained, smaller
       - Top-k gating (k = 1, 2, or 8)
       - Auxiliary-loss-free or sigmoid-gated
       - Output = shared_out + sum_{i ∈ top-k} g_i · expert_i(x)
```

Key knobs:
- **N** total experts: 8 (Mixtral) → 16 (Llama-4 Scout) → 128 (Llama-4 Maverick) → 256 (DeepSeek-V3).
- **k** top selected per token: 1 (Llama-4) or 2-8 (DeepSeek).
- **Shared experts**: 0 (Mixtral) or 1 (DeepSeek, Qwen-3-MoE).
- **Granularity**: each expert sized 1/k of a dense FFN to keep total active params constant.
- **Balance**: aux loss vs aux-loss-free.

---

## 2. Why More, Smaller Experts Help

Original Switch Transformer used 64 large experts with top-1 routing. DeepSeekMoE showed this is suboptimal:

- **Specialization granularity**: fewer, larger experts can only learn coarse specializations. Many smaller experts can carve up the function more precisely.
- **Knowledge sharing via shared expert**: a single shared expert absorbs general computation, freeing routed experts to specialize.
- **More routing options**: top-8 of 256 ≈ C(256,8) ≈ 4×10¹⁴ possible expert combinations vs C(8,2) ≈ 28 for Mixtral. Combinatorial expressivity.

Empirically: DeepSeekMoE matches dense quality with much less active compute. Llama-4 / Qwen-3-MoE follow the same design.

---

## 3. Auxiliary-Loss-Free Load Balancing

### 3.1 The Old Way: Auxiliary Loss

Add a term `L_aux = α · sum_e (f_e · P_e)` where `f_e` is the fraction of tokens routed to expert e and `P_e` is the average gating probability for expert e. Pushes both toward 1/N.

Problem: this loss competes with the main task loss, causing measurable quality degradation at large N.

### 3.2 DeepSeek's Aux-Loss-Free

For each expert e, maintain a bias `b_e` updated by EMA:

```
At each step:
  Compute load f_e = (tokens routed to expert e) / (total tokens)
  Update bias:  b_e ← b_e − γ · (f_e − 1/N)
  Use biased gating: g_e(x) = sigmoid(W_e · x) + b_e
```

If expert e is overloaded (f_e > 1/N), b_e decreases → fewer tokens route to it. If underloaded, b_e increases. The main task loss is unaffected.

Result: balanced load with no quality penalty. Standard in DeepSeek-V3 and emulated by other 2025 designs.

### 3.3 Sigmoid Gating

DeepSeek-V3 uses **sigmoid** gating instead of softmax:
```
g_e(x) = sigmoid(W_e · x + b_e)
top-k = experts with highest g_e(x)
weights = g_e / sum_{i ∈ top-k} g_i  # renormalize
```

Sigmoid is per-expert (no cross-expert competition during gating), making the bias trick natural. Softmax-gated would require careful renormalization after bias.

---

## 4. Top-1 vs Top-k Routing

| Choice | Used by | Pros | Cons |
|--------|---------|------|------|
| Top-1  | Switch Transformer, Llama-4 | Simplest comm (1× all-to-all per layer) | Less knowledge mixing |
| Top-2  | Mixtral | Some mixing, modest comm | Suboptimal capacity utilization |
| Top-k=8| DeepSeek-V3, Qwen-3-MoE | High mixing, expressive | More all-to-all bytes |

Tradeoff: more experts active → better capacity utilization & quality, more comm. With NVL72 and NDR/XDR fabrics, top-8 of 256 became practical at scale.

---

## 5. Group-Limited / Hierarchical Routing

DeepSeek-V3 adds a **group constraint**:
```
- Experts are partitioned into M groups (e.g., 8 groups of 32 experts each)
- For each token, top-k routing is restricted to top-G groups (e.g., G=4 groups)
- Then top-k experts within those groups
```

This bounds the **all-to-all communication scope**: a token's expert routes only span G of M groups, which can be aligned with EP-rank groups. Reduces all-to-all traffic.

Llama-4 Maverick uses a similar concept ("only most relevant experts").

---

## 6. Expert Parallelism (EP) at Scale

### 6.1 EP Layout

Place groups of experts on different GPUs:
```
EP=N: each GPU holds E/N experts.
For DeepSeek-V3 with E=256, EP=32 → 8 experts per GPU.
```

Per MoE layer:
```
1. Compute gating (replicated across all ranks)
2. all-to-all dispatch: token → assigned-expert's rank
3. Run the expert FFN locally on dispatched tokens
4. all-to-all combine: results back to original ranks
```

Two all-to-alls per MoE layer.

### 6.2 EP Bytes Math

Per layer, all-to-all sends:
```
bytes ≈ B · S · d · k / N · b
```
where B = batch, S = seq, d = hidden, k = top-k, N = EP size, b = byte size.

For DeepSeek-V3, B·S=4096, d=7168, k=8, EP=32, FP8:
```
bytes per all-to-all ≈ 4096 · 7168 · 8 / 32 · 1 = 7.3 MB per rank
```
Tiny per layer per token, but with 61 layers and 2 all-to-alls each: 61 × 2 × 7.3 MB = 890 MB per forward pass per rank. At NVLink BW 900 GB/s, ~1 ms — manageable inside NVL domain.

Outside NVL domain (NDR IB at 50 GB/s), this is much more expensive — pushing EP to stay within the NVL fabric.

### 6.3 Expert Replication for Hot Experts

Even with aux-loss-free balancing, some experts get more load than others (e.g., 1.5× average). Solutions:

- **Expert replication**: place hot experts on multiple ranks; shard their input tokens across replicas.
- **DeepSeek's redundant experts**: the architecture has a small number of duplicated experts to absorb hot-spot traffic.

### 6.4 EP × TP × DP Composition

For DeepSeek-V3 inference on 8×H100 with 256 experts:
- TP=8: sharding within attention and the shared expert.
- EP=32 (across 4 nodes of 8 GPUs): each GPU holds 8 routed experts.
- DP=1: single replica.

Or for 8 GPUs only: EP=8 (32 experts per GPU). Quality of fit depends on memory budget.

For training: TP × EP × PP × DP combine. DeepSeek-V3's 14.8T-token run used multi-axis parallelism; details in their report.

---

## 7. MoE Inference Engineering

### 7.1 Decode-Time Routing

Decode is BW-bound and per-token. Each token routes to k experts; only k expert matmuls run. The challenge: tokens in a batch route to different experts → need to **reorganize the batch** by expert.

```
1. Compute gating for batch B tokens.
2. Build per-expert token list (sorted by which expert).
3. For each expert e, do a small matmul on the tokens routed to it.
4. Scatter results back.
```

This is implemented as a **grouped GEMM** or **batched-GEMM** kernel. CUTLASS, FlashInfer, and TRT-LLM all ship optimized grouped-GEMM kernels.

### 7.2 Token Imbalance Across Experts

Within a batch of B tokens, expert i may receive 0 or 5·B/E tokens. Grouped GEMM handles ragged sizes; throughput suffers when the longest group is much longer than the median (the kernel is gated by longest).

Mitigations:
- **Capacity factor**: cap tokens per expert at C · B/E; spillover routes to backup expert. Caps the imbalance.
- **Token drop**: silently discard spillover tokens (rare; quality hit).
- **Re-route based on hot-cold cache**: complex.

### 7.3 EP All-to-All Optimization

Production engines fuse:
- **Dispatch + gating computation**: send token + dest in one message.
- **Expert FFN + combine**: overlap the combine with the next layer's prefix computation.
- **DeepEP** (DeepSeek's open-source EP comm library): hand-tuned all-to-all kernels for Hopper, achieving close-to-peak NVLink BW for MoE workloads.

### 7.4 KV Cache for MoE

Same as dense: per attention layer, store K, V. Routing only affects FFN, not attention. So MoE doesn't change KV math directly — but MLA (DeepSeek's KV compression) often pairs with MoE.

---

## 8. Frameworks and Kernels

| Component | Open-source options |
|-----------|---------------------|
| MoE forward kernel | CUTLASS grouped GEMM, FlashInfer MoE, TRT-LLM MoE plugin, vLLM Fused MoE |
| Routing / gating | Megablocks (Stanford), Tutel (MS), built-in PyTorch |
| EP all-to-all | DeepEP, NCCL all-to-all, MS-CCL+ |
| Training stack | Megatron-Core (NV), DeepSpeed-MoE, ColossalAI, Megablocks |

Production engines (vLLM, SGLang, TRT-LLM) integrate these with continuous batching.

---

## 9. Common Pitfalls

- **Picking small EP for huge expert count**: each rank holds many experts, large per-rank memory; routing imbalance hits hard.
- **EP across slow fabric**: all-to-all on NDR over multi-hop network is brutal. Keep EP in NVL domain.
- **Forgetting capacity factor**: unbounded routing causes the longest expert to dominate kernel time.
- **Mismatched EP and TP**: experts must be evenly distributable across (TP × EP).
- **Aux-loss training mismatch**: switching from aux-loss to aux-loss-free mid-train causes balance shifts.
- **MoE quality regression after FP8**: per-tile scales must be recalibrated; routed experts can be very small and harder to quantize cleanly.

---

## 10. Common Interview Questions

**Q: Why does DeepSeek use 256 small experts instead of 8 big ones?**
A: Specialization granularity — many small experts can carve up the function more precisely than a few large ones. Combined with a shared expert (always active) for general computation, the routed experts can learn niche patterns. Empirically matches or beats dense quality with less active compute.

**Q: What's auxiliary-loss-free load balancing?**
A: Per-expert bias term in the gating logits, updated by EMA: increased if expert is underloaded, decreased if overloaded. Replaces the classic auxiliary load-balancing loss, which competed with the main-task loss and degraded quality at large N.

**Q: Top-1 vs top-8 routing — what's the tradeoff?**
A: Top-1 minimizes all-to-all bytes (one expert per token), simpler kernel. Top-8 enables more expressive expert combinations, better quality, but ~k× more comm bytes. Llama-4 uses top-1; DeepSeek-V3 uses top-8.

**Q: What is a "shared expert" and why does DeepSeek use one?**
A: An expert that always runs for every token, alongside the top-k routed experts. Absorbs general computation, freeing routed experts to specialize on niche patterns. Empirically improves quality and stabilizes training.

**Q: What's the comm cost of a top-k MoE layer?**
A: Two all-to-alls per layer. Bytes per all-to-all ≈ B·S·d·k/N·b for batch B, sequence S, hidden d, top-k, EP=N, byte size b. Cap: stays manageable inside NVL domain; expensive across IB.

**Q: How does EP fit on a single 8-GPU node for DeepSeek-V3 with 256 experts?**
A: EP=8, 32 experts per GPU. The all-to-all stays within NVLink. Memory per GPU: 256/8 = 32 experts × ~5GB each (FP8) = ~160 GB? Doesn't fit. Need EP across multiple nodes (e.g., EP=32 on 4 nodes, 8 experts per GPU = ~40 GB), or use FP4, or shard inside experts with TP.

**Q: Why are MoE models cheap to compute but expensive to host?**
A: Active params per token (e.g., 37B for DeepSeek-V3) determine compute; that's small. But all 671B must reside in HBM somewhere because routing is data-dependent, so any expert might be needed → expensive memory. EP spreads the cost across many GPUs.

**Q: What is DeepEP?**
A: DeepSeek's open-source EP communication library. Hand-tuned Hopper kernels for MoE all-to-all that achieve close-to-peak NVLink bandwidth. Replaces NCCL all-to-all for MoE workloads with measurable speedup.

**Q: What's "expert imbalance" and how do you mitigate it?**
A: Some experts receive many more tokens per batch than others, slowing the grouped-GEMM kernel (gated by longest group). Mitigations: capacity factor (cap tokens per expert), expert replication for hot experts, aux-loss-free balancing during training.

**Q: How does grouped GEMM differ from regular batched GEMM?**
A: Regular batched GEMM: each "batch" is the same shape. Grouped GEMM: each group has its own shape (e.g., M_i, N, K), often varying M (tokens per expert). Implementation packs multiple GEMMs into one kernel launch with per-group offsets.

**Q: Why is MoE training harder than dense training?**
A: Routing is discrete (top-k), making gradients sparse/noisy. Load balancing requires careful tuning. Memory blows up during backward (must save all expert outputs even for tokens not routed to them, depending on impl). Communication scales differently than dense. Mitigations: aux-loss-free balancing, gradient checkpointing in experts, careful EP layout.

**Q: What's the relationship between MoE and MLA?**
A: They're orthogonal. MoE shrinks active compute (FFN only). MLA shrinks KV cache (attention storage). DeepSeek-V3 uses both — they compound: small active FLOPs and small KV bytes per token, enabling huge effective scale.

**Q: Walk me through serving DeepSeek-V3 on 8×H100.**
A: Need to fit 671B FP8 weights ≈ 670 GB across 8 × 80 GB = 640 GB. Tight. Real deployments use FP4 weights (~340 GB) or shard via TP × EP across multiple nodes. With FP4 weights on 8×H100: TP=2 attention, EP=8 across 32 routed experts per rank. Memory per rank: ~40 GB weights + activations + KV pool. Workable.

---

## 11. Further Reading

- DeepSeekMoE paper (Dai et al., 2024).
- DeepSeek-V3 technical report (2024) — aux-loss-free balancing.
- Switch Transformer (Fedus et al., 2022) — top-1 origins.
- GShard (Lepikhin et al., 2020) — top-2 origins.
- Mixtral of Experts (Mistral, 2024).
- Megablocks paper (Stanford, 2022) — block-sparse MoE kernels.
- DeepEP repository — Hopper-optimized MoE all-to-all.
- "ST-MoE: Designing Stable and Transferable Sparse Expert Models" (Zoph et al., 2022).

---

**Next:** [Long_Context_Engineering](Long_Context_Engineering.md), [Disaggregated_Serving_2025](Disaggregated_Serving_2025.md).
**See also:** [Frontier_Models_2025_2026](Frontier_Models_2025_2026.md), [Parallelism_Strategies](Parallelism_Strategies.md), [Networking_and_Interconnect](Networking_and_Interconnect.md).

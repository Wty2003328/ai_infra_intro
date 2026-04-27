# Prefill–Decode Disaggregation

Splitting prefill and decode across separate GPU pools is one of the most consequential serving architecture decisions in 2024–2026. This page explains why phase mixing hurts, what disaggregation buys, the KV transfer engineering, and the failure modes.

**Prerequisites**: [KV_Cache](KV_Cache.md), [Batching_and_Scheduling](Batching_and_Scheduling.md), [Networking_and_Interconnect](Networking_and_Interconnect.md).

---

## 1. The Argument From Bottlenecks

Prefill and decode have opposite resource profiles:

| Phase   | Compute | Bandwidth | KV growth |
|---------|---------|-----------|------------|
| Prefill | Saturates tensor cores | Low (one weight read, sequential) | Builds entire prefix cache |
| Decode  | Idle tensor cores      | Saturates HBM (re-read every step) | Adds 1 row / step |

When a single GPU pool runs both, the *workload mix* — proportion of prefill vs decode — determines where utilization lands. Bursts of long prompts spike prefill demand and stall decode. Decode-heavy moments leave tensor cores idle.

Disaggregation = give each phase a dedicated pool sized to its bottleneck.

### Concrete benefit

Coupled, mixed system on N GPUs:
```
N = N_p + N_d, but they share. Prefill pegs SMs → decode TPOT spikes.
```

Disaggregated:
```
N_p prefill GPUs sized to handle prompt FLOPs at SLO.
N_d decode GPUs sized to handle KV bandwidth at SLO.
```

Because each pool can scale to its own bottleneck, total GPUs to meet SLOs is typically **30–50% lower** than a coupled deployment under bursty workloads (Splitwise, Sarathi-Serve, NVIDIA Dynamo benchmarks).

---

## 2. Architecture

```
        ┌──────────────┐
client →│  Frontend /   │
        │   Router      │
        └─────┬─────────┘
              │
              v
   ┌────────── prefill pool (P) ──────────┐
   │  P1: GPUs   P2: GPUs   ...           │ — runs prefill, builds KV
   └──────────┬───────────────────────────┘
              │  KV transfer (NIXL / NVLink / RDMA)
              v
   ┌────────── decode pool (D) ───────────┐
   │  D1: GPUs   D2: GPUs   ...           │ — runs decode, owns KV
   └──────────┬───────────────────────────┘
              v
            client (streaming response)
```

Lifecycle of one request:

1. **Admit at frontend.** Decide which prefill replica gets the prompt (load balance + prefix-cache locality).
2. **Prefill on P.** Build the KV cache, generate the first token.
3. **Transfer KV to D.** Send all layer's K, V tensors plus state.
4. **Decode on D.** Generate subsequent tokens, stream out.
5. **Release KV** when EOS or max-tokens.

The interesting engineering is mostly in step 3 and the load balancing in steps 1, 2, 4.

---

## 3. KV Transfer Engineering

The transfer payload is the entire prefill KV cache:
```
bytes = 2 · S · L · H_kv · d · dtype
```
For Llama-3-70B, S=4K, FP16: ≈ 1.25 GB per request.

### 3.1 Transport Choices

| Transport            | Bandwidth (per direction) | Latency | When to use            |
|----------------------|---------------------------|---------|------------------------|
| Intra-node NVLink P2P| ~900 GB/s                 | < 1 µs  | P and D on same node   |
| NVL72 NVLink fabric  | ~1.8 TB/s aggregate per GPU | < 5 µs | All P and D in same NVL domain |
| InfiniBand NDR (400G) | ~50 GB/s per port         | ~5 µs  | Multi-node, RoCE/IB    |
| InfiniBand XDR (800G) | ~100 GB/s per port        | ~5 µs  | Newer multi-node       |
| TCP                  | ~10 GB/s sustained        | high   | Last resort            |

### 3.2 NIXL (NVIDIA Inference Xfer Library)

NIXL abstracts GPU↔GPU, GPU↔CPU, GPU↔storage moves with one API and selects the best transport. Used by Dynamo and increasingly by other frameworks. Lets you author the transfer without per-transport code.

### 3.3 Layer-Pipelined Transfer

Instead of "finish all prefill, then transfer all KV", overlap:

```
prefill layer 0 → start sending layer 0 KV → prefill layer 1 → start sending layer 1 KV → ...
```

Decoder begins forward pass once layer-0 KV arrives, attention proceeds layer-by-layer as KV streams in. This hides most of the transfer latency behind the prefill itself.

### 3.4 Multi-Tier Transfer

If P → D goes across nodes, KV may pass through CPU host memory (especially on systems without GPUDirect RDMA properly configured). NIXL provides a path that keeps the data in GPU memory all the way (`GPU on P → NIC on P → fabric → NIC on D → GPU on D`).

### 3.5 Tensor-Parallel Layout Translation

If P uses TP=4 and D uses TP=8, the KV is sharded differently across GPUs in each pool. The transfer must reshape:

```
P: each GPU holds 1/4 of heads
D: each GPU needs 1/8 of heads
```

Either the prefill side concatenates and re-shards, or each GPU sends its slice to two destination GPUs. Modern stacks build a logical "KV manifest" that describes where each shard lives and run a scatter-gather.

---

## 4. Sizing the Pools

Let total request rate be λ requests/s with average prompt length S_p and output length S_d.

### 4.1 Prefill Pool Capacity

Prefill FLOPs per request: roughly `2 · params · S_p` (forward pass of S_p tokens).
Prefill GPUs needed:
```
N_p = λ · 2 · params · S_p / (η_p · peak_FLOPs_per_GPU)
```
with `η_p` the achievable utilization (typically 40–60% on H100 prefill).

### 4.2 Decode Pool Capacity

Decode bytes per request: `(weights + KV(S_p+S_d/2)) · S_d` over its lifetime.
Aggregate bytes per second across active requests:
```
B/s = λ · S_d · (weights + KV(S_p + S_d/2))
```
Decode GPUs needed: `N_d = (B/s) / (η_d · peak_HBM_BW_per_GPU)`, with `η_d` ≈ 70–85%.

### 4.3 Worked Example (Llama-3-70B, FP16)

- λ = 100 req/s, S_p = 2000, S_d = 200, weights = 140 GB, KV ≈ 320 KB/token.
- Prefill FLOPs/sec = 100 · 2 · 70e9 · 2000 = 2.8e16 FLOPs/s. At 50% of 989 TFLOPS H100 peak ≈ 0.5 PFLOPS effective per GPU → 56 GPUs prefill.
- Decode bytes/sec ≈ 100 · 200 · (140 + 0.32·2100/2 GB) = 100 · 200 · 476 GB·1e-3 ≈ 9.5 PB/s? That's wrong scale, let me redo: KV per request grows from 0 to (S_p+S_d) bytes = ~756 MB → average 378 MB. Per-step bytes during decode ≈ weights + average_KV = 140 + 0.378 ≈ 140.4 GB. With B sequences in flight per decode GPU, step time ≈ (140 + B·0.378) / 3.35 TB/s. At B=300 → step ≈ (253) / 3.35 ≈ 75 ms — too slow. So pick B ≈ 64 → 41 ms step → 24 tok/s/seq → 64·24 = 1500 tok/s/GPU. Decode tok/s needed: 100·200 = 20K tok/s → ~14 GPUs.

So at this rate **prefill is the bottleneck** (~56 vs ~14 decode). Pool ratio P:D ≈ 4:1. Different workloads (chat with short prompts and long outputs) flip this — D > P.

The point of disaggregation: you pick the right ratio rather than oversizing both.

---

## 5. Routing & Locality

### 5.1 Prefix Cache Locality

If a prompt's prefix is cached on prefill replica P_i, route the request there to skip prefill. Frontends maintain an LSH or radix tree on prompt prefixes mapped to replica ids. Cache miss rate as small as 5–10% on chat workloads.

### 5.2 Decode Locality

Bind a sequence to a decode replica for its lifetime. If KV transfer cost is dominated by initial ship (one-time), no further locality concerns. Some systems re-balance on long sequences via KV migration, which is rare and expensive.

### 5.3 Multi-Tenant Quotas

Each tenant gets a portion of prefill and decode capacity. Frontend admission enforces these. Hard quotas + burst tokens common.

---

## 6. Failure & Recovery

- **Prefill replica dies during prefill**: client-visible failure; frontend retries on another replica. Idempotent because nothing has been streamed yet.
- **Decode replica dies mid-stream**: KV is lost. Two options: (a) replay from the original prompt + emitted tokens — costly because prefill repeats; (b) preemptively replicate KV to a buddy decode replica. Replication is rare in production due to overhead.
- **Network partition between P and D**: pause new requests, drain in-flight on either side. Health checks at frontend mark unreachable replicas.

---

## 7. Comparison: Coupled vs Chunked Prefill vs Disaggregated

| Approach                 | Hardware | Complexity | Best for                                     |
|--------------------------|----------|-----------|----------------------------------------------|
| Coupled (single pool)    | min      | low       | Small fleets, low traffic, similar lengths   |
| Coupled + chunked prefill| min      | medium    | Medium fleets, longer prompts                |
| Disaggregated            | more nodes (specialization) | high | Large fleets, bursty long prompts, strict SLOs |

The boundary moves with traffic. Most production stacks (Dynamo, llm-d) ship disaggregation as the default at scale; small deployments often start with chunked prefill on a single pool.

---

## 8. KV Migration Without Disaggregation

Some workloads benefit from migrating KV between same-pool GPUs:
- A long sequence may move from a hot decode replica to a cold one for load balancing.
- A multi-turn chat may migrate when the user comes back hours later (warm pool → hot pool).

Mechanisms identical to disaggregation transfer. NIXL or layer-pipelined RDMA. Adds robustness but complexity.

---

## 9. Common Pitfalls

- **Forgetting transfer time** when sizing decode SLO. KV transfer steals from TTFT budget.
- **Mismatched TP** (P uses TP=4, D uses TP=8) without re-sharding logic causes silent slowdowns.
- **Prefix cache invalidation**: if prefill replicas have non-shared caches, prefix locality breaks. Either replicate caches or use a global hash map.
- **Static pool sizing**: traffic patterns change daily. Need autoscaling tied to per-pool SLO breach.
- **Naive transport**: TCP over Ethernet between nodes destroys disaggregation's win. RDMA or bust.

---

## 10. Common Interview Questions

**Q: Why does prefill–decode disaggregation help?**
A: The two phases have opposite bottlenecks (compute vs bandwidth). On a single pool, mixing them either wastes tensor cores during decode-heavy periods or stalls decodes during prefill-heavy periods. Separate pools sized to each bottleneck achieve higher aggregate utilization.

**Q: What's the cost of disaggregation?**
A: KV transfer cost (1–10 GB/request), routing complexity, multi-pool capacity planning, more failure modes, harder to debug. Worth it once your SLO violations from interference exceed the transfer cost.

**Q: What's NIXL?**
A: NVIDIA Inference Xfer Library. A GPU-side transport that abstracts GPU↔GPU/CPU/storage moves and picks the optimal underlying transport (NVLink, RDMA, etc.). Used to ship KV between prefill and decode pools.

**Q: How do you handle TP-mismatched pools?**
A: At transfer time, reshape by gathering shards and re-sharding to the destination TP layout. Either run an all-gather on the prefill side and let the decode side scatter-pull, or build a manifest mapping each output shard to source shards and do a many-to-many transfer.

**Q: How would you size prefill vs decode pools?**
A: Prefill: aggregate prompt FLOPs/s ÷ effective per-GPU FLOPs. Decode: aggregate decode bytes/s (weights + KV) ÷ effective per-GPU HBM bandwidth. Workload analytics drive the ratio; reservations cover bursts.

**Q: What's layer-pipelined KV transfer?**
A: Stream KV from prefill to decode layer-by-layer as the prefill produces it, so the decoder can begin attending while later layers' KV is still in flight. Hides most of the transfer latency behind prefill compute.

**Q: When would you NOT disaggregate?**
A: Small fleets where the routing/transfer overhead exceeds the utilization gain; workloads with very uniform short prompts (chunked prefill suffices); when network between pools is poor (TCP only).

**Q: How do you maintain prefix cache hit rate across a prefill pool?**
A: Either route on prompt-prefix hash to a fixed replica ("affinity routing"), or replicate hot prefix entries across replicas, or use a shared external KV cache (Mooncake's approach with high-bandwidth fabric). Tradeoff: simplicity vs hit rate.

**Q: Where's the failure-tolerance boundary?**
A: Once a request enters decode, KV loss requires replay from prompt. Replication is expensive (2× decode capacity) and rarely worth it. Most systems accept a small failure-rate budget and rely on client retries.

**Q: What does "Splitwise" propose?**
A: A research design (ISCA 2024) showing that disaggregated prefill/decode across heterogeneous GPUs (A100 prefill + cheaper decode pool) reduces cost. Inspired Dynamo/llm-d design choices.

**Q: How does disaggregation interact with speculative decoding?**
A: Decode pool runs the verifier model + draft model. Speculative draft is cheap, fits in decode pool. Disaggregation doesn't really change spec-decode mechanics; it just removes prefill interference.

**Q: How do you ensure SLO when KV transfer takes 10–100 ms?**
A: TTFT budget includes (admit + prefill + transfer + first-decode-step). Transfer is part of TTFT. Make sure (a) layer-pipelined to hide most of it, (b) the decode replica isn't already saturated when KV arrives, (c) backpressure if transfer queue grows.

---

## 11. Further Reading

- Patel et al., "Splitwise: Efficient Generative LLM Inference Using Phase Splitting" (ISCA 2024).
- Holmes et al., "Sarathi-Serve" (OSDI 2024) — chunked prefill, related design space.
- NVIDIA Dynamo product blog and GTC 2024 talks.
- Mooncake paper from Moonshot (2024) — large-scale disaggregated KV pool.
- llm-d open-source project (Meta).

---

**Next:** [Speculative_Decoding](Speculative_Decoding.md).
**See also:** [Batching_and_Scheduling](Batching_and_Scheduling.md), [KV_Cache](KV_Cache.md), [Networking_and_Interconnect](Networking_and_Interconnect.md), [Inference_Frameworks](Inference_Frameworks.md).

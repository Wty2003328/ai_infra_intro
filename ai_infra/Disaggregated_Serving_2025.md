# Disaggregated Serving in 2025–2026

A frontier-focused deep dive on the four canonical disaggregation papers (Splitwise, DistServe, Sarathi-Serve, Mooncake), the production stacks that ship the ideas (NVIDIA Dynamo, llm-d, MoonCloud), and the engineering patterns — KV transfer, locality routing, KV pools, scheduler-side coordination.

**Prerequisites**: [Prefill_Decode_Disaggregation](Prefill_Decode_Disaggregation.md), [Batching_and_Scheduling](Batching_and_Scheduling.md), [Networking_and_Interconnect](Networking_and_Interconnect.md), [KV_Cache](KV_Cache.md).

---

## 1. Why The Field Converged Here

By 2024, three forces made disaggregation the default at scale:

1. **Reasoning workloads** with 10K-100K-token outputs blew up decode time. A single long-output request stalls a coupled engine for minutes.
2. **MLA / MoE architectures** put per-token KV at low cost but moved the bottleneck to FFN compute / EP all-to-all — different bottleneck per phase.
3. **Long-context prompts** (128K-1M) made prefill expensive enough that interleaving with decodes became impossible without aggressive chunking.

Disaggregation lets each phase scale to its own bottleneck and removes head-of-line blocking. By 2026, every major lab and big serving stack supports it.

---

## 2. The Four Canonical Designs

### 2.1 Splitwise (Microsoft, ISCA 2024)

- **Insight**: prefill is compute-bound, decode is memory-bound; serve them on different hardware tiers (e.g., A100s for prefill, V100s for decode) for cost efficiency.
- **Mechanism**: KV transferred from prefill GPU to decode GPU after first token.
- **Result**: 1.4× more throughput per dollar at fixed SLO.
- **Limit**: assumes static pool sizing; doesn't address dynamic workload mix.

### 2.2 DistServe (Peking + UCSD, OSDI 2024)

- **Insight**: prefill–decode co-location creates **interference** that hurts both phases' SLOs. Disaggregating eliminates the interference.
- **Mechanism**: separate prefill and decode replicas. KV transferred via RDMA. Pool sizes derived from independent SLO satisfaction at each phase.
- **Result**: 2-3× more throughput at the same SLO vs co-located baseline.
- **Influence**: most cited in production stacks; the "DistServe formulation" of disaggregation is the modern default.

### 2.3 Sarathi-Serve (Microsoft, OSDI 2024)

- **Different angle**: instead of disaggregating, eliminate interference via **chunked prefill**. Each step admits a small prefill chunk plus decodes; long prompts spread across many steps.
- **Trade-off**: simpler than full disaggregation, less hardware specialization. Achieves much of the win at lower complexity.
- **Production**: chunked prefill is now default in vLLM, SGLang, TRT-LLM. Often combined with disaggregation rather than chosen instead.

### 2.4 Mooncake (Moonshot, 2024)

- **Insight**: extend disaggregation **across the cluster** with a global KV pool. Prefill and decode replicas plus a third tier — a global KV pool — provide **cluster-scale prefix caching**.
- **Mechanism**:
  - Prefill writes KV to local + replicates hot prefixes to the global pool (CPU RAM tier across nodes).
  - Frontend router uses prefix-locality scoring across the cluster.
  - Decode replicas pull from the pool when starting a new sequence.
- **Result**: 80%+ prefix-cache hit rate at cluster scale (vs 30-50% per-replica). Massive throughput gains on chat workloads.
- **Influence**: NIXL-style transport, global KV pools now in NVIDIA Dynamo and llm-d.

---

## 3. The Production Stacks

### 3.1 NVIDIA Dynamo

NVIDIA's flagship multi-node inference fabric, GA in 2024–2025.

Components:
- **Frontend / router**: prefix-aware, multi-tenant, OpenAI-compatible.
- **Engine backend**: TRT-LLM or vLLM as the per-replica engine.
- **NIXL**: KV transfer transport, GPU↔GPU/CPU/storage.
- **Global KV pool**: optional Mooncake-style tier.
- **Disaggregated PD**: native, configurable pool sizing.
- **Autoscaling**: hooks into K8s HPA / NIM operators.

Used internally and shipped to customers. Strong on Blackwell (NVL72-aware routing).

### 3.2 llm-d (Meta, 2024–2025)

Meta's open-source disaggregated serving stack. Conceptually similar to Dynamo but vendor-agnostic.

- vLLM-based engines.
- Distributed KV pool.
- Locality-aware routing.
- K8s-native deployment.

Open-sourced in 2024; production-ready at Meta scale.

### 3.3 SGLang Disaggregated Mode

SGLang has built-in disaggregated PD support since v0.4. Combined with RadixAttention (token-level prefix sharing), excellent for chat / agentic workloads.

### 3.4 vLLM Disaggregated PD

vLLM V1's disaggregated mode integrates with NIXL. Production in 2025.

### 3.5 MoonCloud / Kimi Infrastructure

Moonshot's internal stack — Mooncake's home. Not open-source but heavily inspired the open landscape.

---

## 4. Engineering Patterns

### 4.1 KV Transfer Transport

| Transport | Bandwidth | Use case |
|-----------|-----------|----------|
| Intra-node NVLink P2P | 900 GB/s+ (Hopper) / 1.8 TB/s (Blackwell) | Same node, cheap |
| NVL72 NVLink fabric | full NVLink to any peer | All P/D in one rack |
| InfiniBand XDR (800 Gb/s) | 100 GB/s/port | Multi-node, modern |
| InfiniBand NDR (400 Gb/s) | 50 GB/s/port | Multi-node, prior gen |
| RoCE v2 over Ethernet | 50-100 GB/s/port | Spectrum-X clusters |
| TCP fallback | 10 GB/s | Don't |

NIXL chooses optimally. Production deployments verify with `NCCL_DEBUG=INFO` that the right transport is selected.

### 4.2 Layer-Pipelined Transfer

Don't wait for all layers' KV to be ready before starting transfer:

```
Prefill layer 0 → start sending layer 0 KV → prefill layer 1 → start sending layer 1 KV → …
                                              [decode replica receives layer 0]
                                              [decode begins forward pass once layer 0 ready]
```

Hides most of the transfer latency behind the prefill itself. Standard in modern stacks.

### 4.3 TP / EP Mismatch Handling

Prefill pool may use TP=8, decode pool TP=16 (or different EP layouts). KV must be re-sharded during transfer:
- Build a **manifest**: source-shard → destination-shard mapping.
- Issue many-to-many gather/scatter as part of the transfer.
- Modern implementations (Dynamo) do this transparently.

### 4.4 Locality-Aware Routing

Frontend maintains a **prefix-cache index** across replicas (often a Bloom filter or LSH for prompt-prefix hashes):
1. Compute hash of incoming prompt's prefix blocks.
2. Query the index — which replicas have these blocks cached?
3. Route to the highest-scoring replica that's not overloaded.

Refreshed periodically as caches evolve. Combine with load-aware fallback (power-of-two-choices with locality bias).

### 4.5 Pool Sizing Math

For request rate λ, average prompt length S_p, output length S_d:

**Prefill pool**:
```
GPUs_p = λ · 2 · params · S_p / (η_p · peak_FLOPs)
```

**Decode pool**:
```
GPUs_d = λ · S_d · (weights + KV(S_p + S_d/2) bytes) / (η_d · HBM_BW)
```

Scale each to its own SLO. Adjust ratio dynamically as workload shifts.

For reasoning workloads with S_d=30K, decode dominates: P:D ≈ 1:5.
For RAG with S_d=500, prefill dominates: P:D ≈ 4:1.

### 4.6 Failure Recovery

- **Prefill replica fails mid-prefill**: client-visible failure; retry on another replica.
- **Decode replica fails mid-stream**: KV is lost. Either replay (expensive prefill repeat) or accept failure with client retry.
- **Network partition between P and D**: pause new admissions, drain in-flight on each side, retry once partition heals.
- **Replication for HA**: optional, expensive (~2× decode capacity). Most production accepts a small failure-rate budget.

---

## 5. Mooncake Deep Dive

### 5.1 Architecture

```
                                ┌─── Global KV Pool ─────┐
                                │   distributed CPU RAM   │
                                │   block hash → location │
                                │   eviction: LRU        │
                                └──┬──────────────────┬──┘
                                   │                  │
                        ┌──────────┘                  └──────────┐
                        │                                        │
                        ▼                                        ▼
                ┌──────────────┐                          ┌──────────────┐
                │  Prefill     │ ──── KV transfer ────►   │  Decode      │
                │  replicas    │       (NIXL)              │  replicas    │
                └──────────────┘                          └──────────────┘
                        ▲                                        ▲
                        │                                        │
                        └─────────── Frontend Router ────────────┘
                                  (prefix-locality routing)
```

### 5.2 Block Hashing for Cluster-Wide Cache

Each KV block is hashed by `(parent_hash, token_ids_in_block)`. Hashes are registered in a distributed hash table (DHT) keyed by hash, valued by location (which replica + which physical block).

A new request:
1. Tokenize prompt; compute block hashes.
2. Query DHT for each block — found anywhere?
3. Construct a "KV manifest": for each block, source location.
4. During prefill, **skip** blocks already cached; only prefill the suffix.
5. Pull cached blocks from their hosts during decode initialization.

### 5.3 The Bandwidth Tradeoff

Cluster-wide pool means more prefix hits, but each hit may require remote fetch (CPU RAM via NIC). For chat workloads where hits are 80%+ and prompts are 10K-tokens average, this is dramatically faster than recomputing prefill.

For workloads with low prefix reuse, the overhead of DHT lookups outweighs the gains. Mooncake-style is most beneficial for chat / multi-turn / long-doc QA.

### 5.4 Open-Source Equivalents

Llm-d, Dynamo, and SGLang all have similar global-KV-pool features in some form. Mooncake itself is not open-source, but the patterns are widely adopted.

---

## 6. Multi-Tenant and Multi-Model Disaggregation

### 6.1 Per-Tenant Pools

In a multi-tenant SaaS, segregate prefill / decode pools per tenant for isolation. Costlier than sharing but eliminates noisy-neighbor.

### 6.2 Shared Pools With Priority

More common: shared pools with priority queues. High-priority tenants preempt; low-priority tolerate longer queue. Per-tenant KV-cache quotas prevent runaway sequences.

### 6.3 Multi-Model

Hosting multiple models in disaggregated mode:
- Per-model prefill replicas (model weights are big; prefill replicas pinned per model).
- Per-model decode replicas.
- Or: model-server pooling (one replica swaps models in/out — for cold-start sensitivity).

Routing decisions cascade: pick model → pick prefill replica with prefix locality → pick decode replica with KV space.

### 6.4 LoRA Disaggregation

Per-LoRA traffic can have its own decode pool to avoid LoRA-loading thrashing. Or shared pool with hot-LoRA replication. SGLang and vLLM both support multi-LoRA in a disaggregated mode.

---

## 7. SLO Math With Disaggregation

Disaggregation changes the latency budget breakdown:

```
TTFT = client_to_edge + admit + tokenize + queue_wait + prefill + KV_transfer + first_decode_step
TPOT = decode_step
```

KV transfer adds to TTFT — typically 5-50ms for typical KV sizes within NVL72; 50-200ms across IB. Layer-pipelined transfer hides most of this.

For an SLO of TTFT < 500ms:
- 50ms network + 50ms admission + 200ms prefill + 30ms transfer + 50ms first decode = 380ms ✓.
- For 1M-prompt requests: prefill alone 30s+, transfer 1-10s. TTFT inherently long; users informed.

---

## 8. Common Pitfalls

- **Too-aggressive disaggregation at small scale**: routing + transfer overhead exceeds the utilization gain below ~500 RPS.
- **Forgetting transfer in TTFT budget**: SLO breaks under load when transfer queue grows.
- **TP-mismatched pools without re-sharding**: silent perf drops or correctness bugs.
- **Static pool sizes**: workload mix changes daily; need autoscaling per pool.
- **No global cache with cluster scale**: per-replica prefix cache hit rates plateau; need DHT-style global pool to keep scaling.
- **Failed transport selection**: NIXL falling back to TCP destroys disaggregation's win. Verify in logs.
- **Replication of dead sequences**: failed replicas leaving stale KV blocks orphaned in the pool. GC required.

---

## 9. Common Interview Questions

**Q: What's the difference between DistServe-style and Splitwise-style disaggregation?**
A: Splitwise (ISCA 2024) emphasizes hardware specialization (prefill on better GPUs, decode on cheaper ones) for cost efficiency. DistServe (OSDI 2024) emphasizes interference elimination: even with same hardware, separating phases improves SLOs because they don't fight for resources. Production stacks combine both ideas.

**Q: When does Sarathi-Serve's chunked prefill beat full disaggregation?**
A: At small-to-medium scale (single replica, few hundred RPS), chunked prefill captures most of the interference-elimination win without the operational complexity of two pools and KV transfer. Full disaggregation pulls ahead when traffic is bursty, prompts are long, or cluster scale makes routing complexity worth it.

**Q: How does Mooncake achieve cluster-wide prefix caching?**
A: KV blocks hashed by `(parent_hash, tokens)` and registered in a distributed hash table. Frontend looks up incoming prompt's block hashes; cache hits reuse remote blocks via NIXL transfer. Effective cache size = sum of all replicas' KV pools, so hit rates approach 80%+ on chat workloads vs 30-50% per-replica.

**Q: What's NIXL?**
A: NVIDIA Inference Xfer Library — a unified GPU-side transport for moving KV (or any tensor) across GPU↔GPU, GPU↔CPU, GPU↔storage. Selects the optimal underlying mechanism (NVLink P2P, RDMA, TCP fallback). Used by Dynamo and increasingly by other open stacks for KV transfer in disaggregated serving.

**Q: How do you size a disaggregated PD deployment?**
A: Compute aggregate prefill FLOPs/s and aggregate decode bytes/s from the workload. Size each pool to its own bottleneck under the SLO with η = 40-60% effective utilization. Verify with load tests; autoscale per pool independently as traffic shape evolves.

**Q: How does layer-pipelined KV transfer work?**
A: Start sending layer 0's KV from prefill to decode as soon as layer 0's prefill computation finishes. While later layers prefill, earlier layers' KV is already arriving. Decode begins forward pass once enough layers have arrived. Hides most of the multi-GB transfer latency behind prefill compute.

**Q: How does disaggregation interact with reasoning workloads?**
A: Amplifies its benefits: long decodes (10K-100K tokens) interfere with prefills more in coupled mode; disaggregation cleanly separates them. Decode pool dominates capacity (>5× prefill in reasoning workloads). KV transfer cost is amortized over a long generation.

**Q: What's the failure mode when a decode replica dies mid-stream?**
A: KV is lost. Two options: (a) replay from the original prompt + emitted tokens — costly because prefill repeats, but client-correct; (b) preemptively replicate KV to a buddy replica — expensive (2× decode capacity). Production usually picks (a) and accepts the rare retry tax.

**Q: How does the frontend router handle prefix locality with multiple replicas?**
A: Maintains a prefix-cache index (Bloom filter / LSH / DHT) mapping prompt-prefix hashes to replicas that have them cached. New request: hash prompt blocks, query index, score replicas by hit count + load, route to best. Index refreshes as caches evolve.

**Q: What's the bandwidth requirement for KV transfer between disaggregated pools?**
A: Per-request transfer ≈ KV cache size for the prompt: ~1.25 GB for Llama-3-70B at 4K context FP16. At 100 RPS, that's 125 GB/s of aggregate transfer demand — easily within NVL72 NVLink fabric, marginal across NDR IB (50 GB/s/port → need multiple ports), trivial across XDR (100 GB/s).

**Q: How is TP-mismatch between prefill and decode pools handled?**
A: Each KV shard from the prefill side is mapped to potentially multiple destination shards in the decode TP layout via a manifest. The transfer kernel does scatter-gather based on this manifest. Done transparently in NIXL or by application logic in older stacks.

**Q: Where does Sarathi-Serve sit in the modern stack?**
A: Its chunked-prefill technique is now standard in vLLM, SGLang, and TRT-LLM. Even when disaggregating, chunked prefill remains useful within the prefill pool (long prompts split into chunks). The two are complementary, not alternatives.

---

## 10. Further Reading

- Patel et al., "Splitwise: Efficient Generative LLM Inference Using Phase Splitting" (ISCA 2024).
- Zhong et al., "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving" (OSDI 2024).
- Holmes et al., "Sarathi-Serve: Throughput-Latency Optimized LLM Inference" (OSDI 2024).
- Qin et al., "Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving" (Moonshot, 2024).
- NVIDIA Dynamo product blog, GTC 2024–2025 talks.
- llm-d open-source repository (Meta).
- Anyscale, MosaicML, Together AI engineering blog posts on disaggregation.

---

**Next:** [Modern_Post_Training](Modern_Post_Training.md), [Multimodal_Inference](Multimodal_Inference.md).
**See also:** [Prefill_Decode_Disaggregation](Prefill_Decode_Disaggregation.md), [Modern_KV_Compression](Modern_KV_Compression.md), [Inference_Frameworks](Inference_Frameworks.md).

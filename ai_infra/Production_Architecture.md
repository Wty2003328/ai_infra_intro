# Production Architecture

How a real LLM serving system is laid out end-to-end: from the user's HTTP request to the GPU and back. This page walks through the components, the cross-cutting concerns (caching, routing, security, cost), and the architectural choices a senior engineer makes when designing one.

**Prerequisites**: [Inference_Frameworks](Inference_Frameworks.md), [Kubernetes_and_Orchestration](Kubernetes_and_Orchestration.md), [Prefill_Decode_Disaggregation](Prefill_Decode_Disaggregation.md), [Observability_and_Debugging](Observability_and_Debugging.md).

---

## 1. Reference Architecture

```
                          ┌────────────┐
                          │   Client   │
                          └─────┬──────┘
                                │ HTTPS streaming
                                ▼
            ┌──────────────────────────────────────────────┐
            │         Edge: TLS + WAF + Auth               │
            │   (Cloudflare / nginx / Envoy / API gateway) │
            └─────┬──────────────────────────────┬─────────┘
                  │                              │
                  ▼                              ▼
          ┌─────────────────┐            ┌──────────────────┐
          │ Rate limiter    │            │ Auth / billing   │
          │ Token bucket    │            │ User → quota     │
          └─────┬───────────┘            └──────────────────┘
                │
                ▼
        ┌──────────────────────────────────────────────────┐
        │  Frontend / Router (HA replicas)                 │
        │   - parse OpenAI-style request                   │
        │   - tokenize for routing decisions               │
        │   - look up tenant SLO, model selection          │
        │   - apply moderation / safety filters            │
        │   - choose backend replica (locality-aware)      │
        └─────┬────────────────────────────────────────────┘
              │
              ▼
   ┌──────────────────────────────────────────────────────┐
   │  Inference Tier                                      │
   │                                                      │
   │   ┌─────────────────┐    ┌──────────────────┐        │
   │   │  Prefill pool   │ →  │   Decode pool    │  (PD)  │
   │   │  vLLM / TRT-LLM │    │  vLLM / TRT-LLM  │        │
   │   └─────────────────┘    └──────────────────┘        │
   │                                                      │
   │   - Continuous batching, paged KV, prefix cache      │
   │   - Multi-LoRA adapter routing                       │
   │   - Spec decoding                                    │
   │   - NIXL for KV transfer                             │
   └──────────────────────────────────────────────────────┘
              │
              ▼
   ┌──────────────────────────────────────────────────────┐
   │  Storage Tier                                        │
   │   - Model registry (S3 + local NVMe cache)           │
   │   - LoRA adapter store                               │
   │   - Cold KV cache (NVMe, multi-tenant prefix pool)   │
   │   - Logs, traces, request audit                      │
   └──────────────────────────────────────────────────────┘

   ┌──────────────────────────────────────────────────────┐
   │  Observability Tier (cross-cutting)                  │
   │   - Prometheus + Grafana                             │
   │   - DCGM, OpenTelemetry tracing                      │
   │   - Log aggregation (Loki / ELK)                     │
   │   - Alerting (PagerDuty)                             │
   └──────────────────────────────────────────────────────┘
```

This is the canonical layout for a multi-tenant LLM API service. Single-customer / on-prem deployments simplify some pieces but the structure holds.

---

## 2. Edge Layer

### 2.1 TLS Termination

TLS terminates at the edge (Cloudflare, AWS ALB, custom Envoy). Internal traffic can be mTLS or plain depending on threat model.

### 2.2 Rate Limiting

Per-tenant token bucket. Limits expressed as tokens/min, requests/min, concurrent connections. Defends against runaway clients and noisy neighbors.

### 2.3 Authentication

API keys or OIDC. Map credentials → tenant ID → SLO tier → model allowlist.

### 2.4 Web Application Firewall

For public-facing APIs. Blocks known attack patterns. Often paired with input sanitization (we don't want SQL-injection-style issues even though it's not a SQL endpoint — input length, type, encoding checks).

### 2.5 Geographic Distribution

For low TTFT to global users:
- Anycast at edge.
- Routing to nearest inference region.
- Region selection considers data residency (EU-only data stays in EU).

---

## 3. Frontend / Router

The smartest component. Sits between API and inference engines.

### 3.1 Request Normalization

Convert OpenAI / Anthropic / Bedrock / proprietary format → internal request schema. Apply default sampling parameters per model.

### 3.2 Moderation / Safety

Pre-inference: classifier checks input for policy violations (CSAM, weapons, etc.). Post-inference: classifier checks generated tokens. Streamed: incremental classification, can stop generation mid-stream.

### 3.3 Routing Decisions

For each incoming request, the router picks:
- Which model variant (e.g., latency tier vs quality tier).
- Which replica/shard (locality, load).
- Whether to disaggregate prefill (route to prefill pool, then to decode pool).

Locality-aware routing:
- **Prefix-prefix locality**: route to a replica that has the prompt prefix cached.
- **LoRA locality**: route to a replica with this LoRA loaded.
- **Sticky sessions**: same chat continues on the same replica when possible.

### 3.4 LLM Router Patterns

- **Hash-based**: hash(prompt_prefix) → replica index. Deterministic, good cache locality. Doesn't adapt to load.
- **Power-of-two-choices**: pick two random replicas, send to less-loaded. Adapts.
- **Affinity + load-aware hybrid**: prefer the prefix-locality replica, but fall over if its queue is too deep.

### 3.5 Multi-Model

If hosting many models (e.g. 20 LoRAs of one base, plus 5 base models), the router picks based on the request's `model` field plus available capacity. Hotspot models get more replicas; cold ones share.

---

## 4. Inference Tier

### 4.1 Replica Topology

A replica is a unit that hosts one model with a fixed parallelism config (e.g. 8-GPU TP=8 vLLM). Each replica serves up to its max-batch concurrent users.

For 70B at TP=8 on 8× H100: one replica per node. For 405B at TP=8×PP=2 on 16 H100s: one replica per 2 nodes. For small 7B at TP=1: 8 replicas per 8-GPU node.

### 4.2 Pool Sizes & Autoscaling

Capacity planning per pool: `replicas = peak_RPS / RPS_per_replica × safety_factor`. RPS_per_replica from load testing.

Autoscale:
- HPA on TPOT p95 or queue depth.
- Warm pool of N idle replicas to absorb step-up.
- Scale-down with hysteresis (don't flap).

### 4.3 Disaggregation

For peak workloads with strict SLOs and bursty long prompts, split into prefill pool and decode pool. See [Prefill_Decode_Disaggregation](Prefill_Decode_Disaggregation.md).

### 4.4 Per-Replica Engine Config

- **`max_num_batched_tokens`**: per-step token budget — controls prefill chunk size.
- **`max_num_seqs`**: max concurrent sequences in batch.
- **`gpu_memory_utilization`**: fraction of HBM for KV pool.
- **`speculative-decode`**: spec config.
- **`enable-prefix-caching`**: typically on.
- **TP/PP**: parallelism degrees.

Tune per workload mix; defaults are rarely optimal.

---

## 5. Storage Tier

### 5.1 Model Registry

Authoritative store of all model weights. S3 + cache hierarchy:

```
S3 (cold, durable, central)
   ↓ on first access
Per-region object store (regional cache)
   ↓ on first node access
Per-node NVMe (local cache)
   ↓ at engine startup
HBM (loaded into GPU memory)
```

Cold-start cost dominated by network pull. Pre-warm hot models on every node nightly.

### 5.2 LoRA Adapters

Tens to hundreds of MB each; many per base model. Stored in registry; loaded into engine memory either eagerly (all at startup) or lazily (on first request). Lazy loading + LRU eviction common for fleets with hundreds of LoRAs.

### 5.3 KV Cache Tiering

For prefix-cache spillover beyond HBM:
- HBM (hot)
- Host RAM (warm) — swap on preempt
- NVMe (cold) — Mooncake-style
- Object store (frozen) — multi-tenant pool of canonical prefixes

Each tier has order-of-magnitude latency / bandwidth differences.

### 5.4 Logs / Audit / Traces

Per-request log entry: timestamps, user id, model, prompt hash, output length, latency, tokens, errors. Privacy-aware (don't log raw PII without policy).

---

## 6. Cross-Cutting: Caching

Caching at every level is the only way to make frontier models economical.

### 6.1 Prompt-Level Cache

If a full request (prompt + sampling params) was seen recently and is deterministic, return the cached output. Useful for FAQ-bot workloads.

### 6.2 Prefix Cache

KV cache reuse across requests sharing a prefix. Engine-level. See [KV_Cache §5](KV_Cache.md).

### 6.3 Embedding Cache

For RAG: cache embedding(query) → don't re-embed. Hash queries for de-duplication.

### 6.4 Tokenization Cache

For very long fixed prompts (system prompts, tools), cache the tokenization to avoid CPU work.

### 6.5 Cache Invalidation

When the model or template changes:
- Bump cache key with version.
- Old caches age out via TTL.
- For prefix-block-level caches, hashes naturally diverge.

---

## 7. Cross-Cutting: Multi-Tenancy

### 7.1 Isolation

- **Compute**: dedicated replicas per tier; or shared with priority preemption.
- **Network**: separate VPCs / namespaces for sensitive customers.
- **Data**: per-tenant log buckets, encrypted with per-tenant keys.

### 7.2 Quotas

- Tokens/min, requests/min, concurrent requests, max prompt length.
- Burst allowance; refill rate.
- Hard fail vs soft degrade (queue with backpressure).

### 7.3 Fairness

Across tenants in a shared pool: weighted fair queueing on the router. Without it, one heavy tenant starves others.

### 7.4 Cost Allocation

Track tokens served per tenant. Bill on input + output tokens (typically asymmetric pricing because output is more expensive due to KV BW).

---

## 8. Cross-Cutting: Security

### 8.1 Threat Model

- **Prompt injection**: user input contains instructions trying to override system prompt.
- **Data exfiltration**: model leaks training data or other tenant's data.
- **Resource exhaustion**: very long prompts, infinite outputs.
- **Side channels**: timing or memory observation across tenants.

### 8.2 Mitigations

- Input length cap; rate limits.
- System prompt hardening; never trust user input as instructions.
- Per-tenant model instances when possible.
- Output filtering against known-bad patterns.
- Audit logs.

### 8.3 Compliance

- SOC 2, ISO 27001, HIPAA, GDPR, EU AI Act.
- Data residency: EU users → EU regions; no cross-border transfer.
- Right to deletion: per-user log purge tooling.

---

## 9. Cost Model

For an inference business, GPU is dominant. Rough Hopper economics:

- **8× H100 node** (cloud): $20–30 / hour.
- **At 80% util**: $250–375 / GPU-day.
- **Tokens/sec/H100** for 70B FP8 at decent batch: ~1500.
- **Tokens/dollar**: ~5M tokens/H100/day ÷ $300 ≈ 17K tokens/$.
- Prices for end users (e.g. $0.50–$3 per 1M output tokens) leave margin if you keep utilization high.

Cost optimization levers:
- Quantization (FP8 weights → 2× throughput).
- Spec decoding (1.5–3× throughput).
- Disaggregation (10–30% less hardware for same SLO).
- Prefix caching (variable, sometimes 2× on chat).
- Right-sized batch (TPOT vs throughput sweet spot).
- Spot capacity for non-SLO-critical tiers.

---

## 10. Deployment Patterns

### 10.1 Blue/Green

Two parallel pools (current + new). Cut traffic over instantly; rollback by re-cutting. Doubles cost during migration.

### 10.2 Canary

5% of traffic to new version, watch metrics, ramp to 100% if healthy. Standard for engine upgrades.

### 10.3 Shadow

New version runs alongside old, processing the same requests; outputs compared but not served. Quality regression detection.

### 10.4 Feature Flags

Per-tenant toggles for new engine versions, models, sampling defaults. Lets you roll out features per-customer.

---

## 11. End-to-End Latency Budget

For a "fast chat" SLO of 1s TTFT, 50ms TPOT:

```
Network (client → edge):       50 ms (variable, regional)
Edge processing:               5 ms
Auth + rate limit:             5 ms
Frontend routing + tokenize:   30 ms
Engine queue + admit:          50 ms
Prefill (1-2K tokens):         200 ms
First decode step:             50 ms
─────────────────────────
First token at client:         390 ms  (well under 1s target)
```

Per-token decode 50ms aligns with TPOT SLO. End-to-end for 200-token reply ≈ 390 + 200·50 = 10.4s. If SLO requires under 5s for 200 tokens, must improve TPOT or use spec decoding.

---

## 12. Failure Modes & Mitigations

| Failure | Symptom | Mitigation |
|---------|---------|------------|
| GPU OOM | request fails with 500 | Cap max-tokens, lower gpu_memory_util, restart on flap |
| Replica crash | requests rerouted | Liveness probe + restart; client retry |
| Model file corrupt | startup fails | Checksum verification before load |
| NCCL hang | replica unresponsive | Watchdog timeout; force restart |
| Slow prefill flooding | TPOT spike for everyone | Chunked prefill, prefill budget cap |
| Bad LoRA | quality regression | Per-LoRA shadow eval, version pinning |
| Tokenizer change | prefix cache miss | Bump cache version key |
| Network partition (PD) | requests pile in prefill | Frontend backpressure, drain |

---

## 13. Common Interview Questions

**Q: Walk me through the path of an LLM API request, end to end.**
A: Client → TLS termination at edge → auth + rate limit → frontend router (parse, tokenize, choose backend) → admission to engine → tokenization → continuous batching step (prefill or decode) → sampler → streaming response back through frontend → edge → client. Logs, metrics, and traces collected throughout.

**Q: How does the router decide which replica to send a request to?**
A: Combines (a) prefix-cache locality (route to replica with the prompt prefix already cached), (b) LoRA locality (replica with the LoRA loaded), (c) load (queue depth, TPOT), (d) tenant priority. Often a weighted scoring function over these factors.

**Q: Design an inference platform for 100K RPS across multiple models.**
A: (1) Multi-region edge with anycast TLS termination; (2) regional frontends with local prefix-cache index; (3) per-model replica pools with autoscaling; (4) disaggregated prefill/decode for the top traffic models; (5) LoRA-aware multi-tenant sharing for long-tail; (6) NVMe + S3 weight cache hierarchy; (7) Prometheus/DCGM/OTel observability. Build for 2-3× peak with spot capacity for buffer.

**Q: What's the tradeoff between disaggregated and coupled inference?**
A: Disaggregated achieves higher utilization under bursty mixed workloads at the cost of complexity and KV transfer overhead. Coupled is simpler and fine for low-traffic / similar-length workloads. Crossover usually around 1000+ RPS with bursty long prompts.

**Q: How do you handle a noisy tenant?**
A: Per-tenant rate limits at the edge, per-tenant priority on the queue, weighted fair queueing in the router, hard quota at the API layer. If a tenant repeatedly violates limits, drop their priority or reject.

**Q: What's the cold-start problem and how do you mitigate it?**
A: Loading a model from cold storage takes 1–5 minutes. Mitigations: pre-stage weights on every node's NVMe, warm pool of idle replicas, model server pooling (one process hosts several models, swap quickly), use snapshots if available.

**Q: How does a global LLM service ensure low TTFT for users worldwide?**
A: Multi-region inference with anycast routing; nearest region picked at TLS termination; replicas in each region; cross-region failover for outages; data residency rules respected (some users pinned to specific regions).

**Q: What's "shadow" deployment?**
A: New version runs alongside old, processing the same input but its output is not served — only logged and compared. Used to detect quality regressions before rolling out.

**Q: How do you bill for LLM inference?**
A: Per-token: input tokens at rate A, output tokens at rate B (B > A typically because output costs HBM bandwidth). Volume tiers, free tier, latency tiers. Track per-request token counts for billing accuracy.

**Q: What's the most expensive part of running this and how do you reduce it?**
A: GPU compute. Reduce by (a) maximizing utilization (continuous batching, prefix cache, autoscaling), (b) quantization (FP8/FP4 — 2-4× speedup), (c) speculative decoding (1.5-3×), (d) using cheaper hardware tiers for non-critical workloads, (e) disaggregation to fit each phase to its bottleneck.

**Q: How do you handle a "viral" prompt that hits 1000s of users at once?**
A: Prefix cache automatically dedupes the shared prefix → near-free decode for each user. Router clusters them on the same replica with the cache. Optionally bake the prefix into a faster path (compiled CUDA graph) if it persists.

**Q: What do you log per request?**
A: Timestamp, user id (hashed if needed), model id, prompt hash (not raw if privacy), input/output token counts, latency breakdown (queue, prefill, decode), status, error category. Sample-based raw prompt logging for debugging with strict access controls.

---

## 14. Further Reading

- Anyscale, Mosaic, OctoML, Anthropic, OpenAI engineering blog posts.
- "Site Reliability Engineering" (Google) — applicable beyond GPU systems.
- vLLM and Dynamo deployment guides.
- KServe production case studies.
- USENIX OSDI / NSDI papers on production LLM serving (Mooncake, Splitwise, Sarathi).

---

**Next:** [System_Design_Interview](System_Design_Interview.md).
**See also:** [Inference_Frameworks](Inference_Frameworks.md), [Kubernetes_and_Orchestration](Kubernetes_and_Orchestration.md), [Prefill_Decode_Disaggregation](Prefill_Decode_Disaggregation.md), [Observability_and_Debugging](Observability_and_Debugging.md).

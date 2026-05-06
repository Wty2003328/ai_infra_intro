# System Design Interview

How to approach AI-infra system design questions. Covers the framework you should run for any prompt, six fully worked example designs (LLM serving, training cluster, RAG, agent orchestrator, multi-modal pipeline, eval harness), the numbers you must memorize, and the failure modes that kill candidates.

**Prerequisites**: All preceding pages, especially [Production_Architecture](../L8_Inference_and_Serving/Production_Architecture.md), [Inference_Frameworks](../L8_Inference_and_Serving/Inference_Frameworks.md), [Distributed_Training](../L7_Training_Stack/Distributed_Training.md).

---

## 1. The 7-Step Framework

When a system design question lands, run these steps out loud.

1. **Clarify the problem.** Scope, scale, SLOs, must-haves vs nice-to-haves, multi-tenant or single, data residency. Don't skip — the actual problem is rarely what the prompt says.
2. **Estimate the load.** RPS, tokens/sec, peak vs steady, prompt/output length distributions, growth.
3. **Pick the model & hardware.** Size, precision, target tokens/sec per GPU, fleet size from capacity math.
4. **Lay out the architecture.** Edge / router / engine / storage / observability tiers. Draw it.
5. **Walk a request through.** End-to-end. Forces you to surface gaps in step 4.
6. **Discuss tradeoffs.** Why this disaggregation choice? Why this batch size? Why not the alternative?
7. **Talk about failures, scaling, and ops.** What breaks? How do you detect? How do you recover? What's the on-call life?

Most candidates rush to step 4. Strong candidates spend ~30% of the interview on 1–3.

---

## 2. Numbers You Must Have Memorized

| Quantity | Value |
|----------|-------|
| H100 FP16 peak | ~989 TFLOPS dense |
| H100 FP8 peak | ~1979 TFLOPS dense |
| H100 HBM3 BW | 3.35 TB/s |
| H100 HBM capacity | 80 GB (HBM3) / 96 GB (HBM3e) |
| H100 ridge (FP16) | ~295 FLOP/B |
| B200 HBM | 192 GB @ 8 TB/s |
| B200 FP8 | ~5 PFLOPS dense |
| NVLink 4 (Hopper) per-GPU BW | ~900 GB/s aggregate |
| NVL72 NVLink fabric | ~1.8 TB/s per GPU |
| NDR InfiniBand | 400 Gb/s = 50 GB/s |
| XDR InfiniBand | 800 Gb/s = 100 GB/s |
| PCIe Gen5 x16 | 64 GB/s |
| Llama-3-70B params | 70B → 140 GB FP16, 70 GB FP8 |
| Llama-3-70B KV/token | 320 KB (GQA H_kv=8) |
| Llama-3-8B KV/token | 128 KB |
| MFU target dense BF16 | 45–55% |
| Decode TPOT (70B FP16, batch 1) | ~42 ms (BW-bound) |
| Tokens/sec/H100 (70B FP8 decent batch) | ~1500 |
| Cost (cloud H100/hour) | $20-30/8-GPU node |

For full derivations see [Memory_Hierarchy_and_Roofline](../L3_Microarchitecture/Memory_Hierarchy_and_Roofline.md), [GPU_Architecture](../L3_Microarchitecture/GPU_Architecture.md), [Networking_and_Interconnect](../L4_Systems_and_Interconnects/Networking_and_Interconnect.md).

---

## 3. Worked Design 1: "Design an LLM API Service"

### Clarify

- Models: Llama-3 family + a few finetunes? Confirm one model or many.
- SLO: chat tier (TTFT 500ms, TPOT 50ms) and batch tier (high throughput, lax latency).
- Scale: target peak RPS, e.g., 10K.
- Multi-tenant. Per-tenant rate limits.
- Streaming responses required.
- Geographic: US + EU regions for residency.

### Estimate

- 10K RPS peak, average 200 input + 200 output tokens → 4M tokens/sec.
- Per-replica throughput: 70B FP8 ≈ 1500 tok/s → ~2700 replicas. Each replica is one 8× H100 node.
- 2700 nodes is huge → push for optimization first: prefix caching, spec decoding, smaller model variants.
- Realistic: chat workload has 60% prefix-cache hit, spec decode 2× → effective ~5000 nodes worth of work spread, ~1000 nodes is plausible.

### Hardware

- H100 8× nodes, NVL72 if available. NDR IB cross-node.
- Storage: S3 for cold weights, NVMe per node for hot, regional CDN for downloads.

### Architecture

(Cribbed from [Production_Architecture](../L8_Inference_and_Serving/Production_Architecture.md).)

- Edge: TLS, WAF, auth, rate limit, anycast routing to nearest region.
- Frontend / Router: tokenize, choose model, locality-aware backend selection.
- Inference: vLLM or TRT-LLM replicas; disaggregated PD pools above ~1000 RPS per region.
- Storage: model registry, LoRA store, log/audit pipeline.
- Observability: Prometheus + DCGM + OTel.

### Walk a request

User → TLS → auth → rate limit → router (prefix-cache lookup, choose replica) → tokenize → engine admit (chunked prefill if long) → step loop streams tokens back via SSE.

### Tradeoffs

- vLLM (broad coverage, fast features) vs TRT-LLM (lower latency, harder ops).
- Disaggregation (~30% fewer GPUs but more complex) vs coupled (simpler but higher GPU count).
- Spec decoding (2-3× throughput, requires draft model) vs none.

### Failures

- Replica crash → liveness probe restart, requests rerouted. Client sees retry.
- Hot LoRA causes thrashing → pin, replicate, or warm-pool.
- Region failover → secondary region absorbs; dataloader replication ensures no data loss.

---

## 4. Worked Design 2: "Train a 70B Model on 1024 GPUs"

### Clarify

- Goal: pretrain or fine-tune?
- Data scale: 15T tokens for full pretrain.
- Hardware: 1024 H100 across 128 nodes? Yes.
- Time budget: 30 days.
- Checkpoint cadence + storage.

### Capacity & Time

- 70B params; FLOPs/token ≈ 6 · 70e9 = 4.2e11.
- 15T tokens × 4.2e11 = 6.3e24 total FLOPs.
- 1024 H100 × 989 TFLOPS × 50% MFU = 5e17 FLOPs/sec.
- 6.3e24 / 5e17 = 1.26e7 sec ≈ 146 days. Too long.
- Either reduce tokens (7T, ~70 days), reduce model, or add hardware.
- Move to FP8 → ~30 days achievable on 1024 H100.

### Parallelism

- TP=8 within each 8-GPU node (NVLink).
- DP=128 across nodes (FSDP / ZeRO-3).
- PP=1 (model fits with TP=8 + ZeRO-3).
- Sequence parallel + selective recomputation.

### Network

- NDR IB across nodes; rail-optimized topology.
- 2 NICs per node, NCCL channels = 4-8.

### Storage

- Dataset: 15T tokens × 2 bytes ≈ 30 TB (on Lustre / WekaFS).
- Checkpoints: 1.1 TB each, hourly + on-demand. 24/day. Async DCP. 200 TB checkpoint store.

### Failures

- Per-day GPU failure rate ~1 in 1000 = 1/day expected on 1024.
- Spare nodes; restart from last checkpoint.
- Watchdog on step time to detect hangs / stragglers.

### Hyperparameters

- BF16 + FP8 forward (Transformer Engine). FP32 master weights.
- AdamW; LR cosine schedule, 4K warmup, peak 3e-4.
- Global batch 4M tokens (microbatch=2, accum=16, DP=128, seq=8192 → matches).
- Gradient clip 1.0.

---

## 5. Worked Design 3: "Build a RAG System"

### Clarify

- Corpus size: 100M docs.
- QPS target.
- Freshness: real-time or daily refresh?
- Multi-tenant per-corpus or unified?
- Latency budget for retrieval vs generation.

### Architecture

```
query → embed (1ms) → vector search (10ms) → rerank (20ms) →
        retrieve docs → prompt template → LLM (TTFT 500ms) → stream
```

- Embedding model: BGE / E5 / OpenAI text-embedding-3, on a smaller GPU pool (one A10/L4 per replica).
- Vector DB: FAISS / Pinecone / Weaviate / Milvus. HNSW for ANN.
- Reranker: cross-encoder on smaller GPU pool.
- LLM: vLLM/TRT-LLM as before.

### Capacity

- 100M docs × 1024-dim FP16 embedding = 200 GB. Sharded across vector DB nodes.
- Index update path: nightly batch reindex + incremental updates via streaming.

### Tradeoffs

- HNSW (fast, RAM-bound) vs IVF-PQ (compressed, slightly slower).
- Reranker (better quality, +20ms TTFT) vs no rerank.
- Long context (drop chunks into prompt) vs many short retrievals (cheaper).
- Generate before retrieve ("fusion-in-decoder") for some workloads.

### Caching

- Query → embedding cache (hash-based).
- Query + retrieved → answer cache (TTL).
- Document chunks → KV prefix cache in inference engine for hot docs.

### Failures

- Embedding model down → answer without retrieval (fallback prompt) or 503.
- Vector DB shard down → query other replicas; rebuild from snapshot.
- LLM slow → degrade to smaller model.

---

## 6. Worked Design 4: "Agent Orchestration Platform"

### Clarify

- Agents: tool-calling LLM loops? Multi-agent communication?
- Tool latency: ms to seconds.
- Streaming UX or batch?
- Persistence: agent state must survive restarts.

### Architecture

- LLM tier (as before).
- Tool execution tier: sandboxed runtime per call (Firecracker / Docker / V8 isolate).
- Agent state: Redis / Postgres for short term; object store for transcripts.
- Orchestrator: schedules each step in the agent loop (decide → call tool → observe → decide …).
- Concurrency: each agent run is its own state machine; many concurrent.

### Latency Budget per Step

- LLM call (500ms TTFT + 50 ms·N decode tokens).
- Tool call (10 ms to 30 s — bounded by timeout).
- Total step: 1–5s typical.

### Tool Calling Engine

- LLM emits structured tool calls (JSON schema, grammar-constrained).
- Orchestrator validates against tool registry, dispatches to tool tier, returns result to LLM context.

### Persistence

- Each agent run has a persistent context (transcript). Token-bounded; summarize old steps when context blows.
- Resumable across restarts via state checkpoint.

### Tradeoffs

- ReAct-style loop (LLM-driven control) vs rigid DAG (predictable, less flexible).
- Single-LLM agents vs multi-agent (more capable, harder to debug, ratelimit risk).
- Synchronous tool calls vs async (parallelize independent ones).

### Failures

- Tool timeout → return error to LLM, let it decide next step.
- LLM safety classifier blocks → ask user / abort.
- Infinite loop (LLM keeps calling same tool) → step counter + human-in-loop escalation.

---

## 7. Worked Design 5: "Multi-Modal Inference Pipeline"

### Clarify

- Input modalities: image, audio, video?
- Output: text only or also audio/image?
- Model: VLM (single model) or pipeline (encoder + LLM)?
- Latency budget.

### Architecture

```
Input image → ViT encoder → image tokens →
                                          --> LLM context --> text out
Input text → tokenizer → text tokens   →
```

For VLM (Llava-style): vision encoder runs on the image, projects into LLM token space, prepended to text.

### Capacity

- Image encode: ViT-Large ≈ 30 GFLOPs/image. Cheap on a small GPU.
- LLM: same as before, but context is longer (image tokens).
- KV cache larger because of image tokens.

### Caching

- Cache encoded images (image hash → image tokens) — frequent re-uploads of same images cheap.
- Prefix cache the image-prefix in KV.

### Streaming

- Image arrives → encode → kick off LLM with image prefix while still uploading? Possible if streaming upload is supported.

---

## 8. Worked Design 6: "Eval Harness for LLM Quality"

### Clarify

- Benchmarks: MMLU, GSM8K, HumanEval, MT-Bench, custom.
- Continuous (per commit) or periodic.
- Scale: thousands to millions of eval examples.
- Reproducibility requirements.

### Architecture

```
Eval runner →
   - Loads model under test (vLLM/TRT-LLM).
   - For each benchmark dataset:
        - Streams prompts.
        - Records completions.
        - Scores via reference / judge.
   - Aggregates scores; logs to dashboard.
```

### Capacity

- 1M eval prompts × 200 token output × 50ms TPOT = 10K hours of decode = 1.4M GPU-hours? No — parallelize across replicas.
- 64 replicas × 1500 tok/s = 96K tok/s; 1M·200 tokens / 96K ≈ 35 min. Feasible.

### Determinism

- Fix sampling: temperature=0 or seed-pinned.
- Pin model checkpoint, tokenizer, engine version.
- Per-eval cache results so re-runs are cheap.

### Judge

- For open-ended outputs: LLM-as-judge with another model. Risks (judge bias) → use multiple judges, human spot-check.
- For closed-ended: exact match, F1, BLEU, pass@k.

### Tradeoffs

- Greedy vs sample-N: sample-N catches stochastic failures but expensive.
- Judge model size: smaller judge cheaper but less reliable.
- Online (continuous) vs offline (commit-triggered): online catches regressions sooner.

### Failures

- Eval engine OOM on long prompts → cap input length.
- Judge LLM disagreement on borderline answers → human review pipeline.
- Score drift after engine upgrade → bisect.

---

## 9. Common Tradeoff Axes

These appear in every design.

| Axis | Sketch |
|------|--------|
| Latency vs throughput | Smaller batch → lower latency; bigger batch → higher throughput. Disaggregation breaks the conflict. |
| Quality vs cost | Bigger model = better answers + 5× more $. Quantization + distillation reduce gap. |
| Generality vs efficiency | One model for all vs specialized models per task. Most products land on a small palette of tier-sized models. |
| Build vs buy | Run vLLM yourself vs pay for managed (Bedrock, Together, Anyscale). Buy until hitting scale or compliance pain. |
| Open-source vs closed | Llama-3 / Qwen / DeepSeek vs GPT/Claude. Open gives control + cost; closed gives quality + ops simplicity. |
| Sync vs async | Streaming is sync (kept connection); batch/agent flows async. Streams are state-heavy. |
| Stateless vs stateful | Engines are stateful (KV). Frontends/routers should be stateless for HA. |

---

## 10. Failure Modes That Kill Candidates

- **Jumping to the architecture** without clarifying scale, SLO, multi-tenancy.
- **Vague capacity numbers** ("a lot of GPUs") instead of math.
- **Ignoring failures** — assuming everything works, no replication, no checkpoint.
- **Forgetting observability** — "how would you know if it's broken?"
- **Hand-waving tradeoffs** — "we'd use vLLM" without saying why over TRT-LLM.
- **Designing for max scale on day one** — over-engineering. Mention "start with X, evolve to Y at Z scale."
- **Mixing layers** — putting load balancing inside the engine, putting model logic in the router.
- **Not computing SLO budget** — answering "TPOT 50ms" without checking if your batch size + KV math allows it.

---

## 11. Strong Closing Habits

- Restate the SLOs you're hitting.
- Sketch the next 3–6 month roadmap (autoscaling improvements, FP4 rollout, regional expansion).
- Acknowledge unknowns ("if traffic is spikier than estimated we'd need …").
- Ask the interviewer about a specific area you want to deepen.

---

## 12. Practice Prompts

Try these in 45 minutes each:

1. Design a system to serve a 405B MoE model with 10K concurrent users and TTFT < 1s.
2. Train a 13B model on 64 H100s in a research lab; design the cluster.
3. Build a real-time voice agent (audio-in, audio-out, < 500ms turn latency).
4. Design a structured-output JSON-extraction service over a 70B model with strict schema.
5. Design a fine-tuning service that lets users upload data, kicks off LoRA training, deploys for inference.
6. Multi-region deployment with sub-200ms TTFT for users in NA/EU/Asia.
7. RAG over 10B documents, refreshed hourly.
8. RLHF training pipeline with online sampling for GRPO at the scale of DeepSeek.

For each: run the 7-step framework, write down the numbers, draw the boxes, talk through failures.

---

## 13. Common Interview Questions

**Q: An interviewer asks "how would you design ChatGPT-style serving?" — what do you do first?**
A: Clarify scope: what scale? Single model or family? SLO targets? Multi-tenant? Then estimate load (RPS, tokens/sec). Only then start drawing components.

**Q: How do you decide between vLLM, TRT-LLM, and a custom engine?**
A: Off the shelf wins until proven otherwise. vLLM for general purpose and rapid feature uptake. TRT-LLM when you need NVIDIA peak performance and have a build pipeline. Custom only when you have a very specific bottleneck the OSS engines don't address. Mention you'd benchmark before committing.

**Q: Walk me through the math: 1000 RPS of 70B FP8 inference on H100.**
A: Average 200 input + 200 output tokens. Throughput per H100 ~1500 tok/s. 1000 RPS × 400 = 400K tok/s. 400K / 1500 = 267 H100s. Round up for headroom (~300, ~38 nodes of 8 GPUs).

**Q: How would you make this 50% cheaper?**
A: (1) Spec decoding 2-3× → halve. (2) FP4 / MXFP weights → another 30% if available. (3) Prefix caching at scale (~30% if chat-heavy). (4) Use lower-tier hardware for non-SLO traffic. (5) Disaggregated PD for utilization.

**Q: How do you pick a model parallelism layout?**
A: Start with constraints: model size, NVLink domain, network. Try TP up to NVLink size first. Add ZeRO-3 (FSDP) for DP-shardable memory. PP only if the above don't fit. EP only for MoE. Verify with capacity math before committing.

**Q: How do you decide chunked prefill chunk size?**
A: Start at 2048; if TPOT during long-prompt admission spikes, lower to 512–1024. If TTFT inflated, raise. Workload-dependent — measure on realistic traces.

**Q: How would you migrate from FP16 to FP8 in production?**
A: Stage: (1) Calibrate on a representative dataset; (2) shadow deploy FP8 alongside FP16, compare outputs and metrics on canary traffic; (3) ramp gradually with feature flag; (4) full cutover; (5) keep FP16 fallback for one rollback window.

**Q: How do you prevent one tenant from starving others?**
A: Per-tenant rate limits at the edge; weighted fair queueing in the router; priority classes per SLO tier; dedicated replicas for top tiers. At engine layer, admission control respects these signals.

**Q: How do you handle a sudden 10× traffic spike?**
A: Warm pool absorbs the immediate burst; HPA scales out; rate-limit excess back to clients with 429 + retry-after; degrade non-critical paths (turn off spec decode if it doesn't help; use smaller model for batch tier). Post-incident: increase warm pool baseline and HPA aggressiveness.

**Q: What's the biggest reliability risk in an LLM service?**
A: A few candidates: cascading failures from a hot LoRA / hot prompt; silent quality regressions after upgrades; KV pool exhaustion under burst; cross-tenant noisy neighbor. Defenses: rate limits, shadow tests, observability, isolation.

**Q: How do you balance prefill and decode pool sizes?**
A: Compute per-second prefill FLOPs vs per-second decode bytes for the workload mix. Size each pool to its bottleneck. Re-tune monthly as traffic shape evolves; autoscaling smooths short-term fluctuation.

**Q: How does the design change for batch (offline) vs online inference?**
A: Batch: maximize throughput, longer chunks, no SLO, reuse GPU 100%. Online: respect TTFT/TPOT, smaller chunks, autoscale on latency. They can share hardware via mixed pools with priority preemption.

---

## 14. Further Reading

- "Designing Data-Intensive Applications" (Kleppmann) — system thinking applies broadly.
- Production Engineering blogs from Meta, OpenAI, Anthropic, Anyscale, Mosaic.
- USENIX OSDI / NSDI papers — Mooncake, Splitwise, Sarathi, Orca, vLLM, RadixAttention.
- "AI Infrastructure for Anyone" — MosaicML.
- Various GTC and AI Engineer Summit talks.

---

**Next:** [Common_Interview_Questions](Common_Interview_Questions.md).
**See also:** [Production_Architecture](../L8_Inference_and_Serving/Production_Architecture.md), [Inference_Frameworks](../L8_Inference_and_Serving/Inference_Frameworks.md), [Distributed_Training](../L7_Training_Stack/Distributed_Training.md).

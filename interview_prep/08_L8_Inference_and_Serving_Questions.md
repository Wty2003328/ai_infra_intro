# L8 — Inference and Serving: Interview Questions

Consolidated interview Q&A and worked problems from every page in `L8_Inference_and_Serving/`, moved here to keep topic pages focused on concepts. Each section links back to its source page for full context.

---

## Agentic Inference — Serving Tool-Calling, Multi-Turn, Long-Horizon Workloads

*From [Agentic_Inference.md](../L8_Inference_and_Serving/11_Agentic_Inference.md)*

- **"How is agent serving different from chat?"** → sessions not requests: monotone context, 10–100:1 input:output, per-step SLO compounding, KV reuse as the #1 cost lever, task-level admission control.
- **"Design the KV story."** → paged KV + radix prefix cache → cache-aware routing (overlap−load score) → async tiered offload (DRAM/NVMe) sized so restore hides in tool latency → compaction when context approaches the limit.
- **"Why Rubin CPX / TPU 8i?"** → agentic steps are incremental-prefill-heavy + small-decode; prefill wants cheap dense FLOPs (GDDR7 CPX), decode wants KV residency (HBM/big SRAM). Hardware now mirrors the disaggregation software already proved.
- **"Structured output at scale?"** → grammar → compressed token FSM, compiled per schema-hash off the hot path, masks overlapped with forward pass, jump-forward through forced tokens.

---

## Batching and Scheduling — Continuous Batching, Chunked Prefill, and Admission Control

*From [Batching_and_Scheduling.md](../L8_Inference_and_Serving/03_Batching_and_Scheduling.md)*

**Q: Static vs continuous batching -- what is the difference and why does it matter?**
A: Static batching forms a fixed cohort of requests, pads to max length, runs until the slowest finishes, then returns all results together. Continuous batching re-selects participants every step: sequences join when admitted and leave when finished without disrupting others. Under realistic length variance, continuous batching achieves 2--10$\times$ higher throughput by eliminating padding waste and head-of-line blocking.

**Q: What is prefill--decode interference?**
A: Prefill is compute-bound (tensor cores saturated); decode is bandwidth-bound (HBM streaming). Mixing a large prefill with decodes in one step makes the step take the prefill's compute time, causing a TPOT spike for every active decode sequence. Mitigated by chunked prefill, disaggregation, or prefill budget caps.

**Q: How does chunked prefill work and what does it cost?**
A: A prompt of length $S$ is split into chunks of $C$ tokens (typically 512--2048). Each chunk is a small prefill processed in its own step, interleaved with ongoing decodes. Total FLOPs are identical to a single-pass prefill (chunking spreads but does not add work). The cost is increased TTFT (the prompt is processed across $\lceil S/C \rceil$ steps instead of one) traded for dramatically smoother TPOT.

**Q: How would you determine max-batch for a 70B model on 8$\times$ H100?**
A: Compute TPOT$(B)$ from the decode-step HBM-read model: $(N \cdot b_w + B \cdot \bar{S} \cdot b_{kv}) / (\text{aggregate } \beta)$. Solve for $B$ where TPOT = TPOT\_SLO. Then check KV capacity: $B_{\max} = M_{\text{KV}} / (\bar{S} \cdot b_{kv})$. The binding constraint is the smaller of the two. With 8$\times$ H100, aggregate bandwidth is ~27 TB/s; for TPOT SLO of 50ms, the TPOT constraint gives $B \approx 600$, but KV capacity at $\bar{S} = 4K$ gives $B \approx 90$ on 80 GB GPUs. KV capacity binds.

**Q: What is preemption and when do you choose recompute vs swap?**
A: Preemption evicts an active sequence's KV blocks to free memory for higher-priority requests. Recompute drops the blocks; on resume, the prompt is re-prefilled (cheap if prefix cache hits). Swap copies blocks to host RAM via PCIe and restores them on resume (cheaper for long sequences with valuable KV). Default heuristic: recompute for new requests that haven't started decoding; swap for in-flight decode sequences.

**Q: How does admission control work?**
A: At each step, the scheduler evaluates whether a waiting request can be admitted by checking: (1) sufficient KV pool capacity for the prompt + expected output, (2) room in the per-step token budget for at least one prefill chunk, and (3) the TTFT deadline is achievable given current queue depth. If any check fails, the request is rejected (429), queued, or degraded.

**Q: Why is TPOT nearly flat as batch grows in the bandwidth-bound regime?**
A: Decode reads model weights once per step (fixed cost $\propto N$) plus per-sequence KV (grows linearly with $B$). At small $B$, weights dominate the read. Adding sequences adds KV bytes that are small relative to weights, so step time grows slowly. Batching is "nearly free" until KV reads approach the weight read volume.

**Q: What metrics belong on a serving dashboard?**
A: TTFT p50/p95/p99, TPOT p50/p95/p99, end-to-end latency by output-length bucket, throughput (req/s and tok/s), KV cache occupancy %, prefix cache hit rate, queue depth, admission rate, preemption rate, per-step token budget utilization, GPU SM utilization, HBM bandwidth utilization.

**Q: How would you handle a noisy neighbor sending 32K-prompt requests?**
A: (1) Cap per-step prefill budget so the prompt spreads across many steps. (2) Assign lower priority tier. (3) Apply per-tenant KV quota so one tenant cannot consume more than their share. (4) Isolate prefill to dedicated GPUs (disaggregation). (5) Rate-limit the tenant.

**Q: How does speculative decoding interact with scheduling?**
A: Each sequence in a spec-step produces 1--$K+1$ accepted tokens, creating a ragged batch. The scheduler must account for variable token counts in the per-step budget and handle variable KV write sizes. Prefill sequences are unaffected. Production schedulers fold speculation into the same step loop with dynamic budget adjustment.

---

## GPU Multi-Tenancy — MIG, MPS, Time-Slicing, and Fractional GPUs

*From [GPU_Multitenancy.md](../L8_Inference_and_Serving/12_GPU_Multitenancy.md)*

- **"MIG vs MPS?"** → MIG: hardware partition (SMs+L2+HBM BW+faults), 7 slices max, true QoS, coarse and static. MPS: concurrent kernels in shared context, soft SM caps, no memory/BW/fault isolation, maximal density.
- **"Can two MIG slices serve one model with TP?"** → effectively no — slices are isolated devices without NVLink peering as one domain; TP wants whole GPUs in one NVLink domain.
- **"Fractional GPUs on K8s?"** → three real backings: time-slice (fiction), MPS (soft), MIG (hard); DRA makes the claims expressive but the isolation is whatever the backing gives.
- **"Latency SLO + batch on one GPU?"** → MIG fence (or green contexts within one engine), batch under preemption, never naked MPS — bandwidth interference hits ITL directly.

---

## Inference Frameworks — vLLM, SGLang, TensorRT-LLM, and Beyond

*From [Inference_Frameworks.md](../L8_Inference_and_Serving/08_Inference_Frameworks.md)*

**Q: Compare vLLM and TensorRT-LLM. When would you pick each?**

A: vLLM is open-source Python+CUDA with the broadest model coverage, fast community-driven feature uptake, and simple deployment. TRT-LLM is NVIDIA's compiled C++ engine with the best peak latency and throughput on NVIDIA GPUs but requires per-configuration build pipelines and lags on new architectures. Pick vLLM for general-purpose serving and rapid iteration; TRT-LLM for latency-critical NVIDIA-only deployments where you can amortize the build cost.

**Q: What is RadixAttention and why does it matter?**

A: A token-granularity prefix-sharing scheme implemented as a radix tree over KV cache blocks. Unlike hash-based prefix caching (block-aligned, no sub-block sharing), RadixAttention matches arbitrary-length prefixes and supports branching (multiple requests sharing a common prefix then diverging). It produces 10–30 percentage-point higher cache hit rates on chat and agentic workloads.

**Q: How does PagedAttention enable continuous batching?**

A: Each sequence has an independent block table over a global block pool. Sequences of different lengths grow independently without padding. The scheduler admits and evicts sequences without reshaping tensors. Without paging, ragged batching requires expensive padding or per-request memory regions.

**Q: What is NVIDIA Dynamo's role in the inference stack?**

A: Dynamo 1.0 (successor to Triton Inference Server for LLM serving) is a multi-node orchestration layer built in Rust + Python, not a per-GPU engine. It disaggregates prefill and decode across separate GPU pools, manages KV transfer with tiered eviction (GPU → CPU → SSD → remote storage) via NIXL, routes requests with prefix-locality awareness, and uses SGLang, TRT-LLM, or vLLM as pluggable engine backends. Key additions in 1.0 include the Planner (SLA-driven autoscaler), Grove (K8s operator for NVL72 topology-aware scheduling), ModelExpress (7x faster cold starts), and AIConfigurator (automated deployment configuration). It sits above the engine in the stack hierarchy.

**Q: What does "structured output" mean and how is it implemented?**

A: Constraining generation to match a JSON schema, regex, or grammar. Implementation: at each decode step, a logit mask sets the probability of tokens that would violate the constraint to $-\infty$. Libraries: Outlines, xgrammar, lm-format-enforcer. The per-step masking cost is amortized by GPU-parallel constraint evaluation.

**Q: Why is multi-LoRA serving non-trivial?**

A: Each LoRA adapter modifies projection weights differently. Naive per-request weight loading is slow. Fused multi-LoRA kernels (Punica, S-LoRA) handle a batch with mixed adapter IDs in one matmul by routing rows to the correct adapter weights. The engine must track adapter IDs per row and manage adapter caching.

**Q: How does the inference engine handle EOS?**

A: The sampler outputs the EOS token ID; the scheduler marks the sequence finished, emits the final output to the client, decrements refcounts on shared prefix blocks, frees the sequence's exclusive blocks, and opens the slot for a new request from the waiting queue.

**Q: What is the build-vs-interpret tradeoff in inference engines?**

A: Compiled engines (TRT-LLM) fuse operators and auto-tune kernels at build time, yielding 5–15% better peak performance. The cost is per-configuration build time (minutes to hours), operational complexity (engine versioning, CI), and slower support for new models. Interpreted engines (vLLM, SGLang) load weights dynamically, deploy in seconds, and iterate faster at the cost of slightly lower peak performance.

**Q: How would you debug a sudden TPOT regression in production?**

A: (1) Check if input distribution changed (longer prompts, bigger batch). (2) Inspect KV occupancy and prefix-cache hit rate. (3) GPU utilization from DCGM or Nsight — is the GPU compute-bound or memory-bound? (4) Compare engine version against the known-good baseline. (5) Check NCCL bus bandwidth (inter-node degradation). (6) Look for failure-mode regressions (NaN handling, synchronization barriers, sampling-path changes).

**Q: What is BitNet and how does it differ from quantization-based approaches?**

A: BitNet uses ternary weights ($\{-1, 0, +1\}$) trained natively from scratch rather than post-training quantization of a floating-point model. This eliminates almost all floating-point multiplications — matmul becomes conditional addition/subtraction. The result is 1.37x–6.17x CPU speedup and 55–82% energy reduction versus FP16. However, only BitNet-trained models work; you cannot run standard Llama or Qwen models. The quality gap with FP16 is measurable but narrows at scale. BitNet is best for CPU-only, edge, or power-constrained deployments where GPU inference is not feasible.

**Q: When would you NOT use a data-center inference framework?**

A: Single-user local chat (llama.cpp, MLC). Edge or mobile deployment (MLC-LLM compiled for the target). CPU-only or power-constrained deployment (BitNet/bitnet.cpp runs 100B models on a single CPU at 5–7 tok/s with 55–82% energy reduction). Apple Silicon (SGLang's MLX backend). Research requiring full execution control (raw PyTorch with custom kernels). CPU-only inference where the overhead of GPU-oriented frameworks is wasted.

---

## Prefill/Decode Disaggregation — Different Rooflines, Different Hardware

*From [Prefill_Decode_Disaggregation.md](../L8_Inference_and_Serving/05_Prefill_Decode_Disaggregation.md)*

**Q: Why does prefill/decode disaggregation improve throughput?**
A: Prefill is compute-bound (arithmetic intensity above the roofline ridge point), decode is memory-bandwidth-bound (arithmetic intensity far below). On a single pool, mixing them wastes tensor cores during decode and stalls decodes during prefill. Separate pools let each operate at its own optimum.

**Q: What is the cost of disaggregation?**
A: KV transfer latency (0.5–25 ms depending on fabric), routing complexity, two capacity-planning problems instead of one, more failure domains, and a minimum viable scale of ~4 GPUs. The benefit must exceed these costs, which it does at scale with bursty workloads.

**Q: How do you size the prefill and decode pools?**
A: Prefill: $\lambda \cdot 2N \cdot S_p / (\eta_p \cdot \pi)$. Decode: $\lambda \cdot S_d \cdot \bar{Q}_d / (\eta_d \cdot \beta)$. The ratio depends entirely on the workload: short-prompt long-output (reasoning) is decode-heavy; long-prompt short-output (RAG) is prefill-heavy.

**Q: What is NIXL?**
A: NVIDIA Inference Xfer Library. A unified API for GPU-to-GPU/CPU/storage data movement that selects the optimal transport (NVLink, GPUDirect RDMA, PCIe) at runtime. Used to ship KV between pools without per-transport code.

**Q: How does layer-pipelined KV transfer work?**
A: Transfer each layer's KV to the decode side as soon as prefill of that layer completes, overlapping transfer of layer $l$ with compute of layer $l+1$. The decoder starts its forward pass as each layer's KV arrives, hiding nearly all transfer latency behind prefill compute.

**Q: When would you NOT disaggregate?**
A: Small fleets (< 4 GPUs), uniform short prompts (chunked prefill suffices), poor network fabric (TCP only), or applications requiring sub-10ms TTFT where even 1 ms of transfer overhead is unacceptable.

**Q: How does disaggregation interact with speculative decoding?**
A: Orthogonally. The decode pool runs both draft and verifier models. Disaggregation removes prefill interference from the decode pool but does not change speculative decoding mechanics. They compose well.

**Q: What happens when a decode instance fails mid-generation?**
A: KV is lost. The client sees a truncated response. Recovery requires re-prefilling from the original prompt. Most production systems accept this loss rate and rely on client retries. Buddy replication doubles cost and is rarely justified.

**Q: How do you maintain prefix cache hit rate across a prefill pool?**
A: Affinity routing (hash the prefix to a fixed instance), replicate hot prefixes across instances, or use a global KV pool (Mooncake approach). Tradeoff: simplicity vs. hit rate vs. memory overhead.

**Q: What does Splitwise propose?**
A: A research design (ISCA 2024) showing that disaggregated prefill/decode across heterogeneous GPUs (high-$\pi$ for prefill, high-$\beta$ or cheaper for decode) reduces cost-per-token by 1.4–1.8×. The key insight is matching GPU capabilities to phase bottlenecks.

---

## Production Architecture — Reference Stack, Capacity Planning, and Failure Modes

*From [Production_Architecture.md](../L8_Inference_and_Serving/15_Production_Architecture.md)*

**Q: Design an inference platform for 100K RPS serving 70B.**

A: 70B FP8 at TP=8 = ~60 RPS/replica. Need ~1 700 replicas = 13 600 H100s across 3 regions (~5 700 each). Disaggregated PD for top-traffic region. Edge: anycast TLS, per-tenant rate limiting. Frontend routers: ~200 instances. KV pool: HBM + RAM + NVMe tiering. Autoscaling with 20% warm buffer. Cost at $3.50/GPU-hr: ~$47K/hr (~$1.1M/day).

**Q: GPU OOM just fired. Walk through your response.**

A: (1) Identify replica from alert. (2) Check `kv_cache_utilization` — if > 95%, KV is the cause. (3) Immediate: reduce `max_num_seqs` 25%. (4) Check for tenant sending unusually long prompts. (5) Apply per-tenant `max_model_len` cap if needed. (6) Medium-term: enable KV offloading or add replicas. (7) Post-incident: update capacity model.

**Q: Rollback with zero downtime?**

A: Blue/green with feature flags. New version deploys to green pool while blue serves. Flip flag 0% → 5% (canary) → 100%. Regression at any step: flip back to 0% (instant). No downtime.

**Q: Most cost-efficient 405B serving?**

A: FP8 (2x over BF16), speculative decoding with 70B draft (~2x), PD disaggregation (30% less HW), prefix caching. Combined ~4x throughput: ~$3.60/Mtok → ~$0.90/Mtok on B200.

**Q: Prevent noisy tenant from degrading SLO for others?**

A: (1) Edge rate limiting: per-tenant tokens/min cap, 429 on excess. (2) Router: weighted fair queueing proportional to tier. (3) Engine: priority preemption — low-priority sequences evicted for high-priority arrivals.


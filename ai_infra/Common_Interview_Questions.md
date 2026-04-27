# Common Interview Questions

A consolidated bank of the most frequently asked AI infrastructure interview questions, organized by topic, with crisp model answers. Use this as a final cram sheet — the deeper context lives in the linked pages.

**Categories:**
1. GPU Architecture & Memory
2. CUDA / Kernels
3. Attention & Transformers
4. KV Cache
5. Batching, Scheduling, Disaggregation
6. Quantization
7. Distributed Training
8. Inference Frameworks
9. Networking & Collectives
10. Observability & Reliability
11. Big-Picture / Behavioral

---

## 1. GPU Architecture & Memory

**Q: Why is decode bandwidth-bound and prefill compute-bound?**
A: Decode reads the entire model weights and KV cache once per token but does only `O(d)` FLOPs per token per layer — arithmetic intensity is low (~tens of FLOP/B), well below H100's ridge of 295. Prefill processes many tokens per step, AI scales with batch size, easily exceeding the ridge → compute-bound. See [Memory_Hierarchy_and_Roofline](Memory_Hierarchy_and_Roofline.md).

**Q: What is "arithmetic intensity" and what is the ridge point on an H100?**
A: AI = FLOPs / bytes for a kernel. Ridge = peak_FLOPS / peak_BW. H100 FP16: 989 TFLOPS / 3.35 TB/s ≈ 295 FLOP/B. Below ridge → memory-bound; above → compute-bound.

**Q: Compare HBM3 to HBM3e to GDDR.**
A: HBM stacks DRAM dies vertically with a wide interface (1024-bit), giving very high BW (TB/s). HBM3e is a refresh with higher per-pin rates (B200: 192 GB at 8 TB/s). GDDR (gaming GPUs) is cheaper but ~10× lower BW per chip.

**Q: How big is L2 on H100? What's it for?**
A: 50 MB. Shared across all SMs. Caches HBM reads, helps when multiple SMs touch the same data (e.g., shared weights across thread blocks).

**Q: What is shared memory? How big is it on H100?**
A: SM-local SRAM, manually managed, ~228 KB per block. Used for tile staging in matmul / FA-style kernels. ~10× lower latency, ~10× higher BW than L2.

**Q: What's the difference between a thread, warp, block, and grid?**
A: Thread: scalar execution unit. Warp: 32 threads executed in lockstep (SIMT). Block: ≤1024 threads, share SMEM, scheduled together on one SM. Grid: collection of blocks for one kernel launch. See [CUDA_Programming](CUDA_Programming.md).

**Q: What is occupancy? What limits it?**
A: Active warps per SM divided by max warps per SM. Limited by registers per thread, SMEM per block, and threads per block. Higher occupancy hides latency but isn't always optimal.

---

## 2. CUDA / Kernels

**Q: How do you achieve coalesced memory access?**
A: Threads in a warp access consecutive 128-byte aligned addresses. Pattern `data[threadIdx.x]` is coalesced; `data[threadIdx.x * stride]` is not.

**Q: What is a bank conflict?**
A: SMEM is split into 32 banks; if two threads in a warp access different addresses in the same bank, the access serializes. Avoid via padding (`SMEM[33]` instead of `SMEM[32]`) or swizzling.

**Q: What's TMA and why does it matter?**
A: Tensor Memory Accelerator on Hopper. Async bulk tile transfer from HBM to SMEM with one descriptor — frees registers used for address computation in pre-Hopper LDGSTS, and lets the producer-consumer pattern overlap loads with matmul.

**Q: Difference between `__syncthreads`, `__syncwarp`, and `__threadfence`.**
A: `__syncthreads()` synchronizes all threads in a block. `__syncwarp()` synchronizes 32-thread warp (with mask). `__threadfence()` orders memory operations globally without synchronizing.

**Q: When would you use Triton instead of CUDA?**
A: When the kernel is moderate complexity (matmul, softmax, attention variants) and you want autotuning + Python iteration speed without sacrificing too much performance. CUDA/CUTLASS for the absolute last 10–20% on Hopper-specific features.

**Q: What is warp specialization?**
A: Different warps in a threadblock take different roles — e.g., one group of warps loads (producer) while another computes (consumer). Lets you build a deeper pipeline. Standard in FA-v3.

**Q: Why does a tiled matmul fit in shared memory?**
A: A tile of size `BM × BK` of A and `BK × BN` of B fits in SMEM (well under 228 KB). Compute `BM × BN` output tile in registers via outer-product accumulations. Each tile reuses the SMEM data many times, raising arithmetic intensity above the ridge.

---

## 3. Attention & Transformers

**Q: Walk through the steps of self-attention.**
A: Project X into Q, K, V (per head). Compute scores `S = Q · K^T / sqrt(d)`, apply mask, softmax → P, multiply by V → O, concat heads, output projection. See [Attention_Mechanisms](Attention_Mechanisms.md).

**Q: Why is `1/sqrt(d)` scaling needed?**
A: Without it, dot products grow linearly with d, pushing softmax into a saturated regime where gradients vanish. `1/sqrt(d)` keeps `S` at unit variance.

**Q: Derive the online softmax update.**
A: Maintain running max m and denominator ℓ. New block has local max m' and sum ℓ'. Combined max m_new = max(m, m'). Rescale old by `exp(m - m_new)`, scale new by `exp(m' - m_new)`, sum. Final O divided by ℓ once.

**Q: What is GQA and why does it help?**
A: Grouped-Query Attention: H_kv < H_q, K and V shared across groups of Q heads. Reduces KV memory by group factor. Llama-3-70B: H_q=64, H_kv=8 → 8× KV reduction. Decode is BW-bound, so direct throughput win.

**Q: What is MLA?**
A: Multi-head Latent Attention (DeepSeek). Stores a low-rank latent `c_KV` (~512 dim) per token plus a small rotary part, instead of K and V. ~30× KV compression vs MHA.

**Q: What's RoPE? Why is it position-equivariant?**
A: Rotary Position Embedding rotates Q and K by angles depending on position. The dot product `q_m · k_n` only depends on the relative offset `m - n`, giving the model relative-position behavior naturally. Applied per query-key pair as a 2D rotation in each pair of channels.

**Q: Explain pre-norm vs post-norm.**
A: Pre-norm: `x + Attn(LN(x))`. Post-norm: `LN(x + Attn(x))`. Pre-norm trains more stably at depth (residual stream stays well-conditioned). All modern LLMs are pre-norm.

**Q: What's SwiGLU?**
A: A gated FFN: `down(silu(gate(x)) * up(x))`. Uses 3 projections instead of 2, slightly more compute, much better quality. Standard in Llama-family.

**Q: Why FlashAttention?**
A: Standard attention writes `S` and `P` (size N²) to HBM, making it bandwidth-bound. FA tiles Q, K, V into blocks fitting in SMEM, computes attention with online softmax in-register, never writes S/P to HBM. 2-5× speedup, drops memory from O(N²) to O(N).

**Q: Difference between FA-v1, v2, v3?**
A: v1: tiled forward with online softmax. v2: split-Q work partition + better causal skipping → 2× over v1. v3: Hopper-specific warp specialization, TMA, async wgmma, FP8 → ~75% of peak FP16 on H100. See [FlashAttention_Deep_Dive](FlashAttention_Deep_Dive.md).

---

## 4. KV Cache

**Q: How big is the KV cache for Llama-3-70B at 8K tokens, batch 32?**
A: Per-token: `2 · L · H_kv · d · b = 2 · 80 · 8 · 128 · 2 = 320 KB`. × 8192 × 32 ≈ 80 GB.

**Q: What's PagedAttention?**
A: KV cache split into fixed-size blocks (e.g., 16 tokens). Each sequence has a block table mapping logical positions to physical blocks. Eliminates fragmentation, supports variable-length batching, prefix sharing, and preemption. Kernel takes the block table and indirects per logical block.

**Q: How does prefix caching work?**
A: Hash each block on `(parent_hash, token_ids)`. New requests walk their prompt block-by-block; on hit, reuse the physical block (refcount++). Skip the prefilled prefix, only compute the suffix. Big throughput win for chat / RAG / long-system-prompt workloads.

**Q: What's the difference between hash-based and radix-tree prefix caching?**
A: Hash-based (vLLM): block-level (16 tokens), only matches at block boundaries. Radix-tree (SGLang): token-level, matches at arbitrary boundaries — better hit rate at the cost of more bookkeeping.

**Q: When would you use FP8 vs INT8 KV cache?**
A: FP8 (E4M3) on Hopper: hardware-supported, near-zero accuracy loss, 2× capacity vs FP16. INT8: pre-Hopper, more accuracy risk on long contexts. Default to FP8 if available.

**Q: What's "preemption" in serving?**
A: When KV pool is full, kick out a victim sequence to free blocks. Choices: drop blocks (recompute on resume) or swap to host RAM. vLLM defaults to recompute for incoming, swap for in-flight.

---

## 5. Batching, Scheduling, Disaggregation

**Q: What is continuous batching?**
A: Each step decides which sequences run from a pool of in-flight ones. New sequences join immediately; finished ones leave without disturbing others. Vs static batching (fixed cohort), much higher utilization at length variance. Standard in vLLM/SGLang/TRT-LLM.

**Q: What is prefill–decode interference and how do you fix it?**
A: Mixing a compute-bound prefill with bandwidth-bound decodes makes the step take prefill's time, hurting decode TPOT. Fix: chunked prefill (split long prompts), or disaggregate prefill and decode across separate pools.

**Q: Walk through the lifecycle of a request in vLLM.**
A: Submit → tokenize → admit (check budget, prefix cache) → join waiting queue → admit to running on a step → prefill (chunked if long) → decode steps until EOS or max-tokens → emit output, release KV blocks (refcount-- on shared prefix).

**Q: Why is paging required for continuous batching?**
A: Continuous batching needs each sequence to grow independently with arbitrary length. Padded contiguous layouts can't represent that without massive waste. Paging gives each sequence its own block-table over a global pool — variable-length-friendly.

**Q: When does prefill–decode disaggregation pay off?**
A: When the workload mix has bursty long prompts that interfere with decode SLOs, and at scales where the KV-transfer overhead is paid back by 30%+ utilization gain. Below ~1K RPS, simpler coupled + chunked prefill is usually sufficient.

**Q: How does speculative decoding work and when does it help?**
A: Cheap draft generates K future tokens; target verifies them in one forward pass via rejection sampling. Each accepted token is "free" (extra compute hidden in BW-bound step). Helps when α (acceptance) > 0.5 and T_draft / T_target < 1/K. Hurts otherwise.

**Q: Compare Medusa and EAGLE.**
A: Medusa: K parallel heads predict future tokens directly from target hidden state. EAGLE: small autoregressive head fed by target's hidden state predicts future embeddings, decoded into tokens. EAGLE achieves higher acceptance (0.85–0.95 vs 0.7–0.85) with moderate extra compute.

---

## 6. Quantization

**Q: What's the difference between W4A16 and W8A8 quantization?**
A: W4A16: weights INT4, activations FP16, dequant in-kernel. Easier accuracy, ideal for BW-bound decode. W8A8: weights and activations INT8/FP8, full tensor-core speedup. Higher accuracy risk, especially activation outliers.

**Q: What's GPTQ?**
A: Post-training weight quantization that propagates rounding errors to remaining columns via the input Hessian. Calibration dataset required. INT4 weights with sub-1pp perplexity loss on big models.

**Q: AWQ vs GPTQ?**
A: AWQ scales salient (high-activation) weight channels up before quantization so noise lands on less-important channels. Simpler than GPTQ (no Hessian), often better for chat. GPTQ has slight edge for non-chat distributions.

**Q: SmoothQuant?**
A: Migrates quantization difficulty from activations to weights via per-channel scaling: `(X/s) @ (sW)` is the same product but X becomes smoother (INT8-friendly) and W stays per-channel quantizable. Enables W8A8.

**Q: Why is FP8 better than INT8 for activations?**
A: FP8's exponent handles wide dynamic range — outliers fit naturally. INT8's uniform spacing makes outliers dominate the per-tensor scale. Hopper has hardware-fused FP8 scaling.

**Q: How does FP8 KV cache impact quality?**
A: Typically <1% MMLU drop, <0.05 perplexity. Long-context retrieval (Ruler/NIAH) more sensitive — verify there.

**Q: What should you NOT quantize?**
A: LM head (small, quality-critical). LayerNorm parameters (tiny, no benefit). Embedding lookups (often kept FP16). Sometimes the first/last layer.

---

## 7. Distributed Training

**Q: Compare DP, TP, PP, and EP.**
A: DP: replicate model, split batch, all-reduce grads. TP: split layer weights across ranks, all-reduce inside layer. PP: split layers into stages, point-to-point activations. EP: split MoE experts across ranks, all-to-all per MoE layer. See [Parallelism_Strategies](Parallelism_Strategies.md).

**Q: What does ZeRO-3 (FSDP) do?**
A: Shards parameters, gradients, and optimizer state across DP ranks. Gathers params just-in-time per layer's forward. Enables training models that don't fit per-GPU. Communication cost is roughly the same total bytes as DP all-reduce but spreadable across forward and backward, overlapping with compute.

**Q: When do you use TP vs PP?**
A: TP within an NVLink domain (intra-node, ≤8 typically) because every layer requires high-BW all-reduces. PP across slower fabric (inter-node) because it only sends activations between stage boundaries.

**Q: What's the pipeline bubble?**
A: Idle time at pipeline start/end before stages are filled. Fraction ≈ `(P-1)/(M+P-1)` with P stages and M microbatches. Reduce via interleaved 1F1B or zero-bubble PP, or just more microbatches.

**Q: Why do we use BF16 instead of FP16 for training?**
A: BF16 has the same range as FP32 (8-bit exponent), so no loss scaling needed. FP16's tighter range requires careful loss scaling. BF16 just works.

**Q: What's Transformer Engine FP8 doing?**
A: Wraps each Linear with FP8 logic — maintains a moving max of activations (delayed scaling), picks scales to keep tensors in FP8 range, runs Hopper FP8 matmul at 2× FP16 throughput. Master weights stay FP32; non-matmul ops stay BF16.

**Q: What's selective activation recomputation?**
A: Recompute attention activations in backward (cheap with FA), keep other activations saved. ~5% extra compute, dramatic memory savings.

**Q: What's "critical batch size"?**
A: Batch size beyond which adding tokens stops improving loss-per-step. Empirically estimated; modern frontier runs use 2-16M tokens (large but not unlimited).

**Q: How big are checkpoints for 70B?**
A: Mixed precision: ~1.1 TB (params + master + grads + Adam state, 16 bytes/param). Save async (DCP) hourly + per-N-steps.

---

## 8. Inference Frameworks

**Q: vLLM vs TRT-LLM?**
A: vLLM: Python+CUDA, broad model coverage, fast feature uptake, easy ops. TRT-LLM: NVIDIA-tuned C++ engine, lowest latency on NVIDIA, comprehensive quant, but per-config build pipelines required and lags new architectures. vLLM for general, TRT-LLM for latency-critical.

**Q: What does SGLang add over vLLM?**
A: RadixAttention (token-granularity prefix sharing), first-class structured output via xgrammar, Rust scheduler. Strong for chat / agentic / RAG.

**Q: What is Dynamo (NVIDIA)?**
A: Multi-node serving fabric: disaggregates prefill and decode, routes with prefix locality, manages KV transfer via NIXL. Uses TRT-LLM/vLLM as engine backends.

**Q: How does multi-LoRA serving work?**
A: Each request specifies a LoRA id. Engine groups by LoRA or uses a fused kernel that takes per-row LoRA selection (Punica/S-LoRA). Adapters loaded into engine memory; lazy-loaded with LRU eviction for fleets with many adapters.

**Q: How does structured output (JSON, regex, grammar) work?**
A: At each decode step, mask logits for tokens that would violate the constraint (set to -inf). Outlines/xgrammar/lm-format-enforcer compile the constraint into per-state allowed-token sets. Engine integrates as a per-step logit processor.

---

## 9. Networking & Collectives

**Q: What is NVLink? How fast?**
A: NVIDIA's GPU-to-GPU interconnect. Hopper: ~900 GB/s aggregate per GPU (18 lanes × 50 GB/s/lane). NVLink Switch (NVSwitch) connects all GPUs in a node at full bandwidth. NVL72 extends this to 72 GPUs.

**Q: NVLink vs PCIe vs InfiniBand?**
A: NVLink: GPU-to-GPU, 900 GB/s, intra-node. PCIe Gen5 x16: 64 GB/s, GPU-to-CPU/NIC. InfiniBand NDR: 50 GB/s per port, inter-node. XDR: 100 GB/s.

**Q: What's GPUDirect RDMA?**
A: NIC reads/writes GPU HBM directly, bypassing host CPU. Saves a copy through pinned host memory; cuts cross-node latency and BW pressure on PCIe.

**Q: What's NIXL?**
A: NVIDIA Inference Xfer Library. Abstract GPU↔GPU/CPU/storage moves with one API; selects NVLink, RDMA, etc. Used to move KV between prefill and decode pools.

**Q: How does ring all-reduce work?**
A: N ranks in a logical ring. Phase 1 (reduce-scatter, N-1 steps): each rank sends 1/N to next, receives 1/N from prev, sums locally. Phase 2 (all-gather, N-1 steps): rotate complete chunks around. Total bytes moved per rank: 2(N-1)/N · B. Optimal at large B.

**Q: When does NCCL pick tree over ring?**
A: At small payloads where latency `α·N` dominates `β·B`. Tree has log(N) rounds vs ring's N. Threshold tuned per topology.

**Q: What is NVLink SHARP?**
A: In-fabric reduction by NVSwitch. All-reduce becomes a tree reduce in the switch + broadcast. Halves GPU-side BW for all-reduce. Enabled with `NCCL_NVLS_ENABLE=1` on supported fabrics.

---

## 10. Observability & Reliability

**Q: What metrics matter for an inference service?**
A: TTFT/TPOT/E2E latency p50/p95/p99 by tier; throughput (RPS, tokens/sec); engine state (active seqs, queue depth, KV occupancy %, prefix cache hit rate, preemption rate); SLO violation rate; GPU SM/HBM/tensor-core util.

**Q: How do you debug a hung training run?**
A: `py-spy dump` each rank to find where stuck; compare ranks (which is ahead/behind?); `NCCL_DEBUG=INFO` shows which collective hung; check IB link health, DCGM for hardware events; bisect on recent code changes. Common cause: one rank threw silently, others wait at next collective.

**Q: How do you find a straggler?**
A: Per-rank step time histograms. Variance >2× signals a straggler. DCGM utilization, hardware event counters (ECC, thermal). Cure: evacuate the bad node.

**Q: How would you triage "TPOT regressed 30% after deploy"?**
A: (1) Diff the deploy. (2) Check engine config (chunk size, batch caps). (3) Inspect KV occupancy and prefix cache hit rate. (4) NCCL bus-bw via nccl-tests. (5) DCGM SM/HBM util — kernel-bound or BW-bound? (6) Bisect via canary if needed.

**Q: What do you put in a runbook for an LLM service?**
A: Common failure modes (OOM, hung NCCL, slow disk, NIC down) with diagnostic steps; restart procedures (replica, model server, full pool); SLO definitions and alert thresholds; on-call escalation paths; postmortem template.

---

## 11. Big-Picture / Behavioral

**Q: Walk me through the most interesting performance optimization you've shipped.**
A: (Tell a story.) State the problem, what was tried before, your measurement, the change you made, the quantitative result, lessons learned. Strong answers cite specific kernels / configs / numbers, not vague "made it faster."

**Q: How do you decide what to optimize?**
A: Profile first; pick the top 1-2 bottlenecks. Estimate ROI (engineering hours vs % speedup). Pick the highest ROI. Avoid premature micro-optimization on cold paths.

**Q: How do you stay current in this rapidly evolving field?**
A: Read papers (arXiv, blogs, Twitter/X). Track the OSS engines (vLLM, SGLang, TRT-LLM release notes). Follow GTC, OSDI, MLSys talks. Build small benchmarks to verify claims. Specific people/teams worth following depending on your interest area.

**Q: Tell me about a time you had to debug a really hard distributed-systems issue.**
A: (Story.) A specific incident, what you saw, your hypotheses, how you ruled them out, the final cause, the fix, the post-incident hardening.

**Q: What's one thing you'd improve about a popular framework?**
A: Pick something concrete and grounded — e.g., "vLLM's V0 scheduler had Python-overhead bottleneck at high RPS; V1 fixed it but I'd push further on prefix-tree caching" or "TRT-LLM build pipeline is per-config; an in-memory engine cache would help."

**Q: Why this team / role?**
A: Be specific. "The work on disaggregated serving is the hardest unsolved problem in inference cost" or "I want to ship kernels that real production stacks rely on." Generic answers ("you're doing cool stuff") don't land.

**Q: Where do you see GPU computing going in the next 5 years?**
A: Reasonable answer: (a) FP4 / MXFP becoming standard, (b) bigger NVLink domains (NVL72 → bigger), (c) closer integration of inference engines with hardware features (TMA, async wgmma), (d) MoE / latent-attention compression, (e) post-Transformer architectures emerging, (f) heterogeneous fleets (H100 + B200 + cheaper inference accelerators).

---

## 12. Crisp Answers Cheat Sheet

When you blank, fall back on:

- **"What is X?"** — definition first, then mechanism, then why it matters.
- **"Compare X and Y."** — axes (latency / throughput / quality / complexity / cost), say where each wins.
- **"How would you design …"** — clarify, estimate, lay out tiers, walk a request, tradeoffs, failures.
- **"How would you debug …"** — top-down (metrics → system → kernel), name specific tools.
- **"How do you decide …"** — measurable criteria, prefer numbers over vibes.

---

## 13. Mock Interview Plan

Two weeks before an onsite:
- Day 1-3: Knowledge gaps. Use this notebook's table of contents, mark weak topics, study each in depth.
- Day 4-7: Practice 4-6 system designs (see [System_Design_Interview §12](System_Design_Interview.md)). Time yourself.
- Day 8-10: Coding patterns (see [Coding_Patterns](Coding_Patterns.md)).
- Day 11-13: Mock interviews with peers; verbalize.
- Day 14: Light review; sleep well.

---

## 14. Further Reading

Each section above links to its dedicated page. For broader context:
- [Index](Index.md) — full table of contents.
- [System_Design_Interview](System_Design_Interview.md) — design framework + worked examples.
- [Coding_Patterns](Coding_Patterns.md) — the algorithmic / kernel-level coding side.

---

**Next:** [Coding_Patterns](Coding_Patterns.md).

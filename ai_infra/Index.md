# AI Infrastructure — Master Index

> A senior-level technical notebook for AI infrastructure interviews and practice.
> Target roles: LLM inference/serving engineer, GPU kernel/CUDA engineer, distributed training/MLSys engineer.
> Style: deep theory + proofs + worked examples + production-grade tradeoffs + 2025–2026 frontier coverage.
>
> **39 pages** organized in 8 parts. Parts 1–7 cover the core mechanisms; Part 8 (cutting-edge frontier) covers 2024–2026 architectures, hardware, and engineering patterns specifically.

---

## How to Use This Notebook

Each page is a standalone deep-dive on one topic. Pages are sequenced so that later pages assume the contents of earlier ones, but every page stands on its own for review. The style follows the rest of the notebook: "explain as if the reader is a senior engineer who needs the mechanism, the math, the numbers, and the tradeoffs — not a 101 summary."

Three layers of preparation are interwoven:

1. **Mechanism layer** — what the system actually does at the hardware/algorithm level. Proofs, diagrams, arithmetic.
2. **Engineering layer** — why production systems make the tradeoffs they do. Capacity math, SLAs, failure modes.
3. **Interview layer** — how to articulate the above under time pressure. Sample Q&A, system design walkthroughs.

---

## Part 1 — Hardware Foundations

1. [GPU Architecture](GPU_Architecture.md) — SM internals, tensor cores, Hopper/Blackwell specifics, execution model.
2. [Memory Hierarchy and Roofline](Memory_Hierarchy_and_Roofline.md) — HBM, SRAM, registers, arithmetic intensity, the roofline model, ridge-point math.
3. [Networking and Interconnect](Networking_and_Interconnect.md) — NVLink/NVSwitch, InfiniBand NDR/XDR, RoCE v2, RDMA, GPUDirect, rail-optimized topology.
4. [Storage and Model Loading](Storage_and_Model_Loading.md) — Safetensors, GGUF, GDS, loading pipelines, caching strategies.

## Part 2 — CUDA and Kernel Programming

5. [CUDA Programming](CUDA_Programming.md) — thread/block/warp hierarchy, memory model, streams, synchronization, end-to-end examples.
6. [CUDA Optimization](CUDA_Optimization.md) — coalescing, bank conflicts, occupancy, async copy, warp specialization, Hopper TMA.
7. [Triton and Custom Kernels](Triton_and_Kernels.md) — OpenAI Triton DSL, tiling, autotuning, CUTLASS, writing a matmul, writing FlashAttention.

## Part 3 — Transformers and Attention

8. [Transformer Internals](Transformer_Internals.md) — the full forward pass, positional encodings (RoPE, ALiBi), normalization, MoE basics.
9. [Attention Mechanisms](Attention_Mechanisms.md) — scaled dot-product, MHA/MQA/GQA/MLA, online softmax derivation.
10. [FlashAttention Deep Dive](FlashAttention_Deep_Dive.md) — v1/v2/v3 algorithm walkthrough with tile math and pseudocode.

## Part 4 — Inference Core

11. [KV Cache](KV_Cache.md) — layout, memory math, PagedAttention, prefix caching, radix tree, offloading, NIXL.
12. [Batching and Scheduling](Batching_and_Scheduling.md) — continuous batching, chunked prefill, admission control, preemption.
13. [Prefill-Decode Disaggregation](Prefill_Decode_Disaggregation.md) — motivation, architecture, transfer mechanics, real benchmarks.
14. [Speculative Decoding](Speculative_Decoding.md) — vanilla, self-speculation, Medusa, EAGLE, acceptance rate math.
15. [Quantization](Quantization.md) — FP16/BF16/FP8/FP4/INT8/INT4, GPTQ, AWQ, SmoothQuant, KV cache quantization.

## Part 5 — Distributed Systems

16. [Parallelism Strategies](Parallelism_Strategies.md) — DP, TP, PP, EP, CP, SP, hybrid/3D, communication-volume math.
17. [Collectives and NCCL](Collectives_and_NCCL.md) — AllReduce algorithms (ring, tree, recursive halving-doubling), NCCL internals, bandwidth modeling.
18. [Distributed Training](Distributed_Training.md) — FSDP, ZeRO-1/2/3, checkpointing, fault tolerance, elastic training.
19. [Training Optimization](Training_Optimization.md) — mixed precision (AMP), activation checkpointing, gradient accumulation, gradient compression.

## Part 6 — Frameworks and Deployment

20. [Inference Frameworks](Inference_Frameworks.md) — vLLM, SGLang, TensorRT-LLM, NVIDIA Dynamo, llm-d, TGI — feature-by-feature comparison.
21. [vLLM Internals](vLLM_Internals.md) — engine architecture, scheduler, workers, executor, block manager, APC.
22. [Kubernetes and Orchestration](Kubernetes_and_Orchestration.md) — GPU operator, device plugin, MIG, topology-aware scheduling, HPA/KEDA, OME/llm-d.
23. [Observability and Debugging](Observability_and_Debugging.md) — TTFT/ITL/TPOT metrics, DCGM, Nsight, debugging flow for common pathologies.
24. [Production Architecture](Production_Architecture.md) — reference stack, capacity planning, cost modeling, failure modes.

## Part 7 — Interview Prep

25. [System Design Interview](System_Design_Interview.md) — design ChatGPT, design a multi-tenant inference service, design a training cluster, design a checkpoint store.
26. [Common Interview Questions](Common_Interview_Questions.md) — conceptual Q&A across all topics.
27. [Coding Patterns](Coding_Patterns.md) — CUDA kernel patterns, Python async, tokenizer quirks, vLLM plugin examples.

## Part 8 — Cutting-Edge Frontier (2025–2026)

These pages are deliberately scoped to bleeding-edge content: 2024–2026 architectures, hardware, algorithms, and engineering patterns that the "core" pages above don't cover deeply enough. Read after Parts 1–7 or as standalones for whichever direction you're targeting.

28. [Frontier Models 2025–2026](Frontier_Models_2025_2026.md) — Llama-4, DeepSeek-V3/R1, Qwen-3, Gemma-3, GPT-5/o3, Claude-4, Gemini-2.5; MLA, MTP, aux-loss-free balancing.
29. [Reasoning Models](Reasoning_Models.md) — o1/o3/R1 architecture and post-training; long-CoT RL; test-time compute scaling; thinking budgets; tool-use during reasoning.
30. [Modern MoE](Modern_MoE.md) — DeepSeekMoE, fine-grained / shared experts, auxiliary-loss-free, top-1 vs top-k, EP at NVL72 scale, DeepEP.
31. [Long Context Engineering](Long_Context_Engineering.md) — YaRN, LongRoPE, iRoPE, NSA, MoBA, ring attention, chunked prefill, 1M-10M context.
32. [Modern KV Compression](Modern_KV_Compression.md) — MLA, KIVI, StreamingLLM, H2O, SnapKV, Quest, Mooncake-style global pools.
33. [State Space Models and Hybrids](State_Space_Models_and_Hybrids.md) — Mamba-2, RWKV-7, Jamba, Zamba, hybrid SSM-Transformer stacks; serving engineering.
34. [Blackwell Architecture](Blackwell_Architecture.md) — B100/B200/B300/GB200/GB300, NVLink-5, NVL72/NVL576, TMEM, FP4 hardware, Rubin outlook.
35. [Modern Quantization Frontier](Modern_Quantization_Frontier.md) — MXFP4, NVFP4, FP6, FP4 inference and training, SpinQuant, calibration pipelines.
36. [Disaggregated Serving 2025](Disaggregated_Serving_2025.md) — Mooncake, DistServe, Splitwise, Sarathi-Serve; NVIDIA Dynamo, llm-d; KV pools, layer-pipelined transfer.
37. [Modern Post-Training](Modern_Post_Training.md) — DPO/IPO/KTO/SimPO/ORPO, GRPO, online RL infra (TRL, OpenRLHF, veRL, NeMo-Aligner), distillation from reasoning teachers.
38. [Multimodal Inference](Multimodal_Inference.md) — VLMs (Qwen-VL, InternVL, Llama-4-MM), audio (Whisper, Moshi), video gen (Sora, Veo, Kling), unified any-to-any.
39. [Cutting-Edge Kernels](Cutting_Edge_Kernels.md) — CUTLASS 3.x, CuTe, FlashInfer, DeepEP, TileLang, ThunderKittens, Liger-Kernel; Hopper/Blackwell tile programming.

---

## Cross-Cutting Themes

These show up on almost every page — if you can pattern-match them quickly, interviews become much easier.

**The memory wall.** LLM inference is bandwidth-bound at decode. Every optimization (FlashAttention, PagedAttention, speculation, quantization, GQA, MLA) is a way to reduce bytes moved per token. See [Memory_Hierarchy_and_Roofline](Memory_Hierarchy_and_Roofline.md).

**Arithmetic intensity.** Operations live on a roofline: FLOPs per byte decides whether hardware bandwidth or FLOPS is the bottleneck. Prefill is compute-bound, decode is memory-bound. See [Memory_Hierarchy_and_Roofline](Memory_Hierarchy_and_Roofline.md).

**Phase interference.** Prefill and decode have opposite hardware profiles. Co-locating them on the same GPU creates head-of-line blocking. Modern systems disaggregate. See [Prefill_Decode_Disaggregation](Prefill_Decode_Disaggregation.md).

**The KV cache dominates capacity.** At long contexts, KV cache exceeds model weights. Every capacity planning exercise reduces to: layers × kv_heads × head_dim × 2 × bytes × tokens × concurrent_requests.

**Collectives are latency-critical.** TP AllReduce happens per-layer. The NVLink bandwidth of your node directly caps your latency floor. See [Collectives_and_NCCL](Collectives_and_NCCL.md).

**Batching amortizes weight reads.** A batch of N decodes reads the model weights from HBM once per step instead of N times. This is the entire reason continuous batching exists. See [Batching_and_Scheduling](Batching_and_Scheduling.md).

**The reasoning tax.** Test-time-compute scaling (o1/R1-style) shifts cost from training to inference and from short to long generations. Decode pools must grow 10-100× larger than chat pools per request. See [Reasoning_Models](Reasoning_Models.md).

**MLA + MoE compose.** DeepSeek-V3 demonstrates that compressing KV cache (MLA) and active compute (MoE) are orthogonal — stacking them gives huge effective scale on modest hardware. See [Frontier_Models_2025_2026](Frontier_Models_2025_2026.md), [Modern_MoE](Modern_MoE.md).

**Aux-loss-free balancing.** Replacing the load-balancing auxiliary loss with a per-expert EMA bias removes a quality tax that previously hurt large MoE. Standard in 2025 designs.

**FP4 native.** Blackwell makes FP4 a hardware-supported tensor-core format, doubling FP8 throughput with careful calibration. The next throughput multiplier after FP8. See [Blackwell_Architecture](Blackwell_Architecture.md), [Modern_Quantization_Frontier](Modern_Quantization_Frontier.md).

**Cluster-scale KV pool.** Mooncake-style global pools push prefix-cache hit rates to 80%+ at fleet scale, where per-replica caches plateau at 30-50%. Standard in NVIDIA Dynamo / llm-d. See [Disaggregated_Serving_2025](Disaggregated_Serving_2025.md).

**Inference-engine-inside-RL.** Modern post-training (GRPO, online PPO) is dominated by inference cost; the inference engine becomes a major component of the training stack. See [Modern_Post_Training](Modern_Post_Training.md).

---

## Suggested Reading Order

**For a 2-week interview sprint:**

Week 1 (mechanism): GPU_Architecture → Memory_Hierarchy → Transformer_Internals → Attention_Mechanisms → KV_Cache → Batching_and_Scheduling → Parallelism_Strategies.

Week 2 (production + interview): Prefill_Decode_Disaggregation → Quantization → vLLM_Internals → Kubernetes_and_Orchestration → Observability → System_Design_Interview → Common_Interview_Questions.

**For a CUDA/kernel-engineer interview:**

CUDA_Programming → CUDA_Optimization → Triton_and_Kernels → FlashAttention_Deep_Dive → Attention_Mechanisms → Coding_Patterns.

**For a distributed-training interview:**

GPU_Architecture → Networking_and_Interconnect → Parallelism_Strategies → Collectives_and_NCCL → Distributed_Training → Training_Optimization → System_Design_Interview.

**For a frontier-only crash course (~one weekend):**

Frontier_Models_2025_2026 → Modern_MoE → Reasoning_Models → Long_Context_Engineering → Modern_KV_Compression → Disaggregated_Serving_2025 → Blackwell_Architecture → Modern_Quantization_Frontier → Modern_Post_Training → Cutting_Edge_Kernels → State_Space_Models_and_Hybrids → Multimodal_Inference.

**For 2026 LLM-inference engineer interviews specifically:**

Skim Parts 1–6 for fluency; deep-read Part 8: Frontier_Models_2025_2026, Reasoning_Models, Disaggregated_Serving_2025, Modern_KV_Compression, Modern_Quantization_Frontier. Plus System_Design_Interview from Part 7.

**For 2026 GPU-kernel engineer interviews specifically:**

CUDA_Programming → CUDA_Optimization → FlashAttention_Deep_Dive → Blackwell_Architecture → Modern_Quantization_Frontier → Cutting_Edge_Kernels → Coding_Patterns.

---

## Numbers to Memorize

These numbers come up constantly. Burn them into memory before an interview.

| Quantity | Value | Why it matters |
|---|---|---|
| H100 SXM HBM bandwidth | 3.35 TB/s | Decode bottleneck |
| H100 FP16 tensor TFLOPS | ~990 | Prefill ceiling |
| H100 FP8 tensor TFLOPS | ~1980 | Post-quant prefill |
| B200 HBM bandwidth | 8 TB/s | Next-gen decode |
| B200 HBM capacity | 192 GB | Fits 70B BF16 + KV |
| B200 FP8 tensor TFLOPS | ~4500 | Blackwell prefill ceiling |
| B200 FP4 tensor TFLOPS | ~9000 | NVFP4 production target |
| B300 / Blackwell Ultra HBM | 288 GB @ 12 TB/s | 2026 sweet spot |
| NVLink 5 per-GPU | 1.8 TB/s | Intra-node collectives |
| NVSwitch domain | 72-GPU (NVL72) | Tensor-parallel scope |
| NVL72 HBM total (B200) | 13.5 TB | Cluster KV pool size |
| InfiniBand NDR | 400 Gbps (50 GB/s) | Inter-node |
| InfiniBand XDR | 800 Gbps (100 GB/s) | Quantum-X800 era |
| PCIe Gen5 x16 | 64 GB/s | Host ↔ GPU |
| NVLink-C2C (Grace-Hopper/Blackwell) | 900 GB/s | CPU↔GPU in superchip |
| H100 ridge point (FP16) | ~295 ops/byte | Roofline inflection |
| Llama-3-70B KV per token (FP16, GQA) | 320 KB | 2 × 80 × 8 × 128 × 2 |
| Llama-3-70B weights (FP16 / FP8 / INT4 / FP4) | 140 / 70 / 35 / 35 GB | Quant options |
| DeepSeek-V3 KV per token (MLA) | ~70 KB | 30× smaller than dense MHA |
| DeepSeek-V3 active / total params | 37B / 671B | MoE asymmetry |
| Llama-4 active / total (Maverick) | 17B / 400B | Top-1 routing |
| Typical TTFT SLO (chat) | <500 ms | Interactive UX |
| Typical TPOT SLO (chat) | <50 ms | Smooth streaming |
| Reasoning-mode output length (heavy) | 30K-100K tokens | KV pool sizing target |
| GRPO sample group K | 8-64 | Per-prompt rollouts |
| Mooncake-style cluster prefix hit rate | ~80% | Multi-replica chat workloads |

---

*Last updated: April 2026.*

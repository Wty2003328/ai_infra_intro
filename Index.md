# AI Infrastructure — Master Index

> A senior-level technical notebook for AI infrastructure interviews and practice.
> Target roles: LLM inference/serving engineer, GPU kernel/CUDA engineer, distributed training/MLSys engineer.
> Style: deep theory + proofs + worked examples + production-grade tradeoffs + 2025–2026 frontier coverage.
>
> **66 content pages** organized in 9 layers (L0–L8) plus interview prep. The layers follow a strict bottom-up dependency chain: silicon physics up through inference serving.

---

## How to Use This Notebook

Each page is a standalone deep-dive on one topic. Pages are sequenced so that later pages assume the contents of earlier ones, but every page stands on its own for review. The style follows the rest of the notebook: "explain as if the reader is a senior engineer who needs the mechanism, the math, the numbers, and the tradeoffs — not a 101 summary."

Three layers of preparation are interwoven:

1. **Mechanism layer** — what the system actually does at the hardware/algorithm level. Proofs, diagrams, arithmetic.
2. **Engineering layer** — why production systems make the tradeoffs they do. Capacity math, SLAs, failure modes.
3. **Interview layer** — how to articulate the above under time pressure. Sample Q&A, system design walkthroughs.

---

## L0 — Silicon and Process

1. [Silicon For AI](L0_Silicon_and_Process/01_Silicon_For_AI.md) — CMOS scaling, FinFET/GAA transistors, process nodes (N5/N3/N2), SRAM/Logic scaling divergence, dark silicon, voltage-frequency tradeoffs, defect density, wafer economics, Rubin-era 3NP process outlook.

## L1 — Packaging and Memory

2. [Advanced Packaging](L1_Packaging_and_Memory/01_Advanced_Packaging.md) — 2.5D (CoWoS), 3D stacking, chiplet interconnects (UCIe), co-packaged optics, glass substrates, reticle limits, TSV density, thermal budgets, interposer routing.
3. [HBM Deep Dive](L1_Packaging_and_Memory/02_HBM_Deep_Dive.md) — HBM3/HBM3E/HBM4 architecture, channel organization, bandwidth math, stack height, thermal limits, co-packaged vs discrete, emerging on-chip memory alternatives.

## L2 — Digital Design for AI

4. [On-Chip Memory Hardware](L2_Digital_Design_for_AI/01_On_Chip_Memory_Hardware.md) — SRAM cell design, register files, TMEM (Blackwell), scratchpad vs cache, area/energy per bit, bank conflict hardware, emerging on-chip memory.
5. [FP Unit Design](L2_Digital_Design_for_AI/02_FP_Unit_Design.md) — IEEE 754, FP16/BF16/FP8/FP4/INT8 hardware, BitNet 1-bit multiplier considerations, multiplier trees, fused multiply-add, denormal handling.
6. [Systolic Arrays and Dataflow](L2_Digital_Design_for_AI/03_Systolic_Arrays_and_Dataflow.md) — weight-stationary / output-stationary / row-stationary dataflow, tiling, mapping convolutions and matmuls, throughput modeling.
7. [Digital Design For AI](L2_Digital_Design_for_AI/04_Digital_Design_For_AI.md) — RTL for accelerators, dataflow vs von Neumann, control-flow design, clock gating, power domains, FPGA AI acceleration patterns.

## L3 — Microarchitecture

8. [ISA and Execution Model](L3_Microarchitecture/01_ISA_and_Execution_Model.md) — PTX/SASS, warp scheduling, instruction pipeline, latency/throughput tables, divergence, SIMT vs VLIW vs spatial dataflow.
9. [GPU Architecture](L3_Microarchitecture/02_GPU_Architecture.md) — complete host/front-end/GPC/SM/NoC/L2/GMMU/memory-controller/HBM hierarchy; precise CUDA kernel/grid/block/cluster/warp/thread and memory mapping; SM internals, tensor cores, Hopper/Blackwell specifics, and Threadblock Clusters / DSMEM.
10. [Memory Hierarchy and Roofline](L3_Microarchitecture/03_Memory_Hierarchy_and_Roofline.md) — HBM, SRAM, registers, TMEM, arithmetic intensity, the roofline model, ridge-point math, multi-chip roofline comparisons.
11. [Blackwell Architecture](L3_Microarchitecture/04_Blackwell_Architecture.md) — B100/B200/B300/GB200/GB300 (128 SMs confirmed), NVLink-5, NVL72/NVL576, TMEM, FP4 hardware, Vera Rubin R100 announced at GTC March 2026, NVLink-6, Rubin Ultra quad-die outlook.
12. [AMD Instinct](L3_Microarchitecture/05_AMD_Instinct.md) — CDNA 3/4/Next, MI300/MI350/MI400 "Altair" (432 GB HBM4, TSMC N2), Helios rack architecture, UALink, Infinity Fabric, ROCm 7 compute model, ROCm kernel optimization.
13. [Google TPU](L3_Microarchitecture/06_Google_TPU.md) — TPU v4/v5p/v5e/v6e Trillium/v7 Ironwood, TPU 8t/8i split (Apr 2026: training vs inference silicon, 384 MB SRAM on 8i), MXU systolic array, bfloat16, ICI 3D torus, pod-scale architecture (9,216-chip Ironwood, 9,600-chip 8t superpod).
14. [Huawei Ascend](L3_Microarchitecture/07_Huawei_Ascend.md) — Da Vinci architecture, Ascend 910B/910C/910D, CANN stack, cluster topology (CM-384).
15. [Cloud ASICs](L3_Microarchitecture/08_Cloud_ASICs.md) — AWS Inferentia/Trainium2/Trainium3, Azure Maia, Meta MTIA v2/v3, Intel Gaudi, in-house accelerators.
16. [Specialty Accelerators](L3_Microarchitecture/09_Specialty_Accelerators.md) — Cerebras WSE-3 (IPO May 2026), Groq LPU / LPU II (NVIDIA-Groq acquisition, $20B), Tenstorrent Blackhole, edge AI accelerators, FPGA AI acceleration, RAS for AI accelerators.
17. [Accelerator Landscape 2026](L3_Microarchitecture/10_Accelerator_Landscape_2026.md) — comparative survey of 12+ vendor families, Rubin R100 announced specs, MI455X Altair, TPU v8, Groq 3 LPX, performance/TCO benchmarks, roofline ridge-point comparison, roadmap outlook.

## L4 — Systems and Interconnects

18. [Networking and Interconnect](L4_Systems_and_Interconnects/01_Networking_and_Interconnect.md) — SerDes physics, PCIe/CXL 3.1/NVLink/UALink/Fabric protocols, IB XDR (800 Gbps), fat-tree/dragonfly/torus topologies, DCQCN, GPUDirect RDMA.
19. [Rack-Scale Design](L4_Systems_and_Interconnects/02_Rack_Scale_Design.md) — NVL72/NVL36x2/Helios/TPU pod/Rubin NVL576 architectures, 48V power distribution, liquid cooling math, mechanical constraints, NVIDIA-Marvell NVLink Fusion, Spectrum-X MRC.
20. [Storage and Model Loading](L4_Systems_and_Interconnects/03_Storage_and_Model_Loading.md) — Safetensors/pickle/GGUF formats, GDS, checkpoint I/O math, async checkpointing, KV cache offload tiers.

## L5 — Kernels and Programming

21. [CUDA Programming](L5_Kernels_and_Programming/01_CUDA_Programming.md) — thread/block/warp hierarchy, memory model, streams, synchronization, Hopper Threadblock Clusters, DSMEM, end-to-end examples.
22. [CUDA Optimization](L5_Kernels_and_Programming/02_CUDA_Optimization.md) — the generic optimization ladder: coalescing, shared-memory tiling, bank conflicts, occupancy, vectorization/ILP, launch overhead, Nsight profiling, reduction case study.
23. [Tensor Core Programming](L5_Kernels_and_Programming/03_Tensor_Core_Programming.md) — WMMA→WGMMA instruction hierarchy, FP8 tensor-core programming, 2:4 structured sparsity, TMEM (Blackwell), cp.async/TMA, mbarrier, double buffering, warp specialization, cluster optimizations.
24. [Triton and Kernels](L5_Kernels_and_Programming/04_Triton_and_Kernels.md) — OpenAI Triton 3.6.0 DSL, tiling, autotuning, writing a matmul, writing FlashAttention, CUTLASS/CuTe overview.
25. [FlashAttention Deep Dive](L5_Kernels_and_Programming/05_FlashAttention_Deep_Dive.md) — v1/v2/v3/v4 algorithm walkthrough with tile math and pseudocode; FlashAttention-4 beta for Blackwell and MLA.
26. [Cutting-Edge Kernels](L5_Kernels_and_Programming/06_Cutting_Edge_Kernels.md) — CUTLASS 3.x, CuTe, FlashInfer, DeepEP, TileLang, ThunderKittens, Liger Kernel v0.7.0, BitNet/bitnet.cpp 1-bit kernels; Hopper/Blackwell tile programming.
27. [ML Compilers](L5_Kernels_and_Programming/07_ML_Compilers.md) — torch.compile internals (Dynamo/AOTAutograd/Inductor), guards and graph breaks, CUDA Graphs, XLA/GSPMD, MLIR ecosystem, TensorRT-LLM, autotuning, compilers in vLLM/SGLang, fusion roofline math.

## L6 — Algorithms and Models

28. [Transformer Internals](L6_Algorithms_and_Models/01_Transformer_Internals.md) — the full forward pass, positional encodings (RoPE, ALiBi), normalization, parameter counts, JAX shard_map for distributed transforms.
29. [Attention Mechanisms](L6_Algorithms_and_Models/02_Attention_Mechanisms.md) — scaled dot-product, MHA/MQA/GQA/MLA, online softmax derivation, KV cache formulas.
30. [Modern MoE](L6_Algorithms_and_Models/03_Modern_MoE.md) — DeepSeekMoE, fine-grained / shared experts, aux-loss-free balancing, top-k routing, expert choice routing, DeepEP, MoE quantization, MoE inference optimizations.
31. [State Space Models and Hybrids](L6_Algorithms_and_Models/04_State_Space_Models_and_Hybrids.md) — Mamba-2, RWKV-7, Jamba, Zamba, hybrid SSM-Transformer stacks; serving engineering.
32. [Quantization](L6_Algorithms_and_Models/05_Quantization.md) — FP16/BF16/FP8/FP4/INT8/INT4, GPTQ, AWQ, SmoothQuant, KV cache quantization, error bounds.
33. [Modern Quantization Frontier](L6_Algorithms_and_Models/06_Modern_Quantization_Frontier.md) — MXFP4, NVFP4, NVFP4 KV cache, FP6, FP4 inference and training, TurboQuant 2-bit KV, online MXFP8, SpinQuant, Arcquant/2FP4, Transformer Engine v2.
34. [Frontier Models 2025–2026](L6_Algorithms_and_Models/07_Frontier_Models_2025_2026.md) — DeepSeek V4 Pro, Llama-4 (Scout/Maverick/Muse Spark), Qwen 3/3.5/3.6, Gemma-4, GPT-5/5.5, Claude Opus 4.7, Gemini 3/3.1 Pro, Mistral Medium 3.5/Small 4/Magistral, Kimi K2.6, GLM-5.1; MLA, MTP, MoE as standard.

## L7 — Training Stack

35. [Parallelism Strategies](L7_Training_Stack/01_Parallelism_Strategies.md) — DP, TP, PP, EP, CP, SP, hybrid/3D/5D, context parallelism (PyTorch native), ring attention production, HybridEP, MTP, LatentMoE, communication-volume math.
36. [Collectives and NCCL](L7_Training_Stack/02_Collectives_and_NCCL.md) — AllReduce algorithms (ring, tree, recursive halving-doubling), NCCL internals, SHARP, bandwidth modeling.
37. [Distributed Training](L7_Training_Stack/03_Distributed_Training.md) — FSDP2 compile redesign, ZeRO-1/2/3, async checkpointing, fault tolerance, elastic training, DeepSpeed v0.15, Megatron Core 0.15–0.17, MoE-specific training issues.
38. [Training Optimization](L7_Training_Stack/04_Training_Optimization.md) — mixed precision (AMP), activation checkpointing, gradient accumulation, Transformer Engine, GRPO production deployment, synthetic data distillation.
39. [Modern Post-Training](L7_Training_Stack/05_Modern_Post_Training.md) — DPO/IPO/KTO/SimPO/ORPO, GRPO production, online RL infra, distillation from reasoning teachers.
40. [Reasoning Models](L7_Training_Stack/06_Reasoning_Models.md) — o1/o3/o4-mini/R1/Qwen-3 thinking architecture, long-CoT RL, test-time compute scaling, thinking budgets.

## L8 — Inference and Serving

41. [KV Cache](L8_Inference_and_Serving/01_KV_Cache.md) — layout, memory math, PagedAttention, prefix caching, radix tree, offloading, KV cache quantization, NIXL/Mooncake transfer.
42. [Modern KV Compression](L8_Inference_and_Serving/02_Modern_KV_Compression.md) — StreamingLLM, H2O, KIVI, SnapKV, Quest, MLA, NVFP4 KV cache.
43. [Batching and Scheduling](L8_Inference_and_Serving/03_Batching_and_Scheduling.md) — continuous batching, chunked prefill, admission control, preemption.
44. [Speculative Decoding](L8_Inference_and_Serving/04_Speculative_Decoding.md) — vanilla, self-speculation, Medusa, EAGLE/EAGLE-3 (with MLA), DFLASH pipelined drafting, suffix automaton decoding, adaptive speculative decoding V2, Gemma-4 MTP (multi-token prediction), acceptance rate math.
45. [Prefill-Decode Disaggregation](L8_Inference_and_Serving/05_Prefill_Decode_Disaggregation.md) — motivation, architecture, transfer mechanics, NIXL transport, attention-FFN disaggregation, real benchmarks.
46. [Long Context Engineering](L8_Inference_and_Serving/06_Long_Context_Engineering.md) — YaRN, LongRoPE, iRoPE, NSA, MoBA, ring attention (production), 1M–10M context.
47. [Multimodal Inference](L8_Inference_and_Serving/07_Multimodal_Inference.md) — VLMs (Qwen-VL, InternVL, Llama-4 early-fusion), audio, video gen, unified models, multimodal E/P/D disaggregation.
48. [Inference Frameworks](L8_Inference_and_Serving/08_Inference_Frameworks.md) — vLLM v0.21.0, SGLang v0.5.12, TensorRT-LLM v1.3.0, NVIDIA Dynamo 1.0 (replaced Triton Inference Server), llm-d CNCF Sandbox, TGI archived (March 2026), BitNet/bitnet.cpp — feature-by-feature comparison.
49. [vLLM Internals](L8_Inference_and_Serving/09_vLLM_Internals.md) — V1 engine architecture, compiler-style IR layer, scheduler, V2 ModelRunner, block manager, APC, TurboQuant 2-bit KV compression, FlashAttention-4 for MLA, bi-directional disaggregated KV transfers, NIXL/MooncakeStoreConnector, thinking budgets, adaptive speculative decoding V2.
50. [Disaggregated Serving 2025](L8_Inference_and_Serving/10_Disaggregated_Serving_2025.md) — Mooncake, DistServe, Splitwise, Sarathi-Serve; NVIDIA Dynamo 1.0 disaggregation, llm-d; multimodal E/P/D disaggregation, bi-directional KV transfer, MooncakeStoreConnector, NIXL, attention-FFN disaggregation, TurboQuant 2-bit in serving; cluster KV pools.
51. [Agentic Inference](L8_Inference_and_Serving/11_Agentic_Inference.md) — serving tool-calling/multi-turn workloads, prefix-cache economics, cache-aware routing, KV offload tiers, compressed-FSM structured output, session scheduling, task-level SLOs, TPU 8i / Rubin CPX rationale.
52. [GPU Multi-Tenancy](L8_Inference_and_Serving/12_GPU_Multitenancy.md) — MIG/MPS/time-slicing/green contexts, isolation properties, interference math, K8s DRA and fractional GPUs, multi-model packing economics, confidential computing.
53. [Kubernetes and Orchestration](L8_Inference_and_Serving/13_Kubernetes_and_Orchestration.md) — GPU operator, device plugin, MIG, topology-aware scheduling, HPA/KEDA, OME/llm-d.
54. [Observability and Debugging](L8_Inference_and_Serving/14_Observability_and_Debugging.md) — TTFT/ITL/TPOT metrics, DCGM, Nsight, debugging flow for common pathologies.
55. [Production Architecture](L8_Inference_and_Serving/15_Production_Architecture.md) — reference stack, capacity planning, cost modeling, failure modes.

## Interview Prep

Per-layer Q&A consolidated out of the L0–L8 topic pages above (worked interview problems + snap answers, moved here to keep topic pages focused on concepts), followed by cross-cutting interview material:

56. [L0 Questions](interview_prep/01_L0_Silicon_and_Process_Questions.md) — silicon physics, process nodes, yield, packaging preview.
57. [L1 Questions](interview_prep/02_L1_Packaging_and_Memory_Questions.md) — advanced packaging, HBM.
58. [L2 Questions](interview_prep/03_L2_Digital_Design_for_AI_Questions.md) — pipelining/CDC, FP units, on-chip memory, systolic arrays.
59. [L3 Questions](interview_prep/04_L3_Microarchitecture_Questions.md) — GPU/TPU/ASIC microarchitecture (AMD, NVIDIA, Google, cloud ASICs, ISA models, roofline, specialty accelerators).
60. [L4 Questions](interview_prep/05_L4_Systems_and_Interconnects_Questions.md) — networking/interconnect, rack-scale design, storage and model loading.
61. [L5 Questions](interview_prep/06_L5_Kernels_and_Programming_Questions.md) — CUDA programming/optimization, cutting-edge kernels, FlashAttention, ML compilers, Triton.
62. [L6 Questions](interview_prep/07_L6_Algorithms_and_Models_Questions.md) — transformer internals.
63. [L8 Questions](interview_prep/08_L8_Inference_and_Serving_Questions.md) — batching/scheduling, multi-tenancy, inference frameworks, disaggregation, production architecture.
64. [Common Interview Questions](interview_prep/09_Common_Interview_Questions.md) — conceptual Q&A across all topics, updated for 2026 frontier models and hardware.
65. [System Design Interview](interview_prep/10_System_Design_Interview.md) — design ChatGPT, multi-tenant inference, training cluster, RAG, agent orchestrator, eval harness.
66. [Coding Patterns](interview_prep/11_Coding_Patterns.md) — CUDA kernel patterns, parallel reduction, scheduler algorithms, numerical algorithms, systems C++, BitNet kernel patterns.

---

## Cross-Cutting Themes

These show up on almost every page — if you can pattern-match them quickly, interviews become much easier.

**The memory wall.** LLM inference is bandwidth-bound at decode. Every optimization (FlashAttention, PagedAttention, speculation, quantization, GQA, MLA) is a way to reduce bytes moved per token. See [Memory_Hierarchy_and_Roofline](L3_Microarchitecture/03_Memory_Hierarchy_and_Roofline.md).

**Arithmetic intensity.** Operations live on a roofline: FLOPs per byte decides whether hardware bandwidth or FLOPS is the bottleneck. Prefill is compute-bound, decode is memory-bound. See [Memory_Hierarchy_and_Roofline](L3_Microarchitecture/03_Memory_Hierarchy_and_Roofline.md).

**Phase interference.** Prefill and decode have opposite hardware profiles. Co-locating them on the same GPU creates head-of-line blocking. Modern systems disaggregate. See [Prefill_Decode_Disaggregation](L8_Inference_and_Serving/05_Prefill_Decode_Disaggregation.md).

**The KV cache dominates capacity.** At long contexts, KV cache exceeds model weights. Every capacity planning exercise reduces to: layers × kv_heads × head_dim × 2 × bytes × tokens × concurrent_requests.

**Collectives are latency-critical.** TP AllReduce happens per-layer. The NVLink bandwidth of your node directly caps your latency floor. See [Collectives_and_NCCL](L7_Training_Stack/02_Collectives_and_NCCL.md).

**Batching amortizes weight reads.** A batch of N decodes reads the model weights from HBM once per step instead of N times. This is the entire reason continuous batching exists. See [Batching_and_Scheduling](L8_Inference_and_Serving/03_Batching_and_Scheduling.md).

**The reasoning tax.** Test-time-compute scaling (o1/R1-style) shifts cost from training to inference and from short to long generations. Decode pools must grow 10–100x larger than chat pools per request. See [Reasoning_Models](L7_Training_Stack/06_Reasoning_Models.md).

**MLA + MoE compose.** DeepSeek-V3/V4 Pro demonstrates that compressing KV cache (MLA) and active compute (MoE) are orthogonal — stacking them gives huge effective scale on modest hardware. See [Frontier_Models_2025_2026](L6_Algorithms_and_Models/07_Frontier_Models_2025_2026.md), [Modern_MoE](L6_Algorithms_and_Models/03_Modern_MoE.md).

**MoE is now standard, not niche.** By mid-2026, Mixture-of-Experts is the default architecture across nearly all frontier model families — both open and closed. The "total-A-active" notation (e.g., 397B A17B) is now the industry standard. Expert parallelism, all-to-all communication, and load balancing are first-class concerns for every inference deployment. See [Frontier_Models_2025_2026](L6_Algorithms_and_Models/07_Frontier_Models_2025_2026.md), [Modern_MoE](L6_Algorithms_and_Models/03_Modern_MoE.md).

**1-bit LLMs as extreme quantization endpoint.** BitNet demonstrates that ternary/1-bit weight representations are viable at scale, potentially eliminating the multiplier entirely. BitNet/bitnet.cpp inference is a new category in the framework landscape. See [Quantization](L6_Algorithms_and_Models/05_Quantization.md), [Cutting_Edge_Kernels](L5_Kernels_and_Programming/06_Cutting_Edge_Kernels.md).

**Disaggregated serving is now production, not just research.** NVIDIA Dynamo 1.0 ships disaggregated prefill/decode as a first-class feature with NIXL transport and global KV pooling. Meta's llm-d provides a vendor-neutral open-source counterpart. SGLang and vLLM both support disaggregated modes natively. Attention-FFN disaggregation extends the idea beyond prefill/decode. See [Disaggregated_Serving_2025](L8_Inference_and_Serving/10_Disaggregated_Serving_2025.md).

**NVIDIA-Groq acquisition enables heterogeneous GPU+LPU inference.** The $20B acquisition (2026) gives NVIDIA access to deterministic SRAM-only inference at microsecond latency alongside traditional GPU throughput. GroqLink compilers are being integrated into the CUDA ecosystem. See [Specialty_Accelerators](L3_Microarchitecture/09_Specialty_Accelerators.md), [Accelerator_Landscape_2026](L3_Microarchitecture/10_Accelerator_Landscape_2026.md).

**Framework convergence toward compiler-style IR.** vLLM's V1 redesign introduces a compiler-style IR layer for model execution. TensorRT-LLM compiles models to optimized execution graphs. The trend is away from eager-mode execution and toward graph-level optimization with compilation. See [vLLM_Internals](L8_Inference_and_Serving/09_vLLM_Internals.md), [Inference_Frameworks](L8_Inference_and_Serving/08_Inference_Frameworks.md).

**Aux-loss-free balancing.** Replacing the load-balancing auxiliary loss with a per-expert EMA bias removes a quality tax that previously hurt large MoE. Standard in 2025–2026 designs.

**FP4 native.** Blackwell makes FP4 a hardware-supported tensor-core format, doubling FP8 throughput with careful calibration. The next throughput multiplier after FP8. See [Blackwell_Architecture](L3_Microarchitecture/04_Blackwell_Architecture.md), [Modern_Quantization_Frontier](L6_Algorithms_and_Models/06_Modern_Quantization_Frontier.md).

**Cluster-scale KV pool.** Mooncake-style global pools push prefix-cache hit rates to 80%+ at fleet scale, where per-replica caches plateau at 30–50%. Standard in NVIDIA Dynamo / llm-d. See [Disaggregated_Serving_2025](L8_Inference_and_Serving/10_Disaggregated_Serving_2025.md).

**Inference-engine-inside-RL.** Modern post-training (GRPO, online PPO) is dominated by inference cost; the inference engine becomes a major component of the training stack. See [Modern_Post_Training](L7_Training_Stack/05_Modern_Post_Training.md).

---

## Suggested Reading Order

**For a 2-week interview sprint:**

Week 1 (mechanism): GPU_Architecture → Memory_Hierarchy → Transformer_Internals → Attention_Mechanisms → KV_Cache → Batching_and_Scheduling → Parallelism_Strategies.

Week 2 (production + interview): Prefill_Decode_Disaggregation → Quantization → vLLM_Internals → Kubernetes_and_Orchestration → Observability → System_Design_Interview → Common_Interview_Questions.

**For a CUDA/kernel-engineer interview:**

CUDA_Programming → CUDA_Optimization → Tensor_Core_Programming → Triton_and_Kernels → FlashAttention_Deep_Dive → Attention_Mechanisms → Cutting_Edge_Kernels → Coding_Patterns.

**For a distributed-training interview:**

GPU_Architecture → Networking_and_Interconnect → Parallelism_Strategies → Collectives_and_NCCL → Distributed_Training → Training_Optimization → System_Design_Interview.

**For a frontier-only crash course (~one weekend):**

Frontier_Models_2025_2026 → Modern_MoE → Reasoning_Models → Long_Context_Engineering → Modern_KV_Compression → Disaggregated_Serving_2025 → Blackwell_Architecture → Modern_Quantization_Frontier → Modern_Post_Training → Cutting_Edge_Kernels → State_Space_Models_and_Hybrids → Multimodal_Inference.

**For 2026 LLM-inference engineer interviews specifically:**

Skim L3–L8 for fluency; deep-read: Frontier_Models_2025_2026, Reasoning_Models, Disaggregated_Serving_2025, Modern_KV_Compression, Modern_Quantization_Frontier, vLLM_Internals. Plus System_Design_Interview.

**For 2026 GPU-kernel engineer interviews specifically:**

CUDA_Programming → CUDA_Optimization → Tensor_Core_Programming → FlashAttention_Deep_Dive → Blackwell_Architecture → Modern_Quantization_Frontier → Cutting_Edge_Kernels → Coding_Patterns.

---

## Numbers to Memorize

These numbers come up constantly. Burn them into memory before an interview.

| Quantity | Value | Why it matters |
|---|---|---|
| H100 SXM HBM bandwidth | 3.35 TB/s | Decode bottleneck |
| H100 FP16 tensor TFLOPS | ~990 | Prefill ceiling |
| H100 FP8 tensor TFLOPS | ~1980 | Post-quant prefill |
| B200 SM count | 128 (confirmed) | Dual-die, 2× 64-SM |
| B200 HBM bandwidth | 8 TB/s | Next-gen decode |
| B200 HBM capacity | 192 GB | Fits 70B BF16 + KV |
| B200 FP8 tensor TFLOPS | ~4500 | Blackwell prefill ceiling |
| B200 FP4 tensor TFLOPS | ~9000 | NVFP4 production target |
| B300 / Blackwell Ultra HBM | 288 GB @ 12 TB/s | 2026 sweet spot |
| Rubin R100/R200 (GTC Mar 2026, volume H2 2026) | 336 B transistors, 288 GB HBM4, 22 TB/s, ~25 PFLOP FP8, 50 PFLOP NVFP4; VR200 NVL144 rack 3.3× GB300 FP4, ~190–230 kW | Next generation |
| MI455X Altair (CES 2026, ships H2 2026) | 320 B transistors, 432 GB HBM4, 19.6 TB/s, 20 PFLOP FP8, 40 PFLOP FP4; Helios 72-GPU: 31 TB, 1.4 PB/s, 2.9 EF FP4 | AMD next-gen |
| TPU v7 Ironwood | 192 GB HBM3e, 7.37 TB/s, 4614 FP8 TFLOPS, 9216-chip pod | Google production |
| TPU 8t / 8i (Apr 2026) | 8t: 12.6 PF FP4, 216 GB, 9600-chip superpod (2 PB HBM); 8i: 10.1 PF FP4, 288 GB @ 8.6 TB/s, 384 MB SRAM | Google next-gen, GA H2 2026 |
| Groq LPU II / 3 LPX (NVIDIA-Groq) | deterministic SRAM-only inference, $20B acquisition | Microsecond-latency inference |
| NVLink 5 per-GPU | 1.8 TB/s | Intra-node collectives |
| NVLink 6 per-GPU (Rubin) | ~3.6 TB/s (announced) | Next-gen intra-node |
| NVSwitch domain | 72-GPU (NVL72), 576-GPU (NVL576, Rubin) | Tensor-parallel scope |
| NVL72 HBM total (B200) | 13.5 TB | Cluster KV pool size |
| InfiniBand NDR | 400 Gbps (50 GB/s) | Inter-node |
| InfiniBand XDR | 800 Gbps (100 GB/s) | Quantum-X800 era |
| PCIe Gen5 x16 | 64 GB/s | Host ↔ GPU |
| NVLink-C2C (Grace-Hopper/Blackwell) | 900 GB/s | CPU↔GPU in superchip |
| H100 ridge point (FP16) | ~295 ops/byte | Roofline inflection |
| Llama-3-70B KV per token (FP16, GQA) | 320 KB | 2 × 80 × 8 × 128 × 2 |
| Llama-3-70B weights (FP16 / FP8 / INT4 / FP4) | 140 / 70 / 35 / 35 GB | Quant options |
| DeepSeek-V4 Pro KV per token (MLA) | ~70 KB | ~8.5× smaller than dense MHA |
| DeepSeek-V4 Pro active / total params | 37B / 671B | MoE asymmetry |
| Llama-4 Maverick active / total | 17B / 400B | Top-1 routing |
| Llama-4 Scout context window | 10M tokens | Extreme context target |
| Qwen-3.6 Max Preview Intelligence Index | 52 | Top open-weights contender |
| Kimi K2.6 Intelligence Index | 54 | Highest-ranked open-weights |
| Typical TTFT SLO (chat) | <500 ms | Interactive UX |
| Typical TPOT SLO (chat) | <50 ms | Smooth streaming |
| Reasoning-mode output length (heavy) | 30K–100K tokens | KV pool sizing target |
| GRPO sample group K | 8–64 | Per-prompt rollouts |
| Mooncake-style cluster prefix hit rate | ~80% | Multi-replica chat workloads |
| Cerebras WSE-3 die area | 46,225 mm² (full wafer) | Wafer-scale compute |
| Cerebras IPO date | May 15, 2026 | Public market milestone |

---

*Last updated: May 2026.*

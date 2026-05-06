# AI Infrastructure — Master Index

> A senior-level technical notebook for AI infrastructure interviews and practice.
> Target roles: LLM inference/serving engineer, GPU kernel/CUDA engineer, distributed training/MLSys engineer.
> Style: deep theory + proofs + worked examples + production-grade tradeoffs + 2025–2026 frontier coverage.
>
> **65 pages** organized in 9 layers (L0–L8) plus interview prep. The layers follow a strict bottom-up dependency chain: silicon physics up through inference serving.

---

## How to Use This Notebook

Each page is a standalone deep-dive on one topic. Pages are sequenced so that later pages assume the contents of earlier ones, but every page stands on its own for review. The style follows the rest of the notebook: "explain as if the reader is a senior engineer who needs the mechanism, the math, the numbers, and the tradeoffs — not a 101 summary."

Three layers of preparation are interwoven:

1. **Mechanism layer** — what the system actually does at the hardware/algorithm level. Proofs, diagrams, arithmetic.
2. **Engineering layer** — why production systems make the tradeoffs they do. Capacity math, SLAs, failure modes.
3. **Interview layer** — how to articulate the above under time pressure. Sample Q&A, system design walkthroughs.

---

## L0 — Silicon and Process

1. [Silicon For AI](L0_Silicon_and_Process/Silicon_For_AI.md) — CMOS scaling, FinFET/GAA transistors, process nodes (N5/N3/N2), SRAM/Logic scaling divergence, dark silicon, voltage-frequency tradeoffs, defect density, wafer economics.

## L1 — Packaging and Memory

2. [Advanced Packaging](L1_Packaging_and_Memory/Advanced_Packaging.md) — 2.5D (CoWoS), 3D stacking, chiplet interconnects (UCIe), reticle limits, TSV density, thermal budgets, interposer routing.
3. [HBM Deep Dive](L1_Packaging_and_Memory/HBM_Deep_Dive.md) — HBM3/HBM3E/HBM4 architecture, channel organization, bandwidth math, stack height, thermal limits, co-packaged vs discrete.

## L2 — Digital Design for AI

4. [Digital Design For AI](L2_Digital_Design_for_AI/Digital_Design_For_AI.md) — RTL for accelerators, dataflow vs von Neumann, control-flow design, clock gating, power domains.
5. [FP Unit Design](L2_Digital_Design_for_AI/FP_Unit_Design.md) — IEEE 754, FP16/BF16/FP8/FP4/INT8 hardware, multiplier trees, fused multiply-add, denormal handling.
6. [On-Chip Memory Hardware](L2_Digital_Design_for_AI/On_Chip_Memory_Hardware.md) — SRAM cell design, register files, scratchpad vs cache, area/energy per bit, bank conflict hardware.
7. [Systolic Arrays and Dataflow](L2_Digital_Design_for_AI/Systolic_Arrays_and_Dataflow.md) — weight-stationary / output-stationary / row-stationary dataflow, tiling, mapping convolutions and matmuls, throughput modeling.

## L3 — Microarchitecture

8. [GPU Architecture](L3_Microarchitecture/GPU_Architecture.md) — SM internals, tensor cores, Hopper/Blackwell specifics, execution model.
9. [Memory Hierarchy and Roofline](L3_Microarchitecture/Memory_Hierarchy_and_Roofline.md) — HBM, SRAM, registers, arithmetic intensity, the roofline model, ridge-point math.
10. [ISA and Execution Model](L3_Microarchitecture/ISA_and_Execution_Model.md) — PTX/SASS, warp scheduling, instruction pipeline, latency/throughput tables, divergence.
11. [Blackwell Architecture](L3_Microarchitecture/Blackwell_Architecture.md) — B100/B200/B300/GB200/GB300, NVLink-5, NVL72/NVL576, TMEM, FP4 hardware, Rubin outlook.
12. [AMD Instinct](L3_Microarchitecture/AMD_Instinct.md) — CDNA 3/4, MI300/MI400, XDNA, Infinity Fabric, ROCm compute model.
13. [Google TPU](L3_Microarchitecture/Google_TPU.md) — TPU v4/v5/v6, MXU systolic array, bfloat16, ICI, pod-scale architecture.
14. [Huawei Ascend](L3_Microarchitecture/Huawei_Ascend.md) — Da Vinci architecture, Ascend 910B/910C, CANN stack, cluster topology.
15. [Cloud ASICs](L3_Microarchitecture/Cloud_ASICs.md) — AWS Inferentia/Trainium2, Azure Maia, MTIA, in-house accelerators.
16. [Specialty Accelerators](L3_Microarchitecture/Specialty_Accelerators.md) — networking NPUs, video encode/decode ASICs, GenAI inference ASICs, edge NPUs.
17. [Accelerator Landscape 2026](L3_Microarchitecture/Accelerator_Landscape_2026.md) — comparative survey, performance/TCO benchmarks, roadmap outlook.

## L4 — Systems and Interconnects

18. [Networking and Interconnect](L4_Systems_and_Interconnects/Networking_and_Interconnect.md) — SerDes physics, PCIe/CXL/NVLink/UALink protocols, fat-tree/dragonfly/torus topologies, DCQCN, GPUDirect RDMA.
19. [Rack-Scale Design](L4_Systems_and_Interconnects/Rack_Scale_Design.md) — NVL72/Helios/TPU pod architectures, 48V power distribution, liquid cooling math, mechanical constraints.
20. [Storage and Model Loading](L4_Systems_and_Interconnects/Storage_and_Model_Loading.md) — Safetensors/pickle/GGUF formats, GDS, checkpoint I/O math, KV cache offload tiers.

## L5 — Kernels and Programming

21. [CUDA Programming](L5_Kernels_and_Programming/CUDA_Programming.md) — thread/block/warp hierarchy, memory model, streams, synchronization, end-to-end examples.
22. [CUDA Optimization](L5_Kernels_and_Programming/CUDA_Optimization.md) — coalescing, bank conflicts, occupancy, async copy, warp specialization, Hopper TMA.
23. [Triton and Kernels](L5_Kernels_and_Programming/Triton_and_Kernels.md) — OpenAI Triton DSL, tiling, autotuning, CUTLASS, writing a matmul, writing FlashAttention.
24. [FlashAttention Deep Dive](L5_Kernels_and_Programming/FlashAttention_Deep_Dive.md) — v1/v2/v3 algorithm walkthrough with tile math and pseudocode.
25. [Cutting-Edge Kernels](L5_Kernels_and_Programming/Cutting_Edge_Kernels.md) — CUTLASS 3.x, CuTe, FlashInfer, DeepEP, TileLang, ThunderKittens, Liger-Kernel; Hopper/Blackwell tile programming.

## L6 — Algorithms and Models

26. [Transformer Internals](L6_Algorithms_and_Models/Transformer_Internals.md) — the full forward pass, positional encodings (RoPE, ALiBi), normalization, parameter counts.
27. [Attention Mechanisms](L6_Algorithms_and_Models/Attention_Mechanisms.md) — scaled dot-product, MHA/MQA/GQA/MLA, online softmax derivation, KV cache formulas.
28. [Modern MoE](L6_Algorithms_and_Models/Modern_MoE.md) — DeepSeekMoE, fine-grained / shared experts, aux-loss-free balancing, top-k routing, DeepEP.
29. [State Space Models and Hybrids](L6_Algorithms_and_Models/State_Space_Models_and_Hybrids.md) — Mamba-2, RWKV-7, Jamba, Zamba, hybrid SSM-Transformer stacks; serving engineering.
30. [Quantization](L6_Algorithms_and_Models/Quantization.md) — FP16/BF16/FP8/FP4/INT8/INT4, GPTQ, AWQ, SmoothQuant, KV cache quantization, error bounds.
31. [Modern Quantization Frontier](L6_Algorithms_and_Models/Modern_Quantization_Frontier.md) — MXFP4, NVFP4, FP6, FP4 inference and training, SpinQuant, Transformer Engine.
32. [Frontier Models 2025–2026](L6_Algorithms_and_Models/Frontier_Models_2025_2026.md) — Llama-4, DeepSeek-V3/R1, Qwen-3, Gemma-3, GPT-5/o3, Claude-4, Gemini-2.5; MLA, MTP.

## L7 — Training Stack

33. [Parallelism Strategies](L7_Training_Stack/Parallelism_Strategies.md) — DP, TP, PP, EP, CP, SP, hybrid/3D, communication-volume math.
34. [Collectives and NCCL](L7_Training_Stack/Collectives_and_NCCL.md) — AllReduce algorithms (ring, tree, recursive halving-doubling), NCCL internals, SHARP, bandwidth modeling.
35. [Distributed Training](L7_Training_Stack/Distributed_Training.md) — FSDP, ZeRO-1/2/3, checkpointing, fault tolerance, elastic training.
36. [Training Optimization](L7_Training_Stack/Training_Optimization.md) — mixed precision (AMP), activation checkpointing, gradient accumulation, Transformer Engine.
37. [Modern Post-Training](L7_Training_Stack/Modern_Post_Training.md) — DPO/IPO/KTO/SimPO/ORPO, GRPO, online RL infra, distillation from reasoning teachers.
38. [Reasoning Models](L7_Training_Stack/Reasoning_Models.md) — o1/o3/R1 architecture, long-CoT RL, test-time compute scaling, thinking budgets.

## L8 — Inference and Serving

39. [KV Cache](L8_Inference_and_Serving/KV_Cache.md) — layout, memory math, PagedAttention, prefix caching, radix tree, offloading.
40. [Modern KV Compression](L8_Inference_and_Serving/Modern_KV_Compression.md) — StreamingLLM, H2O, KIVI, SnapKV, Quest, MLA.
41. [Batching and Scheduling](L8_Inference_and_Serving/Batching_and_Scheduling.md) — continuous batching, chunked prefill, admission control, preemption.
42. [Speculative Decoding](L8_Inference_and_Serving/Speculative_Decoding.md) — vanilla, self-speculation, Medusa, EAGLE, acceptance rate math.
43. [Prefill-Decode Disaggregation](L8_Inference_and_Serving/Prefill_Decode_Disaggregation.md) — motivation, architecture, transfer mechanics, real benchmarks.
44. [Long Context Engineering](L8_Inference_and_Serving/Long_Context_Engineering.md) — YaRN, LongRoPE, iRoPE, NSA, MoBA, ring attention, 1M–10M context.
45. [Multimodal Inference](L8_Inference_and_Serving/Multimodal_Inference.md) — VLMs (Qwen-VL, InternVL, Llama-4-MM), audio, video gen, unified models.
46. [Inference Frameworks](L8_Inference_and_Serving/Inference_Frameworks.md) — vLLM, SGLang, TensorRT-LLM, NVIDIA Dynamo, llm-d, TGI — feature-by-feature comparison.
47. [vLLM Internals](L8_Inference_and_Serving/vLLM_Internals.md) — engine architecture, scheduler, workers, executor, block manager, APC.
48. [Disaggregated Serving 2025](L8_Inference_and_Serving/Disaggregated_Serving_2025.md) — Mooncake, DistServe, Splitwise, Sarathi-Serve; NVIDIA Dynamo, llm-d; KV pools.
49. [Kubernetes and Orchestration](L8_Inference_and_Serving/Kubernetes_and_Orchestration.md) — GPU operator, device plugin, MIG, topology-aware scheduling, HPA/KEDA, OME/llm-d.
50. [Observability and Debugging](L8_Inference_and_Serving/Observability_and_Debugging.md) — TTFT/ITL/TPOT metrics, DCGM, Nsight, debugging flow for common pathologies.
51. [Production Architecture](L8_Inference_and_Serving/Production_Architecture.md) — reference stack, capacity planning, cost modeling, failure modes.

## Interview Prep

52. [System Design Interview](interview_prep/System_Design_Interview.md) — design ChatGPT, multi-tenant inference, training cluster, RAG, agent orchestrator, eval harness.
53. [Common Interview Questions](interview_prep/Common_Interview_Questions.md) — conceptual Q&A across all topics.
54. [Coding Patterns](interview_prep/Coding_Patterns.md) — CUDA kernel patterns, parallel reduction, scheduler algorithms, numerical algorithms, systems C++.

---

## Cross-Cutting Themes

These show up on almost every page — if you can pattern-match them quickly, interviews become much easier.

**The memory wall.** LLM inference is bandwidth-bound at decode. Every optimization (FlashAttention, PagedAttention, speculation, quantization, GQA, MLA) is a way to reduce bytes moved per token. See [Memory_Hierarchy_and_Roofline](L3_Microarchitecture/Memory_Hierarchy_and_Roofline.md).

**Arithmetic intensity.** Operations live on a roofline: FLOPs per byte decides whether hardware bandwidth or FLOPS is the bottleneck. Prefill is compute-bound, decode is memory-bound. See [Memory_Hierarchy_and_Roofline](L3_Microarchitecture/Memory_Hierarchy_and_Roofline.md).

**Phase interference.** Prefill and decode have opposite hardware profiles. Co-locating them on the same GPU creates head-of-line blocking. Modern systems disaggregate. See [Prefill_Decode_Disaggregation](L8_Inference_and_Serving/Prefill_Decode_Disaggregation.md).

**The KV cache dominates capacity.** At long contexts, KV cache exceeds model weights. Every capacity planning exercise reduces to: layers × kv_heads × head_dim × 2 × bytes × tokens × concurrent_requests.

**Collectives are latency-critical.** TP AllReduce happens per-layer. The NVLink bandwidth of your node directly caps your latency floor. See [Collectives_and_NCCL](L7_Training_Stack/Collectives_and_NCCL.md).

**Batching amortizes weight reads.** A batch of N decodes reads the model weights from HBM once per step instead of N times. This is the entire reason continuous batching exists. See [Batching_and_Scheduling](L8_Inference_and_Serving/Batching_and_Scheduling.md).

**The reasoning tax.** Test-time-compute scaling (o1/R1-style) shifts cost from training to inference and from short to long generations. Decode pools must grow 10–100× larger than chat pools per request. See [Reasoning_Models](L7_Training_Stack/Reasoning_Models.md).

**MLA + MoE compose.** DeepSeek-V3 demonstrates that compressing KV cache (MLA) and active compute (MoE) are orthogonal — stacking them gives huge effective scale on modest hardware. See [Frontier_Models_2025_2026](L6_Algorithms_and_Models/Frontier_Models_2025_2026.md), [Modern_MoE](L6_Algorithms_and_Models/Modern_MoE.md).

**Aux-loss-free balancing.** Replacing the load-balancing auxiliary loss with a per-expert EMA bias removes a quality tax that previously hurt large MoE. Standard in 2025 designs.

**FP4 native.** Blackwell makes FP4 a hardware-supported tensor-core format, doubling FP8 throughput with careful calibration. The next throughput multiplier after FP8. See [Blackwell_Architecture](L3_Microarchitecture/Blackwell_Architecture.md), [Modern_Quantization_Frontier](L6_Algorithms_and_Models/Modern_Quantization_Frontier.md).

**Cluster-scale KV pool.** Mooncake-style global pools push prefix-cache hit rates to 80%+ at fleet scale, where per-replica caches plateau at 30–50%. Standard in NVIDIA Dynamo / llm-d. See [Disaggregated_Serving_2025](L8_Inference_and_Serving/Disaggregated_Serving_2025.md).

**Inference-engine-inside-RL.** Modern post-training (GRPO, online PPO) is dominated by inference cost; the inference engine becomes a major component of the training stack. See [Modern_Post_Training](L7_Training_Stack/Modern_Post_Training.md).

---

## Suggested Reading Order

**For a 2-week interview sprint:**

Week 1 (mechanism): GPU_Architecture → Memory_Hierarchy → Transformer_Internals → Attention_Mechanisms → KV_Cache → Batching_and_Scheduling → Parallelism_Strategies.

Week 2 (production + interview): Prefill_Decode_Disaggregation → Quantization → vLLM_Internals → Kubernetes_and_Orchestration → Observability → System_Design_Interview → Common_Interview_Questions.

**For a CUDA/kernel-engineer interview:**

CUDA_Programming → CUDA_Optimization → Triton_and_Kernels → FlashAttention_Deep_Dive → Attention_Mechanisms → Cutting_Edge_Kernels → Coding_Patterns.

**For a distributed-training interview:**

GPU_Architecture → Networking_and_Interconnect → Parallelism_Strategies → Collectives_and_NCCL → Distributed_Training → Training_Optimization → System_Design_Interview.

**For a frontier-only crash course (~one weekend):**

Frontier_Models_2025_2026 → Modern_MoE → Reasoning_Models → Long_Context_Engineering → Modern_KV_Compression → Disaggregated_Serving_2025 → Blackwell_Architecture → Modern_Quantization_Frontier → Modern_Post_Training → Cutting_Edge_Kernels → State_Space_Models_and_Hybrids → Multimodal_Inference.

**For 2026 LLM-inference engineer interviews specifically:**

Skim L3–L8 for fluency; deep-read: Frontier_Models_2025_2026, Reasoning_Models, Disaggregated_Serving_2025, Modern_KV_Compression, Modern_Quantization_Frontier. Plus System_Design_Interview.

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
| Reasoning-mode output length (heavy) | 30K–100K tokens | KV pool sizing target |
| GRPO sample group K | 8–64 | Per-prompt rollouts |
| Mooncake-style cluster prefix hit rate | ~80% | Multi-replica chat workloads |

---

*Last updated: May 2026.*

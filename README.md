# AI Infrastructure Notebook

A comprehensive, bottom-up technical reference covering the full AI infrastructure stack — from CMOS physics and silicon fabrication through GPU microarchitecture, CUDA kernels, transformer algorithms, distributed training, and production LLM inference serving.

**65 pages | 31,000+ lines | 240 Mermaid diagrams | LaTeX math throughout**

Built for senior-level interview preparation and professional reference. Target roles: LLM inference/serving engineer, GPU kernel/CUDA engineer, distributed training/MLSys engineer.

## Structure

The notebook is organized in 9 layers following a strict bottom-up dependency chain. Each layer assumes the content of all layers below it.

```
ai_infra/
├── L0_Silicon_and_Process/          CMOS, FinFET/GAA, process nodes, wafer economics
├── L1_Packaging_and_Memory/         2.5D/3D packaging, HBM3/HBM4, chiplet interconnects
├── L2_Digital_Design_for_AI/        FP units, systolic arrays, on-chip memory, dataflow
├── L3_Microarchitecture/            GPU/TPU/ASIC architectures, roofline model, ISA
├── L4_Systems_and_Interconnects/    Networking, rack-scale design, storage, model loading
├── L5_Kernels_and_Programming/      CUDA, Triton, FlashAttention, CUTLASS, Hopper kernels
├── L6_Algorithms_and_Models/        Transformers, attention, MoE, quantization, SSMs
├── L7_Training_Stack/               Parallelism, collectives, FSDP/ZeRO, post-training, RL
├── L8_Inference_and_Serving/        KV cache, batching, disaggregation, frameworks, production
└── interview_prep/                  System design walkthroughs, Q&A, coding patterns
```

### Page Index

| Layer | Pages | Lines | Coverage |
|-------|-------|-------|----------|
| L0 — Silicon & Process | 2 | 742 | CMOS scaling, process nodes, wafer economics |
| L1 — Packaging & Memory | 3 | 1,157 | CoWoS, HBM3/3E/4, chiplets, TSVs |
| L2 — Digital Design | 5 | 2,145 | FP units, systolic arrays, SRAM, RTL |
| L3 — Microarchitecture | 11 | 3,348 | H100/B200, TPU v5, AMD MI300, roofline |
| L4 — Systems & Interconnects | 4 | 1,657 | NVLink, InfiniBand, NVL72, liquid cooling |
| L5 — Kernels & Programming | 6 | 4,012 | CUDA, Triton, FlashAttention v1–v3, CUTLASS 3.x |
| L6 — Algorithms & Models | 8 | 4,090 | Transformers, MLA, MoE, quantization, SSMs |
| L7 — Training Stack | 7 | 4,089 | 3D parallelism, NCCL, FSDP, GRPO, reasoning models |
| L8 — Inference & Serving | 14 | 8,435 | vLLM, SGLang, KV cache, disaggregation, K8s, prod |
| Interview Prep | 3 | 1,384 | System design, Q&A, CUDA coding patterns |

## Content Style

Every page follows the same structure:

- **Section 0: Why this page exists** — one-paragraph framing of what problem this page solves
- **Numbered sections** — deep technical content with derivations, not summaries
- **Mermaid diagrams** — architecture, dataflow, cause-effect chains
- **LaTeX math** — all formulas fully derived, not just stated
- **Numbers to memorize** — tables of constants that come up in interviews
- **Worked problems** — 4–5 interview-style problems with full solutions
- **Cross-references** — links to prerequisite and downstream pages

The tone is declarative: "here is the mechanism, the math, the numbers, and the tradeoffs" — written for a senior engineer, not a beginner tutorial.

## Reading Paths

**2-week interview sprint:**

Week 1 (mechanism): GPU Architecture → Memory Hierarchy → Transformer Internals → Attention → KV Cache → Batching → Parallelism

Week 2 (production): Prefill-Decode Disaggregation → Quantization → vLLM Internals → Kubernetes → Observability → System Design Interview

**CUDA/kernel interview:**

CUDA Programming → CUDA Optimization → Triton → FlashAttention → Cutting-Edge Kernels → Coding Patterns

**Distributed training interview:**

GPU Architecture → Networking → Parallelism Strategies → NCCL → Distributed Training → Training Optimization

**Frontier crash course (weekend):**

Frontier Models → MoE → Reasoning Models → Long Context → KV Compression → Disaggregated Serving → Blackwell → Quantization Frontier

## How to Use

**Obsidian (recommended):** Open this folder as a vault. All cross-references, Mermaid diagrams, and LaTeX render natively. The included `mermaid-zoom.css` snippet auto-fits diagrams to the reading pane.

**GitHub:** All pages are readable directly on GitHub. Mermaid diagrams and LaTeX math render in the GitHub markdown preview.

**Any markdown viewer:** Pages use standard markdown with relative links — they work anywhere.

## Topics Covered

**Hardware:** CMOS physics, FinFET/GAA transistors, TSMC N5/N3/N2, CoWoS/SoIC packaging, HBM3E/HBM4, GPU SM architecture, tensor cores, NVLink 4/5, NVL72/NVL576, InfiniBand NDR/XDR, PCIe Gen5, CXL, UALink, Grace-Blackwell superchips, AMD MI300/MI400, Google TPU v5p/v6, Huawei Ascend 910, cloud ASICs (Trainium2, Inferentia2, MTIA)

**Kernels:** CUDA thread hierarchy, memory coalescing, bank conflicts, occupancy, TMA (Hopper), warp specialization, Triton DSL, CUTLASS 3.x/CuTe, FlashAttention v1/v2/v3, FlashInfer, DeepEP, Liger Kernel, ThunderKittens

**Algorithms:** Transformer forward pass, RoPE/ALiBi, MHA/MQA/GQA/MLA, online softmax, MoE gating & routing, aux-loss-free balancing, Mamba-2/RWKV-7, GPTQ/AWQ/SmoothQuant, FP8/FP4/MXFP4 quantization, Transformer Engine

**Training:** Data/tensor/pipeline/expert/sequence parallelism, AllReduce algorithms, NCCL internals, SHARP, FSDP/ZeRO-1/2/3, gradient checkpointing, mixed precision, fault tolerance, PPO/DPO/GRPO, reasoning models (o1/o3/R1), test-time compute

**Inference:** KV cache memory math, PagedAttention, prefix caching, continuous batching, chunked prefill, speculative decoding (Medusa/EAGLE), prefill-decode disaggregation, long context (YaRN/NSA/MoBA), multimodal inference, vLLM/SGLang/TRT-LLM/Dynamo/llm-d internals, Kubernetes GPU orchestration, observability, production architecture

**Interview:** System design walkthroughs (LLM API, training cluster, RAG, agent orchestrator), conceptual Q&A, CUDA kernel coding patterns, scheduler algorithms, capacity math

## License

MIT — use freely for study, interview prep, or teaching.

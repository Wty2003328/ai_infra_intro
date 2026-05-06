# AI Infrastructure & Hardware Design Notebook

A comprehensive, bottom-up technical reference covering the full AI infrastructure stack and digital IC design — from CMOS physics and silicon fabrication through GPU microarchitecture, CUDA kernels, transformer algorithms, distributed training, production LLM inference serving, CPU design, physical implementation, and SystemVerilog verification.

**92 pages | 65,000+ lines | 240 Mermaid diagrams | LaTeX math throughout**

Built for senior-level interview preparation and professional reference. Target roles: LLM inference/serving engineer, GPU kernel/CUDA engineer, distributed training/MLSys engineer, RTL design engineer, physical design/STA engineer, verification engineer.

## Structure

```
notebook/
├── ai_infra/                         AI infrastructure (silicon → inference serving)
│   ├── L0_Silicon_and_Process/
│   ├── L1_Packaging_and_Memory/
│   ├── L2_Digital_Design_for_AI/
│   ├── L3_Microarchitecture/
│   ├── L4_Systems_and_Interconnects/
│   ├── L5_Kernels_and_Programming/
│   ├── L6_Algorithms_and_Models/
│   ├── L7_Training_Stack/
│   ├── L8_Inference_and_Serving/
│   └── interview_prep/
└── hardware_design/                  Digital IC design & verification
    ├── Fundamentals/
    ├── Architecture/
    ├── Implementation/
    ├── Clocking_and_Signals/
    ├── Power/
    └── SystemVerilog/
```

---

## AI Infrastructure (`ai_infra/`)

65 pages organized in 9 layers following a strict bottom-up dependency chain. Each layer assumes the content of all layers below it.

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

**Topics:** GPU/TPU/ASIC architectures, CUDA/Triton kernels, transformer internals, MoE, quantization (FP8/FP4/MX), distributed training (FSDP/ZeRO/3D parallelism), KV cache, continuous batching, speculative decoding, prefill-decode disaggregation, vLLM/SGLang/TRT-LLM internals, Kubernetes orchestration, production serving.

See [ai_infra/Index.md](ai_infra/Index.md) for the full page index and reading paths.

---

## Hardware Design (`hardware_design/`)

27 pages organized in 6 thematic sections covering digital IC design from transistor physics through verification signoff.

| Section | Pages | Lines | Coverage |
|---------|-------|-------|----------|
| Fundamentals | 5 | ~5,669 | CMOS physics, fabrication, logic basics, adders, floating point |
| Architecture | 3 | ~3,593 | CPU pipeline, memory design, AMBA bus protocols |
| Implementation | 6 | ~8,009 | Synthesis, PnR, STA, DFT/ATPG, formal verification, IC packaging |
| Clocking & Signals | 3 | ~4,827 | Clock dividers, async/CDC, signal integrity, reliability |
| Power | 5 | ~5,697 | Power fundamentals, reduction, analysis/signoff, UPF, low-power design |
| SystemVerilog | 5 | ~5,336 | Data types, processes, OOP, assertions/coverage, IPC/verification |

**Topics:** MOSFET/FinFET physics, fab process (DUV/EUV), adder architectures, IEEE 754, 5-stage CPU pipeline, branch prediction, MESI coherence, AXI4/AHB/APB, SRAM/DRAM design, synthesis, place-and-route, static timing analysis (OCV/POCV), scan chains/ATPG, formal verification (LEC/model checking), async FIFOs, CDC, crosstalk/EM/IR drop, power gating/DVFS, UPF, SystemVerilog OOP/randomization, UVM-style testbench, SVA.

See [hardware_design/Index.md](hardware_design/Index.md) for the full page index and reading paths.

---

## Reading Paths

### AI Infrastructure

**2-week interview sprint:** GPU Architecture → Memory Hierarchy → Transformer Internals → Attention → KV Cache → Batching → Parallelism → Prefill-Decode Disaggregation → Quantization → vLLM Internals → Kubernetes → System Design Interview

**CUDA/kernel interview:** CUDA Programming → CUDA Optimization → Triton → FlashAttention → Cutting-Edge Kernels → Coding Patterns

**Distributed training interview:** GPU Architecture → Networking → Parallelism → NCCL → Distributed Training → Training Optimization

### Hardware Design

**RTL design interview:** CMOS Fundamentals → Basic Knowledge → Adders → Floating Point → CPU Architecture → Synthesis → STA → SystemVerilog basics

**Physical design / STA interview:** CMOS → Synthesis → Physical Design → STA → Signal Integrity → Power Analysis → Clock Division

**Verification interview:** Data Types → Procedural → OOP → Assertions → IPC and Verification → Formal Verification → Async/CDC

**Power-aware design interview:** Power Fundamentals → Low Power Design → Power Reduction → UPF → Power Analysis and Signoff

---

## Content Style

Every page follows the same structure:

- **Section 0: Why this page exists** — one-paragraph framing of what problem this page solves
- **Numbered sections** — deep technical content with derivations, not summaries
- **Mermaid diagrams** — architecture, dataflow, cause-effect chains
- **LaTeX math** — all formulas fully derived, not just stated
- **Numbers to memorize** — tables of constants that come up in interviews
- **Worked problems** — interview-style problems with full solutions
- **Cross-references** — links to prerequisite and downstream pages

The tone is declarative: "here is the mechanism, the math, the numbers, and the tradeoffs" — written for a senior engineer, not a beginner tutorial.

## How to Use

**Obsidian (recommended):** Open this folder as a vault. All cross-references, Mermaid diagrams, and LaTeX render natively. The included `mermaid-zoom.css` snippet auto-fits diagrams to the reading pane.

**GitHub:** All pages are readable directly on GitHub. Mermaid diagrams and LaTeX math render in the GitHub markdown preview.

**Any markdown viewer:** Pages use standard markdown with relative links — they work anywhere.

## License

MIT — use freely for study, interview prep, or teaching.

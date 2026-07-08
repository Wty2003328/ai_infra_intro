# Silicon to Serving

**The AI infrastructure & chip-design notebook — from CMOS physics to production LLM serving.**

A comprehensive, bottom-up technical reference covering the full AI infrastructure stack and digital IC design — from silicon fabrication through GPU microarchitecture, CUDA kernels and ML compilers, transformer algorithms, distributed training, production inference serving, RISC-V CPU design, out-of-order execution, cache hierarchy, NoC, and SystemVerilog/UVM verification.

**108 pages | 100,000+ lines | 290+ Mermaid diagrams | LaTeX math throughout**

Built for senior-level interview preparation and professional reference. Target roles: LLM inference/serving engineer, GPU kernel/CUDA engineer, distributed training/MLSys engineer, RTL design engineer, physical design/STA engineer, verification engineer.

## Getting Started

```bash
git clone https://github.com/Wty2003328/silicon-to-serving.git
```

- **Obsidian (recommended):** open the cloned folder as a vault — cross-references, Mermaid, and LaTeX render natively.
- **GitHub:** every page renders directly in the GitHub preview (Mermaid + LaTeX supported).
- **Any markdown viewer:** standard markdown with relative links throughout.

Start at [ai_infra/Index.md](ai_infra/Index.md) or [hardware_design/Index.md](hardware_design/Index.md), or jump into a reading path below.

## Dependencies & Setup

To view this notebook exactly as intended, the following Obsidian configurations are required:

1. **Custom CSS Snippets (Code & Math Formatting)**
   * This project includes a custom CSS snippet (`snippet-formatting.css`) located in `.obsidian/snippets/`.
   * **Why it's needed:** It wraps long code lines and LaTeX equations (preventing ugly horizontal scrollbars), applies Mac-style UI headers to code blocks, and correctly sizes ASCII-art architectures.
   * **Setup:** Go to Obsidian Settings -> Appearance -> CSS snippets, and ensure `snippet-formatting` is enabled.

2. **Mermaid Zoom Plugin (Architecture Diagrams)**
   * It is **highly recommended** to install the [mermaid-zoom](https://github.com/xiaozhuang0433/mermaid-zoom) community plugin.
   * **Why it's needed:** The repository contains massive, highly dense hardware architecture diagrams. This plugin allows you to click any Mermaid diagram to open it in a full-screen, scrollable modal overlay, letting you fluidly pan across complex architectures without text squishing.
   * **Setup:** Go to Obsidian Settings -> Community Plugins -> Browse, search for `mermaid-zoom`, and install/enable it.

*(Note: If you opened this repository directly as an Obsidian vault by cloning the `.obsidian` folder, both the CSS snippet and the plugin are already installed and enabled for you. Just hit `Ctrl+R` / `Cmd+R` to ensure they load).*

## Structure

```ascii-graph
silicon-to-serving/
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
    ├── 00_Fundamentals/
    ├── 01_Architecture_and_PPA/
    ├── 02_Power_and_Low_Power/
    ├── 03_Frontend_RTL_and_Verification/
    ├── 04_Synthesis/
    ├── 05_Backend_Physical_Design/
    ├── 06_Signoff/
    ├── 07_Manufacturing_and_Bringup/
    └── interview_prep/
```

---

## AI Infrastructure (`ai_infra/`)

57 content pages (plus layer indexes) organized in 9 layers following a strict bottom-up dependency chain. Each layer assumes the content of all layers below it.

| Layer | Pages | Lines | Coverage |
|-------|-------|-------|----------|
| L0 — Silicon & Process | 1 | 832 | CMOS scaling, process nodes, wafer economics |
| L1 — Packaging & Memory | 2 | 1,280 | CoWoS, HBM3/3E/4, chiplets, TSVs |
| L2 — Digital Design | 4 | 2,173 | FP units, systolic arrays, SRAM, RTL |
| L3 — Microarchitecture | 10 | 4,054 | H100/B200/Rubin, TPU 8t/8i, AMD MI455X, roofline |
| L4 — Systems & Interconnects | 3 | 2,227 | NVLink, InfiniBand, NVL72, liquid cooling |
| L5 — Kernels & Programming | 6 | 5,282 | CUDA, Triton, FlashAttention v1–v3, CUTLASS 3.x, ML compilers (torch.compile/XLA/MLIR) |
| L6 — Algorithms & Models | 7 | 5,608 | Transformers, MLA, MoE, quantization, SSMs |
| L7 — Training Stack | 6 | 4,689 | 3D parallelism, NCCL, FSDP, GRPO, reasoning models |
| L8 — Inference & Serving | 15 | 10,580 | vLLM, SGLang, KV cache, disaggregation, agentic serving, GPU multi-tenancy, K8s, prod |
| Interview Prep | 3 | 1,585 | System design, Q&A, CUDA coding patterns |

**Topics:** GPU/TPU/ASIC architectures, CUDA/Triton kernels, ML compilers (torch.compile, CUDA Graphs, XLA/GSPMD, MLIR), transformer internals, MoE, quantization (FP8/FP4/MX), distributed training (FSDP/ZeRO/3D parallelism), KV cache, continuous batching, speculative decoding, prefill-decode disaggregation, agentic/tool-calling inference, GPU multi-tenancy (MIG/MPS/DRA), vLLM/SGLang/TRT-LLM internals, Kubernetes orchestration, production serving.

See [ai_infra/Index.md](ai_infra/Index.md) for the full page index and reading paths.

---

## Hardware Design (`hardware_design/`)

39 content pages organized in 7 thematic sections covering digital IC design from transistor physics through verification signoff, plus an interview-prep drill set.

| Section | Pages | Lines | Coverage |
|---------|-------|-------|----------|
| Fundamentals | 5 | 7,175 | CMOS physics, fabrication, logic basics, adders, floating point |
| Architecture | 12 | 22,274 | RISC-V ISA, OoO execution, branch prediction, caches, TLB, Xiangshan, DDR, ACE/CHI, CPU, memory, AXI, Network-on-Chip |
| Implementation | 6 | 14,375 | Synthesis, PnR, STA, DFT/ATPG, formal verification, IC packaging |
| Clocking & Signals | 3 | 5,116 | Clock dividers, async/CDC, signal integrity, reliability |
| Power | 5 | 6,453 | Power fundamentals, reduction, analysis/signoff, UPF, low-power design |
| SystemVerilog | 6 | 6,799 | Data types, processes, OOP, assertions/coverage, IPC/verification, UVM methodology |
| Interview Prep | 2 | 557 | RTL coding canon with solutions; worked timing/power/cache problems + snap answers |

**Topics:** RISC-V RV64G ISA (instruction encoding, privilege modes, Sv39/Sv48), out-of-order execution (ROB, issue queue, LSQ, rename, misprediction recovery), branch prediction (TAGE, gshare, perceptron, TAGE-SC as in Xiangshan), cache microarchitecture (MSHR, prefetch, MESI/MOESI), TLB and virtual memory (hardware page walker, VIPT), Xiangshan open-source RISC-V processor case study, DDR4/DDR5 memory controller, AMBA ACE/CHI coherence protocols, Network-on-Chip (topology/bisection math, wormhole/VC flow control, router microarchitecture, deadlock theory, CMN-class coherent mesh), multi-port SRAM, register file design, ECC (SECDED), CAM/TCAM, AXI CDC bridges, TrustZone, AXI ATOP, MOSFET/FinFET physics, adder architectures, IEEE 754, synthesis, STA (OCV/POCV), DFT/ATPG, formal verification, async FIFOs, power gating/DVFS, UPF, SystemVerilog OOP/randomization, UVM (phasing, sequences, factory, RAL), SVA.

See [hardware_design/Index.md](hardware_design/Index.md) for the full page index and reading paths.

---

## Reading Paths

### AI Infrastructure

**2-week interview sprint:** GPU Architecture → Memory Hierarchy → Transformer Internals → Attention → KV Cache → Batching → Parallelism → Prefill-Decode Disaggregation → Quantization → vLLM Internals → Agentic Inference → Kubernetes → System Design Interview

**CUDA/kernel interview:** CUDA Programming → CUDA Optimization → Triton → FlashAttention → Cutting-Edge Kernels → ML Compilers → Coding Patterns

**Distributed training interview:** GPU Architecture → Networking → Parallelism → NCCL → Distributed Training → Training Optimization

**Serving platform / multi-tenant fleet interview:** Batching → KV Cache → GPU Multi-Tenancy → Agentic Inference → Kubernetes → Production Architecture

### Hardware Design

**CPU design / microarchitecture interview:** RISC-V ISA → CPU Architecture → OoO Execution → Branch Prediction Deep Dive → Cache Microarchitecture → TLB and Virtual Memory → Xiangshan CPU Design

**Memory subsystem / SoC interconnect interview:** Memory → Cache Microarchitecture → TLB and Virtual Memory → DDR Controller → AHB/AXI/APB → ACE and CHI → Network-on-Chip

**RTL design interview:** CMOS Fundamentals → Basic Knowledge → Adders → Floating Point → CPU Architecture → RISC-V ISA → Synthesis → STA → SystemVerilog basics → RTL Coding Questions

**Physical design / STA interview:** CMOS → Synthesis → Physical Design → STA → Signal Integrity → Power Analysis → Clock Division

**Verification interview:** Data Types → Procedural → OOP → Assertions → IPC and Verification → UVM Methodology → Formal Verification → Async/CDC

**Power-aware design interview:** Power Fundamentals → Low Power Design → Power Reduction → UPF → Power Analysis and Signoff

**Interview week (hardware):** RTL Coding Questions → Hardware Interview Questions

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

## Currency

Updated June 2026: TPU 8t/8i, NVIDIA Vera Rubin / Rubin CPX, AMD MI455X "Altair" + Helios, vLLM V1, SGLang, GPT-5.5 / Claude Opus 4.7 / Gemini 3.5 / DeepSeek V4 Pro / Qwen 3.7 era coverage.

## License

[MIT](LICENSE) — use freely for study, interview prep, or teaching.

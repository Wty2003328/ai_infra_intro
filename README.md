# Silicon to Serving — AI Infrastructure Notebook

**The AI infrastructure notebook — from CMOS physics to production LLM serving.**

A comprehensive, bottom-up technical reference covering the full AI infrastructure stack — from silicon fabrication through GPU microarchitecture, CUDA kernels and ML compilers, transformer algorithms, distributed training, and production inference serving.

Built for senior-level interview preparation and professional reference. Target roles: LLM inference/serving engineer, GPU kernel/CUDA engineer, distributed training/MLSys engineer.

This notebook covers the AI-infrastructure half of the original combined project. The digital IC design / hardware half now lives in its own sibling notebook: **Silicon Design Notebook** (RISC-V CPU microarchitecture, RTL/verification, synthesis, physical design, signoff, fabrication).

## Getting Started

```bash
git clone https://github.com/Wty2003328/silicon-to-serving.git
```

- **Obsidian (recommended):** open the cloned folder as a vault — cross-references, Mermaid, and LaTeX render natively.
- **GitHub:** every page renders directly in the GitHub preview (Mermaid + LaTeX supported).
- **Any markdown viewer:** standard markdown with relative links throughout.

Start at [Index.md](Index.md), or jump into a reading path below.

## Dependencies & Setup

To view this notebook exactly as intended, the following Obsidian configurations are required:

1. **Custom CSS Snippets (Code & Math Formatting)**
   * This project includes a custom CSS snippet (`snippet-formatting.css`) located in `.obsidian/snippets/`.
   * **Why it's needed:** It wraps long code lines and LaTeX equations (preventing ugly horizontal scrollbars), applies Mac-style UI headers to code blocks, and correctly sizes ASCII-art architectures.
   * **Setup:** Go to Obsidian Settings -> Appearance -> CSS snippets, and ensure `snippet-formatting` is enabled.

2. **Mermaid Zoom Plugin (Architecture Diagrams)**
   * It is **highly recommended** to install the [mermaid-zoom](https://github.com/xiaozhuang0433/mermaid-zoom) community plugin.
   * **Why it's needed:** The repository contains massive, highly dense architecture diagrams. This plugin allows you to click any Mermaid diagram to open it in a full-screen, scrollable modal overlay, letting you fluidly pan across complex architectures without text squishing.
   * **Setup:** Go to Obsidian Settings -> Community Plugins -> Browse, search for `mermaid-zoom`, and install/enable it.

*(Note: If you opened this repository directly as an Obsidian vault by cloning the `.obsidian` folder, both the CSS snippet and the plugin are already installed and enabled for you. Just hit `Ctrl+R` / `Cmd+R` to ensure they load).*

## Structure

```ascii-graph
ai_infra_intro/                       AI infrastructure (silicon → inference serving)
├── Index.md                          master index — start here
├── L0_Silicon_and_Process/
├── L1_Packaging_and_Memory/
├── L2_Digital_Design_for_AI/
├── L3_Microarchitecture/
├── L4_Systems_and_Interconnects/
├── L5_Kernels_and_Programming/
├── L6_Algorithms_and_Models/
├── L7_Training_Stack/
├── L8_Inference_and_Serving/
└── interview_prep/
```

## Contents

66 content pages (plus layer indexes) organized in 9 layers following a strict bottom-up dependency chain. Each layer assumes the content of all layers below it. Pages within each folder are numbered in reading order.

| Layer | Pages | Lines | Coverage |
|-------|-------|-------|----------|
| L0 — Silicon & Process | 1 | 800 | CMOS scaling, process nodes, wafer economics |
| L1 — Packaging & Memory | 2 | 1,234 | CoWoS, HBM3/3E/4, chiplets, TSVs |
| L2 — Digital Design | 4 | 3,116 | FP units, systolic arrays, SRAM, RTL |
| L3 — Microarchitecture | 10 | 4,872 | GPU hardware/software and memory hierarchy, H100/B200/Rubin, TPU 8t/8i, AMD MI455X, roofline |
| L4 — Systems & Interconnects | 3 | 1,986 | NVLink, InfiniBand, NVL72, liquid cooling |
| L5 — Kernels & Programming | 7 | 4,969 | CUDA, tensor-core programming (WGMMA/TMA/warp specialization), Triton, FlashAttention v1–v4, CUTLASS 3.x, ML compilers (torch.compile/XLA/MLIR) |
| L6 — Algorithms & Models | 7 | 5,441 | Transformers, MLA, MoE, quantization, SSMs |
| L7 — Training Stack | 6 | 4,671 | 3D parallelism, NCCL, FSDP, GRPO, reasoning models |
| L8 — Inference & Serving | 15 | 10,418 | vLLM, SGLang, KV cache, disaggregation, agentic serving, GPU multi-tenancy, K8s, prod |
| Interview Prep | 11 | — | See [interview_prep/](interview_prep) — per-layer Q&A (L0-L8) plus system design, common Q&A, and coding patterns |

**Topics:** GPU/TPU/ASIC architectures, CUDA/Triton kernels, ML compilers (torch.compile, CUDA Graphs, XLA/GSPMD, MLIR), transformer internals, MoE, quantization (FP8/FP4/MX), distributed training (FSDP/ZeRO/3D parallelism), KV cache, continuous batching, speculative decoding, prefill-decode disaggregation, agentic/tool-calling inference, GPU multi-tenancy (MIG/MPS/DRA), vLLM/SGLang/TRT-LLM internals, Kubernetes orchestration, production serving.

See [Index.md](Index.md) for the full page index and reading paths.

## Reading Paths

**2-week interview sprint:** GPU Architecture (including hardware–software mapping) → Memory Hierarchy → Transformer Internals → Attention → KV Cache → Batching → Parallelism → Prefill-Decode Disaggregation → Quantization → vLLM Internals → Agentic Inference → Kubernetes → System Design Interview

**CUDA/kernel interview:** CUDA Programming → CUDA Optimization → Tensor Core Programming → Triton → FlashAttention → Cutting-Edge Kernels → ML Compilers → Coding Patterns

**Distributed training interview:** GPU Architecture → Networking → Parallelism → NCCL → Distributed Training → Training Optimization

**Serving platform / multi-tenant fleet interview:** Batching → KV Cache → GPU Multi-Tenancy → Agentic Inference → Kubernetes → Production Architecture

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

## History

The digital IC design / hardware content was split out into its own repository (with git history preserved) — the Silicon Design Notebook.

## License

[MIT](LICENSE) — use freely for study, interview prep, or teaching.

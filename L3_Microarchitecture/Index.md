# L3 — Microarchitecture

> The layer where L2 atoms (memory, FP units, dataflow, RTL) become **chips**: Streaming Multiprocessors, MXUs, DaVinci Cubes, NeuronCores, Tensix tiles. L3 is where vendor architecture diverges. NVIDIA, AMD, Google, AWS, Huawei, Meta, Cerebras, Groq, and Tenstorrent each chose different answers to the same physics — and those answers determine how every layer above (kernels, frameworks, serving) must look.

---

## Scope of this layer

L3 covers the **chip-level architecture**: how SMs/cores are organized, how the warp/wave/wavefront scheduler works, how the ISA exposes execution to compilers, and how the roofline analytical framework predicts achievable throughput on each design.

1. **The execution model and ISA** — SIMT (NVIDIA, AMD), VLIW (TPU, Trainium), spatial dataflow (Cerebras, Tenstorrent). PTX vs SASS. Warp / wave / wavefront semantics.
2. **The reference architecture** — the NVIDIA SM (Hopper / Blackwell). Warp scheduler, register file, tensor cores, SMEM, TMEM, TMA, async wgmma.
3. **The roofline framework** — derivation, ridge point, why decode is memory-bound and prefill is compute-bound. Worked examples on every chip.
4. **Per-vendor deep dives** — Blackwell (canonical), MI300/MI355 (AMD), TPU v5p/v6e/v7 (Google systolic), Ascend 910 (Huawei DaVinci 3D Cube), Trainium (AWS VLIW), MTIA (Meta), Cerebras WSE-3 (wafer-scale mesh), Groq LPU (SRAM-only deterministic), Tenstorrent (NoC-coupled tiles).
5. **Comparative summary** — the 2026 accelerator landscape on standardized axes.

L3 explicitly does **not** cover system / rack-scale interconnects (→ L4), kernel programming (→ L5), or model-architecture algorithms (→ L6).

---

## Pages (read in order)

1. [ISA_and_Execution_Model](ISA_and_Execution_Model.md) — what code looks like at the hardware boundary. Foundational; every later page assumes this.
2. [GPU_Architecture](GPU_Architecture.md) — the canonical SIMT chip (Hopper SM as reference). Reads like the textbook GPU.
3. [Memory_Hierarchy_and_Roofline](Memory_Hierarchy_and_Roofline.md) — the analytical framework that predicts throughput on every chip below. **Memorize the ridge-point math here.**
4. [Blackwell_Architecture](Blackwell_Architecture.md) — the 2025–2026 frontier NVIDIA part. Specialization of the GPU page with FP4/MX, TMEM, NV-HBI, NVL72.
5. [AMD_Instinct](AMD_Instinct.md) — CDNA 3 → CDNA 4 → CDNA-Next. Chiplet vs monolithic; xGMI vs UALink.
6. [Google_TPU](Google_TPU.md) — non-GPU paradigm. MXU, SparseCore, ICI 3D torus, OCS.
7. [Huawei_Ascend](Huawei_Ascend.md) — DaVinci 3D Cube; SMIC-node compensations; HCCS.
8. [Cloud_ASICs](Cloud_ASICs.md) — AWS Trainium (VLIW NeuronCore), Meta MTIA (mesh inference).
9. [Specialty_Accelerators](Specialty_Accelerators.md) — Cerebras WSE-3, Groq LPU, Tenstorrent.
10. [Accelerator_Landscape_2026](Accelerator_Landscape_2026.md) — comparative landscape with standardized axes (FLOPS, BW, TDP, scale-up domain, software stack).

---

## What L3 hands off to L4 and above

| L3 fact | Higher-layer consequence |
|---|---|
| H100 ridge point ≈ 295 ops/B (FP16); B200 ≈ 1 125 ops/B (FP4) | Decode is memory-bound on every modern accelerator (L8) |
| NVIDIA SM with 64 warps × 32 threads = 2 048 threads | Occupancy + register pressure are the L5 kernel knobs |
| Hopper TMA + wgmma async | FlashAttention-3 hits 70%+ utilization (L5/L6) |
| Blackwell adds TMEM | wgmma operand throughput unblocked → FP4 dense GEMM viable |
| TPU MXU 128×128 weight-stationary | TPU dominates dense-GEMM training; falls behind on small matmuls |
| Groq is SRAM-only, deterministic | Strange TTFT/throughput numbers; needs ~600 chips to host 70B (L8) |
| Cerebras WSE has 21 PB/s of on-die BW | Inverts the roofline — compute-bound on almost every kernel |
| AMD MI300X chiplet design adds NUMA latency | ROCm runtime must be topology-aware (L5/L7) |
| Trainium uses VLIW with software-scheduled pipelines | Neuron compiler does what nvcc + warp scheduler do at runtime |

---

## Reading order

Top-down for orientation: skim Index → `Accelerator_Landscape_2026` for context → return to start and read in order.

Strict bottom-up for depth: ISA → GPU → Roofline → Blackwell → others → Landscape.

After this layer, ascend to [L4 — Networking & Interconnects](../L4_Systems_and_Interconnects/Index.md) where chips become racks.


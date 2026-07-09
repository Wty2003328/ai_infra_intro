# L2 — Digital Design for AI

> The layer where physics becomes architecture. L0 (silicon) and L1 (packaging + HBM) deliver transistors and bandwidth; L3 (microarchitecture) consumes them as SMs and tensor cores. L2 is the RTL-level glue: SRAM bitcells, FP units, multipliers, register files, dataflow patterns, and the FSMs that orchestrate them.

---

## Scope of this layer

L2 covers the **synthesizable digital logic** of an accelerator — what an RTL engineer actually writes in SystemVerilog and what a synthesis tool maps to standard cells from L0:

1. **On-chip memory hardware** — 6T SRAM cell physics (SNM, β-ratios), banking and bank-conflict math, register-file multi-port physics, operand collector, TMEM, L2 distributed slices.
2. **FP unit design** — IEEE-754 vs OCP MX formats; integer multiplier scaling ($O(M^2)$); Wallace/Dadda partial-product reduction; FMA pipeline (multiply, align, add, normalize, round); LZA in parallel with CPA; the exact 2× FP4-vs-FP8 throughput law; MX sum-together hardware optimization; 2:4 sparsity gating.
3. **Systolic arrays & dataflow** — weight-/output-/input-stationary tradeoffs; 2D systolic mesh latency math; TPU MXU vs Tenstorrent Tensix vs Cerebras WSE vs NVIDIA warp+TMA hybrid; sparsity-aware dataflow.
4. **AI-specific RTL patterns** — pipelining and timing closure ($f_{\max}$ derivation, FO4 budget); CDC and mesochronous interfaces for cross-die signaling; async copy engines (TMA-like); NoC topologies; clock/power gating; verification reality.

L2 explicitly does **not** cover transistor or process physics (→ L0), HBM internals (→ L1), GPU SM scheduling or roofline (→ L3), or kernels (→ L5+).

---

## Pages

1. [On_Chip_Memory_Hardware](01_On_Chip_Memory_Hardware.md) — SRAM/RF/TMEM/L2. **Read first** because every later page assumes operand-fetch capacity.
2. [FP_Unit_Design](02_FP_Unit_Design.md) — multiplier, FMA tree, MX hardware. The compute atom.
3. [Systolic_Arrays_and_Dataflow](03_Systolic_Arrays_and_Dataflow.md) — how memory + FP units are arranged in space to amortize operand reads. (New page.)
4. [Digital_Design_For_AI](04_Digital_Design_For_AI.md) — the integration layer: pipelining, CDC, NoC, TMA-style async copies, verification, errata. **Read last** because it relies on the prior three.

---

## What L2 hands off to L3 and above

| L2 fact | Higher-layer consequence |
|---|---|
| 6T SRAM density stalled at TSMC N3+ | GPU L1/SMEM stuck at ~256 KB across 3 generations → drives L3 cache-hierarchy decisions |
| FP4 multiplier area = ¼ FP8, but FP32 accumulator is fixed-cost ⇒ FP4 throughput = 2× FP8 | Blackwell FP4 datasheet TFLOPS exactly |
| MX sum-together saves 31 of 32 alignment shifters per dot product | Why MXFP4 / NVFP4 hardware is not 32× area of FP4 |
| TMEM = 256 KB dedicated tensor-operand SRAM | Why Blackwell exposes a separate `wgmma`-only memory tier |
| 32-bank SMEM, 4-byte word interleave, mod-32 conflict math | Why CUDA padding tricks matter at L5 (kernels) |
| Operand collector hides 1R/1W bank conflicts | Warp scheduler at L3 can ignore RF bank contention |
| Mesochronous CDC across NV-HBI adds 2–4 cycles | Cross-die memory access penalty in dual-die GPUs (B200) |
| Weight-stationary dataflow ⇒ each weight reused N times for N output rows | TPU MXU energy-per-FLOP advantage; how systolic beats general-purpose at GEMM |
| 2:4 sparsity uses 4:2 mux + clock gating | 2× sparse throughput "for free" — only at coarse 4-element granularity |

---

## Reading order

1. **[On_Chip_Memory_Hardware](01_On_Chip_Memory_Hardware.md)** — establishes the operand-bandwidth ceiling everything else fights against.
2. **[FP_Unit_Design](02_FP_Unit_Design.md)** — the unit being fed.
3. **[Systolic_Arrays_and_Dataflow](03_Systolic_Arrays_and_Dataflow.md)** — how the fed-units are spatially arranged.
4. **[Digital_Design_For_AI](04_Digital_Design_For_AI.md)** — the integration glue (pipelining, CDC, NoC, async copies).

After this layer, ascend to [L3 — Microarchitecture](../L3_Microarchitecture/00_Index.md) where these L2 atoms become SMs, tensor cores, ISAs, and rooflines.


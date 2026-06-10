# L0 — Silicon & Process

> The bottom of the stack. Everything above this layer — HBM stacks, tensor cores, NVLink switches, CUDA kernels, vLLM schedulers — is a derived consequence of decisions made here. If you cannot reason about leakage current, reticle limits, and yield curves, you cannot honestly reason about why a B200 costs $40k or why Rubin needs backside power delivery.

---

## Scope of this layer

L0 covers the physics and economics of the silicon itself, *before* it is packaged into HBM stacks (L1) or used to build FP units and SRAM banks (L2). Concretely:

1. **Transistor architecture** — planar / FinFET / GAAFET (nanosheet) / CFET, with the electrostatics that distinguish them.
2. **Standard-cell libraries** — track height, drive current, and the area/frequency/power knob it gives the architect.
3. **Process nodes** — what TSMC N4/N3/N2/A16, Intel 18A, and SMIC N+1/N+2 actually mean in CPP / metal pitch / density / nominal V_dd.
4. **Lithography** — the EUV resolution / depth-of-focus equations and the **858 mm² reticle limit** that single-handedly forces multi-die packaging.
5. **Yield modeling** — Murphy / Poisson / negative-binomial models, and why *every* shipped accelerator is internally redundant.
6. **Power delivery** — IR droop, di/dt droop, and the move to back-side PDN at A16 / 18A.
7. **Thermal** — the 500 W/cm² wall that killed air cooling.

This layer does **not** cover HBM internals (→ L1), on-die SRAM hardware (→ L2), or microarchitectural design (→ L3). Cross-references are explicit.

---

## Pages

1. [Silicon_For_AI](Silicon_For_AI.md) — the canonical L0 deep-dive: physics → transistor → cell → node → lithography → yield → PDN → thermal.

(L0 is intentionally a single page. The breadth is here, but most of the depth flows naturally into L1 packaging and L2 digital-design pages.)

---

## What L0 hands off to L1

| L0 fact | L1 consequence |
|---|---|
| Reticle limit ≈ 858 mm² (33 mm × 26 mm at 0.33 NA EUV); halved to 429 mm² at 0.55 NA High-NA | Multi-die packaging is *mandatory* at frontier scale → CoWoS-S/R/L, EMIB, SoIC |
| Silicon interposer reticle-stitching ceiling ≈ 3.3× field (~2 800 mm²) | CoWoS-S cannot host 2× B200 dies + 8× HBM3e → forces CoWoS-L |
| Inter-die signaling energy: organic ~1 pJ/bit, silicon-bridge ~0.2 pJ/bit | NV-HBI 10 TB/s die-to-die only thermally feasible on silicon bridges |
| Defect density D₀ ≈ 0.1 / cm² at TSMC N4 maturity | HBM/logic die area drives package economics; spatial redundancy mandatory |

## What L0 hands off to L2

| L0 fact | L2 consequence |
|---|---|
| Subthreshold swing S → 60 mV/decade limit | SRAM idle leakage dominates large on-chip memory → low-Vt only on critical paths |
| HD (6T) cells = small + slow; HP (9T) cells = big + fast | Tensor-core FMA pipelines use HP cells; L2 / TMEM uses HD cells |
| GAAFET reduces I_off by ~10× at iso-V_th | N2 generation enables large TMEM without thermal blowup |

## What L0 hands off to L3 and above

| L0 fact | Higher-layer consequence |
|---|---|
| 500 W/cm² heat-flux limit | Direct-to-chip liquid cooling is mandatory → rack-scale (L4) thermal budget |
| di/dt droop → can't sustain 4 GHz on tensor cores | Boost clock policies, kernel pacing (L5), wgmma rate-limiting |
| ~$1 pJ/bit organic die-to-die vs ~10 pJ/bit cross-package | Why TP fits inside NVL72 NVSwitch domain (L4) but never spans cabinets |

---

## Reading order

Read [Silicon_For_AI](Silicon_For_AI.md) once end-to-end. It's the only page in this layer because every L0 concept feeds something concrete one or two layers up — there is no benefit to splitting into smaller pages.

After this layer, ascend to [L1 — Packaging & Memory Stack](../L1_Packaging_and_Memory/Index.md).

---

*Cross-domain references:*
- [`digital_design/Fabrication_Process.md`](../../hardware_design/Fundamentals/Fabrication_Process.md) — process flow at the chemistry/etch level (deeper than this layer goes).
- [`digital_design/CMOS_Fundamentals.md`](../../hardware_design/Fundamentals/CMOS_Fundamentals.md) — CMOS device physics primer.
- [`power/Power_Analysis_and_Signoff.md`](../../hardware_design/Power/Power_Analysis_and_Signoff.md) — IR-droop and PDN signoff math at the chip level.

# L1 — Packaging & Memory Stack

> The layer that turns logic dies and DRAM stacks into a *single piece of hardware*. L0 explains why we cannot build big monolithic dies; L1 explains how the industry works around it. Every "TB/s" number above this layer — HBM bandwidth, NV-HBI cross-die bandwidth, NVLink at the package edge — is determined by the geometry, signaling, and thermal limits of the package itself.

---

## Scope of this layer

L1 covers the **substrate and the memory** that sit immediately above L0 silicon and immediately below L2/L3 logic:

1. **Packaging substrates** — organic vs. silicon-interposer vs. embedded-bridge architectures (CoWoS-S/R/L, EMIB, IF-AP, SoIC).
2. **Inter-die signaling** — bump pitch and density math, ultra-short-reach signaling, eye-opening at 9.6+ Gbps, NV-HBI / EMIB / Foveros bandwidth derivations.
3. **HBM** — channel/pseudo-channel/bank topology, TSV physics, base-die logic, sense-amp analog reality, refresh overhead, thermal feedback loops.
4. **Hybrid bonding** — Cu-Cu direct bonding physics; the only path to <10 µm pitch.
5. **TSV electromigration, KOZ, CTE stress** — the long-tail reliability problems that limit stack height and lifetime.
6. **Power and thermal at the package level** — multi-die PDN, lid-vs-lidless, liquid-cold-plate interfaces.

L1 explicitly does **not** cover transistor physics or process nodes (→ L0), on-die SRAM hardware or FP units (→ L2), or chip microarchitecture (→ L3). HBM is here, on-die memory is L2.

---

## Pages

1. [Advanced_Packaging](Advanced_Packaging.md) — interposers, embedded bridges, hybrid bonding, NV-HBI / EMIB / IF-AP, package-level PDN and thermal.
2. [HBM_Deep_Dive](HBM_Deep_Dive.md) — HBM channel topology, sense-amp physics, TSV RC, base-die logic, HBM3e/HBM4 transition, refresh-vs-thermal feedback, processing-in-memory.

Read in order; `Advanced_Packaging` establishes the geometry and signaling environment that `HBM_Deep_Dive` then specializes for DRAM.

---

## What L1 hands off to L2 and above

| L1 fact | Higher-layer consequence |
|---|---|
| HBM3e: 1024-bit, 9.6 Gbps/pin → ~1.23 TB/s/stack; 8 stacks → ~9.8 TB/s/package | The "decode is bandwidth-bound" wall in [L3 Memory_Hierarchy_and_Roofline](../L3_Microarchitecture/Memory_Hierarchy_and_Roofline.md) |
| HBM4: 2048-bit @ ~10 Gbps → ~2 TB/s/stack; 8 stacks → ~16 TB/s/package | The Rubin/MI400 generational doubling |
| NV-HBI: ~10 TB/s die-to-die at ~0.2 pJ/bit | Why a 2-die B200 looks like a single GPU at the CUDA level |
| Microbump pitch ceiling (~25 µm), forced hybrid bonding for HBM4 | Per-stack capacity scaling slows; PIM and on-base-die compute become realistic |
| HBM stack thermal: 5–8 W/stack with $T_j \le 95\,°C$ | Refresh-rate escalation cap; mandatory cold-plate cooling at L4 rack design |
| CoWoS-L scales to 4–6× reticle (~3 500–5 800 mm²) | How many HBM stacks fit on a B200/MI355X/Rubin package |
| Hybrid-bonding energy: ~0.05 pJ/bit | The roadmap floor for inter-die / 3D-stacked compute |

---

## Reading order

Start with [Advanced_Packaging](Advanced_Packaging.md) for the package geometry, then [HBM_Deep_Dive](HBM_Deep_Dive.md) for the memory specialization. Cross-link forward to:

- [L3 — Microarchitecture](../L3_Microarchitecture/Index.md) for how this bandwidth is consumed by SMs / tensor cores.
- [L4 — Systems & Interconnects](../L4_Systems_and_Interconnects/Index.md) for what NVLink / NVSwitch does *across* packages, on top of NV-HBI *inside* the package.

Cross-domain references in this vault:

- [`digital_design/IC_Packaging.md`](../../hardware_design/07_Manufacturing_and_Bringup/IC_Packaging.md) — package-engineer view of the same ground.
- [`digital_design/Memory.md`](../../hardware_design/01_Architecture_and_PPA/Memory.md) — DRAM device physics from the design-engineer angle.
- [`digital_design/Signal_Integrity_Reliability.md`](../../hardware_design/05_Backend_Physical_Design/Signal_Integrity_Reliability.md) — eye-diagram, jitter, ISI math.

# Depth Audit — Is Everything at "Power-Gating Level"?

> Audit date: 2026-06-19. Scope: all 92 content pages across `ai_infra/` (L0–L8 + interview_prep) and `hardware_design/`. Index/README files excluded.
> Benchmark: `hardware_design/Power/Low_Power_Design.md` §6 "Power Gating — Deep Dive", which carries a topic through five levels — **(1) mechanism → (2) the actual cell/structure → (3) the cell's effect → (4) how-to / design procedure → (5) numbers & tradeoffs** (header/footer switch schematics, rush-current sizing math, daisy-chain power-on, balloon-latch retention FF transistor diagram, full power-down/up sequencing).

---

## Round 2 — deep body audit (read inside files, not just structure)

Read flagship-hard topics end-to-end to test for hand-waving in the hardest places: `GPU_Architecture` (SM scoreboard/GTO scheduling, operand collector, MMA dataflow, TMA microarch), `Attention_Mechanisms` incl. **MLA §5.6 decoupled-RoPE + matrix absorption**, and `STA` §7 **OCV/AOCV/POCV** (worked identical-path comparison). Plus a notebook-wide scan for self-admitted gloss phrases ("beyond scope", "for brevity", "hand-wave", "omitted"…).

**Finding: content depth is uniformly at/above the benchmark — no content gaps and no hand-waving found in this round.** The gloss-phrase scan returned essentially nothing (the few hits are cross-references or describe candidate pitfalls, not the page glossing).

**Concrete fix this round — notebook-wide subsection numbering.** A scan for `###` children whose major number disagrees with their parent `##` found 8 files with mismatches; all fixed:
- *Introduced by the Round-1 `##` renumber and now corrected:* `RISC_V_ISA` (§14–17 children re-keyed 11–14 → 14–17), `DFT_and_ATPG` (§15 children 11.x → 15.x).
- *Pre-existing, now corrected:* `GPU_Architecture` (§5 had 4.2/4.3), `CUDA_Optimization` (§8 had 7.1/7.2), `CUDA_Programming` (9.5 → 10.3), `Training_Optimization` (§10 had 9.1–9.3), `AHB_AXI_APB` (§16 had 14.5/14.6), `IC_Packaging` (§16 had 20.1). Final scan: zero parent/child mismatches remain.

**Net:** after two rounds, the one genuine *content* depth gap was NoC (fixed in Round 1). Everything else has been consistency/polish. If specific topics still feel thin to you, name them and I'll go straight there; otherwise the highest-value remaining work is correctness/currency verification, not adding depth.

---

## Progress log (updated 2026-06-19)

**Done:**
- ✅ **A1 — NoC router internals** (`Network_on_Chip.md`): added the router datapath block diagram, the G·R·O·P·C input-unit state table, flit/packet format, the credit-counter circuit (§3.3), the separable/matrix-arbiter allocator microarchitecture with iSLIP (§4.1B), plus a numbers row and a router-timing worked problem. Now at benchmark depth.
- ✅ **B — both pages that lacked closers**: added `Numbers to Memorize` + `Worked Problems` to `Observability_and_Debugging.md` and `Multimodal_Inference.md`.
- ✅ **C — numbering/dedup pass**: `CUDA_Optimization` (§8B), `Cache_Microarchitecture` (§11A), `UPF_Power_Intent` (removed repeated §2 header), `RISC_V_ISA` (§14–17), `Branch_Prediction` (un-numbered Navigation), `vLLM_Internals` (§15B), `DFT_and_ATPG` (§15). **Plus a real content fix:** `Quantization.md` had two near-identical "KV Cache Quantization" sections — merged into one §9 (kept all unique content: attention-error math, calibration, vLLM flag, mixed-precision, GQA/MLA orthogonality) and **corrected a wrong KV-size figure (32 GB → ~40 GiB)**.

- ✅ **B — consolidated `Numbers to memorize` tables** added to the 10 flagged pages: Power folder (`Power_Fundamentals`, `Power_Reduction_Techniques`, `Low_Power_Design`, `UPF_Power_Intent` — the last as a commands/cells reference) and L8/L7 (`Speculative_Decoding`, `Prefill_Decode_Disaggregation`, `Modern_Post_Training`, `Inference_Frameworks`, `Batching_and_Scheduling`, `Production_Architecture`). Each table drawn from the page's own figures; trailing References/Further-Reading sections renumbered accordingly.

**Remaining (lower value):**
- A few very large hardware pages (`CPU_Architecture`, `Memory`, `AHB_AXI_APB`, several SystemVerilog pages) still close on "Interview Q&A" rather than a `Numbers to memorize` table. They're already comprehensive — adding one is marginal. Say the word and I'll do them too.
- Optional Tier-A polish: ESD-clamp cross-link in `Signal_Integrity`; a drawn droop-detector schematic in `Block_Activity`.
- Tier D: web-verified currency refresh of `Frontier_Models_2025_2026` and `Accelerator_Landscape_2026`.

---

## Headline

The notebook is **already at or very near the benchmark depth across the large majority of pages.** Low word counts are usually topic-appropriate, not shallow: the thinnest pages I read in full (`Network_on_Chip`, `GPU_Multitenancy`, `Cloud_ASICs`) all carry block diagrams, comparison tables, worked problems, and numbers. Cell-level structure where it matters is genuinely present — `Memory.md` has a transistor-level 6T/8T/10T SRAM treatment with sense-amps and write-drivers, `Basic_Knowledge.md` has latch/FF at transistor level, `Clock_Division.md` has the PLL with charge-pump/VCO/loop-filter, `Adders.md` derives every adder cell, the power cells (ICG, isolation, level-shifter, retention, header/footer switch) are covered in `Low_Power_Design` + `UPF_Power_Intent`.

So this is **not** a "fill in the stubs" job. The remaining work is a small, specific set of (A) genuine structural-depth gaps, (B) template-consistency gaps, and (C) redundancy/numbering cleanup. They are listed below in priority order.

---

## Tier A — Genuine structural-depth gap (below benchmark)

After verifying candidate gaps against the actual text, **only one page has a real structural-depth gap.** (Two others I first suspected — Signal Integrity and Block-Activity — turned out to already cover their material well; see "Verified non-gaps" below. I'm leaving that note in deliberately so the audit is honest about what was checked.)

### A1. `hardware_design/Architecture/Network_on_Chip.md` — router datapath internals *(the one real gap)*
Strong on topology math, flow-control disciplines, deadlock theorems (Dally–Seitz, Duato), CHI-on-mesh, and worked problems. But the **router itself is described only in prose** — the benchmark would draw it. The page mentions a "two-stage separable allocator" in one sentence and never shows it. Missing:
- **Router datapath block diagram**: per-VC input FIFOs → route-compute → VC allocator → switch allocator → crossbar → output latches.
- **Allocator microarchitecture**: the separable allocator (input-first arbiters → output-first arbiters), the request matrix, round-robin/matrix-arbiter pointers and their fairness effect. This is exactly the "what is the structure and what does it do" level that power gating hits and NoC currently skips.
- **Flit/packet format**: head/body/tail flit fields (dest, VC id, type, route bits, sideband).
- **Credit counter logic**: the up/down counter + credit-return wire at gate level (the page derives the credit-loop *math* but not the *circuit*).
- A short allocator/credit **RTL sketch**, paralleling the power-gating sequencing block.

**Verified non-gaps** (checked, already at depth — do *not* rewrite):
- `Signal_Integrity_Reliability.md` already has aging at benchmark depth (§8.2: NBTI/PBTI/HCI table, ΔVth ∝ tⁿ·exp(−Ea/kT) with 10-year numbers, aging-aware STA + PrimeTime flow) and antenna effect (mechanism, antenna ratio, Fowler-Nordheim, fixes). *Only nit:* antenna/ESD live in the Q&A rather than as standalone sections, an
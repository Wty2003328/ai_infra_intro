# L2 — Digital Design for AI: Interview Questions

Consolidated interview Q&A and worked problems from every page in `L2_Digital_Design_for_AI/`, moved here to keep topic pages focused on concepts. Each section links back to its source page for full context.

---

## Digital Design for AI — Pipelining, CDC, NoC, Async Copies

*From [Digital_Design_For_AI.md](../L2_Digital_Design_for_AI/Digital_Design_For_AI.md)*

**Q1.** *Why can't NVIDIA push tensor cores to 4 GHz on Blackwell?*

Three compounding limits: (a) The 5-stage FMA pipeline already has only ~80 ps per stage; halving to 2 GHz → 4 GHz target would need ~10 stages, doubling pipeline-FF area (which already approaches the multiplier area). (b) Voltage scaling: $V_{dd}$ would need to rise ~100 mV to maintain noise margin at the higher frequency, blowing up dynamic power as $V^2$. (c) di/dt droop becomes uncontrollable — the L0 deep-trench caps can't supply 2× the burst current without physically more cap area. Net: 4 GHz is achievable only at TDP that exceeds the 1 000 W package budget by ~2×.

**Q2.** *Estimate the cost in cycles of accessing memory on the "other die" of a B200 dual-die GPU.*

NV-HBI mesochronous CDC: 2–4 cycles. Local NoC hops on remote die: ~6 cycles. L2 access on remote slice: ~50 cycles. HBM access if miss: 300+ cycles. If the request is satisfied at the remote die's L2: ~60 cycles total vs ~50 on local L2 — ~20% penalty. If it goes to HBM: ~310 vs ~300 — <4% penalty. This is why two-die GPUs work despite the geometry: at HBM granularity the cross-die hop is invisible.

**Q3.** *Why does the TMA + wgmma combination give a step-function in achievable utilization?*

Pre-TMA: warp issues 32 individual loads per tile → ~50 cycles of issue overhead per tile × scheduler resource starved. Effective utilization on memory-bound kernels: ~30%.
Post-TMA: warp issues 1 TMA descriptor per tile → 2 cycles of overhead, async completion. Scheduler freed to issue compute. Effective utilization: 70%+.
The step function is because TMA *removes* a quadratic scaling problem (issue × tile count) and replaces it with constant overhead.

**Q4.** *What happens if a designer omits the latch in the ICG cell?*

The enable signal can glitch (briefly toggle) due to combinational hazards in the controller. Without the latch, glitches pass through the AND gate → spurious rising clock edges → flip-flops in the gated cluster sample mid-cycle → state corruption that's nearly impossible to debug because it depends on signal-arrival timing. The latch holds EN stable across a clock period, eliminating the glitch path. This is one of the canonical "if you remember nothing else about clock gating, remember the latch" rules.

**Q5.** *Why is dimension-order routing provably deadlock-free?*

Cycles in routing dependency graph require packets that route in non-monotonic dimension orderings (e.g., one packet routes X-then-Y while another routes Y-then-X, creating a hold-and-wait cycle). DOR forces every packet to traverse X first, then Y — eliminates the cycle by construction. Tradeoff: bandwidth utilization drops because some long X+Y trips can't shortcut through alternative routes. VCs let you have BOTH adaptive routing AND deadlock freedom by giving certain traffic classes their own dedicated channels that can't be blocked by congestion in the other classes.

---

## FP Unit Design — Multipliers, Wallace Trees, FMA, MX Formats

*From [FP_Unit_Design.md](../L2_Digital_Design_for_AI/FP_Unit_Design.md)*

**Q1.** *Derive why the Wallace tree depth is logarithmic in $N$ but not in base 2.*

A 3:2 compressor takes 3 input rows and produces 2 output rows. Per level, row count is multiplied by 2/3. After $L$ levels, $N \cdot (2/3)^L = 2$ (so a final CPA can sum the last 2 rows). $L = \log_{3/2}(N/2)$. Base 1.5, not 2, because each level's compression is a factor of 1.5×, not 2×.

**Q2.** *MXFP4 has 4.25 bits/element amortized but uses 4-bit elements. Explain the discrepancy.*

The shared exponent is 8 bits per 32 elements = 0.25 bits/element. Total: 4 + 0.25 = 4.25 bits/element. The "4-bit element" is just the mantissa+sign storage; the exponent overhead is amortized.

**Q3.** *A Blackwell SM provides 9000 TFLOPS FP4 = 4500 TFLOPS FP8. The mantissa multiplier shrinks 4×. Why isn't the throughput 4× as well?*

Because (a) the FP32 accumulator is shared between FP4 and FP8 and represents a fixed area cost (~half the MAC), so MAC area shrinks only ~1.5×, and (b) operand-fetch bandwidth is the second factor: FP4 packs 2 per byte, halving operand traffic and letting the same SMEM port feed 2× more MACs. Multiply: 1.5 × (2/1.5) = 2.0 exactly.

**Q4.** *Why is the LZA used instead of computing leading zeros after the CPA?*

If you waited for the CPA, the leading-zero count and the subsequent shift would add ~6–8 FO4 to the critical path, blowing the 50 ps budget per FMA stage and forcing $f_{\max}$ down ~25%. The LZA is a parallel boolean network that predicts the LZ count in ~3.5 FO4, completing before or at the same time as the CPA. The shifter starts immediately.

**Q5.** *Why does FP6 underperform what mantissa-area math predicts?*

FP6 has M=3, so a 3×3 = 9 partial-product multiplier — much smaller than FP8's 16. By the area-only argument, FP6 should give ~1.4× FP8 throughput. In practice it gives ~1.1×. Two reasons: (a) the 6-bit datapath doesn't divide evenly into power-of-two byte-addressable memory (8 ÷ 6 = 1.33 elements/byte → packing wastes 25% of bits or requires complex unpacking logic), and (b) 6-bit operands need an additional MUX layer to share the multiplier with the 4 / 8 / 16-bit pathways, eating most of the area win. Architects therefore prefer pure powers of two; FP6 only makes sense as a research format.

---

## On-Chip Memory Hardware — SRAM, Register File, TMEM, L2

*From [On_Chip_Memory_Hardware.md](../L2_Digital_Design_for_AI/On_Chip_Memory_Hardware.md)*

**Q1.** *A B200 SM at FP8 needs ~50 TB/s of operand bandwidth. SMEM provides only ~6.5 TB/s. Why doesn't the SM stall?*

Two reasons. (a) Not every cycle is a wgmma — about 60–70% of cycles execute tensor ops; the rest are scheduling, address calc, etc. Effective demand is lower than the raw peak. (b) Register-file reuse: each operand fetched into RF is reused across multiple wgmma instructions in the same tile. The RF (with wider per-bank ports and the operand collector) supplies the high-bandwidth path (~30 TB/s), while SMEM only stages data into RF/TMEM at tile boundaries — its ~6.5 TB/s is sufficient for inter-tile prefetch. With FP4 (B300), even this two-tier scheme falls short and TMEM becomes mandatory.

**Q2.** *You're allocating a 64×64 tile in SMEM as `__shared__ float A[64][64]` and 32 threads in a warp transpose-load column-major (each thread reads `A[k][threadIdx.x]` for k∈[0,63]). What's the bank conflict?*

Stride per thread is 64 * 4 = 256 B = stride 64 in 4-B words. Bank = 64·i mod 32 = 0 for all i. **32-way conflict** on every load. Fix: pad to `A[64][65]` — stride becomes 260 B = 65 in words, bank = 65i mod 32 = (i + 32i) mod 32 = i mod 32 → 0-way conflict. Cost: ~1.6% extra SMEM. Speedup: ~32×.

**Q3.** *Why isn't TMEM just a renamed bigger SMEM?*

Three reasons. (a) Read-port topology: TMEM has wide (1024-bit) read ports geometrically matched to wgmma tile rows; SMEM has 32 narrow (32-bit) banks for arbitrary thread-pattern access. (b) Isolation: SMEM can be touched by any CUDA thread instruction, creating probabilistic conflict with wgmma. TMEM is addressable only by tensor-core ops — zero contention. (c) Address mapping: SMEM is byte-addressed and bank-interleaved; TMEM uses tensor-tile addressing matched to mma matrix layout. Effectively, TMEM is an *operand cache* for the tensor core, not a general scratchpad.

**Q4.** *Estimate the bitcell area cost of a true 6R/2W register file vs 1R/1W banked + operand collector, both for 256 KB.*

Multi-port: $A_{cell} \propto (P+W)^2 = 64$. So 64× the area-per-bit of a single-port cell. 256 KB at 64× ≈ 16 MB-equivalent of 1R/1W bitcell area — about 200 mm² at frontier nodes, larger than a whole SM. Banked + collector: 256 KB at 1× = ~3 mm². The operand collector itself adds maybe 0.1 mm² of FFs and arbitration logic. Total: ~50× area savings, comparable bandwidth. This is why the GPU industry universally chose the second path.

**Q5.** *Why does an LLM with a 70 GB working set still benefit from L2 at all?*

The full working set doesn't fit, but *transient* working sets do: FlashAttention's row sums and exp-max statistics live entirely in L2 across tile passes (~MB scale); KV-cache pages just fetched from HBM are reused once (in fmha epilogue) before being evicted; activations from forward pass are re-read by the next layer's backward pass within microseconds. L2 catches these short-lived high-reuse buffers. Without L2, every reduction would round-trip through HBM and decode would slow ~3×.

---

## Systolic Arrays & Dataflow — Spatial MAC Organization

*From [Systolic_Arrays_and_Dataflow.md](../L2_Digital_Design_for_AI/Systolic_Arrays_and_Dataflow.md)*

**Q1.** *A 128×128 systolic array does $C = A \cdot B$ where $A \in \mathbb{R}^{128 \times 1024}$, $B \in \mathbb{R}^{1024 \times 128}$. How many cycles total? Effective utilization?*

Pre-load weights: 128 cycles. Stream activations: K = 1024 cycles. Drain output: 127 cycles. Total: ≈ 1 280 cycles. Useful work: $128 \cdot 128 \cdot 1024 = 16.78$ M FMAs across $128^2 = 16\,384$ MACs. Each MAC averages ~$16.78\text{ M}/(16\,384 \cdot 1\,280) \approx 80\%$ utilization. The 20% loss is the fill+drain — amortizable by tiling.

**Q2.** *Why does a TPU MXU outperform a GPU SM on dense GEMM despite having fewer total MACs?*

Three reasons: (a) Weight-stationary dataflow reuses each weight 128 times before eviction → ~128× lower weight bandwidth. (b) Operands flow PE-to-PE through register-style pipelined links — no SMEM/RF round-trips. (c) The systolic array is uniformly utilized (no operand-collector arbitration losses). The GPU pays for general-purpose flexibility with operand-fetch bandwidth.

**Q3.** *Why is output-stationary preferred for FlashAttention's per-tile inner loop?*

Within an FA tile, $K = $ tile-dim × head-dim, which is small (~64–128). Weight-stationary's fill/drain cost is large relative to K — wastes cycles. Output-stationary keeps the partial sums of $S = QK^T$ in registers; both Q and K stream through. K dominates so this minimizes shifts and updates — the OS choice gives ~2× higher utilization than WS at this granularity.

**Q4.** *Cerebras has no HBM. How does it serve a 70 B-parameter LLM at 8-bit?*

70 GB at FP8 doesn't fit on the wafer's 40 GB on-die SRAM. Cerebras streams weights from external MemoryX nodes via SwarmX (~1.2 TB/s aggregate). The wafer holds activations + KV cache + a sliding window of weights. For inference, this works because each layer's weights are touched once per token — streaming throughput suffices. For training, MemoryX bandwidth limits batch size; Cerebras typically uses batch parallelism across multiple wafers.

**Q5.** *Estimate the operand-bandwidth saving of weight-stationary 128×128 systolic vs equivalent throughput general-purpose MAC array.*

Systolic at 1 GHz × 16 384 FMAs/cycle = 16 K FMAs/ns. Operand bandwidth at array boundary: 128 in + 128 out = 256 B/cycle × 1 GHz = 256 GB/s. General-purpose: 16 K FMAs × 2 inputs × 2 B = 64 KB/cycle = 64 TB/s. Ratio: 256× lower for systolic. This is the structural energy advantage of TPUs over GPUs on pure GEMM workloads.


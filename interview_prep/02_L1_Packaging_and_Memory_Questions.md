# L1 — Packaging and Memory: Interview Questions

Consolidated interview Q&A and worked problems from every page in `L1_Packaging_and_Memory/`, moved here to keep topic pages focused on concepts. Each section links back to its source page for full context.

---

## Advanced Packaging — Interposers, Bridges, NV-HBI, Hybrid Bonding

*From [Advanced_Packaging.md](../L1_Packaging_and_Memory/01_Advanced_Packaging.md)*

**Q1.** *A new accelerator wants 4 compute dies + 12 HBM4 stacks on one package. HBM4 footprint with margin is ~140 mm²; compute die is ~700 mm² each. Will it fit on CoWoS-L 6× (3 500 mm²)? What about CoWoS-L 9× (5 800 mm²)?*

Active-die area: $4 \cdot 700 + 12 \cdot 140 = 2 800 + 1 680 = 4 480\,\text{mm}^2$. Add ~15% for routing/power-fanout perimeter: $\sim 5 150\,\text{mm}^2$. Doesn't fit on CoWoS-L 6×; fits on CoWoS-L 9× with little margin. The "9× by 2027" roadmap is exactly to enable this geometry.

**Q2.** *Why is hybrid bonding mandatory for HBM4 but optional for HBM3e?*

HBM3e is 1 024 bits @ 40 µm bump pitch — fits within the SnAg microbump regime. HBM4 doubles the bus to 2 048 bits within roughly the same base-die footprint, halving per-pin pitch into the 20–25 µm region where SnAg bridging and IMC-fatigue defects spike. Hybrid bonding's <10 µm pitch handles this, plus drops energy/bit ~10× — critical because doubling pin count doubles the PHY power budget unless energy/bit is cut.

**Q3.** *A package draws 600 A in 0.5 ns from idle. Substrate inductance is 25 pH; on-die deep-trench cap is 1.2 µF. Is the di/dt droop survivable?*

$di/dt = 1.2 \times 10^{12}$ A/s. $\Delta V = 25\times 10^{-12} \cdot 1.2\times 10^{12} = 30$ mV. The DTC absorbs the sub-ns transient (its impedance at 1 GHz is $1/(2\pi f C) \approx 0.13$ mΩ — vanishing). So the 30 mV droop appears at the package level; with ~50 mV IR droop on top, total is ~80 mV. On a 0.7 V rail with V_dd_min ~0.6 V, margin ~20 mV. Tight but survivable; engineers would likely add LSCs to absorb the longer-tail of the transient, leaving more headroom.

**Q4.** *Estimate B200 package-level signaling power assuming 8 HBM3e stacks at 1.23 TB/s each + 10 TB/s NV-HBI, both at 0.2 pJ/bit.*

Total signaling BW: $8 \cdot 1.23 + 10 = 19.84$ TB/s = $1.587 \times 10^{14}$ bits/s. At 0.2 pJ/bit:

$$
P \;=\; 1.587\times 10^{14} \cdot 0.2\times 10^{-12} \;=\; 31.7\ \text{W}
$$

About 3% of the 1 000 W package — non-trivial, and the reason silicon-bridge (not organic) routing is a forced choice. On organic at 1 pJ/bit it'd be 158 W, ~16% of TDP.

**Q5.** *Why does AMD's IF-AP architecture have a NUMA penalty NVIDIA's NV-HBI doesn't? Estimate the latency difference.*

NV-HBI: ~10–20 ns die-to-die (cache-coherent, single GPU view). IF-AP across the IOD: every cross-XCD memory access traverses XCD→IOD fabric (~10 ns) → IOD routing (~10 ns) → IOD→destination XCD (~10 ns) → HBM access (~80 ns). Local: ~80 ns. Remote: ~110 ns. The runtime must place tensor blocks on the local XCD or pay a 30%+ access-latency tax — exactly what NCCL+ROCm topology-aware scheduling tries to do.

---

## HBM Deep Dive — Channels, TSVs, Sense Amps, Base Die

*From [HBM_Deep_Dive.md](../L1_Packaging_and_Memory/02_HBM_Deep_Dive.md)*

**Q1.** *A B200 8-HBM3e package serves 64 concurrent requests of a 70 B FP8 model. How many tokens/sec aggregate? Identify the bottleneck.*

Effective BW ≈ 0.85 × 9.83 = 8.36 TB/s. Per-step weight read = 70 GB. Tokens/step = 64 (one per request). So tokens/sec = $\frac{8.36\,\text{TB/s} \cdot 64}{70\,\text{GB}} \approx 7 640$ tok/s aggregate. Bottleneck: HBM bandwidth (decode-bound). Compute is barely touched at this batch size; FP8 tensor-core capacity is ~10× idle.

**Q2.** *Why can't HBM raise per-pin to 20 Gbps to double bandwidth without doubling pin count?*

Three compounding reasons: (a) eye opening collapses — single-ended USR signaling at 20 Gbps requires equalization (DFE/CTLE) that quintuples PHY power; (b) the TSV-stack acts as a lossy bus with skin-effect attenuation rising as √f, so 4× higher loss at 4× the Nyquist; (c) the protocol overhead (training, refresh) becomes a larger fraction of throughput. Total energy per bit at 20 Gbps would rise 2.5×, defeating the purpose. Doubling pin count via hybrid bonding stays linear in pJ/bit.

**Q3.** *Estimate the HBM stack power that pushes a stack from refresh zone 1 to zone 2.*

In the steady state, $T_j = T_{\text{ambient}} + P \cdot \theta_{\text{stack}}$. With $T_{\text{ambient}} = 35\,°$C (cold-plate inlet), zone 2 at 85 °C, and $\theta_{\text{stack}} \approx 8\,°$C/W (typical 12-Hi):

$$
P_{\text{zone2}} \;=\; \frac{85 - 35}{8} \;=\; 6.25\ \text{W}
$$

So a 12-Hi HBM3e stack hits zone-2 at roughly steady ~6 W. A peak workload putting >6 W per stack will halve refresh interval and cost ~9% of bandwidth. Realistic ops budget: hold each stack to ≤5 W average.

**Q4.** *Why does HBM4 require a logic-node base die?*

Three reasons stacking: (a) PHY power per bit must drop ~30% to keep total package signaling power flat as bus width doubles; only logic-node transistors get there; (b) on-die ECC scoring at 16 TB/s requires nontrivial error-correction circuits, which need logic-density transistors; (c) the optionality of putting GPU memory-controller fragments and PIM in the base die is unlocked, reducing inter-package data movement.

**Q5.** *A kernel does fully random 64 B reads from a 70 B model. Estimate effective HBM bandwidth on B200.*

Random access destroys row-buffer locality. Each access pays ~30 ns. 64 B per access ⇒ effective per-channel BW = 64 / (30 ns) = 2.13 GB/s. With 8 stacks × 16 channels × 2 PCs = 256 independent channels: 256 × 2.13 = 545 GB/s. Versus 9.8 TB/s peak: **5.5% of peak**. This is why prefix-cached / structured attention patterns dominate decode-time access in production engines.


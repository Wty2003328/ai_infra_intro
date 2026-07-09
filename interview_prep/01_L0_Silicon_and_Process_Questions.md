# L0 — Silicon and Process: Interview Questions

Consolidated interview Q&A and worked problems from every page in `L0_Silicon_and_Process/`, moved here to keep topic pages focused on concepts. Each section links back to its source page for full context.

---

## Silicon for AI — Physics, Process, Yield, Power

*From [Silicon_For_AI.md](../L0_Silicon_and_Process/01_Silicon_For_AI.md)*

These are not "name the term" questions — each requires deriving from L0 invariants.

**Q1.** *A foundry quotes you a defect density of 0.12/cm² with α = 1.8. What die area gives 50% raw yield (no redundancy)?*

Solve $0.5 = (1 + AD_0/\alpha)^{-\alpha}$ for $A$:
$1 + 0.12A/1.8 = 0.5^{-1/1.8} = 1.470$
$0.12A/1.8 = 0.470 \Rightarrow A = 7.05\,\text{cm}^2 = 705\,\text{mm}^2$.
Just under the reticle limit. Adding 5% spare SMs would push effective yield ~95%+.

**Q2.** *A 1 000 W package switches 600 A in 0.8 ns. PDN inductance is 80 pH. What's the di/dt droop, and is it survivable on a 0.7 V rail with 0.6 V V_dd_min?*

$di/dt = 600 / 0.8\times 10^{-9} = 7.5 \times 10^{11}$ A/s.
$\Delta V = 80\times 10^{-12} \cdot 7.5\times 10^{11} = 60\,\text{mV}$.
With ~50 mV IR droop on top, $V_{\text{trans}} = 700 - 50 - 60 = 590\,\text{mV} < 600\,\text{mV}$.
**Not survivable** without DTC — would corrupt or crash. Fix: add ~700 nF of DTC, or move to BSPDN to halve IR droop.

**Q3.** *Why does Rubin's roadmap re-introduce silicon-bridge cost reductions even though TSMC perfected CoWoS-L?*

High-NA EUV halves the reticle field to 429 mm². Rubin's logic budget per package roughly doubles vs B200, so per-package die count goes from 2 → 4–6. The number of silicon bridges per package scales super-linearly; bridge yield, alignment, and inter-die latency all become bigger problems. Cost-down silicon bridges (thinner, hybrid-bonded, smaller area) are how the package economics stay sane.

**Q4.** *Walk through why N2 (GAAFET) matters more for idle SRAM power than for tensor-core peak power.*

Tensor cores are dynamic-dominated: $P = \alpha C V^2 f$. Going from 0.70 V to 0.65 V buys ~14% — useful but not transformative.
SRAM idle is leakage-dominated: $P_{\text{leak}} \propto I_{\text{off}} \propto 10^{-V_{th}/S}$. GAAFET cuts $S$ from ~70 to ~62 mV/decade, which at constant $V_{th}$ reduces $I_{\text{off}}$ by $10^{V_{th} (1/62 - 1/70)} \approx 10^{0.020}\approx 1.05$ at $V_{th}=200$ mV — but the *real* GAAFET advantage is enabling lower $V_{th}$ at constant leakage, recovering frequency at lower V_dd. Stacking both effects, total SRAM idle leakage drops ~3×.

**Q5.** *Estimate the maximum HBM-stack count on a CoWoS-S package given 800 mm² logic die.*

CoWoS-S interposer ceiling ~2 800 mm². Each HBM3e 12-Hi stack footprint ~110 mm² + ~30 mm² perimeter. Logic die + 8 HBM = $800 + 8 \cdot 140 = 1\,920$ mm². Fits. 12 HBM = $800 + 1\,680 = 2\,480$ mm². Tight. 16 HBM doesn't fit, period — that's the structural reason 8-stack and 12-stack HBM packaging is the practical ceiling on CoWoS-S, and why CoWoS-L was needed to break it.


# L3 — Microarchitecture: Interview Questions

Consolidated interview Q&A and worked problems from every page in `L3_Microarchitecture/`, moved here to keep topic pages focused on concepts. Each section links back to its source page for full context.

---

## AMD Instinct — CDNA 3 / CDNA 4 / Helios

*From [AMD_Instinct.md](../L3_Microarchitecture/AMD_Instinct.md)*

**Q1.** *Why does AMD's chiplet approach work despite the NUMA penalty?*

Yield economics. An 800 mm² monolithic die at TSMC N5 with $D_0 = 0.1 / cm²$ yields ~50% (negative-binomial). 8 × 100 mm² chiplets at the same defect density yield ~95% per chiplet → ~67% combined good packages (any chiplet bad → reject). But wafer cost amortization across many small dies plus binning gives ~30% lower per-package cost than monolithic. Combined with 3D-SoIC bonding eliminating most of the inter-chiplet latency, NUMA cost is small enough to be worth the silicon savings.

**Q2.** *MI355X has higher peak FP4 (20 PFLOPS) than B200 (9 PFLOPS). Why isn't it dominating in production?*

Three reasons: (a) **Software** — ROCm has caught up but lags CUDA in obscure-kernel optimization; many production fp4 kernels aren't yet co-tuned for AMD. (b) **Scale-up** — xGMI tops out at 8 GPUs vs NVL72's 72; MoE / large TP workloads need the wider domain. (c) **Ecosystem** — vLLM, SGLang, TensorRT-LLM all ship CUDA-first; AMD ports come with delay. Helios/UALink+ROCm 7 closes the scale-up and software gaps; expect AMD's market share to rise through 2026–2027.

**Q3.** *Estimate cross-chiplet bandwidth available on MI300X for an 8-way tensor parallel within one package.*

Each XCD has Infinity Fabric to the IOD layer. Aggregate intra-package BW (XCD↔IOD): ~10 TB/s (matches Infinity Cache BW). Spread across 8 XCDs in TP: ~1.25 TB/s/XCD bidirectional. For TP all-reduce of activations (~8 GB activation matrix), reduce time ~6.4 ms — comparable to NVL72 NVLink for the same operation but without NVSwitch hop overhead.

**Q4.** *Why does AMD prefer 64-thread wavefronts when NVIDIA chose 32?*

Historical: AMD's GCN ISA used 64-lane SIMD; CDNA inherited it. Pros of wave64: amortizes scheduler overhead (one issue per cycle drives twice the threads), better for graphics-style large pixel groups. Cons: divergence cost is twice as bad (a 50/50 if-else stalls 32 lanes instead of 16); harder to occupy fully on small kernels. CDNA-3 added wave32 support to address this; modern AMD code paths choose width per kernel.

**Q5.** *Compare ROCm RCCL with NCCL for AllReduce performance.*

Both implement ring + tree algorithms. RCCL's bandwidth in xGMI domain (8 GPUs, MI300X) reaches ~85% of peak (~380 GB/s out of 448). NCCL on NVL8 (H100) reaches ~95% (~850 GB/s out of 900). NVIDIA's edge: lower-overhead algorithm-selector that switches ring/tree/recursive halving by message size dynamically. RCCL ships with similar logic but less battle-tested. At rack scale (NVL72 vs Helios), the gap narrows because both fabrics are limited by the same physics.

---

## Blackwell Architecture — B100 / B200 / B300 / GB200 / GB300, Rubin Outlook

*From [Blackwell_Architecture.md](../L3_Microarchitecture/Blackwell_Architecture.md)*

**Q1.** *Why does Blackwell need TMEM but Hopper doesn't?*

Hopper's tensor cores operate on FP8 (1 byte/operand). FP8 wgmma operand demand ≈ 25 TB/s/SM, fitting within SMEM's ~30 TB/s budget. Blackwell adds FP4 (½ byte/operand) which doubles the *effective* operand throughput requirement to ~50 TB/s/SM — exceeding SMEM. Doubling SMEM doesn't help (port count, not capacity, is the bottleneck). TMEM is a separate SRAM with wide ports geometrically matched to wgmma tile shapes.

**Q2.** *Estimate the cross-die latency penalty for accessing remote HBM on B200.*

Local HBM access: ~250 ns (DRAM activation + return). Cross-die: + NV-HBI traversal (~10–20 ns) + remote-die NoC (~6 cycles ≈ 4 ns at 1.6 GHz). Total ~270 ns, or ~8% slower. Well within HBM's normal latency variance — usually invisible. For L2-resident data the penalty is bigger (30 ns local vs 80 ns cross-die ≈ 2.7×) but L2 hits are rare in LLM workloads.

**Q3.** *NVL72 is non-blocking with 130 TB/s switch capacity matching GPU aggregate. What workload does this enable that NVL8 (Hopper) couldn't?*

MoE expert-parallel all-to-all. For a 16-expert MoE with 16 GPUs, each GPU sends ~1/16 of its activation to every other GPU. Bandwidth required: $N \cdot d \cdot \text{bytes} \cdot (E-1)/E$ where N is batch tokens, d is hidden, E is expert count. At rack scale (E = 32, N = 4 K, d = 8 K), this is ~10 TB/s of all-to-all per second. NVL72 sustains this; NVL8 saturates the inter-rack network (~400 Gb/s IB) and prefill stalls. NVL72 brings expert-parallel MoE training "into the rack".

**Q4.** *How does NVLink-C2C change long-context decode economics?*

KV cache for 1M tokens at 70 B FP16 model with GQA = ~140 GB → exceeds B200's 192 GB HBM (after weights). Without C2C, swapping to host PCIe (64 GB/s) means each KV read costs 100× HBM time → decode collapses. With C2C (900 GB/s), KV bytes from Grace LPDDR cost ~10× HBM time but stays workable. Effective context length per GB200 superchip rises from ~512 K to ~4 M tokens.

**Q5.** *Why does Blackwell's FP4 throughput reach only ~70% in real benchmarks despite the 2× theoretical advantage?*

Three factors: (a) FP4 wgmma's K dimension doubles to 64 — more partial-product accumulation in a single instruction lengthens the tensor-core pipeline, increasing per-issue latency; (b) MX shared-exponent decode adds ~1 extra cycle of overhead per block; (c) operand-fetch from TMEM is at the bandwidth ceiling already, so any cache miss into SMEM costs a full HBM round-trip. In aggregate, sustained FP4 throughput is ~2/3 of theoretical 2× peak — still a huge speedup, just not the marketing number.

---

## Cloud-Vendor ASICs — AWS Trainium / Inferentia, Meta MTIA

*From [Cloud_ASICs.md](../L3_Microarchitecture/Cloud_ASICs.md)*

**Q1.** *Why does AWS use SRD instead of RoCE for EFA?*

RoCE assumes a near-lossless Ethernet (PFC, ECN). At AWS-scale (hundreds of thousands of nodes, commodity hardware), packet loss is statistically inevitable. RoCE responds badly to loss (TCP-style retransmit timers in the millisecond range). SRD (a) sprays packets across multiple ECMP paths simultaneously, randomizing congestion; (b) detects loss via packet sequence numbers and retransmits in microseconds; (c) doesn't require lossless network configuration. For 100 K-chip training, this is the difference between "training runs" and "training stalls every 10 minutes".

**Q2.** *Estimate Trainium 3's per-package training throughput on a 70 B-parameter model.*

For tensor-parallel TP=8 across the 16-chip UltraServer, each chip holds ~9 GB of weights at FP8. Forward pass: $2 \cdot 70\text{ B}$ FLOPs per token = 140 GFLOPS per token at 1.3 PFLOPS FP8 effective (50% utilization typical) → 9.3 ns/token. Tokens/sec/server = $1 / 9.3$ ns × 8 chips = ~860 M tok/s aggregate forward (ignoring backward pass cost). Real numbers are lower due to communication overhead but the order is right.

**Q3.** *Why is MTIA's DDR5-based design considered ASIC-good rather than legacy?*

For DLRM, $I = 0.1$ FLOP/B. Ridge point $\pi/\beta$ — for MTIA v2: 4 TFLOPS / 200 GB/s = 20 FLOP/B. **Even at 20 FLOP/B ridge, DLRM is memory-bound**. Adding HBM (10 TB/s) wouldn't help because the bottleneck is the *random* access pattern, not aggregate bandwidth. MTIA's many small DDR5 channels have higher effective utilization on small random accesses than HBM's wide-bus design. Cheaper too.

**Q4.** *When does Trainium beat GPU and when does it lose?*

Beats: (a) steady-state inference of mainstream models (Claude, Llama-3) at >10 K chip scale — wins ~30% on $/inference; (b) pre-training of fixed-architecture models — Project Rainier serves Anthropic at scale.
Loses: (a) novel architectures requiring custom CUDA kernels; (b) interactive R&D where iteration speed matters more than per-chip cost; (c) exotic data types not in the Neuron Compiler's library; (d) workloads that can't be statically scheduled (RL with dynamic exploration loops).

**Q5.** *How does Triton-MTIA differ from upstream Triton?*

Same DSL syntax. Different backend: instead of emitting PTX, it emits Neuron VLIW. The kernel writer doesn't see the difference for most kernels. The differences leak through when:
(a) **Tile shapes** — Trainium's systolic dim is 512×128, not 32×16 like Hopper wgmma; tile sizing must change.
(b) **Synchronization primitives** — `tl.atomic` works differently (no warp-level shuffles).
(c) **Numerical formats** — MXFP8 is native; FP4 isn't supported as of 2026.
For ~80% of kernels, a CUDA-Triton kernel ports to Triton-MTIA with minor tile-size changes. The other 20% need rewrites.

---

## GPU Architecture — The SIMT Reference Chip

*From [GPU_Architecture.md](../L3_Microarchitecture/GPU_Architecture.md)*

**Q1.** *A kernel uses 80 registers/thread on Blackwell. What's the max occupancy?*

Per-SM RF = 65 536 regs. Threads = 65 536 / 80 = 819 threads. Warp count = ⌊819/32⌋ = 25 warps. Out of 64 warp slots → ~39% occupancy. Whether this is acceptable depends on $W_{\min}$ for the kernel's HBM dependency pattern. If the kernel is compute-bound (dense GEMM) then 25 warps is plenty (independent wgmmas hide each other). If memory-bound, occupancy is the limit.

**Q2.** *Why does Blackwell add TMEM as a *new* memory tier instead of just doubling SMEM?*

SMEM bandwidth varies by generation: ~19 TB/s/SM on A100 (164 KB SMEM), ~30 TB/s/SM on H100/B200 (228 KB SMEM). The ~30 TB/s/SM figure here is for Hopper/Blackwell. FP4 wgmma operand demand on Blackwell is ~50 TB/s/SM. Doubling SMEM doubles capacity, not port count, so it doesn't help. TMEM has wide read ports (1 024 b each) geometrically matched to wgmma tile rows, and is accessible only by tensor cores → no contention with general SMEM ops. Two separate memory tiers ≠ doubling one.

**Q3.** *Estimate the dense BF16 GEMM throughput on B200 for $M=N=K=8192$.*

$2 \cdot M \cdot N \cdot K = 1.1 \times 10^{12}$ FLOPs. Bytes loaded ≈ $2(MK + KN) + 4 MN = 2.4 \times 10^9$ bytes. AI = 458 FLOP/B. B200 FP16/BF16 ridge ≈ 280 FLOP/B → compute-bound. At 70% utilization of the ~4 500 TFLOPS peak: 3.1 PFLOPS effective. Elapsed: $1.1 \text{ TFLOPs} / 3.1 \text{ PFLOPS} = 0.35$ ms.

**Q4.** *Why does GPU decode (bs=1, 70B model) get <5% utilization regardless of generation?*

Decode reads all 70 GB of weights per token. AI = $2 \cdot 70\text{B} / (70\text{B} \cdot 2 B) = 1$ FLOP/B. Ridge point on B200 FP4 = 1 125 FLOP/B. Operating at AI = 1 means **1/1 125 ≈ 0.09%** of compute used. Throughput = $2 \cdot 70\text{B} / 8\text{TB/s} = 17.5$ ms/token = 57 tok/s. Compute-FLOPS-utilization is meaningless here; HBM-BW utilization is the meaningful metric and approaches 100%.

**Q5.** *How does TMA improve decode throughput when decode is HBM-bound, not issue-bound?*

It mostly doesn't. TMA helps prefill (compute-bound) by freeing scheduler issue slots for wgmma. In decode, the bottleneck is HBM reads of weights — TMA does the same number of HBM reads as a manual loop, just packaged into one descriptor. The marginal gain on decode is the small overhead of warp-issue-side address calculation, which is ~2% of total decode time. Decode-bound kernels can use TMA but it's not a step-function speedup.

---

## Google TPU — v4 / v5p / v5e / v6e Trillium / v7 Ironwood

*From [Google_TPU.md](../L3_Microarchitecture/Google_TPU.md)*

**Q1.** *Why does TPU OCS matter for production AI?*

OCS lets Google reconfigure the inter-rack interconnect to give each job a contiguous, dedicated torus slice. Three benefits: (a) **predictable performance** — no inter-job network contention; (b) **fault tolerance** — a failed link just gets routed around; (c) **flexible topology** — small jobs get $4×4×4$, big jobs get $32×32×32$, on the same physical hardware. NVIDIA's electrical-switched IB cannot do this — every job sees the same fabric and shares bandwidth with neighbors.

**Q2.** *Estimate TPU v5p performance on a per-head attention with $S=2048$, $d=128$.*

Per-head $Q \cdot K^T$: shape $S \times S$. Compute = $2 S^2 d = 1.07 \times 10^9$ FLOPs. With $M = S = 2048$, $K = 128$, $N = S = 2048$ → systolic K is 128, fits the MXU once with no fill loss; 2 048 / 128 = 16 sub-tiles in M, similar in N. Utilization should be ~85% on the array. Actual measured: ~50% because XLA softmax fusion adds non-MXU cycles. Pallas-based FlashAttention-equivalent recovers the gap.

**Q3.** *Why are TPUs commonly chosen for LLM pretraining but not for inference?*

Pretraining = dense GEMM over fixed shapes on long batches. Plays to TPU strengths: 95% MXU utilization, predictable, OCS scales to 9 K chips. Inference = variable batch + variable sequence + small batches at low TTFT. TPU loses on (a) small-shape MXU underutilization, (b) recompilation per dynamic shape, (c) lack of NVL72-style fabric for MoE EP at scale, (d) inference-engine ecosystem (vLLM/SGLang/TRT-LLM are CUDA-first).

**Q4.** *Why does Pallas exist when XLA already targets TPU?*

XLA is operator-level — it fuses + tiles known patterns (matmul, conv, softmax). For non-standard kernels (FlashAttention, custom MoE routing, exotic quantization), XLA underperforms because it can't match a hand-tuned algorithm. Pallas exposes a kernel-level Python DSL that compiles to direct VLIW bundles, letting users write FlashAttention-equivalent kernels at GPU-Triton-style productivity. As of 2026, Pallas is mainstream for advanced TPU users.

**Q5.** *Why is TPU's v5e ("Efficient") TF32 throughput much lower than v5p ("Performance")?*

v5e is cost-optimized for inference: smaller MXU count per chip, no OCS, smaller HBM. Per-chip throughput is ~½ of v5p. The bet: at ~½ price, $/TFLOP is comparable, and inference doesn't need the OCS-scale topology. v5e is what hosts most Google production inference today; v5p/v7 are reserved for training.

---

## Huawei Ascend — 910B / 910C / 910D

*From [Huawei_Ascend.md](../L3_Microarchitecture/Huawei_Ascend.md)*

**Q1.** *Why does the DaVinci 3D Cube help on SMIC 7nm specifically?*

SMIC's 7 nm has ~1.6× higher wire capacitance vs TSMC N4. Dynamic power = $\alpha C V^2 f$ — wire capacitance dominates at high frequency. The 3D Cube reduces operand-routing distance by ~30% relative to a 2D systolic array of equivalent throughput. Result: ~30% lower dynamic power for the same FLOPS, fitting within thermal budget. On TSMC N4, this effect is smaller and the 2D approach (NVIDIA, AMD, TPU) is fine; on SMIC 7nm, 3D is necessary.

**Q2.** *Estimate CM-384's aggregate FP8 throughput vs an NVL72 B200 rack.*

CM-384: 384 × ~600 TFLOPS FP8 (estimated, given 910C lacks native FP8) ≈ 230 PFLOPS. NVL72 B200: 72 × 4 500 = 324 PFLOPS. NVL72 wins ~40% on raw FP8 throughput, but CM-384 has ~3.5× the HBM (50 TB vs 14 TB). For inference (memory-bound), CM-384 can serve more concurrent requests; for training (compute-bound), NVL72 is faster.

**Q3.** *Why is the 910C's per-package throughput so much lower than B200's?*

Three multiplicative factors: (a) **process** — SMIC 7 nm has ~½ the transistor density of TSMC N4 → fewer Cubes fit per die; (b) **frequency** — 910C runs ~1.2 GHz vs B200's ~1.6 GHz, ~25% throughput delta; (c) **format** — 910C native is BF16; B200's FP4 doubles peak again. Combined: ~3× throughput gap at iso-precision, ~6× at FP4.

**Q4.** *What's the path forward for Ascend if SMIC can't access EUV?*

Two paths: (1) **Multi-patterning DUV all the way** — SMIC has demonstrated 5 nm-class via aggressive double-patterning, at lower yield. Ascend 910D is built on this. Cost is yield economics worse than TSMC by ~3×. (2) **Architectural innovation** — bigger packages (more dies stitched), more aggressive 3D stacking (logic-on-logic via SoIC equivalents), spatial-dataflow techniques. Long-term: SMIC EUV indigenization (5+ year horizon) or moving to non-CMOS (carbon nanotubes, tunnel FETs — research only).

**Q5.** *Why does CANN's PyTorch performance lag CUDA by 10–25%?*

Three reasons: (a) **Compiler maturity** — Huawei's CCE compiler is ~5 years behind nvcc in optimization passes; common patterns work, exotic kernels lose; (b) **Library coverage** — TBE has ~80% of cuDNN's primitive coverage; missing primitives fall back to slower vector-engine paths; (c) **Hardware** — DaVinci's 3D Cube is best for square matmuls; non-square shapes need padding or kernel restructuring.

---

## ISA and Execution Model — SIMT, VLIW, Spatial Dataflow

*From [ISA_and_Execution_Model.md](../L3_Microarchitecture/ISA_and_Execution_Model.md)*

**Q1.** *A CUDA kernel achieves 30% occupancy. Diagnose three possible causes.*

(a) **Register pressure**: kernel uses too many registers per thread → fewer threads fit → fewer warps → low occupancy. Check via `nvcc --ptxas-options=-v`. (b) **SMEM allocation**: per-block SMEM exceeds 1/3 of total SMEM, capping resident blocks at 3. (c) **Block size**: kernel launches with very small blocks (e.g., 32 threads), preventing the SM from filling its 64-warp slots even at full block count.

**Q2.** *Why is divergence less of a problem in AI workloads than in graphics?*

AI tensor ops (matmul, conv, attention) are **uniform across threads** — every thread does the same work on different data. The divergence cases (e.g., softmax masking, loss computation) are localized to small kernels and easy to refactor into branchless code (predicated execution, mask-based gating). Graphics has frequent per-pixel branching (different shaders, different materials), which is much harder to make uniform.

**Q3.** *A Trainium kernel needs to run with sequence length 1024 today and 2048 next week. What changes?*

The Neuron compiler must **re-compile** for the new shape because VLIW schedules every cycle. The new schedule has different unroll factors, different SRAM tile layouts, possibly different DMA patterns. There is no "dynamic shape" support without compilation. Production teams pre-compile per common shape and ship a shape catalog. CUDA, by contrast, can ingest dynamic shapes at runtime because the warp scheduler adapts.

**Q4.** *Why does Cerebras claim "no kernel API" as a feature?*

Spatial-dataflow execution is graph-defined — the chip is a physical pipeline shaped like the model. There's no equivalent of a CUDA kernel to write because there's no warp/thread to write code for. The compiler partitions the graph onto tiles. Selling point: data scientists never need to drop into CUDA. Cost: ML systems engineers can't squeeze 5% more out of Cerebras the way they can on a GPU; you get what the compiler gives you.

**Q5.** *Why is SASS undocumented while PTX is public?*

PTX is the forward-compatible interface contract — published so third-party tooling (CUDA libraries, compilers) can target NVIDIA hardware across generations. SASS is generation-specific and changes radically (e.g., Hopper added wgmma, Blackwell added new tile-loading instructions). NVIDIA reserves the right to change SASS without notice; documenting it would be a contract that constrains future hardware choices. Reverse-engineering tools (CUDA-MEMCHECK, cuobjdump) expose SASS, but only NVIDIA-internal compilers know its full semantics.

---

## Memory Hierarchy & Roofline — The Universal Performance Model

*From [Memory_Hierarchy_and_Roofline.md](../L3_Microarchitecture/Memory_Hierarchy_and_Roofline.md)*

**Q1.** *A kernel reports 2 TFLOPS on B200 (peak 4 500 TFLOPS FP8). Memory-bound or compute-bound? How do you tell?*

Achieved P = 2 TF; π = 4.5 PF. Way below peak ⇒ **suspected memory-bound**. To confirm, measure $Q$ (bytes moved). If $Q$ corresponds to ~0.25 TB/s of HBM (2 TF / 8 ridge ≈), and $\beta_{\text{HBM}} = 8$ TB/s, then BW utilization is only 3% → **neither bound** — kernel has issue/scheduling overhead. If $Q$ shows 7+ TB/s of HBM traffic → bw-bound, peak in this regime. Use Nsight Compute to get $Q$.

**Q2.** *Compute decode tokens/sec for a 405B FP8 model on B200 with 192 GB HBM.*

Weights = 405 GB → doesn't fit on one B200. Need 3+ B200 in TP. Tokens/sec per replica: $\beta / N\text{bytes} = 8 \text{ TB/s} / 405 \text{ GB} = 19.7$ tok/s. With TP=4 (one B200 holds ~100 GB of weights), each B200 reads ~100 GB/token but bandwidth scales: tokens/sec = 8 / 101 = ~79 tok/s. Per-replica throughput ~79 tok/s; at batch 64: 5 056 tok/s aggregate.

**Q3.** *Why doesn't doubling HBM bandwidth from 8 TB/s to 16 TB/s double prefill throughput?*

Prefill is compute-bound (AI = 8 192 ≫ ridge 562). Bandwidth is not the bottleneck. $P = \pi$ regardless of $\beta$. Doubling β shifts the ridge from 562 to 281 — a kernel formerly at AI = 400 (memory-bound) becomes compute-bound, but a kernel at AI = 8 192 was already compute-bound and gets nothing. Bandwidth doubling helps decode (linear gain) and short-context attention; doesn't help prefill or large GEMM.

**Q4.** *What's the AI of FlashAttention v2 forward on Hopper for $S = 4096$, $d = 128$?*

FA-2 keeps Q, K, V, O in HBM; everything else in SMEM. Per-block: load $B_r d$ Q + $S d$ K + $S d$ V; compute attention. Across whole sequence: load $S d$ Q (1×), full K and V $S/B_r$ times. Total Q ≈ $3 S d$ bytes loaded, vs $W \approx 4 S^2 d$ FLOPs. AI ≈ $4 S^2 d / (3 S d \cdot 2) = 2 S / 3$ ≈ 2 730. Way above H100 ridge → compute-bound. Hence FA-2 hits 70%+ utilization on H100.

**Q5.** *On Cerebras WSE-3 with ridge 6, why is utilization still not 100% on every kernel?*

Three reasons: (a) **inter-PE communication overhead** — even SRAM-only ops require routing across the 2D mesh, adding NoC cycles. (b) **Tile size mismatch** — fixed PE granularity wastes cycles when the kernel doesn't divide evenly. (c) **MemoryX streaming** — large models exceed on-die 44 GB; weights must stream from external MemoryX nodes, reintroducing a slower memory tier. Achieved utilization on 70B inference: ~50–70%, very impressive for the workload but not "all kernels at 100%".

---

## Specialty Accelerators — Cerebras, Groq, Tenstorrent

*From [Specialty_Accelerators.md](../L3_Microarchitecture/Specialty_Accelerators.md)*

**Q1.** *Why does Cerebras "invert the roofline"?*

GPU ridge: π/β ≈ 1 000 FLOP/B (FP4 B200) — most workloads are far below ridge → memory-bound. Cerebras ridge: 125 PFLOPS / 21 PB/s ≈ 6 FLOP/B — every workload above 6 FLOP/B (which is essentially everything: matmul has ~K, attention ~$d$, even decode at FP16 is 1 FLOP/B → just barely below). Compute-bound regime → tensor cores stay full → ~95% silicon utilization vs ~30% on a GPU running decode.

**Q2.** *Estimate per-token cost on a Groq 70 B inference vs B200.*

Groq: ~600 chips × $20K capex / 5-year amortization / 86 400 s/day / 365 days = ~$0.075 per chip-hour aggregate ⇒ ~$45/hr for the 600-chip pod. Throughput: ~500 tok/s → $0.025 per 1K tokens.

B200: $30K / 5 yr / hr → $0.7/hr. With batching, single B200 serves ~2 000 tok/s aggregate at FP8 70B. → ~$0.0001 per 1K tokens.

Groq is **250× more expensive per token but offers <1ms TTFT vs B200's 200 ms**. For latency-sensitive use cases, Groq wins; for cost-sensitive, B200.

**Q3.** *Why doesn't Cerebras dominate frontier-model training?*

Three reasons: (a) **per-system cost** — $3 M per CS-3, similar to 10 B200s. For a 100 K-chip equivalent training cluster, Cerebras is ~10 K wafers = $30 B, vs ~$3 B for GPU equivalent. (b) **weight streaming bottleneck** for >44 GB models — MemoryX is fast (~1.2 TB/s) but constrained vs HBM aggregate; large dense models bottleneck on weight load. (c) **ecosystem** — frontier-model researchers iterate on PyTorch + CUDA. Cerebras requires graph-level rewrites that slow R&D.

**Q4.** *Why is Tenstorrent Blackhole not a Cerebras competitor despite having a similar mesh structure?*

Tenstorrent uses HBM (chip + memory boards), so it's a normal accelerator with NoC dataflow. Cerebras eliminates HBM entirely. Tenstorrent inherits HBM's bandwidth ceiling (~10 TB/s) → same memory wall as a GPU. Cerebras beats the ceiling by going to on-die SRAM. Different architectural philosophies despite both using meshes.

**Q5.** *What workload would convince you to deploy specialty silicon over GPUs in production?*

(a) **Realtime trading or robotics inference** — sub-ms TTFT requirement → Groq.
(b) **Pharmaceutical sequence modeling** — long-context (1M+ tokens) on a small model → Cerebras.
(c) **Inference-only DLRM at <10K-chip scale where you can't justify MTIA NRE** — Tenstorrent.
For typical LLM serving, GPUs win on $/inference at any scale where ecosystem maturity matters more than architectural niche optimization.


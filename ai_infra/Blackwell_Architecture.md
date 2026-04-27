# Blackwell Architecture (B100 / B200 / GB200 / GB300 and Rubin Outlook)

The 2024–2026 NVIDIA hardware roadmap that anyone in AI infra needs to be fluent in. Blackwell brought 5th-generation tensor cores, FP4 / NVFP4, the second-gen Transformer Engine, NVLink-5, NVL72 / NVL576, and the GB200/GB300 superchip. This page covers the chip-level changes, the system-level (DGX, NVL domain), the new precision ecosystem, and the systems implications.

**Prerequisites**: [GPU_Architecture](GPU_Architecture.md), [Networking_and_Interconnect](Networking_and_Interconnect.md), [Modern_Quantization_Frontier](Modern_Quantization_Frontier.md).

---

## 1. The Generation Map

| Family | First shipping | Process | Highlights |
|--------|----------------|---------|------------|
| Volta (V100)  | 2017 | TSMC 12nm | 1st-gen tensor cores |
| Ampere (A100) | 2020 | TSMC 7N   | 3rd-gen, BF16, TF32 |
| Hopper (H100) | 2022 | TSMC 4N   | 4th-gen TC, FP8, TMA, wgmma, NVLink-4, Transformer Engine v1 |
| Blackwell (B100/B200) | 2024 | TSMC 4NP, dual-die | 5th-gen TC, FP4, FP6, NVFP4, MXFP, Transformer Engine v2, NVLink-5 |
| Blackwell Ultra (B300/B300A) | 2025 | TSMC 4NP, dual-die | Bigger HBM, more compute, Transformer Engine v2.x |
| Rubin (R100) | 2026 (announced) | TSMC 3N expected | Next-gen; details limited |
| Rubin Ultra (R200) | 2027 (announced) | – | – |

(I write Blackwell Ultra = B300; NVIDIA's commercial naming has been B100 / B200 / B300 across SKUs.)

---

## 2. Blackwell Chip Architecture

### 2.1 Dual-Die Design

B200 is a **dual-reticle GPU**: two compute dies connected by an ultra-high-bandwidth chip-to-chip interconnect (NV-HBI, ~10 TB/s die-to-die). To software, it appears as **one** GPU — the runtime / driver hides the die boundary.

This breaks the reticle-size limit on monolithic chips. Two Hopper-sized dies fused gives roughly 2× compute and 2× HBM in one logical GPU.

### 2.2 SM Updates

- **5th-gen tensor cores**: native FP4, FP6, MXFP4/6/8 support, plus all earlier formats.
- **Larger SMEM** per SM (~256 KB).
- **TMEM** (Tensor Memory): a new dedicated tensor-core scratchpad separate from SMEM, accessible only by tensor-core ops. Frees SMEM for non-TC work.
- **More registers** per SM, supports bigger tiles.
- **Async wgmma extensions**: support for FP4/FP6 inputs.

### 2.3 HBM

| Variant | HBM | BW |
|---------|-----|-----|
| H100    | 80 GB HBM3 | 3.35 TB/s |
| H200    | 141 GB HBM3e | 4.8 TB/s |
| B100    | 192 GB HBM3e | 8 TB/s |
| B200    | 192 GB HBM3e | 8 TB/s |
| B300    | 288 GB HBM3e | 12 TB/s (planned) |

The capacity bumps each reset the "fits / doesn't fit" math for big models. 192 GB B200 fits Llama-3-405B FP8 alone; 288 GB B300 fits 671B FP8.

### 2.4 Compute Throughput

Headline numbers (sparse, dense roughly half):

| Format | H100 (TFLOPS) | B200 (TFLOPS) | B300 (TFLOPS) |
|--------|---------------|---------------|---------------|
| FP16   | 989           | 2250          | 2700          |
| FP8    | 1979          | 4500          | 5400          |
| FP6    | —             | 4500          | 5400          |
| FP4    | —             | 9000          | 10800         |

Real applications hit 30-50% of peak, as on Hopper. FP4 doubles theoretical FP8 throughput; achievable speedup depends on calibration and kernel maturity.

---

## 3. NVFP4 and the New Precision Stack

Blackwell introduced **MXFP** (microscaling FP) formats, native to the tensor cores:

- **MXFP4 / NVFP4**: 4-bit mantissa-exponent values with a per-block (32-element) shared 8-bit exponent. ~4× compression vs FP16.
- **MXFP6 / NVFP6**: 6-bit version, ~2.7× compression.
- **MXFP8**: 8-bit microscaled version, 2× over FP16.

The "block" (32 elements) shares an exponent → captures dynamic range without per-element exponent bits. This is the OCP MX standard.

NVFP4 is NVIDIA's slight refinement (different block size details, possibly per-tensor scale on top). Treat them as the same family for systems work.

See [Modern_Quantization_Frontier](Modern_Quantization_Frontier.md) for the full quantization treatment.

### 3.1 Transformer Engine v2

NVIDIA's TE library, v2 for Blackwell, automates:
- Per-tensor and per-block scale tracking.
- FP4/FP6/FP8 selection per op.
- Smooth integration with PyTorch and Megatron.

A model wraps its Linear layers in TE's `Linear`, and TE picks precisions based on calibration. Out-of-the-box ~1.8× speedup over BF16 in FP8 mode; up to 3× in FP4 mode for matmul-heavy phases.

---

## 4. NVLink-5 and NVL Domains

### 4.1 Per-GPU NVLink

- **H100**: NVLink-4, 18 lanes × 50 GB/s = 900 GB/s aggregate per GPU (bidir).
- **B200**: NVLink-5, 18 lanes × 100 GB/s = 1.8 TB/s aggregate per GPU.

### 4.2 NVL72

GB200 NVL72: 72 GPUs in a single rack with NVLink fabric (NVSwitch backbone) connecting **all 72** at full NVLink BW. Effectively one giant logical NVLink domain.

For collectives (all-reduce, all-to-all) within NVL72, comm is ~6× faster than scaling out across an InfiniBand fabric. Critical for MoE EP, big TP, and large training runs.

### 4.3 NVL576 (planned)

GB200 NVL576 (and GB300 variants) extend the NVL fabric across multiple racks via NVLink Switch System. The whole domain operates as one big GPU pool.

### 4.4 NVLink-C2C (Grace-Hopper / Grace-Blackwell)

CPU-GPU link in superchips:
- **Grace-Hopper** (H100 + Grace CPU): 900 GB/s C2C, much faster than PCIe Gen5.
- **GB200** (2× B200 + Grace CPU per superchip): NVLink-5 between the GPUs, NVLink-C2C to Grace.

Enables CPU memory to act as a fast HBM extension (KV swap, weight tiering).

### 4.5 Spectrum-X / Quantum-X800

NVIDIA's Ethernet (Spectrum-X) and InfiniBand (Quantum-X800) switch families for Blackwell-era clusters:
- Quantum-X800: 800 Gb/s = 100 GB/s per port (XDR IB).
- Spectrum-X: 800 GbE with adaptive routing for AI workloads.

Plus BlueField-3 DPUs handle storage, security, and network offload — keeping the GPU-to-GPU network as clean as possible for collectives.

---

## 5. GB200 / GB300 Superchips

A **GB200** superchip:
- 1 × Grace CPU (72 Arm Neoverse-V2 cores, LPDDR5X memory).
- 2 × B200 GPUs (each with 192 GB HBM3e).
- Connected via NVLink-C2C and NVLink-5.
- 384 GB total HBM in one superchip.

A **GB200 NVL72 rack**:
- 36 superchips × 2 GPUs = 72 GPUs.
- 13.5 TB HBM3e in one NVL domain.
- 576 PFLOPS FP8 dense, 1.4 EFLOPS FP4 dense.

A **GB300** generation (Blackwell Ultra) bumps HBM to 288 GB per GPU (~21 TB per rack) and slightly higher compute.

This is the new building block for serving and training frontier models. Many production deployments target GB200 NVL72 racks as the smallest unit.

---

## 6. Systems Implications

### 6.1 KV Pool Math at NVL72 Scale

A single NVL72 with B200s holds 13.5 TB HBM. After model weights (e.g., 670 GB for DeepSeek-V3 FP8), KV budget is ~12.8 TB → million+ token slots across batch.

This radically changes the prefill-decode disaggregation calculus: a single rack can host a huge decode pool with massive prefix-cache hit potential.

### 6.2 EP Domain Size

With NVL72, EP=72 is now intra-NVL — all-to-all stays on NVLink. DeepSeek-V3 (256 experts) maps cleanly: 256 / 72 ≈ 3.5 experts per GPU (with replication for hot ones). Cluster-wide EP no longer requires inter-node IB.

### 6.3 TP at Scale

Llama-3-405B with TP=72? Possible, but diminishing returns past 16-32 due to communication overhead per layer. More common: TP=8 or 16 with DP/EP fanning out to fill the NVL72.

### 6.4 Training Throughput

GB200 NVL72 for pretraining 70B-class:
- ~6 EFLOPS FP8 dense per rack.
- At 50% MFU, ~3 EFLOPS effective.
- 70B at 15T tokens × 6 · 70e9 = 6.3e24 FLOPs → 2 days per rack? Surprisingly small. Reality 5-10× slower due to scheduling, comm, etc., but still order-of-magnitude faster than equivalent H100 fleet.

### 6.5 Liquid Cooling Becomes Mandatory

Per-GPU TDP:
- H100 SXM: 700 W.
- B200: 1000 W.
- GB200 superchip: 2700 W.
- GB300 superchip: ~3500 W projected.

NVL72 rack: ~120 kW. Air cooling is impossible at this density; **liquid cooling** (cold plates, often direct-to-chip) is standard. Implications for data-center facilities: new builds must be liquid-cooled-ready.

---

## 7. Rubin Outlook (Announced; Sparse Details)

Announced at Computex 2024:
- **Rubin GPU** (R100): 3N process, HBM4 (8 TB/s+), shipping ~2026.
- **Rubin Ultra**: ~2027.
- **Vera Rubin** superchip: with Vera CPU (next-gen Grace).
- **NVLink-6**, **Spectrum-Ultra**, **Quantum-Ultra** networking.
- **NVL576+** domains.

Treat Rubin as "Blackwell with more of everything" until concrete details land.

---

## 8. Competitive Landscape (Brief)

For context — questions sometimes go beyond NVIDIA:

| Vendor | Chip | HBM | FP8 (TFLOPS) | Notes |
|--------|------|-----|---------------|-------|
| AMD    | MI300X | 192 GB | ~2600 | ROCm, growing inference share |
| AMD    | MI325X | 256 GB | ~2700 | Late 2024 |
| AMD    | MI350X / MI355X | 288 GB | – | 2025 |
| Intel  | Gaudi-3 | 128 GB | ~1800 | OPI, niche use |
| Google | TPU v5p, v6e (Trillium) | 192-384 GB HBM | ~900-1800 | XLA-only, internal scale |
| Amazon | Trainium2 | 96 GB | ~1300 | AWS internal |
| Cerebras | WSE-3 | 44 GB SRAM (weights), no HBM | ~125K BF16 | Wafer-scale, niche workloads |
| Groq   | LPU | 230 MB SRAM, no HBM | – | Inference-only, super low latency |

NVIDIA dominates training; AMD MI300X has gained inference share at major clouds; specialized accelerators (Cerebras, Groq) win specific niches.

---

## 9. Programming Blackwell

### 9.1 Kernel-Level Changes

- **wgmma extensions** for FP4/FP6/FP8 mixed inputs.
- **TMEM access**: dedicated tensor-memory scratchpad. New PTX intrinsics; CUTLASS 3.5+ supports it.
- **CTA Pair / Distributed Shared Memory** extensions: clusters of CTAs (introduced on Hopper) get more capabilities.
- **Async barriers**: improved arrival-wait semantics for producer-consumer pipelines.

CUTLASS 3.5+, Triton 3.x (Blackwell support 2025), TileLang (BitMagic, 2024–2025), and direct PTX are the kernel-author paths.

### 9.2 Software Stack Maturity (early 2026)

- **PyTorch + Inductor**: native Blackwell support, FP8 / FP4 codegen via Inductor.
- **Transformer Engine v2**: production.
- **Megatron-Core**: full support including FP8 training pipelines.
- **vLLM**: Blackwell tested, FP4 inference experimental in late 2025, mainstream in 2026.
- **TensorRT-LLM**: full Blackwell support, FP4 mature.
- **NVIDIA Dynamo, NIM**: Blackwell-first.

For interview purposes: expect Blackwell-targeted production stacks to be standard reference.

---

## 10. Common Pitfalls

- **Treating B200 as Hopper-but-faster**: it's dual-die with NV-HBI, has TMEM, and supports FP4/FP6 — kernels need updating, not just rebuilding.
- **NVL72 bandwidth assumptions**: between two arbitrary GPUs in NVL72, BW is full NVLink-5 (1.8 TB/s) — but only if they're properly addressed; cross-die NV-HBI bandwidth differs.
- **Forgetting liquid cooling needs**: inheriting an air-cooled deployment plan onto Blackwell hardware doesn't work; thermal envelopes are bigger.
- **Underestimating FP4 calibration effort**: the precision is tight; production deployment requires careful per-block scale calibration and fallback paths.
- **Power budgeting**: a GB200 rack is 100+ kW; existing data-center power circuits may need upgrades.
- **NVL576 as one big GPU**: it isn't; comm cost still varies with hop count and link contention. Most apps still optimize for intra-NVL72 first.

---

## 11. Common Interview Questions

**Q: What's new in Blackwell vs Hopper?**
A: Dual-die GPU (NV-HBI chip-to-chip), 5th-gen tensor cores with native FP4/FP6/MX support, TMEM (dedicated tensor-core scratchpad), 192 GB HBM3e at 8 TB/s, NVLink-5 (1.8 TB/s per GPU), NVL72/NVL576 fabric, Transformer Engine v2. Roughly 2.25× FP16 throughput, 4× FP4 vs FP8 on Hopper.

**Q: What's NVFP4 and why does it matter?**
A: 4-bit floating-point format with per-block (e.g., 32-elem) shared exponent. Native tensor-core support on Blackwell. ~4× compression vs FP16 with much better accuracy than naive INT4. Halves matmul bytes again over FP8, doubling theoretical throughput. Critical for the next round of inference cost reduction.

**Q: Why is NVL72 a big deal?**
A: 72 GPUs in one NVLink domain at full NVLink-5 bandwidth. Collectives (all-reduce, all-to-all) that previously crossed slow IB now stay on fast NVLink. Enables intra-fabric EP=72 for MoE, larger TP for big dense models, and 13.5 TB of pooled HBM for KV cache at serving scale.

**Q: What does TMEM do and why is it needed?**
A: Dedicated tensor-core scratchpad memory separate from SMEM. Tensor-core operations on Blackwell have grown so fast that they can starve on SMEM bandwidth shared with other ops. TMEM is exclusive to TC operands, freeing SMEM for cooperative work and reducing register spills.

**Q: How does the dual-die B200 appear to software?**
A: As a single GPU. The driver and runtime hide the die boundary; kernels and CUDA contexts treat it as one. Cross-die memory access via NV-HBI is fast (~10 TB/s) but non-zero — careful kernel designers can get incremental wins by die-aware tile placement; most don't bother.

**Q: What's a GB200 superchip?**
A: One Grace CPU + two B200 GPUs on a single board, connected by NVLink-C2C (CPU-GPU) and NVLink-5 (GPU-GPU). 384 GB total HBM. The building block of NVL72: 36 superchips per rack = 72 GPUs.

**Q: What's the implication of Blackwell's higher TDP?**
A: Per-GPU TDP from 700W (H100) to 1000W (B200); NVL72 racks at 100+ kW. Air cooling is impossible at this density — liquid cooling (direct-to-chip cold plates) is mandatory. Existing data centers may need power and cooling upgrades to deploy GB200/GB300.

**Q: How do you decide between H100, B200, and B300 for a deployment?**
A: H100 is the well-understood baseline (still purchasable, abundant cloud capacity). B200 is the throughput choice (~2.25× FP16, ~4× FP4) when supply and software stack support are sufficient. B300 is the long-context / largest-model choice (288 GB HBM). Production teams in 2026 typically have all three in their fleet, with serving routed by model size and SLO.

**Q: Why would FP4 inference deliver "less than 2× speedup over FP8" in practice despite 2× peak throughput?**
A: Real workloads are partially memory-bound (decode), have fixed comm and overhead, and not 100% of ops are matmuls (norms, activations stay higher precision). Plus FP4 accumulation still happens in FP32 — no extra compute saved there. Realistic FP4-vs-FP8 speedup: 1.4-1.7× on decode-heavy inference, 1.7-1.9× on prefill-heavy.

**Q: What's NVLink-C2C and where does it fit?**
A: GPU-CPU link in Grace-Hopper / Grace-Blackwell superchips. ~900 GB/s — much faster than PCIe Gen5 (64 GB/s). Lets the CPU's LPDDR5X act as a fast extension of GPU memory: KV swap to host RAM has 14× higher BW than PCIe, making host-tier KV practical.

**Q: How does Spectrum-X differ from Quantum-X800?**
A: Spectrum-X is Ethernet (800 GbE) optimized for AI with adaptive routing and lossless flow control. Quantum-X800 is InfiniBand (800 Gb/s = 100 GB/s, XDR generation). Both achieve similar AI workload performance; Ethernet is more compatible with general DC infrastructure, IB is the historical AI default.

**Q: What changes for kernel writers on Blackwell?**
A: Use 5th-gen wgmma with FP4/FP6 inputs; access TMEM via new PTX; expect slightly different SMEM/register budgets per SM; pipeline producer-consumer with TMA + TMEM. CUTLASS 3.5+ and Triton 3.x abstract this; raw PTX writers need to learn the new mnemonics.

**Q: When would you target Rubin?**
A: For long-term roadmap planning, since Rubin ships in late 2026 onwards. For immediate work, target Blackwell. Expect Rubin to bring HBM4, NVLink-6, and another generational compute uplift — but software stack maturity will lag behind hardware availability for the first 6-12 months.

---

## 12. Further Reading

- NVIDIA Blackwell architecture whitepaper.
- NVIDIA GTC 2024 / 2025 keynotes.
- Transformer Engine v2 documentation.
- Open Compute Project MX format specification.
- NVIDIA NVLink-5 / Quantum-X800 / Spectrum-X technical briefs.
- "Microscaling Data Formats for Deep Learning" (OCP, 2023).
- HotChips presentations on Blackwell, GB200, Rubin.

---

**Next:** [Modern_Quantization_Frontier](Modern_Quantization_Frontier.md), [Disaggregated_Serving_2025](Disaggregated_Serving_2025.md).
**See also:** [GPU_Architecture](GPU_Architecture.md), [Networking_and_Interconnect](Networking_and_Interconnect.md), [Frontier_Models_2025_2026](Frontier_Models_2025_2026.md).

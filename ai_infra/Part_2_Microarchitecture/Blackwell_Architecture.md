# Blackwell Architecture (B100 / B200 / GB200 / GB300 and Rubin Outlook)

A rigorous, graduate-level architectural analysis of the NVIDIA Blackwell generation. This text details the microarchitectural mechanisms that enable sub-word SIMD execution, the mathematical imperatives driving the introduction of Tensor Memory (TMEM), and the physical infrastructure of the NVL72 Clos network.

**Prerequisites**: [GPU_Architecture](GPU_Architecture.md), [Networking_and_Interconnect](../Part_3_Hardware_Systems_and_Interconnects/Networking_and_Interconnect.md), [Modern_Quantization_Frontier](../Part_5_Algorithms_and_Quantization/Modern_Quantization_Frontier.md).

---

## 1. The Generation Map

| Family | First shipping | Process | Architectural Highlights |
|--------|----------------|---------|------------|
| Volta (V100)  | 2017 | TSMC 12nm | 1st-gen tensor cores |
| Ampere (A100) | 2020 | TSMC 7N   | 3rd-gen, BF16, TF32 |
| Hopper (H100) | 2022 | TSMC 4N   | 4th-gen TC, FP8, TMA, wgmma, NVLink-4, Transformer Engine v1 |
| Blackwell (B100/B200) | 2024 | TSMC 4NP, dual-die | 5th-gen TC, FP4, FP6, NVFP4, MXFP, Transformer Engine v2, NVLink-5 |
| Blackwell Ultra (B300) | 2025 | TSMC 4NP, dual-die | 288 GB HBM3e, expanded thermal envelopes |
| Rubin (R100) | 2026 (announced) | TSMC 3NP expected | HBM4, native NVLink-6 (3.6 TB/s bidir) |

---

## 2. Microarchitecture: The FP4 Regime and TMEM

### 2.1 Dual-Die Design and NV-HBI
B200 utilizes a **dual-reticle GPU** architecture to circumvent the ASML extreme ultraviolet (EUV) reticle limit ($858 \text{ mm}^2$). Two compute dies are bridged via NV-HBI (NVIDIA High Bandwidth Interface) over a CoWoS-L silicon interposer.
- **Aggregate Bandwidth**: $\sim 10$ TB/s die-to-die.
- **Clocking Dynamics**: The crossing is mesochronous. A shared reference clock minimizes phase skew, allowing Clock Domain Crossing (CDC) to occur via extremely shallow elastic buffers (1-2 clock cycles latency). While the CUDA programming model presents this as a unified SM array, maximal arithmetic intensity requires compiler-enforced thread-block affinity to avoid cross-die memory thrashing.

### 2.2 Sub-Word SIMD and `wgmma` Execution
Blackwell's 5th-generation Tensor Cores achieve exactly $2\times$ the throughput of FP8 by natively supporting the OCP Microscaling (MXFP4) format.

This is executed via **sub-word SIMD**. When a warp-group matrix-multiply-accumulate instruction (`wgmma.mma_async.m64n256k16`) is issued for FP4:
1. The hardware does not replicate discrete physical FP4 multipliers. Instead, it dynamically reconfigures the Wallace tree of a 16-bit MAC block to perform four independent $2 \times 2$-bit mantissa multiplications simultaneously.
2. A shared 8-bit exponent (from the MX block format) is decoded once and applied to all elements in the block, amortizing the silicon area of exponent alignment logic.

### 2.3 The Mathematical Necessity of TMEM
Let $P_{FP4}$ be the peak compute rate: $P_{FP4} = 9.0 \times 10^{15}$ FLOPs/s.
A matrix multiplication $C = A \times B$ requires fetching operands $A$ and $B$. For every 2 FLOPs (one multiply-add), 2 input operands (FP4 = 0.5 Bytes each) are required. 
The internal SRAM read bandwidth required to feed the Tensor Cores without starvation is:
$$\beta_{SRAM} = \frac{P_{FP4}}{2} \times 1.0 \text{ Byte} = 4.5 \text{ TB/s per GPU}$$
*(Note: This is an idealized lower bound ignoring block reuse; actual temporal staging demands even higher peak instantaneous bandwidth).*

On Hopper, the Tensor Cores read directly from Shared Memory (SMEM). On Blackwell, sustaining this read bandwidth would cause catastrophic bank conflicts with normal CUDA threads.
**Solution: TMEM (Tensor Memory)**. 
- TMEM is a dedicated, highly multi-ported scratchpad SRAM tightly coupled to the Tensor Cores. 
- The **TMA (Tensor Memory Accelerator)** operates asynchronously, issuing bulk PCIe-style DMA transactions to stream tiles directly from HBM/L2 into TMEM.
- The `wgmma` instruction fetches exclusively from TMEM. This physical segregation of operand staging from general-purpose SMEM is the architectural keystone of the Blackwell SM.

---

## 3. Scale-Up Fabric: NVLink-5 and the NVL72 Clos Network

### 3.1 NVLink-5 Signaling
NVLink-5 achieves 100 GB/s per lane (unidirectional) via **224 Gbps PAM4 SerDes**. At a Nyquist frequency of 56 GHz, copper trace insertion loss is extreme, limiting passive copper run lengths to a single rack.
Each B200 features 18 NVLink-5 links, yielding an aggregate bidirectional bandwidth of $18 \times 200 \text{ GB/s} = 3.6$ TB/s (or 1.8 TB/s unidirectional).

### 3.2 NVL72 Network Topology
An NVL72 rack creates a single logical 72-GPU memory domain, critical for the All-to-All collectives in Mixture of Experts (MoE) routing.
- **Topology**: It utilizes a non-blocking 2-tier folded Clos network.
- **Hardware Mapping**: The backbone consists of 9 NVSwitch trays. Each tray contains 2 NVSwitch ASICs.
- **Switch Radix Math**: An NVSwitch ASIC has a radix $r = 144$ (at 50 GB/s per port). 
- In a single-rack topology, the 72 GPUs connect directly to the switch trays via dense orthogonal copper backplanes. The total switch port capacity ($9 \text{ trays} \times 288 \text{ ports/tray} = 2592 \text{ ports}$) drastically exceeds the GPU port count ($72 \text{ GPUs} \times 36 \text{ ports/GPU} = 2592 \text{ ports}$). Wait, the math aligns perfectly: the network is precisely fully populated, offering purely non-blocking 1.8 TB/s any-to-any bandwidth.

---

## 4. Systems Implications and Thermodynamics

### 4.1 NVLink-C2C and KV Cache Pooling
The GB200 Superchip integrates 2 $\times$ B200 GPUs with a Grace Arm CPU via **NVLink-C2C** (900 GB/s coherent interconnect). 
In LLM inference, when the KV cache overflows the HBM pool, swapping to host memory over standard PCIe Gen5 (64 GB/s) causes severe pipeline bubbles. NVLink-C2C elevates the Grace CPU's LPDDR5X (up to 480 GB) into a performant Tier-2 KV cache, radically expanding the maximum concurrent batch size for million-token context workloads.

### 4.2 Rack Power Math
A GB200 NVL72 rack consumes $P = 120$ kW. This energy density fundamentally invalidates air cooling.
Using the sensible heat equation for Direct-to-Chip (D2C) liquid cooling:
$$Q = \dot{m} C_p \Delta T$$
Where $Q = 120,000$ J/s, and water's specific heat $C_p = 4184$ J/(kg·K). Assuming a strict $\Delta T = 10^\circ$C to prevent thermal throttling of the temperature-sensitive HBM3e modules:
$$\dot{m} = \frac{120,000}{4184 \times 10} \approx 2.87 \text{ kg/s}$$
This sustained flow rate of $\sim 45$ gallons per minute mandates continuous, pressurized single-phase liquid loops supported by facility-scale Coolant Distribution Units (CDUs).

---

## 5. Further Reading

- [Accelerator_Landscape_2026](Accelerator_Landscape_2026.md)
- [AMD_Instinct](AMD_Instinct.md)
- Transformer Engine v2 documentation.
- OCP MX format spec (Microscaling).

**Next:** [Cloud_ASICs](Cloud_ASICs.md), [Google_TPU](Google_TPU.md), [ISA_and_Execution_Model](ISA_and_Execution_Model.md).

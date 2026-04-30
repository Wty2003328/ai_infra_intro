# AMD Instinct (CDNA 3 / CDNA 4 / "Helios" Rack — MI300X → MI455X)

A comprehensive architectural analysis of AMD's datacenter accelerator lineage, acting as the primary counterbalance to NVIDIA's dominance in 2025–2026. This chapter explores the microarchitectural evolution from CDNA 3 to CDNA 4, the mathematical realities of operand bandwidth in the FP4 regime, and the system-level topology of the Helios rack utilizing UALink.

**Prerequisites**: [GPU_Architecture](GPU_Architecture.md), [Blackwell_Architecture](Blackwell_Architecture.md), [Networking_and_Interconnect](../Part_3_Hardware_Systems_and_Interconnects/Networking_and_Interconnect.md), [Modern_Quantization_Frontier](../Part_5_Algorithms_and_Quantization/Modern_Quantization_Frontier.md).

---

## 1. Generation Map and Silicon Evolution

The CDNA architecture is optimized exclusively for highly parallel data center workloads, stripping away the graphics pipelines present in RDNA.

| Family    | Part                     | Year      | Process              | HBM          | Architectural Highlights                                                              |
| --------- | ------------------------ | --------- | -------------------- | ------------ | ------------------------------------------------------------------------------------- |
| CDNA 3    | MI300X                   | 2023      | TSMC N5+N6 (chiplet) | 192 GB HBM3  | First broadly-deployed AMD AI GPU; 8 XCD (compute) chiplets                           |
| CDNA 3    | MI325X                   | 2024      | N5+N6                | 256 GB HBM3e | Memory capacity iteration                                                             |
| CDNA 4    | MI350X                   | 2025      | TSMC N3 + N6 base    | 288 GB HBM3e | Air-cooled; matrix-core overhaul, FP4/FP6, VGPR forwarding                            |
| CDNA 4    | MI355X                   | 2025      | TSMC N3 + N6 base    | 288 GB HBM3e | Liquid-cooled, higher TDP, full FP4/FP6 throughput                                    |
| CDNA-Next | MI430X / MI440X / MI455X | 2026      | TSMC N2 (compute)    | 432 GB HBM4  | "Helios" rack-scale, native UALink SerDes integration                                 |

---

## 2. CDNA 4 Microarchitecture: The FP4 Memory Wall and VGPR Forwarding

### 2.1 Compute Unit (CU) Composition
A CDNA Compute Unit (CU) serves as the fundamental scheduling block (analogous to an NVIDIA SM). It contains:
- 4 SIMD32 vector ALUs (wave64 native execution).
- A Matrix Core (MFMA — Matrix Fused Multiply-Add), acting as the tensor engine.
- LDS (Local Data Share), functionally equivalent to Shared Memory (SMEM).
- Scalar units, schedulers, and a highly banked Register File.

### 2.2 The Operand Fetch Problem in FP4
In CDNA 4, the Matrix Core throughput is doubled compared to FP8 by shifting to FP4/MXFP4 (OCP Microscaling).
Let $P_{FP4}$ be the FP4 throughput (e.g., 20.1 PFLOPS on MI355X). The MAC engines require operands to feed these multipliers. For every fused multiply-add (2 FLOPs), three operands are involved ($A \times B + C$). The read bandwidth required from the register file to feed the Matrix Core scales linearly with compute.

With FP4 operations yielding 2x the FLOPs per clock of FP8, the required register file read bandwidth doubles. NVIDIA's Blackwell solved this by introducing TMEM (a dedicated SRAM). AMD chose a different, compiler-dependent path.

### 2.3 VGPR Operand Forwarding
CDNA 4 handles the massive $4\times$ overall operand fetch requirement (relative to older FP16 baselines) by physically expanding the **Vector General Purpose Register (VGPR)** file and increasing LDS bandwidth via wider banks. 

Instead of an autonomous SRAM like TMEM, CDNA 4 relies on **VGPR operand forwarding**. 
- Operands are loaded from HBM into LDS, then staged from LDS into the VGPRs via heavily unrolled, software-pipelined double-buffering.
- **Operand Forwarding Mechanism**: Data read from the VGPR is piped directly into the sub-word SIMD multipliers within the Matrix Core, bypassing intermediate operand latches. Concurrently, the next matrix tile is loaded from LDS into an alternate VGPR bank.
- **Mathematical constraint**: To sustain $P_{FP4}$, the LDS-to-VGPR transfer rate $\beta_{LDS}$ and the VGPR-to-MAC read rate $\beta_{VGPR}$ must satisfy: $\beta_{VGPR} \ge \frac{P_{FP4}}{2 \text{ FLOPs}} \times \text{Bytes per Operand}$.
This shifts immense register allocation and instruction scheduling complexity onto the compiler (Triton-AMD) and AMD's Composable Kernel (CK) library.

---

## 3. Scale-Up Fabrics: Infinity Fabric vs. UALink

### 3.1 Infinity Fabric / xGMI
Infinity Fabric forms the intra-node coherent link. On MI300X, GPUs are fully meshed at 896 GB/s aggregate bandwidth via xGMI. While optimal for 8-GPU nodes, it lacks the switch-based architecture required to scale to the 72-GPU domains demanded by modern Mixture of Experts (MoE) Expert Parallelism.

### 3.2 UALink (Ultra Accelerator Link) Physical Layer Math
To counter NVLink, the MI400 series integrates native **UALink v1.0**.
- **Signaling Rate**: UALink relies on 224 Gbps PAM4 SerDes. 
- **Topology Formulation**: A Helios rack implements a switched UALink fabric. To connect $N = 72$ GPUs, if each GPU exposes $L$ links and the UALink switches have radix $R$, the network forms a spine-leaf or fully-connected topology. 
Assuming an equivalent of NVLink-5's 1.8 TB/s bidirectional bandwidth per GPU, the all-to-all bisection bandwidth of a Helios rack is $72 \times 900 \text{ GB/s} = 64.8$ TB/s.

---

## 4. The Helios Rack System (CES 2026)

Helios is AMD's structural counterpart to the GB200 NVL72, transforming the MI455X from an isolated accelerator into a rack-scale computing instrument.

| System Property | Helios (Standard Rack) | Helios Double-Wide |
|---|---|---|
| Accelerator Count | 72 $\times$ MI455X | 144 $\times$ MI455X |
| Total HBM Pool | 31.1 TB (HBM4) | 62.2 TB (HBM4) |
| Aggregate HBM BW | ~1.4 PB/s | ~2.8 PB/s |
| FP4 Dense Peak | ~2.9 EFLOPS ($2.9 \times 10^{18}$) | ~5.8 EFLOPS |
| Rack Power (kW) | ~150 kW | ~300 kW |
| Scale-up Fabric | UALink (Intra-rack) | UALink |
| Scale-out Fabric| Ultra Ethernet (800 GbE) | Ultra Ethernet |

### 4.1 KV Cache Math at Rack Scale
For long-context LLM inference, the Memory Capacity Ceiling dictates throughput. 
Total Rack HBM on Helios is 31.1 TB. A massive dense model (e.g., Llama-4 400B) in FP8 requires $\sim 400$ GB.
Remaining capacity for KV Cache: $31.1 \text{ TB} - 0.4 \text{ TB} = 30.7 \text{ TB}$.
This massive coherent memory pool—enabled by UALink's load/store semantics—allows Helios to cache tens of millions of active tokens across the cluster simultaneously, mathematically dominating smaller 8-GPU islands.

---

## 5. Software Stack: ROCm 7 and Matrix Primitives

### 5.1 Composable Kernel (CK) vs. wgmma
NVIDIA's PTX provides the `wgmma` instruction for Hopper/Blackwell. AMD provides analogous MFMA intrinsics.
- **Canonical Tile**: CDNA 4 utilizes an MFMA $16 \times 16 \times 128$ FP8 instruction block. 
- **Composable Kernel (CK)**: Because CDNA lacks a hardware TMEM, developers rely on CK (AMD's CUTLASS equivalent). CK uses heavily templated C++ to generate perfectly interleaved `global_load` $\rightarrow$ LDS $\rightarrow$ `ds_read` $\rightarrow$ VGPR $\rightarrow$ `v_mfma` pipelines.

---

## 6. Thermodynamics and Liquid Cooling

An MI455X package consumes $\ge 1200$ W. A 72-GPU Helios rack consumes $\sim 150$ kW.
Using standard sensible heat transfer ($Q = \dot{m} C_p \Delta T$), removing 150 kW requires high-pressure single-phase Direct-to-Chip (D2C) liquid loops or two-phase boiling systems. The thermal resistance ($\theta_{JA}$) from the HBM4 die stacks through the CoWoS interposer to the cold plate must be minimized, as HBM4 is highly temperature-sensitive (maximum junction temperature $T_j \le 95^\circ$C). If $T_j$ exceeds this, the HBM controller throttles refresh rates, severely degrading the $19.6$ TB/s bandwidth ceiling.

---

## 7. Further Advanced Reading

- [Accelerator_Landscape_2026](Accelerator_Landscape_2026.md) - For Roofline comparisons.
- [Google_TPU](Google_TPU.md) - Google's alternative OCS-based scale-up architecture.
- [Modern_Quantization_Frontier](../Part_5_Algorithms_and_Quantization/Modern_Quantization_Frontier.md) - OCP MXFP4 formats and numerical limits.

**Next:** [Blackwell_Architecture](Blackwell_Architecture.md), [Networking_and_Interconnect](../Part_3_Hardware_Systems_and_Interconnects/Networking_and_Interconnect.md)
# Accelerator Landscape 2026 — Master Comparison

A comprehensive, mathematically rigorous synthesis of the AI accelerator ecosystem in 2026. This master comparison evaluates every major accelerator architecture—encompassing chip microarchitecture, scale-up fabrics, systems software, and power density—on standardized axes. This serves as the root index for graduate-level analysis of AI infrastructure. For deep architectural dives into specific platforms, follow the linked references.

**Prerequisites**: Familiarity with quantitative computer architecture, Roofline performance models, high-speed SerDes signaling, and multi-tier network topologies. Recommended initial reading: [Blackwell_Architecture](Blackwell_Architecture.md), [AMD_Instinct](AMD_Instinct.md), [Google_TPU](Google_TPU.md), [Cloud_ASICs](Cloud_ASICs.md), [Specialty_Accelerators](Specialty_Accelerators.md), [Huawei_Ascend](Huawei_Ascend.md).

---

## 1. The 2026 Production Lineup

The following table details the silicon that frontier-AI infrastructure teams will plausibly evaluate, procure, or operate at scale in 2026. Note that "dense" compute metrics are utilized, as sparse utilization ($\sim 2\times$ theoretical) heavily depends on compiler-enforced block sparsity.

| Vendor | Part | Microarch | Process | HBM (GB) | Mem BW (TB/s) | FP8 dense (PFLOPS) | FP4 dense (PFLOPS) | Scale-up domain (Max GPUs) | Software Stack | Max TDP (W) |
|---|---|---|---|---|---|---|---|---|---|---|
| NVIDIA | H100 | Hopper | TSMC 4N | 80 | 3.35 | 1.98 | – | NVLink-4 (8) | CUDA | 700 |
| NVIDIA | H200 | Hopper | TSMC 4N | 141 | 4.8 | 1.98 | – | NVLink-4 (8) | CUDA | 700 |
| NVIDIA | B200 | Blackwell | TSMC 4NP dual-die | 192 | 8.0 | 4.5 | 9.0 | NVL72 (72) | CUDA + TE2 | 1000 |
| NVIDIA | B300 | Blackwell Ultra | TSMC 4NP dual-die | 288 | 8.0 | 5.4 | 10.8 | NVL72 (72) | CUDA + TE2 | 1200 |
| NVIDIA | R100 | Rubin | TSMC 3NP dual-die | 288 (HBM4) | 22.0 | ~25.0 | ~50.0 | NVL576 (576) | CUDA + TE3 | ~1500 |
| AMD | MI300X | CDNA 3 | N5+N6 chiplet | 192 | 5.3 | 2.6 | – | xGMI (8) | ROCm | 750 |
| AMD | MI355X | CDNA 4 | N3 + N6 base | 288 | 8.0 | 10.1 | 20.1 | xGMI (8) | ROCm 7 | 1000 |
| AMD | MI455X | CDNA-Next | TSMC N2 | 432 (HBM4) | 19.6 | ~20.0 | ~40.0 | UALink (72-144) | ROCm 7+ | 1200 |
| Google | TPU v5p | TPU v5p | – | 95 | 2.7 | 0.459 (BF16) | – | ICI (8960-chip) + OCS | XLA / JAX | ~600 |
| Google | TPU v6e | Trillium | – | 32 | 1.6 | ~1.8 (INT8) | – | ICI (256-chip) | XLA / JAX | ~400 |
| Google | TPU v7 | Ironwood | – | 192 | 7.37 | 4.6 | – | ICI (9216-chip) | XLA / JAX / Pallas | ~900 |
| AWS | Trainium2| NeuronCore-v3 | TSMC 5nm | 96 | 2.9 | ~1.3 | – | NeuronLink (16) | Neuron + NKI | ~700 |
| AWS | Trainium3| NeuronCore-v4 | TSMC 3nm | 144 | 4.9 | 2.52 | (MXFP4) | NeuronLink (16) | Neuron + NKI | ~900 |
| Meta | MTIA v2 | Artemis (Mesh) | TSMC 5nm | 128 (LPDDR5)| 0.2 | inference-tier | – | PCIe / Meta-internal | Triton-MTIA | 90 |
| Cerebras | WSE-3 | SLAC mesh | TSMC 5nm wafer| 0.044 (SRAM) | 21,000 | 125 (FP16) | – | SwarmX (192 nodes) | Cerebras SDK | 23,000 (system)|
| Huawei | Ascend 910C| DaVinci 3.0 | SMIC 7nm | 128 (HBM3) | 3.2 | (INT8 emulation) | – | HCCS (384) | CANN + MindIE | ~600 |

*(Note: 1 PFLOPS = $10^{15}$ floating-point operations per second. Performance numbers reflect thermal steady-state dense throughput.)*

---

## 2. Quantitative Performance Modeling: The Memory Wall

To rigorously evaluate these architectures, we rely on the **Roofline Performance Model**, which maps the attainable performance of an algorithmic kernel based on its arithmetic intensity and the hardware's peak parameters.

### 2.1 Roofline Equations and Arithmetic Intensity

For a given kernel, let $W$ be the total floating-point operations (FLOPs) and $Q$ be the total memory bytes accessed from High Bandwidth Memory (HBM). The **Arithmetic Intensity** ($I$) is defined as:
$$I = \frac{W}{Q} \quad \text{(FLOPs/Byte)}$$

The hardware imposes two strict ceilings:
1. **Peak Compute ($\pi$)**: The maximum achievable operations per second (e.g., PFLOPS).
2. **Peak Memory Bandwidth ($\beta$)**: The maximum data transfer rate (e.g., TB/s).

The attainable performance ($P$) of a kernel is bounded by:
$$P = \min(\pi, I \times \beta)$$

The inflection point where a kernel transitions from memory-bound to compute-bound is the **Ridge Point ($I_{ridge}$)**:
$$I_{ridge} = \frac{\pi}{\beta}$$

### 2.2 The FP4 Math

As architectures shift to lower precision (FP8 to FP4), the multiplier silicon area shrinks quadratically, allowing a doubling of $\pi$. However, $\beta$ (HBM bandwidth) does not scale as rapidly. Consequently, the ridge point shifts dramatically rightward.

Let's calculate $I_{ridge}$ for contemporary architectures:
- **NVIDIA H100 (FP8)**: $\pi = 1.98 \times 10^{15}$, $\beta = 3.35 \times 10^{12} \implies I_{ridge} \approx 591$ FLOPs/Byte.
- **NVIDIA B200 (FP8)**: $\pi = 4.5 \times 10^{15}$, $\beta = 8.0 \times 10^{12} \implies I_{ridge} \approx 562$ FLOPs/Byte.
- **NVIDIA B200 (FP4)**: $\pi = 9.0 \times 10^{15}$, $\beta = 8.0 \times 10^{12} \implies I_{ridge} \approx 1125$ FLOPs/Byte.
- **AMD MI355X (FP4)**: $\pi = 20.1 \times 10^{15}$, $\beta = 8.0 \times 10^{12} \implies I_{ridge} \approx 2512$ FLOPs/Byte.
- **NVIDIA R100 (FP4, HBM4)**: $\pi = 50.0 \times 10^{15}$, $\beta = 22.0 \times 10^{12} \implies I_{ridge} \approx 2272$ FLOPs/Byte.

**Implications:**
For FP4 operations on an MI355X, a kernel must perform over 2500 FLOPs for every byte fetched from HBM. Standard attention mechanisms in LLM generation (autoregressive decode) possess an arithmetic intensity of $O(1)$, heavily anchoring them to the $\beta$ ceiling. Only extremely large batched matrix multiplications (prefill phase) or heavily fused kernels can approach $I \ge 2000$. This necessitates the integration of **HBM4**, which introduces a 2048-bit wide interface (double that of HBM3e) integrated directly on the base die, delivering the $>20$ TB/s required to keep 50 PFLOPS compute engines fed.

---

## 3. Scale-Up Fabric Architecture: NVLink vs. UALink vs. ICI

The "scale-up domain" refers to the maximum number of accelerators that can communicate via ultra-high-bandwidth, load/store cache-coherent interconnects before resorting to standard networking stacks (Ethernet/InfiniBand).

### 3.1 Signaling and Physical Layer Rigor
Scale-up interconnects operate at the edge of physical signaling limits.
- **NVLink-5 / UALink v1**: Both rely on **224 Gbps PAM4 SerDes** (Pulse Amplitude Modulation, 4 levels). At 224 Gbps, the Nyquist frequency is 56 GHz. At these frequencies, insertion loss over copper traces is extreme.
- A single NVLink-5 lane operates at 224 Gbps. Each "link" bundles multiple lanes. In Blackwell, 2 lanes per direction form a 50 GB/s link (unidirectional). Blackwell features 18 such links, totaling 900 GB/s per direction ($1.8$ TB/s bidirectional).

### 3.2 Topologies: NVL72 Clos vs. TPU Torus
**NVIDIA NVL72 (Clos Network)**
The NVL72 utilizes a non-blocking 2-tier folded Clos network.
- **Radix Math**: Let $r$ be the radix (number of ports) of a switch ASIC. An NVSwitch chip has a radix of 144 ports (at 50 GB/s each).
- In a 2-tier configuration, leaf switches connect to servers, and spine switches connect the leaves. The maximum number of nodes $N$ in a non-blocking 2-tier Clos is $N = \frac{r^2}{2}$.
- With $r=144$, the theoretical limit is $10,368$ endpoints, but physical copper backplane trace limits constrain NVL72 to 72 GPUs. The 9 NVSwitch trays (each containing 2 ASICs) provide $9 \times 288 = 2592$ total switch ports, forming a radically overprovisioned backbone that guarantees $1.8$ TB/s non-blocking all-to-all bandwidth.

**Google TPU v7 Ironwood (3D Torus + OCS)**
Unlike Clos networks, Google's Inter-Core Interconnect (ICI) employs a 3D Torus.
- A 3D Torus connects nodes in an $X \times Y \times Z$ grid. Each TPU has 6 ICI links (+X, -X, +Y, -Y, +Z, -Z).
- **Optical Circuit Switches (OCS)** dynamically reconfigure the torus dimensions by physically rotating MEMS mirrors, providing hardware-level fault tolerance and topology shaping (e.g., molding a $4 \times 4 \times 16$ torus into a $2 \times 8 \times 16$ for specific model parallel partitioning).

**AMD Helios UALink**
UALink introduces an open-standard switch-based topology analogous to NVLink, utilizing standard AST/Broadcom retimers and switches to form a coherent 72-144 GPU domain.

---

## 4. Scale-Out Fabrics and Congestion Control

When scaling beyond a single NVL72 or UALink pod, workloads rely on **Scale-Out** fabrics.

| Fabric | Speed (Per Port) | Transport | Congestion Control |
|---|---|---|---|
| InfiniBand XDR | 800 Gbps | Credit-based | Hardware flow control, lossless |
| Ultra Ethernet | 800 / 1600 GbE | RoCEv2 / UEC | Packet spraying, DCQCN, PFC/ECN |
| AWS EFAv3 | 400 Gbps+ | SRD (Scalable Reliable Datagram) | Multi-path dispersion, custom retransmission |

**Mathematical realities of RoCEv2:**
RDMA over Converged Ethernet (RoCEv2) relies on Priority Flow Control (PFC) and Data Center Quantized Congestion Notification (DCQCN).
- **DCQCN**: Adapts the sending rate $R_c$. If an ECN (Explicit Congestion Notification) packet is received, the sender cuts its rate: $R_c = R_c (1 - \frac{\alpha}{2})$, where $\alpha$ is an exponentially weighted moving average of the fraction of marked packets.
- **Incast problem**: In AI All-to-All collectives (e.g., MoE routing), $N-1$ nodes simultaneously transmit to 1 node, causing microbursts that overwhelm switch buffers. Ultra Ethernet Consortium (UEC) addresses this via packet spraying (breaking flows across equal-cost multi-path routes dynamically) rather than relying strictly on reactive ECN.

---

## 5. Thermodynamics and Power Math

As rack densities push past 120 kW (GB200 NVL72) to 150 kW+ (Helios / Rubin), the thermodynamics dictate infrastructure design.

### 5.1 Rack Power Density Math
Consider a GB300 NVL72 rack consuming $P = 140$ kW.
To remove this heat using Single-Phase Direct-to-Chip (D2C) liquid cooling, we apply the sensible heat equation:
$$Q = \dot{m} C_p \Delta T$$
Where:
- $Q = 140,000$ Watts (J/s)
- $C_p \approx 4184$ J/(kg·K) (Specific heat of water)
- $\Delta T$ is the allowable temperature rise of the coolant (typically $\sim 10$ K).

Solving for mass flow rate $\dot{m}$:
$$\dot{m} = \frac{140,000}{4184 \times 10} \approx 3.35 \text{ kg/s}$$
This translates to roughly 3.35 liters per second (or $\sim 53$ gallons per minute) of fluid continuously pumped through the microchannels of a single rack. This extreme flow rate demands highly redundant Coolant Distribution Units (CDUs) and reinforced facility plumbing, moving AI data centers functionally closer to chemical plants. Phase-change cooling (two-phase boiling) handles higher latent heat ($Q = \dot{m} h_{fg}$) but introduces massive pressure regulation complexity.

---

## 6. Workload-First Decision Matrix (Mathematical Context)

1. **Frontier Dense Pretraining ($\ge$ 100B params)**: Dominated by massive matrix multiplications. Requires high $\pi$ and mature distributed checkpointing. Default: NVL72, Ironwood Pods.
2. **Mixture of Experts (MoE) Pretraining**: Requires large Expert Parallel (EP) domains to minimize All-to-All latency. MoE All-to-All requires $\frac{N-1}{N} M$ data transferred across the fabric, where $M$ is total activation size. NVL72 (1.8 TB/s) and Helios UALink are mandatory to avoid Ethernet bottlenecks.
3. **Long-Context Inference ($>1$M tokens)**: KV cache size scales as $O(\text{batch} \times \text{seq\_len} \times \text{hidden\_dim})$. Memory capacity dictates constraints. AMD MI455X (432 GB) mathematically dominates B300 (288 GB) here, supporting $1.5\times$ the concurrent long-context requests per node before encountering Out-Of-Memory (OOM) or swapping to CPU via C2C.

---

## 7. Further Advanced Reading

For deeper systems engineering, mathematical derivation, and hardware synthesis:
- [Blackwell_Architecture](Blackwell_Architecture.md) -> Microarchitectural tensor pipeline behavior.
- [Networking_and_Interconnect](../Part_3_Hardware_Systems_and_Interconnects/Networking_and_Interconnect.md) -> Deep dive on DCQCN and Clos properties.
- [Production_Architecture](../Part_6_Inference_Engines_and_Serving/Production_Architecture.md) -> System-level fault tolerance.
- [Modern_Quantization_Frontier](../Part_5_Algorithms_and_Quantization/Modern_Quantization_Frontier.md) -> The calculus of FP4/MX microscaling.

**Next:** [AMD_Instinct](AMD_Instinct.md) | [Google_TPU](Google_TPU.md) | [ISA_and_Execution_Model](ISA_and_Execution_Model.md)
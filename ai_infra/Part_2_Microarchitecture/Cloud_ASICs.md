# Cloud-Vendor ASICs (AWS Trainium / Inferentia, Meta MTIA)

A rigorous, graduate-level analysis of hyperscaler in-house silicon. Unlike general-purpose GPUs, these ASICs are co-designed against specific internal workloads, trading programmability for extreme thermodynamic and cost efficiency at scale. AWS Trainium and Meta MTIA represent the two most consequential non-merchant silicon architectures in 2026.

**Prerequisites**: [GPU_Architecture](GPU_Architecture.md), [Networking_and_Interconnect](../Part_3_Hardware_Systems_and_Interconnects/Networking_and_Interconnect.md), [Quantization](../Part_5_Algorithms_and_Quantization/Quantization.md).

---

## 1. AWS Trainium / Inferentia Architecture

### 1.1 Generation Map and Scaling

| Family | Year | Process | Memory Subsystem | Compute Peak (Dense) | Scale-Up Domain | Architectural Focus |
|---|---|---|---|---|---|---|
| Inferentia 2 | 2023 | TSMC 7nm | 32 GB HBM2 | ~190 TFLOPS BF16 | Single Node | First HBM inference target |
| **Trainium 2** | 2024 | TSMC 5nm | 96 GB HBM3 (2.9 TB/s) | 1.3 PFLOPS FP8 | NeuronLink (16 chips) | Project Rainier scale |
| **Trainium 3** | 2026 | TSMC 3nm | 144 GB HBM3e (4.9 TB/s) | 2.52 PFLOPS FP8 / MXFP8 | NeuronLink (16 chips) | MXFP4 native, inference-training parity |

### 1.2 NeuronCore Pipelines and Microarchitecture
A Trainium 3 device contains **8 NeuronCore-v4** cores. The architecture fundamentally differs from SIMT (Single Instruction, Multiple Thread) execution on GPUs; it utilizes a statically scheduled, VLIW (Very Long Instruction Word) heterogeneous pipeline.

Inside a single NeuronCore:
- **Tensor Engine**: A dense systolic array. In v4, the matrix engine is dimensioned at $512 \times 128$ for MXFP8/MXFP4. A systolic array does not rely on a monolithic register file; instead, data flows latch-to-latch between Multiply-Accumulate (MAC) cells, dramatically reducing dynamic power consumption ($P \propto C V^2 f$).
- **Vector Engine**: An autonomous SIMD ALU for element-wise operations (e.g., GeLU, Softmax) containing specialized transcendentals (`exp`, `log`).
- **Scalar Engine**: Dedicated exclusively to loop control, address generation, and pointer arithmetic.
- **DMA Engine**: Handles asynchronous bulk transfers between HBM and the 32 MiB on-chip SRAM scratchpad.

**The Pipeline Math**:
To achieve peak utilization, the Neuron Compiler overlaps these engines in software via modulo scheduling. Let $T_{DMA}$, $T_{Tensor}$, and $T_{Vector}$ be the latencies of data load, matrix multiply, and activation. The compiler constructs an instruction stream such that at cycle $k$, the hardware is simultaneously executing:
- DMA load for tile $i+1$
- Tensor Engine MAC for tile $i$
- Vector Engine activation for tile $i-1$
Because there is no hardware warp scheduler to hide latency dynamically, if the compiler miscalculates loop bounds or memory dependency latency, the entire pipeline bubbles, reducing $I_{ridge}$ efficiency catastrophically.

### 1.3 Topology: NeuronLink and EFAv3
- **Intra-Server (NeuronLink)**: A 16-chip UltraServer relies on a direct-connect mesh topology yielding all-to-all non-blocking bandwidth.
- **Inter-Server (EFAv3)**: Cross-rack communication relies on AWS Elastic Fabric Adapter (EFA). Instead of standard RoCEv2 (which relies on reactive ECN/PFC), EFA utilizes **SRD (Scalable Reliable Datagram)**. SRD mathematically circumvents Ethernet's hashing limitations by employing multipath dispersion: it sprays packets across all equal-cost multipath (ECMP) routes simultaneously. Upon packet drop, SRD retransmits in $\mu$s rather than waiting for standard TCP RTO (Retransmission TimeOut) timers (typically milliseconds).

---

## 2. Meta MTIA (Meta Training and Inference Accelerator)

### 2.1 The Ranking and Recommendation Mandate
MTIA is co-designed for DLRM (Deep Learning Recommendation Models), where arithmetic intensity $I$ is astronomically low due to massive embedding table lookups.

### 2.2 MTIA v2 (Artemis) Architecture
- **Mesh of Processing Elements (PEs)**: MTIA v2 replaces the SM concept with an $8 \times 8$ grid of lightweight PEs.
- **Network-on-Chip (NoC)**: A dedicated crossbar NoC routes traffic from PEs to four DDR5 memory controllers.
- **Memory Bandwidth Math**: Unlike GPUs leveraging HBM, MTIA v2 utilizes DDR5, yielding $\sim 204.8$ GB/s. For an embedding lookup, reading a 128-dimensional FP16 vector (256 bytes) performs negligible FLOPs. A GPU with 8 TB/s HBM is severely underutilized on the MAC side during this. MTIA right-sizes the MAC-to-Bandwidth ratio to maximize Perf/Watt ($PFLOPS / kW$) on purely memory-bound irregular graphs.

### 2.3 MTIA v3 (Iris)
Moving to TSMC 3nm and incorporating HBM, v3 bridges the gap to LLM inference, addressing the massive KV-cache capacity demands while maintaining the highly predictable, compiler-managed static mesh routing of the PEs.

---

## 3. Comparison and Thermodynamic Efficiency

Cloud ASICs universally prioritize **Deterministic Execution**. By eliminating hardware branch predictors, out-of-order execution logic, and dynamic warp schedulers, the transistor budget allocated strictly to MACs and SRAM is vastly higher than on merchant GPUs.

**Implication for Infrastructure**:
Deploying Trainium or MTIA delivers disproportionate ROI ($>40\%$ cost reduction per inference query) strictly when the workload is mathematically stable (e.g., continuous Llama-3 decoding or DLRM serving). They are profoundly suboptimal for high-churn R&D where dynamic control flow and immediate PTX/CUDA availability are required.

---

## 4. Further Reading

- [Google_TPU](Google_TPU.md) - Google's approach to the systolic array.
- [GPU_Architecture](GPU_Architecture.md) - Contrast with dynamic SIMT execution.
- [Quantization](../Part_5_Algorithms_and_Quantization/Quantization.md) - The MXFP4 numeral formats used in Trainium 3.

**Next:** [Specialty_Accelerators](Specialty_Accelerators.md), [Huawei_Ascend](Huawei_Ascend.md)
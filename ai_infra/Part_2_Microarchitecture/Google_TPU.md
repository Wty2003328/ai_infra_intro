# Google TPU (v4 → v5p / v5e → v6e Trillium → v7 Ironwood)

A rigorous, textbook-level analysis of Google's custom AI silicon. The Tensor Processing Unit (TPU) stands as the singular non-NVIDIA fabric credibly hosting frontier-class training at NVL576-scale. This architecture diverges fundamentally from the GPU paradigm, prioritizing deterministic static scheduling via systolic arrays and utilizing a radically different scale-up topology: the 3D-torus Inter-Core Interconnect (ICI) coupled with Optical Circuit Switching (OCS).

**Prerequisites**: [GPU_Architecture](GPU_Architecture.md), [Networking_and_Interconnect](../Part_3_Hardware_Systems_and_Interconnects/Networking_and_Interconnect.md), [Parallelism_Strategies](../Part_7_Distributed_Systems_and_Scale/Parallelism_Strategies.md).

---

## 1. Generation Map and Silicon Evolution

| Generation | Codename | Year | Memory Subsystem | Compute Peak (FP8/BF16) | Topology | Pod Scale |
|---|---|---|---|---|---|---|
| TPU v4 | – | 2021 | 32 GB HBM2 | 275 TFLOPS BF16 | 3D Torus + OCS | 4096 chips |
| TPU v5p | "Performance" | 2023 | 95 GB HBM3 | 459 TFLOPS BF16 | 3D Torus + OCS | 8960 chips |
| TPU v5e | "Efficient" | 2023 | 16 GB HBM2 | 197 TFLOPS INT8 | 2D Torus (No OCS) | 256 chips |
| TPU v6e | **Trillium** | 2024 | 32 GB HBM3 | ~918 TFLOPS BF16 | 2D Torus | 256 chips |
| TPU v7 | **Ironwood** | 2025/2026 | **192 GB HBM3e** (7.37 TB/s) | **4614 FP8 TFLOPS** | 3D Torus + OCS | 9216 chips |

---

## 2. Microarchitecture: The MXU and SparseCore

### 2.1 The Systolic Array (MXU)
Instead of independent Streaming Multiprocessors (SMs) dispatching warps to execution units, the TPU relies on the Matrix Multiply Unit (MXU)—a massive systolic array.
- **Dataflow Math**: In a $128 \times 128$ MXU, data moves latch-to-latch across the grid. An element from matrix $A$ propagates horizontally, while an element from matrix $B$ propagates vertically. At each intersection $(i, j)$, a MAC operation occurs.
- **Throughput**: A single $128 \times 128$ MXU performs $128 \times 128 = 16,384$ MACs ($32,768$ FLOPs) per cycle. At $1.0$ GHz, this yields $32.7$ TFLOPS per array.
- **Utilization Penalty**: Systolic arrays achieve $>80\%$ utilization exclusively on large, dense, statically sized matrices. If a matrix dimension $K$ is not a multiple of 128, the compiler must pad the tensor with zeros, mathematically wasting active array cycles.

### 2.2 The SparseCore
Modern LLMs, Retrieval-Augmented Generation (RAG), and Deep Learning Recommendation Models (DLRM) require massive embedding lookups.
- **Mechanism**: The SparseCore is a dedicated asynchronous DMA and ALU engine. Instead of stalling the MXU to perform memory-bound vector gathers, the TPU dispatches a list of embedding indices to the SparseCore.
- **Bandwidth Optimization**: The SparseCore autonomously traverses the HBM modules to hash, gather, and reduce embedding vectors. The aggregated vectors are then placed into a unified software-managed SRAM for the TensorCore to consume at maximum bandwidth, entirely masking the latency of irregular DRAM accesses.

---

## 3. Scale-Up Fabric: 3D Torus ICI and Routing Protocols

While NVIDIA utilizes massive non-blocking Clos networks (NVL72), Google links TPUs directly to one another via the Inter-Core Interconnect (ICI) forming a 3D Torus.

### 3.1 3D Torus Topology Math
In a 3D torus, chips are arranged in an $X \times Y \times Z$ grid. Every TPU has exactly 6 bidirectional ICI links connecting to its immediate neighbors ($+X, -X, +Y, -Y, +Z, -Z$).
- Let $N$ be the total number of TPUs (e.g., $N = 4096$).
- The maximum distance (diameter) of an $X \times Y \times Z$ Torus is $\lfloor X/2 \rfloor + \lfloor Y/2 \rfloor + \lfloor Z/2 \rfloor$.
- **Bisection Bandwidth**: The bisection bandwidth $B_{bisect}$ of a symmetric 3D Torus is roughly $2 \times N^{2/3} \times b$, where $b$ is the per-link bandwidth. While a Clos network provides full $N \times b$ bisection bandwidth, a torus is much cheaper to wire but mathematically constrains all-to-all collectives.

### 3.2 ICI Routing Protocols: DOR and VLB
To prevent catastrophic congestion in the torus during heavy Ring All-Reduce or All-to-All operations, Google implements advanced hardware routing protocols:
1. **Dimension-Ordered Routing (DOR)**: Traffic is strictly routed first along the X-axis, then Y, then Z. This mathematically guarantees deadlock-free routing in a mesh, but can create intense thermal/bandwidth hotspots in the center of the torus during asymmetric traffic patterns.
2. **Valiant Load Balancing (VLB)**: To defeat hotspots, the TPU network stack optionally routes a packet to a *randomly selected intermediate node* before sending it to its final destination. While VLB strictly doubles the average path length (and thus base latency), it mathematically guarantees non-blocking worst-case throughput by perfectly uniformly distributing traffic across the entire $X \times Y \times Z$ volume.

### 3.3 Optical Circuit Switching (OCS)
To scale beyond a single physical cluster of racks without massive NVSwitch backplanes, Google utilizes OCS.
- OCS trays contain MEMS (Micro-Electro-Mechanical Systems) mirrors.
- By physically tilting microscopic mirrors, OCS dynamically reconfigures the $X, Y, Z$ wraparound links of the Torus.
- This allows a 9216-chip Ironwood Pod to mathematically partition itself. If a job requires 256 chips, the OCS physically redirects the light paths to form a perfect, isolated $4 \times 8 \times 8$ sub-torus in roughly $\sim 10$ milliseconds. It also inherently provides hardware-level fault tolerance by routing around failed silicon.

---

## 4. Software Stack: The XLA Compiler

Because TPUs lack a hardware warp scheduler, out-of-order execution, or dynamic branch prediction, the execution is entirely deterministic.
- **XLA (Accelerated Linear Algebra)**: The compiler analyzes the entire computation graph (from JAX or PyTorch/XLA) ahead of time. It issues VLIW instructions that dictate exactly which cycle data moves from HBM to the Vector Memory (VMEM), and which cycle the MXU consumes it.
- **Implications**: "Triton for TPU" does not exist in the same way because PTX dynamic dispatch doesn't exist. Instead, developers use **Pallas** (a JAX-embedded kernel DSL) to author precise tile-level memory movements that XLA lowers to static instructions.

---

## 5. Further Reading

- [GPU_Architecture](GPU_Architecture.md) - For contrast with SIMT dynamic scheduling.
- [Networking_and_Interconnect](../Part_3_Hardware_Systems_and_Interconnects/Networking_and_Interconnect.md) - For contrast with InfiniBand and Clos Networks.
- "TPU v4: An Optically Reconfigurable Supercomputer" (arXiv 2304.01433) - Canonical OCS mathematical analysis.

**Next:** [Cloud_ASICs](Cloud_ASICs.md), [Accelerator_Landscape_2026](Accelerator_Landscape_2026.md).

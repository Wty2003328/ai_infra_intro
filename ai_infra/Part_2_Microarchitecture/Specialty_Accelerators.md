# Specialty Accelerators (Cerebras, Groq, Tenstorrent)

A rigorous architectural analysis of the alternative hardware landscape. While GPUs (SIMT architectures) and TPUs (Systolic Arrays) dominate massive-scale frontier training, specialty accelerators exploit extreme hardware niches—specifically wafer-scale integration and deterministic SRAM execution—to shatter the traditional Roofline ceilings for latency-critical and batch-size-1 inference workloads.

**Prerequisites**: [GPU_Architecture](GPU_Architecture.md), [Memory_Hierarchy_and_Roofline](Memory_Hierarchy_and_Roofline.md), [Disaggregated_Serving_2025](../Part_6_Inference_Engines_and_Serving/Disaggregated_Serving_2025.md).

---

## 1. Groq: The Deterministic LPU (Language Processing Unit)

Groq's architecture completely eliminates High Bandwidth Memory (HBM), hardware warp schedulers, and all non-deterministic execution elements. It is an **SRAM-only, software-scheduled** streaming pipeline.

### 1.1 The GroqNode Microarchitecture
- **Memory Subsystem**: A Groq LPU contains no HBM. Instead, it features $\approx 230$ MB of ultra-high-speed, highly multi-ported SRAM strictly managed by the compiler.
- **Latency Math**: The fundamental breakthrough of Groq is predictability. Because there are no cache misses or hardware resource contentions, every instruction executes in a mathematically fixed number of clock cycles $T_{inst}$.
- Let the total number of sequential instructions for a transformer decode step be $N$. The execution latency $L_{decode}$ is perfectly calculable at compile-time: 
$$L_{decode} = \sum_{i=1}^{N} T_i \times \frac{1}{f_{clock}}$$
- By stringing hundreds of LPUs together via the GroqLink fabric, Groq guarantees sub-millisecond Time-To-First-Token (TTFT) and extreme token generation rates. For a 70B model with $\approx 140$ GB of weights, Groq requires spreading the model across roughly $140,000 \text{ MB} / 230 \text{ MB} \approx 600$ chips, performing pipelined execution where weights remain physically stationary in SRAM and activations stream through the fabric.

---

## 2. Cerebras: Wafer-Scale Integration (WSE-3)

Cerebras solves the interconnect latency and packaging constraints of traditional chiplets by simply not dicing the wafer.

### 2.1 The WSE-3 (Wafer-Scale Engine)
- **Scale**: The WSE-3 is a single $46,225$ mm$^2$ piece of silicon cut directly from a 300mm TSMC 5nm wafer. It contains 4 trillion transistors.
- **Compute and Memory**: It features 900,000 independent cores interconnected by a 2D mesh, and 44 GB of strictly on-die SRAM.
- **Memory Bandwidth Math**: With 44 GB of SRAM distributed across the cores, the aggregate internal memory bandwidth reaches $21$ Petabytes/second (PB/s).
- **The Arithmetic Intensity Inversion**: On a GPU, an $I_{ridge} \approx 1000$ FLOPs/Byte is required to saturate the compute units. On the WSE-3, $\beta_{SRAM} = 21,000$ TB/s. With a peak compute $\pi \approx 125$ PFLOPS (FP16), the ridge point collapses:
$$I_{ridge} = \frac{125 \times 10^{15}}{21 \times 10^{15}} \approx 5.9 \text{ FLOPs/Byte}$$
This implies that virtually all AI workloads—including inherently memory-bound autoregressive decoding and Sparse MoE—operate entirely within the compute-bound plateau, provided the model parameters fit within the 44 GB SRAM envelope.

### 2.2 SwarmX and MemoryX Disaggregation
For models exceeding 44 GB (e.g., GPT-4 or Llama-3-405B), Cerebras relies on **MemoryX** (external memory cabinets) and **SwarmX** (broadcast reduction fabric). The architecture shifts from storing weights on-wafer to streaming weights sequentially from MemoryX cabinets into the WSE-3. The wafer holds the activations, and the immense internal 21 PB/s bandwidth ensures the compute fabric is perfectly utilized during the matrix-vector products.

---

## 3. Tenstorrent: RISC-V and the Tensix Fabric

Tenstorrent's architecture (Blackhole / Quasar) blends RISC-V scalar processors with dedicated tensor hardware, connected via a massive on-chip Network-on-Chip (NoC) utilizing a standard Ethernet routing protocol internally.

### 3.1 The Tensix Core
- A Tensix core contains 5 RISC-V "Baby" cores for data routing and control, paired with a specialized math engine for dense MACs.
- This creates extreme programmability: instead of writing complex CUDA PTX or navigating XLA's static graph, a developer can mathematically express data movement using simple RISC-V C++ routines, while the tensor hardware autonomously dispatches the heavy linear algebra.

---

## 4. Further Reading

- [GPU_Architecture](GPU_Architecture.md) - For the contrast with SIMT execution.
- [Cloud_ASICs](Cloud_ASICs.md) - For AWS Trainium's statically scheduled VLIW approach vs. Groq's SRAM-only determinism.
- [Memory_Hierarchy_and_Roofline](Memory_Hierarchy_and_Roofline.md) - Understanding the $I_{ridge}$ math governing WSE-3's memory bandwidth advantage.
# GPU Architecture — Senior Engineer Level

This chapter details the fundamental hardware primitives that execute Large Language Models (LLMs). We dissect the Streaming Multiprocessor (SM), the mathematical imperatives of the Single Instruction, Multiple Thread (SIMT) execution model, the arithmetic density of Tensor Cores, and the architectural divergence between Hopper and Blackwell generations.

---

## 1. The Physics of GPU Compute: Latency vs. Throughput

### 1.1 The Workload Profile
Transformer inference is characterized by massive dense matrix multiplications (GEMMs) alongside memory-bound element-wise operations.
For a forward pass of an $N$-parameter model over a sequence of $S$ tokens, the required operations are approximately $2 \cdot N \cdot S$ FLOPs. 
A 70B parameter model processing a 4K context demands $\approx 5.7 \times 10^{14}$ FLOPs. CPUs, optimized for scalar latency, peak at a few TFLOPS. GPUs, optimized for vector throughput, deliver tens of PetaFLOPS.

### 1.2 The SIMT Latency Hiding Mechanism
A CPU avoids memory latency via massive L3 caches, branch prediction, and deep Out-Of-Order (OOO) execution windows. 
A GPU **hides** memory latency via massive thread oversubscription. 
When a warp (a group of 32 threads) stalls on an HBM load ($\sim 400$ cycles), the hardware warp scheduler performs a zero-cycle context switch to another resident warp. 

**Mathematical Occupancy Requirement**:
Let $L_{mem}$ be the latency of a memory load (in cycles), and $I_{warp}$ be the number of instructions a warp can execute before stalling on that load. If the SM scheduler issues 1 instruction per cycle, maintaining 100% pipeline utilization requires a minimum number of active warps $W_{min}$:
$$W_{min} = \frac{L_{mem}}{I_{warp}}$$
If $L_{mem} = 400$ cycles and $I_{warp} = 5$, the SM requires $\ge 80$ active warps. Given an SM max capacity of 64 warps (2048 threads), maintaining high **Occupancy** is mathematically mandatory to hide latency.

---

## 2. The Streaming Multiprocessor (SM) Microarchitecture

The SM is the atomic compute block of the GPU.

### 2.1 SM Partitioning and Schedulers
A modern SM (e.g., Hopper H100) is partitioned into 4 processing blocks. Each block contains:
- 1 Warp Scheduler & Dispatch Unit.
- 16 FP32 CUDA Cores (scalar ALUs).
- 1 Tensor Core (the matrix engine).
- A 16 KB slice of the Register File.

### 2.2 Register File Allocation and Register Pressure
The 256 KB Register File (RF) per SM is immense but statically partitioned among resident threads.
If a kernel launches with 256 threads per block, and requests 3 blocks per SM (768 threads total), the maximum registers per thread is:
$$R_{max} = \left\lfloor \frac{65536 \text{ registers}}{768 \text{ threads}} \right\rfloor = 85 \text{ registers/thread}$$
If the compiler (e.g., `nvcc`) requires 96 registers, it will either **spill** to local memory (devastating performance by dumping RF reads to L1/HBM) or forcibly reduce the number of resident blocks to 2, halving occupancy.

---

## 3. Tensor Cores and the MMA Pipeline

Tensor Cores are fixed-function matrix multiply-accumulate (MMA) units responsible for $>95\%$ of the SM's FLOPS.

### 3.1 Sub-Word SIMD and `wgmma`
On Hopper and Blackwell, the `wgmma` (Warp-Group MMA) instruction binds 4 warps (128 threads) to compute a $64 \times N \times 16$ tile collaboratively.
- **Arithmetic Density**: A $16 \times 16 \times 16$ MMA performs 4096 Fused Multiply-Adds (FMAs), or 8192 FLOPs, per instruction.
- **Asynchronous Execution**: `wgmma` is asynchronous. The warp group issues the instruction and immediately proceeds, allowing the SM to overlap index arithmetic and memory loads with the mathematical matrix computation.

---

## 4. The Memory Hierarchy: SRAM vs. HBM

### 4.1 Shared Memory (SMEM) and Bank Conflicts
Shared Memory is a 256 KB (on Hopper) programmer-managed SRAM.
It is physically divided into **32 banks** (each 32-bits wide). Memory accesses are served in parallel *only* if the 32 threads in a warp access 32 distinct banks.
- **The Bank Conflict Math**: If Thread 0 accesses Bank 0, and Thread 16 also accesses Bank 0 (e.g., due to a stride-2 access pattern), a 2-way bank conflict occurs. The hardware serializes the access into two cycles, instantly halving SMEM bandwidth. 
- **Mitigation**: High-performance kernels (like FlashAttention) rely on **XOR Swizzling** pointers to mathematically guarantee uniform bank distribution regardless of matrix stride.

### 4.2 Tensor Memory Accelerator (TMA)
Introduced in Hopper, TMA is an asynchronous hardware DMA engine.
Instead of consuming warp cycles calculating 2D array offsets, a single thread issues a TMA instruction specifying a multidimensional tensor coordinate. The TMA engine autonomously traverses the HBM, performs bounds checking, swizzles the data to avoid SMEM bank conflicts, and deposits the tile into SMEM, raising an `mbarrier` flag upon completion.

### 4.3 High Bandwidth Memory (HBM) Math
HBM achieves immense bandwidth via bus width, not clock speed.
A Blackwell Ultra B300 utilizes 8 HBM3e stacks. Each stack has a 1024-bit interface.
$$\text{Total Width} = 8 \times 1024 = 8192 \text{ bits} = 1024 \text{ Bytes/clock}$$
At a DDR (Double Data Rate) clock of $\approx 5.85$ GHz, the aggregate bandwidth yields $\approx 12$ TB/s.
For autoregressive LLM decode (batch size 1), the arithmetic intensity is $\approx 2$ FLOPs/Byte. Thus, decode throughput is strictly and exclusively bounded by this 12 TB/s HBM limit, rendering the SM's 15 PFLOPS largely idle.

---

## 5. Further Reading

- [Blackwell_Architecture](Blackwell_Architecture.md) - Deep dive on TMEM and FP4.
- [Memory_Hierarchy_and_Roofline](Memory_Hierarchy_and_Roofline.md) - Advanced Roofline analysis.
- [ISA_and_Execution_Model](ISA_and_Execution_Model.md) - PTX and SASS compilation targets.
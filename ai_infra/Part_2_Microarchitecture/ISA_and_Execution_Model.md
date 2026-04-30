# ISA and Execution Model (PTX, SASS, and SIMT)

A rigorous analysis of the GPU Instruction Set Architecture (ISA) and the compiler stack that translates high-level PyTorch or CUDA code into hardware-executable binaries. Understanding the dichotomy between PTX and SASS is mandatory for authoring peak-performance kernels, as the abstraction often masks performance-destroying latency and register spills.

**Prerequisites**: [GPU_Architecture](GPU_Architecture.md), [Memory_Hierarchy_and_Roofline](Memory_Hierarchy_and_Roofline.md).

---

## 1. The Compilation Stack: PTX vs. SASS

NVIDIA employs a two-stage compilation model.

### 1.1 PTX (Parallel Thread Execution)
PTX is a stable, virtual ISA. It presents a machine-agnostic execution model:
- **Infinite Virtual Registers**: A PTX program can assume an unlimited supply of virtual registers (e.g., `%f1, %f2, ...`).
- **Forward Compatibility**: A PTX binary compiled for Ampere can generally execute on Blackwell. The CUDA driver Just-In-Time (JIT) compiles the PTX into the target GPU's native machine code upon launch.
- **Limitation**: Because PTX assumes infinite registers, analyzing a `.ptx` file cannot reveal register pressure, spilling, or true execution latency.

### 1.2 SASS (Streaming ASSembler)
SASS is the true hardware ISA. It is undocumented, proprietary, and changes radically between generations.
- **Physical Register Allocation**: SASS operates strictly on physical registers (e.g., `R0, R1, ... R255`).
- **Instruction Latency and Yield**: The `nvcc` compiler (specifically `ptxas`) schedules SASS instructions to hide latency. SASS includes control codes indicating instruction dependencies and required stall cycles.
- **Analysis**: Advanced kernel optimization requires analyzing SASS dumps (via `cuobjdump -sass`) to verify that the PTX `wgmma` virtual instructions lowered into optimal, register-efficient hardware instructions without spilling to local memory.

---

## 2. SIMT Mathematical Execution Constraints

The Single Instruction, Multiple Thread (SIMT) model forces 32 threads (a warp) to execute in lockstep.

### 2.1 Warp Divergence Penalty
If a warp encounters a data-dependent branch (`if/else`), and $k$ threads evaluate to `true` while $32-k$ threads evaluate to `false` (where $0 < k < 32$), the hardware serializes the execution.
- **Math**: Let $T_{true}$ be the latency of the true branch, and $T_{false}$ be the latency of the false branch. The total execution time $T_{warp}$ for the divergent warp is mathematically:
$$T_{warp} = T_{true} + T_{false}$$
Throughput drops proportionally to the magnitude of the divergence.

### 2.2 Independent Thread Scheduling (Volta+)
Prior to Volta, diverging warps relied on a single Program Counter (PC) and an active mask. Volta introduced Independent Thread Scheduling, where each thread maintains its own PC and call stack. While this mathematically prevents deadlocks in complex concurrent algorithms (like fine-grained spinlocks), it does **not** eliminate the $T_{true} + T_{false}$ throughput penalty; divergent execution is still serialized over the ALUs.

---

## 3. Instruction Dispatch and Latency Hiding

An SM scheduler issues SASS instructions. Let the arithmetic pipeline depth be $D$ cycles (e.g., $D=4$ for a basic FMA).
If Warp 0 issues an FMA on registers $R1, R2 \rightarrow R3$, the result $R3$ is not mathematically available until cycle $t+4$.
To prevent the ALU pipeline from stalling during cycles $t+1, t+2, t+3$, the warp scheduler must issue independent instructions from Warp 1, Warp 2, and Warp 3.
This dictates the **Minimum Pipeline Occupancy**:
$$W_{compute\_bound} \ge \text{Pipeline Depth} \times \text{Instructions Per Cycle}$$
For memory loads hitting HBM where $D \approx 400$ cycles, the requisite $W_{memory\_bound}$ demands maximum warp residency.

---

## 4. Further Reading

- [GPU_Architecture](GPU_Architecture.md) - For the SM physical layout.
- [Memory_Hierarchy_and_Roofline](Memory_Hierarchy_and_Roofline.md) - For the memory bandwidth math.

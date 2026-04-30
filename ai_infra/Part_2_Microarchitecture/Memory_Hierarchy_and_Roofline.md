# Memory Hierarchy and Roofline Models

A mathematically rigorous framework for diagnosing AI hardware performance constraints. The Roofline Model maps algorithmic Arithmetic Intensity against the strict physical ceilings of peak compute ($\pi$) and peak memory bandwidth ($\beta$). This dictates whether a kernel (e.g., Matrix Multiplication vs. LayerNorm) fundamentally requires more Tensor Cores or wider HBM channels.

**Prerequisites**: [GPU_Architecture](GPU_Architecture.md), [Accelerator_Landscape_2026](Accelerator_Landscape_2026.md).

---

## 1. The Roofline Performance Model

### 1.1 The Fundamental Equations
The theoretical attainable performance ($P$) of an algorithmic kernel is strictly bounded:
$$P = \min(\pi, I \times \beta)$$
Where:
- $\pi$: Peak mathematical throughput of the execution units (e.g., FLOPS/sec).
- $\beta$: Peak memory bandwidth of the memory tier being accessed (e.g., Bytes/sec).
- $I$: **Arithmetic Intensity** ($W / Q$), the ratio of total computational operations ($W$) to total data bytes transferred ($Q$) from the specified memory tier.

### 1.2 The Ridge Point ($I_{ridge}$)
The inflection point on the Roofline graph, where an algorithm transitions from memory-bound to compute-bound, is defined as:
$$I_{ridge} = \frac{\pi}{\beta} \quad \text{(FLOPs/Byte)}$$

**Mathematical Realities**:
- **Matrix Multiplication (Compute Bound)**: For an $M \times K$ by $K \times N$ dense matrix multiplication, $W = 2 M K N$ operations, and $Q \approx M K + K N + M N$ words transferred. The intensity $I = O(K)$. As dimensions scale, $I \gg I_{ridge}$, saturating the Tensor Cores.
- **Autoregressive Decode (Memory Bound)**: Generating a single token involves a matrix-vector product across the entire parameter footprint. For $N$ parameters, $W = 2N$ FLOPs, and $Q = N \times \text{bytes\_per\_param}$. The intensity is $I \approx \frac{2}{\text{bytes\_per\_param}}$. For FP16, $I = 1$ FLOP/Byte. Because $1 \ll I_{ridge}$ (e.g., $1125$ on B200 FP4), decode is catastrophically memory-bound.

---

## 2. Memory Tiers: HBM vs. SRAM

### 2.1 High Bandwidth Memory (HBM) Physical Limits
HBM is integrated directly onto the silicon interposer (e.g., CoWoS-L), circumventing the limited pin count of the PCIe boundary.
The theoretical bandwidth $\beta_{HBM}$ is:
$$\beta_{HBM} = \text{Stacks} \times \text{Channels/Stack} \times \text{Bus Width/Channel} \times \text{Data Rate}$$
For HBM3e on an NVIDIA B200:
- 8 Stacks
- 1024-bit total bus width per stack
- Data Rate: $\approx 8.0$ Gbps per pin.
$$\beta_{HBM} = 8 \times 1024 \text{ pins} \times 8.0 \text{ Gbps/pin} = 65,536 \text{ Gbps} = 8.192 \text{ TB/s}$$
*Note*: Actual attainable bandwidth is typically $\sim 80-90\%$ of theoretical due to DRAM refresh cycles (tRFC), bank conflicts, and controller overhead.

### 2.2 SRAM (L1, L2, TMEM)
SRAM provides latencies of $10-30$ cycles, contrasting with HBM's $\sim 400$ cycles.
- **L2 Cache**: A massive $\approx 50$ MB pool (Hopper) routing all HBM requests.
- **L1 / Shared Memory**: $256$ KB per SM. If an SM computes $16,384$ FLOPs per cycle, it requires multiple TB/s of *internal* SRAM bandwidth to feed the ALUs without stalling.
- **TMEM**: Blackwell's dedicated Tensor Memory decouples the high-bandwidth 4-bit/8-bit operand broadcast requirements of the Tensor Cores from general-purpose SMEM bank conflicts.

---

## 3. Optimizing for the Roofline

To shift a workload from the memory-bound slope to the compute-bound plateau:
1. **Reduce $Q$ (Quantization)**: Decreasing the precision from FP16 to FP4 exactly halves $Q$, instantly doubling $I$. This shifts the operating point rightwards on the Roofline.
2. **Operator Fusion**: Fusing an activation function (e.g., GeLU) directly into the epilogue of a Matrix Multiplication. Instead of $Q_{store}$ (write to HBM) and $Q_{load}$ (read from HBM for GeLU), the intermediate result is held entirely in registers. This eliminates HBM traffic, keeping the kernel constrained only by the SM computation logic.

---

## 4. Further Reading

- [GPU_Architecture](GPU_Architecture.md) - For the physical SMEM and register structure.
- [Quantization](../Part_5_Algorithms_and_Quantization/Quantization.md) - For how FP4 microscaling mathematically manipulates $I$.

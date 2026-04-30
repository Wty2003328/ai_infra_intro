# Huawei Ascend (910B / 910C / 910D)

A rigorous, graduate-level analysis of the Huawei Ascend AI accelerator family. Facing profound geopolitical and supply-chain constraints, the Ascend architecture (specifically the DaVinci core) has evolved to maximize compute density using older process nodes. Ascend represents the only fully vertically integrated, non-US ecosystem deployed at scale for frontier model training in 2026.

**Prerequisites**: [GPU_Architecture](GPU_Architecture.md), [Networking_and_Interconnect](../Part_3_Hardware_Systems_and_Interconnects/Networking_and_Interconnect.md), [Parallelism_Strategies](../Part_7_Distributed_Systems_and_Scale/Parallelism_Strategies.md).

---

## 1. The Ascend Generation Map

Due to Entity List restrictions, Huawei relies on SMIC for fabrication, limiting access to EUV (Extreme Ultraviolet) lithography. Consequently, the architecture relies heavily on advanced 2.5D/3D packaging to aggregate silicon area.

| Generation | Year | Process Node | Architecture | Memory | Compute Peak (Dense) | Scale-Up Topology |
|---|---|---|---|---|---|---|
| Ascend 910 | 2019 | TSMC 7nm+ | DaVinci v1 | 32 GB HBM2 | 256 TFLOPS FP16 | HCCS (Mesh) |
| Ascend 910B | 2023 | SMIC 7nm | DaVinci v2 | 64 GB HBM2e | ~300 TFLOPS FP16 | HCCS (8-way) |
| Ascend 910C | 2024/25 | SMIC 7nm | DaVinci v3 (Dual-Die) | 128 GB HBM3 | ~150 TFLOPS BF16 | HCCS / CM-384 |
| Ascend 910D | 2026 | SMIC 5nm class | DaVinci-Next (Multi-Die) | 256+ GB HBM3e | ~300+ PFLOPS FP8 | HCCS-Next |

---

## 2. Microarchitecture: The DaVinci Core

The fundamental compute unit is the **AI Core**, based on the DaVinci architecture.

### 2.1 The 3D Cube Matrix Engine
Unlike NVIDIA's 2D Tensor Cores or Google's large Systolic Arrays, the DaVinci architecture centers on a **3D Cube Compute Engine**.
- **Execution Math**: A standard DaVinci $16 \times 16 \times 16$ Cube performs 4096 MACs (8192 FLOPs) per cycle.
- **Data Routing**: To sustain this without exhausting SRAM bandwidth, the 3D Cube leverages broadcast data routing. Matrix $A$ is broadcast along one spatial axis, Matrix $B$ along another, and the partial sums accumulate along the third.
- **Power Efficiency**: The 3D Cube architecture mathematically minimizes data movement distance per FLOP. For older process nodes where wire capacitance ($C$) is high, reducing the routing distance $d$ linearly reduces dynamic power $P_{dyn} = \alpha C V^2 f$, allowing SMIC 7nm silicon to remain thermally viable.

### 2.2 Vector and Scalar Units
Adjacent to the Cube, the AI Core contains:
- **Vector Unit**: A highly parallel SIMD engine for element-wise ops (LayerNorm, SwiGLU).
- **Scalar Unit**: For loop unrolling and address generation.
- **Local Memory**: L0 (registers), L1 (private SRAM), and an L2 shared across all AI cores on the die.

---

## 3. Scale-Up Fabric: HCCS and CloudMatrix-384

Huawei's scale-up fabric is **HCCS (Huawei Cache Coherent System)**.

### 3.1 Intra-Node HCCS
Within a single 8-GPU node, Ascend 910C utilizes HCCS to form a fully connected mesh or high-radix ring.
- **Bandwidth**: The per-link signaling rate yields roughly 400 GB/s aggregate inter-chip bandwidth (comparable to NVLink-3).

### 3.2 Rack-Scale CloudMatrix-384 (CM-384)
To counter the NVL72, Huawei introduced the CloudMatrix architecture, scaling a single coherent domain to 384 chips.
- **Topology**: CM-384 utilizes a massive hierarchical ring and mesh topology, integrated heavily with optical transceivers.
- **Scale-Up Math**: If 384 chips must perform an All-to-All collective for MoE Expert routing, the total data volume $V$ transmitted across the network is $V = M \frac{N-1}{N}$, where $N=384$. Without the non-blocking Clos network of NVLink, the CM-384 mathematically requires sophisticated congestion control algorithms and intelligent MoE placement to avoid saturating bisection bandwidth.
- **Power Density**: At 384 chips, utilizing 7nm/5nm process technology, the thermal density is extreme. A single CM-384 deployment requires multi-megawatt facility provisioning, resulting in a significantly lower $PFLOPS/kW$ ratio compared to Blackwell.

---

## 4. Software Ecosystem: CANN and MindSpore

Because Huawei cannot rely on CUDA, it maintains a completely independent software stack.

### 4.1 CANN (Compute Architecture for Neural Networks)
CANN is the equivalent of the cuDNN/cuBLAS and PTX layers.
- It provides a Graph Engine (GE) for high-level graph optimization and TBE (Tensor Boost Engine) for kernel generation.
- **TBE DSL**: Developers author kernels using a Python-based DSL that resembles Triton, which the TBE compiler lowers to DaVinci's specific Cube/Vector/Scalar instructions.

### 4.2 MindIE and PyTorch Integration
To capture the OSS ecosystem, Huawei has rigorously optimized the `torch_npu` PyTorch backend. For inference, **MindIE** (MindSpore Inference Engine) serves as the vLLM/TensorRT-LLM equivalent, implementing PagedAttention and continuous batching specifically tuned for the DaVinci memory hierarchy.

---

## 5. Further Reading

- [GPU_Architecture](GPU_Architecture.md) - Contrast the 3D Cube with the SM.
- [Networking_and_Interconnect](../Part_3_Hardware_Systems_and_Interconnects/Networking_and_Interconnect.md) - Contrast HCCS with NVLink.
- [Cloud_ASICs](Cloud_ASICs.md) - Contrast with Google TPU and AWS Trainium.
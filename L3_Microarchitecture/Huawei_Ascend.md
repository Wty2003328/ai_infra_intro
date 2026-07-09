# Huawei Ascend — 910B / 910C / 910D

> **Layer:** L3.
> **Prerequisites:** [GPU_Architecture](GPU_Architecture.md), [L2 Systolic_Arrays_and_Dataflow](../L2_Digital_Design_for_AI/Systolic_Arrays_and_Dataflow.md).
> **Hands off to:** [L4 Networking & Interconnects](../L4_Systems_and_Interconnects/Index.md) (HCCS), [Accelerator_Landscape_2026](Accelerator_Landscape_2026.md).

---

## 0. The geopolitical context

Huawei is on the US Entity List → no access to TSMC EUV → Ascend is built on **SMIC's 7 nm-class node** (DUV multi-patterning). This is the *only* fully-vertically-integrated, non-US AI ecosystem deployed at frontier-model scale in 2026.

The architecture compensates for the older process node with:

1. **Aggressive 2.5D / 3D packaging** to aggregate silicon area.
2. **Lower per-FLOP frequency** (~1.2 GHz vs ~1.6 GHz on Blackwell) to keep within thermal/power budgets.
3. **3D Cube Matrix Engine** that minimizes intra-chip data-movement distance, reducing dynamic power $P = \alpha C V^2 f$.
4. **Wide HCCS interconnect** (similar to NVLink) for in-domain coherency.

Result: per-chip throughput ~1/3 of B200, but rack-scale CM-384 (384-chip mesh) delivers competitive aggregate inference throughput on Chinese-ecosystem deployments.

---

## 1. Generation map

| Gen | Year | Process | Architecture | Memory | BF16 dense (TFLOPS) | Topology | Domain |
|---|---|---|---|---|---|---|---|
| Ascend 910 | 2019 | TSMC 7nm+ | DaVinci v1 | 32 GB HBM2 | 256 | HCCS mesh | 8 |
| Ascend 910B | 2023 | SMIC 7nm | DaVinci v2 | 64 GB HBM2e | ~300 | HCCS | 8 |
| Ascend 910C | 2024–25 | SMIC 7nm dual-die | DaVinci v3 | 128 GB HBM3 | ~150 (per die, 300 dual) | HCCS / CM-384 | 384 |
| Ascend 910D | 2026 | SMIC 5nm-class multi-die | DaVinci-Next | 256+ GB HBM3e | ~300+ (FP8, est) | HCCS-Next | 384–1024 |

The Ascend 910C is the production part for 2024–2026; 910D is on the roadmap pending SMIC's 5 nm ramp.

---

## 2. The DaVinci AI Core

### 2.1 The 3D Cube Matrix Engine

Unlike NVIDIA's small-tile 2D tensor cores or TPU's large 2D MXU, DaVinci uses a **3D Cube** — a $16 \times 16 \times 16$ array of MAC cells:

```mermaid
%%{init: {"flowchart": {"defaultRenderer": "elk", "nodeSpacing": 60, "rankSpacing": 60, "htmlLabels": false}}}%%
flowchart TB
    subgraph CUBE["DaVinci 3D Cube · 16×16×16 = 4096 MACs · 8192 FLOPs/cycle FP16"]
        direction TB
        L0["Layer 0: 16×16 MAC plane"]:::layer
        L1["Layer 1: 16×16 MAC plane"]:::layer
        LD["… 14 more layers …"]:::layer
        L15["Layer 15: 16×16 MAC plane"]:::layer
        L0 --- L1 --- LD --- L15
    end
    A["Matrix A<br/>broadcast along X axis"]:::a
    B["Matrix B<br/>broadcast along Y axis"]:::b
    C["Partial sums accumulate along Z axis"]:::c
    A --> CUBE
    B --> CUBE
    CUBE --> C
    classDef layer fill:#fde68a,stroke:#b45309,color:#000
    classDef a fill:#bae6fd,stroke:#0369a1,color:#000
    classDef b fill:#bbf7d0,stroke:#15803d,color:#000
    classDef c fill:#fbcfe8,stroke:#9d174d,color:#000
```

Operation:

- $A$ broadcasts horizontally (X-axis): one row of A into all 16 columns per layer.
- $B$ broadcasts vertically (Y-axis): one column of B into all 16 rows per layer.
- Each layer's $16 \times 16 = 256$ MAC cells produce an $A_i B_j$ partial product.
- Partial sums accumulate along the Z-axis (16 layers stacked).

Total: $16 \times 16 \times 16 = 4\,096$ MACs per cycle = 8 192 FLOPs/cycle at FP16. At ~1.2 GHz: ~10 TFLOPS BF16 per Cube.

### 2.2 Why 3D, not 2D?

The 3D structure minimizes the Manhattan distance between operand SRAM and the MAC sites. For a 2D systolic array, operands travel up to $D$ wires (where $D$ is the array side). For a 3D cube, max distance is $D$ on any one axis but the *third* dimension reduces operand-fetch routing by half.

At SMIC 7 nm, wire capacitance is ~1.6× higher than TSMC N4 (longer wires due to lower density). Reducing routing distance by 30% directly cuts dynamic power $\alpha C V^2 f$ by ~30%, recovering the thermal budget needed to fit a useful chip.

### 2.3 AI Core composition

Each Ascend 910C die has multiple AI Cores. Per AI Core:

- 1 × Cube Engine (16×16×16).
- 1 × Vector Engine (SIMD ALU for elementwise ops).
- 1 × Scalar Engine (loop control, address generation).
- L1 buffer (~256 KB).
- L0 buffers feeding the Cube directly.

The 910C has 32 AI Cores per die × 2 dies = 64 AI Cores per package. Theoretical peak: 64 × ~10 TFLOPS = ~640 BF16 TFLOPS dense. **Spec-sheet value is ~300 TFLOPS.** The gap arises from: (a) the dual-die package cannot sustain ~1.2 GHz across all 64 Cubes simultaneously — thermal and power delivery constraints force clock-frequency reduction (~750–900 MHz typical sustained); (b) cross-die synchronization overhead on the SMIC 7nm interposer adds stalls that reduce achievable throughput; (c) the quoted ~300 TFLOPS reflects a realistic sustained envelope with a ~0.45–0.50 utilization factor, which is typical for multi-die designs on a constrained process node. Report the ~300 TFLOPS figure for comparisons; the ~640 TFLOPS is the architectural ceiling that real silicon cannot reach.

---

## 3. Memory hierarchy

| Tier | Capacity | BW (per package) | Latency |
|---|---|---|---|
| Cube registers | per-MAC | n/a | 1 cycle |
| L0 buffer | per-Cube ~32 KB | high | ~2 cycles |
| L1 buffer | per-AI Core ~256 KB | ~3 TB/s | ~10 cycles |
| L2 buffer | per-die ~64 MB | ~5 TB/s | ~50 cycles |
| HBM | 128 GB HBM3 | 3.2 TB/s | ~400 cycles |

L2 buffer is *huge* (64 MB per die, 128 MB per dual-die package) — much larger than B200's 50 MB L2. This compensates for HBM3 bandwidth that's ~½ of HBM3e.

---

## 4. CANN software stack

CANN (Compute Architecture for Neural Networks) is Huawei's CUDA-equivalent. Layers:

| CANN component | NVIDIA equivalent |
|---|---|
| ACL (Ascend Computing Language) | CUDA Runtime |
| CCE (CANN Custom Engine) compiler | nvcc |
| HCCL (Huawei Collective Comm Lib) | NCCL |
| MindSpore | PyTorch (deep integration) |
| MindIE | TensorRT |
| TBE (Tensor Boost Engine) | cuDNN |
| AscendC kernel DSL | CUDA C / Triton |

CANN is the most mature non-CUDA stack outside ROCm. PyTorch on Ascend works via the `torch_npu` plugin; loss is ~10–25% vs same-precision CUDA at the kernel level (compiler is less optimized but improving rapidly).

---

## 5. HCCS and CM-384 fabric

### 5.1 HCCS (Huawei Cache Coherent System)

Per-chip: 7 HCCS links × ~56 GB/s/link = ~392 GB/s/chip. Domain: 8 chips per server (similar to xGMI on AMD).

### 5.2 CM-384 cluster mesh

To scale beyond 8, Huawei built **CM-384** — a 384-chip rack-scale mesh fabric. Topology:

- 48 servers × 8 chips = 384 chips.
- Each server has internal HCCS (8-chip coherent).
- Servers connect via a **2-tier optical fabric** for inter-server traffic.
- Aggregate domain: 384 chips treated as one logical training cluster.

```mermaid
%%{init: {"flowchart": {"defaultRenderer": "elk", "nodeSpacing": 60, "rankSpacing": 60, "htmlLabels": false}}}%%
flowchart TB
    subgraph CM384[CM-384 mesh — 384 Ascend 910C]
        direction TB
        subgraph S0[Server 0 — 8 chips]
            G0[Chip 0]:::g
            G1[Chip 1]:::g
            GD0[…]:::g
            G7[Chip 7]:::g
        end
        subgraph SD[… 47 more servers …]
            X[…]:::g
        end
        subgraph S47[Server 47 — 8 chips]
            G376[Chip 376]:::g
            G377[Chip 377]:::g
            GD47[…]:::g
            G383[Chip 383]:::g
        end
        OF[Optical inter-server fabric<br/>2-tier]:::opt
        S0 <--> OF
        SD <--> OF
        S47 <--> OF
    end
    classDef g fill:#fde68a,stroke:#b45309,color:#000
    classDef opt fill:#bae6fd,stroke:#0369a1,color:#000
```

Aggregate CM-384 capacity:

- ~120 PFLOPS BF16 dense (384 × ~300 / 1 000)
- ~50 TB HBM3 (384 × 128 GB / 1 000)
- Sufficient to host a Llama-3-405B-class model in BF16 with TP/PP across the mesh.

### 5.3 Comparison with NVL72 / Helios

| | CM-384 | NVL72 (B200) | Helios (MI455X) |
|---|---|---|---|
| Chip count | 384 | 72 | 72 |
| Per-chip BF16 TFLOPS | ~300 | ~2 250 | ~5 050 |
| Aggregate BF16 PFLOPS | ~120 | ~162 | ~360 |
| Per-chip HBM | 128 GB HBM3 | 192 GB HBM3e | 432 GB HBM4 |
| Aggregate HBM | ~50 TB | ~14 TB | ~31 TB |
| Domain coherency | partial (server-level fully, mesh logically) | fully coherent NVL | fully coherent UALink |

CM-384 wins on aggregate HBM capacity (largest in 2026), loses on per-chip throughput. The bet: large memory + many chips compensates for slower per-chip silicon.

---

## 6. End-to-end cause / effect

```mermaid
%%{init: {"flowchart": {"defaultRenderer": "elk", "nodeSpacing": 60, "rankSpacing": 60, "htmlLabels": false}}}%%
flowchart TD
    A[SMIC 7nm DUV process] --> B[Lower density vs TSMC N4]
    B --> C[Wire capacitance ~1.6× higher]
    C --> D[Need to minimize routing distance<br/>per FLOP]
    D --> E[3D Cube architecture]

    F[3D Cube 16×16×16 = 4 096 MACs] --> G[~10 TFLOPS BF16 per Cube]
    G --> H[~640 TFLOPS per package · ~3× lower<br/>than B200]

    I[64 MB L2 per die] --> J[Compensate for HBM3 vs HBM3e BW]

    K[CM-384 scale-up] --> L[384 chips · ~120 PFLOPS aggregate]
    L --> M[Competitive at rack scale despite<br/>per-chip gap]

    N[CANN + MindSpore stack] --> O[~10–25% perf gap vs CUDA]

    H & M --> P[Ascend dominates Chinese AI<br/>deployments through 2026]
```

---

## 7. Numbers to memorize

| Quantity | Value | Why |
|---|---|---|
| Ascend 910C process | SMIC 7nm | Geopolitical |
| 910C die | dual-die (2 dies) | Reticle workaround |
| 910C HBM | 128 GB HBM3 | spec |
| 910C HBM BW | 3.2 TB/s | spec |
| 910C BF16 dense | ~300 TFLOPS (per package) | spec |
| AI Cores per 910C package | 64 (32 per die) | structure |
| DaVinci 3D Cube | 16×16×16 = 4 096 MACs | per Cube |
| Per-Cube throughput | ~10 TFLOPS BF16 at 1.2 GHz | derived |
| L2 buffer per die | ~64 MB | larger than B200 L2 |
| HCCS per chip | ~392 GB/s | 7 links × 56 GB/s |
| HCCS coherent domain | 8 chips | per server |
| CM-384 chip count | 384 | rack scale |
| CM-384 aggregate BF16 | ~120 PFLOPS | spec |
| CM-384 aggregate HBM | ~50 TB | largest in 2026 |
| Ascend 910D HBM | 256 GB HBM3e (proj) | future |

---

## 8. References

- Liao, Bai et al., *Ascend Native AI Software Stack*, Hot Chips 2023.
- *Ascend 910 Architecture Overview*, Huawei white papers (2019–2024).
- *MindSpore Documentation*, Huawei.
- SMIC technology disclosures (limited; mostly inferred from Ascend benchmarks).

---

**Up the stack:** [L4 Networking & Interconnects](../L4_Systems_and_Interconnects/Index.md), [Accelerator_Landscape_2026](Accelerator_Landscape_2026.md).
**Down the stack:** [GPU_Architecture](GPU_Architecture.md), [L2 Systolic_Arrays_and_Dataflow](../L2_Digital_Design_for_AI/Systolic_Arrays_and_Dataflow.md).

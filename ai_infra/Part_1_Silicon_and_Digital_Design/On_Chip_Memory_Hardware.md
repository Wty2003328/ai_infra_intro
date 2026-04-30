# On-Chip Memory Hardware — SRAM, Register File, TMEM, LDS

The L1 deep-dive on the microscopic memory cells that encapsulate the tensor core. Tensor-core arithmetic throughput is ultimately dictated by the Operand Delivery Bandwidth—a constraint bounded by the physics of SRAM bitcells and capacitive bitline charging. This chapter provides a rigorous mathematical and structural analysis of the local memory tiers (SRAM, Register Files, and TMEM) that constitute the immediate staging ground for AI computation.

**Layer**: L1.
**Prerequisites**: [Digital_Design_For_AI](Digital_Design_For_AI.md), [FP_Unit_Design](FP_Unit_Design.md). Familiarity with CMOS transistor sizing and parasitic capacitance. [HBM_Deep_Dive](../Part_1_Silicon_and_Digital_Design/HBM_Deep_Dive.md) covers the off-chip memory domain.

---

## 1. The Bandwidth Cliff and the Physical Memory Wall

The defining crisis of 2026 AI architecture is the sheer volume of operand data required to keep a dense MAC array saturated.
Consider a theoretical Blackwell B300 SM delivering $10.8 \text{ PFLOPS}$ at FP4. At $1.6 \text{ GHz}$, this requires completing $\sim 6,750$ operations per clock cycle. Each FMA operation ($C \leftarrow A \cdot B + C$) necessitates:
- Reading operand $A$ (1 byte at FP8/FP4 mixed precision).
- Reading operand $B$ (1 byte).
- Reading accumulator $C$ (4 bytes, FP32).
- Writing accumulator $C$ (4 bytes, FP32).

Total intrinsic datapath bandwidth demand: $\sim 67,500 \text{ bytes/cycle}$. Multiplied by $1.6 \text{ GHz} \approx 108 \text{ TB/s}$ per SM.

If the architecture cannot supply this from immediate local storage, it hits the **Bandwidth Cliff**, stalling the MAC pipeline and plummeting Model FLOP Utilization (MFU).

| Architectural Tier | Bandwidth Capacity (Per SM) | Typical Capacity | Access Latency | Physical Implementation |
|---|---|---|---|---|
| **Register File (RF)** | $\sim 100+ \text{ TB/s}$ | $256 \text{ KB}$ | 1 cycle | Highly multi-ported custom SRAM |
| **TMEM (Blackwell)** | $\sim 50-80 \text{ TB/s}$ | $256 \text{ KB}$ | $2-4$ cycles | Multi-banked, wide-read 6T SRAM |
| **SMEM / LDS** | $\sim 30 \text{ TB/s}$ | $128-256 \text{ KB}$ | $8-20$ cycles | 32/64-bank word-interleaved SRAM |
| **L2 Cache** | $\sim 10 \text{ TB/s}$ | $50 \text{ MB}$ (Chip wide) | $30-80$ cycles | High-density 6T SRAM blocks |
| **HBM (Off-Chip)** | $\sim 8 \text{ TB/s}$ (Chip wide) | $288 \text{ GB}$ | $300-500$ cycles | Stacked 1T1C DRAM cells |

Feeding the MAC array requires operands to reside exclusively in the Register File or TMEM. Everything lower in the hierarchy exists merely to stage data upward.

---

## 2. The Foundation: The 6T-SRAM Cell Mathematics

The fundamental atomic unit of on-chip AI memory is the 6-Transistor (6T) SRAM cell. 

```text
   WL ────┬───────────────┬────
          │               │
       M5│              M6│       ← Access Transistors (Pass Gates)
   ┌───┴────┐         ┌────┴───┐
   │ ┌────┐ │         │ ┌────┐ │
   │ │M2  │ │   ↔    │ │M4  │ │   ← Pull-Up PMOS (Cross-coupled)
   │ ┕━━━━┘ │         │ ┕━━━━┘ │
   │ ┌────┐ │         │ ┌────┐ │
   │ │M1  │ │   ↔    │ │M3  │ │   ← Pull-Down NMOS (Cross-coupled)
   │ ┕━━━━┘ │         │ ┕━━━━┘ │
   └─────┬──┘         └────┬───┘
        BL                 BLB
```

### 2.1 The Physics of Reading and Writing

The 6T cell relies on a precarious balance of transistor drive strengths ($\beta$-ratios).

**Read Stability (Static Noise Margin - SNM):**
During a read, Bitline (BL) is precharged to $V_{DD}$. When Wordline (WL) asserts, access transistor M5 activates, and current flows from BL through M5 and pull-down M1 to Ground. 
By voltage division, the internal node voltage $V_{internal}$ rises slightly above Ground. 
If $V_{internal}$ rises above the threshold voltage ($V_{th}$) of the opposite inverter (M3/M4), the cell will accidentally flip its state (a catastrophic destructive read).
*Mathematical Constraint:* M1 must be physically wider and exponentially more conductive than M5. The Cell Ratio ($CR = \beta_{M1} / \beta_{M5}$) must strictly be $\ge 1.2$ to $1.5$.

**Writeability (Write Margin):**
During a write, the external circuitry drives BL to $0\text{V}$. M5 must overpower the pull-up PMOS M2 to drag the internal node below the switching threshold of the cross-coupled pair.
*Mathematical Constraint:* M5 must be significantly stronger than M2. The Pull-Up Ratio ($PR = \beta_{M2} / \beta_{M5}$) must strictly be $< 1.0$.

### 2.2 The Scaling Wall (N3 and Beyond)

At the TSMC N3E node, the physical gate length is so constrained that fine-tuning the $W/L$ ratios of M1, M2, and M5 to satisfy both Read SNM and Writeability is extraordinarily difficult due to random dopant fluctuation and Line Edge Roughness. 
Consequently, high-density SRAM cell area scaling has effectively flatlined (shrinking by barely $10-15\%$ from N5 to N3). This physical stall is precisely why GPU L1/SMEM capacities have remained locked at $\sim 256 \text{ KB}$ for three architecture generations.

---

## 3. Memory Architecture: Banking and Conflict Mathematics

A monolithic SRAM array suffers from massive bitline capacitance ($C_{BL}$), rendering the $RC$ delay intolerable for gigahertz operation. To achieve AI-class bandwidth, the macro is aggressively partitioned into **Banks**.

### 3.1 The Geometry of Banking

A $256 \text{ KB}$ Shared Memory (SMEM) is not a single block; it is typically partitioned into 32 banks of $8 \text{ KB}$ each. 
If each bank independently possesses its own sense amplifiers and column decoders, the array can theoretically serve 32 independent memory requests per clock cycle, boosting structural bandwidth by $32\times$.

### 3.2 Bank Conflict Probabilities

A bank conflict occurs when $N$ parallel threads in a warp attempt to access differing addresses that mathematically alias to the exact same physical bank. The hardware cannot service multiple accesses to the same 1R/1W bank in a single cycle, forcing the memory controller to serialize the requests, inflicting a massive cycle penalty.

In CUDA, SMEM utilizes a 32-bank, 4-byte (word) interleaved topology.
The bank index for a given byte address is:
$$ \text{Bank\_Index} = \left\lfloor \frac{\text{Address}}{4} \right\rfloor \pmod{32} $$

**Conflict Scenarios:**
- **Stride-1 Access:** Thread $i$ accesses Address $4 \times i$. 
  Bank Index $= (i) \pmod{32}$. Since $i \in [0, 31]$, every thread hits a unique bank. Zero conflict.
- **Stride-2 Access:** Thread $i$ accesses Address $8 \times i$.
  Bank Index $= (2i) \pmod{32}$. Threads $0$ and $16$ both hit Bank 0. This produces a $2$-way bank conflict, halving effective memory bandwidth.

Triton and CUTLASS autotuners spend massive compilation resources injecting precise padding bytes into SMEM arrays to alter the arithmetic modulo, mathematically guaranteeing conflict-free tile loading.

---

## 4. Register File Architecture: The Ultimate Bandwidth Tier

The Register File (RF) is the most intensively accessed logic structure in the accelerator.

### 4.1 Multi-Port Physics

To supply a warp scheduler, the RF must routinely supply 4 to 8 operands per cycle while absorbing 2 to 4 writebacks.
A true 8R/4W SRAM cell requires 12 independent wordlines and 24 bitlines per cell. The parasitic wire capacitance ($C_{wire}$) of crossing dozens of metal traces over a single bitcell expands the physical cell area geometrically ($\mathcal{O}(Ports^2)$). A true multi-port RF is too large and slow for modern AI.

### 4.2 The Operand Collector Abstraction

Instead of building massive cells, modern GPUs build the RF identically to SMEM: 
- $\sim 32$ deeply isolated banks.
- Each bank is constructed using compact 1R/1W or 2R/1W dual-port cells.
- **The Operand Collector:** A dedicated hardware buffer sitting between the instruction decode and the ALUs. It analyzes the register addresses required by the next instruction. If the registers reside in different banks, it fetches them simultaneously. If they alias to the same bank, the Operand Collector sequences the reads across consecutive clock cycles, buffering the successful reads until the full operand vector is assembled, effectively hiding bank conflict latency through deep pipelining.

---

## 5. TMEM: The Blackwell Hardware Paradigm

The most significant memory hierarchy evolution in 2026 AI architecture is NVIDIA's **Tensor Memory (TMEM)**.

### 5.1 The Mathematical Necessity of TMEM

As established in §1, FP4 transitions expanded MAC operand bandwidth demands to $\sim 50+ \text{ TB/s}$ per SM.
Standard SMEM is structurally capped. Adding more read ports to the SMEM SRAM macros would destroy the storage density. Increasing the bank count beyond 32 incurs devastating area overhead in the crossbar routing logic.

**The Bifurcation Solution:** When bandwidth cannot scale vertically, architects bifurcate the hierarchy horizontally.
TMEM is a completely independent $256 \text{ KB}$ SRAM block provisioned strictly for the tensor cores.
- It utilizes specialized ultra-wide read interfaces that directly match the geometric dimension of the `wgmma` multiplier array.
- It is decoupled from the asynchronous copy engines (TMA) and legacy CUDA thread loads, preventing the MAC array from experiencing jitter or bank-conflict stalling induced by overlapping operations.

Without TMEM, the B200's $9000 \text{ TFLOPS}$ capability would be stranded behind an insurmountable SMEM port-contention wall. AMD compensates for lacking TMEM on CDNA 4 by aggressively expanding LDS capacity to $\sim 128 \text{ KB}$ and relying on compiler-enforced double-buffering.

---

## 6. The L2 Cache and Coherency

The L2 Cache ($~50 \text{ MB}$ on Blackwell, $~64 \text{ MB}$ Infinity Cache on MI355X) serves as the centralized cross-SM communication hub.

### 6.1 Structural Parameters

- Constructed from high-density 6T SRAM variants (sacrificing speed for extreme density).
- Divided into $32-64$ slices geographically distributed around the die, interfaced via the primary Network-on-Chip (NoC) crossbar.
- Employs Pseudo-LRU replacement policies.

### 6.2 The AI Workload Reality

In legacy graphics, L2 acts to capture spatial locality. In Large Language Models, the active working set (a $70\text{GB}$ parameter matrix and a $100\text{k}$ token KV cache) exceeds the $50\text{MB}$ L2 capacity by 3 orders of magnitude. 
Consequently, L2 acts strictly as a **transient absorption buffer**. It catches partial sums during massive reduction operations (e.g., FlashAttention sequence accumulation) and provides an ultra-low latency staging tier for hot activations during the forward pass, ensuring they survive to be re-read during the immediate backward pass window before being evicted.

---

## 7. High-Speed I/O Memory Interfaces (SerDes)

The interfaces driving data off-chip to HBM and NVLink domains represent the pinnacle of analog mixed-signal memory logic.

- **HBM PHY:** Driving 2048 single-ended channels at $10 \text{ Gbps}$ over $5 \text{ mm}$ of silicon interposer. Requires rigorous active Per-Pin Write Leveling—sweeping a digital delay line to precisely center the latching clock edge within a sub-$50 \text{ ps}$ data eye window.
- **NVLink/UALink PHY:** Driving 224 Gbps utilizing PAM4 (Pulse Amplitude Modulation 4-level) differential signaling. At these frequencies, the copper traces act as violent low-pass filters. The RTL must implement heavy Continuous-Time Linear Equalization (CTLE) and Decision Feedback Equalization (DFE) DSP blocks to reconstruct the decimated signal prior to data latching.

---

## 8. Common Architectural Interview Inquiries

**Q: Explain mathematically why a 6T SRAM cell cannot be indiscriminately shrunken to match standard logic gate scaling.**
A: The 6T cell's functionality relies strictly on the current drive ratios ($\beta$-ratios) between the Pull-Down, Access, and Pull-Up transistors to maintain the Static Noise Margin (SNM) during reads and permit successful writes. At extreme nodes (N3/N2), shrinking the physical width of the fin or nanosheet introduces high variance in dopant concentration. If a read transistor is randomly weak, the SNM collapses, and reading the cell destroys the stored data. To guarantee yield across billions of cells, the transistors must be kept physically larger than standard logic devices, stalling SRAM density scaling.

**Q: How does the Operand Collector prevent structural deadlocks in the Register File?**
A: If multiple instructions require registers housed in the same physical bank, the 1R/1W bank can only serve one per cycle. The Operand Collector contains multiple independent queue slots. It analyzes incoming instructions, arbitrates bank access, and buffers successfully read operands. If an instruction is stalled waiting for a conflicted bank, the collector allows independent, non-conflicting instructions in the warp scheduler to bypass it and access free banks, maintaining continuous pipeline saturation.

**Q: Why was TMEM architected specifically for Blackwell, rather than just doubling the SMEM size?**
A: Doubling SMEM size to $512 \text{ KB}$ increases capacity but fundamentally does not increase the available Read/Write ports per bank. The tensor cores transitioned to FP4, effectively quadrupling their data consumption rate. They require massive, simultaneous, conflict-free parallel read bandwidth. TMEM isolates the tensor core traffic into a dedicated, physically wide-ported SRAM macro, ensuring that TMA asynchronous copies feeding SMEM from HBM do not probabilistically collide with MAC operand fetches.

**Q: In CUDA, why does a `float` array access with stride 32 produce zero bank conflicts, but a stride of 16 produces a 2-way conflict?**
A: SMEM contains 32 banks, word-interleaved. The mapping is $(\text{Word\_Address}) \pmod{32}$. 
If 32 threads access `A[i * 32]`, their addresses map to $(i \times 32) \pmod{32} = 0$. All 32 threads simultaneously request data from Bank 0. However, the hardware supports a broadcast mechanism; if all threads request the exact same address, Bank 0 reads it once and broadcasts it to all threads in 1 cycle. (Zero conflicts).
If 32 threads access `A[i * 16]`, threads $0$ and $2$ request addresses mapping to $(0) \pmod{32} = 0$ and $(32) \pmod{32} = 0$, respectively. These are *different* addresses within the identical Bank 0. The bank must serially serve thread 0, then thread 2, creating a 2-way conflict and halving bandwidth.

---

## 9. Further Academic and Engineering Reading

**Foundational Semiconductor Texts:**
- Weste, Neil H. E., and Harris, David Money. *CMOS VLSI Design: A Circuits and Systems Perspective.* (Essential math for SRAM SNM and $\beta$-ratios).
- Rabaey, Jan M. *Digital Integrated Circuits.*

**GPU Architecture and Tuning:**
- "Programming Massively Parallel Processors" (Kirk & Hwu). Crucial detailing of SMEM banking algorithms.
- "Dissecting the NVIDIA Hopper Architecture" (Community Reverse Engineering Papers).
- Triton compiler source code and documentation detailing autotuned block padding algorithms.

**Cross-Referenced System Documentation:**
- [../../digital_design/Memory](../../digital_design/Memory.md) — Comprehensive logic memory primitives.
- [../../digital_design/Physical_Design](../../digital_design/Physical_Design.md) — Floorplanning and SRAM macro placement.

---

**Up the stack:** [GPU_Architecture](../Part_2_Microarchitecture/GPU_Architecture.md), [CUDA_Optimization](../Part_4_GPU_Kernel_Engineering/CUDA_Optimization.md).
**Down the stack:** [HBM_Deep_Dive](../Part_1_Silicon_and_Digital_Design/HBM_Deep_Dive.md).
**See also:** [Digital_Design_For_AI](Digital_Design_For_AI.md), [FP_Unit_Design](FP_Unit_Design.md).

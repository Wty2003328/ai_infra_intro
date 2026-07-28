# L4 — Systems & Interconnects

> The layer where L3 chips become **nodes, racks, and clusters**. A single GPU is a roofline; a rack of 72 GPUs connected through NVSwitch is a system. The interconnect fabric — signaling physics, topology mathematics, congestion control, storage fabric — determines whether a trillion-parameter model trains in days or stalls on stragglers. L4 is the bridge between silicon physics (L0–L1), microarchitecture (L2–L3), and the software stack above (L5–L8).

---

## Scope of this layer

L4 covers the **rack-scale and cluster-scale interconnect** and the **storage subsystem** that feeds it.

1. **Networking and interconnect** — the complete inter-chip path: remote mapping and endpoint transaction state; package-parallel and SerDes PHYs; FEC/CRC/replay/credits; UCIe, PCIe, CXL, NVLink/Infinity-class and UALink semantic boundaries; switches, bisection, collectives; then InfiniBand/RoCE/Ultra Ethernet scale-out and congestion control.
2. **Rack-scale design** — the system architecture of a single rack: GB200 NVL72, Helios UALink, TPU v7 pod, Cerebras CS-3. Power distribution (48 V busbars, VRM), thermal (sensible heat, two-phase, CDU), mechanical (floor loading, cabling density).
3. **Storage and model loading** — how trillion-parameter checkpoints move from NVMe to HBM: GPUDirect Storage, safetensors, sharded checkpoints, NVMe-oF, KV cache offload tiers.

L4 covers the **system contract** of chip-to-chip links and fabrics. Detailed GPU port counts or vendor core placement remain in L3, gate/PHY/router implementation is introduced in L2, kernel communication primitives remain in L5, and collective algorithms remain in L7.

---

## Pages (read in order)

1. [Networking_and_Interconnect](01_Networking_and_Interconnect.md) — from a GPU/CPU transaction through remote mapping, endpoint/link/PHY/switch hardware and memory semantics, then into topology, congestion control, RDMA, and collective offload. Foundational for understanding everything above.
2. [Rack_Scale_Design](02_Rack_Scale_Design.md) — what a single rack looks like: NVL72, Helios, TPU pod, Cerebras. Power, thermal, mechanical constraints that make the topology real.
3. [Storage_and_Model_Loading](03_Storage_and_Model_Loading.md) — the storage fabric: how checkpoints load in seconds, not minutes. GDS, safetensors, sharded formats, NVMe-oF.

---

## What L4 hands off to L5 and above

| L4 fact | Higher-layer consequence |
|---|---|
| 224 Gbps PAM4 Nyquist = 56 GHz; passive copper ≤ 1 m | Scale-up domain is rack-bound; scale-out requires optics (L7, L8) |
| Endpoint port sum is not fabric bisection; link ACK is not remote visibility | collective placement and synchronization must use the actual route/semantic contract (L7) |
| UALink 1.0 uses software coherence across accelerator caches | kernel/runtime phase boundaries must flush, signal, acquire, and invalidate as required |
| CXL separates `.io`, `.cache`, and `.mem` roles | pooled/tiered memory is topology- and home-agent-aware, not a flat DRAM extension |
| 3D torus bisection BW ∝ N^{2/3} vs Clos ∝ N | Google must use OCS + careful scheduling to avoid hotspots (L7) |
| DCQCN rate halving on ECN mark | AllReduce bandwidth must account for 30–50% congestion backoff (L7) |
| Incast: N−1→1 bursts overflow 128 MB ToR buffers | Packet spraying + large static buffers required for All-to-All (L7) |
| 48 V distribution reduces I²R loss 16× vs 12 V | Rack density can reach 120–140 kW without copper busbar meltdown (L8) |
| 120 kW rack requires ~45 GPM liquid flow | Data center design: CDU sizing, dry cooler capacity, water-side economics |
| GDS bypasses CPU at ~63 GB/s per GPU | Checkpoint save/restore fits in seconds, not minutes (L7) |
| Safetensors zero-copy mmap | Model loading latency is I/O-bound, not deserialization-bound (L8) |
| NVMe-oF delivers 600+ GB/s aggregate | Trillion-param checkpoint < 10 s save time (L7) |

---

## Reading order

Top-down for orientation: read this Index → skim [Networking_and_Interconnect](01_Networking_and_Interconnect.md) sections 1–3 for the taxonomy → jump to [Rack_Scale_Design](02_Rack_Scale_Design.md) for concrete system pictures → [Storage_and_Model_Loading](03_Storage_and_Model_Loading.md) for the I/O story.

Strict bottom-up for depth: Networking → Rack → Storage, in order.

After this layer, ascend to [L5 — Kernels & Programming](../L5_Kernels_and_Programming/00_Index.md) where the hardware becomes software.


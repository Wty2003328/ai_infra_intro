# L4 — Systems & Interconnects

> The layer where L3 chips become **nodes, racks, and clusters**. A single GPU is a roofline; a rack of 72 GPUs connected through NVSwitch is a system. The interconnect fabric — signaling physics, topology mathematics, congestion control, storage fabric — determines whether a trillion-parameter model trains in days or stalls on stragglers. L4 is the bridge between silicon physics (L0–L1), microarchitecture (L2–L3), and the software stack above (L5–L8).

---

## Scope of this layer

L4 covers the **rack-scale and cluster-scale interconnect** and the **storage subsystem** that feeds it.

1. **Networking and interconnect** — the physical layer (NRZ, PAM4, SerDes generations), the protocol layer (PCIe, CXL, NVLink, UALink, InfiniBand, RoCE v2), the topology layer (fat-tree, Clos, dragonfly, torus, dragonfly-plus), and the congestion-control layer (PFC, ECN, DCQCN, packet spraying, Ultra Ethernet).
2. **Rack-scale design** — the system architecture of a single rack: GB200 NVL72, Helios UALink, TPU v7 pod, Cerebras CS-3. Power distribution (48 V busbars, VRM), thermal (sensible heat, two-phase, CDU), mechanical (floor loading, cabling density).
3. **Storage and model loading** — how trillion-parameter checkpoints move from NVMe to HBM: GPUDirect Storage, safetensors, sharded checkpoints, NVMe-oF, KV cache offload tiers.

L4 explicitly does **not** cover chip-level interconnects like NVLink lane counts or ICI link widths (→ L3), kernel-level communication primitives (→ L5), or distributed-training collective algorithms (→ L7).

---

## Pages (read in order)

1. [Networking_and_Interconnect](Networking_and_Interconnect.md) — the physics and mathematics of the network fabric. Signaling, topologies, congestion control, RDMA. Foundational for understanding everything above.
2. [Rack_Scale_Design](Rack_Scale_Design.md) — what a single rack looks like: NVL72, Helios, TPU pod, Cerebras. Power, thermal, mechanical constraints that make the topology real.
3. [Storage_and_Model_Loading](Storage_and_Model_Loading.md) — the storage fabric: how checkpoints load in seconds, not minutes. GDS, safetensors, sharded formats, NVMe-oF.

---

## What L4 hands off to L5 and above

| L4 fact | Higher-layer consequence |
|---|---|
| 224 Gbps PAM4 Nyquist = 56 GHz; passive copper ≤ 1 m | Scale-up domain is rack-bound; scale-out requires optics (L7, L8) |
| NVL72 is strictly non-blocking at 130 TB/s aggregate | TP ≤ 72 without leaving the rack; no cross-rack TP traffic (L7) |
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

Top-down for orientation: read this Index → skim [Networking_and_Interconnect](Networking_and_Interconnect.md) sections 1–3 for the taxonomy → jump to [Rack_Scale_Design](Rack_Scale_Design.md) for concrete system pictures → [Storage_and_Model_Loading](Storage_and_Model_Loading.md) for the I/O story.

Strict bottom-up for depth: Networking → Rack → Storage, in order.

After this layer, ascend to [L5 — Kernels & Programming](../L5_Kernels_and_Programming/Index.md) where the hardware becomes software.

Cross-domain references:
- [`digital_design/SerDes_Design.md`](../../digital_design/SerDes_Design.md) — transistor-level SerDes and CDR circuits.
- [`power/Power_Distribution.md`](../../power/Power_Distribution.md) — VRM design and multiphase buck converters.
- [`systemverilog/PCIe_Controller.md`](../../systemverilog/PCIe_Controller.md) — RTL for the PCIe/CXL transaction layer.

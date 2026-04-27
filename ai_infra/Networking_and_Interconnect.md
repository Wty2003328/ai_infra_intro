# Networking and Interconnect

LLM serving and training are distributed workloads. Every collective operation — AllReduce in TP, point-to-point sends in PP, scatter/gather in EP — lives on the interconnect. Understanding NVLink, InfiniBand, and RDMA is the difference between a system that scales and one that stalls.

---

## 1. The Interconnect Hierarchy

GPUs talk to each other at three distinct levels, each with its own bandwidth, latency, and topology:

```
Intra-SM  (shared memory, registers)  — nanoseconds, TB/s
   │
Intra-GPU (HBM, L2 cache)              — ~100 ns, 3-12 TB/s
   │
Intra-Node (NVLink + NVSwitch)         — ~1 μs, 0.9-1.8 TB/s per GPU
   │
Intra-Rack (NVL72, copper)             — ~2-5 μs, 1.8 TB/s per GPU
   │
Inter-Node (InfiniBand / RoCE)         — ~2-10 μs, 400-800 Gbps per GPU
   │
Inter-Datacenter (dedicated fiber)     — 1-10 ms, 100-400 Gbps
```

**Rule of thumb for architecting distribution:**
- Put bandwidth-heavy collectives (tensor parallel AllReduce) on NVLink — **intra-node only** (or intra-NVL72).
- Put latency-tolerant collectives (pipeline parallel point-to-point) on InfiniBand — **cross-node**.
- Keep high-frequency small messages (scheduler signals) on the host network; keep high-bandwidth streams (KV transfer) on the RDMA fabric.

---

## 2. NVLink — Intra-Node GPU-to-GPU

NVLink is NVIDIA's proprietary GPU-to-GPU interconnect, designed to bypass PCIe. PCIe Gen5 x16 gives 64 GB/s per direction; NVLink gives 10-20× that. It's not routed through the CPU.

### 2.1 Generations

| Gen | GPU | Per-link BW | Links/GPU | Total per GPU | Year |
|---|---|---|---|---|---|
| NVLink 1 | P100 | 20 GB/s | 4 | 80 GB/s | 2016 |
| NVLink 2 | V100 | 25 GB/s | 6 | 150 GB/s → 300 GB/s bi-dir | 2017 |
| NVLink 3 | A100 | 25 GB/s | 12 | 300 GB/s → 600 GB/s bi-dir | 2020 |
| NVLink 4 | H100 | 25 GB/s | 18 | 450 GB/s → 900 GB/s bi-dir | 2022 |
| NVLink 5 | B200 | 50 GB/s | 18 | 900 GB/s → 1800 GB/s bi-dir | 2024 |

The quoted "900 GB/s" or "1.8 TB/s" is typically the **bidirectional** total across all links. Uni-directional is half. When the NCCL docs say "send BW", they mean uni-directional.

### 2.2 Electrical

NVLink is a point-to-point differential SerDes link using PAM-4 signaling. On H100 it runs at 200 Gbps per lane, 8 lanes per link. The physical medium is copper traces on the PCB (intra-node) or copper cables (within an NVL72 tray).

### 2.3 NVLink over fiber (NVL72)

In Blackwell's NVL72 systems (2024-2025), NVLink extends to 72 GPUs via a rack-scale NVSwitch fabric built from copper cable trays. The fabric latency is ~1 μs between any two GPUs in the rack, and bandwidth remains 1.8 TB/s per GPU.

For TP=72 (which emerging trillion-parameter models require), this is a game-changer: prior-gen NVLink topped out at 8 GPUs per NVLink domain, forcing you to cross InfiniBand for anything larger.

---

## 3. NVSwitch

NVSwitch is a crossbar switch that creates all-to-all NVLink connectivity. Without NVSwitch, NVLink is point-to-point — 8 GPUs would need a crossbar of 28 NVLink pairs, consuming too many ports.

### 3.1 Topology in a DGX H100 / HGX H100

```
┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐
│ GPU0 │  │ GPU1 │  │ GPU2 │  │ GPU3 │
└───┬──┘  └───┬──┘  └───┬──┘  └───┬──┘
    │         │         │         │
    └────┐    │    ┌────┘         │
         ▼    ▼    ▼              ▼
    ┌───────────────────────────────┐
    │   NVSwitch × 4 (900 GB/s      │
    │   per GPU aggregate)          │
    └───────────────────────────────┘
         ▲    ▲    ▲              ▲
    ┌────┘    │    └────┐         │
    │         │         │         │
┌───┴──┐  ┌───┴──┐  ┌───┴──┐  ┌───┴──┐
│ GPU4 │  │ GPU5 │  │ GPU6 │  │ GPU7 │
└──────┘  └──────┘  └──────┘  └──────┘
```

With NVSwitch, any GPU can talk to any other GPU at full 900 GB/s bandwidth, regardless of topology. This is essential for AllReduce in tensor parallelism, which is all-to-all in its data flow.

### 3.2 NVSwitch generations

| Gen | Bandwidth switched | Deployed in | Notes |
|---|---|---|---|
| NVSwitch 1 | 300 GB/s per GPU | DGX-1 / DGX A100 | 6 switches per node |
| NVSwitch 3 | 900 GB/s per GPU | DGX H100 / HGX H100 | 4 switches per node |
| NVSwitch 4 | 1.8 TB/s per GPU | DGX B200, NVL72 | SerDes upgraded to 200 Gbps |

---

## 4. PCIe — Host-to-Device and Non-NVLink GPUs

PCIe is still essential for:
- CPU ↔ GPU memory transfers (feeding batches from DataLoader)
- GPU ↔ NIC (for RDMA)
- GPU ↔ NVMe (for GDS)
- GPUs without NVLink (consumer cards, some cloud SKUs)

### 4.1 PCIe generations

| Gen | GT/s | x16 BW (per direction) | Typical use |
|---|---|---|---|
| Gen 3 | 8 | 16 GB/s | Legacy |
| Gen 4 | 16 | 32 GB/s | A100 systems |
| Gen 5 | 32 | 64 GB/s | H100 systems |
| Gen 6 | 64 | 128 GB/s | B200+ systems (2024-2025) |

Note: these are **per direction**. PCIe is full-duplex, so bidirectional peak is 2×.

### 4.2 PCIe bottlenecks

On a typical DGX H100, CPU↔GPU goes via PCIe Gen5 x16 = 64 GB/s. Compare to 900 GB/s NVLink between GPUs. This is why:
- You stage data from CPU to GPU once, then keep it there
- You avoid "hot" CPU-GPU transfers in the decode loop
- You use GPUDirect RDMA for NIC↔GPU (bypassing CPU bounce buffer)

---

## 5. InfiniBand and RoCE — Inter-Node Fabric

### 5.1 InfiniBand

InfiniBand (IB) is the premium AI-cluster fabric: kernel-bypass, RDMA-native, very low latency. Mellanox (now NVIDIA Networking) dominates.

| Gen | Per-link speed | Typical lane count | Practical per-port BW |
|---|---|---|---|
| HDR | 200 Gbps | x4 | ~25 GB/s |
| NDR | 400 Gbps | x4 | ~50 GB/s |
| XDR | 800 Gbps | x4 | ~100 GB/s |

Typical NIC: 1-4 IB ports per node, each at NDR 400 Gbps. An 8-GPU node with 8 NICs (one per GPU) gets 8 × 50 GB/s = 400 GB/s inter-node bandwidth.

### 5.2 RoCE v2

RoCE (RDMA over Converged Ethernet) puts RDMA semantics on top of Ethernet. Uses UDP for routing across L3 (v2 specifically). Bandwidth parity with IB, but less tuned latency.

| Advantages | Disadvantages |
|---|---|
| Reuses Ethernet skills, switches, cabling | Requires careful QoS (PFC, ECN) to avoid congestion collapse |
| No vendor lock-in | Higher latency than IB at same speed |
| Scales to larger fabrics (meta-scale) | More sensitive to buffer sizing |

Meta uses RoCE v2 at scale for GenAI (RoCEv2 + DCQCN). NVIDIA/OpenAI tend to use IB. Google uses their custom protocol (not exposed externally).

### 5.3 RDMA — the core primitive

**RDMA** (Remote Direct Memory Access) allows NIC A to read/write memory on node B directly, bypassing:
- The remote CPU
- The remote OS kernel
- Intermediate copy buffers

```
Without RDMA:
  GPU_A → CPU_A bounce → NIC_A → network → NIC_B → CPU_B bounce → GPU_B
  
With RDMA + GPUDirect:
  GPU_A → NIC_A → network → NIC_B → GPU_B    (zero CPU touches)
```

Key RDMA operations:
- `WRITE`: NIC pushes data to remote memory, no action on remote CPU
- `READ`: NIC pulls data from remote memory
- `SEND`/`RECV`: two-sided, like sockets but zero-copy
- `ATOMIC`: compare-and-swap, fetch-and-add for locks

NCCL uses mostly `WRITE` (and `SEND`/`RECV` for certain ring protocols).

---

## 6. GPUDirect — Bypassing the CPU

NVIDIA's family of technologies for removing CPU/memory bounces from GPU data paths.

### 6.1 GPUDirect RDMA

NIC writes directly into GPU HBM and vice versa, using PCIe peer-to-peer. Requires:
- PCIe ACS disabled (or specific switch config)
- GPU and NIC on the same PCIe root complex (ideally same NUMA node)
- NVIDIA peermem driver (or modern kernel's DMA-BUF support)

Without GPUDirect RDMA: NIC writes to CPU memory, then CPU → GPU via PCIe (two hops, double bandwidth). With it: one hop.

### 6.2 GPUDirect Storage (GDS)

NVMe SSDs write directly to GPU HBM. Requires NVIDIA GPU Driver + GDS kernel module + filesystem support.

For model loading: a 140 GB model loads in ~20 s from local NVMe at 7 GB/s. With GDS, the data doesn't bounce through CPU memory first, saving ~10 s and CPU RAM pressure.

### 6.3 GPUDirect Async (GDR-A)

Skip the host CPU for network progress. `ibv_post_send` without host involvement.

---

## 7. Network Topology for AI Clusters

### 7.1 Fat-tree

Classic datacenter topology. Multi-tier switch fabric where each node has a deterministic oversubscription ratio.

```
                    ┌─────┐  ┌─────┐  ┌─────┐
     Spine layer    │ SW  │  │ SW  │  │ SW  │
                    └──┬──┘  └──┬──┘  └──┬──┘
                       │       ...       │
                    ┌──┴──┐  ┌─────┐  ┌──┴──┐
     Leaf layer     │ SW  │  │ SW  │  │ SW  │
                    └──┬──┘  └──┬──┘  └──┬──┘
                       │        │        │
                       ▼        ▼        ▼
                     Node 0   Node 1   Node N
```

Non-blocking fat-tree: sum of uplink BW = sum of downlink BW at every layer. Typical non-blocking 2-tier design supports 1024-4096 nodes.

### 7.2 Rail-optimized

Standard for modern AI clusters. Each node has 8 GPUs, each with a dedicated NIC. GPUs of the same rank across all nodes connect to the same leaf switch (one "rail" per GPU slot).

```
Rail 0: GPU0 (every node) ── Leaf Switch 0 ── Spine
Rail 1: GPU1 (every node) ── Leaf Switch 1 ── Spine
Rail 2: GPU2 (every node) ── Leaf Switch 2 ── Spine
...
Rail 7: GPU7 (every node) ── Leaf Switch 7 ── Spine
```

Why this layout: TP collectives run on NVLink (intra-node), PP/DP collectives run on InfiniBand but follow the same rank everywhere (e.g., GPU 0 talks to GPU 0 on other nodes). Keeping same-rank traffic on dedicated rails minimizes cross-traffic congestion and simplifies routing.

### 7.3 Cluster design decisions

- **Oversubscription:** 1:1 (non-blocking) is ideal for training; 2:1 or 4:1 is acceptable for inference where cross-node traffic is lighter.
- **Cable reach:** copper for <3m (intra-rack), active optical for 3-100m (cross-rack), single-mode fiber for longer distances.
- **Switch radix:** modern IB switches support 64 × 400 Gbps ports per switch (NVIDIA Quantum-2 Q9700).

---

## 8. Collective Communication and Bandwidth Modeling

Understanding collective patterns is essential for capacity planning. See [Collectives_and_NCCL](Collectives_and_NCCL.md) for algorithms; here we focus on interconnect implications.

### 8.1 AllReduce bandwidth (TP)

For TP=8 on one node with 900 GB/s NVLink (per-GPU bidirectional):
- Ring AllReduce bytes on the wire: 2·(N-1)/N · data_size ≈ 1.75 × data_size for N=8
- Peak BW for the ring step ≈ min(link BW) = 900 GB/s
- For a 128 KB AllReduce (typical single-layer residual): wire time ≈ 128·1.75/900 ≈ 0.25 μs per AR

An 80-layer model has ~4 ARs per layer (QKV, attention output, MLP up, MLP down). 80 × 4 × 0.25 μs = 80 μs per step just in TP collectives — significant at 10 ms decode steps.

### 8.2 AllGather and ReduceScatter

FSDP and sequence parallelism use AllGather (for weights) and ReduceScatter (for gradients). Bytes on the wire: same as AllReduce when implemented as RS+AG.

### 8.3 PP send/recv

Pipeline parallel is point-to-point — a single send of activations from stage i to stage i+1. For 70B model, hidden = 8192, seq = 4096, fp16 → activation chunk = 8192 × 4096 × 2 = 64 MB.

At 50 GB/s NDR → 1.3 ms per pipeline step.

### 8.4 EP all-to-all

Expert parallel routing: tokens scatter to expert GPUs, then gather back. All-to-all is the most bandwidth-hungry collective.

For DeepSeek-V3 (256 routed experts, top-8 activation) on 8 nodes with EP=8:
- Total tokens: batch × seq × top-k = 32 × 4096 × 8 = 1M token-routings per step
- Each token-routing = hidden × fp16 = 16 KB
- All-to-all data volume: 16 GB, traversing inter-node fabric

This can be 30-50% of decode-step latency for MoE models if not optimized (DeepEP, DeepSpeed-MoE kernels).

---

## 9. Latency Components

Understanding where microseconds go helps diagnose problems.

### 9.1 NVLink latency breakdown

For a small (64-byte) transfer:
- SM-to-NIC DMA: ~200 ns
- NVLink propagation: 20-50 ns/meter of copper
- Remote DMA to SM: ~200 ns
- Total: ~500 ns-1 μs

For large (1 MB) transfer:
- Bandwidth-limited: 1 MB / 900 GB/s = 1.1 μs per direction

### 9.2 InfiniBand latency breakdown

For a 64-byte RDMA WRITE:
- Local QP processing: ~500 ns
- Switch hops (1-2): ~100 ns each
- Remote QP processing: ~500 ns
- Total: ~1.5-2 μs

### 9.3 Why you care

- Per-collective overhead is essentially a fixed cost that dominates small collectives
- For many small all-reduces per layer (TP), the overhead compounds
- "NCCL fusion" (grouping collectives) amortizes this

---

## 10. KV Cache Transfer in Disaggregated Inference

Prefill-decode disaggregation requires moving the KV cache (~GB per request) from prefill GPUs to decode GPUs.

### 10.1 NIXL (NVIDIA Inference Transfer Library)

Emerging standard in 2026. Handles:
- One-sided GPU-to-GPU transfers via UCX or native NVLink
- Multi-path striping across NICs
- Deterministic latency SLOs
- Layer-wise incremental transfer (overlap with prefill)

Typical transfer for Llama-70B 4K context KV cache (1.25 GB):
- Same-node over NVLink: 1.25 / 900 ≈ 1.4 ms
- Cross-node over NDR (400 Gbps): 1.25 / 50 ≈ 25 ms

### 10.2 Transfer patterns

**Synchronous:** prefill completes fully, then transfers KV, then decode begins. Simplest, highest TTFT.

**Layer-wise (streaming):** as each layer's prefill completes, immediately start transferring that layer's KV while computing the next layer. Hides transfer latency inside compute.

**Pipelined:** decode starts after the first K layers have transferred; remaining layers stream in as decode proceeds. Lowest TTFT but complex.

Modern disaggregated systems (vLLM + NIXL, NVIDIA Dynamo) use layer-wise streaming by default.

---

## 11. Congestion Control and QoS

### 11.1 Why it matters

An AI cluster during a large training run generates bursts of 50+ GB/s from every node simultaneously. Without congestion control:
- Shared links drop packets
- Retransmits multiply traffic
- Tail latency blows up (P99 >> P50)
- Training steps stall waiting for stragglers

### 11.2 InfiniBand's approach

IB uses **credit-based flow control**: senders never transmit unless receiver has buffer space. No packet drops, guaranteed delivery. Flow control operates at the link level.

### 11.3 RoCE's approach (PFC + ECN)

RoCE (Ethernet-based) uses:
- **PFC (Priority Flow Control):** pause specific traffic classes when a switch's egress queue fills
- **ECN (Explicit Congestion Notification):** mark packets with congestion bits; receiver signals sender to slow down
- **DCQCN (Data Center QCN):** the congestion-control algorithm combining ECN signals with rate reduction

Meta's "RoCEv2 at Scale" paper describes tuning PFC thresholds per-link, separating training traffic onto its own priority class, and running large-MTU (4 KB) jumbo frames for efficiency.

### 11.4 Common failure modes

- **PFC deadlock** — cyclic pause dependencies lock up the fabric. Requires careful routing that avoids cycles.
- **ECN under-reaction** — DCQCN not aggressive enough, leading to buffer overruns and drops.
- **PFC over-reaction** — pauses propagate back to uninvolved senders (head-of-line blocking on switches).

---

## 12. NIC Tuning and Multi-Rail Striping

### 12.1 NCCL sharding across NICs

Modern NCCL (2.18+) supports **multi-rail**: it shards a single collective across all available NICs on a node. If a node has 8 NICs at 400 Gbps each, aggregate BW is 3200 Gbps across the collective.

Needs correct topology detection via `NCCL_TOPO_FILE` or NCCL's automatic PCI topology walk.

### 12.2 Queue Pair (QP) tuning

RDMA uses QP pairs per connection. More QPs = more parallel DMA engines = better BW for small messages.

Typical settings:
- `NCCL_IB_QPS_PER_CONNECTION=4` — 4 QPs per peer for NCCL
- `NCCL_MIN_NCHANNELS=16` — more rings/trees for parallelism
- Messages under 1 KB: use separate "eager" path (no RDMA setup overhead)

### 12.3 Adaptive routing

Modern IB fabrics support **adaptive routing**: per-packet route selection based on switch congestion. Prevents a single congested link from blocking an entire flow.

NCCL does not require this (it uses static routing by default), but enabling it improves worst-case latency significantly.

---

## 13. Storage Fabric

AI clusters have two data paths:
1. **Compute fabric** (NVLink + IB): GPU-to-GPU
2. **Storage fabric**: GPU → network-attached storage

Storage fabric bandwidth requirements:
- Training: ~10 GB/s per node for dataloading (sustained)
- Inference: ~100 GB/s per node for model loading on cold start (bursty)
- Checkpointing: ~100 GB/s per node (bursty, during periodic checkpoints)

**Lustre / WekaFS / DAOS** are the dominant scale-out filesystems. Object stores (S3, GCS) are used for cold tier.

See [Storage_and_Model_Loading](Storage_and_Model_Loading.md) for the full storage picture.

---

## 14. Common Interview Questions on Networking

**Q: Why is NVLink used for tensor parallel but InfiniBand for pipeline parallel?**
A: TP AllReduce happens at every layer (multiple times) with relatively small tensors — it's latency-sensitive and requires very high bandwidth. NVLink's 900 GB/s is an order of magnitude more than IB's ~50 GB/s. PP is just per-stage send/recv with large activation tensors that tolerate millisecond latencies — IB's bandwidth suffices and the all-to-all topology isn't needed.

**Q: What is RDMA and how is it different from regular TCP networking?**
A: RDMA (Remote Direct Memory Access) lets a NIC read/write remote GPU/CPU memory directly, bypassing the remote OS kernel and CPU. TCP requires both endpoints to be involved at the CPU level for each packet. RDMA latency is ~2 μs; TCP is ~20-100 μs. Bandwidth is higher too because there's no kernel copy.

**Q: Explain GPUDirect RDMA.**
A: GPUDirect RDMA enables a NIC to write directly into GPU HBM via PCIe peer-to-peer, without the data bouncing through CPU memory. Critical for disaggregated KV cache transfer and for NCCL AllReduce when GPUs span multiple nodes.

**Q: What's the difference between InfiniBand and RoCE?**
A: Both support RDMA. IB is a separate physical layer (its own switches, NICs, cabling) with built-in credit-based flow control. RoCE puts RDMA on Ethernet — reuses datacenter Ethernet gear but requires careful QoS tuning (PFC, ECN, DCQCN). IB has lower latency at the same speed; RoCE is more flexible and scales to larger fabrics.

**Q: What is a rail-optimized topology and why is it used?**
A: Each GPU slot (0-7) in every node connects to its own dedicated leaf switch. "Rail 0" is all node's GPU 0s talking through the same switch. It minimizes cross-traffic congestion because DP/PP collectives for GPU i only use switch i's capacity. Plus it gives deterministic latency between same-rank GPUs across the cluster.

**Q: I have a TP=8 AllReduce of 4 MB every decode step, running at 10 ms/step. How much of that is communication?**
A: Ring AllReduce for 4 MB across 8 GPUs = 2·7/8 × 4 MB = 7 MB on the wire per GPU. At 900 GB/s NVLink, that's ~8 μs of bandwidth time plus ~1 μs overhead = ~9 μs. Multiple ARs per layer × 80 layers ≈ 80-100 μs per step — about 1% of your 10 ms step. Small, but measurable.

**Q: What happens when you try to do TP across two nodes via InfiniBand?**
A: Bandwidth drops from 900 GB/s (NVLink intra-node) to ~50 GB/s (NDR IB inter-node) — 18×. Decode latency for the same model is dominated by the AR time and grows dramatically. In practice, you never do TP across nodes in production (unless the model truly cannot fit intra-node); you use PP or EP across nodes.

**Q: Your inference deployment has 8-GPU nodes with NVL4 only (first 4 GPUs connected via NVLink, other 4 only via PCIe). How do you configure TP and placement?**
A: TP=4 maximum, pinned to the NVLink-connected GPUs. TP=8 would include the PCIe-connected GPUs and suffer 10-20× slower collectives. If you need the full node's compute, run two replicas of TP=4 (model replicated twice), each on its own NVL4 subgroup.

**Q: In a disaggregated deployment, prefill GPU 3 on node A needs to send 1.25 GB KV to decode GPU 5 on node B. Walk me through the data path.**
A: 1) Prefill completes, KV tensors in GPU 3 HBM on node A. 2) NIXL API call initiates an RDMA WRITE from GPU 3 HBM → node A NIC → fabric → node B NIC → GPU 5 HBM. 3) GPUDirect RDMA means the CPU is untouched on both sides. 4) Path uses the rail dedicated to GPU-slot 3 (or 5 — depends on rail topology). 5) At 50 GB/s NDR: 25 ms. 6) NIXL signals decode-side engine that KV is ready; decode starts.

**Q: What's the largest NVLink domain today, and what does that unlock?**
A: NVL72 — 72 GPUs in a single rack, all connected by NVLink 5 at 1.8 TB/s per GPU. It unlocks TP=72 for trillion-parameter models, which previously would have required 9 × TP=8 nodes stitched by InfiniBand (10× slower). NVL72 is the architectural enabler for models like GPT-4 class and DeepSeek-V3 scale serving.

---

## 15. Further Reading

- NVIDIA NVLink Whitepapers (Hopper and Blackwell)
- "Meta's RoCEv2-based Research SuperCluster" (2022) and "GenAI Training at Scale" (2024)
- RDMA programming primer: "An Introduction to the InfiniBand Verbs Interface" (Mellanox)
- NCCL documentation — NCCL_TOPO_FILE and multi-rail configuration
- Google's "Vertical and horizontal networking for large-scale ML" talks
- OpenAI's "Scaling Kubernetes to 7500 Nodes" (describes their network architecture)

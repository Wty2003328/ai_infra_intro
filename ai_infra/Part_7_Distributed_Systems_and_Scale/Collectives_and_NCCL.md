# Collectives and NCCL

NCCL (NVIDIA Collective Communications Library) is the GPU-aware MPI of deep learning. Every parallelism strategy ultimately calls a NCCL collective. This page covers the collective semantics, ring/tree/pipeline algorithms, the cost models, NCCL's internals (channels, protocols, environment vars), and how to debug perf problems.

**Prerequisites**: [Parallelism_Strategies](Parallelism_Strategies.md), [Networking_and_Interconnect](../Part_3_Hardware_Systems_and_Interconnects/Networking_and_Interconnect.md).

---

## 1. The Collectives

Let `N` = ranks, `B` = bytes per rank input.

| Collective       | Semantics | Output per rank | Used by |
|------------------|-----------|------------------|---------|
| broadcast        | rank 0 → all | size B          | initial weight broadcast |
| reduce           | all → rank 0 (sum/max/min) | B at root | rare alone |
| all-reduce       | all → all (sum/max/min) | B            | DP grad sync, TP all-reduce |
| reduce-scatter   | sum + each rank gets a slice | B/N      | ZeRO grad reduction |
| all-gather       | concat from all → all | N·B            | ZeRO param fetch, TP-SP |
| all-to-all       | each rank sends unique chunk to each other | B (with N×N pattern) | MoE EP, sequence parallel |
| barrier          | sync only | – | rare |
| send / recv      | point-to-point | – | PP activations |

Identity: **all-reduce ≡ reduce-scatter + all-gather** with the same total data movement. Modern frameworks lean on this identity to overlap halves with compute.

---

## 2. Algorithm: Ring vs Tree vs Pipeline

NCCL picks an algorithm based on payload size, topology, and bandwidth.

### 2.1 Ring All-Reduce (the default at large size)

Arrange ranks in a logical ring. Two phases:

**Reduce-scatter phase** (N-1 steps): each rank sends `B/N` bytes to next rank, receives `B/N` from previous, sums into local. After N-1 steps, each rank holds the **complete sum of one chunk**.

**All-gather phase** (N-1 steps): rotate chunks around the ring; each rank passes its complete chunk along.

Total bytes moved per rank: `2·(N-1)/N · B`. Time: `2·(N-1)/N · B / BW` plus `2·(N-1)·α` latency. Optimal for large B.

### 2.2 Tree All-Reduce (small payloads)

A binary tree: leaves send up, internal nodes sum, root sends back down. Latency `O(log N · α)`. Used by NCCL for small messages where ring's `O(N)` latency dominates.

### 2.3 Double-Binary-Tree (NCCL's default for medium)

Two binary trees with disjoint links → fully utilize bidirectional links. Commonly used for medium payloads on NVLink topologies.

### 2.4 Pipelined Variants

Subdivide payload into chunks and pipeline ring stages → overlaps comm with itself, hides the `(N-1)` factor up to a point. Used implicitly inside NCCL.

### 2.5 Cost Model (α/β)

Latency-bandwidth model: `T = α + β · B`, where `α` is per-message overhead, `β = 1/BW`. NCCL chooses the algorithm minimizing the model:

| Algorithm     | T model |
|---------------|---------|
| Ring AR       | `2·α·(N-1) + 2·β·B·(N-1)/N` |
| Tree AR       | `2·α·log₂(N) + β·B` (simplified) |
| Double-tree AR| `2·α·log₂(N) + β·B` (with BW factor) |

---

## 3. Topology Awareness

NCCL discovers the topology at init (via NVML, `nvidia-smi topo -m`-style logic) and picks rings/trees that map well to:

- **Intra-node**: NVLink/NVSwitch — fully connected effectively. NCCL builds rings using all NVLink lanes (typically 4–8 channels).
- **Inter-node**: PCIe + NIC + IB/RoCE. NCCL builds rings with one rank per NIC at the edge, switching to NVLink rings inside nodes.
- **NVL72**: a 72-GPU NVL domain looks like a fully connected node to NCCL → big rings stay on NVLink.
- **Rail-optimized**: NIC i on node A talks directly to NIC i on node B via the same rail switch. NCCL `rail` plugin assigns rings rail-by-rail.

`NCCL_TOPO_DUMP_FILE` exports the discovered topology for inspection.

---

## 4. Channels

A "channel" is one independent SM-driven engine running a piece of the collective. NCCL spawns multiple channels (typically 2–32) to parallelize across NVLink lanes and NICs.

```
NCCL_NCHANNELS_PER_PEER=2
NCCL_MIN_NCHANNELS=4
NCCL_MAX_NCHANNELS=32
```

More channels → more BW utilization and more SM occupancy used by NCCL. Trade against compute kernels running concurrently.

---

## 5. Protocols

NCCL has three protocols for moving data:

| Protocol | Description | Latency | BW efficiency |
|----------|-------------|---------|---------------|
| LL  (low-latency)        | tiny chunks, busy-loop ack | smallest | low |
| LL128                    | 128B chunks, slightly more overhead | small | medium |
| Simple                   | large chunks, BW-optimal | large | high |

NCCL picks per-message based on size and topology.

---

## 6. Performance Knobs

### 6.1 Environment Variables

| Var | What it does |
|-----|--------------|
| `NCCL_DEBUG=INFO` | logs algorithm/protocol choice |
| `NCCL_DEBUG_SUBSYS=ALL` | finer logging |
| `NCCL_ALGO=Ring,Tree` | restrict algorithms |
| `NCCL_PROTO=LL,LL128,Simple` | restrict protocols |
| `NCCL_IB_HCA=mlx5_0:1` | bind specific HCA ports |
| `NCCL_IB_GID_INDEX=3` | RoCEv2 GID for IPv4 |
| `NCCL_SOCKET_IFNAME=eth0` | force socket interface |
| `NCCL_P2P_LEVEL=NVL` | restrict P2P to NVLink |
| `NCCL_NVLS_ENABLE=1` | enable NVLink SHARP if available |
| `NCCL_BUFFSIZE=4194304` | per-channel buffer size |

### 6.2 Common Tuning

- Multi-NIC training: ensure `NCCL_IB_HCA` lists the right HCAs and one rank per NIC has `CUDA_VISIBLE_DEVICES` matched to the NUMA-local GPU.
- Cloud / TCP fallback: if IB is missing, NCCL falls back to TCP — much slower. Verify with `NCCL_DEBUG=INFO`.

### 6.3 NVLink SHARP / NVLS

On Hopper+ NVLink fabrics, switches can perform reductions in-fabric ("SHARP"). All-reduce becomes a single tree reduce in the switch + broadcast, halving the GPU-side BW pressure. Free win when available.

---

## 7. Overlap with Compute

To hide collective time:

1. **Issue early**: run all-reduce on layer i's grads while computing layer i-1's backward.
2. **Use streams**: NCCL ops on a separate CUDA stream from compute. Synchronize only when output is needed.
3. **Bucket gradients**: combine many small grad tensors into bigger buckets so each all-reduce amortizes its `α` cost. PyTorch DDP defaults to 25 MB buckets.
4. **ZeRO-3 prefetch**: gather params N+1 layers ahead while computing layer N.

Quality of overlap depends on:
- Comm BW vs compute time per layer.
- Whether NCCL's SM usage steals from compute kernels (channels can be tuned down).

---

## 8. Debugging Performance

### 8.1 Symptoms

- **Step time stochastic** with high variance: another job sharing NICs, or a slow GPU dragging down all-reduce.
- **Step time scales linearly with N**: ring algorithm is right but BW-bound — investigate NIC saturation or topology.
- **Step time scales with `log N` worse than expected**: tree algorithm or imbalanced subgroups.

### 8.2 Tools

- `nccl-tests` — `all_reduce_perf -b 8 -e 1G -f 2 -g <gpus>` — measure achieved BW vs payload size.
- Nsight Systems trace with NCCL annotations — see exact comm timeline.
- `NCCL_DEBUG=INFO` — confirms algorithm chosen.
- `ibstat`, `ibdiagnet` — IB fabric health.
- `nvidia-smi nvlink --status` — verify NVLink lanes are up.

### 8.3 Worked Diagnostic Flow

1. Are all-reduce calls actually NCCL? Confirm via Nsight, not assumed.
2. What payload size? Match the protocol/algorithm to expected best.
3. What BW are you achieving vs theoretical? `nccl-tests` busbw should be >70% of NIC BW for IB and >70% of NVLink BW intra-node.
4. Is it bursty or sustained? Bursty implies overlap broken.
5. Is one rank slow? Median vs max step time — straggler check.

---

## 9. NCCL vs Alternatives

- **Gloo** — CPU-friendly fallback, used by PyTorch for things like model loading. Doesn't use GPU, slow.
- **MSCCL / MSCCL++** — Microsoft's collective compiler. Synthesizes algorithms tailored to topology. Sometimes 1.5–2× faster than NCCL on novel shapes (e.g. small all-to-all on InfiniBand).
- **RCCL** — AMD GPU equivalent.
- **OpenMPI / IntelMPI** — HPC libraries, not GPU-first.

Most production stacks default to NCCL and fall back to Gloo for CPU bookkeeping.

---

## 10. Common Pitfalls

- **Mismatched group ordering**: TP group on `[0..7]` per node assumed, but actual rank assignment differs → NCCL all-reduce works but maps to wrong physical links.
- **Mixing NCCL streams without proper sync**: stale data read by compute kernel.
- **Not pinning CUDA device**: `CUDA_VISIBLE_DEVICES` and NCCL device selection mismatched.
- **TCP fallback unnoticed**: `NCCL_DEBUG=INFO` would show "via TCP" — orders of magnitude slow.
- **MIG slices**: NCCL behavior with MIG is limited — disabled in many configurations.

---

## 11. Common Interview Questions

**Q: What's the difference between all-reduce and reduce-scatter+all-gather?**
A: They move the same total bytes (about 2·B per rank) but reduce-scatter+all-gather can be split: do the reduce-scatter at the end of forward (or backward) and the all-gather at the start of the next layer's forward, hiding both inside compute. ZeRO-3 and Megatron sequence-parallel exploit this.

**Q: Why is ring all-reduce optimal at large payloads?**
A: At large B, the cost is dominated by `β·B`. Ring achieves `2·B·(N-1)/N` total bytes → near-optimal for a single full-bandwidth ring. Latency `α` is small relative.

**Q: Why use tree at small payloads?**
A: Latency-dominated regime. Tree has `log N` rounds vs ring's `N`, so tree wins when `α·N >> β·B`.

**Q: How does NCCL handle multi-NIC nodes?**
A: One ring channel per NIC (rail-aligned). NCCL discovers PCIe topology, assigns ranks to NICs, and builds rings that traverse the right physical paths. Tunable via `NCCL_IB_HCA`.

**Q: What is NVLink SHARP and what does it buy you?**
A: Switch-side reduction. The NVSwitch performs the sum across links in-network and broadcasts the result, halving GPU-side BW for all-reduce. Enabled with `NCCL_NVLS_ENABLE=1` on supported fabrics.

**Q: How do you overlap NCCL with compute in PyTorch?**
A: Use separate CUDA streams. PyTorch DDP does this internally with gradient buckets. For custom ops, launch NCCL on `c10d.ProcessGroupNCCL`'s stream and synchronize only when the result is consumed.

**Q: What does `nccl-tests` measure?**
A: All-reduce, all-gather, reduce-scatter, broadcast bandwidth at varying payload sizes. Reports `algbw` (bytes/(time)) and `busbw` (effective hardware throughput, normalized for the algorithm). Compare against theoretical NIC/NVLink BW.

**Q: How does all-to-all scale with N?**
A: Each rank sends N-1 messages → quadratic message count. BW-wise, total volume is `N·B` (each rank sends and receives B). Time scales `B/(BW/(N-1))` plus `α·(N-1)`. Performance-sensitive at small B; rail-aligned topology helps.

**Q: Why is NCCL faster than CPU MPI for collectives?**
A: GPU-resident buffers — no copy through host memory. Direct GPU↔GPU via NVLink or GPUDirect RDMA. NCCL is also tuned for GPU compute streams.

**Q: Walk me through what happens during a ring all-reduce of a 4-rank, 4-element vector.**
A: Step 1 (reduce-scatter): rank i sends element `i+1 mod 4`, receives element `i mod 4`, accumulates. Step 2: rotation of partial sums. After 3 steps total, each rank holds one fully reduced element. Step 4–6 (all-gather): rotate complete elements around. After 6 total steps, every rank has the full reduced vector.

**Q: When should you NOT use NCCL?**
A: Pure CPU collectives (use Gloo or MPI). Single-GPU. Tiny messages where launch overhead dominates (use a custom shared-memory primitive). Topologies where MSCCL has hand-tuned algorithms that beat NCCL.

**Q: How would you debug an inexplicably slow all-reduce in a multi-node training job?**
A: Set `NCCL_DEBUG=INFO`, run `nccl-tests` matching the topology, check NIC/HCA bindings and NUMA, verify no TCP fallback, check straggler GPU via per-rank step times, confirm sufficient channels (`NCCL_MIN_NCHANNELS`), compare achieved bus-bw to theoretical NIC BW.

---

## 12. Further Reading

- NCCL documentation and source code on GitHub.
- Patarasuk & Yuan, "Bandwidth Optimal All-reduce Algorithms for Clusters of Workstations" (2009) — the ring AR paper.
- Sanders et al., "Two-tree algorithms for full bandwidth broadcast, reduction and scan" (2009).
- NVIDIA SHARP whitepapers.
- MSCCL/MSCCL++ papers (MSR).
- `nccl-tests` benchmark suite.

---

**Next:** [Distributed_Training](Distributed_Training.md).
**See also:** [Parallelism_Strategies](Parallelism_Strategies.md), [Networking_and_Interconnect](../Part_3_Hardware_Systems_and_Interconnects/Networking_and_Interconnect.md), [Observability_and_Debugging](../Part_6_Inference_Engines_and_Serving/Observability_and_Debugging.md).

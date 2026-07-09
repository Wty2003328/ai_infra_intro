# L4 — Systems and Interconnects: Interview Questions

Consolidated interview Q&A and worked problems from every page in `L4_Systems_and_Interconnects/`, moved here to keep topic pages focused on concepts. Each section links back to its source page for full context.

---

## Networking and Interconnect — From SerDes Physics to Cluster Topology

*From [Networking_and_Interconnect.md](../L4_Systems_and_Interconnects/01_Networking_and_Interconnect.md)*

**Q1.** *A 65,536-GPU cluster uses a two-tier fat-tree of NDR (400 Gb/s, 64-port) switches. Each GPU connects via a single NDR port. Calculate the total number of switches, the aggregate bisection bandwidth, and the network cost (at $30k per switch).*

Number of ToR (edge) switches: each ToR has 64 ports; 32 are downlinks to GPUs, 32 are uplinks. So each ToR serves 32 GPUs. Total ToR switches: $65{,}536 / 32 = 2{,}048$.

Number of spine switches: each spine has 64 ports, all downlinks to ToRs. But each ToR has 32 uplinks. Total uplinks = $2{,}048 \times 32 = 65{,}536$. Each spine switch receives 64 uplinks. Number of spine switches: $65{,}536 / 64 = 1{,}024$.

Total switches: $2{,}048 + 1{,}024 = 3{,}072$.

Aggregate bisection bandwidth: full bisection in a two-tier fat-tree. Bisection cuts $\frac{N}{2} = 32{,}768$ links. Each link is 400 Gb/s = 50 GB/s. $B_{bisect} = 32{,}768 \times 50 = 1{,}638{,}400$ GB/s $= 1.64$ PB/s.

Network cost: $3{,}072 \times \$30{,}000 = \$92.16$ M.

**Q2.** *A 224 Gbps PAM4 SerDes link uses Megtron-6 laminate (α = 0.7 dB/(inch·GHz^{1/2})). Package loss is 5 dB, connector loss is 2 dB, and the total equalization budget is 30 dB. What is the maximum PCB trace length?*

Insertion loss per inch at Nyquist frequency $f = 56$ GHz:

$$IL/inch = \alpha \cdot \sqrt{f} = 0.7 \times \sqrt{56} = 0.7 \times 7.48 = 5.24 \text{ dB/inch}$$

Available budget for PCB trace:

$$IL_{trace} = 30 - 5 - 2 = 23 \text{ dB}$$

Maximum trace length:

$$\ell = \frac{23}{5.24} = 4.39 \text{ inches} \approx 0.11 \text{ m}$$

With Megtron-7G ($\alpha = 0.4$): $IL/inch = 0.4 \times 7.48 = 2.99$ dB/inch, $\ell = 23 / 2.99 = 7.7$ inches $= 0.20$ m. Still well under 1 m. This explains why 224G PAM4 requires either very short traces or active retimers.

**Q3.** *64 GPUs perform an All-to-All collective where each GPU sends 100 MB to every other GPU. The total data transferred is 64 × 63 × 100 MB = 403.2 GB. The interconnect provides 50 GB/s per GPU of usable bandwidth. How long does the All-to-All take? What is the efficiency if DCQCN causes 40% bandwidth reduction due to congestion?*

Without congestion: each GPU sends $63 \times 100 = 6{,}300$ MB. At 50 GB/s:

$$T_{ideal} = \frac{6{,}300 \text{ MB}}{50 \text{ GB/s}} = \frac{6.3 \text{ GB}}{50 \text{ GB/s}} = 0.126 \text{ s} = 126 \text{ ms}$$

With DCQCN 40% reduction ($BW_{eff} = 0.6 \times 50 = 30$ GB/s):

$$T_{actual} = \frac{6.3 \text{ GB}}{30 \text{ GB/s}} = 0.21 \text{ s} = 210 \text{ ms}$$

Efficiency: $126 / 210 = 60\%$. Nearly half the potential bandwidth is lost to congestion control.

**Q4.** *Compare the bisection bandwidth of a 3D torus of 8,960 nodes (20 × 16 × 28) with a fat-tree of the same size, assuming per-link bandwidth b = 50 GB/s.*

3D torus: the smallest cross-section is $X \times Y = 20 \times 16 = 320$ links (cutting along Z). Each link carries $b = 50$ GB/s in each direction. Bisection bandwidth:

$$B_{torus} = 2 \times 320 \times 50 = 32{,}000 \text{ GB/s} = 32 \text{ TB/s}$$

Fat-tree: $\frac{N}{2} \cdot b = \frac{8{,}960}{2} \times 50 = 224{,}000$ GB/s $= 224$ TB/s.

Ratio: $224 / 32 = 7×$. The fat-tree has 7× more bisection bandwidth but requires thousands of switches; the torus requires none.

**Q5.** *A B200 GPU loads a 1.4 TB checkpoint (FP16 weights of a 700B model) from storage. Compare load times for: (a) PCIe Gen5 x16 at 63 GB/s, (b) 8 NVMe drives with GDS at 56 GB/s aggregate, (c) NVLink from a peer GPU at 1.8 TB/s.*

(a) PCIe: $T = 1{,}400 / 63 = 22.2$ s. The model is on host RAM, DMA'd over PCIe.

(b) GDS: $T = 1{,}400 / 56 = 25.0$ s. Limited by NVMe bandwidth. GDS saves latency but not throughput here — the bottleneck is the storage device, not the bus. However, GDS avoids the 2× memory copy, saving ~100 ms of setup time.

(c) NVLink P2P: $T = 1{,}400 / 1{,}800 = 0.78$ s. 28× faster than PCIe. This is why disaggregated serving pre-loads models into a "warm pool" of GPUs that feed inference instances via NVLink.

---

## Rack-Scale Design — Power, Thermal, and Mechanical Engineering of AI Racks

*From [Rack_Scale_Design.md](../L4_Systems_and_Interconnects/02_Rack_Scale_Design.md)*

**Q1.** *A data center operator plans to deploy 50 NVL72 racks. Each rack draws 130 kW. The facility receives power at $0.08/kWh. Calculate the annual power cost and the required cooling plant capacity.*

Per-rack power: 130 kW.

50 racks: $50 \times 130 = 6{,}500$ kW = 6.5 MW.

Annual energy: $6.5 \times 10^6 \times 8{,}760 = 5.694 \times 10^{10}$ Wh = 56,940 MWh.

Annual cost: $56{,}940 \times 1{,}000 \times \$0.08 = \$4{,}555{,}200 \approx \$4.56$ M/year.

Cooling plant: PUE (Power Usage Effectiveness) for liquid-cooled data centers is ~1.15–1.25. Total facility power:

$$P_{facility} = P_{IT} \times PUE = 6.5 \text{ MW} \times 1.2 = 7.8 \text{ MW}$$

Cooling plant capacity: $7.8 - 6.5 = 1.3$ MW for CDUs, dry coolers, pumps, lighting, and overhead.

Dry coolers needed (at 500 kW each): $1{,}300 / 500 \approx 3$ dry coolers (with redundancy, 4–5 units).

**Q2.** *A B300 GPU draws 1,200 W at 0.7 V. The PCB trace from VRM to die has inductance L = 0.3 nH and resistance R = 0.05 mΩ. The GPU transitions from 10% to 100% load in 20 ns. Calculate the voltage droop. Is it within the ±5% tolerance?*

Steady-state current at 100%: $I_{ss} = 1{,}200 / 0.7 = 1{,}714$ A.

Transient current change: $\Delta I = 1{,}714 \times 0.9 = 1{,}543$ A.

di/dt: $\frac{di}{dt} = \frac{1{,}543}{20 \times 10^{-9}} = 7.71 \times 10^{10}$ A/s.

Inductive droop: $\Delta V_L = L \cdot \frac{di}{dt} = 0.3 \times 10^{-9} \times 7.71 \times 10^{10} = 23.1$ mV.

Resistive droop (at full current): $\Delta V_R = R \cdot I = 0.05 \times 10^{-3} \times 1{,}714 = 0.086$ mV.

Total droop: $23.1 + 0.086 \approx 23.2$ mV.

Allowable: $\pm 5\% \times 700 = 35$ mV. The droop of 23.2 mV is within tolerance (3.3%). The inductive term dominates; the resistive term is negligible.

**Q3.** *Compare coolant flow rates for (a) single-phase water cooling of a 140 kW rack with ΔT = 8°C, and (b) two-phase cooling using R-1234ze where 80% of heat is removed by evaporation and 20% by sensible heat with ΔT = 5°C. For R-1234ze: h_fg ≈ 180 kJ/kg, C_p ≈ 1,200 J/(kg·K).*

(a) Single-phase water:

$$\dot{m}_{water} = \frac{140{,}000}{4{,}184 \times 8} = 4.18 \text{ kg/s} \approx 66.3 \text{ GPM}$$

(b) Two-phase R-1234ze:

$$\dot{m}_{evap} = \frac{0.8 \times 140{,}000}{180{,}000} = 0.622 \text{ kg/s}$$

$$\dot{m}_{sensible} = \frac{0.2 \times 140{,}000}{1{,}200 \times 5} = 4.67 \text{ kg/s}$$

$$\dot{m}_{total} = 0.622 + 4.67 = 5.29 \text{ kg/s}$$

Comparison: single-phase water requires 4.18 kg/s of water; two-phase requires 5.29 kg/s of R-1234ze. The mass flow is actually higher for two-phase because R-1234ze's C_p and h_fg are much lower than water's. However, the **volumetric** flow differs: R-1234ze density at saturation is ~1,150 kg/m³, giving $\dot{V}_{R1234ze} = 5.29/1{,}150 = 4.6$ L/s vs $\dot{V}_{water} = 4.18/1{,}000 = 4.18$ L/s. Nearly the same. The real benefit of two-phase is **lower pump power** (lower pressure drop) and the ability to maintain near-constant temperature (isothermal boiling), which simplifies thermal management.

**Q4.** *An NVL72 rack needs to load a 1 Trillion parameter model in FP16 (2 TB total) distributed across 72 GPUs. Each GPU needs 2 TB/72 ≈ 27.8 GB. If the model is stored on a shared NVMe array delivering 56 GB/s via GDS to each GPU node, and loading is pipelined across 72 GPUs, how long does the full load take? Compare with loading over 100 GbE (12.5 GB/s per GPU).*

Each GPU loads 27.8 GB from local NVMe (56 GB/s available from 8 drives):

$$T_{local} = \frac{27.8}{56} = 0.50 \text{ s}$$

If all 72 GPUs load simultaneously from their local NVMe, the load is parallel:

$$T_{parallel} = 0.50 \text{ s}$$

Total time is bounded by the slowest GPU, not the aggregate bandwidth.

Over 100 GbE (12.5 GB/s per GPU), loading from a remote storage server:

$$T_{network} = \frac{27.8}{12.5} = 2.22 \text{ s}$$

The 100 GbE path is 4.4× slower. For a multi-rack deployment where each rack loads from a central model store, the network becomes the bottleneck: a single 400 GbE link serving 72 GPUs would give only $50/72 = 0.69$ GB/s per GPU, resulting in $27.8/0.69 = 40$ s.

**Q5.** *A 10,000-GPU cluster uses NVL72 racks (139 racks, with one partial). Each rack has 36 NDR400 uplinks. Calculate the total scale-out bandwidth and the number of spine switches needed for a two-tier fat-tree.*

Total uplinks per rack: 36 × 400 Gb/s = 14.4 Tb/s = 1,800 GB/s per rack.

Total scale-out BW: 139 × 1,800 = 250,200 GB/s ≈ 250 TB/s.

For a fat-tree: 10,000 GPUs, each connected via one NDR400 link (400 Gb/s). Number of edge switches: 10,000 / 32 (half ports down, half up) = 313 edge switches (with 7 unused ports on the last one).

Total uplinks from edge to spine: 313 × 32 = 10,016 uplinks.

Spine switches (64-port): 10,016 / 64 = 157 spine switches.

Total switches: 313 + 157 = 470 NDR switches.

---

## Storage and Model Loading — From NVMe to HBM

*From [Storage_and_Model_Loading.md](../L4_Systems_and_Interconnects/03_Storage_and_Model_Loading.md)*

**Q1.** *A training cluster of 8 NVL72 racks (576 GPUs) trains a 400B-parameter FP16 model. The model uses FSDP with optimizer state sharding. Calculate: (a) the per-GPU checkpoint size, (b) the per-GPU checkpoint save time with local NVMe (56 GB/s), (c) the total checkpoint bandwidth across all 576 GPUs, and (d) the network bandwidth needed to replicate all checkpoints to a remote array.*

(a) Model weights: 400B × 2 bytes = 800 GB. Adam states: 400B × 2 × 4 bytes (FP32 momentum + variance) = 3,200 GB. Total: 4,000 GB.

Per GPU (FSDP sharding): $4{,}000 / 576 = 6.94$ GB per GPU.

(b) Save time with local NVMe at 56 GB/s:

$$T_{save} = \frac{6.94}{56} = 0.124 \text{ s} \approx 124 \text{ ms}$$

(c) Total aggregate checkpoint bandwidth:

$$BW_{total} = \frac{576 \times 6.94}{0.124} = \frac{3{,}997}{0.124} \approx 32{,}234 \text{ GB/s} \approx 32 \text{ TB/s}$$

(d) For async replication to remote storage: each GPU replicates its 6.94 GB shard over the network. With 4 NDR uplinks per rack (200 GB/s per rack), and 8 racks: $8 \times 200 = 1{,}600$ GB/s aggregate network bandwidth.

$$T_{replicate} = \frac{576 \times 6.94}{1{,}600} = \frac{3{,}997}{1{,}600} = 2.5 \text{ s}$$

This overlaps with training, so the effective impact is zero if training continues during replication.

**Q2.** *A 70B model in BF16 generates KV cache at 4 MB per token. An inference server has 8 × B300 GPUs (192 GB HBM each, 1,536 GB total). The model weights occupy 140 GB, leaving 1,396 GB for KV cache. What is the maximum batch size × sequence length product that fits in HBM? If 50% of KV is offloaded to host DDR5, what does the effective capacity become?*

Available HBM for KV: 1,396 GB = 1,396,000 MB.

KV per (batch × seq_len): 4 MB.

$$batch \times seq\_len = \frac{1{,}396{,}000}{4} = 349{,}000$$

For example: batch=256, seq_len=1,363 tokens. Or batch=128, seq_len=2,727 tokens.

With 50% offload to DDR5: 698 GB stays in HBM, 698 GB in DDR5. HBM holds:

$$batch \times seq\_len = \frac{698{,}000}{4} = 174{,}500$$

But the **effective** batch×seq_len doubles because the offloaded KV is still accessible (with higher latency). Total capacity: 1,396 GB (HBM) + host DDR5 capacity. If the host has 1 TB DDR5:

$$batch \times seq\_len = \frac{1{,}396{,}000 + 1{,}000{,}000}{4} = 599{,}000$$

This is 1.7× the HBM-only capacity, at the cost of ~100 ns latency per offloaded KV access.

**Q3.** *Using Young's formula, calculate the optimal checkpoint interval for a 100,000-GPU cluster. Assume MTBF = 1.5 hours and checkpoint save time = 15 s (network-bottlenecked).*

$MTBF = 1.5 \times 3{,}600 = 5{,}400$ s. $T_{ckpt} = 15$ s.

$$T_{interval} = \sqrt{2 \times 15 \times 5{,}400} - 15 = \sqrt{162{,}000} - 15 = 402.5 - 15 = 387.5 \text{ s} \approx 6.5 \text{ min}$$

Training efficiency: fraction of time doing useful work:

$$E = \frac{T_{interval}}{T_{interval} + T_{ckpt}} = \frac{387.5}{387.5 + 15} = 96.3\%$$

Lost work per failure: on average, half the interval ($T_{interval}/2 = 193.75$ s). With MTBF = 5,400 s, the expected fraction of lost work:

$$\frac{T_{interval}/2}{MTBF} = \frac{193.75}{5{,}400} = 3.6\%$$

Total efficiency: $96.3\% \times (1 - 0.036) = 92.8\%$. About 7.2% of compute is lost to checkpointing and failures.

**Q4.** *Compare cold-start times for a 140 GB model across three paths: (a) S3 over 100 GbE, (b) local NVMe via GDS, (c) NVLink P2P from a warm-pool GPU. Assume S3 first-byte latency = 150 ms.*

(a) S3 over 100 GbE (12.5 GB/s):

$$T_{S3} = 0.15 + \frac{140}{12.5} = 11.35 \text{ s}$$

(b) Local NVMe via GDS (56 GB/s):

$$T_{GDS} = \frac{140}{56} = 2.5 \text{ s}$$

(c) NVLink P2P (1.8 TB/s):

$$T_{NVLink} = \frac{140}{1{,}800} = 0.078 \text{ s} = 78 \text{ ms}$$

Speedups: GDS is 4.5× faster than S3; NVLink P2P is 145× faster than S3. This is why disaggregated serving architectures maintain warm pools of GPUs that can transfer models via NVLink in milliseconds.


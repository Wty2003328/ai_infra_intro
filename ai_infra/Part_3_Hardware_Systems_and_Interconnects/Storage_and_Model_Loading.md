# Storage, Model Loading, and Checkpointing

A rigorous mathematical analysis of the AI storage tier. As models scale past 1 Trillion parameters (e.g., DeepSeek-V3, Llama-4), the storage subsystem transforms from a trivial capacity constraint into a catastrophic bandwidth bottleneck governing cluster uptime and training efficiency.

**Prerequisites**: [GPU_Architecture](../Part_2_Microarchitecture/GPU_Architecture.md), [Networking_and_Interconnect](Networking_and_Interconnect.md), [Parallelism_Strategies](../Part_7_Distributed_Systems_and_Scale/Parallelism_Strategies.md).

---

## 1. The PCIe Bottleneck

While GPUs possess immense internal memory bandwidth (e.g., $12$ TB/s HBM3e on Blackwell Ultra), loading a model from persistent storage requires traversing the host CPU's PCIe bus.

### 1.1 PCIe Gen5 Math
A modern GPU connects to the host via a PCIe Gen5 x16 slot.
- **Raw Bitrate**: 32 GigaTransfers per second (GT/s) per lane.
- **Encoding**: 128b/130b encoding yields $\approx 98.46\%$ efficiency.
- **Theoretical Bandwidth**: $16 \text{ lanes} \times 32 \text{ GT/s} \times \frac{1 \text{ byte}}{8 \text{ bits}} \times 0.9846 \approx 63.0 \text{ GB/s}$ unidirectional ($126$ GB/s bidirectional).

### 1.2 Model Loading Time
To load a dense 1 Trillion parameter model in FP8 ($1 \text{ TB}$ of data) over a single PCIe Gen5 x16 bus:
$$T_{load} = \frac{1000 \text{ GB}}{63.0 \text{ GB/s}} \approx 15.8 \text{ seconds}$$
*Note*: This assumes a single unified host. In an NVL72 rack, the $1 \text{ TB}$ is sharded across $72$ GPUs, meaning each GPU only loads $\approx 13.8$ GB. Assuming the storage fabric can feed all 72 GPUs simultaneously at $63$ GB/s, the load time collapses to $\approx 0.22$ seconds.

---

## 2. GPUDirect Storage (GDS)

Traditional POSIX read/write operations pull data from NVMe drives, route it into the CPU's main system RAM (bounce buffer), and then DMA it over PCIe to the GPU.
- **CPU Bottleneck**: The bounce buffer saturates CPU memory controllers and consumes massive CPU cycles.
- **GDS Architecture**: GPUDirect Storage leverages RDMA to establish a direct Peer-to-Peer (P2P) PCIe path between the NVMe controller and the GPU HBM, mathematically bypassing the CPU entirely.
- By bypassing the CPU, GDS reduces latency $L_{IO}$ and mathematically guarantees that the throughput $\beta_{IO}$ reaches the PCIe physical limit ($63$ GB/s).

---

## 3. Checkpointing in Distributed Training

During training, large clusters experience regular hardware failures (Mean Time Between Failures, MTBF $\approx 1$ per 1000 GPU-days). To prevent catastrophic loss of compute hours, models must periodically save checkpoints.

### 3.1 Checkpoint Bandwidth Math
A checkpoint includes model weights, optimizer states (e.g., Adam requires 2 additional parameters per weight), and activation buffers. For a 1 Trillion parameter FP16 model:
- Weights: 2 TB
- Adam States: 4 TB (momentum + variance)
- Total Checkpoint Size: $\approx 6$ TB.

Let the cluster contain $N$ nodes. If the cluster trains for $H$ hours between failures, and checkpoints take time $T_{ckpt}$, the **Training Efficiency** $E$ is:
$$E = \frac{H - T_{ckpt}}{H}$$
If $T_{ckpt}$ takes $15$ minutes due to slow NFS storage, training efficiency plummets. To achieve $T_{ckpt} \le 10$ seconds, the parallel file system (e.g., Lustre, WEKA, IBM DAOS) must sustain an aggregate write bandwidth:
$$\beta_{write} = \frac{6000 \text{ GB}}{10 \text{ seconds}} = 600 \text{ GB/s}$$

### 3.2 NVMe over Fabrics (NVMe-oF)
To deliver $600$ GB/s, storage is disaggregated. NVMe-oF extends the NVMe protocol across the RoCEv2/InfiniBand network.
- **RDMA Write**: The GPUs dump their HBM states directly over the network into remote NVMe flash arrays utilizing RDMA, avoiding any CPU intervention.

---

## 4. Further Reading

- [Networking_and_Interconnect](Networking_and_Interconnect.md) - For the RoCEv2 fabric carrying NVMe-oF traffic.
- [Rack_Scale_Design](Rack_Scale_Design.md) - For the physical limitations of these storage deployments.
- [Parallelism_Strategies](../Part_7_Distributed_Systems_and_Scale/Parallelism_Strategies.md) - For how parameter sharding dictates checkpointing patterns.
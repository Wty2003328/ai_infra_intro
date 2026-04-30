# Networking and Interconnect Architectures

A rigorous, graduate-level analysis of the network topologies and congestion control algorithms underpinning hyperscale AI infrastructure. The exponential growth of distributed training (e.g., 100k GPU clusters) dictates that the network is the computer. We examine the fundamental mathematics of scale-up domains (Clos vs. Torus), the physical realities of SerDes signaling, and the evolution of RoCEv2 congestion control (DCQCN).

**Prerequisites**: [GPU_Architecture](../Part_2_Microarchitecture/GPU_Architecture.md), [Memory_Hierarchy_and_Roofline](../Part_2_Microarchitecture/Memory_Hierarchy_and_Roofline.md), [Parallelism_Strategies](../Part_7_Distributed_Systems_and_Scale/Parallelism_Strategies.md).

---

## 1. Scale-Up Domains: NVLink, UALink, and ICI

A "scale-up" domain is defined by cache-coherent, low-latency interconnects (typically $<500$ ns) enabling thousands of gigabytes per second of raw throughput per device.

### 1.1 Physical Layer Limits: 224G PAM4
Modern interconnects—NVLink-5 and UALink v1.0—rely on 224 Gbps PAM4 SerDes.
- **PAM4**: Pulse Amplitude Modulation 4-level encodes 2 bits per clock cycle. At $224$ Gbps, the Nyquist frequency is $56$ GHz.
- **Signal Integrity**: At $56$ GHz, signal attenuation across copper PCB traces ($\approx 1$ dB/inch) is immense. Without active retimers or optical transceivers, a passive copper trace cannot exceed approximately $0.5$ to $1.0$ meters, physically restricting these topologies to a single rack or contiguous server chassis.

### 1.2 Topology 1: The Non-Blocking Clos Network (NVL72)
NVIDIA's NVL72 architecture relies on a 2-tier folded Clos network.
- **Mathematical Definition**: A network is strictly non-blocking if any idle input port can connect to any idle output port regardless of existing traffic.
- Let $r$ be the radix (number of ports) of an NVSwitch ASIC. For NVL72, $r=144$ ports per ASIC, each $50$ GB/s.
- The maximum number of nodes $N$ for a 2-tier folded Clos is:
$$N = \frac{r^2}{2} = \frac{144^2}{2} = 10,368 \text{ endpoints}$$
- By constraining the domain to $72$ GPUs, NVIDIA massively overprovisions the backbone, ensuring $1.8$ TB/s per-GPU full-bisection bandwidth. For an All-to-All collective across $72$ GPUs, each GPU can transmit its data simultaneously without any statistical blocking probability.

### 1.3 Topology 2: The 3D Torus (Google TPU)
Google's Inter-Core Interconnect (ICI) utilizes a 3D Torus.
- In a $X \times Y \times Z$ Torus, each node has 6 connections.
- The average hop count (latency) $H$ is $\approx \frac{X+Y+Z}{4}$.
- **Bisection Bandwidth**: The bisection bandwidth $B_{bisect}$ of a 3D Torus of $N$ nodes, with link bandwidth $b$, is mathematically:
$$B_{bisect} \approx 2 \cdot b \cdot N^{2/3}$$
- Contrast with a Clos network, where $B_{bisect} = \frac{N}{2} \cdot b$. As $N \rightarrow 100,000$, the torus bisection bandwidth scales poorly ($\propto N^{2/3}$) compared to a Clos network ($\propto N$). Thus, Google must rely heavily on Dimension-Ordered Routing (DOR) and Optical Circuit Switching (OCS) to mitigate structural bottlenecks.

---

## 2. Scale-Out Fabrics: RoCEv2 and Congestion Control

When scaling beyond a single rack (e.g., connecting NVL72 racks or UALink pods), the cluster relies on Ethernet or InfiniBand.

### 2.1 RoCEv2 (RDMA over Converged Ethernet)
RoCEv2 enables Remote Direct Memory Access (RDMA) over standard UDP/IP Ethernet. It requires a "lossless" fabric, historically provided by Priority Flow Control (PFC).

### 2.2 The Incast Problem
In distributed training (e.g., Ring AllReduce or All-to-All), the communication pattern often results in $N-1$ nodes transmitting simultaneously to $1$ receiving node. This creates a microburst "incast", instantly overflowing the Top-of-Rack (ToR) switch buffers (typically 32-128 MB of shared SRAM).

### 2.3 DCQCN (Data Center Quantized Congestion Notification)
To prevent PFC from triggering a cascading pause (Head-of-Line blocking), RoCEv2 employs DCQCN, integrating ECN (Explicit Congestion Notification).
- **The DCQCN Math**:
When a switch buffer queue depth $Q$ exceeds a threshold $K_{min}$, the switch statistically marks packets with an ECN bit.
The receiving NIC generates a Congestion Notification Packet (CNP) back to the sender.
The sender maintains an exponentially weighted moving average $\alpha$ of the fraction of marked packets:
$$\alpha \leftarrow (1-g) \alpha + g \cdot F$$
Where $F=1$ if a CNP arrived, $0$ otherwise ($g \approx 1/256$).
The sender linearly decreases its transmission rate $R$:
$$R \leftarrow R \left(1 - \frac{\alpha}{2}\right)$$
This control-theoretic loop ensures that flows back off before switch buffers drop packets.

### 2.4 Ultra Ethernet Consortium (UEC)
Modern topologies are migrating from DCQCN to UEC standards. UEC deprecates strict reactive ECN in favor of **packet spraying** (multipath dispersion) and selective retransmission, mathematically maximizing ECMP (Equal Cost Multi-Path) utilization.

---

## 3. Further Reading

- [Blackwell_Architecture](../Part_2_Microarchitecture/Blackwell_Architecture.md) - How NVL72 implements the Clos network.
- [Google_TPU](../Part_2_Microarchitecture/Google_TPU.md) - How OCS modulates the 3D Torus.
- [Parallelism_Strategies](../Part_7_Distributed_Systems_and_Scale/Parallelism_Strategies.md) - How All-to-All and All-Reduce collectives map onto these topologies.
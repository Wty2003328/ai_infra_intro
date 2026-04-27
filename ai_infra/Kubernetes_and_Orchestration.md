# Kubernetes and Orchestration

How LLM workloads actually run in production fleets: the GPU operator, MIG, scheduling for multi-GPU jobs, autoscaling for inference, and the cross-cutting tools (KubeRay, KServe, NIM, NVIDIA Run:ai, Skypilot). This page focuses on what an MLSys engineer needs to design and debug, not full Kubernetes from scratch.

**Prerequisites**: [Networking_and_Interconnect](Networking_and_Interconnect.md), [Inference_Frameworks](Inference_Frameworks.md), [Production_Architecture](Production_Architecture.md).

---

## 1. Why Kubernetes for AI

LLM inference and training run on heterogeneous fleets (many GPU types, network rails, data tiers). Kubernetes provides:

- **Schedulability** — bin-packing workloads onto nodes with constraints (GPU type, NUMA, network).
- **Lifecycle** — restart on crash, rolling upgrade, version pinning.
- **Service discovery** — clients find live replicas without hardcoded IPs.
- **Autoscaling** — scale replicas based on metrics (RPS, queue depth, GPU util).
- **Multi-tenancy** — namespaces, RBAC, quotas.

Production at scale almost always builds on K8s, even when the "GPU job" feels alien to web-app patterns.

---

## 2. The NVIDIA GPU Operator

A meta-controller that installs and manages GPU-related components on each node:

- **NVIDIA driver** — version pinned, kernel-module signed.
- **Container Toolkit** — runtime hook making GPUs visible to containers.
- **Device plugin** — exposes `nvidia.com/gpu` as a schedulable resource.
- **DCGM exporter** — Prometheus metrics for GPU health/util.
- **MIG manager** — partitions H100/A100 into MIG slices.
- **GPUDirect RDMA modules** — for IB-attached fleets.
- **Node Feature Discovery** — tags nodes by GPU model, NVLink topology.

Without the GPU Operator, you're hand-managing driver versions across a fleet — painful at scale.

```yaml
# Pod requesting 1 GPU
resources:
  limits:
    nvidia.com/gpu: 1
```

---

## 3. MIG (Multi-Instance GPU)

H100 / A100 / B200 can be partitioned into hardware-isolated slices. Example H100 partitions:

| Profile  | SMs | HBM | Use case |
|----------|-----|-----|----------|
| 7g.80gb  | 132 | 80 GB | full GPU |
| 4g.40gb  | 60  | 40 GB | half |
| 3g.40gb  | 42  | 40 GB | three of these per H100 |
| 2g.20gb  | 28  | 20 GB | small jobs |
| 1g.10gb  | 14  | 10 GB | smallest |

Use MIG when:
- Hosting many small models.
- Multi-tenant inference where strong isolation is needed.
- Latency-critical workloads that must not share with noisy neighbors.

Don't use MIG when:
- Single workload spans the full GPU.
- Need NVLink (MIG slices have **no inter-slice NVLink** — they live in a single GPU only).

The GPU operator's MIG manager configures profiles per node.

---

## 4. Scheduling Multi-GPU Workloads

A single pod can request multiple GPUs:
```yaml
resources:
  limits:
    nvidia.com/gpu: 8
```

But for proper multi-node training/inference, a single pod isn't enough. Patterns:

### 4.1 Pod-per-Worker

One pod per GPU, communicating via NCCL over the pod network. Use a `StatefulSet` to get stable hostnames; init container syncs the rendezvous before launch.

### 4.2 Gang Scheduling

All-or-nothing: launch all N pods together, or none. Without it, partial allocations deadlock. Tools: Kueue (modern K8s native), Volcano, KubeRay (built-in for Ray jobs).

### 4.3 Topology-Aware Scheduling

Pack pods onto nodes that share NVLink fabric, IB rails, or NUMA domains. Without it, your TP=8 ends up split across PCIe links.

Implementations:
- **Topology Aware Volume / Pod Affinity** — labels nodes with topology, pod affinities prefer them.
- **NodeFeatureDiscovery + PodAffinity** — tag nodes by NVLink domain, prefer same-domain colocation.
- **Custom schedulers** — `nvidia.com/topology` aware, e.g. NVIDIA's K8s topology plugin.

### 4.4 Network Bandwidth Reservation

Some clusters reserve dedicated network rails per training job. Implementation: SR-IOV virtual functions, network operator pinning a HCA per pod, manifest selectors.

---

## 5. KubeRay (Ray on K8s)

Ray is a distributed compute framework heavily used for ML training, RL, and increasingly inference (Anyscale). KubeRay maps Ray clusters onto K8s resources.

```
RayCluster CRD → head pod + N worker pods
```

Each worker is a Ray Actor host. Ray handles dispatch, autoscaling, and inter-actor RPC. For LLM inference: Ray Serve hosts vLLM/TRT-LLM replicas with built-in load balancing, health checks, and request batching.

KubeRay strengths: Python-native, scales to thousands of GPUs, integrates with Tune (HPO), RLHF pipelines.

---

## 6. KServe / Seldon / NVIDIA NIM

Higher-level inference deployment platforms.

- **KServe** — open-source, multi-framework, K8s-native CRD `InferenceService`. Adds canary, transformers (preprocess), explainers, pluggable runtimes.
- **Seldon Core / Seldon Deploy** — similar territory, longer history.
- **NVIDIA NIM** — pre-built containers per model, optimized with TRT-LLM, OpenAI-compatible API. Pull, deploy, done.

Pick KServe for full control, NIM for fast time-to-deploy on standard models.

---

## 7. Inference Autoscaling

### 7.1 Horizontal Pod Autoscaler (HPA)

Standard K8s mechanism: scale replicas based on a metric (CPU, custom). For inference, the right metric is **queue depth** or **TPOT p95**, not CPU.

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metrics:
  - type: External
    external:
      metric:
        name: vllm_inference_queue_depth
      target:
        type: AverageValue
        averageValue: 10
```

### 7.2 KEDA

Kubernetes Event-Driven Autoscaling. Scale to/from zero based on Prometheus, Kafka lag, or custom queue triggers. Useful for spiky traffic.

### 7.3 Cold Starts

LLM replicas have **long start times** (10–60s for model load even from local NVMe; minutes from S3). Cold start of a replica blows latency SLOs. Mitigations:

- **Warm pools** — keep N idle replicas preloaded.
- **Local caching tier** — pre-stage weights on every node's NVMe.
- **Model server pooling** — one process hosts multiple models, swap in/out without restart.
- **GPU memory snapshot/restore** — rare; some research.

### 7.4 Right-Sizing

Track per-replica request rate, batch size, and TPOT. Add replicas before SLO breaks; remove when steady-state utilization drops below threshold.

---

## 8. Networking in K8s for AI

### 8.1 Pod Network

Default CNI plugins (Calico, Cilium, Flannel) work fine for control plane traffic, but RDMA over IB needs special setup:

- **SR-IOV CNI** — exposes virtual functions of physical HCAs to pods.
- **Multus** — multiple network interfaces per pod (control net + RDMA net).
- **NVIDIA Network Operator** — installs Mellanox/IB drivers, RDMA shared device plugin.

A typical training pod:
```yaml
metadata:
  annotations:
    k8s.v1.cni.cncf.io/networks: rdma-net
spec:
  containers:
    - name: trainer
      resources:
        limits:
          nvidia.com/gpu: 8
          rdma/hca: 1   # exposed by network operator
```

### 8.2 Service & Ingress

For inference, a `Service` exposes pod replicas; ingress (NGINX, Envoy, Istio) handles TLS, auth, rate limiting. Modern stacks add **token-aware routing** (route by prompt-prefix hash for cache locality).

### 8.3 Kubernetes Service Mesh — careful

Istio sidecars terminate connections per pod, hurting throughput on streaming workloads. Many inference deployments skip the sidecar mesh on the data plane and use it only for the control plane.

---

## 9. Storage in K8s for AI

### 9.1 Model Weights

Hot path needs sub-second access. Options:
- **NVMe PV** on the node — fastest. Synced from S3 by an init container or DaemonSet.
- **CSI driver to S3 / GCS** — convenient but slow (every cold start re-downloads).
- **Persistent shared FS** (Lustre, WekaFS via CSI) — central, hot.
- **OCI image with weights baked in** — large image, slow image pull, but no separate sync.

A common pattern: a DaemonSet pre-stages models on each node's NVMe via a sidecar that watches a model registry.

### 9.2 Datasets (training)

Streaming over network or mounted shared FS. CSI driver to Lustre/WekaFS. PVCs sized to dataset; multiple jobs share read-only mounts.

### 9.3 Checkpoints

S3 or shared FS via CSI. PyTorch DCP writes to a directory; the path is a PVC mount.

---

## 10. Failure & Recovery

K8s helps but doesn't solve everything.

- **Pod crash** — restart count rises; if it loops, K8s backs off. Watch `CrashLoopBackOff` for OOMs / NaNs.
- **Node failure** — pods evicted; controller (StatefulSet, etc.) reschedules. For training, this means losing a rank → all-or-nothing restart.
- **PriorityClass** — preempt lower-priority jobs so high-priority training lands on free GPUs quickly.
- **Job retries** — `Job.spec.backoffLimit` for batch; `RayJob` has its own retry logic.
- **Checkpoint discipline** — see [Distributed_Training §8](Distributed_Training.md).

---

## 11. Multi-Tenancy

In a shared cluster:
- **Namespaces** + **ResourceQuotas** — cap GPUs per tenant.
- **PriorityClass + preemption** — high-priority gets GPUs first.
- **NetworkPolicies** — isolate traffic between tenants.
- **Quota dashboards** — per-tenant GPU-hours reporting.

Run:ai and other commercial schedulers add fairness queueing, dynamic quota, oversubscription with preemption.

---

## 12. Common Pitfalls

- **Wrong GPU resource name**: `nvidia.com/gpu`, not `nvidia/gpu`. Pods schedule but don't get GPUs.
- **MIG enabled but workload expects full GPU**: TP=8 won't work on slices.
- **Shared HCA contention**: two pods on the same HCA → bandwidth halved silently.
- **Unbounded model image size**: 200 GB images take forever to pull. Strip pretraining artifacts; use sidecar weight downloaders.
- **Liveness probe too aggressive on inference replicas**: a long prefill blocks the probe → kill loop.
- **No topology pinning for TP**: TP=8 gets scattered across PCIe → 5× slowdown invisible to SLO until p99 explodes.
- **Forgetting hugepages**: CUDA performance suffers without 2MB hugepages enabled in some kernels.

---

## 13. Common Interview Questions

**Q: How do you schedule a TP=8 inference workload on Kubernetes?**
A: Either a single pod requesting `nvidia.com/gpu: 8` (one node), or 8 pods with topology-aware scheduling that lands them on the same node's NVLink domain. The GPU Operator + Node Feature Discovery + a topology-aware scheduler is the standard setup.

**Q: What does the NVIDIA GPU Operator do?**
A: Installs and manages all GPU-related components on each node — driver, container toolkit, device plugin, DCGM exporter, MIG manager, GPUDirect RDMA modules, NFD. Without it, fleet management of GPU drivers is a nightmare.

**Q: When would you use MIG?**
A: When you need hardware-isolated GPU slices for many small workloads (single-replica small-model serving, multi-tenant), and don't need NVLink between slices. Not for big LLM inference (one model spans full GPU and probably more).

**Q: How do you autoscale inference replicas?**
A: HPA (or KEDA) keyed on a meaningful metric — queue depth, TPOT p95, or RPS — not CPU. Maintain a warm pool to avoid cold-start latency. Pre-stage weights on each node so spin-up is sub-second.

**Q: What's the cold-start problem for LLM serving?**
A: Loading a 70B model from object store takes minutes; even from local NVMe, 10s+ to mmap and warm up. Autoscaling responds too slowly. Mitigated by warm pools, node-level weight caching, and process-level model pooling.

**Q: How does GPUDirect RDMA work in Kubernetes?**
A: NVIDIA Network Operator + Multus exposes a second network interface per pod backed by SR-IOV virtual functions of an HCA. NCCL detects the RDMA interface and uses it directly for GPU-GPU communication, bypassing CPU.

**Q: What's KServe?**
A: A Kubernetes CRD (InferenceService) for deploying models with built-in canary, transformers, and explainers. Pluggable runtimes (vLLM, TRT-LLM, ONNX). Standard at companies that want platform-as-a-service inference without writing custom controllers.

**Q: How do you give a high-priority training job free GPUs without disrupting inference?**
A: PriorityClass with preemption. The training Pod has higher priority; K8s evicts lower-priority pods to free resources. For inference, use PodDisruptionBudget to limit eviction during high traffic.

**Q: How do you debug a "GPU not visible to pod" error?**
A: Check (1) pod's resource request includes `nvidia.com/gpu`, (2) device plugin DaemonSet is healthy, (3) GPU operator components are running, (4) CUDA driver version compatible with toolkit, (5) `nvidia-smi` works on the host. Often a driver-version mismatch.

**Q: Why is Istio sidecar bad for streaming inference?**
A: Sidecars proxy every connection; on streaming SSE/WebSocket the sidecar holds the connection and incurs CPU overhead per token. Disable sidecar injection on inference data plane, keep it for control plane only.

**Q: How would you design GPU autoscaling for variable demand?**
A: HPA on TPOT p95 + queue depth, with min replicas to absorb baseline traffic, warm pool sized to expected step-up, and aggressive scale-up policy (small step, fast trigger). Keep node pools warm via cluster autoscaler with min nodes set to expected baseline + buffer.

**Q: What's the Run:ai (now NVIDIA) value-add over vanilla K8s?**
A: GPU-aware fairness scheduling, dynamic quotas, fractional GPU sharing without MIG, gang scheduling, queue prioritization, multi-tenant isolation. Vanilla K8s lacks these out of the box.

**Q: Why is Volcano or Kueue used for training jobs?**
A: Native K8s doesn't gang-schedule (all-or-nothing). A training job spread across 64 pods would deadlock if half don't get scheduled. Volcano/Kueue ensure all pods come up together or none do.

---

## 14. Further Reading

- NVIDIA GPU Operator docs.
- KServe documentation.
- NVIDIA NIM blog and product pages.
- Kueue docs (Kubernetes SIG-scheduling).
- KubeRay docs.
- Anyscale Ray + Kubernetes guides.
- "Operating LLMs at scale" — various NVIDIA, Mosaic, Anyscale blog posts.

---

**Next:** [Observability_and_Debugging](Observability_and_Debugging.md).
**See also:** [Production_Architecture](Production_Architecture.md), [Networking_and_Interconnect](Networking_and_Interconnect.md), [Inference_Frameworks](Inference_Frameworks.md).

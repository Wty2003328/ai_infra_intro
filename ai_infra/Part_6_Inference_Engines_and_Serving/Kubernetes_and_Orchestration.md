# Kubernetes and Orchestration

How LLM workloads actually run in production fleets: the GPU operator, MIG, scheduling for multi-GPU jobs, autoscaling for inference, and the cross-cutting tools (KubeRay, KServe, NIM, NVIDIA Run:ai, Skypilot). This page focuses on what an MLSys engineer needs to design and debug, not full Kubernetes from scratch.

**Prerequisites**: [[Networking_and_Interconnect]], [[Inference_Frameworks]], [[Production_Architecture]].

---

## 1. Why Kubernetes for AI

LLM inference and training run on heterogeneous fleets (many GPU types, network rails, data tiers). Kubernetes provides:

- **Schedulability** — bin-packing workloads onto nodes with constraints (GPU type, NUMA, network).
- **Lifecycle** — restart on crash, rolling upgrade, version pinning.
- **Service discovery** — clients find live replicas without hardcoded IPs.
- **Autoscaling** — scale replicas based on metrics (RPS, queue depth, GPU util).
- **Multi-tenancy** — namespaces, RBAC, quotas.

Production at scale almost always builds on K8s, even when the "GPU job" feels alien to web-app patterns.

### 1.1 K8s vs. Alternative Orchestrators for AI

While Kubernetes dominates, it is worth understanding when alternatives apply:

| Orchestrator | Strengths | Weaknesses |
|---|---|---|
| Kubernetes | Ecosystem, multi-tenancy, mature scheduling | Complexity, GPU support requires add-ons |
| SLURM | Native GPU/MPI support, HPC heritage, simple for training | Weak multi-tenancy, no autoscaling, poor for serving |
| Nomad (HashiCorp) | Simpler than K8s, device plugins | Smaller ecosystem, fewer GPU integrations |
| Ray (standalone) | Python-native, elastic, built-in ML primitives | Less mature for non-ML workloads |

Most production LLM platforms use K8s for serving and increasingly for training, with SLURM remaining common in pure-research HPC environments.

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

Without the GPU Operator, you're hand-managing driver versions across a fleet -- painful at scale.

```yaml
# Pod requesting 1 GPU
resources:
  limits:
    nvidia.com/gpu: 1
```

### 2.1 Operator Lifecycle and Upgrade Strategy

The GPU Operator itself runs as a Helm chart. Upgrades are node-rolling: the operator cordons a node, drains GPU workloads, installs the new driver, uncordons. Key operational concerns:

- **Driver-toolkit compatibility matrix**: every CUDA version requires a minimum driver version. The operator enforces this, but mismatches between operator version and the underlying OS kernel can block upgrades.
- **Validation pod**: after each driver install, the operator runs a validation container that exercises `nvidia-smi` and basic CUDA calls. If validation fails, the node stays cordoned.
- **Rollback**: pin a known-good operator version in Helm values; if a new driver causes regressions, revert the Helm release.

### 2.2 GPU Operator Component Interaction

```
┌──────────────────────────────────────────────────────────┐
│  Node                                                     │
│                                                           │
│  ┌──────────────┐   ┌────────────────┐   ┌─────────────┐ │
│  │ NVIDIA Driver │   │ Container      │   │ Device      │ │
│  │ (kernel mod)  │──►│ Toolkit (hook) │──►│ Plugin      │ │
│  └──────────────┘   └────────────────┘   │ (exposes    │ │
│                                           │  resource)  │ │
│  ┌──────────────┐   ┌────────────────┐   └──────┬──────┘ │
│  │ MIG Manager   │   │ DCGM Exporter  │          │        │
│  └──────────────┘   └────────────────┘    kubelet │        │
│                                                   ▼        │
│                                            Pod gets GPU    │
└──────────────────────────────────────────────────────────┘
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
- Need NVLink (MIG slices have **no inter-slice NVLink** -- they live in a single GPU only).

The GPU operator's MIG manager configures profiles per node.

### 3.1 MIG Capacity Planning

When partitioning GPUs with MIG, compute the effective throughput per slice. For a 1g.10gb slice on H100:

- Compute: $\frac{14}{132}$ of total SMs $\approx 10.6\%$ of peak TFLOPS
- Memory bandwidth: $\frac{10}{80}$ of HBM $\approx 12.5\%$ of peak BW

For a small model (e.g., 7B FP8 $\approx 7$ GB weights) on a 1g.10gb slice:

$$\text{Decode throughput (BW-bound)} \approx \frac{0.125 \times 3.35 \text{ TB/s}}{7 \text{ GB}} \approx 60 \text{ tokens/s}$$

This is sufficient for low-QPS single-user serving but not for high-throughput batched workloads.

### 3.2 MIG vs. Time-Slicing vs. MPS

| Approach | Isolation | Performance | Use case |
|---|---|---|---|
| MIG | Hardware (memory + compute) | Predictable, no interference | Multi-tenant production |
| Time-slicing | None (context switching) | Unpredictable, noisy-neighbor | Dev/test only |
| MPS (Multi-Process Service) | Partial (shared memory space) | Good for cooperative workloads | Same-team batch jobs |

MIG is the only option that provides true hardware isolation suitable for production multi-tenant workloads.

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

**Example StatefulSet pattern for distributed training:**

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: training-workers
spec:
  replicas: 8
  serviceName: "workers"
  selector:
    matchLabels:
      app: training
  template:
    metadata:
      labels:
        app: training
    spec:
      containers:
      - name: trainer
        image: my-training:latest
        env:
        - name: MASTER_ADDR
          value: "training-workers-0.workers"
        - name: WORLD_SIZE
          value: "8"
        - name: RANK
          valueFrom:
            fieldRef:
              fieldPath: metadata.labels['apps.kubernetes.io/pod-index']
        resources:
          limits:
            nvidia.com/gpu: 1
            rdma/hca: 1
```

### 4.2 Gang Scheduling

All-or-nothing: launch all N pods together, or none. Without it, partial allocations deadlock. Tools: Kueue (modern K8s native), Volcano, KubeRay (built-in for Ray jobs).

**The deadlock problem without gang scheduling:**

Consider a 64-GPU training job on a cluster with 80 free GPUs. Two such jobs are submitted simultaneously. Without gang scheduling, each job might get 40 GPUs, but neither can proceed because both need 64. This is a resource deadlock. Gang schedulers either allocate all 64 at once or queue the job.

**Kueue vs. Volcano comparison:**

| Feature | Kueue | Volcano |
|---|---|---|
| K8s native | Yes (SIG-scheduling) | CRD-based |
| Gang scheduling | Yes | Yes |
| Fair sharing | Cohorts + ClusterQueues | Queue-level |
| Preemption | Priority-based | Priority-based |
| Multi-tenant quotas | Built-in | Limited |
| Maturity | Newer, actively developed | Older, stable |

### 4.3 Topology-Aware Scheduling

Pack pods onto nodes that share NVLink fabric, IB rails, or NUMA domains. Without it, your TP=8 ends up split across PCIe links.

**Quantifying the penalty**: NVLink 4 provides $\sim 900$ GB/s bidirectional, while PCIe Gen5 x16 provides $64$ GB/s. An all-reduce across 8 GPUs via PCIe instead of NVLink is:

$$\text{Slowdown factor} \approx \frac{900}{64} \approx 14\times \text{ less bandwidth}$$

For TP=8, where every forward pass requires an all-reduce, this turns a 2 ms collective into a 28 ms collective -- catastrophic for TPOT.

Implementations:
- **Topology Aware Volume / Pod Affinity** — labels nodes with topology, pod affinities prefer them.
- **NodeFeatureDiscovery + PodAffinity** — tag nodes by NVLink domain, prefer same-domain colocation.
- **Custom schedulers** — `nvidia.com/topology` aware, e.g. NVIDIA's K8s topology plugin.

### 4.4 Network Bandwidth Reservation

Some clusters reserve dedicated network rails per training job. Implementation: SR-IOV virtual functions, network operator pinning a HCA per pod, manifest selectors.

**Bandwidth calculation for DP=128 training of a 70B model with FSDP:**

Each gradient all-reduce exchanges $\approx 2 \times$ the model size (reduce-scatter + all-gather):

$$\text{Data per step} = 2 \times 70 \text{ GB (FP16)} = 280 \text{ GB}$$

With NDR IB at 50 GB/s per link, using a ring all-reduce across 128 nodes:

$$\text{All-reduce time} \approx \frac{2 \times (128 - 1)}{128} \times \frac{280 \text{ GB}}{50 \text{ GB/s}} \approx 11 \text{ s}$$

This must overlap with backward compute to hide the latency, which is why gradient bucketing and communication-computation overlap are critical.

---

## 5. KubeRay (Ray on K8s)

Ray is a distributed compute framework heavily used for ML training, RL, and increasingly inference (Anyscale). KubeRay maps Ray clusters onto K8s resources.

```
RayCluster CRD → head pod + N worker pods
```

Each worker is a Ray Actor host. Ray handles dispatch, autoscaling, and inter-actor RPC. For LLM inference: Ray Serve hosts vLLM/TRT-LLM replicas with built-in load balancing, health checks, and request batching.

KubeRay strengths: Python-native, scales to thousands of GPUs, integrates with Tune (HPO), RLHF pipelines.

### 5.1 KubeRay Architecture Deep Dive

```
┌────────────────────────────────────────────────────┐
│  KubeRay Operator (control plane)                   │
│   - Watches RayCluster / RayJob / RayService CRDs   │
│   - Manages head + worker pod lifecycle              │
└──────────────┬─────────────────────────────────────┘
               │ creates/manages
               ▼
┌────────────────────────────────────────────────────┐
│  RayCluster                                         │
│   ┌──────────┐   ┌─────────┐   ┌─────────┐        │
│   │ Head Pod  │   │ Worker  │   │ Worker  │  ...   │
│   │ (GCS,     │   │ (GPU×8) │   │ (GPU×8) │        │
│   │  dashboard│   │         │   │         │        │
│   │  driver)  │   │ vLLM    │   │ vLLM    │        │
│   └──────────┘   │ replica  │   │ replica │        │
│                   └─────────┘   └─────────┘        │
└────────────────────────────────────────────────────┘
```

**RayService** is the CRD for serving: it creates a RayCluster and deploys a Ray Serve application. Key advantage: the Ray autoscaler can scale individual serve replicas (e.g., add more vLLM instances) without scaling the underlying K8s pods, providing finer-grained control.

### 5.2 KubeRay for RLHF Pipelines

Ray is especially strong for RLHF/GRPO because a single Ray cluster can orchestrate:
1. **Inference actor group**: generates rollouts using vLLM.
2. **Reward model actor group**: scores rollouts.
3. **Training actor group**: runs PPO/GRPO updates.

All three share the same cluster, with Ray handling data movement between actor groups via object store, avoiding expensive serialization to external storage.

---

## 6. KServe / Seldon / NVIDIA NIM

Higher-level inference deployment platforms.

- **KServe** — open-source, multi-framework, K8s-native CRD `InferenceService`. Adds canary, transformers (preprocess), explainers, pluggable runtimes.
- **Seldon Core / Seldon Deploy** — similar territory, longer history.
- **NVIDIA NIM** — pre-built containers per model, optimized with TRT-LLM, OpenAI-compatible API. Pull, deploy, done.

Pick KServe for full control, NIM for fast time-to-deploy on standard models.

### 6.1 KServe InferenceService Example

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: llama-70b
spec:
  predictor:
    model:
      modelFormat:
        name: vllm
      runtime: kserve-vllm
      storageUri: "s3://models/llama-70b-fp8"
      resources:
        limits:
          nvidia.com/gpu: 8
    minReplicas: 2
    maxReplicas: 20
    scaleTarget: 10  # target concurrent requests
  transformer:
    containers:
    - name: tokenizer
      image: my-tokenizer:latest
```

KServe handles:
- **Scale-to-zero** (if configured) and scale-from-zero with serverless.
- **Canary rollouts**: split traffic between revisions.
- **Transformer pipeline**: pre/post-processing containers chained before/after the predictor.

### 6.2 NIM vs. Custom Deployment Decision Matrix

| Factor | NVIDIA NIM | Custom (vLLM/TRT-LLM on K8s) |
|---|---|---|
| Time to deploy | Hours | Days to weeks |
| Model coverage | Standard models only | Any model |
| Optimization control | Black-box | Full control |
| Cost | License fee | Engineering time |
| Customization | Limited | Unlimited |
| Updates | Tied to NIM releases | Your cadence |

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

**KEDA ScaledObject example for Prometheus-driven scaling:**

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: vllm-scaler
spec:
  scaleTargetRef:
    name: vllm-deployment
  minReplicaCount: 2
  maxReplicaCount: 50
  triggers:
  - type: prometheus
    metadata:
      serverAddress: http://prometheus:9090
      metricName: vllm_tpot_p95
      query: |
        histogram_quantile(0.95,
          rate(vllm_request_latency_tpot_seconds_bucket[5m]))
      threshold: "0.060"  # 60ms TPOT p95 target
```

### 7.3 Cold Starts

LLM replicas have **long start times** (10--60 s for model load even from local NVMe; minutes from S3). Cold start of a replica blows latency SLOs. Mitigations:

- **Warm pools** — keep $N$ idle replicas preloaded.
- **Local caching tier** — pre-stage weights on every node's NVMe.
- **Model server pooling** — one process hosts multiple models, swap in/out without restart.
- **GPU memory snapshot/restore** — rare; some research.

**Sizing the warm pool:**

Given traffic that can spike by $S\times$ from baseline within $t$ seconds, and new replica startup time $T_{\text{start}}$:

$$N_{\text{warm}} = \left\lceil \frac{(S - 1) \cdot R_{\text{baseline}}}{R_{\text{per\_replica}}} \cdot \frac{T_{\text{start}}}{T_{\text{start}} - t} \right\rceil$$

where $R_{\text{baseline}}$ is baseline RPS and $R_{\text{per\_replica}}$ is per-replica capacity. In practice, teams often use a simpler heuristic: warm pool = 20--30% of peak replica count.

### 7.4 Right-Sizing

Track per-replica request rate, batch size, and TPOT. Add replicas before SLO breaks; remove when steady-state utilization drops below threshold.

**Utilization-based capacity formula:**

$$\text{Replicas}_{\text{target}} = \left\lceil \frac{\text{RPS}_{\text{current}}}{\text{RPS}_{\text{per\_replica}} \times U_{\text{target}}} \right\rceil + N_{\text{warm}}$$

where $U_{\text{target}} \approx 0.7$--$0.8$ provides headroom for bursts without SLO violations.

### 7.5 Autoscaling Dynamics and Stability

A poorly tuned autoscaler oscillates (flapping). Key parameters:

- **Scale-up policy**: react within 15--30 s to queue depth spikes. Use `stabilizationWindowSeconds: 0` for fast response.
- **Scale-down policy**: use a longer stabilization window (5--10 min) to avoid premature scale-down.
- **Behavior asymmetry**: scale up aggressively (percent or pods-per-minute), scale down conservatively.

```yaml
behavior:
  scaleUp:
    stabilizationWindowSeconds: 0
    policies:
    - type: Percent
      value: 100
      periodSeconds: 60
  scaleDown:
    stabilizationWindowSeconds: 300
    policies:
    - type: Pods
      value: 1
      periodSeconds: 120
```

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

### 8.3 Kubernetes Service Mesh -- careful

Istio sidecars terminate connections per pod, hurting throughput on streaming workloads. Many inference deployments skip the sidecar mesh on the data plane and use it only for the control plane.

**Measured overhead**: Istio sidecar adds 1--3 ms per-hop latency and 5--15% CPU overhead per pod. For streaming inference where each token traverses the proxy, this compounds to measurable TPOT regression.

**Recommended pattern**: annotate inference pods with `sidecar.istio.io/inject: "false"` and handle mTLS at the ingress gateway instead.

---

## 9. Storage in K8s for AI

### 9.1 Model Weights

Hot path needs sub-second access. Options:
- **NVMe PV** on the node — fastest. Synced from S3 by an init container or DaemonSet.
- **CSI driver to S3 / GCS** — convenient but slow (every cold start re-downloads).
- **Persistent shared FS** (Lustre, WekaFS via CSI) — central, hot.
- **OCI image with weights baked in** — large image, slow image pull, but no separate sync.

A common pattern: a DaemonSet pre-stages models on each node's NVMe via a sidecar that watches a model registry.

**Model load time comparison for a 70B FP8 model (70 GB):**

| Source | Bandwidth | Load time |
|---|---|---|
| Local NVMe (Gen4) | $\sim 7$ GB/s | $\sim 10$ s |
| Lustre/WekaFS | $\sim 2$--$5$ GB/s | $14$--$35$ s |
| S3 (single stream) | $\sim 0.5$ GB/s | $\sim 140$ s |
| S3 (multi-stream, 64 threads) | $\sim 10$ GB/s | $\sim 7$ s |

### 9.2 Datasets (training)

Streaming over network or mounted shared FS. CSI driver to Lustre/WekaFS. PVCs sized to dataset; multiple jobs share read-only mounts.

### 9.3 Checkpoints

S3 or shared FS via CSI. PyTorch DCP writes to a directory; the path is a PVC mount.

**Checkpoint size for common models:**

$$\text{Checkpoint size} \approx \text{params} \times \text{bytes per param} \times (1 + \text{optimizer states})$$

For a 70B model with AdamW (2 optimizer states in FP32, weights in BF16):

$$70\text{B} \times 2 \text{ bytes} + 70\text{B} \times 4 \text{ bytes} \times 2 = 140 + 560 = 700 \text{ GB}$$

With FSDP sharded checkpointing, each rank writes $\frac{700}{N_{\text{ranks}}}$ GB, enabling parallel I/O.

---

## 10. Failure & Recovery

K8s helps but doesn't solve everything.

- **Pod crash** — restart count rises; if it loops, K8s backs off. Watch `CrashLoopBackOff` for OOMs / NaNs.
- **Node failure** — pods evicted; controller (StatefulSet, etc.) reschedules. For training, this means losing a rank -> all-or-nothing restart.
- **PriorityClass** — preempt lower-priority jobs so high-priority training lands on free GPUs quickly.
- **Job retries** — `Job.spec.backoffLimit` for batch; `RayJob` has its own retry logic.
- **Checkpoint discipline** — see [[Distributed_Training]].

### 10.1 Failure Rate Math for Large Clusters

For a cluster with $N$ GPUs, each with per-day failure probability $p$:

$$P(\text{at least one failure per day}) = 1 - (1 - p)^N$$

With $p = 0.001$ (typical) and $N = 1024$:

$$P = 1 - (1 - 0.001)^{1024} = 1 - 0.999^{1024} \approx 1 - e^{-1.024} \approx 0.64$$

So on a 1024-GPU cluster, expect a failure roughly every 1.5 days. This is why checkpoint frequency matters -- with hourly checkpoints, you lose at most 1 hour of work per failure. The expected wasted compute fraction is:

$$\text{Waste} \approx \frac{T_{\text{checkpoint\_interval}}}{2 \times T_{\text{mean\_time\_between\_failures}}}$$

### 10.2 PodDisruptionBudget for Inference

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: inference-pdb
spec:
  minAvailable: "80%"
  selector:
    matchLabels:
      app: vllm-inference
```

This ensures that during voluntary disruptions (node upgrades, scaling), at least 80% of replicas stay available, preventing SLO violations.

---

## 11. Multi-Tenancy

In a shared cluster:
- **Namespaces** + **ResourceQuotas** — cap GPUs per tenant.
- **PriorityClass + preemption** — high-priority gets GPUs first.
- **NetworkPolicies** — isolate traffic between tenants.
- **Quota dashboards** — per-tenant GPU-hours reporting.

Run:ai and other commercial schedulers add fairness queueing, dynamic quota, oversubscription with preemption.

### 11.1 GPU Quota Example

```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: team-a-gpu-quota
  namespace: team-a
spec:
  hard:
    requests.nvidia.com/gpu: "32"
    limits.nvidia.com/gpu: "32"
    persistentvolumeclaims: "10"
```

### 11.2 Fair-Share Scheduling

Without fairness, a tenant submitting many jobs can monopolize the cluster. Fair-share schedulers track cumulative GPU-hours per tenant and prioritize under-served tenants:

$$\text{Priority}(t) = \frac{\text{Quota}(t)}{\text{Used}(t) + \epsilon}$$

Tenants who have used less than their fair share get higher priority. Kueue's `ClusterQueue` with `BorrowingLimit` and `Cohort` implements this natively.

---

## 12. Common Pitfalls

- **Wrong GPU resource name**: `nvidia.com/gpu`, not `nvidia/gpu`. Pods schedule but don't get GPUs.
- **MIG enabled but workload expects full GPU**: TP=8 won't work on slices.
- **Shared HCA contention**: two pods on the same HCA -> bandwidth halved silently.
- **Unbounded model image size**: 200 GB images take forever to pull. Strip pretraining artifacts; use sidecar weight downloaders.
- **Liveness probe too aggressive on inference replicas**: a long prefill blocks the probe -> kill loop.
- **No topology pinning for TP**: TP=8 gets scattered across PCIe -> $5\times$ slowdown invisible to SLO until p99 explodes.
- **Forgetting hugepages**: CUDA performance suffers without 2MB hugepages enabled in some kernels.
- **Missing `terminationGracePeriodSeconds`**: default 30 s is too short for an LLM server draining in-flight requests. Set to 120--300 s.
- **No resource requests (only limits)**: scheduler cannot bin-pack properly; GPUs wasted.

---

## 13. Common Interview Questions

**Q: How do you schedule a TP=8 inference workload on Kubernetes?**
A: Either a single pod requesting `nvidia.com/gpu: 8` (one node), or 8 pods with topology-aware scheduling that lands them on the same node's NVLink domain. The GPU Operator + Node Feature Discovery + a topology-aware scheduler is the standard setup. The single-pod approach is simpler and preferred for TP within a node. Multi-pod is needed when TP spans nodes (rare for inference, common for training with PP). Verify with `nvidia-smi topo -m` that the 8 GPUs are actually connected via NVLink, not PCIe.

**Q: What does the NVIDIA GPU Operator do?**
A: Installs and manages all GPU-related components on each node -- driver, container toolkit, device plugin, DCGM exporter, MIG manager, GPUDirect RDMA modules, NFD. Without it, fleet management of GPU drivers is a nightmare. It runs as a set of DaemonSets orchestrated by a controller. Upgrades are rolling and validated per-node. The operator also handles MIG profile configuration and can reconfigure profiles without node restart.

**Q: When would you use MIG?**
A: When you need hardware-isolated GPU slices for many small workloads (single-replica small-model serving, multi-tenant), and don't need NVLink between slices. Not for big LLM inference (one model spans full GPU and probably more). MIG is also useful for dev/test environments where you want to give each developer a guaranteed GPU slice. Each slice gets its own memory controller, cache partition, and compute units -- true hardware isolation, not time-sharing.

**Q: How do you autoscale inference replicas?**
A: HPA (or KEDA) keyed on a meaningful metric -- queue depth, TPOT p95, or RPS -- not CPU. Maintain a warm pool to avoid cold-start latency. Pre-stage weights on each node so spin-up is sub-second. Scale-up policy should be aggressive (react in 15--30 s), scale-down should be conservative (5--10 min stabilization) to prevent flapping. The key formula: $\text{Replicas} = \lceil \text{RPS} / (\text{RPS}_{\text{per\_replica}} \times U_{\text{target}}) \rceil + N_{\text{warm}}$.

**Q: What's the cold-start problem for LLM serving?**
A: Loading a 70B model from object store takes minutes; even from local NVMe, 10 s+ to mmap and warm up. Autoscaling responds too slowly. Mitigated by warm pools, node-level weight caching, and process-level model pooling. Quantitatively: 70 GB from NVMe at 7 GB/s = 10 s; from S3 single-stream at 0.5 GB/s = 140 s. Multi-stream S3 with 64 threads can achieve 7--10 s. The best strategy is a DaemonSet that pre-stages hot models on every node's NVMe.

**Q: How does GPUDirect RDMA work in Kubernetes?**
A: NVIDIA Network Operator + Multus exposes a second network interface per pod backed by SR-IOV virtual functions of an HCA. NCCL detects the RDMA interface and uses it directly for GPU-GPU communication, bypassing CPU. The pod manifest requests `rdma/hca: 1` as a resource. The network operator also installs the necessary kernel modules (mlx5_core, ib_uverbs) and device plugins. Key gotcha: ensure the GPU and HCA are on the same PCIe switch (NUMA-aware placement) for optimal DMA performance.

**Q: What's KServe?**
A: A Kubernetes CRD (InferenceService) for deploying models with built-in canary, transformers, and explainers. Pluggable runtimes (vLLM, TRT-LLM, ONNX). Standard at companies that want platform-as-a-service inference without writing custom controllers. Key features include scale-to-zero (for cost), autoscaling, traffic splitting between revisions, and a standard predict/explain API.

**Q: How do you give a high-priority training job free GPUs without disrupting inference?**
A: PriorityClass with preemption. The training Pod has higher priority; K8s evicts lower-priority pods to free resources. For inference, use PodDisruptionBudget to limit eviction during high traffic. Specifically: inference gets PDB with `minAvailable: 80%`, training gets a higher PriorityClass. The scheduler will only preempt inference pods if it can still satisfy the PDB. For truly critical inference, use dedicated node pools with taints that training cannot tolerate.

**Q: How do you debug a "GPU not visible to pod" error?**
A: Check (1) pod's resource request includes `nvidia.com/gpu`, (2) device plugin DaemonSet is healthy, (3) GPU operator components are running, (4) CUDA driver version compatible with toolkit, (5) `nvidia-smi` works on the host. Often a driver-version mismatch. Also check: (6) node has allocatable GPUs (`kubectl describe node | grep nvidia.com/gpu`), (7) MIG is not enabled when full GPUs are expected, (8) the container runtime is configured to use the NVIDIA runtime hook.

**Q: Why is Istio sidecar bad for streaming inference?**
A: Sidecars proxy every connection; on streaming SSE/WebSocket the sidecar holds the connection and incurs CPU overhead per token. Measured overhead: 1--3 ms per hop and 5--15% CPU per pod. For a 200-token response, that is 200--600 ms of added latency. Disable sidecar injection on inference data plane, keep it for control plane only.

**Q: How would you design GPU autoscaling for variable demand?**
A: HPA on TPOT p95 + queue depth, with min replicas to absorb baseline traffic, warm pool sized to expected step-up, and aggressive scale-up policy (small step, fast trigger). Keep node pools warm via cluster autoscaler with min nodes set to expected baseline + buffer. Scale-down with hysteresis (5--10 min window) to prevent flapping. For cost optimization, combine warm GPU nodes (fast scale-up) with cluster autoscaler (slower, adds new nodes) as a two-tier approach.

**Q: What's the Run:ai (now NVIDIA) value-add over vanilla K8s?**
A: GPU-aware fairness scheduling, dynamic quotas, fractional GPU sharing without MIG, gang scheduling, queue prioritization, multi-tenant isolation. Vanilla K8s lacks these out of the box. Key differentiator: Run:ai can oversubscribe GPU quotas -- if Team A isn't using their 32 GPUs, Team B can borrow them, with automatic preemption when Team A needs them back. This dramatically improves cluster utilization from typical 30--40% to 70--80%.

**Q: Why is Volcano or Kueue used for training jobs?**
A: Native K8s doesn't gang-schedule (all-or-nothing). A training job spread across 64 pods would deadlock if half don't get scheduled. Volcano/Kueue ensure all pods come up together or none do. Additionally, Kueue provides multi-tenant fair-sharing via ClusterQueues and Cohorts, priority-based preemption, and resource borrowing between teams -- features essential for shared GPU clusters.

**Q: How do you handle node maintenance on a GPU cluster running inference?**
A: (1) Cordon the node to prevent new pods. (2) Respect PDB -- drain will wait until min replicas are satisfied on other nodes. (3) For inference pods, set `terminationGracePeriodSeconds` to 120--300 s to allow in-flight requests to complete. (4) Use pre-stop hooks to signal the engine to stop accepting new requests and drain the queue. (5) Once drained, the pod terminates and the node is free for maintenance. (6) Uncordon and let the autoscaler repopulate.

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

**Next:** [[Observability_and_Debugging]].
**See also:** [[Production_Architecture]], [[Networking_and_Interconnect]], [[Inference_Frameworks]].

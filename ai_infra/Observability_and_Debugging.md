# Observability and Debugging

What to instrument and how to read it when something's wrong. Covers metrics (request, system, GPU), profiling tools (Nsight Systems / Compute, PyTorch Profiler), distributed-system debugging (NCCL, hangs, stragglers), and the practical hierarchy of "where to look first."

**Prerequisites**: [Inference_Frameworks](Inference_Frameworks.md), [Distributed_Training](Distributed_Training.md), [Collectives_and_NCCL](Collectives_and_NCCL.md).

---

## 1. The Three Levels of Observability

| Level | Question | Tools |
|-------|----------|-------|
| Application | Are users happy? | Prometheus metrics, OpenTelemetry traces, request logs |
| System | Are GPUs / network / storage healthy? | DCGM, `nvidia-smi`, ibstat, node_exporter |
| Kernel | Is this code efficient? | Nsight Systems, Nsight Compute, PyTorch Profiler |

Most production debugging starts at the top and drills down. Most kernel optimization starts at the bottom.

---

## 2. Application-Level Metrics for Inference

The metrics that matter for an LLM serving system:

### 2.1 Request Latency

- **TTFT** (Time To First Token) — admit + prefill + first decode step.
- **TPOT** (Time Per Output Token) — average decode-step latency.
- **End-to-end latency** — TTFT + TPOT × output_tokens.

Always track p50, p95, p99 — averages hide tail issues.

### 2.2 Throughput

- **Requests/sec** (RPS).
- **Tokens/sec** — input, output, total.
- **Throughput per replica** vs aggregate.

### 2.3 Engine State

- **Active sequences** (in step batch).
- **Waiting queue depth**.
- **KV cache occupancy %**.
- **Prefix cache hit rate** (block-level).
- **Preemption rate**.
- **Speculative decode acceptance rate** (if enabled).

### 2.4 SLO Tracking

For each priority class:
- Was the per-request SLO met? (boolean per request)
- Aggregate SLO violation rate.
- Burn-down rate against monthly error budget.

### 2.5 Sample Prometheus Layout

```
vllm_request_latency_ttft_seconds{model, replica}        Histogram
vllm_request_latency_tpot_seconds{model, replica}        Histogram
vllm_requests_running{replica}                           Gauge
vllm_requests_waiting{replica}                           Gauge
vllm_kv_cache_usage_perc{replica}                        Gauge
vllm_prefix_cache_hit_rate{replica}                      Gauge
vllm_tokens_generated_total{model, replica}              Counter
vllm_request_failures_total{reason}                      Counter
```

vLLM, SGLang, TRT-LLM, TGI all expose Prometheus endpoints with similar metrics.

---

## 3. System-Level Metrics

### 3.1 GPU (DCGM)

NVIDIA Data Center GPU Manager exports per-GPU:
- **SM utilization** (DCGM_FI_DEV_SM_ACTIVE)
- **Memory utilization** (DCGM_FI_DEV_MEM_COPY_UTIL)
- **HBM bandwidth used** (DCGM_FI_PROF_DRAM_ACTIVE)
- **Tensor core utilization** (DCGM_FI_PROF_PIPE_TENSOR_ACTIVE)
- **Power draw**, **temperature**, **clock**.
- **PCIe TX/RX BW**.
- **NVLink TX/RX**.

Reading the metrics:
- High SM utilization but low Tensor core util → kernel not using tensor cores.
- High memory util but low SM util → memory-bound kernel; expected for decode.
- Spiky utilization → scheduling stalls.

### 3.2 Network (IB/RoCE)

- `ibstat` — link state, port speeds.
- `perfquery` — port-level counters (xmit, recv, errors, drops).
- `ibdiagnet` — fabric-wide health scan.
- `nccl-tests` — achieved NCCL BW.

### 3.3 Storage

- `iostat`, `nvme list`, `lustre stat`.
- Per-PV throughput dashboards.

### 3.4 Node

- CPU utilization (especially for tokenizer, scheduler).
- RAM (KV swap targets, dataloader buffers).
- Hugepages.

---

## 4. PyTorch Profiler

The first profiling tool for any PyTorch workload.

```python
import torch
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=tensorboard_trace_handler("./logs"),
    record_shapes=True,
    with_stack=True,
) as prof:
    for step in range(10):
        train_step()
        prof.step()
```

Outputs:
- Per-op CPU and CUDA time.
- Memory timeline.
- Stack traces.
- Visualize in Chrome `chrome://tracing` or TensorBoard.

What to look for:
- **Big `cudaLaunchKernel` blocks** — kernel launch overhead (host bound). Fix with CUDA graphs or fusion.
- **Long matmul kernel times** — expected; verify hits tensor cores.
- **Many short kernels** — fuse them.
- **Idle GPU gaps** — find what the host is doing (dataloader? logging? optimizer?).

---

## 5. Nsight Systems

System-wide timeline. Best tool for finding orchestration bugs.

```bash
nsys profile -o myrun \
  --trace=cuda,nvtx,osrt,nccl \
  python train.py
```

Open in Nsight Systems UI. See:
- CUDA stream timeline per GPU.
- NCCL collective markers.
- CPU thread activity, including dataloader workers.
- Memory transfers.

What problems Nsys exposes:
- NCCL not overlapping with backward (sequential bars).
- Dataloader stalls (GPU idle while CPU works).
- Mis-ordered streams.
- Excessive `cudaStreamSynchronize` calls.

Use NVTX ranges to label your code regions:
```python
import torch.cuda.nvtx as nvtx
with nvtx.range("forward_pass"):
    out = model(x)
```
These appear in the Nsys timeline.

---

## 6. Nsight Compute

Per-kernel deep dive. Use after Nsys identifies a slow kernel.

```bash
ncu --set full -o profile \
  --launch-skip 5 --launch-count 1 \
  python kernel_test.py
```

Outputs:
- **Roofline analysis** — where the kernel sits vs peak.
- **Memory throughput** — % of HBM BW used.
- **Compute throughput** — % of tensor core / FP32 / FP64 peak.
- **Warp scheduler stats** — issue/active/eligible warps; stall reasons (memory, compute, sync, etc.).
- **Occupancy** — achieved vs theoretical.
- **Source view** — line-by-line cost.

What problems Ncu exposes:
- **Memory dependency stalls** — coalesce, prefetch, use TMA.
- **Long scoreboard stalls** — instruction dependencies; use `cp.async`.
- **Low occupancy from register pressure** — reduce per-thread state.
- **Non-tensor-core matmul** — verify dtype and tile shape.

---

## 7. Distributed Debugging

### 7.1 Hangs

Most painful failure mode. Symptoms: training stops emitting logs, NCCL timeout fires after 10–30 min.

Diagnostic steps:
1. **Per-rank stack traces** — `py-spy dump --pid <each rank's PID>` — find where each rank is stuck.
2. **NCCL_DEBUG=INFO** logs — identifies which collective hung.
3. **Compare ranks** — which rank is "ahead" or "behind"? Often a single straggler.
4. **GPU state** — `nvidia-smi`: any GPU wedged? ECC errors?
5. **NIC state** — `ibstat`: link down on one node?

Common causes:
- One rank threw an exception in a custom op; others wait at the next collective forever.
- Dataloader hang on a single rank (shard read failure).
- NIC link flap (transient IB issue).
- Mismatched collective parameters (different ranks call all-reduce with different sizes — NCCL hangs).

Mitigations:
- Wrap forward in a watchdog that prints `py-spy` traces if step > N×median.
- Set `NCCL_TIMEOUT_MS` aggressive enough to fail fast (default 30 min is long).
- Use `torch.distributed.barrier` strategically to flush state.

### 7.2 Stragglers

Identical work, one rank slower. Symptoms: step time variance >2×, all-reduce takes longer than expected.

Diagnose:
- DCGM utilization per rank.
- Per-rank step time histogram.
- Hardware events (clock throttling, ECC).

Common causes:
- Thermal throttling on one node.
- ECC errors silently slowing memory.
- A different MIG profile / GPU model in the fleet.
- NIC slowness (older firmware, congestion).

Cure: identify the slow node, evacuate, replace.

### 7.3 NaN / Loss Spikes

Loss goes to NaN or jumps suddenly.

Diagnose:
- Per-layer activation max/min logging.
- Per-parameter gradient norm logging.
- Save the offending step's input data — repro on small scale.

Common causes:
- Bad data sample (very long or weirdly tokenized).
- FP16 overflow (use BF16).
- Numerical instability in custom kernel (e.g. softmax without subtracting max).
- LR too high after warmup, or warmup too short.

Cure: gradient clipping (always), data validation, restart from last checkpoint.

---

## 8. Inference-Specific Debugging

### 8.1 Sudden TPOT Regression

1. Was a config changed? (engine version, precision, scheduler params)
2. Look at KV occupancy and prefix cache hit rate over time.
3. Did input distribution shift? (longer prompts, different LoRAs)
4. Check NCCL bus-bw via nccl-tests.
5. nvidia-smi for power/thermal.

### 8.2 High TTFT

1. Prefill chunk size — too small means many steps for long prompts.
2. Queue wait time — admission backlog.
3. Prefix cache miss rate.
4. Tokenizer overhead — sometimes a giant prompt's tokenization dominates TTFT (Python regex tokenizers can be slow; pre-tokenize when possible).

### 8.3 Memory Issues at Inference

- **OOM**: gpu_memory_utilization too high; reduce.
- **Fragmentation**: long-running engines may suffer over hours; restart periodically.
- **Slow allocator**: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True helps.

### 8.4 Quality Regression

1. Diff the engine version and quantization config.
2. Re-run on same prompts pre/post.
3. Check tokenizer (chat template change).
4. Sampling parameter change (temperature, min_p, etc.).

---

## 9. The Debug Hierarchy (Where To Look First)

When a production issue lands:

1. **Are SLOs being violated?** Check Prometheus.
2. **What changed recently?** Engine deploys, model swaps, config.
3. **Is one replica or many?** One replica often = infrastructure (host issue, NIC). All = systemic.
4. **What does nvidia-smi say on the bad host?** Util? Power? ECC?
5. **What does the engine log say?** Preemption? OOM warnings? Slow kernels?
6. **If training**, what's per-rank step time?
7. **Drill down only if needed**: Nsys → Ncu.

Most production "performance" issues are misconfigurations or infrastructure problems, not kernel performance. Don't reach for Nsight Compute first.

---

## 10. Common Pitfalls

- **Averages hide tails** — always track p99.
- **Prometheus scrape interval too long** — 30s misses bursty issues; 5–15s is better for inference.
- **Ignoring DCGM `XID` errors** — silent hardware failures lurk; alert on them.
- **No NVTX annotations** — Nsys output is unreadable without them.
- **Profiling without warmup** — first few steps include kernel JIT, autotune, allocator setup. Skip them.
- **Conflating GPU `Volatile GPU-Util`% with real utilization** — Volatile is binary "any kernel running"; use DCGM for real metrics.

---

## 11. Common Interview Questions

**Q: What metrics would you put on an LLM inference dashboard?**
A: TTFT/TPOT/E2E latency p50/p95/p99 by request priority; throughput (RPS, tokens/sec); engine state (active seqs, queue depth, KV occupancy %, prefix cache hit rate, preemption rate); SLO compliance %; GPU SM/HBM/tensor-core util via DCGM; error rates by reason.

**Q: A user reports occasional TPOT spikes. How do you investigate?**
A: (1) Check if spikes correlate with long-prompt arrivals → prefill–decode interference; mitigate with chunked prefill. (2) Check if KV cache near full → preemptions causing recompute. (3) Inspect per-request log for the spike: was it a long output? Was prefix cache missing? (4) GPU/NIC metrics during spike windows.

**Q: How would you debug a hung training run?**
A: `py-spy dump` each rank to find stack states; compare ranks; check `NCCL_DEBUG=INFO` to see which collective hung; inspect IB link health and DCGM for hardware issues; look at logs for the last action per rank. Common cause: one rank crashed silently, others stuck at next collective.

**Q: What's the difference between PyTorch Profiler, Nsys, and Ncu?**
A: PyTorch Profiler — per-op CPU/CUDA time, easy framework-level view. Nsys — system timeline including NCCL, multi-process; ideal for orchestration debugging. Ncu — per-kernel deep dive with warp-level metrics; for kernel optimization. Use top-down: PyTorch Profiler / Nsys for "where does time go," Ncu for "why is this kernel slow."

**Q: What's a "straggler" and how do you find one?**
A: A rank significantly slower than others, dragging down all-reduce. Find by per-rank step-time histograms, DCGM SM utilization variance, or hardware events (ECC, thermal). Symptoms: step time mean dominated by max rank, all-reduce time 2× expected.

**Q: How do you measure prefix cache hit rate?**
A: Engine exposes `cached_blocks_hit / total_blocks_requested` per request and aggregate. Useful per-tenant dashboards: hit rate by user, by application. Drops in hit rate after a release suggest tokenizer or template change.

**Q: What's NVTX and why use it?**
A: NVIDIA Tools Extension — a way to annotate code regions with named markers visible in Nsys/Ncu timelines. Without NVTX, the timeline is just kernel names and stack traces; with NVTX, you see "forward_pass / attention / qkv_proj" as a hierarchy. Essential for any nontrivial profiling.

**Q: How would you investigate a sudden quality regression in a deployed inference replica?**
A: Diff (engine version, quantization config, model checkpoint, sampling defaults, tokenizer / chat template). Replay the same prompts pre/post; A/B with a known-good replica. Once isolated, check whether it's a quantization calibration issue, a kernel correctness regression, or a default-changed.

**Q: What does DCGM_FI_PROF_PIPE_TENSOR_ACTIVE measure?**
A: Fraction of time the tensor core pipeline was active. High value (>50%) during matmul-heavy phases; low even during compute-heavy phases means kernels aren't hitting tensor cores (wrong dtype, tile shape, layout).

**Q: How would you size your monitoring infrastructure for a 1000-GPU cluster?**
A: DCGM exporter on each node (~10s scrape, low overhead). Centralized Prometheus federation + long-term store (Thanos/Cortex) since per-GPU metrics are voluminous. Grafana dashboards by team, by job, by hardware tier. Alertmanager for paging; SLO dashboards for engineering review.

**Q: What's a CUDA Graph and when does it help?**
A: A captured sequence of CUDA operations replayed with one launch. Reduces kernel launch overhead — beneficial for inference-style workloads with many small kernels per step. PyTorch supports it via `torch.cuda.graph(...)`. Particularly helpful for low-batch decoding where launch overhead is non-trivial.

**Q: How do you alert on SLO burn-down rate?**
A: Compute `(SLO_violation_rate / total_request_rate) / SLO_target` over a sliding window. If burning faster than budget allows for the period (e.g., 2× normal), page on-call. Standard SRE error-budget alerting.

---

## 12. Further Reading

- NVIDIA DCGM documentation.
- NVIDIA Nsight Systems and Compute user guides.
- PyTorch Profiler tutorials.
- "Site Reliability Engineering" (Google) — SLO/error-budget chapters.
- Anyscale, Mosaic, Anthropic engineering blog posts on production inference.
- USENIX SREcon talks on ML system reliability.

---

**Next:** [Production_Architecture](Production_Architecture.md).
**See also:** [Distributed_Training](Distributed_Training.md), [Inference_Frameworks](Inference_Frameworks.md), [Collectives_and_NCCL](Collectives_and_NCCL.md).

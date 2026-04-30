# Training Optimization

How to make a large training run actually hit its theoretical throughput. This page covers the kernel-level optimizations (Flash, fused kernels, custom CUDA), system-level (TorchCompile, mixed precision, memory schedules), and the empirical tuning loop (MFU targets, profiling, ablation discipline).

**Prerequisites**: [Distributed_Training](Distributed_Training.md), [CUDA_Optimization](../Part_4_GPU_Kernel_Engineering/CUDA_Optimization.md), [Memory_Hierarchy_and_Roofline](../Part_2_Microarchitecture/Memory_Hierarchy_and_Roofline.md).

---

## 1. The Goal: MFU

Model FLOPs Utilization — actual training FLOPs divided by hardware peak FLOPs:

```
MFU = (model_FLOPs_per_step) / (peak_FLOPs · step_time)
```

Where `model_FLOPs_per_step ≈ 6 · params · tokens_per_batch` for a transformer (3 forward, 3 backward — slightly low estimate; some count attention separately).

Targets:
- **A100 BF16**: ~50–55% MFU is excellent for dense models.
- **H100 BF16**: ~45–55% (lower because peak is harder to hit).
- **H100 FP8**: ~35–45% (peak is 2× higher, baseline relatively lower).
- **B200**: targets emerging, ~35–45% FP8.

If MFU is below ~30%, something is wrong: comm-bound, kernel-bound, or scheduler-bound. Profile.

---

## 2. Optimization Tiers

In rough order of biggest payoff for fewest hours:

1. **Hardware** — H100 over A100, FP8 over BF16, NVLink over PCIe. Sometimes the "optimization" is buying better silicon.
2. **Algorithm-level** — FlashAttention, fused kernels, RMSNorm vs LayerNorm.
3. **Mixed precision** — BF16 / FP8 instead of FP32.
4. **Activation recomputation** — selective.
5. **Compilation** — TorchCompile, XLA, NeMo.
6. **Sharding tuning** — ZeRO stage, TP/PP balance.
7. **Comm tuning** — NCCL channels, bucket sizes, overlap.
8. **Data loading** — async, prefetch, packed sequences.
9. **Custom kernels** — write or import optimized CUDA / Triton for hot ops.

---

## 3. Kernel-Level Wins

### 3.1 FlashAttention

Mandatory. Drops attention from compute-and-memory-bound to compute-bound, saves O(N²) memory. See [FlashAttention_Deep_Dive](../Part_4_GPU_Kernel_Engineering/FlashAttention_Deep_Dive.md).

### 3.2 Fused LayerNorm / RMSNorm

Single kernel that computes mean/var, normalizes, applies scale/bias. Saves the multiple HBM round-trips of naive PyTorch. Apex / NeMo / `flash_attn` ship them.

### 3.3 Fused MLP (SwiGLU)

```
y = (gate_proj(x) ⊙ silu(gate_proj(x))) · up_proj(x)   ← gated
out = down_proj(y)
```
Fuse the gate + up into a single matmul (concat weights), then the SiLU + mul + down can sometimes fuse with the down projection's epilogue (in CUTLASS).

### 3.4 Fused Cross-Entropy

Last-layer logits + softmax + log-likelihood + backward all in one fused kernel that streams over vocab dim without ever materializing `(B, S, V)` logits in HBM. Critical at large vocab (V=128K+ in modern models). Liger-Kernel, Apex, Megatron-LM all ship variants.

### 3.5 Fused AdamW

Combines weight update + master-precision conversion + grad clipping. Single kernel sweep over params instead of multiple. PyTorch's `foreach` + `fused=True` works for AdamW.

### 3.6 RoPE Fused

Fuse rotary embedding application with the QK projection or directly into FlashAttention. `flash_attn_func` with `rotary_emb` arg does this.

### 3.7 Liger-Kernel

LinkedIn's open-source "Triton + a few CUDA" kernel collection: SwiGLU, GeGLU, RMSNorm, RoPE, fused cross-entropy, FusedLinearCE, FusedJSD. ~10–20% speedup, ~50% memory reduction on full training. Drop-in via `liger_kernel.transformers.apply_liger_kernel_to_llama`.

---

## 4. Compilation: TorchCompile and XLA

### 4.1 TorchCompile

`torch.compile(model)` traces and lowers PyTorch ops to optimized fused kernels via TorchInductor (which often generates Triton). Best results when:
- Static shapes (use `dynamic=False`).
- Limited Python overhead.
- Eligible operator coverage (ops outside the supported set fall back).

Reported gains for training: 10–30% on dense models. Pairs well with FSDP.

### 4.2 XLA / JAX

JAX programs lower to XLA → HLO → tile-level codegen. SPMD program with `jax.shard_map` and `Mesh`. Used in Pax/MaxText, Google's internal stacks. Strong for TPU; competitive for GPU.

### 4.3 NeMo / Megatron Inductor

Megatron precompiles its kernels (Transformer Engine, Apex). Often you don't need TorchCompile because the hot ops are already CUDA. Composing them: TE handles linear/attention; TorchCompile can wrap remaining Python.

---

## 5. Mixed-Precision Strategies

### 5.1 BF16 Default

Simple and robust. Use FP32 master weights for stability:
```
W_master FP32 → cast to BF16 for forward
forward in BF16
backward in BF16
gradients in BF16
master weight update in FP32
```

### 5.2 FP8 (Hopper Transformer Engine)

NVIDIA Transformer Engine wraps each linear / layer in FP8 logic with **per-tensor delayed scaling**: maintain a moving max of activation magnitudes, choose a scale that keeps the tensor in FP8 range. Hopper tensor cores provide 2× FP16 throughput.

Sketch:
```python
import transformer_engine.pytorch as te

class FP8Linear(te.Linear):
    def forward(self, x):
        # automatic FP8 cast & scale management
        return super().forward(x)
```

What stays in BF16/FP32: layer norms, optimizer state, master weights. Saves HBM and gets ~1.5–2× wall-clock speedup over BF16 with negligible quality loss when scales are tuned.

### 5.3 FP4 / MXFP (Blackwell)

Emerging. Microscaling formats with 32-element blocks that share an exponent. Tensor cores 2× over FP8. Requires more careful calibration; mostly research as of 2026.

---

## 6. Activation Schedules

### 6.1 Selective Recompute

Only attention recomputed in backward (because FA is compute-bound and quick). Saves the bulk of activation memory at ~5% extra FLOPs.

```python
torch.utils.checkpoint.checkpoint(attention_block, x)
```

### 6.2 Layer-Granular

Recompute every other transformer block. Trade memory for FLOPs precisely.

### 6.3 Memory-Efficient Optimizer Step

PyTorch FSDP can keep one layer's params materialized at a time. Optimizer state is sharded (ZeRO-3). Combined: peak memory ≈ a single layer's activations + sharded state.

---

## 7. Communication Optimization

### 7.1 Bucket Sizes

DDP / FSDP bucket gradients to amortize NCCL launch overhead. Default 25 MB for DDP; FSDP uses module-level buckets via the wrapper. Smaller buckets → more launches; larger → less overlap with backward.

### 7.2 Overlap Backward with Reduce-Scatter

FSDP issues `reduce-scatter` for layer i's gradients during backward of layer i-1. Critical for hiding comm under compute.

### 7.3 NCCL Channel Tuning

```
NCCL_MIN_NCHANNELS=4
NCCL_MAX_NCHANNELS=16
```
More channels → more BW used for collectives. Trade with compute SM occupancy. Sweet spot per topology.

### 7.4 Comm-Compute Co-tuning

Profile (Nsight) to see where comm overlaps fail. Common fix: reorder operations so big matmuls come **before** comm-blocking points; small element-wise ops after the comm starts.

---

## 8. Data Loading

### 8.1 Prefetch & Pipeline

Dataloader workers build the next microbatch on CPU while GPU is doing the previous step. Pinned memory + `non_blocking=True` H2D copies overlap with compute.

```python
DataLoader(..., num_workers=8, prefetch_factor=4, pin_memory=True)
```

### 8.2 Tokens, Not Sequences

Pre-tokenize and pack: store `uint16`/`uint32` arrays, slice random `seq_len` chunks. Avoids on-the-fly tokenization (CPU-bound).

### 8.3 Sequence Packing

Pack multiple short documents into one `seq_len`-long sample, separated by EOS. Add an attention mask preventing cross-document attention. Eliminates padding waste; near 100% token utilization.

### 8.4 Dataloader Throughput Math

For 8K H100s training Llama-3-70B at MFU 50%:
```
tokens/sec ≈ 0.5 · 989e12 · 8192 / (6 · 70e9) ≈ 9.6e6 tokens/sec
data BW (FP16, 2 bytes/tok): ~20 MB/s — easy
```
Token data is small. Real bottleneck is the **streaming/shuffling** of large datasets, not raw bytes.

---

## 9. Profiling Workflow

### 9.1 Tools

- **PyTorch Profiler** (`torch.profiler`) — kernel timeline, memory, top-ops.
- **Nsight Systems** — system-level GPU/CPU timeline with NCCL annotations.
- **Nsight Compute** — kernel-level, per-warp metrics, roofline.
- **DLProf** — NVIDIA framework profiler.
- **Megatron's `--log-throughput`** — TFLOPS, MFU, step time printouts.

### 9.2 What to Look For

- **GPU idle gaps** in the timeline — sync points, dataloader stalls, kernel launch overhead.
- **NCCL not overlapping** — backward and reduce-scatter on different streams but appearing serial.
- **Many short kernels** — fuse them.
- **Register spills** in Nsight Compute — kernel using too many regs, drops occupancy.
- **Memory throughput vs theoretical** — if a kernel hits 60%+ of HBM BW it's BW-bound, optimize for compute.

### 9.3 Roofline View

Plot achieved FLOPs/s vs arithmetic intensity. Compute-bound kernels hover near peak FLOPs; BW-bound near `BW · AI`. Identifies whether to optimize compute (fuse, use tensor cores) or memory (reduce traffic, tile better).

---

## 10. Ablation Discipline

When tuning at scale:

- Change one thing at a time.
- Match seeds and data order.
- Measure for ≥1000 steps to average out warm-up.
- Compare TFLOPS, step time, and loss curve — not just one metric.
- Test convergence at smaller scale before committing 10K-GPU runs.

A common failure mode: a "speedup" turns out to silently hurt loss after 100K steps. Run ablations long enough.

---

## 11. Common Pitfalls

- **Disabling activation recomputation thinking it'll be faster** — it usually OOMs at scale; selective is the right knob.
- **TorchCompile graph breaks** — too many Python branches; cache miss → slowdowns. Use `TORCH_LOGS=+dynamo` to debug.
- **NaN with FP8** — calibration scales not yet stable. Warm up FP32 / BF16 for ~100 steps before switching.
- **Memory fragmentation** — frequent tensor alloc/free. Use `expandable_segments` allocator (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`).
- **Dataloader hangs** — too many workers, fd exhaustion. Reduce `num_workers` or use streaming.
- **Comm not overlapping** because of stream sync calls. Audit `torch.cuda.synchronize()` usage.

---

## 12. Common Interview Questions

**Q: What is MFU and what's a reasonable target?**
A: Model FLOPs Utilization — actual training FLOPs over hardware peak. Targets: ~50% on A100 BF16, ~45–55% H100 BF16, ~35–45% H100 FP8. <30% indicates a problem.

**Q: How do you achieve >50% MFU on H100 BF16 training?**
A: FlashAttention, fused norms / cross-entropy / Adam, BF16 mixed precision, selective activation recomputation, FSDP with overlapped reduce-scatter, sequence packing for token utilization, tuned NCCL channels, and TorchCompile or pre-compiled kernels.

**Q: Why is fused cross-entropy important?**
A: Logits at large vocab (128K+) are huge — `(B, S, V)`. Materializing them in HBM blows memory and BW. Fused CE streams over vocab in shared memory, computes loss + gradient without ever staging the full logits tensor. Memory drops dramatically; BW too.

**Q: What does `torch.compile` do?**
A: Traces a Python function into a graph, applies fusions, and lowers to optimized kernels via TorchInductor (often Triton-generated). Speedups 10–30% on training when the trace is stable. Limitations: dynamic shapes, custom Python control flow, unsupported ops cause graph breaks.

**Q: How does FP8 training work on Hopper?**
A: Transformer Engine wraps linears with FP8 forward/backward + FP32 master. Maintains a moving max of activations to choose scales (delayed scaling). Tensor cores compute 2× faster than BF16. Most non-matmul ops stay BF16/FP32 for stability.

**Q: What's selective activation recomputation?**
A: Recompute only attention activations in backward (since FA is fast and compute-bound). Saves bulk of activation memory at ~5% FLOP overhead vs ~33% for full recomputation.

**Q: How do you overlap NCCL all-reduces with backward compute?**
A: PyTorch DDP/FSDP bucket gradients per layer. As soon as a bucket's gradients finish, fire the all-reduce on a separate CUDA stream while the next layer's backward continues. ZeRO-3 also does this for params (gather one layer ahead).

**Q: What's "sequence packing" and why does it matter?**
A: Concatenate multiple short documents into a fixed seq_len chunk separated by EOS, with attention masks that prevent cross-document attention. Eliminates padding waste — token utilization near 100% instead of 70–85% with naive padding.

**Q: How do you debug "training is 30% slower than expected"?**
A: (1) Profile a step with PyTorch Profiler / Nsight to see GPU timeline. (2) Look for idle gaps (data loader, sync points, kernel launch overhead). (3) Check NCCL bus-bw via nccl-tests. (4) Check MFU; compare against the theoretical model. (5) Compare compute time per layer vs expected; check for kernel regressions.

**Q: What's "expandable_segments" allocator and why does it help?**
A: PyTorch CUDA allocator setting that allows segments to grow dynamically. Reduces fragmentation when tensors of varying sizes alloc/free repeatedly. Helpful for FSDP and TorchCompile workloads.

**Q: How would you optimize a custom transformer block kernel?**
A: (1) Fuse independent ops (norm + linear + bias). (2) Use FlashAttention for the attention. (3) Tile sizes that fit shared memory but maximize tensor core utilization. (4) Async pipeline (TMA on Hopper). (5) Verify register pressure isn't dropping occupancy. (6) Compare against CUTLASS reference. See [CUDA_Optimization](../Part_4_GPU_Kernel_Engineering/CUDA_Optimization.md).

**Q: When do you choose Triton over CUDA for a custom kernel?**
A: Triton when the kernel is moderate complexity, you need autotuning, and you're not chasing the absolute last percent. CUDA / CUTLASS when you need cluster-launch / TMA / wgmma / warp specialization on Hopper or beyond what Triton compiles efficiently.

**Q: What's an example of a fused kernel that gives 10%+ training speedup?**
A: Liger's fused linear + cross-entropy: replaces last-layer matmul + softmax + CE + backward with one streaming kernel that never materializes the full logits. Reduces both compute and memory dramatically at large vocab.

---

## 13. Further Reading

- Korthikanti et al., "Reducing Activation Recomputation" (2022).
- NVIDIA Transformer Engine (FP8) docs.
- "FlashAttention-3" paper (2024).
- Liger-Kernel GitHub & blog.
- PyTorch FSDP and DTensor tutorials.
- TorchCompile benchmarking guides.
- Megatron-LM and NeMo source code.

---

**Next:** [Inference_Frameworks](../Part_6_Inference_Engines_and_Serving/Inference_Frameworks.md).
**See also:** [CUDA_Optimization](../Part_4_GPU_Kernel_Engineering/CUDA_Optimization.md), [Distributed_Training](Distributed_Training.md), [Quantization](../Part_5_Algorithms_and_Quantization/Quantization.md), [Observability_and_Debugging](../Part_6_Inference_Engines_and_Serving/Observability_and_Debugging.md).

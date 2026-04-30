# Batching and Scheduling

How an inference server packs concurrent requests onto a finite number of GPUs determines almost every performance metric users care about — TTFT, TPOT, throughput, fairness. This page covers static vs continuous batching, prefill–decode interference, chunked prefill, request-level admission control, and the SLO math.

**Prerequisites**: [KV_Cache](KV_Cache.md), [Memory_Hierarchy_and_Roofline](../Part_2_Microarchitecture/Memory_Hierarchy_and_Roofline.md).

---

## 1. The Two Phases: Prefill vs Decode

Inference has two computationally distinct phases:

| Phase   | Token count per step | Compute character | Bottleneck         |
|---------|----------------------|--------------------|---------------------|
| Prefill | many (whole prompt) | Compute-bound (matmuls dominate) | Tensor core throughput |
| Decode  | exactly 1 / sequence | Memory-bound (re-reads KV + weights) | HBM bandwidth |

The **same GPU running the same model** has totally different throughput in the two phases. Mixing them wastes one or the other resource.

### Numerics for Llama-3-70B on H100

- **Prefill**, batch=1, S=4K: ~140 GB weights · 1 read + ~280 GB activations ≈ a few hundred ms. FLOPs ~`2·N·params·S/B` heavy. Tensor cores saturate.
- **Decode**, batch=1: re-read 140 GB weights every token → 140 / 3.35 ≈ 42 ms. ~24 tok/s.
- **Decode**, batch=64: still ~42 ms (BW-bound) → 64·24 ≈ 1500 tok/s aggregate. **Batching is free in the BW-bound regime.**

This asymmetry is the foundation of every batching/scheduling decision.

---

## 2. Static Batching (Don't)

Old-style: collect N requests, pad to max length, run forward, return all N together. Wait for the slowest.

Problems:
- Fixed shape requires waiting for batch to fill → tail latency suffers.
- Padding wastes compute proportional to length variance.
- Requests with different output lengths block each other.
- KV cache fragmentation if you preallocate `(B, max_seq, …)`.

Useful only for offline batch jobs with similar-length items. Production never uses static batching for online serving.

---

## 3. Continuous (In-Flight) Batching

Originated in Orca (OSDI 2022), now standard everywhere (vLLM, TRT-LLM, TGI, SGLang, Dynamo).

**Key idea**: each *step* (one forward pass) decides which sequences participate. Sequences enter when admitted, leave when they emit EOS or hit max-tokens — without disturbing other in-flight sequences.

### 3.1 Step Loop

```
loop forever:
    runnable = scheduler.select(current_seqs, pending_queue)   # who runs this step?
    inputs   = build_step_inputs(runnable)                     # tokens, KV refs
    logits   = model.forward(inputs)
    samples  = sampler(logits, runnable.sampling_params)
    for s in runnable:
        s.append_token(samples[s.id])
        if s.is_finished():
            release_kv(s); emit_response(s)
```

A "step" is a single GPU forward pass. Across steps the participating set changes.

### 3.2 What Makes It Possible

- **Paged KV cache** — sequences can have different lengths and storage independent of batch size. (See [KV_Cache §4](KV_Cache.md).)
- **Variable-length attention kernels** — FlashAttention with packed sequences and per-sequence offsets.
- **Per-sequence sampling state** — temperatures, top-p, biases, grammar masks, repetition penalty differ across requests.
- **Streaming response delivery** — partial tokens go out before the request is done.

### 3.3 Throughput Wins

Naive estimate: under continuous batching, latency ≈ static-batching latency for the slowest, but **utilization** ≈ 100% because no GPU cycles are wasted on padding or finished sequences. Real-world throughput improvement vs static batching at realistic length variance: 2–10×.

---

## 4. Prefill–Decode Interference

Continuous batching mixes prefill and decode in the same step. This **interferes** because they have opposite bottlenecks.

### 4.1 The Problem

Run a 4K-token prefill alongside 32 ongoing decodes:
- Prefill compute: tens of ms — saturates the tensor cores.
- Decodes are memory-bound, would otherwise take ~5 ms each step.
- Combined step takes the *prefill time* (compute-bound) — every decode token in this step gets a TPOT spike.

Symptoms users see:
- TPOT (time per output token) **stutters** when long prompts arrive.
- Long prompts hurt the latency of everyone else's decoding.

### 4.2 Mitigations

1. **Chunked Prefill** (vLLM, SGLang): split a long prompt into chunks of, say, 512 tokens. Each chunk is a small prefill that interleaves cleanly with decodes. TPOT smooths out at the cost of slightly higher TTFT.
2. **Prefill–Decode Disaggregation**: dedicate separate GPUs (or SM partitions) to prefill and decode. Eliminates interference but adds a KV transfer step. See [Prefill_Decode_Disaggregation](Prefill_Decode_Disaggregation.md).
3. **Prefill Caps**: limit max prefill tokens per step (e.g., 8192). Long prompts spread across multiple steps.
4. **Priority Scheduling**: schedule decodes first, prefills second; prefills only run when there's compute slack.

Real systems combine these. vLLM defaults to chunked prefill with a per-step prefill budget.

---

## 5. Chunked Prefill in Detail

For a prompt of length S split into chunks of C:

```
total prefill steps = ceil(S / C)
each chunk does compute proportional to (positions seen so far)
```

For each chunk i (1-indexed), we attend over positions `[1, i·C]`. So compute is:

```
sum_{i=1..K} 2 · C · (i·C) · per_token_attn_cost
≈ K · C² · per_token_attn_cost   ≈ S² · per_token_attn_cost / 1
```

Same total FLOPs as one big prefill — chunking doesn't add work, it spreads the work into pieces small enough to interleave with decodes.

**Optimal C** balances (a) chunk small enough that one chunk + N decodes fits in a step's TPOT budget, and (b) chunk large enough to keep tensor cores reasonably loaded. Typical C: 256–2048.

---

## 6. Scheduling Policies

Choosing which sequences run each step.

### 6.1 First-Come-First-Served (FCFS)

Default. Simple, predictable TTFT, fair across users.

### 6.2 SJF / SRJF (Shortest Remaining Job First)

Schedule sequences with fewer remaining tokens first. Improves average latency but starves long sequences. Need an estimate of remaining length — token-budget proxies, length-classifier models, or just "min(decode_so_far, max_tokens)".

### 6.3 Priority + SLO-Aware

Each request has a priority (tier) and an SLO (e.g. p99 TTFT < 500 ms, TPOT < 50 ms). Scheduler tracks slack:
```
slack = SLO_deadline - now
```
Lower slack = higher priority. Implementations: EDF (earliest deadline first), weighted fair queueing.

### 6.4 Preemption

When KV pool is full, evict a victim. Common policies:
- **Recompute on resume** (vLLM default for incoming requests): drop blocks, redo prefill on next admission. Simple, no swap.
- **Swap-to-CPU**: copy KV to host, restore on resume. Lower recomputation cost, higher PCIe traffic.

Choosing victims: lowest priority, longest waiting, most tokens generated, or LRU.

### 6.5 Admission Control

Refuse requests that would (a) exceed token budget, (b) overflow KV pool, (c) miss SLO. Return 429 / queue with backpressure. Alternative: accept-and-degrade (e.g., shorten max-tokens cap).

---

## 7. SLO Math: TTFT vs TPOT vs Throughput

Two user-visible latency metrics:

- **TTFT** (Time to First Token) — admission + prefill latency.
- **TPOT** (Time Per Output Token) — average decode-step latency for the user's slot.
- **End-to-end latency** = TTFT + TPOT · (output_tokens − 1).

Throughput-vs-latency tradeoff:

```
TPOT(B) ≈ (weight_bytes + B·KV_bytes_per_seq) / HBM_BW
       ≈ A + B·k                      # affine in B once you cross BW
Throughput(B) ≈ B / TPOT(B)
```

Define `B*` as the batch where adding more requests starts hurting individual TPOT enough to violate SLO. Goldilocks point: at or below B*. Typical: 32–128 on H100 for 7B–70B models depending on context length.

### Example

For Llama-70B FP16 on 2× H100 (NVLink), KV at 4K context = 1.25 GB/seq:
```
TPOT(B) ≈ (140 + 1.25·B) GB / 6.7 TB/s
       = 20.9 ms + B·0.187 ms
```
SLO TPOT = 50 ms → max B ≈ 155. Pick B around 100–128 for headroom.

---

## 8. Sampling Inside the Batch

Each sequence has its own sampling parameters. The serving step:

1. Compute logits for all sequences in the step.
2. Apply per-sequence transforms: temperature divide, top-k mask, top-p mask, repetition / frequency / presence penalties, logit bias, structured-output constraint masks (JSON, regex, grammar).
3. Sample.

These are tiny GPU kernels that take per-sequence params via tensors. Continuous batching means parameters are not uniform — the implementation typically dispatches a kernel that takes parameter arrays of length B.

---

## 9. Speculative Decoding Inside the Batch

Speculative decoding (see [Speculative_Decoding](Speculative_Decoding.md)) drafts K future tokens cheaply and verifies in parallel. Inside continuous batching:

- The "step" now produces 1..K+1 accepted tokens per sequence.
- The draft model (or Medusa head, or EAGLE) is part of the same scheduler.
- Variable acceptance length per sequence ⇒ batch shape becomes ragged.

vLLM and TRT-LLM both fold speculation into their step loop with little user visibility.

---

## 10. Multi-LoRA / Multi-Tenant

When the server hosts a base model + many LoRA adapters:
- Each request specifies a LoRA id.
- Engine groups requests by LoRA in the step or uses a fused kernel that loads the right adapter weights per row.
- Punica / S-LoRA / vLLM's multi-LoRA all handle this with batched LoRA matmul kernels.

The scheduler's job grows: respect LoRA placement (some LoRAs may be cold-loaded from CPU/NVMe), avoid pathological packings.

---

## 11. Common Pitfalls

- **Batching prefills and decodes together with no chunking**: latency stutters. Always chunk or disaggregate.
- **Allowing unbounded max-tokens**: a single 8K-output request can starve everyone else when KV pool is small.
- **Picking batch by TPS only**: ignoring TPOT distribution. Track p99 TPOT, not just mean.
- **Forgetting prefix cache on scheduling decisions**: a request that hits prefix cache effectively has zero prefill cost — schedule it ahead of similar-priority cold requests.
- **Static batch size = max in-flight requests**: continuous batching has no fixed cap; the limit is KV pool slots.

---

## 12. Common Interview Questions

**Q: Static vs continuous batching — what's the difference and why does it matter?**
A: Static: form a fixed batch, run, return all together. Continuous: each step picks who runs from a pool of in-flight sequences, so newly admitted requests join immediately and finished ones leave without disturbing others. Dramatically better utilization on workloads with length variance.

**Q: What is prefill–decode interference?**
A: Mixing a compute-bound prefill with bandwidth-bound decodes in the same step makes the step take the prefill's time, hurting decode TPOT for everyone in the batch. Mitigated by chunked prefill or disaggregation.

**Q: How does chunked prefill work and what does it cost?**
A: Long prompts split into chunks of 256–2K tokens. Each chunk is a small prefill step. Same total FLOPs but spread across steps, so they interleave with decodes. Slightly higher TTFT (you process the prompt across multiple steps) for much better TPOT smoothness.

**Q: How would you decide max-batch on a 70B model on 8× H100?**
A: Compute decode-step latency as `(weight_bytes + B·KV_bytes_per_seq)/aggregate_HBM_BW` (with TP across the GPUs), pick B such that TPOT comfortably meets SLO p99 with headroom. Verify against the real model — kernel launch overhead and non-attention layers add fixed cost.

**Q: What's the role of paging in continuous batching?**
A: Continuous batching needs sequences with arbitrary, time-varying lengths to coexist. Padded contiguous layouts can't represent that without massive waste. Paging gives each sequence its own block-table over a global pool. (See [KV_Cache §4](KV_Cache.md).)

**Q: What scheduling policy would you pick for a chat product with strict TTFT SLO?**
A: SLO-aware EDF on TTFT for new arrivals, FCFS within priority tier, chunked prefill, and a preemption policy that recomputes rather than swaps if memory pressure spikes. Keep prefill budget per step bounded.

**Q: What's "preemption" in this context and when do you choose recompute vs swap?**
A: Kicking a sequence out to free KV blocks. Recompute = drop blocks, redo prefill on resume — cheap memory-wise, repeats work. Swap = copy KV to CPU, restore later — pays PCIe BW but no FLOP repeat. vLLM default: recompute for new requests, swap for in-flight.

**Q: Why is `TPOT` nearly flat as you grow the batch in the BW-bound regime?**
A: Decode reads weights once + KV per sequence; weights dominate at small batch. Adding sequences adds linear KV bytes but doesn't change weight bytes, so step time grows slowly. This is why batching is "free" until KV reads dominate.

**Q: What metrics would you put on a serving dashboard?**
A: TTFT p50/p95/p99, TPOT p50/p95/p99, end-to-end latency by output length bucket, throughput (req/s and tok/s), KV cache occupancy %, prefix cache hit rate, queue depth, preemption rate, GPU SM utilization, HBM bandwidth utilization.

**Q: How does multi-LoRA serving change scheduling?**
A: Each request has a LoRA id; the engine must load the adapter weights (cheaply because they're small) and use a fused matmul kernel that handles per-row LoRA selection. Scheduling should avoid thrashing if LoRAs are cold-loaded from CPU.

**Q: Where does prefix caching plug into the scheduler?**
A: At admission and at chunk boundary. On admission, check prefix-cache hit and adjust the prefill cost. On chunk-prefill boundaries, also re-check (deeper hits become visible after the first chunk).

**Q: How would you deal with a "noisy neighbor" 32K-prompt request that hurts everyone else?**
A: Cap the per-step prefill budget so it spreads across many steps; assign it lower priority; isolate prefill GPUs (disaggregation); apply admission control with a priority quota.

---

## 13. Further Reading

- Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models" (OSDI 2022).
- Kwon et al., vLLM SOSP 2023.
- Holmes et al., "Sarathi-Serve" — chunked prefill paper (OSDI 2024).
- Patel et al., "Splitwise" — disaggregation (ISCA 2024).
- NVIDIA Dynamo blog posts on disaggregated serving.

---

**Next:** [Prefill_Decode_Disaggregation](Prefill_Decode_Disaggregation.md).
**See also:** [KV_Cache](KV_Cache.md), [Speculative_Decoding](Speculative_Decoding.md), [Inference_Frameworks](../Part_6_Inference_Engines_and_Serving/Inference_Frameworks.md).

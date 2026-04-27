# vLLM Internals

vLLM is the most studied open-source inference engine. This page is a code-level tour: how an in-flight request flows through the system, how PagedAttention is implemented, how the scheduler decides each step, and where to look in the source when something goes wrong.

**Prerequisites**: [Inference_Frameworks](Inference_Frameworks.md), [KV_Cache](KV_Cache.md), [Batching_and_Scheduling](Batching_and_Scheduling.md).

---

## 1. The High-Level Architecture (V1 Engine)

```
┌─────────────────────────────────────────┐
│  AsyncLLMEngine (entrypoint)            │
│   - HTTP/gRPC API                       │
│   - request → output stream             │
└──────┬──────────────────────────────────┘
       │ submit(request)
       ▼
┌─────────────────────────────────────────┐
│  EngineCore (V1 scheduler/coordinator)  │
│   - admission                           │
│   - chunked prefill planning            │
│   - per-step batch building             │
│   - prefix-cache lookup                 │
└──────┬──────────────────────────────────┘
       │ step(model_inputs)
       ▼
┌─────────────────────────────────────────┐
│  Worker[i] for i in TP group (per GPU)  │
│   - ModelRunner                         │
│   - KVCacheManager (block pool, etc.)   │
└──────┬──────────────────────────────────┘
       │ kernel calls
       ▼
┌─────────────────────────────────────────┐
│  Custom CUDA / Triton kernels:          │
│   - paged_attention                     │
│   - flash_attn                          │
│   - quantized matmul                    │
│   - sampler                             │
└─────────────────────────────────────────┘
```

In V1, the EngineCore runs in its own process; workers are sub-processes. IPC via shared memory queues (zmq / multiprocessing). This isolates Python overhead from the model run.

---

## 2. Request Lifecycle

### 2.1 Submission

User calls the OpenAI-compatible endpoint:
```
POST /v1/chat/completions
{
  "model": "meta-llama/Llama-3-70B-Instruct",
  "messages": [...],
  "stream": true,
  "max_tokens": 512,
  "temperature": 0.7
}
```

Frontend turns it into an internal `Request` object: tokenized prompt, sampling params, request id, stream queue.

### 2.2 Admission

Scheduler checks:
- Token budget remaining (max_num_batched_tokens)?
- KV pool slots available?
- Priority/SLO?

If admitted, request enters the **waiting queue**. If rejected, returns 429.

### 2.3 Prefix Cache Lookup

Hash blocks of the prompt against the prefix cache. Successful matches → reuse those KV blocks (refcount++). Tail of the prompt becomes the prefill workload.

### 2.4 First Step (Prefill)

Scheduler decides: full prefill in one step, or chunked? Bound by `max_num_batched_tokens`.
Prefill produces KV for all prompt tokens and the first output logits.

### 2.5 Decode Steps

Each subsequent step adds 1 token (or K with speculative decoding) for each in-flight sequence. Step's batch is built fresh.

### 2.6 Termination

Sampler emits EOS, or hits max_tokens, or stop sequence. Sequence marked finished, blocks released (refcount-- on shared prefix; freed when zero), final output streamed.

---

## 3. The KV Cache Manager

### 3.1 Block Pool

A single pre-allocated tensor of shape `(num_blocks, 2, block_size, num_kv_heads, head_dim)` per layer. `num_blocks = (HBM_for_kv) / block_size_bytes`.

Free list of physical block IDs, allocator pops on demand.

### 3.2 Per-Sequence Block Table

```python
block_table: Dict[seq_id, List[int]]    # logical → physical block ids
```

When a sequence appends a token: if last block is partial, slot in the next position; if full, allocate a new block and append to the table.

### 3.3 Prefix Cache

```python
hash_to_block: Dict[block_hash, BlockMetadata]
block_hash = sha256(parent_hash || token_ids_in_block)
```

When a new request arrives, walk its tokens block-by-block computing hashes. On hit, share the physical block (refcount++). On miss, create new physical blocks, register them in the cache.

Eviction: when free list is empty, drop oldest unrefcounted prefix blocks.

### 3.4 Memory Math

For Llama-3-70B (L=80, H_kv=8, d=128, FP16, block_size=16):
```
bytes_per_block = 2 · 16 · 8 · 128 · 2 · 80 = 5.24 MB per block × layers? Wait.
```
Actually each layer has its own per-block tensor. Per layer: `2 · 16 · 8 · 128 · 2 = 65,536 bytes = 64 KB`. Total per block across all 80 layers: `80 · 64 KB = 5 MB`. Token-level: 5 MB / 16 = 320 KB/token (matches our [KV_Cache](KV_Cache.md) numbers).

For 60 GB of KV pool: ~12,000 blocks ≈ 192,000 token slots.

---

## 4. The Scheduler (V1)

### 4.1 Per-Step Decision

```python
def schedule_step():
    # 1. Move freshly arrived requests from waiting → ready
    # 2. Try to add waiting requests to the running set
    while waiting and can_admit(waiting[0]):
        seq = waiting.popleft()
        running.append(seq)

    # 3. Build the batch:
    #    - All running decoding sequences (1 token each)
    #    - Some prefill chunks if budget allows
    batch = []
    token_budget = max_num_batched_tokens
    for seq in running:
        if seq.in_prefill:
            chunk = min(seq.tokens_remaining_to_prefill, token_budget)
            batch.append((seq, chunk))
            token_budget -= chunk
        else:
            batch.append((seq, 1))   # decode
            token_budget -= 1
        if token_budget <= 0: break

    # 4. Preempt if KV pool insufficient
    while not enough_kv_for(batch):
        victim = pick_preemption_victim(running)
        evict(victim)

    return batch
```

### 4.2 Chunked Prefill

`max_num_batched_tokens` typically 2K–8K. A 32K-token prompt is split: first step has 8K prefill (no decodes), next 8K (now decodes can interleave), etc. After 4 chunks the prefix is built; subsequent steps are pure decode + 1 first-token slot.

### 4.3 Prefix Cache Awareness

Before prefilling, check prefix cache for the prompt's block hashes. Skip already-cached blocks; only prefill the suffix. Reduces work and frees the engine to take more concurrent requests.

### 4.4 Preemption

When KV pool is full and a new request needs admission:
- **Recompute mode**: drop a victim's blocks; victim re-enters waiting and will redo prefill on next admission.
- **Swap mode**: copy victim's KV blocks to CPU host memory; bring them back when victim resumes.

Default: recompute for incoming requests, swap for in-flight (configurable).

---

## 5. Model Runner Forward Pass

### 5.1 Build Inputs

```python
input_ids        : shape (total_tokens,)              # packed
position_ids     : shape (total_tokens,)              # per-seq positions
block_tables     : shape (batch, max_logical_blocks)  # per-seq block table
slot_mapping     : shape (total_tokens,)              # where to write KV (which physical slot)
seq_lens         : shape (batch,)                     # current cached length per seq
query_lens       : shape (batch,)                     # how many new tokens this step (1 for decode, N for prefill chunk)
```

### 5.2 Layer Forward

Per layer:
```python
# 1. RMSNorm
x = rmsnorm(hidden_states)

# 2. QKV projection (TP-sharded)
q, k, v = qkv_proj(x)

# 3. RoPE
q, k = apply_rope(q, k, position_ids)

# 4. Attention (write new K,V to cache, then attend)
write_kv_to_cache(k, v, slot_mapping)
out = paged_attention(q, k_cache, v_cache, block_tables, seq_lens)

# 5. Output projection (TP-sharded, all-reduce after)
out = o_proj(out)

# 6. Residual + RMSNorm
x = rmsnorm(hidden_states + out)

# 7. MLP (gate/up + SwiGLU + down, TP all-reduce)
mlp_out = mlp(x)

# 8. Residual
hidden_states = hidden_states + out + mlp_out
```

The "paged_attention" kernel is the heart of vLLM's perf — it dereferences block tables to load K,V from non-contiguous physical blocks and computes attention without staging.

### 5.3 Final Logits & Sampling

```python
logits = lm_head(final_hidden)             # (total_tokens, vocab)
# Slice to last token of each query span (for prefill, only the last token's logits matter)
last_logits = gather_last_per_seq(logits, query_lens)
sampled = sampler(last_logits, sampling_params)
```

---

## 6. The PagedAttention Kernel

### 6.1 What It Does

Computes attention output for one query position per sequence, given:
- Q: `(num_seqs, num_heads, head_dim)` — new token's queries.
- K_cache, V_cache: physical block storage.
- block_tables: which physical blocks belong to each sequence.
- seq_lens: current cached length per sequence.

### 6.2 Inner Loop (Sketch)

```cuda
__global__ void paged_attention_kernel(...) {
    int seq_id = blockIdx.x;
    int head_id = blockIdx.y;
    int seq_len = seq_lens[seq_id];

    // Load Q for this (seq, head) into registers
    Q[head_dim] = load_query(...);

    // Online softmax accumulators
    float m = -INF, ell = 0;
    float O[head_dim] = {0};

    // Iterate over logical blocks
    for (int blk = 0; blk < div_ceil(seq_len, BLOCK_SIZE); ++blk) {
        int phys = block_tables[seq_id][blk];
        // Load K, V block from physical storage
        load_kv_block(K_cache[phys], V_cache[phys], head_id);

        // Compute partial attention for this block (mask out beyond seq_len)
        float s[BLOCK_SIZE] = compute_qk_dot(Q, K_block);
        if (blk == last_block) mask_beyond(s, seq_len);

        // Online softmax update
        float m_new = max(m, max(s));
        float alpha = exp(m - m_new);
        float p[BLOCK_SIZE] = exp(s - m_new);
        ell = alpha * ell + sum(p);
        for (int d = 0; d < head_dim; ++d) {
            O[d] = alpha * O[d] + dot(p, V_block[:, d]);
        }
        m = m_new;
    }

    // Final normalization
    for (int d = 0; d < head_dim; ++d) O[d] /= ell;
    store_output(...);
}
```

For decode (query_len=1) this runs one threadblock per (seq, head). Implements flash-decoding-paged.

### 6.3 Prefill

For prefill chunks, queries are larger (chunk size). vLLM uses FlashAttention's prefill kernel adapted to the paged layout, with block_table passed in.

### 6.4 Variants Shipped

- `paged_attention_v1` and `paged_attention_v2` (with split-K parallelism for long sequences during decode).
- FA-v3 with paged tables (Hopper).
- Quantized variants for FP8 KV.

---

## 7. Tensor Parallelism Inside the Worker

vLLM uses Megatron-style TP:
- QKV projections: column-parallel (each rank owns H/N heads' Q,K,V).
- O projection: row-parallel (each rank takes its slice of attention output, all-reduce after).
- MLP: gate+up column-parallel; down row-parallel; all-reduce after.

Distribution via `torch.distributed` NCCL groups initialized at engine startup.

KV is naturally sharded across TP ranks (each rank holds its own heads' KV). So the block pool is per-rank.

---

## 8. The Sampler

Per-sequence sampling parameters live in tensors of length `batch`. Stages:

1. Apply temperature: `logits /= temperature`.
2. Apply repetition / frequency / presence penalties: scan recent tokens, modify their logits.
3. Apply logit bias (per token).
4. Apply structured-output mask (Outlines/xgrammar).
5. Top-k filter: keep top-k logits, set others to -inf.
6. Top-p filter: cumulative sort, keep tokens whose cumulative prob ≤ p.
7. Sample: multinomial or argmax.

All on GPU; no host round-trip.

---

## 9. Where to Look in the Source

```
vllm/
├── engine/
│   ├── async_llm_engine.py     # AsyncLLMEngine (V1 entrypoint)
│   └── engine_core.py          # The scheduler / coordinator
├── core/
│   ├── scheduler.py            # Step-level scheduling logic
│   ├── block_manager.py        # KV pool & block tables
│   └── prefix_caching.py       # Hash-based prefix cache
├── worker/
│   ├── worker.py               # Per-GPU process
│   └── model_runner.py         # Forward pass orchestration
├── model_executor/
│   ├── models/                 # Per-architecture (llama.py, mixtral.py, ...)
│   └── layers/                 # Attention, RMSNorm, FFN building blocks
├── attention/
│   ├── ops/
│   │   ├── paged_attn.py       # Python wrapper
│   │   └── flash_attn.py       # FA wrapper
│   └── backends/               # FA / xformers / native
├── csrc/                       # Custom CUDA kernels
│   ├── attention/
│   ├── quantization/
│   └── activation/
└── entrypoints/
    ├── openai/                 # OpenAI-compatible HTTP frontend
    └── api_server.py
```

This is the V1 layout (approximate; rapidly evolving). Major debugging lives in `engine_core.py` + `scheduler.py`.

---

## 10. Performance Knobs

```
--max-num-batched-tokens       # per-step token budget (chunked prefill cap)
--max-num-seqs                 # max sequences in one step (decode batch cap)
--gpu-memory-utilization       # fraction of HBM for KV pool (default 0.9)
--block-size                   # KV block size (default 16)
--swap-space                   # CPU swap GB
--enable-prefix-caching        # toggle (default on in V1)
--enable-chunked-prefill       # toggle
--tensor-parallel-size         # TP degree
--pipeline-parallel-size       # PP degree (less common)
--dtype                        # auto / bfloat16 / float16 / fp8
--kv-cache-dtype               # auto / fp8 / fp8_e4m3
--quantization                 # awq / gptq / fp8 / etc.
--speculative-config           # spec decode setup
```

Tuning: increase `max_num_seqs` until TPOT SLO breaks; increase `max_num_batched_tokens` until prefill chunks deliver acceptable TTFT smoothness.

---

## 11. Common Failure Modes

- **`CUDA OOM during init`**: gpu-memory-utilization too high vs model + activations. Reduce.
- **`Stuck` request that never streams**: stop tokens or grammar deadlock. Check the sampler's mask computation.
- **TPOT spikes correlate with long-prompt admissions**: chunked prefill not aggressive enough. Lower `max_num_batched_tokens`.
- **Prefix cache hit rate near zero**: prompt prefixes vary too much, or block_size mismatched (pre-tokenized chat templates change tokens slightly).
- **Workers hung on NCCL**: NCCL_TIMEOUT_MS too short, or one GPU dropped link. `NCCL_DEBUG=INFO` reveals.

---

## 12. Common Interview Questions

**Q: How does vLLM achieve high throughput?**
A: Three pillars: (1) PagedAttention for variable-length batching without padding, (2) continuous batching letting sequences enter/leave the active set per step, (3) prefix caching for hot system prompts and chat history. Plus optimized kernels (FA, paged_attn, quantized matmul).

**Q: What's vLLM's V1 engine?**
A: A 2024–2025 rewrite of the core. Decouples scheduler from worker dispatch via process IPC, moves hot-path scheduling to C++, simplifies the API. Roughly 1.5–2× higher peak throughput than V0 with identical model code.

**Q: Walk me through the scheduling loop.**
A: Each step: (1) try to admit waiting requests if KV and token budgets allow; (2) build a batch of decodes (1 token each) plus a prefill chunk if room remains; (3) preempt low-priority sequences if KV is exhausted; (4) dispatch to workers; (5) update sequence state with sampled tokens.

**Q: How is the KV block pool allocated?**
A: Single tensor per layer of shape `(num_blocks, 2, block_size, H_kv, d)`. `num_blocks` derived from `gpu_memory_utilization` × HBM minus model weights and activation buffers. Free list manages allocation; refcounts handle prefix sharing.

**Q: What does the PagedAttention kernel do that FlashAttention doesn't?**
A: Dereferences a block table per logical block to find physical KV storage, then runs flash-style online-softmax tile loop. Adds O(num_logical_blocks) integer indirections per query, otherwise the same algorithm. Production FA kernels in vLLM/SGLang now natively support paged tables.

**Q: How does vLLM handle a 64K-token prompt?**
A: Chunked prefill: split the prompt into 8K-token chunks (or whatever `max_num_batched_tokens` is), prefill one chunk per step, allowing decode of other in-flight sequences to interleave. Spreads compute and smooths TPOT.

**Q: What's the role of refcounts in the block manager?**
A: Each prefix-cache hit increments the block's refcount. When a sequence releases or a cached prefix is evicted, refcount decrements. Block returns to free list when refcount hits zero. Same block can serve many concurrent sequences.

**Q: How does TP work inside vLLM?**
A: Each worker process owns one GPU, holds 1/N of the model weights (Megatron column/row pattern). Workers run forward in lockstep with NCCL collectives at all-reduce points. KV cache is naturally sharded across heads.

**Q: How would you add a new model architecture to vLLM?**
A: Implement the architecture in `model_executor/models/<name>.py`: layer modules using vLLM's TP-aware Linear / RMSNorm / Attention building blocks; register in the model registry; add a config mapping. Most LLaMA-family additions are <500 lines.

**Q: How is prefix caching invalidated if prompt templates change between releases?**
A: Block hashes include token IDs, so any change in tokenization produces different hashes — old cached blocks remain valid for old prompts, new blocks created for new ones. Cold start: first request after a template change repays full prefill.

**Q: What's the worst case if `gpu_memory_utilization` is too high?**
A: Insufficient slack for activations / scratch buffers → OOM during prefill of a long prompt. Visible as `CUDA OOM` during step. Default 0.9 is usually safe; lower to 0.8 if running close.

---

## 13. Further Reading

- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (SOSP 2023).
- vLLM blog: V1 engine, prefix caching, structured output, multi-LoRA.
- vLLM GitHub source — best learning resource by far.
- Anyscale and Mosaic blog posts on vLLM tuning.

---

**Next:** [Kubernetes_and_Orchestration](Kubernetes_and_Orchestration.md).
**See also:** [Inference_Frameworks](Inference_Frameworks.md), [KV_Cache](KV_Cache.md), [Batching_and_Scheduling](Batching_and_Scheduling.md), [Speculative_Decoding](Speculative_Decoding.md).

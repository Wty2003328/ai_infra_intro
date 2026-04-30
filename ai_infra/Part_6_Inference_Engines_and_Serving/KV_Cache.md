# KV Cache

The KV cache is the dominant memory consumer in LLM inference and the primary determinant of throughput economics. This page covers what it is, why it exists, how big it gets, how production systems manage it (paging, prefix sharing, offload, eviction), and the math you need to reason about capacity.

**Prerequisites**: [Attention_Mechanisms](../Part_5_Algorithms_and_Quantization/Attention_Mechanisms.md), [Transformer_Internals](../Part_5_Algorithms_and_Quantization/Transformer_Internals.md), [Memory_Hierarchy_and_Roofline](../Part_2_Microarchitecture/Memory_Hierarchy_and_Roofline.md).

---

## 1. Why a KV Cache Exists

In autoregressive decoding, each new token attends to **all previous tokens'** keys and values. If we recomputed K and V from scratch every step, we'd do quadratic work — O(N²·d) per token instead of O(N·d).

The KV cache stores the K and V activations of every layer for every position seen so far. At each decode step:

1. Compute Q, K, V only for the **new** token (1 row each).
2. Append the new K, V row to the cache.
3. Run attention `Q · K_cache^T` then `softmax(...) · V_cache`.

This turns each decode step from O(N²·d) into O(N·d) — a huge win, but it shifts the bottleneck from compute to **memory bandwidth** (we re-read the entire cache every token) and **memory capacity** (we must hold the cache in HBM).

---

## 2. KV Cache Size Math

For a transformer with `L` layers, `H` KV heads (with GQA: H_kv ≤ H_q), head dim `d`, sequence length `S`, batch `B`, and dtype size `b` bytes:

```
KV_bytes = 2 · B · S · L · H_kv · d · b
         (factor 2 = K and V)
```

### Examples

**Llama-3-8B** (L=32, H_q=32, H_kv=8, d=128, FP16 b=2):
```
per-token: 2 · 32 · 8 · 128 · 2 = 131,072 bytes = 128 KB / token / sequence
For S=8192, B=1: 128 KB · 8192 = 1.0 GB
```

**Llama-3-70B** (L=80, H_q=64, H_kv=8, d=128, FP16):
```
per-token: 2 · 80 · 8 · 128 · 2 = 327,680 bytes = 320 KB / token / sequence
For S=8192, B=1:  320 KB · 8192 ≈ 2.5 GB
For S=8192, B=64: 2.5 GB · 64   ≈ 160 GB  (already exceeds one H100!)
```

**DeepSeek-V3** (L=61, MLA: latent dim 512+64 RoPE, FP16):
```
MLA stores compressed (c_KV) instead of K,V — see §6.
per-token: ~70 KB instead of ~600 KB for naive MHA.
```

### Memory Wall Quick Estimate

If a model fits in a fraction `f` of GPU memory and total memory is `M`, then KV cache budget is `(1-f)·M`. Max-tokens served = `(1-f)·M / (per_token_bytes)`. On an 80GB H100 hosting Llama-3-70B (≈140GB FP16, so it doesn't even fit on one — needs 2× H100):

```
2× H100 = 160 GB
Weights: 140 GB FP16 (or 70 GB FP8/INT8)
Activations + framework overhead: ~5 GB
KV budget (FP16 weights): ~15 GB → 47K tokens total across batch
KV budget (FP8 weights):  ~85 GB → 270K tokens total across batch
```

This is why **weight quantization is a throughput multiplier** — it frees HBM for KV cache.

---

## 3. Decode Bandwidth: The Memory Wall

Each decode step reads the entire KV cache from HBM:

```
bytes_per_token = KV_bytes_per_seq + weight_bytes
```

For Llama-3-70B FP16, S=4K, B=1:
- KV per token: 320 KB · 4096 ≈ 1.3 GB
- Weights:    140 GB
- Total per token: ~141.3 GB → at 3.35 TB/s → 42 ms/token → 24 tok/s

Wait — that's the per-step bytes. Note weights dominate at small S, KV dominates at large S·B.

**Crossover** between weights and KV happens when:
```
B · S · per_seq_token_bytes ≈ weight_bytes
```
For Llama-3-70B FP16, ≈ 140 GB / (320 KB · S) → at S=4K, B≈110 to balance. So at typical serving batch sizes, **KV reads dominate**.

Decisions like **MQA, GQA, MLA, FP8 KV, KV compression** all attack this bandwidth/capacity wall.

---

## 4. Layout in Memory

Naive contiguous layout: `kv_cache[layer, k_or_v, batch, head, seq, d_head]` — but this fragments badly when sequences have different lengths and when requests come and go.

### 4.1 Padded Contiguous (legacy)

```
kv_cache: shape (B, max_seq, L, 2, H_kv, d)
```
- Pre-allocate for max_seq. Wasted memory if requests are short. Cannot easily evict or migrate.
- Used by HF Transformers' default `past_key_values`.

### 4.2 Paged (vLLM PagedAttention)

Cache is divided into fixed-size **blocks** (a.k.a. pages), typically 16 tokens worth of K,V:

```
block_shape = (block_size, L, 2, H_kv, d)
```

The system holds a pool of physical blocks. Each sequence has a **block table**: a list mapping logical block indices → physical block indices. When a sequence appends a token, it grabs a fresh physical block if needed.

Benefits:
- No internal fragmentation beyond the partial last block.
- Sequences can grow independently.
- Easy to share blocks (prefix caching, see §5).
- Supports preemption: kick out a sequence's blocks, swap to CPU, reload later.

Costs:
- Indirection: attention kernel must dereference the block table per logical block.
- Block-size tuning: too small → indirection overhead; too big → fragmentation.

### 4.3 Block-Sparse / Token-Granularity (newer)

Some systems (e.g. SGLang's RadixAttention) use a tree-structured cache to support arbitrary prefix sharing. Tokens, not blocks, are the unit — implemented as fixed pages but indexed by a radix tree of token sequences.

---

## 5. Prefix Caching & Sharing

Many requests share a prefix:
- A long system prompt.
- A few-shot template.
- A retrieved document, identical across users.
- Multi-turn chat: every turn re-includes the entire conversation.

**Prefix caching** stores the KV of seen prefixes and reuses them across requests.

### 5.1 Hash-Based (vLLM, TRT-LLM)

For each block, hash `(parent_hash, token_ids_in_block)`. If a new request's first block's hash matches a cached block, point to the same physical block. Walk forward greedily until the hashes diverge. Skipped tokens save full prefill compute.

### 5.2 Radix Tree (SGLang RadixAttention)

Build a trie keyed on token sequences; each node owns the KV of its tokens. Hits common prefixes naturally and supports sub-block matches. Adds bookkeeping but excels for chat replay where prefixes stack.

### 5.3 Hit Rate Effects

Realistic hit rates from production reports:
- Long system prompts: 30–80% bytes saved.
- Multi-turn chat: 50–90% on the second+ turn.
- RAG with long retrieved chunks: depends on chunk reuse.

Prefix caching is a **multi-x throughput improvement** when active and is essentially free given paging — every modern serving stack ships it.

---

## 6. KV Cache Compression Schemes

### 6.1 GQA / MQA (Architectural)

Use fewer KV heads than Q heads (Llama-3: 4× sharing factor; Mistral, GPT-4 rumored 8×). Multiplies the KV memory reduction directly. See [Attention_Mechanisms §5](../Part_5_Algorithms_and_Quantization/Attention_Mechanisms.md).

### 6.2 MLA (Multi-Head Latent Attention)

DeepSeek-V2/V3 trick: store a low-rank latent `c_KV` of dim ~512 plus a small rotary part, instead of full K and V. At attention time, project up.

```
KV_bytes_per_token ≈ 2 · L · (d_c + d_rope) · b   # vs 2·L·H·d for MHA
```

For DeepSeek-V3: `(512 + 64) · 2 · 61 · 2 ≈ 140 KB` vs `2 · 128 · 128 · 61 · 2 ≈ 4 MB` for naive. ~30× reduction.

Cost: extra matmuls inside attention; offset because KV bytes dominate.

### 6.3 KV Quantization

Store K, V in INT8 / FP8 / INT4 with per-token or per-head scales:
- **FP8 KV**: trivial drop-in via Hopper's E4M3 — typical 1–2% perplexity hit.
- **INT8 KV**: needs careful scaling; some perplexity hit on long contexts.
- **INT4 KV with KIVI / SmoothQuant-KV**: aggressive, 4× over FP16, more accuracy risk.

Production stacks (TRT-LLM, vLLM, SGLang) all support FP8 KV; INT4 KV is opt-in for capacity-constrained deployments.

### 6.4 Token Eviction (StreamingLLM, H2O, etc.)

Drop "unimportant" KV entries (low attention scores, far past). Useful for very long contexts but shifts the model off-distribution. Mostly research-grade for now.

### 6.5 Sliding Window

Architectural: only the last W tokens are kept (Mistral W=4096). Hard cap on KV memory at the cost of true long context.

---

## 7. Offload to CPU / NVMe

When HBM is exhausted, frameworks can spill less-active KV blocks to:

1. **CPU RAM** — typical bandwidth 50–500 GB/s aggregate via PCIe Gen5 (~64 GB/s/GPU). Dynamo and vLLM support swap_in/swap_out.
2. **NVMe** — ~10–25 GB/s, only viable for cold prefixes (large RAG corpora, multi-tenant warm pools).
3. **Remote GPU memory via NVLink** — across NVL72 in NVL domain, an SM can read another GPU's HBM at ~1.8 TB/s sustained. Used by DeepSeek for expert sharding.
4. **NIXL** (NVIDIA Inference Xfer Library) — a transport that abstracts GPU↔GPU, GPU↔CPU, GPU↔storage moves with the same API. See [Networking_and_Interconnect](../Part_3_Hardware_Systems_and_Interconnects/Networking_and_Interconnect.md).

The fundamental tradeoff: every offloaded byte must be paid back when the request becomes active again. Useful for **prefix pools** where hit rates are high; risky for **active sequences** because the bandwidth gap is huge.

---

## 8. PagedAttention Kernel (Sketch)

```
# Per request, given block_table[seq] : list of phys_block_ids, len = ceil(S/B)

def paged_attention_decode(q, K_pages, V_pages, block_table, slot_in_last_block):
    # q: (1, head, d)  the new token only
    O = zeros(head, d)
    m = -inf; ell = 0
    for logical in range(num_logical_blocks(seq)):
        phys = block_table[logical]
        K_blk = K_pages[phys]  # (block_size, head, d)
        V_blk = V_pages[phys]

        # Apply mask if logical is the last block (only positions ≤ slot are valid)
        s = q @ K_blk.transpose(-1, -2) / sqrt(d)
        if logical == last:
            s[:, slot_in_last_block+1:] = -inf

        m_new = max(m, s.max())
        alpha = exp(m - m_new)
        p = exp(s - m_new)
        ell = alpha * ell + p.sum()
        O = alpha * O + (p @ V_blk)
        m = m_new

    return O / ell
```

Both v1 of PagedAttention (kernel by vLLM team) and FlashDecoding-paged variants follow this pattern. The block table indirection adds <5% overhead vs contiguous.

---

## 9. Eviction Policies

When the block pool is exhausted:

- **LRU on prefix-cache blocks** — evict cold cached prefixes first; they can be recomputed.
- **Preempt sequences** — lowest-priority active request gets its blocks freed; either swap to CPU or recompute on next schedule. vLLM uses recomputation for new requests, swap for in-flight ones it can't afford to drop.
- **Prefer evicting before throttling** — admission control should refuse new requests rather than thrash the cache.

Recomputation cost: a partial prefill at the saved length, paid only when the sequence is rescheduled.

---

## 10. KV Movement in Disaggregated Inference

In **prefill-decode disaggregation** (see [Prefill_Decode_Disaggregation](Prefill_Decode_Disaggregation.md)), the prefill GPU produces a full KV cache for the prompt and must hand it to the decode GPU. This is a multi-GB transfer per request.

Options:
- **NVLink** — when both GPUs are in the same NVL domain, ~900 GB/s P2P. Cheap.
- **InfiniBand / RoCE RDMA** — across nodes, ~50–400 GB/s per NIC. NIXL handles this.
- **Layer-pipelined** — start streaming layer-0 KV while layer-31 is still computing; decoder starts attending as layers arrive. Hides much of the transfer.

A 70B prefill at 4K tokens ≈ 1.25 GB KV. At 200 GB/s effective, ~6 ms — small vs the prefill time itself (~50 ms), so disaggregation is feasible.

---

## 11. Common Pitfalls

- **Forgetting GQA factor**: people compute KV using H_q rather than H_kv. For Llama-3-70B that's an 8× overestimate.
- **Wrong dtype assumption**: weights might be FP8 but KV often stays FP16 (or BF16). Check the runtime config.
- **Ignoring layer count**: 70B has 80 layers vs 8B's 32 — the same per-token bytes scales linearly with depth.
- **Counting padding**: paging eliminates per-sequence padding but the last block can still be partial.
- **MLA confusion**: MLA's compressed cache is per-layer scalar `(d_c + d_rope)`, not per-head. Don't multiply by H.

---

## 12. Common Interview Questions

**Q: Why do we need a KV cache, and what is its size for Llama-3-70B at 8K tokens, batch 32, FP16?**
A: Without it, each decode step re-runs full attention over the prompt, which is O(N²·d) per step. With a cache it becomes O(N·d) per step. Size: 320 KB/token (using the formula `2·L·H_kv·d·b` with L=80, H_kv=8, d=128, b=2) × 8192 × 32 ≈ 80 GB.

**Q: What is GQA and why does it help inference?**
A: Grouped-Query Attention shares K and V tensors across groups of Q heads. With H_kv = H_q / g, KV memory is reduced by `g`× (typically 4×–8×). At decode this is a direct bandwidth and capacity win.

**Q: What is PagedAttention and why is it useful?**
A: It splits the KV cache into fixed-size blocks held in a global pool, with per-sequence block tables mapping logical positions to physical blocks. Eliminates fragmentation, enables variable-length batching, supports prefix sharing, and allows preemption. Used by vLLM, copied by virtually every modern serving stack.

**Q: Walk me through prefix caching.**
A: Hash each block by `(parent_hash, token_ids)`. When a new request arrives, walk its prompt block by block; if the hash matches an existing block in the pool, increment a refcount and reuse it. Stop matching at the first divergence. The new request only prefills the suffix — multi-x throughput in chat / RAG / long-system-prompt workloads.

**Q: How does MLA differ from GQA?**
A: GQA reduces the **number of KV heads** (architectural sharing). MLA stores a **low-rank latent** representation of K,V (compression) and reconstructs at attention time. MLA achieves much higher KV compression (~30× over MHA in DeepSeek-V3) at the cost of extra projection compute.

**Q: When does KV cache reads dominate over weight reads in decode?**
A: When `B · S · per_seq_token_bytes > weight_bytes`. For Llama-70B FP16, around B·S ≈ 440K — i.e. batch 110 at S=4K, or batch 50 at S=8K. Production batches usually live past the crossover, so KV BW dominates.

**Q: What's the kernel overhead of the block-table indirection in PagedAttention?**
A: Typically <5% vs contiguous because the lookup (1 int per logical block) is cheap and block sizes are large enough (16 tokens) to amortize.

**Q: How would you decide between FP8 KV and INT8 KV?**
A: FP8 (E4M3) keeps an exponent → handles long-tail magnitudes from softmax better; on Hopper it's almost free in the tensor cores. INT8 needs per-token or per-head scale factors; risk of accuracy loss on long contexts. Default to FP8 on Hopper; keep INT8 only if you target Ampere or older.

**Q: How is the KV cache moved in prefill-decode disaggregated inference?**
A: Via NVLink intra-node (~900 GB/s P2P), RDMA over IB/RoCE inter-node (~50–400 GB/s per NIC). NIXL provides a unified API. Layer-pipelined transfer overlaps with compute to hide latency.

**Q: What is preemption in the context of LLM serving?**
A: When the cache is full, the scheduler kicks out a low-priority running sequence — either freeing its blocks (forces recomputation when rescheduled) or swapping them to CPU memory. This avoids OOM and lets the system trade latency for throughput.

**Q: Why is prefix caching essentially free with paging but expensive with contiguous layout?**
A: Pages are sharable with refcounts. Contiguous layouts would require copying or fragile pointer aliasing across separate per-request allocations.

**Q: A user is hitting OOM on long contexts. List 5 mitigations.**
A: (1) Switch to GQA/MLA model, (2) FP8 KV cache, (3) reduce max_seq or chunked prefill, (4) FP8 weights to free HBM, (5) enable swap-to-CPU or NVMe offload, (6) shrink max-batch.

**Q: Why is the KV cache the dominant capacity bottleneck rather than weights?**
A: Weights are O(model_size) — fixed. KV is O(B·S·model_depth·H_kv·d) — scales with batch and context. As context windows grow (32K, 128K, 1M) and batch sizes grow, KV easily exceeds weights. Hence the industry obsession with KV reduction.

**Q: What's "in-flight batching" / "continuous batching" and how does it interact with KV?**
A: Sequences enter and leave the active batch each step instead of waiting on a fixed cohort. KV cache must support per-sequence variable-length state — paging makes this easy because each sequence holds its own block-table; ragged contiguous layouts can't.

---

## 13. Further Reading

- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (SOSP 2023).
- Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs" (NeurIPS 2024) — RadixAttention.
- DeepSeek-V2 / V3 technical reports — MLA derivation.
- KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache (2024).
- Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (EMNLP 2023).

---

**Next:** [Batching_and_Scheduling](Batching_and_Scheduling.md).
**See also:** [Attention_Mechanisms](../Part_5_Algorithms_and_Quantization/Attention_Mechanisms.md), [Quantization](../Part_5_Algorithms_and_Quantization/Quantization.md), [Prefill_Decode_Disaggregation](Prefill_Decode_Disaggregation.md).

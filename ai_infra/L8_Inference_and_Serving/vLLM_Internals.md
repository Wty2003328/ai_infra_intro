# vLLM Internals — Engine Architecture, Scheduler, and Block Manager

> **Layer:** L8. **Prerequisites:** [Inference_Frameworks](Inference_Frameworks.md), [KV_Cache](KV_Cache.md), [Batching_and_Scheduling](Batching_and_Scheduling.md). **Hands off to:** [Production_Architecture](Production_Architecture.md), [Kubernetes_and_Orchestration](Kubernetes_and_Orchestration.md).

---

## 0. Why This Page Exists

vLLM is the most studied open-source LLM inference engine and the reference implementation for PagedAttention, continuous batching, and automatic prefix caching. Understanding its internals is a forcing function for understanding every other serving system: SGLang, TensorRT-LLM, Dynamo, and llm-d all implement variants of the same scheduler--block-manager--worker decomposition.

This page is a code-level architectural tour. It traces a single request from HTTP arrival to streamed output, dissects the V1 engine's process topology, derives the scheduler's per-step decision loop, explains the block manager's virtual-memory-inspired allocation and copy-on-write prefix sharing, and maps every component back to its source module. Every design choice is grounded in the scheduling theory from [Batching_and_Scheduling](Batching_and_Scheduling.md) and the memory math from [KV_Cache](KV_Cache.md).

### Invariants

| Symbol | Meaning | Typical range |
|--------|---------|---------------|
| $B_{\text{tok}}$ | Per-step token budget (`max_num_batched_tokens`) | 2048--8192 |
| $B_{\text{seq}}$ | Maximum concurrent sequences (`max_num_seqs`) | 64--4096 |
| $B_s$ | KV block size in tokens | 16 (default) |
| $M_{\text{KV}}$ | Total KV cache memory pool | 10--85 GB |
| $N_{\text{blk}}$ | Number of physical KV blocks in pool | 5K--40K |
| $n_l$ | Transformer layers | 32--126 |
| $n_{\text{kv}}$ | KV heads per layer | 1--128 |
| $d_h$ | Head dimension | 64--256 |
| $b$ | Bytes per KV element | 1 (FP8) or 2 (FP16/BF16) |

---

## 1. V1 Engine Architecture

### 1.1 Process topology

The V1 engine (shipped late 2024, default since v0.6) separates the scheduling critical path from the model execution path by placing them in different processes connected via shared-memory queues.

```mermaid
flowchart TD
    CLIENT["Client<br/>HTTP / gRPC / SDK"] -->|"POST /v1/chat/completions"| FRONTEND["Frontend Process<br/>AsyncLLMEngine<br/>tokenization, request obj"]
    FRONTEND -->|"zmq push<br/>Request proto"| CORE["EngineCore Process<br/>scheduler + block manager"]
    CORE -->|"zmq push<br/>ModelInput proto"| WORKER0["Worker Process 0<br/>GPU 0<br/>ModelRunner"]
    CORE -->|"zmq push<br/>ModelInput proto"| WORKER1["Worker Process 1<br/>GPU 1<br/>ModelRunner"]
    CORE -->|"zmq push<br/>ModelInput proto"| WORKERN["Worker Process N-1<br/>GPU N-1<br/>ModelRunner"]
    WORKER0 -->|"zmq push<br/>SampleResult"| CORE
    WORKER1 -->|"zmq push<br/>SampleResult"| CORE
    WORKERN -->|"zmq push<br/>SampleResult"| CORE
    CORE -->|"zmq push<br/>RequestOutput"| FRONTEND
    FRONTEND -->|"SSE stream"| CLIENT

    style CORE fill:#dbeafe,stroke:#1e40af,color:#000
    style FRONTEND fill:#fef3c7,stroke:#92400e,color:#000
    style WORKER0 fill:#d1fae5,stroke:#065f46,color:#000
    style WORKER1 fill:#d1fae5,stroke:#065f46,color:#000
    style WORKERN fill:#d1fae5,stroke:#065f46,color:#000
```

Each Worker process is pinned to one GPU and holds its own model-weight copy (or shard under tensor parallelism). The EngineCore process owns the scheduler and block manager -- it never touches the GPU. Communication is entirely via zero-copy shared-memory queues (multiprocessing queues for intra-node, zmq for the API layer), eliminating Python GIL contention between scheduling and execution.

### 1.2 Component responsibilities

| Component | Process | Responsibilities |
|-----------|---------|-----------------|
| AsyncLLMEngine | Frontend | HTTP/gRPC API, tokenization, Request object creation, output streaming |
| EngineCore | Core | Admission, scheduling, KV block allocation, prefix cache lookup, preemption |
| Worker | Per-GPU | ModelRunner forward pass, KV cache tensors, sampling, custom kernel dispatch |
| ModelRunner | Inside Worker | Layer-by-layer execution, input tensor packing, slot mapping |

### 1.3 Why the split?

The V0 engine ran scheduler and workers in the same Python process. The GIL serialized Python work on the scheduling path against Python work on the post-processing path. At high batch sizes this added 2--5 ms of scheduling overhead per step -- material when decode steps themselves are 20--50 ms. The V1 split removes scheduling from the GPU-critical path entirely: by the time the Worker finishes one forward pass, the next step's ModelInput is already waiting in the queue.

Performance impact: roughly 1.5--2$\times$ higher peak throughput at identical model code, with the gains concentrated at high batch sizes where scheduling overhead was previously the binding constraint.

---

## 2. Request Lifecycle — End to End

### 2.1 Arrival

The client sends an OpenAI-compatible request:

```json
POST /v1/chat/completions
{
  "model": "meta-llama/Llama-3.1-70B-Instruct",
  "messages": [{"role": "system", "content": "You are..."}, {"role": "user", "content": "Explain..."}],
  "stream": true,
  "max_tokens": 512,
  "temperature": 0.7
}
```

The frontend tokenizes the chat template into token IDs, wraps them into an internal `Request` object (request ID, token IDs, sampling parameters, output stream reference), and pushes it to the EngineCore via shared memory.

### 2.2 Waiting queue

The EngineCore appends the request to the **waiting queue**. The request remains here until the scheduler admits it (Section 3). If the queue is full or the system is overloaded, the frontend returns HTTP 429 immediately.

### 2.3 Admission and prefix cache lookup

On the next scheduling step, the scheduler evaluates admission (Section 3.1). If admitted, the block manager computes block hashes for the prompt's token IDs and checks the prefix cache (Section 5). Matching blocks are shared via reference-count increment; the unmatched suffix becomes the prefill workload.

### 2.4 Prefill

The scheduler allocates a prefill chunk: up to $B_{\text{tok}}$ tokens of the prompt suffix, minus tokens consumed by concurrent decode sequences. For a 4096-token prompt with $B_{\text{tok}} = 2048$ and 32 active decodes: first chunk gets $\min(4096, 2048 - 32) = 2016$ tokens. The remaining 2080 tokens are deferred to subsequent steps.

The scheduler builds a `ModelInput` containing packed token IDs, position IDs, per-sequence block tables, and slot mappings (which physical cache slot each token writes to). This is dispatched to all workers.

### 2.5 Forward pass and KV write

Each worker's ModelRunner executes the forward pass layer by layer. At each attention layer, the new $K$, $V$ vectors are written into the physical KV cache at the slot locations specified by the slot mapping. The PagedAttention kernel reads existing cached $K$, $V$ through the block table and computes attention output for the new tokens.

### 2.6 Sampling and output

After the final layer, logits are computed ($\text{lm\_head}$ projection). The sampler applies temperature, top-$k$, top-$p$, repetition penalties, and grammar masks per-sequence on GPU. Sampled token IDs are returned to the EngineCore. If the request streams, the frontend pushes the token to the client immediately.

### 2.7 Decode loop

On subsequent steps, the scheduler includes the request in the decode batch (1 token budget per sequence). Each step produces one more token. The request remains in the **running** set until EOS, stop string, or `max_tokens`.

### 2.8 Completion

When the sampler emits EOS or `max_tokens` is reached, the request moves to the **finished** state. Its block table is released: each block's reference count is decremented. Blocks with refcount zero return to the free list (or remain in the prefix cache if they are hash-registered). The final output is streamed to the client.

```mermaid
stateDiagram-v2
    [*] --> Waiting : Request arrives
    Waiting --> Running : Scheduler admits<br/>(KV budget + token budget)
    Running --> Running : Decode step<br/>(1 token per step)
    Running --> Waiting : Preempted<br/>(recompute or swap)
    Running --> Finished : EOS / max_tokens / stop
    Finished --> [*] : Blocks released<br/>refcount decremented
```

---

## 3. The Scheduler

### 3.1 Per-step decision loop

The scheduler runs once per step and must decide: which waiting requests to admit, which running sequences to schedule, and whether any preemption is necessary. The decision is greedy and order-dependent.

```mermaid
flowchart TD
    START["Step begins"] --> SWAP_IN["Swap in preempted sequences<br/>if swap space available"]
    SWAP_IN --> ADMIT{"Waiting queue<br/>non-empty?"}
    ADMIT -->|Yes| CHECK_BUDGET{"Token budget +<br/>KV budget available?"}
    CHECK_BUDGET -->|Yes| PREFIX["Prefix cache lookup<br/>allocate blocks"]
    PREFIX --> ADMIT_REQ["Move request to Running"]
    ADMIT_REQ --> ADMIT
    CHECK_BUDGET -->|No| BUILD["Build batch"]
    ADMIT -->|No| BUILD
    BUILD --> CHECK_KV{"Enough KV blocks<br/>for batch?"}
    CHECK_KV -->|Yes| DISPATCH["Dispatch ModelInput<br/>to Workers"]
    CHECK_KV -->|No| PREEMPT["Preempt victim<br/>(recompute or swap)"]
    PREEMPT --> BUILD
    DISPATCH --> RECEIVE["Receive SampleResults<br/>from Workers"]
    RECEIVE --> UPDATE["Update sequences<br/>append tokens<br/>check EOS"]
    UPDATE --> FINISHED{"Any requests<br/>finished?"}
    FINISHED -->|Yes| RELEASE["Release blocks<br/>decrement refcounts"]
    RELEASE --> START
    FINISHED -->|No| START

    style DISPATCH fill:#dbeafe,stroke:#1e40af,color:#000
    style PREEMPT fill:#fecaca,stroke:#991b1b,color:#000
    style RELEASE fill:#d1fae5,stroke:#065f46,color:#000
```

### 3.2 Budget enforcement

At each step the scheduler enforces three hard constraints:

$$\sum_{\text{prefill chunks}} C_i + |\text{decodes}| \le B_{\text{tok}}$$

$$|\text{running sequences}| \le B_{\text{seq}}$$

$$\sum_{i \in \text{running}} \left\lceil \frac{S_i}{B_s} \right\rceil \le N_{\text{blk}}$$

The token budget ($B_{\text{tok}}$) limits total compute per step. The sequence budget ($B_{\text{seq}}$) caps the batch dimension. The block budget ($N_{\text{blk}}$) limits total KV memory.

### 3.3 Chunked prefill within the scheduler

A request with a prompt longer than the remaining token budget is split across steps. The scheduler tracks how many prompt tokens have been prefilled so far and only schedules the remaining chunk. Between chunks, the request stays in the running set but does not consume decode tokens -- it is purely in the prefill phase.

The interleaving of prefill chunks with decode tokens is governed by the token budget. If 32 decode sequences consume 32 tokens from the budget and $B_{\text{tok}} = 4096$, the prefill chunk gets at most 4064 tokens. This ensures decode TPOT is not disrupted by large prefills, as analyzed in [Batching_and_Scheduling](Batching_and_Scheduling.md) Section 5.

### 3.4 V1 vs V0 scheduler differences

The V0 scheduler was a single Python class (`Scheduler`) that maintained separate `self.waiting`, `self.running`, and `self.swapped` deques. It ran inside the same Python process as the API server and workers, serialized by the GIL. The V1 scheduler introduces several fundamental changes:

| Dimension | V0 Scheduler | V1 Scheduler |
|-----------|-------------|--------------|
| Process | Same Python process as API server + workers | Dedicated EngineCore process |
| Language | Pure Python (Scheduler class) | C++ core with Python bindings |
| Queue structure | Separate `waiting`, `running`, `swapped` deques | Unified priority queue with state flags |
| Prefill/decode distinction | Explicit: requests move between `waiting` (prefill) and `running` (decode) | Implicit: every request in the running set has a `num_computed_tokens` counter; prefill vs decode is determined by whether `num_computed_tokens < total_num_tokens` |
| Scheduling overhead | ~500 $\mu$s/step (Python iteration + GIL contention) | ~50 $\mu$s/step (C++ iteration, no GIL) |
| Chunked prefill | Supported but required explicit `running_queue` management for partial prefills | Native: partial-prefill sequences stay in the unified queue with their `num_computed_tokens` counter |
| Communication | Python `multiprocessing.Queue` (pickle serialization) | Zero-copy shared-memory queues (zmq) |

**Unified scheduling in V1.** The V0 scheduler treated prefill and decode as distinct phases with separate code paths. In V1, the scheduler makes no explicit distinction: it simply iterates over all requests in the running set and checks each request's state:

```
for req in running_set:
    if req.num_computed_tokens < req.num_prompt_tokens:
        # This request still needs prefill
        chunk = min(remaining_tokens, tokens_budget)
        schedule_prefill_chunk(req, chunk)
        tokens_budget -= chunk
    else:
        # This request is in decode phase
        if tokens_budget >= 1:
            schedule_decode(req)
            tokens_budget -= 1
```

This eliminates the V0 pattern of shuffling requests between separate waiting/running/swapped lists. The unified queue simplifies the code and avoids list-management overhead.

**How V1 handles chunked prefill internally.** When a request with a 6K-token prompt arrives and the token budget is 4096 with 48 active decodes:

1. V1 computes the available prefill budget: $4096 - 48 = 4048$ tokens.
2. The first chunk processes 4048 tokens. After the forward pass, `req.num_computed_tokens = 4048`.
3. The request stays in the running set. On the next step, the scheduler sees `num_computed_tokens (4048) < num_prompt_tokens (6144)` and schedules the remaining 1952 tokens as chunk 2.
4. After chunk 2, `num_computed_tokens = 6144 = num_prompt_tokens`. The first output token is sampled. On subsequent steps, the request is decoded normally.

No explicit state machine transitions between "waiting," "prefilling," and "decoding" -- the `num_computed_tokens` counter implicitly encodes the phase.

### 3.5 Preemption decisions

When the block pool is exhausted but a higher-priority request must be admitted, the scheduler selects a victim for preemption. The decision matrix:

| Victim type | Mechanism | When preferred |
|-------------|-----------|---------------|
| New request not yet decoded | Recompute (drop blocks) | Always: no investment lost |
| Decode sequence, short context | Recompute | Re-prefill cost is low |
| Decode sequence, long context, low prefix hit | Swap to CPU | Preserves expensive KV |
| Decode sequence, long context, high prefix hit | Recompute | Prefix cache absorbs most re-prefill cost |

Victim selection among equal-priority candidates: largest KV footprint first (maximizes freed blocks per eviction). Among different priorities: always preempt the lowest priority.

### 3.6 Scheduling policy

V1 uses FCFS (first-come-first-served) as the default admission policy. Priority can be set per-request via the API, mapping to integer priority levels. Within the same priority level, FCFS applies. The scheduler does not implement SRJF or EDF natively; these are left to the routing layer above (see [Production_Architecture](Production_Architecture.md)).

---

## 4. The Worker and Model Runner

### 4.1 Worker process

Each Worker is a Python subprocess pinned to one GPU. It holds:

- **Model weights**: loaded at startup, possibly sharded for tensor parallelism.
- **KV cache tensors**: pre-allocated per-layer tensors of shape $(N_{\text{blk}}, 2, B_s, n_{\text{kv}} / \text{TP}, d_h)$.
- **ModelRunner**: the object that orchestrates the forward pass.

The Worker receives a `ModelInput` from the EngineCore, calls `ModelRunner.execute()`, and returns `SampleResult` (sampled token IDs and related metadata).

### 4.1.1 How the LLMEngine dispatches to workers

The EngineCore communicates with Workers via shared-memory queues (V1) or multiprocessing queues (V0). The dispatch flow:

1. **EngineCore** runs the scheduler (Section 3), producing a `SchedulerOutput` containing: which sequences are scheduled, their prefill chunks, and any preemption/swapping directives.
2. The `SchedulerOutput` is serialized into a `ModelInput` proto containing packed `input_ids`, `positions`, `block_tables`, `slot_mapping`, and `seq_lens` tensors.
3. The `ModelInput` is pushed to each Worker's input queue via zero-copy shared memory. Under TP, all workers receive the *same* `ModelInput` (identical input IDs and block tables; each worker uses its own weight shard).
4. Each Worker's `ModelRunner.execute()` deserializes the `ModelInput`, moves tensors to GPU, and runs the forward pass.
5. After the forward pass, the sampler produces `SampleResult` (one token ID per sequence). This is pushed back to the EngineCore via a return queue.
6. The EngineCore updates sequence states, emits streaming outputs, and begins the next scheduling step.

Under TP, Workers execute in lockstep because the forward pass includes NCCL all-reduce operations that synchronize all ranks. The EngineCore does not proceed to the next step until all Workers have returned their `SampleResult`.

### 4.1.2 Model weight loading

At startup, each Worker loads model weights through the following path:

1. **Safetensors scan**: read the model's `model.safetensors.index.json` to determine the weight file layout and tensor metadata.
2. **Shard selection**: under TP, each Worker loads only its shard of each tensor. For column-parallel layers (QKV, gate, up projections), Worker $i$ loads column range $[i \cdot H/\text{TP}, (i+1) \cdot H/\text{TP})$. For row-parallel layers (output, down projections), Worker $i$ loads row range $[i \cdot d/\text{TP}, (i+1) \cdot d/\text{TP})$.
3. **Quantization**: if `--quantization` is set, weights are quantized during loading (e.g., FP16 $\to$ FP8 via absmax per-tensor or per-channel).
4. **GPU transfer**: each weight tensor is moved to GPU via `cudaMemcpy` (PCIe) or direct HBM mapping. Total load time: ~30 seconds for a 70B model from local NVMe, dominated by storage read bandwidth.
5. **Memory accounting**: after all weights are loaded, the remaining GPU memory is computed and the KV cache tensor is allocated to fill the available space (subject to `gpu_memory_utilization`).

### 4.1.3 Attention backend selection

vLLM supports multiple attention backends, selected at startup based on GPU architecture, block size, and kernel availability:

| Backend | When selected | Implementation |
|---------|--------------|----------------|
| FlashAttention v3 | Hopper GPUs (H100+), FP16/BF16 KV | `flash_attn_varlen_func` with paged KV support |
| FlashAttention v2 | Ampere+ GPUs (A100+), FP16/BF16 KV | `flash_attn_varlen_func` with block-table wrapper |
| xFormers | Ampere GPUs, custom block sizes | `memory_efficient_attention_forward` with paged wrapper |
| ROCm FlashAttention | AMD MI300X, MI250 | ROCm-ported FlashAttention kernel |
| PagedAttention v1/v2 (custom CUDA) | Fallback for all GPUs | `csrc/attention/attention_kernels.cu` |

Selection priority: FlashAttention v3 > FlashAttention v2 > xFormers > custom PagedAttention. The selection happens in `vllm/attention/selector.py` based on `torch.cuda.get_device_capability()`, KV dtype, and block size. Users can override via `--attention-backend`.

The PagedAttention kernel wraps the selected backend by: (1) resolving block tables to physical memory addresses, (2) constructing the `seq_lens` and `block_tables` tensors in the format expected by the backend, and (3) handling the slot mapping for KV writes. The backend kernel itself is unaware of paging -- the indirection is handled in the wrapper layer.

### 4.2 ModelRunner forward pass

The ModelRunner unpacks the `ModelInput` into GPU tensors and executes the model layer by layer:

| Tensor | Shape | Purpose |
|--------|-------|---------|
| `input_ids` | $(T,)$ | Packed token IDs for all sequences ($T = \sum$ tokens in step) |
| `positions` | $(T,)$ | Position IDs (RoPE input) |
| `block_tables` | $(B, \text{max\_blocks})$ | Per-sequence logical-to-physical block mapping |
| `slot_mapping` | $(T,)$ | Physical KV slot for each token's new K, V |
| `seq_lens` | $(B,)$ | Current cached length per sequence |

At each layer:

1. **RMSNorm** on hidden states.
2. **QKV projection** (column-parallel under TP): $x \to Q, K, V$.
3. **RoPE**: apply rotary positional embeddings to $Q$ and $K$.
4. **KV write**: scatter new $K$, $V$ into the cache at `slot_mapping` locations.
5. **PagedAttention**: attend over cached $K$, $V$ via `block_tables`.
6. **Output projection** (row-parallel under TP, followed by all-reduce).
7. **Residual addition** + **RMSNorm** (post-attention).
8. **MLP** (gate/up projections, SwiGLU activation, down projection; TP all-reduce after down).
9. **Residual addition**.

After the final layer, the **LM head** projects hidden states to logits of shape $(T, V)$ where $V$ is the vocabulary size. For prefill sequences, only the last token's logits are needed (the rest produced KV but do not generate output). The sampler operates on these final logits.

### 4.3 Sampler

The sampler applies per-sequence parameters entirely on GPU:

1. Scale logits by temperature: $\ell_i / \tau$.
2. Apply repetition, frequency, and presence penalties by scanning recent token IDs.
3. Apply logit bias (per-token additive adjustments).
4. Apply structured-output mask (xgrammar / Outlines grammar constraints).
5. Top-$k$ filtering: retain only the $k$ highest logits.
6. Top-$p$ (nucleus) filtering: cumulative probability cutoff.
7. Multinomial sample (or argmax for $\tau = 0$).

All stages operate on batched tensors with no host round-trip. The sampler returns one token ID per sequence.

---

## 5. The Block Manager

### 5.1 BlockSpaceManager

The `BlockSpaceManager` is the single owner of all KV cache memory. It maintains:

- **Free list**: a stack of available physical block IDs.
- **Block tables**: `Dict[request_id, List[int]]` mapping each sequence's logical block indices to physical block IDs.
- **Reference counts**: `Dict[block_id, int]` tracking how many sequences or cache entries reference each block.
- **Prefix hash table**: `Dict[block_hash, block_id]` for automatic prefix caching.

### 5.2 Physical vs virtual blocks

The analogy to OS virtual memory is exact:

| OS Virtual Memory | vLLM Block Manager |
|-------------------|-------------------|
| Virtual address space | Logical token positions (0, 1, ..., $S-1$) |
| Physical pages | Physical KV blocks in the pre-allocated tensor |
| Page table | Block table per sequence |
| Page fault (demand paging) | Block allocation on first write to a logical slot |
| Shared memory (`mmap`) | Prefix-shared blocks (refcounted) |
| Swap to disk | Swap to CPU host memory |

Logical block $j$ of a sequence spans token positions $[j \cdot B_s,\; (j+1) \cdot B_s - 1]$. When a sequence generates a token at position $p$ that falls into a new logical block, the block manager pops a fresh physical block from the free list and appends it to the sequence's block table.

### 5.3 Memory math

The pre-allocated KV cache tensor per layer per GPU has shape:

$$\text{cache\_shape} = (N_{\text{blk}},\; 2,\; B_s,\; n_{\text{kv}} / \text{TP},\; d_h)$$

Total memory:

$$M_{\text{KV}} = N_{\text{blk}} \times 2 \times B_s \times \frac{n_{\text{kv}}}{\text{TP}} \times d_h \times b \times n_l$$

The system computes $N_{\text{blk}}$ at startup:

$$N_{\text{blk}} = \left\lfloor \frac{M_{\text{GPU}} \times \text{gpu\_mem\_util} - W - M_{\text{act}}}{B_s \times 2 \times (n_{\text{kv}} / \text{TP}) \times d_h \times b \times n_l} \right\rfloor$$

where $M_{\text{GPU}}$ is total GPU HBM, $W$ is model weight memory, and $M_{\text{act}}$ is activation scratch space.

### 5.4 Slot mapping

For each token in the step, the slot mapping determines where in the physical KV cache to write the new $K$ and $V$ vectors. Given token at logical position $p$ in a sequence with block table $T$:

$$\text{logical\_block} = \lfloor p / B_s \rfloor, \quad \text{offset} = p \mod B_s$$
$$\text{physical\_block} = T[\text{logical\_block}]$$
$$\text{slot} = \text{physical\_block} \times B_s + \text{offset}$$

The GPU kernel uses `slot_mapping` as a scatter index: `kv_cache[slot] = new_kv_value`. This is a single integer indirection per token -- negligible cost compared to the $d_h$-dimensional write.

---

## 6. Automatic Prefix Caching (APC)

### 6.1 Motivation

Production workloads share prompt prefixes across requests: system prompts, few-shot examples, RAG documents, multi-turn conversation history. Without caching, every request reprefills the entire prefix, wasting compute and inflating TTFT. With APC, the block manager reuses KV blocks computed by prior requests for any matching prefix.

### 6.2 Hash chain

Each block is identified by a content hash that encodes both the block's tokens and its position in the prefix:

$$h_j = \text{Hash}(h_{j-1} \;\|\;\; \text{token\_ids}[j \cdot B_s : (j+1) \cdot B_s])$$

with $h_{-1} = 0$ (null seed). The hash chain ensures that identical tokens at different prefix positions produce different hashes (because the parent hash differs), and that two prompts sharing a prefix share the same hash values for the shared blocks.

### 6.3 Copy-on-write sharing

When a new request's prefix hash matches an existing cached block, the block manager does **not** copy the KV data. Instead, it increments the block's reference count and points the new sequence's block table entry at the shared physical block.

```mermaid
flowchart TD
    subgraph "Request A (existing)"
        A0["Block 0<br/>hash: h₀<br/>refcount: 2"]
        A1["Block 1<br/>hash: h₁<br/>refcount: 2"]
        A2["Block 2<br/>(unique)<br/>refcount: 1"]
    end
    subgraph "Request B (new, shares prefix)"
        B0["Block 0<br/>same h₀<br/>refcount: 2"]
        B1["Block 1<br/>same h₁<br/>refcount: 2"]
        B2["Block 3<br/>(unique)<br/>refcount: 1"]
    end

    A0 -.->|"shared physical block #42"| B0
    A1 -.->|"shared physical block #107"| B1

    style A0 fill:#dbeafe,stroke:#1e40af,color:#000
    style B0 fill:#dbeafe,stroke:#1e40af,color:#000
    style A1 fill:#dbeafe,stroke:#1e40af,color:#000
    style B1 fill:#dbeafe,stroke:#1e40af,color:#000
    style A2 fill:#fef3c7,stroke:#92400e,color:#000
    style B2 fill:#fef3c7,stroke:#92400e,color:#000
```

Copy-on-write semantics: if a sequence later modifies a shared block (which cannot happen for KV cache -- blocks are append-only), the modification would trigger a physical copy first. In practice, KV blocks are immutable once written, so sharing is always safe without copies.

### 6.4 Lookup algorithm

When a new request is admitted:

1. Walk the prompt tokens block-by-block, computing $h_0, h_1, \ldots$.
2. For each $h_j$, probe the hash table:
   - **Hit**: increment `refcount[block_id]`, add `block_id` to the sequence's block table. Move to $h_{j+1}$.
   - **Miss**: stop. All tokens from this position onward must be prefilled. Allocate fresh physical blocks for them and register their hashes as they are computed.
3. The **prefix match length** is $j \cdot B_s$ tokens. The remaining prompt length is $S_{\text{prompt}} - j \cdot B_s$.

### 6.5 Eviction

Unreferenced cached blocks (refcount = 0) are eligible for eviction when the free list is empty. Eviction policy: LRU (least recently used). Evicted blocks are removed from the hash table and returned to the free list. The cost of eviction is zero for active requests because only unreferenced blocks are evicted.

### 6.6 Throughput impact

For a chat application with a 2048-token shared system prompt and 512-token average user message:

- **Without APC**: every request prefills 2560 tokens. Prefill FLOPs = $2 N \times 2560$.
- **With APC (full hit)**: only 512 tokens are prefilled. Prefill FLOPs = $2 N \times 512$.
- **Savings**: $(2560 - 512) / 2560 = 80\%$ of prefill compute eliminated.

TTFT drops proportionally. At 1000 req/s, this frees approximately one GPU's worth of prefill capacity.

### 6.7 APC integration with the V1 scheduler and block manager

APC in V1 is not a separate subsystem -- it is deeply integrated into the scheduler's admission path and the block manager's allocation path.

**Integration at admission time.** When the scheduler considers admitting a new request from the waiting queue (Section 3.1), the prefix cache lookup happens *before* block allocation:

1. The block manager walks the request's prompt tokens, computing hash chain values $h_0, h_1, \ldots$.
2. For each hash hit, the block manager increments the physical block's refcount and appends the block ID to the request's block table -- **without allocating new blocks**.
3. At the first hash miss, the walk stops. The remaining suffix length determines how many new blocks to allocate.
4. The scheduler now knows the *effective* prefill cost: only the suffix tokens need computation. If the suffix is small enough to fit in the remaining token budget, the request is admitted even if the full prompt would not fit.

This means APC directly affects admission control: a request with a 90% prefix cache hit has 10% of the prefill cost and requires 10% of the new blocks. The scheduler can admit more such requests than cold requests.

**Integration with KV block manager V1.** The block manager maintains:

- **Free list**: a stack (LIFO) of available physical block IDs, stored as a bitmask for $O(1)$ pop/push.
- **Block tables**: `Dict[request_id, List[int]]` -- one list per sequence, mapping logical block index to physical block ID.
- **Reference counts**: `Dict[int, int]` -- tracks how many sequences and cache entries reference each physical block.
- **Hash table**: `Dict[Tuple[int, bytes], int]` -- maps `(parent_hash, token_ids)` to physical block ID. The parent hash is the hash of the preceding block, ensuring positional uniqueness.
- **LRU list**: a doubly-linked list of unreferenced cached blocks, ordered by last access time. Supports $O(1)$ move-to-front on cache hit and $O(1)$ eviction of the tail.

**GPU memory pool management.** At startup, the block manager computes the total number of physical blocks:

$$N_{\text{blk}} = \left\lfloor \frac{M_{\text{GPU}} \times \text{gpu\_mem\_util} - W - M_{\text{act}}}{B_s \times 2 \times (n_{\text{kv}} / \text{TP}) \times d_h \times b \times n_l} \right\rfloor$$

The KV cache is allocated as a single contiguous tensor of shape $(N_{\text{blk}}, 2, B_s, n_{\text{kv}} / \text{TP}, d_h)$ per layer. The total memory reserved is:

$$M_{\text{KV}} = N_{\text{blk}} \times 2 \times B_s \times (n_{\text{kv}} / \text{TP}) \times d_h \times b \times n_l$$

This memory is pre-allocated at startup and never grows or shrinks. The block manager never calls `cudaMalloc` during serving -- all allocation is a metadata operation (pop from free list). This eliminates GPU memory fragmentation and makes allocation latency deterministic ($< 1\;\mu$s per block).

The split between KV cache memory and activation memory is determined by `gpu_memory_utilization` (default 0.90). The remaining 10% is reserved for activation tensors (intermediate layer outputs during the forward pass). If the activation spikes exceed this reservation, the forward pass triggers a CUDA OOM. This is why the default is 0.90 rather than 0.99.

**How APC reduces prefill compute.** When the prefix cache hits $M$ blocks for a new request:

1. The request's block table is pre-populated with $M$ shared physical blocks.
2. The scheduler constructs the `ModelInput` with `input_ids` containing only the suffix tokens (positions $M \cdot B_s$ through $S-1$).
3. The `positions` tensor maps these suffix tokens to their correct positions in the sequence (not starting from 0).
4. The PagedAttention kernel during the forward pass reads the cached prefix KV from the shared blocks and the newly computed suffix KV from the freshly allocated blocks.
5. The forward pass computes FLOPs proportional to the suffix length, not the full prompt length. For a 90% cache hit: 10% of the FLOPs.

---

## 7. Executor Backends

### 7.1 Single GPU

The simplest configuration: one EngineCore process, one Worker process. The Worker holds the full model and the full KV block pool. No inter-GPU communication.

### 7.2 Tensor parallelism (TP)

For models too large for one GPU, vLLM uses Megatron-style TP:

- **QKV projections**: column-parallel. Each rank computes $H / \text{TP}$ query heads and its share of KV heads.
- **Output projection**: row-parallel with all-reduce via NCCL.
- **MLP**: gate+up column-parallel; down row-parallel with all-reduce.
- **KV cache**: naturally sharded. Each rank stores $n_{\text{kv}} / \text{TP}$ heads' worth of KV data. Block pool is per-rank.

Workers run in lockstep: every step, all ranks receive the same ModelInput and execute the same forward pass. All-reduce at two points per layer (attention output, MLP output) via NCCL.

### 7.3 Pipeline parallelism (PP)

Less commonly used for inference (adds pipeline bubbles), but supported. The model is split into $P$ stages, each running on a separate GPU or GPU group. Micro-batches flow through stages sequentially. PP is beneficial when model depth is large enough to amortize the bubble overhead (e.g., 405B models on 8 GPUs).

### 7.4 Configuration mapping

| CLI flag | Executor | Communication |
|----------|----------|--------------|
| `--tensor-parallel-size 1` | UniProcExecutor | None |
| `--tensor-parallel-size N` (N > 1) | MPClient (multiprocess) | NCCL all-reduce |
| `--pipeline-parallel-size P` | PipelineParallel | Point-to-point (inter-stage) |
| Both TP and PP | Combined | NCCL within stage, P2P across stages |

---

## 8. The PagedAttention Kernel

### 8.1 Decode kernel

At decode time, each sequence contributes one query token. The kernel is launched with one threadblock per (sequence, head) pair. Each threadblock:

1. Loads the query vector $q \in \mathbb{R}^{d_h}$ into registers.
2. Iterates over the sequence's logical blocks, looking up the physical block ID from the block table.
3. For each physical block, loads $K$ and $V$ from the cache tensor.
4. Computes partial attention scores ($q \cdot K$), applies online softmax accumulation.
5. Produces the output vector $o \in \mathbb{R}^{d_h}$.

The block-table indirection adds one integer read per $B_s$ tokens. With $B_s = 16$ and $d_h = 128$, the indirection reads 4 bytes per $16 \times 128 \times 2 = 4096$ bytes of KV data -- an overhead below 0.1%.

### 8.2 Split-K parallelism (v2)

For long sequences where a single threadblock would iterate over many blocks, the v2 kernel splits the KV dimension across multiple threadblocks, each computing a partial attention output. A final reduction kernel combines the partial results. This improves SM utilization for $S > 4096$.

### 8.3 Prefill kernel

Prefill processes multiple query tokens per sequence. vLLM uses FlashAttention-style tiled attention adapted to accept block-table indirection. The query tile is a $C \times d_h$ matrix (where $C$ is the chunk size), and the KV tiles are loaded block-by-block through the block table.

### 8.4 Kernel variants shipped

| Variant | Use case |
|---------|----------|
| `paged_attention_v1` | Short-to-medium decode ($S \le 4096$) |
| `paged_attention_v2` | Long-sequence decode ($S > 4096$), split-K |
| FlashAttention v3 (Hopper) | FA-native paged attention on H100+ |
| Quantized variants | FP8 KV cache, INT8 KV cache |

---

## 9. Cause and Effect: Request Flow

```mermaid
flowchart TD
    ARR["Request arrives<br/>prompt = S tokens"] --> TOK["Tokenize<br/>→ token IDs"]
    TOK --> WAIT["Append to Waiting Queue"]
    WAIT --> ADMIT{"Scheduler:<br/>budget available?"}
    ADMIT -->|No| REJECT["HTTP 429 / queue"]
    ADMIT -->|Yes| PREFIX["Prefix cache lookup<br/>match M blocks"]
    PREFIX --> ALLOC["Allocate blocks<br/>for unmatched suffix"]
    ALLOC --> PREFILL["Prefill chunk<br/>up to B_tok tokens"]
    PREFILL --> KV["Write K,V to cache<br/>via slot_mapping"]
    KV --> LOGITS["Compute logits<br/>lm_head projection"]
    LOGITS --> SAMPLE["Sample token<br/>(temperature, top-p, grammar)"]
    SAMPLE --> FIRST["Stream first token<br/>TTFT measured"]
    FIRST --> DECODE{"More tokens<br/>to generate?"}
    DECODE -->|Yes| STEP["Next step:<br/>1 token budget allocated"]
    STEP --> FWD["Forward pass<br/>(decode path)"]
    FWD --> SAMPLE2["Sample next token"]
    SAMPLE2 --> STREAM["Stream token to client"]
    STREAM --> DECODE
    DECODE -->|No: EOS / max_tokens| FIN["Mark finished"]
    FIN --> FREE["Release blocks<br/>refcount--<br/>free if refcount = 0"]
    FREE --> DONE["Output complete"]

    style REJECT fill:#fecaca,stroke:#991b1b,color:#000
    style FIRST fill:#bbf7d0,stroke:#15803d,color:#000
    style DONE fill:#dbeafe,stroke:#1e40af,color:#000
```

---

## 10. Step-by-Step Scheduling Trace

Consider Llama-3-70B on 2$\times$ H100 with TP=2:

- $B_{\text{tok}} = 4096$, $B_{\text{seq}} = 256$
- $B_s = 16$, $N_{\text{blk}} = 12{,}000$ blocks (~60 GB KV pool)
- Per-block cost: 5.24 MB (from [KV_Cache](KV_Cache.md) Section 4.2)

**Trace:**

| Time | Event | Running | Waiting | Blocks used | Step tokens |
|------|-------|---------|---------|-------------|-------------|
| $t_0$ | R1 arrives (prompt=2048) | 0 | 1 | 0 | -- |
| $t_1$ | Admit R1. Prefix cache miss. Allocate 128 blocks. Prefill 2048 tokens. | 1 | 0 | 128 | 2048 |
| $t_2$ | R1 produces first token. Enters decode. R2 arrives (prompt=512). | 1 | 1 | 129 | -- |
| $t_3$ | Admit R2. Prefill 512 + R1 decode(1) = 513 tokens. | 2 | 0 | 161 | 513 |
| $t_4$ | R1 decode + R2 decode. | 2 | 0 | 162 | 2 |
| $t_5$ | R3 arrives (prompt=8192). Admit. Allocate 512 blocks needed. Prefill chunk 1: 4094 tokens (4096 - 2 decodes). | 3 | 0 | 674 | 4096 |
| $t_6$ | R3 chunk 2: 4094 tokens + 2 decodes. | 3 | 0 | 674+ | 4096 |
| $t_7$ | R3 produces first token. 3 decodes. R1 finishes (generated 50 tokens). Release 131 blocks. | 2 | 0 | ~548 | 3 |

**Key observation**: chunked prefill spreads R3's 8192-token prompt across two steps, keeping each step's total tokens at $B_{\text{tok}} = 4096$. Without chunking, R3 would consume 8192 tokens in one step, pushing step time to ~240 ms and causing a TPOT spike for R1 and R2.

---

## 11. Numbers to Know

| # | Quantity | Value | Derivation |
|---|----------|-------|------------|
| 1 | Default block size ($B_s$) | 16 tokens | Balance of waste vs. overhead |
| 2 | Internal fragmentation | $\le 6.25\%$ | $(B_s - 1) / B_s$ worst case |
| 3 | Block size (Llama-3-70B, GQA, FP16) | 5.24 MB | $16 \times 2 \times 80 \times 8 \times 128 \times 2$ |
| 4 | Blocks per 4K-token sequence | 256 | $\lceil 4096 / 16 \rceil$ |
| 5 | KV pool blocks (2x H100, FP8 weights) | ~32K | $(160 - 70 - 5) \text{ GB} / 2.62 \text{ MB}$ |
| 6 | Default `gpu_memory_utilization` | 0.90 | 90% of HBM for KV after weights |
| 7 | Typical `max_num_batched_tokens` ($B_{\text{tok}}$) | 2048--8192 | Controls step time |
| 8 | Typical `max_num_seqs` ($B_{\text{seq}}$) | 64--256 | Caps batch dimension |
| 9 | V1 scheduling overhead per step | < 0.5 ms | Shared-memory IPC, no Python GIL |
| 10 | Prefix cache TTFT savings (shared system prompt) | 80--95% | Long prefix fully cached |
| 11 | APC hash computation cost | ~0.01 ms/block | SHA-256 on 16 token IDs |
| 12 | PagedAttention indirection overhead | < 0.1% | 1 int read per 4096 B KV |
| 13 | Prefill chunk compute time (512 tokens, 70B, H100) | ~15--25 ms | Compute-bound on tensor cores |
| 14 | Decode step time (B=64, S=2K, 70B, 2xH100) | ~45 ms | Bandwidth-bound |
| 15 | PCIe swap-in time (2.5 GB) | ~39 ms | $2.5 / 64$ GB/s |
| 16 | V1 vs V0 throughput improvement | 1.5--2x | Eliminates GIL scheduling overhead |
| 17 | NCCL all-reduce per layer (TP=2, NVLink) | ~0.05 ms | 900 GB/s interconnect |
| 18 | Sampler latency (B=128, top-p) | ~0.2 ms | All-GPU, no host trip |
| 19 | Block allocation latency | ~0.001 ms | Free-list pop |
| 20 | APC prefix match length (shared chat template) | 64--2048 tokens | Depends on template length |

---

## 12. Worked Problems

### Problem 1: Block Pool Capacity

**Q:** Llama-3-70B ($n_l=80$, $n_{\text{kv}}=8$, $d_h=128$) runs on a single H100 (80 GB) with FP16 weights (140 GB is too large; use FP8 at 70 GB) and 5 GB activation overhead. Block size $B_s = 16$, FP16 KV. How many KV blocks fit? How many concurrent 4K-token sequences?

**A:**

KV budget: $80 - 70 - 5 = 5$ GB.

Per-block bytes: $16 \times 2 \times 80 \times 8 \times 128 \times 2 = 5{,}242{,}880$ B $\approx 5.24$ MB.

Total blocks: $\lfloor 5 \times 10^9 / 5{,}242{,}880 \rfloor = 953$ blocks.

Blocks per 4K sequence: $\lceil 4096 / 16 \rceil = 256$ blocks.

Concurrent sequences: $\lfloor 953 / 256 \rfloor = 3$ sequences.

This is why 70B on a single H100 requires FP8 KV cache or multi-GPU TP. With FP8 KV ($b = 1$), per-block cost halves to 2.62 MB, yielding 1906 blocks and 7 concurrent 4K sequences.

### Problem 2: Prefill Budget Allocation

**Q:** The server runs with $B_{\text{tok}} = 4096$ and currently has 48 decode sequences. A new request arrives with prompt length 6000 tokens. No prefix cache hit. How many chunks is the prefill split into? How many steps before the first token?

**A:**

Step 1: decode budget = 48 tokens. Prefill budget = $4096 - 48 = 4048$. Chunk 1 = 4048 tokens. Remaining: $6000 - 4048 = 1952$ tokens.

Step 2: decode budget = 48 (same sequences, plus the new one now in running state adds 1 decode token = 49). Prefill budget = $4096 - 49 = 4047$. Chunk 2 = 1952 tokens (fits in budget).

Total steps: 2. TTFT = $2 \times T_{\text{step}}$. At $T_{\text{step}} \approx 45$ ms: TTFT $\approx 90$ ms.

Without chunked prefill: TTFT would be dominated by a single step processing all 6000 + 48 = 6048 tokens, taking $\sim 180$ ms and causing a TPOT spike for all 48 decoders.

### Problem 3: Prefix Caching Hit Rate

**Q:** A chat service has a 1024-token system prompt shared by all users. Average user message is 256 tokens. Block size $B_s = 16$. What fraction of prompt blocks are served from cache for the second and subsequent requests?

**A:**

System prompt: 1024 tokens = 64 blocks. User message: 256 tokens = 16 blocks. Total: 80 blocks.

Prefix cache hit: the first 64 blocks (1024 tokens) are an exact match of the shared system prompt. The 65th block straddles the boundary: tokens 1024--1039 contain the tail of the system prompt (0 tokens) and the start of the user message (16 tokens). The hash for block 64 depends on the user message content -- different for each request. So only blocks 0--63 (the full 1024-token prefix) are shared.

Hit rate: $64 / 80 = 80\%$ of blocks served from cache.

Prefill work reduction: only 256 tokens (the user message) need prefilling, vs 1280 without caching. Savings: $(1280 - 256) / 1280 = 80\%$.

### Problem 4: Preemption Cost Analysis

**Q:** The block pool is full. A high-priority request (prompt = 4096 tokens) arrives. Two candidates for preemption: Sequence A (generated 200 tokens, 0 prefix cache hits, total context = 8192), Sequence B (generated 10 tokens, 75% prefix cache hit, total context = 1024). Block size = 16. Which should be preempted and why?

**A:**

**Sequence A:** 8192 tokens = 512 blocks. Recompute cost = re-prefill all 8192 tokens ($2 N \times 8192$ FLOPs) because prefix cache hit rate is 0. Swap cost = $8192 \times 320 \text{ KB} / 64 \text{ GB/s} \approx 41$ ms.

**Sequence B:** 1024 tokens = 64 blocks. Recompute cost = re-prefill only $0.25 \times 1024 = 256$ tokens (75% hits prefix cache). Swap cost = $1024 \times 320 \text{ KB} / 64 \text{ GB/s} \approx 5$ ms.

**Decision:** Preempt Sequence A. Although it frees more blocks (512 vs 64), the high-priority request needs $4096/16 = 256$ blocks -- Sequence A frees enough in one eviction; Sequence B would require evicting multiple victims. The recompute cost is irrelevant to the decision because it is borne by Sequence A when it is re-admitted, not by the high-priority request. The priority system explicitly accepts this cost to serve the urgent request.

If Sequence B were sufficient (only 256 blocks needed): preempt Sequence B. Its recompute cost is minimal (256 tokens with 75% cache hit = 64 tokens to re-prefill), and it frees enough blocks.

### Problem 5: TP Memory Math

**Q:** Llama-3-70B ($n_l=80$, $n_{\text{kv}}=8$, $d_h=128$, FP16 weights = 140 GB) is served on 4x H100 with TP=4. How much memory per GPU is available for KV cache with `gpu_memory_utilization=0.9`?

**A:**

Per GPU: 80 GB HBM. Weight shard: $140 / 4 = 35$ GB. Activation overhead: ~3 GB per GPU.

Usable per GPU: $80 \times 0.9 = 72$ GB. KV budget per GPU: $72 - 35 - 3 = 34$ GB.

Per-block cost per GPU (TP=4, so $n_{\text{kv}} / 4 = 2$ heads per rank):
$16 \times 2 \times 80 \times 2 \times 128 \times 2 = 1{,}310{,}720$ B $\approx 1.31$ MB per block per rank.

Blocks per GPU: $\lfloor 34 \times 10^9 / 1{,}310{,}720 \rfloor = 25{,}936$ blocks.

Total logical blocks across all 4 GPUs: 25{,}936 (each block is sharded across ranks; the block table is identical on all ranks). This serves $\lfloor 25{,}936 / 256 \rfloor = 101$ concurrent 4K-token sequences -- a significant improvement over the single-GPU case (3 sequences).

---

## 13. Source Map

The V1 codebase (approximate; rapidly evolving):

```mermaid
flowchart TB
    subgraph vllm["vllm/"]
        subgraph eng["engine/"]
            A["async_llm_engine.py — AsyncLLMEngine (frontend entry point)"]:::py
            B["engine_core.py — EngineCore (scheduler + coordinator)"]:::py
        end
        subgraph core["core/"]
            C["scheduler.py — Per-step scheduling logic"]:::py
            D["block_manager.py — KV pool, block tables, free list"]:::py
            E["prefix_caching.py — Hash-chain prefix cache"]:::py
        end
        subgraph wrk["worker/"]
            F["worker.py — Per-GPU process, weight loading"]:::py
            G["model_runner.py — Forward pass, input packing"]:::py
        end
        subgraph mx["model_executor/"]
            H["models/ — Per-architecture model definitions"]:::py
            I["layers/ — TP-aware building blocks"]:::py
        end
    end
    classDef py fill:#fde68a,stroke:#b45309,color:#000
```

Additional key directories: `attention/` (PagedAttention dispatch, ops wrappers for paged and FlashAttention kernels, backend selection), `csrc/` (custom CUDA/Triton kernels for paged attention, quantization, and activation), `entrypoints/openai/` (OpenAI-compatible HTTP frontend), and `sampling/sampler.py` (top-k, top-p, grammar, penalties).

Major debugging concentrates in `engine_core.py` (scheduling decisions), `block_manager.py` (KV allocation and prefix cache), and `model_runner.py` (forward pass correctness).

---

## 14. Performance Tuning Knobs

| Flag | Default | Effect |
|------|---------|--------|
| `--max-num-batched-tokens` | 2048 (varies) | Per-step token budget. Higher = bigger prefill chunks, lower = smoother decode TPOT |
| `--max-num-seqs` | 256 | Maximum concurrent sequences. Increase until TPOT SLO breaks |
| `--gpu-memory-utilization` | 0.90 | Fraction of HBM for KV pool after weights. Too high risks OOM during activation spikes |
| `--block-size` | 16 | KV block size in tokens. Rarely needs changing |
| `--swap-space` | 4 GB | CPU swap buffer for preempted sequences |
| `--enable-prefix-caching` | On (V1) | Toggle APC. Off for workloads with no prefix sharing |
| `--enable-chunked-prefill` | On (V1) | Toggle chunked prefill. Off only for latency-insensitive batch processing |
| `--tensor-parallel-size` | 1 | TP degree. Must match GPU count |
| `--pipeline-parallel-size` | 1 | PP degree. Rarely needed for inference |
| `--kv-cache-dtype` | auto | FP8 KV halves cache memory at minor quality cost |
| `--quantization` | None | Weight quantization (awq, gptq, fp8). Frees HBM for KV |

**Tuning heuristic:** increase `max_num_seqs` until p99 TPOT reaches the SLO limit. Then increase `max_num_batched_tokens` until TTFT meets its target. If KV pool is the bottleneck (frequent preemption), either quantize weights or enable FP8 KV.

---

## 15. Speculative Decoding Integration

Speculative decoding (draft-then-verify) accelerates decode throughput by having a small draft model propose $k$ candidate tokens that the target model verifies in a single batched forward pass. In vLLM, this flows through the scheduler and block manager as follows.

### 15.1 Draft token generation and speculative slots

When speculative decoding is enabled, the scheduler allocates **speculative slots** in the block table for $k$ draft tokens (typically $k = 5$) *before* verification. The draft model (a smaller model sharing the same tokenizer) runs a lightweight forward pass to produce the $k$ candidates. These tokens are appended to the sequence's KV cache speculatively -- the block manager allocates blocks for them as if they were real decode output, but marks them as unverified.

### 15.2 Block manager handling of draft tokens

Draft tokens are written into the KV cache immediately. The block manager appends them to the sequence's block table using the normal allocation path (pop from free list). If the verification pass rejects some or all draft tokens, the block manager **truncates back** to the last verified position: the blocks beyond the accepted length have their reference counts decremented and return to the free list. If all $k$ tokens are accepted, the blocks are committed and the sequence advances by $k+1$ positions (the $k$ draft tokens plus the one token the target model itself produces at the acceptance boundary).

### 15.3 Verification pass

The target model runs a single batched verification forward pass over all $k$ draft tokens simultaneously -- not sequentially. The acceptance check produces a mask that determines how many tokens to keep. The acceptance criterion is the standard rejection-sampling check: draft token $t_i$ is accepted iff the target model's probability $p_{\text{target}}(t_i) \ge p_{\text{draft}}(t_i)$, with a resampled token drawn from $p_{\text{target}} - p_{\text{draft}}$ (clamped to non-negative) at the first rejection point.

### 15.4 Scheduler changes

The token budget must account for both draft and verified tokens. When a speculative sequence is scheduled for verification, the scheduler reserves $k$ tokens in the budget (one per draft position to verify), not just the single decode token. The scheduler must know the draft model's speculative length $k$ at configuration time (`--speculative-max-model-len` or `num_speculative_tokens`). Prefill sequences are never speculatively decoded -- only decode-phase sequences participate.

### 15.5 Integration with PagedAttention

Speculative blocks use the same block pool as normal decode blocks. Rejected draft tokens simply free their blocks via the standard free-list return path. No copy-on-write is needed because speculative blocks are only referenced by one sequence -- there is no sharing scenario. The slot mapping for draft tokens uses the same `physical_block * B_s + offset` computation as regular decode tokens.

---

## 16. V1 vs V0 Benchmark Comparison

The V1 architectural changes (Section 1.3) yield measurable throughput and latency improvements. The table below summarizes approximate figures from public benchmarks (vLLM blog posts, SGLang comparison papers, and community benchmarks). Exact numbers vary by hardware, model, and workload.

| Metric | V0 (async engine) | V1 (single-process) | Improvement |
|--------|-------------------|---------------------|-------------|
| TTFT (Llama-3-8B, $S=1024$) | ~45 ms | ~32 ms | 1.4x |
| Decode throughput (8B, $B=256$) | ~12,000 tok/s | ~18,500 tok/s | 1.54x |
| Decode throughput (70B, $B=64$) | ~1,100 tok/s | ~1,400 tok/s | 1.27x |
| End-to-end latency (streaming) | Higher (IPC overhead) | Lower (no IPC) | ~30% |
| GIL contention (Python-heavy) | Present | Eliminated | Major |
| Scheduling overhead per step | ~500 $\mu$s | ~50 $\mu$s | 10x |

**Sources of V1 gains:**

1. **Eliminating inter-process IPC.** The V0 engine used Python `multiprocessing.Queue` between the API server, scheduler, and workers. Each queue hop added serialization and deserialization overhead. V1 uses zero-copy shared-memory queues.
2. **Removing the Python GIL from the scheduling hot path.** V0's scheduler ran in the same Python interpreter as post-processing code. V1 isolates scheduling into a dedicated path with minimal Python.
3. **Fused scheduling + sampling kernel.** V1 co-designs the scheduler output format with the sampler input, avoiding redundant tensor construction.
4. **Zero-copy tensor sharing.** The scheduler and model runner share GPU tensor memory directly, avoiding host-device copies for metadata.

---

## 17. KV Transfer for Disaggregated Prefill

vLLM supports a KV transfer API (`--kv-transfer-config` or the `kv_transfer` flag) that serializes KV blocks from dedicated prefill workers to dedicated decode workers, enabling **disaggregated prefill** (see [Prefill_Decode_Disaggregation](Prefill_Decode_Disaggregation.md)).

**Block-level granularity.** Entire PagedAttention blocks are transferred, not individual token KV vectors. When a prefill worker finishes computing the prompt's KV cache, it serializes the filled blocks (physical block IDs + their KV tensor data) and sends them to the decode worker pool.

**Transport.** Intra-node transfer uses NCCL (GPU-to-GPU over NVLink or PCIe). Inter-node transfer uses PyNIXL or custom RDMA transports. The transfer is pipelined: the prefill worker begins sending completed blocks before the full prefill finishes.

**Decode-side allocation.** The decode worker allocates matching blocks in its local pool and populates them from the received data. The block table on the decode side mirrors the prefill worker's logical-to-physical mapping, but with the decode worker's own physical block IDs.

**Mooncake-style disaggregation.** This enables architectures where prefill runs on a pool of high-compute GPUs (e.g., H100 with high tensor-core FLOPS) and decode runs on a pool of high-bandwidth GPUs (e.g., H100 with high HBM bandwidth), connected via RDMA. The prefill workers never run decode steps; the decode workers never run prefill. KV transfer is the bridge.

---

## 18. Common Failure Modes

| Symptom | Root cause | Fix |
|---------|-----------|-----|
| `CUDA OOM during init` | `gpu_memory_utilization` too high for model + activations | Reduce to 0.85 or quantize weights |
| Requests stuck (never stream) | Stop tokens or grammar deadlock in sampler | Check structured-output grammar; test with `temperature > 0` |
| TPOT spikes correlate with long-prompt arrivals | Chunked prefill not aggressive enough | Lower `max_num_batched_tokens` |
| Prefix cache hit rate near zero | Prompt prefixes vary (e.g., dynamic system prompts, different tokenizers) | Fix shared prefix length; verify tokenizer consistency |
| Workers hung on NCCL | NCCL timeout or GPU link failure | Set `NCCL_DEBUG=INFO`; increase `NCCL_TIMEOUT_MS` |
| High scheduling overhead (>5 ms/step) | V0 engine (Python GIL) | Upgrade to V1; check that EngineCore runs in separate process |
| Frequent preemption under load | KV pool too small for offered concurrency | Quantize weights to free HBM; reduce `max_num_seqs`; add GPUs |

---

## 19. References

1. Kwon, W. et al. (2023). "Efficient Memory Management for Large Language Model Serving with PagedAttention." *SOSP 2023*.
2. vLLM Project. "vLLM: Easy, Fast, and Cheap LLM Serving." [GitHub](https://github.com/vllm-project/vllm).
3. vLLM Blog (2024--2025). "V1 Engine Architecture," "Automatic Prefix Caching," "Structured Output."
4. Yu, G.I. et al. (2022). "Orca: A Distributed Serving System for Transformer-Based Generative Models." *OSDI 2022*.
5. Zheng, L. et al. (2024). "SGLang: Efficient Execution of Structured Language Model Programs." *NeurIPS 2024*.
6. Holmes, C. et al. (2024). "Sarathi-Serve: Taming Compute-Utilization and Memory-Bottlenecks in LLM Serving." *OSDI 2024*.
7. Dao, T. (2023). "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." *ICLR 2024*.

---

## 20. Stack Links

**Up (deeper):**
- [KV_Cache](KV_Cache.md) -- PagedAttention memory layout, block math, prefix caching derivations
- [Batching_and_Scheduling](Batching_and_Scheduling.md) -- continuous batching theory, chunked prefill, preemption policies
- [FlashAttention_Deep_Dive](../L5_Kernels_and_Programming/FlashAttention_Deep_Dive.md) -- tiled attention kernel consumed by the PagedAttention dispatch
- [Parallelism_Strategies](../L7_Training_Stack/Parallelism_Strategies.md) -- Megatron-style TP pattern used by the executor

**Down (higher level):**
- [Production_Architecture](Production_Architecture.md) -- capacity planning, SLO targets, deployment topology
- [Kubernetes_and_Orchestration](Kubernetes_and_Orchestration.md) -- GPU operator, pod placement, autoscaling

**Lateral:**
- [Inference_Frameworks](Inference_Frameworks.md) -- vLLM vs SGLang vs TRT-LLM feature comparison
- [Speculative_Decoding](Speculative_Decoding.md) -- draft-then-verify integrated into the vLLM step loop
- [Prefill_Decode_Disaggregation](Prefill_Decode_Disaggregation.md) -- splitting the engine across prefill and decode GPU pools

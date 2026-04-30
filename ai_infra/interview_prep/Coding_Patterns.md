# Coding Patterns

The coding rounds for AI-infra roles are unusual: kernel writing, parallel reduction patterns, scheduler algorithms, and only sometimes leetcode-style. This page collects the patterns that come up most often, with reference implementations.

**Prerequisites**: [CUDA_Programming](../Part_4_GPU_Kernel_Engineering/CUDA_Programming.md), [CUDA_Optimization](../Part_4_GPU_Kernel_Engineering/CUDA_Optimization.md), [Triton_and_Kernels](../Part_4_GPU_Kernel_Engineering/Triton_and_Kernels.md), [Batching_and_Scheduling](../Part_6_Inference_Engines_and_Serving/Batching_and_Scheduling.md).

---

## 1. The Coding Round Categories

Roles vary, but the patterns cluster:

| Round type | What you'll be asked | Where to drill |
|------------|---------------------|----------------|
| Kernel writing | Write a CUDA / Triton kernel for X (matmul, softmax, layernorm, attention) | this page §3-5 |
| Parallel reduction | Sum, max, prefix sum, top-k on GPU | §6 |
| Scheduler / serving | Implement continuous batching scheduler, paged KV manager, rate limiter | §7 |
| Numerical algorithms | Online softmax, beam search, sampling, KV update | §8 |
| Systems C++ | Memory allocator, lock-free queue, threadpool | §9 |
| Leetcode-style | Trees, graphs, DP — less common but happens | §10 |

Most companies emphasize 1, 2, 4. Senior roles add 7. Some still ask 10 to filter.

---

## 2. Setup You Should Have

- CUDA toolchain installed locally OR a remote H100/A100.
- PyTorch with CUDA, with at least one custom op you've built end-to-end.
- Triton installed; have written ≥3 nontrivial kernels.
- Familiar with `nvcc`, `nsys`, `ncu`.
- A few benchmarks ready: matmul, softmax, attention against `torch.bmm` / `F.scaled_dot_product_attention`.

For interviews, you should be able to write CUDA on a whiteboard or shared editor without compiler help.

---

## 3. Reference Kernel: Matmul (CUDA)

Tiled SMEM matmul. Memorize the structure; the variations come from tile sizes and tensor-core variants.

```cuda
// C = A @ B, A: (M, K) row-major, B: (K, N) row-major, C: (M, N) row-major
template <int BM, int BN, int BK>
__global__ void matmul_tiled(
    const float* A, const float* B, float* C, int M, int N, int K) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    float c_reg = 0.0f;

    for (int k_tile = 0; k_tile < K; k_tile += BK) {
        // Cooperatively load tiles into SMEM
        As[ty][tx] = A[(by*BM + ty) * K + (k_tile + tx)];
        Bs[ty][tx] = B[(k_tile + ty) * N + (bx*BN + tx)];
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            c_reg += As[ty][kk] * Bs[kk][tx];
        }
        __syncthreads();
    }

    C[(by*BM + ty) * N + (bx*BN + tx)] = c_reg;
}
```

Improvements you should know how to bolt on:
- Each thread computes a `TM × TN` micro-tile (not just one element) → register tiling, reduces SMEM traffic.
- Async copy via `cp.async` (Ampere) or TMA (Hopper) overlapping load with compute.
- Tensor core `wmma` / `wgmma` for FP16/BF16/FP8 inputs.
- Bank-conflict-free SMEM layout (pad columns or swizzle).

For interviews focused on "write the simplest correct version, then optimize": start as above and iterate.

---

## 4. Reference Kernel: Softmax (Triton)

```python
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(X_ptr, Y_ptr, M, N, stride_xm, stride_ym, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=-float("inf"))
    x_max = tl.max(x, axis=0)
    x_shift = x - x_max
    num = tl.exp(x_shift)
    den = tl.sum(num, axis=0)
    y = num / den

    tl.store(Y_ptr + row * stride_ym + cols, y, mask=mask)
```

Key: subtract max **before** exp for numerical stability. Online (block-wise) softmax needed when N exceeds BLOCK_N — see [Attention_Mechanisms §3](../Part_5_Algorithms_and_Quantization/Attention_Mechanisms.md).

---

## 5. Reference Kernel: Attention (Triton, Causal Forward)

A condensed FA-v2-style forward. Real production kernels have many more knobs, but this is the structure to reproduce on whiteboard.

```python
@triton.jit
def attn_fwd(Q, K, V, O, sm_scale,
             stride_qb, stride_qh, stride_qs, stride_qd,
             stride_kb, stride_kh, stride_ks, stride_kd,
             stride_vb, stride_vh, stride_vs, stride_vd,
             stride_ob, stride_oh, stride_os, stride_od,
             N_CTX, HEAD_DIM: tl.constexpr,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    start_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // n_heads
    h = bh % n_heads

    # Pointers into (B, H, S, D)
    Q_blk = tl.make_block_ptr(
        base=Q + b*stride_qb + h*stride_qh,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qs, stride_qd),
        offsets=(start_m*BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0))
    q = tl.load(Q_blk)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Causal: only iterate up to the diagonal
    hi = (start_m + 1) * BLOCK_M

    for start_n in range(0, hi, BLOCK_N):
        K_blk = tl.make_block_ptr(...)  # K offsets
        V_blk = tl.make_block_ptr(...)
        k = tl.load(K_blk)
        v = tl.load(V_blk)

        s = tl.dot(q, tl.trans(k)) * sm_scale  # (BLOCK_M, BLOCK_N)

        # Causal mask within the diagonal block
        m_offsets = start_m*BLOCK_M + tl.arange(0, BLOCK_M)
        n_offsets = start_n + tl.arange(0, BLOCK_N)
        mask = m_offsets[:, None] >= n_offsets[None, :]
        s = tl.where(mask, s, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = alpha * l_i + tl.sum(p, 1)
        acc = alpha[:, None] * acc + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    acc = acc / l_i[:, None]
    O_blk = tl.make_block_ptr(...)
    tl.store(O_blk, acc.to(O.dtype.element_ty))
```

Whiteboard expectation: you should be able to write this skeleton — outer loop over Q blocks, inner over K/V, online softmax accumulator — in 20-30 minutes. See [FlashAttention_Deep_Dive §12](../Part_4_GPU_Kernel_Engineering/FlashAttention_Deep_Dive.md) for the full version.

---

## 6. Parallel Reduction Patterns

### 6.1 Block-level Sum

```cuda
__global__ void block_sum(const float* x, float* out, int N) {
    __shared__ float smem[BLOCK];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    smem[tid] = (gid < N) ? x[gid] : 0.0f;
    __syncthreads();

    // Tree reduction inside block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = smem[0];
}
```

### 6.2 Warp Shuffle (faster)

```cuda
__inline__ __device__ float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, offset);
    return v;  // first lane has full sum
}
```

Use within block: do warp reduces then a tiny SMEM-staged final.

### 6.3 Top-K (Heap or Bitonic)

For top-k of vocab logits in sampling: maintain a per-thread heap or use a bitonic sort. CUB's `cub::DeviceRadixSort::SortPairsDescending` followed by gather works in production.

### 6.4 Prefix Sum (Scan)

Hillis-Steele or Kogge-Stone scans. Use CUB's `cub::DeviceScan` in real code.

---

## 7. Scheduler / Serving Patterns

### 7.1 Continuous Batching Skeleton

```python
class Scheduler:
    def __init__(self, max_batched_tokens, max_seqs):
        self.waiting = deque()
        self.running = []
        self.max_batched_tokens = max_batched_tokens
        self.max_seqs = max_seqs
        self.kv = KVPool()

    def admit(self, req):
        self.waiting.append(req)

    def step(self):
        # Try to admit waiting
        while self.waiting and len(self.running) < self.max_seqs:
            r = self.waiting[0]
            if self.kv.can_alloc_for(r):
                self.waiting.popleft()
                self.kv.reserve(r)
                self.running.append(r)
            else:
                break

        # Build the step batch
        batch = []
        token_budget = self.max_batched_tokens
        for r in self.running:
            if r.in_prefill:
                chunk = min(r.tokens_remaining_to_prefill, token_budget)
                if chunk == 0: break
                batch.append((r, chunk))
                token_budget -= chunk
            else:
                if token_budget <= 0: break
                batch.append((r, 1))  # 1 decode token
                token_budget -= 1

        # Preempt if KV insufficient (rare with reservation)
        return batch
```

### 7.2 Paged KV Allocator

```python
class KVPool:
    def __init__(self, num_blocks, block_size):
        self.free = list(range(num_blocks))
        self.block_size = block_size
        self.refcount = [0] * num_blocks

    def alloc(self):
        if not self.free: return None
        b = self.free.pop()
        self.refcount[b] = 1
        return b

    def share(self, b):  # prefix cache hit
        self.refcount[b] += 1

    def free_block(self, b):
        self.refcount[b] -= 1
        if self.refcount[b] == 0:
            self.free.append(b)
```

### 7.3 Token Bucket Rate Limiter

```python
import time
class TokenBucket:
    def __init__(self, rate, capacity):
        self.rate = rate          # tokens / sec
        self.capacity = capacity
        self.tokens = capacity
        self.last = time.monotonic()

    def consume(self, n):
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
        self.last = now
        if self.tokens >= n:
            self.tokens -= n
            return True
        return False
```

---

## 8. Numerical Algorithm Patterns

### 8.1 Online Softmax (numerically stable)

```python
def online_softmax_update(m_old, l_old, x_new):
    m_new = max(m_old, x_new.max())
    alpha = math.exp(m_old - m_new)
    beta = 1.0  # if x_new is a single block, scale is 1
    l_new = alpha * l_old + (np.exp(x_new - m_new)).sum()
    return m_new, l_new
```

Used in FA, ring attention, paged decode.

### 8.2 Sampling (top-k + top-p)

```python
def sample(logits, temperature, top_k, top_p):
    logits = logits / temperature
    # Top-k
    if top_k is not None:
        kth = np.partition(logits, -top_k)[-top_k]
        logits = np.where(logits >= kth, logits, -np.inf)
    # Softmax
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    # Top-p
    if top_p is not None and top_p < 1:
        order = np.argsort(probs)[::-1]
        cumsum = np.cumsum(probs[order])
        cutoff = np.argmax(cumsum >= top_p) + 1
        keep = order[:cutoff]
        mask = np.zeros_like(probs)
        mask[keep] = 1
        probs *= mask
        probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)
```

In production this is a fused GPU kernel.

### 8.3 Beam Search

```python
def beam_search(model, prompt, beam_width, max_len):
    beams = [(0.0, prompt)]  # (logprob, tokens)
    for _ in range(max_len):
        candidates = []
        for score, seq in beams:
            logits = model(seq)[-1]
            log_probs = log_softmax(logits)
            top = np.argpartition(log_probs, -beam_width)[-beam_width:]
            for tok in top:
                candidates.append((score + log_probs[tok], seq + [tok]))
        beams = sorted(candidates, key=lambda x: -x[0])[:beam_width]
    return beams
```

### 8.4 KV Cache Update

```python
def append_kv(kv_cache, layer, k_new, v_new, slot):
    # kv_cache shape: (L, 2, B, S_max, H, D)
    kv_cache[layer, 0, :, slot] = k_new
    kv_cache[layer, 1, :, slot] = v_new
```

In paged form:
```python
def append_kv_paged(kv_blocks, block_table, layer, k_new, v_new, seq_id, pos):
    block_size = kv_blocks.block_size
    blk_idx = pos // block_size
    slot_in_blk = pos % block_size
    if blk_idx >= len(block_table[seq_id]):
        block_table[seq_id].append(kv_blocks.alloc())
    phys = block_table[seq_id][blk_idx]
    kv_blocks.K[layer, phys, slot_in_blk] = k_new
    kv_blocks.V[layer, phys, slot_in_blk] = v_new
```

---

## 9. Systems C++ Patterns

### 9.1 Lock-Free SPSC Queue

For request handoff between scheduler thread and dispatcher thread.

```cpp
template <typename T, size_t N>
class SPSCQueue {
    std::array<T, N> buf;
    std::atomic<size_t> head{0}, tail{0};
public:
    bool push(T v) {
        size_t t = tail.load(std::memory_order_relaxed);
        size_t n = (t + 1) % N;
        if (n == head.load(std::memory_order_acquire)) return false;
        buf[t] = std::move(v);
        tail.store(n, std::memory_order_release);
        return true;
    }
    bool pop(T& v) {
        size_t h = head.load(std::memory_order_relaxed);
        if (h == tail.load(std::memory_order_acquire)) return false;
        v = std::move(buf[h]);
        head.store((h + 1) % N, std::memory_order_release);
        return true;
    }
};
```

### 9.2 Slab Allocator

For variable-size KV blocks or activation buffers:

```cpp
class SlabAllocator {
    struct Slab { void* ptr; size_t size; bool used; };
    std::vector<Slab> slabs;
public:
    void* alloc(size_t sz) {
        for (auto& s : slabs)
            if (!s.used && s.size >= sz) { s.used = true; return s.ptr; }
        void* p = ::operator new(sz);
        slabs.push_back({p, sz, true});
        return p;
    }
    void free_(void* p) {
        for (auto& s : slabs) if (s.ptr == p) { s.used = false; return; }
    }
};
```

Real allocators (PyTorch CUDA caching allocator) are much more sophisticated.

---

## 10. Leetcode Patterns That Show Up

When companies still ask leetcode, AI-infra teams favor:

- **Sliding window** — implementing token caches, recent-N statistics.
- **Heap / priority queue** — top-k logits, scheduler priorities.
- **Prefix sum / range queries** — running stats on streams.
- **Topological sort** — DAG scheduling for kernel pipelines.
- **Trie / suffix automaton** — prefix caching (RadixAttention is a trie).
- **LRU cache** — model/LoRA caching.
- **Producer-consumer queues** — pipeline parallelism schedulers.

Have crisp implementations of each ready.

---

## 11. Practice Problem Bank

Try each in 30–45 minutes:

1. Implement a CUDA kernel for `y = x * x` with proper coalescing, optimal block size, and a benchmark vs `torch.square`.
2. Write a Triton kernel for layer norm, including affine. Match `torch.nn.LayerNorm` output to 1e-5.
3. Implement a paged-attention decode kernel for one head, given a block table.
4. Write a continuous-batching scheduler in Python that admits requests, builds a step batch with chunked prefill, and emits done flags.
5. Implement online softmax over a stream of arrays, then verify it matches `np.softmax` on the concatenation.
6. Write a top-p sampler in Python (CPU is fine) that handles temperature and top-k.
7. Implement a hash-based prefix cache: blocks of size 16, hash by (parent_hash, block_tokens), refcount-aware.
8. Implement a trie-based RadixAttention-style prefix cache that supports partial-block hits.
9. Speculative decoding loop: vanilla with a draft and target, including the rejection-sampling step.
10. Implement a token-bucket rate limiter that fairly serves multiple tenants.

---

## 12. Coding Round Tips

- **Talk through the problem first.** Restate constraints. State tradeoffs before code.
- **Start simple.** A correct slow version beats an incomplete fancy one. Optimize only if asked or if time permits.
- **Test as you go.** Write a tiny driver / unit test next to your kernel.
- **Mind units.** Bytes vs elements, FLOPs vs FLOP/s, ms vs s. Common interview slip.
- **Show numerical-stability awareness.** Subtract max before exp; use FP32 accumulators; pad SMEM for bank conflicts.
- **Sketch the optimization plan before writing it.** Tile sizes, async copies, tensor-core variants — say what you'd add and why.

---

## 13. Common Interview Questions

**Q: Write a CUDA kernel for vector addition.**
A: Trivial — main pitfalls are (1) bounds checking when `N` isn't a multiple of block size, (2) coalesced access, (3) using `__restrict__` and pointer aliasing hints, (4) considering vectorized loads (`float4`).

**Q: How would you optimize a softmax kernel for very long sequences?**
A: Online softmax: split the row into blocks, maintain running (max, sum), do block reductions with warp shuffles, then a final SMEM-based reduce. Use FP32 accumulators even with FP16 inputs.

**Q: Implement a tiled matmul. What's the optimal tile size on H100?**
A: Standard: BM=128, BN=128, BK=64 for FP16 with tensor cores. Each thread computes a TM×TN micro-tile, e.g., 8×8. Use cp.async or TMA for loads. Choose based on register budget and SMEM (228 KB available per block).

**Q: How do you ensure thread-safety in a kernel that writes to a shared output?**
A: For exact reductions, use `atomicAdd` (slow at high contention). For tensor reductions, use a tree reduction inside the block, then either a single thread per block writes, or a final atomicAdd to global. Splits-K with a separate reduction pass also works.

**Q: Write the scheduler step for continuous batching.**
A: See §7.1.

**Q: How do you implement paged KV cache?**
A: Pool of fixed-size blocks; per-sequence block table mapping logical → physical. Allocator with refcounts for prefix sharing. See §7.2.

**Q: Write online softmax and explain how it generalizes to attention.**
A: Maintain (m, ℓ). For each block, compute local max m', shift+exp, sum to ℓ', combine with `α=exp(m-m_new), β=exp(m'-m_new)`. In attention, also maintain a running output O, rescaled the same way; final divide by ℓ.

**Q: What's the difference between `__syncthreads` and `__threadfence`?**
A: `__syncthreads` is a barrier — all threads in the block wait. `__threadfence` only orders memory operations from the calling thread relative to others; doesn't block. Use the latter when you need ordering but not synchronization.

**Q: How would you implement a tokenizer-free request rate limiter?**
A: Token bucket per tenant; tokens = "credits"; consume credits per request based on input size. Refill at fixed rate. Reject if not enough credits. See §7.3.

**Q: Implement top-k for sampling.**
A: For small k (≤ 64), warp-level partial sort works. For larger or sparse, partition + selection. In production, CUB radix sort suffices. CPU reference: `np.argpartition(logits, -k)[-k:]`.

---

## 14. Further Reading

- "CUDA C++ Best Practices Guide" (NVIDIA).
- Triton tutorials.
- CUTLASS examples and walkthroughs.
- NVIDIA's `nccl-tests` source for collective patterns.
- vLLM's `csrc/` directory for production-quality CUDA.
- "Programming Massively Parallel Processors" (Hwu/Kirk).
- Recent FA-v3 and CUTLASS-3 source code for Hopper-era patterns.

---

**Next:** [Index](../Index.md) for the master TOC.
**See also:** [CUDA_Programming](../Part_4_GPU_Kernel_Engineering/CUDA_Programming.md), [CUDA_Optimization](../Part_4_GPU_Kernel_Engineering/CUDA_Optimization.md), [Triton_and_Kernels](../Part_4_GPU_Kernel_Engineering/Triton_and_Kernels.md), [System_Design_Interview](System_Design_Interview.md).

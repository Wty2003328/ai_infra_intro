# FlashAttention Deep Dive

FlashAttention is the reference implementation of IO-aware attention and is now standard in every production inference stack. Understanding it at the tile level is mandatory for a kernel-facing or inference-facing interview. This page walks through v1, v2, and v3 forward passes, the backward pass, tile-sizing math, and the common interview questions.

**Prerequisites**: [[Attention_Mechanisms]] (online softmax derivation), [[Memory_Hierarchy_and_Roofline]] (arithmetic intensity, tiling), [[CUDA_Optimization]] (shared memory, warp specialization, TMA).

---

## 1. Why Standard Attention Is IO-Bound

Recall the standard attention formulation for a single head:

$$S = Q K^\top \quad (N \times N)$$
$$P = \text{softmax}(S) \quad (N \times N)$$
$$O = P V \quad (N \times d)$$

Both $S$ and $P$ are materialized in HBM, which is the root cause of the IO bottleneck.

For sequence length $N$ and head dimension $d$, standard attention does:

- **FLOPs**: $2 N^2 d$ (for $QK^\top$) + $2 N^2 d$ (for $PV$) $\approx 4 N^2 d$ plus the softmax (tiny).
- **HBM bytes**: we must read $Q, K, V$ once ($3 N d$), but we write $S$ and $P$ to HBM ($2 N^2$) and read them back. So total HBM traffic $\approx 4 N^2$ (dominant at large $N$) plus $O(N d)$.

### 1.1 Arithmetic Intensity Derivation

Arithmetic intensity (AI) is the ratio of compute to memory traffic. For standard attention:

$$\text{AI} = \frac{\text{FLOPs}}{\text{bytes}} \approx \frac{4 N^2 d}{4 N^2 \cdot b}= \frac{d}{b}$$

where $b$ is the dtype size in bytes. For FP16 ($b = 2$) and $d = 128$:

$$\text{AI} \approx \frac{128}{2} = 64 \;\text{FLOP/B}$$

On H100 the ridge point is:

$$\text{ridge} = \frac{\text{peak FLOPS}}{\text{peak BW}} = \frac{989 \;\text{TFLOPS}}{3.35 \;\text{TB/s}} \approx 295 \;\text{FLOP/B}$$

Since $64 \ll 295$, standard attention is deeply **memory-bound** -- we are roughly $4\times$ below the ridge point, so we hit about 25% of peak FLOPs at best. The bottleneck is **$S$ and $P$ round-tripping through HBM**, not the matmul.

The idea of FlashAttention: **never materialize $S$ or $P$ in HBM**. Compute them in shared memory / registers, combine partial softmax statistics using the online softmax trick, and write only the final output $O$.

---

## 2. Online Softmax Recap

From [[Attention_Mechanisms]] section 3, the online softmax maintains three running statistics as we stream over blocks of keys:

- $m$ = running max logit
- $\ell$ = running denominator (sum of exponentials)
- $O$ = running weighted sum of values

When a new block arrives with local max $m'$ and local sum $\ell'$:

$$m_{\text{new}} = \max(m, \, m')$$

$$\alpha = e^{m - m_{\text{new}}}, \quad \beta = e^{m' - m_{\text{new}}}$$

$$\ell \leftarrow \alpha \cdot \ell + \beta \cdot \ell'$$

$$O \leftarrow \alpha \cdot O + \beta \cdot (P_{\text{new}} \, V_{\text{new}})$$

$$m \leftarrow m_{\text{new}}$$

The final $O$ is divided by $\ell$ once at the end. This is numerically identical to softmax (proof in the attention page).

### 2.1 Why This Works: Numerical Equivalence

To see why this is equivalent, consider the full softmax over all $N$ keys:

$$o_i = \frac{\sum_{j=1}^{N} e^{s_j} v_j}{\sum_{j=1}^{N} e^{s_j}}$$

Splitting into two blocks $A$ and $B$ with local maxima $m_A$ and $m_B$:

$$o_i = \frac{e^{m_A - m_{\text{new}}} \sum_{j \in A} e^{s_j - m_A} v_j + e^{m_B - m_{\text{new}}} \sum_{j \in B} e^{s_j - m_B} v_j}{e^{m_A - m_{\text{new}}} \sum_{j \in A} e^{s_j - m_A} + e^{m_B - m_{\text{new}}} \sum_{j \in B} e^{s_j - m_B}}$$

where $m_{\text{new}} = \max(m_A, m_B)$. This is exactly what the running update computes. The subtraction of the max ensures all exponents are non-positive, preventing overflow -- the same trick as the stable softmax, applied incrementally.

---

## 3. FlashAttention v1 (2022)

### 3.1 Algorithm

Tile $Q$ into blocks of size $B_r$ (rows) and $K, V$ into blocks of size $B_c$ (cols). The outer loop is over Q-blocks, the inner over K,V-blocks.

```
# Inputs: Q (N, d), K (N, d), V (N, d) in HBM
# Output: O (N, d), softmax stats (m, l) in HBM
# Block sizes: B_r (query rows per block), B_c (key cols per block)

for i in range(0, N, B_r):                       # outer loop: over Q blocks
    Q_i = load Q[i:i+B_r, :] -> SRAM            # (B_r, d)
    O_i = zeros(B_r, d) -> SRAM
    m_i = -inf  (vector of B_r)
    l_i = 0     (vector of B_r)

    for j in range(0, N, B_c):                   # inner loop: over K,V blocks
        K_j = load K[j:j+B_c, :] -> SRAM        # (B_c, d)
        V_j = load V[j:j+B_c, :] -> SRAM        # (B_c, d)

        S_ij = Q_i @ K_j^T                       # (B_r, B_c) -- stays in registers/SRAM
        m_ij = rowmax(S_ij)                      # (B_r,)
        P_ij = exp(S_ij - m_ij)                  # (B_r, B_c)
        l_ij = rowsum(P_ij)                      # (B_r,)

        m_new = max(m_i, m_ij)
        alpha = exp(m_i - m_new)
        beta = exp(m_ij - m_new)
        l_i = alpha * l_i + beta * l_ij
        O_i = diag(alpha) @ O_i + diag(beta) @ (P_ij @ V_j)
        m_i = m_new

    O_i = diag(1/l_i) @ O_i                      # final normalization
    store O_i -> HBM
    store m_i, l_i -> HBM                        # for backward pass
```

### 3.2 HBM Traffic Analysis

- **Q**: loaded once per Q-block -- total $N \cdot d$ reads.
- **K, V**: loaded once per (Q-block, K-block) pair. Since each Q-block reads all K and V: $\frac{N}{B_r} \cdot N \cdot d$ bytes per matrix.
- **O**: written once -- $N \cdot d$ bytes.
- **$m$, $\ell$**: tiny, $N$ bytes each.

**Total HBM** $\approx N d + \frac{2 N^2 d}{B_r}$ (no more $O(N^2)$ term from $S$, $P$ round-tripping!).

Compare this to standard attention's $O(N^2)$ HBM traffic. As long as $B_r$ is reasonably large (e.g. 128), the $\frac{N^2 d}{B_r}$ term is much smaller than $N^2$.

Tile size must be chosen so that $B_r \cdot d + B_c \cdot d + B_r \cdot B_c + B_r \cdot d$ fits in shared memory ($\approx$ 228 KB/SM on H100). For $d = 128$, $B_r = B_c = 128$ works comfortably.

### 3.3 Arithmetic Intensity

The inner loop does $2 B_r B_c d$ FLOPs for $Q K^\top$ plus $2 B_r B_c d$ for $P V$ = $4 B_r B_c d$, and loads $(B_c d + B_c d) = 2 B_c d$ bytes per iteration ($K_j$ and $V_j$; $Q_i$ is already in SRAM).

$$\text{AI}_{\text{inner}} \approx \frac{4 B_r B_c d}{2 B_c d \cdot b} = \frac{2 B_r}{b}$$

For FP16 with $B_r = 128$:

$$\text{AI} \approx \frac{2 \times 128}{2} = 128 \;\text{FLOP/B}$$

This is below the H100 ridge ($\sim 295$) but much better than standard attention's 64 FLOP/B. Bigger $B_r$ pushes AI higher, capped by shared memory capacity.

### 3.4 v1 Limitations

1. **Outer loop over Q, inner over K/V**. This means each Q-block does one long sequential scan of K,V and finalizes its $O$ before moving on. Fine for parallelism across Q-blocks (one per threadblock), but wastes available parallelism along the sequence dimension when batch $\times$ heads is small.
2. **Each threadblock emits a full O-row**. Non-rescaled partial $O$'s live in shared memory; register pressure is high because you carry $O_i$ inside the inner loop and rescale each iteration.
3. **No support for causal masking shortcuts** -- v1 still visits all K,V blocks even for diagonal-upper-zero entries.

---

## 4. FlashAttention v2 (2023)

Key insight: on modern GPUs with many SMs, FA-v1 leaves parallelism on the table because one Q-block = one threadblock. v2 makes three improvements.

### 4.1 Swap the Loops (Parallelize Over Seq Dim)

FA-v2 parallelizes the **forward pass** along the sequence dimension by launching one threadblock per **Q-tile per head per batch-element** (same as v1) -- but it also parallelizes the **backward pass** differently. More importantly, it restructures the inner computation:

```
# v2 inner loop: scale AT THE END, not every iteration
O_i = zeros(B_r, d)
l_i = 0
m_i = -inf

for j in range(0, N, B_c):
    S_ij = Q_i @ K_j^T
    m_new = max(m_i, rowmax(S_ij))
    P_ij = exp(S_ij - m_new)
    l_i = exp(m_i - m_new) * l_i + rowsum(P_ij)
    O_i = exp(m_i - m_new) * O_i + P_ij @ V_j    # still rescale each iter
    m_i = m_new

O_i = O_i / l_i   # single final scale
```

The mathematical steps are the same, but v2 writes them in a way that minimizes non-matmul FLOPs. Empirically, v2 achieves $\sim 2\times$ the throughput of v1 on A100 at long sequence lengths because:

1. **Fewer non-matmul FLOPs** in the inner loop (one rescale, not two).
2. **Better work partitioning across warps** inside a threadblock -- v1 partitioned K across warps (so each warp computed a slice of $P_{ij} V_j$), v2 partitions Q across warps (so each warp owns full rows of $O$, avoiding inter-warp reductions at the end).
3. **Causal masking skip**: for causal attention, inner-loop iterations where $j > i$ can be skipped entirely. v2 makes this skip efficient.

### 4.2 Warp Partitioning: "Split-Q" vs "Split-K"

- **Split-K (v1)**: 4 warps per block, each warp computes $P \cdot V$ for a slice of the K dimension. Then warps reduce across the K dimension to produce $O_i$. Extra shared-memory communication.
- **Split-Q (v2)**: 4 warps per block, each warp owns a set of Q-rows. No inter-warp reduction because each warp independently produces its rows of $O_i$.

Split-Q is strictly better once you're memory-efficient enough to keep $Q_i$ and the full $K_j$, $V_j$ tiles visible to every warp.

### 4.3 Backward Pass Parallelism

The backward recomputes $P$ using saved $(m, \ell)$ to avoid storing the $N^2$ attention matrix. v2 parallelizes the backward over the K sequence dimension (one block per K-tile), which increases parallelism when batch $\times$ heads is small.

### 4.4 Throughput Improvement Breakdown

The $\sim 2\times$ speedup of v2 over v1 decomposes approximately as:

| Source | Estimated contribution |
|--------|----------------------|
| Reduced non-matmul FLOPs | $\sim 1.3\times$ |
| Split-Q warp partitioning (no reduction) | $\sim 1.3\times$ |
| Causal masking skip (causal only) | $\sim 1.5\times$ on top |

These factors multiply. For non-causal attention v2 is roughly $1.7\times$ faster; for causal the skip adds another boost, reaching $\sim 2\times$ overall.

---

## 5. FlashAttention v3 (2024, Hopper)

FA-v3 is a Hopper-specific rewrite that exploits four new hardware features:

1. **TMA (Tensor Memory Accelerator)** -- async bulk tile loads from HBM to SMEM with one descriptor. Frees registers previously used for LDGSTS address computation.
2. **wgmma (warp-group matmul)** -- 4-warp async tensor core ops against shared memory operands. Bigger tiles, higher throughput than v7 `mma.sync`.
3. **Thread block clusters + DSMEM** -- blocks can read each other's shared memory via the SM-to-SM network.
4. **FP8 tensor cores** -- $2\times$ throughput vs FP16 at the cost of needing per-tile scaling.

### 5.1 Warp Specialization (Producer-Consumer)

v3 uses 3 warp groups per block (12 warps total) in a producer-consumer pipeline:

| Role          | Warps | Job                                                   |
|---------------|-------|-------------------------------------------------------|
| Producer      | 4     | Issue TMA loads for next K,V tile into shared memory |
| Consumer 1    | 4     | wgmma for $S = Q \cdot K^\top$, online-softmax stats           |
| Consumer 2    | 4     | wgmma for $O \mathrel{+}= P \cdot V$                                  |

The producer fills an N-deep pipeline (typically 2-3 stages) of SMEM buffers; consumers drain. Consumer-1 produces $P_{ij}$ which gets consumed by Consumer-2 -- they run concurrently on different tiles, hiding softmax latency behind the next matmul.

### 5.2 2-Stage Pipelined Softmax

Non-matmul ops ($\exp$, rowmax, rowsum) run on CUDA cores in parallel with the tensor-core $O \mathrel{+}= P \cdot V$ that uses the previous iteration's $P$. This is the single biggest reason FA-v3 hits 75%+ of theoretical FP16 peak on H100 ($\sim 540$ TFLOPS of 989 TFLOPS dense FP16).

The pipeline looks like:

```
Iteration i:
  Tensor cores: O += P_{i-1} * V_{i-1}    (uses PREVIOUS P)
  CUDA cores:   S_i = Q * K_i^T            (partial, via wgmma)
                 m_i, l_i = softmax_stats(S_i)
                 P_i = exp(S_i - m_i)      (ready for next iteration)
```

This overlap hides the latency of the non-matmul softmax operations almost entirely.

### 5.3 FP8 Forward

In FP8 mode, $Q$, $K$, $V$ are stored as E4M3 with per-block scales. The accumulator is FP32. Because the softmax requires full dynamic range, $P$ is computed in higher precision inside the register file before being cast back to FP8 for $O \mathrel{+}= P \cdot V$. Reaches $\sim 1.2$ PFLOPS FP8 on H100 at $d = 128$, $N = 16384$ (roughly $2\times$ FP16).

### 5.4 Performance Summary (H100, $d = 128$, FP16)

| Kernel          | TFLOP/s | % of peak |
|-----------------|---------|-----------|
| Standard (cuBLAS)| $\sim 100$    | $\sim 10\%$      |
| FA-v1           | $\sim 200$    | $\sim 20\%$      |
| FA-v2           | $\sim 335$    | $\sim 34\%$      |
| FA-v3           | $\sim 540$    | $\sim 55\%$      |
| FA-v3 FP8       | $\sim 1200$   | $\sim 60\%$ of FP8 peak |

Numbers drift across CUDA versions; use them for relative reasoning, not benchmarks.

---

## 6. Backward Pass

The forward saves $(O, m, \ell)$ per row. Backward recomputes $P$ on the fly rather than saving the $N^2$ attention matrix.

### 6.1 Math

Given $dO$, we need $dQ$, $dK$, $dV$. Let $L_i = m_i + \log(\ell_i)$ (the log-sum-exp). Then:

$$S = Q K^\top$$

$$P = \exp(S - L) \quad \text{(recomputed, rows normalized)}$$

$$dV = P^\top \cdot dO$$

$$dP = dO \cdot V^\top$$

$$D = \text{rowsum}(dO \odot O) \quad \text{(scalar per row, computable from saved } O\text{)}$$

$$dS = P \odot (dP - D)$$

$$dQ = dS \cdot K$$

$$dK = dS^\top \cdot Q$$

### 6.2 Derivation of $dS = P \odot (dP - D)$

This is the chain rule through the softmax. Let $P_i = \text{softmax}(S_i)$ for row $i$. The Jacobian of softmax is:

$$\frac{\partial P_i}{\partial S_i} = \text{diag}(P_i) - P_i P_i^\top$$

Applying this to the upstream gradient $dP_i$:

$$dS_i = P_i \odot dP_i - P_i \cdot (P_i^\top dP_i) = P_i \odot (dP_i - D_i)$$

where $D_i = P_i^\top dP_i = \sum_j P_{ij} \, dP_{ij}$. Since $P_{ij} = \frac{e^{S_{ij}}}{\ell_i}$ and $O_i = \sum_j P_{ij} V_j$, we can show $D_i = \text{rowsum}(dO_i \odot O_i)$. This avoids ever forming the $N \times N$ Jacobian.

### 6.3 Tiling Strategy

- Outer loop over **K,V-tiles** (one threadblock per K-tile).
- Inner loop over Q-tiles, reading $L_i$, $D_i$, $dO_i$, $Q_i$.
- Atomic-add into $dQ$ because multiple K-blocks contribute to the same Q-row.

The atomic-add into $dQ$ is the main cost; v2/v3 use split-K over Q-tiles as a post-step reduction to avoid atomics on hot rows.

### 6.4 Memory Savings

Without FA backward:
- must save $P$ ($N^2$) $\rightarrow$ $O(N^2)$ HBM.

With FA backward:
- save only $O$, $L$ ($O(N \cdot d)$ + $O(N)$).
- recompute $P$ during backward (extra FLOPs, but still faster overall because we stay in SRAM).

**Concrete example**: For $N = 8192$, $d = 128$, FP16:
- Standard: save $P$ = $8192^2 \times 2$ bytes $= 128$ MB per head.
- FlashAttention: save $O$ = $8192 \times 128 \times 2 = 2$ MB + $L$ = $8192 \times 4 = 32$ KB per head.
- Savings: $64\times$ less memory per head.

---

## 7. Tile-Sizing Math

For each tile, shared memory must hold:

```
Q_i:  B_r x d
K_j:  B_c x d
V_j:  B_c x d
S_ij: B_r x B_c    (only during matmul; can reuse SMEM)
```

On H100 with 228 KB usable SMEM per block, for $d = 128$, FP16 (2 bytes):

$$\text{mem} = 2 \cdot (B_r \cdot d + 2 B_c \cdot d + B_r \cdot B_c)$$

$$= 2 \cdot (128 B_r + 256 B_c + B_r B_c)$$

For $B_r = B_c = 128$:

$$\text{mem} = 2 \cdot (128 \times 128 + 256 \times 128 + 128 \times 128) = 2 \cdot (16384 + 32768 + 16384) = 131{,}072 \;\text{bytes} = 128 \;\text{KB} \; \checkmark$$

For $B_r = B_c = 256$:

$$\text{mem} = 2 \cdot (128 \times 256 + 256 \times 256 + 256 \times 256) = 2 \cdot (32768 + 65536 + 65536) = 327{,}680 \;\text{bytes} = 320 \;\text{KB} \; \times$$

Hence the standard $(128, 128)$ choice for $d = 128$. FA-v3 uses $(128, 128)$ for FP16 forward and up to $(128, 256)$ for FP8 (half the byte footprint).

**Key rule**: bigger $B_r$ helps arithmetic intensity but burns more SMEM and more registers. $B_c$ mainly trades off HBM traffic (smaller $B_c$ means more trips) vs wgmma efficiency.

### 7.1 Tile Size Selection Heuristic

Given SMEM budget $M_{\text{smem}}$, head dim $d$, and dtype size $b$:

$$B_r \leq \frac{M_{\text{smem}} / b - 2 B_c d}{d + B_c}$$

For maximum AI, maximize $B_r$ subject to this constraint. In practice, kernel authors also consider:

- **Register pressure**: each thread carries $B_r / \text{warps}$ rows of the $O$ accumulator. Too large $B_r$ spills registers to local memory.
- **Occupancy**: larger tiles reduce the number of blocks that can run concurrently on an SM. On H100 with 228 KB SMEM, a 128 KB tile allows 1 block/SM; a 64 KB tile allows 2-4 blocks/SM with better latency hiding.
- **wgmma alignment**: on Hopper, wgmma instructions require tile dimensions to be multiples of 16 (FP16) or 32 (FP8).

---

## 8. Causal, Sliding-Window, and Masked Variants

### 8.1 Causal

For causal attention, the mask is $j \leq i$. FA-v2 has three regimes in the inner loop:

1. **Fully unmasked blocks** ($j < i - B_c + 1$): no mask apply, full speed.
2. **Diagonal block** ($j$ overlaps $i \ldots i + B_r$): apply triangular mask inside the tile.
3. **Fully masked blocks** ($j > i$): **skip entirely** -- this is where most of the $\sim 2\times$ speedup on causal comes from. About half the blocks are skipped on average.

**Worked example**: For $N = 4096$ and $B_c = 128$, there are $\frac{N}{B_c} = 32$ K-blocks. For the Q-block at position $i = 2048$ (block index 16):
- Fully unmasked: blocks 0-14 (15 blocks)
- Diagonal: block 15-16 (1-2 blocks)
- Fully masked: blocks 17-31 (15 blocks, **skipped**)

So roughly half the work is skipped, matching the fact that a causal mask zeros out half the $N \times N$ matrix.

### 8.2 Sliding Window

Only visit K-tiles within $[i - W, \, i]$. Reduces the inner loop to $O(W \cdot d)$ per Q-block, giving $O(N \cdot W \cdot d)$ total FLOPs. Used by Mistral ($W = 4096$), Gemma-2.

### 8.3 Arbitrary Masks (ALiBi, document masks)

- **ALiBi**: add a bias $-m \cdot |i - j|$ to $S_{ij}$ inside the tile. Adds a few FLOPs, no memory cost.
- **Document/packed masks**: pass a block-sparse table so the inner loop can skip entire tiles. Implemented by FlexAttention (PyTorch 2.5+) and in SGLang.

---

## 9. Comparison to Alternatives

| Kernel                      | Strategy                   | Strengths                        | Weaknesses                          |
|-----------------------------|----------------------------|----------------------------------|-------------------------------------|
| Standard ($Q K^\top$, softmax) | Naive                      | simplest                         | IO-bound, $O(N^2)$ HBM                 |
| Memory-efficient attention  | Recompute in backward only | easy drop-in                     | forward is still IO-bound           |
| FlashAttention v1           | Fused forward, tiled       | $O(N)$ HBM, mem-saving             | suboptimal parallelism              |
| FlashAttention v2           | Split-Q, fewer non-mm FLOPs | $\sim 2\times$ over v1                     | pre-Hopper only                     |
| FlashAttention v3           | Warp specialization, TMA, FP8 | 75%+ of Hopper peak          | Hopper+ only                         |
| FlashDecoding               | Split-K decoding           | Parallelizes decode (tiny Q)     | only useful during generation       |
| FlashDecoding++             | Unified scaling            | Handles async KV tiles           | complex, niche                      |
| FlexAttention (PyTorch)     | User-defined masks/scores  | flexible, still flash-style      | slower than hand-tuned FA-v3        |
| xformers.memory_efficient   | Similar to FA              | pre-FA lineage                   | largely superseded                  |

---

## 10. FlashDecoding: Inference-Time Variant

During decoding, the query is just 1 token ($N_q = 1$), so $Q K^\top$ is a skinny matmul $(1, d) \times (d, N_{kv}) = (1, N_{kv})$. One threadblock per head per batch-element starves the GPU because $N_{kv}$ is large but there's no batch parallelism to spread across SMs.

**FlashDecoding** splits the K,V sequence across SMs:

```
for each batch, head:
    split KV into S chunks of size B_c
    launch S blocks in parallel
    each block computes partial (m_s, l_s, O_s) on its KV chunk
    final kernel reduces (m_s, l_s, O_s) across chunks to (m, l, O)
```

The reduction uses online softmax to combine partial stats. This saturates the GPU during decoding at batch=1, where standard FA leaves SMs idle. vLLM, TRT-LLM, and SGLang all ship FlashDecoding-style kernels.

### 10.1 Why Standard FA Fails at Decode

Consider Llama-3-70B with 8 KV heads. At batch=1, there are only 8 threadblocks to schedule across 132 SMs on H100 -- utilization is $\frac{8}{132} \approx 6\%$. FlashDecoding with $S = 32$ chunks per head gives $8 \times 32 = 256$ blocks, more than enough to fill the GPU.

The tradeoff is a small reduction kernel at the end that combines partial stats, but this is negligible compared to the utilization gain.

---

## 11. Integration with KV Cache & Paged Memory

Production inference uses **paged KV cache** (see [[KV_Cache]]): K and V are not contiguous in memory -- they live in fixed-size pages (typically 16 tokens) scattered by an allocator. FA kernels are adapted:

- **PagedAttention (vLLM)**: the kernel takes a block table `(batch, logical_block) -> physical_block_idx` and loads K/V pages indirectly.
- **Radix attention / prefix sharing**: multiple requests share KV pages at the common prefix; the kernel doesn't care -- it just reads through the block table.
- **Chunked prefill**: the prefill of a long prompt is broken into chunks that interleave with decodes. FA handles an arbitrary `(n_new_tokens, n_cached_tokens)` split per request.

---

## 12. Pseudocode Summary (Triton-style Forward)

This is a compact Triton kernel for FA-v2 forward. It's close to the real `flash_attn_triton` reference. Full version in [[Triton_and_Kernels]].

```python
@triton.jit
def fa_fwd(
    Q, K, V, O, L,           # L = log-sum-exp stats
    sm_scale,
    stride_qb, stride_qh, stride_qs, stride_qd,  # and similar for K,V,O
    N_CTX, HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz  = tl.program_id(1)  # batch*heads

    # Pointers for this Q-tile
    q_ptrs = Q + ...  # shape (BLOCK_M, HEAD_DIM)
    q = tl.load(q_ptrs)  # stays in registers

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Loop over K,V blocks
    hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else N_CTX
    for start_n in range(0, hi, BLOCK_N):
        k = tl.load(K + ...)              # (BLOCK_N, HEAD_DIM)
        v = tl.load(V + ...)              # (BLOCK_N, HEAD_DIM)

        s = tl.dot(q, tl.trans(k)) * sm_scale   # (BLOCK_M, BLOCK_N)
        if IS_CAUSAL:
            s = tl.where(causal_mask, s, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = alpha * l_i + tl.sum(p, 1)
        acc = alpha[:, None] * acc + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(O + ..., acc)
    tl.store(L + ..., m_i + tl.log(l_i))
```

---

## 13. Common Interview Questions

**Q: Why is standard attention memory-bound and what is the arithmetic intensity?**
A: Because $S$ and $P$ (both $N \times N$) must be written to and read from HBM. $\text{AI} \approx \frac{d}{b}$, e.g. 64 for FP16 $d = 128$, well below H100's ridge of $\sim 295$.

**Q: How does FlashAttention avoid materializing $S$?**
A: Tile $Q$, $K$, $V$ into blocks that fit in shared memory. For each (Q-tile, K/V-tile) pair, compute $S_{ij}$ in registers/SRAM, apply softmax with online statistics $(m, \ell)$, and accumulate into $O$ without ever writing $S$ to HBM.

**Q: Walk me through the online softmax update.**
A: Maintain running max $m$ and denominator $\ell$. When a new block's max is $m'$, compute $m_{\text{new}} = \max(m, m')$, rescale the old running values by $e^{m - m_{\text{new}}}$ and the new contributions by $e^{m' - m_{\text{new}}}$, and sum. Final output divided by $\ell$ once at the end. Mathematically identical to vanilla softmax.

**Q: Prove the online softmax is numerically identical to standard softmax.**
A: Consider $N$ scores split into blocks $B_1, \ldots, B_K$. After processing all blocks, $\ell = \sum_{j=1}^{N} e^{s_j - m}$ where $m = \max_j s_j$ (the global max accumulates correctly because $\max$ is associative). The output numerator is $\sum_{j=1}^{N} e^{s_j - m} v_j$, which after dividing by $\ell$ gives $\sum_j \frac{e^{s_j}}{\sum_k e^{s_k}} v_j$ -- exactly the standard softmax-weighted sum. The rescaling by $e^{m_{\text{old}} - m_{\text{new}}}$ at each step is the algebraic identity that makes the running sum track the correct normalization.

**Q: What are the main differences between FA-v1 and v2?**
A: v2 (1) does fewer non-matmul FLOPs per inner iteration (one rescale instead of two), (2) switches warp partitioning from split-K to split-Q inside the threadblock to avoid inter-warp reductions, (3) adds efficient causal skipping, and (4) parallelizes backward over the K dimension. End result: $\sim 2\times$ over v1 on A100.

**Q: What's new in FA-v3?**
A: Warp specialization (producer-consumer with 3 warp groups), TMA for async HBM-to-SMEM loads, wgmma for async tensor-core matmul against SMEM operands, a 2-stage pipeline that overlaps softmax with the next $P \cdot V$ matmul, and FP8 support with per-tile scales. Brings Hopper utilization from $\sim 35\%$ to $\sim 75\%$ of peak FP16.

**Q: Why is split-Q better than split-K for warp partitioning inside a block?**
A: In split-K each warp owns a slice of the K dimension, so computing $P \cdot V$ for that slice produces partial rows of $O$ that must be reduced across warps via shared memory. In split-Q each warp owns a slice of Q-rows and independently computes its final rows of $O$ -- no cross-warp reduction needed.

**Q: How does FlashDecoding differ from FlashAttention?**
A: FlashDecoding is for generation ($N_q = 1$). At low batch, the single Q-token per sequence doesn't give enough parallelism. FD splits the KV sequence across SMs, computes partial softmax stats per chunk, then reduces with a second kernel. Keeps the GPU busy at small batch.

**Q: What memory gets saved between forward and backward?**
A: $O$ (the output), and $L = m + \log(\ell)$ (log-sum-exp). Not $P$ or $S$ -- those are recomputed from $Q$, $K$, $V$ in the backward. This brings activation memory from $O(N^2)$ to $O(N \cdot d)$.

**Q: Why is the backward pass harder to parallelize than the forward?**
A: Each K-tile contributes to $dQ$ for multiple Q-tiles, creating write conflicts on $dQ$. FA-v2 partitions the backward by K, then does a reduction over K-tiles to accumulate $dQ$ (via atomics or a separate split-K reduce pass). Forward had no such conflict because each Q-tile writes its own O-rows.

**Q: When is FA NOT the right tool?**
A: (1) Very small sequence length ($N < \sim 256$) where the kernel launch / tile overhead dominates -- standard cuBLAS + softmax is fine. (2) Head dimension $d > 256$ where the tile doesn't fit comfortably in SMEM. (3) Attention with complex custom biases or non-softmax normalization -- FlexAttention may be easier.

**Q: Why doesn't FA use atomic operations on $O$ in the forward?**
A: Because each Q-tile is owned by exactly one threadblock, which writes its full O-row contribution once at the end. The rescaling ($\text{diag}(\alpha) \cdot O_i + \ldots$) is private to that block's registers/SRAM. No cross-block contention on $O$.

**Q: How does paged KV cache interact with FA?**
A: Instead of treating $K$, $V$ as contiguous matrices, the kernel takes a block table mapping logical positions to physical pages. Each tile load indirects through the block table. Adds address-computation overhead but preserves the fused forward structure. Used by vLLM's PagedAttention.

**Q: What's the FLOP count of FA forward vs backward?**
A: Forward: $4 N^2 d$ per head (two matmuls of $N^2 d$). Backward: roughly $5 N^2 d$ (recompute $P = N^2 d$, plus $dP$, $dS$, $dQ$, $dK$, $dV$ each $\sim N^2 d$). So backward is about $2.25\times$ forward FLOPs, matching typical training ratios.

**Q: What is the "ridge point" and where does FA land?**
A: Ridge = $\frac{\text{peak FLOPs}}{\text{peak BW}}$. For H100 FP16: $\frac{989 \;\text{TFLOPS}}{3.35 \;\text{TB/s}} \approx 295$ FLOP/B. FA's effective AI scales with $B_r$ ($\sim 128$ FLOP/B for $d = 128$ FP16), so it's still slightly memory-bound but much closer to compute-bound than standard attention, explaining the $3$-$5\times$ speedup.

**Q: If you had to estimate the wall-clock speedup of FA over standard attention for a given $N$ and $d$, how would you do it?**
A: Standard attention is memory-bound, so its runtime is dominated by HBM traffic: $\sim \frac{4 N^2 \cdot b}{\text{HBM BW}}$. FA's HBM traffic is $\sim \frac{2 N^2 d \cdot b}{B_r \cdot \text{HBM BW}} + \frac{N d \cdot b}{\text{HBM BW}}$. The speedup ratio is roughly $\frac{4 N^2}{2 N^2 d / B_r + N d} \approx \frac{2 B_r}{d}$ for large $N$. With $B_r = d = 128$, that's $\sim 2\times$ from IO alone, plus the benefit of fused kernels avoiding launch overhead.

---

## 14. Further Reading

- Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (NeurIPS 2022) -- v1.
- Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning" (ICLR 2024) -- v2.
- Shah et al., "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision" (2024) -- v3.
- Hong et al., "FlashDecoding++" (2024).
- vLLM PagedAttention paper (SOSP 2023).
- PyTorch FlexAttention docs (2.5+).

---

**Next:** [[KV_Cache]] -- how the K,V tensors are stored, paged, shared across requests, and moved across GPUs.

**See also:** [[Attention_Mechanisms]], [[CUDA_Optimization]], [[Triton_and_Kernels]], [[Memory_Hierarchy_and_Roofline]].

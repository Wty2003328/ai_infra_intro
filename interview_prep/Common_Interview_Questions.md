# Common Interview Questions

A consolidated bank of the most frequently asked AI infrastructure interview questions, organized by topic, with crisp model answers. Use this as a final cram sheet — the deeper context lives in the linked pages.

**Categories:**
1. GPU Architecture & Memory
2. CUDA / Kernels
3. Attention & Transformers
4. KV Cache
5. Batching, Scheduling, Disaggregation
6. Quantization
7. Distributed Training
8. Inference Frameworks
9. Networking & Collectives
10. Observability & Reliability
11. Big-Picture / Behavioral

---

## 1. GPU Architecture & Memory

**Q: Why is decode bandwidth-bound and prefill compute-bound?**
A: Decode reads the entire model weights and KV cache once per token but does only $O(d)$ FLOPs per token per layer — arithmetic intensity is low (~tens of FLOP/B), well below H100's ridge of ~295. Prefill processes many tokens per step, so arithmetic intensity scales with batch size and easily exceeds the ridge, making it compute-bound. The crossover happens at the **roofline ridge point**: any kernel below the ridge is memory-bound, any above is compute-bound. In practice, a single decode token with a 70B model does roughly $2 \times 70 \times 10^9 \approx 140\text{G}$ FLOPs but reads ~70 GB of weights — AI $\approx 2$ FLOP/B, far below the ridge. See [[Memory_Hierarchy_and_Roofline]].

**Q: What is "arithmetic intensity" and what is the ridge point on an H100?**
A: Arithmetic intensity (AI) is the ratio of FLOPs to bytes transferred for a kernel:

$$\text{AI} = \frac{\text{FLOPs}}{\text{Bytes}}$$

The ridge point is where peak compute equals peak bandwidth:

$$\text{Ridge} = \frac{\text{Peak FLOPS}}{\text{Peak BW}} = \frac{989\;\text{TFLOPS}}{3.35\;\text{TB/s}} \approx 295\;\text{FLOP/B}$$

Below the ridge, the kernel is memory-bound (bandwidth is the bottleneck). Above, it is compute-bound (ALUs are the bottleneck). For H100 SXM with FP16 tensor cores, the ridge is ~295. For FP8 (1979 TFLOPS), it doubles to ~590, making even more kernels memory-bound.

**Q: Compare HBM3 to HBM3e to GDDR.**
A: HBM stacks DRAM dies vertically with a wide interface (1024-bit), giving very high BW (TB/s class). HBM3e is a refresh with higher per-pin rates (B200: 192 GB at 8 TB/s). GDDR (gaming GPUs) is cheaper but ~10x lower BW per chip. Comparison:

| Memory | Bandwidth | Capacity (typical) | Use case |
|--------|-----------|---------------------|----------|
| HBM3 (H100) | 3.35 TB/s | 80 GB | Training/inference GPUs |
| HBM3e (B200) | 8 TB/s | 192 GB | Next-gen datacenter |
| GDDR6X (RTX 4090) | ~1 TB/s | 24 GB | Consumer/edge |

HBM's advantage is critical for LLM inference where decode is bandwidth-bound.

**Q: How big is L2 on H100? What's it for?**
A: 50 MB. Shared across all 132 SMs. Caches HBM reads, helps when multiple SMs touch the same data (e.g., shared weights across thread blocks). The L2 is particularly important for **persistent kernels** that keep data resident, and for the **wave quantization** effect: if a kernel launch's data fits in L2, subsequent accesses avoid HBM entirely. For context, A100 had 40 MB L2; the 50 MB on H100 is a modest bump that helps at tile boundaries.

**Q: What is shared memory? How big is it on H100?**
A: SM-local SRAM, manually managed by the programmer, ~228 KB per block on H100 (configurable split with L1 cache). Used for tile staging in matmul / FlashAttention-style kernels. ~10x lower latency and ~10x higher bandwidth than L2. On H100, each SM has 256 KB of combined shared memory/L1, configurable as up to 228 KB shared memory. The key programming pattern: load a tile from HBM into shared memory, then reuse it many times from shared memory, amortizing the HBM read.

**Q: What's the difference between a thread, warp, block, and grid?**
A: Thread: scalar execution unit. Warp: 32 threads executed in lockstep (SIMT) — the fundamental scheduling unit. Block (threadblock): up to 1024 threads that share SMEM, scheduled together on one SM. Grid: collection of blocks for one kernel launch. Hierarchy matters because synchronization only works within a block (`__syncthreads`), warps are the unit of instruction issue (divergent branches serialize within a warp), and grid dimensions determine total parallelism. See [[CUDA_Programming]].

**Q: What is occupancy? What limits it?**
A: Active warps per SM divided by max warps per SM (64 on H100). Limited by three resources: registers per thread (if each thread uses 128 registers, only 4 warps fit in 64K registers), SMEM per block (228 KB total), and threads per block (max 1024). Higher occupancy hides memory latency by giving the scheduler more warps to switch between, but isn't always optimal — some kernels achieve peak performance at 50% occupancy by using more registers/SMEM per thread for data reuse.

**Q: What are Tensor Cores and how do they differ from CUDA cores?**
A: CUDA cores execute scalar FP32/FP64/INT32 operations, one per clock per core. Tensor Cores execute small matrix-multiply-accumulate (MMA) operations in a single instruction — e.g., $D = A \times B + C$ where A, B are $16 \times 16$ matrices in FP16/BF16/FP8 and C, D are FP32 accumulators. On H100, Tensor Cores deliver ~989 TFLOPS FP16 vs ~67 TFLOPS FP32 from CUDA cores — a 15x gap. All modern LLM workloads are built around Tensor Core utilization; if your kernel isn't using Tensor Cores, you're leaving >90% of the chip's throughput on the table.

---

## 2. CUDA / Kernels

**Q: How do you achieve coalesced memory access?**
A: Threads in a warp access consecutive 128-byte aligned addresses. Pattern `data[threadIdx.x]` is coalesced; `data[threadIdx.x * stride]` is not. The memory controller combines coalesced accesses into a single wide transaction (128 bytes = 32 floats for 32 threads), achieving peak bandwidth. Non-coalesced access can degrade bandwidth by 10-32x. For matrix operations, this means row-major access along the fast dimension, or transposing data in shared memory when the natural access pattern is strided.

**Q: What is a bank conflict?**
A: SMEM is split into 32 banks; if two threads in a warp access different addresses in the same bank, the access serializes (N-way conflict = Nx latency). Avoid via padding (`SMEM[row][33]` instead of `SMEM[row][32]` so consecutive rows shift by 1 bank) or swizzling (XOR-based index remapping). A 32-way bank conflict is the worst case — essentially serial access. Broadcast (all threads reading the same address in one bank) is free.

**Q: What's TMA and why does it matter?**
A: Tensor Memory Accelerator on Hopper. Async bulk tile transfer from HBM to SMEM with one descriptor — frees registers used for address computation in pre-Hopper LDGSTS, and lets the producer-consumer pattern overlap loads with matmul. Key advantages: (1) single instruction initiates a multi-dimensional tile copy, (2) hardware handles address calculation and boundary checks, (3) integrates with the async pipeline (TMA + wgmma overlap). FlashAttention-3 uses TMA heavily for its warp-specialization pipeline.

**Q: Difference between `__syncthreads`, `__syncwarp`, and `__threadfence`.**
A: `__syncthreads()` synchronizes all threads in a block — a barrier, all must arrive before any proceed. `__syncwarp(mask)` synchronizes threads within a 32-thread warp specified by the mask. `__threadfence()` orders memory operations from the calling thread so that writes are visible to all other threads globally, but does **not** synchronize (other threads don't wait). Use `__syncthreads` for block-level coordination (tile loads), `__syncwarp` for warp-level algorithms (shuffles), and `__threadfence` for producer-consumer patterns across blocks (e.g., signaling completion via a flag).

**Q: When would you use Triton instead of CUDA?**
A: When the kernel is moderate complexity (matmul, softmax, attention variants) and you want autotuning + Python iteration speed without sacrificing too much performance. Triton's block-program model handles tiling, memory coalescing, and some SMEM management automatically. CUDA/CUTLASS for the absolute last 10-20% on Hopper-specific features (TMA descriptors, warp specialization, explicit wgmma). Rule of thumb: prototype in Triton, ship in CUDA only if the 10-20% gap matters for your workload.

**Q: What is warp specialization?**
A: Different warps in a threadblock take different roles — e.g., one group of warps loads data (producer) while another computes (consumer). Lets you build a deeper software pipeline: while consumer warps compute on tile N, producer warps prefetch tile N+1. Standard in FlashAttention-3 on Hopper. The programming model uses `setmaxnreg` to give producer warps fewer registers (they just move data) and consumer warps more (they do the math).

**Q: Why does a tiled matmul fit in shared memory?**
A: A tile of size $BM \times BK$ of A and $BK \times BN$ of B fits in SMEM (e.g., $128 \times 64 \times 2 + 64 \times 128 \times 2 = 32\;\text{KB}$, well under 228 KB). Compute the $BM \times BN$ output tile in registers via outer-product accumulations over the $BK$ dimension. Each element in the SMEM tile is reused $BM$ or $BN$ times, raising arithmetic intensity to $\frac{BM \cdot BN \cdot BK \cdot 2}{(BM \cdot BK + BK \cdot BN) \cdot 2} = \frac{BM \cdot BN}{BM + BN}$ FLOP/B, which exceeds the ridge for typical tile sizes.

---

## 3. Attention & Transformers

**Q: Walk through the steps of self-attention.**
A: Project input $X$ into queries, keys, and values per head: $Q = XW_Q$, $K = XW_K$, $V = XW_V$. Compute attention scores:

$$S = \frac{QK^T}{\sqrt{d_k}}$$

Apply causal mask (set future positions to $-\infty$), then softmax row-wise to get attention weights $P = \text{softmax}(S)$. Multiply by values: $O = PV$. Concatenate outputs from all heads and apply output projection $W_O$. Total cost per layer: $8d^2$ FLOPs for projections plus $4Nd$ FLOPs for attention, where $N$ is sequence length and $d$ is model dimension. See [[Attention_Mechanisms]].

**Q: Why is $\frac{1}{\sqrt{d_k}}$ scaling needed?**
A: Without it, dot products grow linearly with $d_k$ — specifically, if $q$ and $k$ have entries with unit variance, then $q \cdot k$ has variance $d_k$. This pushes softmax into a saturated regime where gradients vanish (all probability mass concentrates on one or two entries). Scaling by $\frac{1}{\sqrt{d_k}}$ keeps the variance of scores at $O(1)$, maintaining a well-behaved softmax distribution throughout training.

**Q: Derive the online softmax update.**
A: Maintain running max $m$ and denominator $\ell$. Given a new block with local max $m'$ and local exponential sum $\ell'$:

$$m_{\text{new}} = \max(m, m')$$

$$\ell_{\text{new}} = e^{m - m_{\text{new}}} \cdot \ell + e^{m' - m_{\text{new}}} \cdot \ell'$$

The running output accumulator is also rescaled:

$$O_{\text{new}} = e^{m - m_{\text{new}}} \cdot O + e^{m' - m_{\text{new}}} \cdot P'V'$$

Final output is $O / \ell$. This is the core of FlashAttention: process K/V blocks one at a time, never materializing the full $N \times N$ attention matrix.

**Q: What is GQA and why does it help?**
A: Grouped-Query Attention: $H_{kv} < H_q$, with K and V shared across groups of Q heads. Reduces KV memory by the group factor $H_q / H_{kv}$. Example: Llama-3-70B uses $H_q = 64$, $H_{kv} = 8$, giving 8x KV reduction. Decode is BW-bound (must read entire KV cache per token), so reducing KV size directly improves throughput. GQA sits between MHA ($H_{kv} = H_q$, no compression) and MQA ($H_{kv} = 1$, maximum compression but quality loss). GQA is the standard choice for 2024-2025 dense models.

**Q: What is MLA?**
A: Multi-head Latent Attention (DeepSeek-V2/V3). Instead of storing full K, V per head, stores a low-rank latent $c_{KV}$ (~512 dim) per token plus a small decoupled rotary part (~64 dim). Full K and V are reconstructed at attention time via cheap projections. KV per token per layer: $(512 + 64) \times 2 = 1152$ bytes in FP16, vs ~16 KB for MHA and ~2 KB for GQA-8. Roughly 30x compression vs MHA, 3x vs GQA. The cost is extra projection FLOPs at attention time, which are tiny compared to the bandwidth savings during decode.

**Q: What's RoPE? Why is it position-equivariant?**
A: Rotary Position Embedding rotates Q and K vectors by position-dependent angles. For position $m$, each pair of dimensions $(i, i+1)$ in Q is rotated by angle $m\theta_i$ where $\theta_i = 10000^{-2i/d}$. The key property: the dot product $q_m \cdot k_n$ depends only on the relative offset $m - n$ because:

$$\langle R_m q, R_n k \rangle = \langle R_{m-n} q, k \rangle$$

This gives the model relative-position behavior without explicit relative-position encoding. Applied per query-key pair as a 2D rotation in each pair of channels. RoPE is standard in Llama, Mistral, Qwen, and most modern LLMs.

**Q: Explain pre-norm vs post-norm.**
A: Pre-norm: $x + \text{Attn}(\text{LN}(x))$. Post-norm: $\text{LN}(x + \text{Attn}(x))$. Pre-norm trains more stably at depth because the residual stream stays well-conditioned — gradients flow through the skip connection without being distorted by the normalization. Post-norm can achieve slightly better final quality but is much harder to train without careful learning-rate warmup. All modern LLMs use pre-norm (often with RMSNorm instead of LayerNorm for efficiency).

**Q: What's SwiGLU?**
A: A gated FFN activation: $\text{FFN}(x) = W_{\text{down}} \cdot (\text{SiLU}(W_{\text{gate}} \cdot x) \odot W_{\text{up}} \cdot x)$. Uses 3 weight matrices instead of 2 in the standard FFN, slightly more parameters and compute per layer, but consistently better quality. SiLU (Sigmoid Linear Unit) is $x \cdot \sigma(x)$. The gating mechanism allows the network to learn which features to pass through. Standard in Llama, Mistral, Qwen, and most 2024+ architectures.

**Q: Why FlashAttention?**
A: Standard attention writes $S$ and $P$ (size $N^2$) to HBM, making it bandwidth-bound for long sequences. FA tiles Q, K, V into blocks fitting in SMEM, computes attention with online softmax in-register, never writes $S/P$ to HBM. Result: 2-5x speedup, memory from $O(N^2)$ to $O(N)$. The I/O complexity drops from $O(N^2 d)$ HBM reads to $O(N^2 d^2 / M)$ where $M$ is SRAM size — a factor of $M/d$ improvement.

**Q: Difference between FA-v1, v2, v3?**
A: FA-v1: tiled forward with online softmax, first to avoid materializing the $N \times N$ matrix. FA-v2: split-Q work partition (parallelize over Q blocks instead of K/V blocks for better GPU utilization) + better causal masking that skips empty blocks entirely -- 2x over v1. FA-v3: Hopper-specific optimizations including warp specialization (producer/consumer warps), TMA for async tile loads, FP8 via wgmma instructions, achieving ~75% of peak FP16 TFLOPS on H100. See [[FlashAttention_Deep_Dive]].

---

## 4. KV Cache

**Q: How big is the KV cache for Llama-3-70B at 8K tokens, batch 32?**
A: Per-token KV size:

$$\text{KV per token} = 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{bytes} = 2 \times 80 \times 8 \times 128 \times 2 = 327{,}680\;\text{bytes} \approx 320\;\text{KB}$$

Total: $320\;\text{KB} \times 8192 \times 32 \approx 80\;\text{GB}$. This is the entire HBM of one H100. This is why KV compression (GQA, MLA, FP8 KV cache) and paging are essential for production serving.

**Q: What's PagedAttention?**
A: KV cache split into fixed-size blocks (e.g., 16 tokens). Each sequence has a block table mapping logical positions to physical blocks in a global KV pool. Eliminates internal fragmentation (no padding to max length), supports variable-length batching, enables prefix sharing (multiple sequences share common prefix blocks via refcounting), and supports preemption (can free/swap individual blocks). The attention kernel takes the block table as input and indirects through it per logical block. Introduced by vLLM; now standard in all production engines.

**Q: How does prefix caching work?**
A: Hash each block on `(parent_hash, token_ids)` to create a content-addressable cache. New requests walk their prompt block-by-block; on hash hit, reuse the physical block (refcount++). Skip the prefilled prefix, only compute the suffix. The hash chain ensures that matching only happens when the full prefix matches, not just a single block in isolation. Big throughput win for chat (system prompt reuse), RAG (shared document chunks), and long-system-prompt workloads. Typical hit rates: 60-90% for chat with system prompts.

**Q: What's the difference between hash-based and radix-tree prefix caching?**
A: Hash-based (vLLM): block-level granularity (e.g., 16 tokens), only matches at block boundaries. Simple to implement, low overhead, but misses partial-block matches. Radix-tree (SGLang): token-level granularity using a trie data structure, matches at arbitrary token boundaries. Better hit rate (catches partial overlaps), but more bookkeeping (pointer chasing, more complex eviction). The difference matters most when prompts share long prefixes that don't align to block boundaries.

**Q: When would you use FP8 vs INT8 KV cache?**
A: FP8 (E4M3) on Hopper: hardware-supported (no dequant overhead), near-zero accuracy loss on most benchmarks, 2x capacity vs FP16. INT8: available on pre-Hopper hardware (A100), but slightly more accuracy risk on long contexts due to uniform quantization spacing struggling with outlier attention values. Default to FP8 if Hopper is available. For very long contexts (100K+), always verify with retrieval benchmarks (NIAH, Ruler) since long-range attention patterns are more sensitive to quantization.

**Q: What's "preemption" in serving?**
A: When the KV pool is full and a new high-priority request arrives, kick out a victim sequence to free blocks. Two strategies: (1) **recompute** — drop the victim's KV blocks entirely, recompute them from the prompt when the request resumes; (2) **swap** — copy blocks to host RAM, copy back when resumed. vLLM defaults to recompute for new incoming requests and swap for in-flight requests (which have already generated partial output). The choice is a latency-vs-compute tradeoff: recompute costs prefill FLOPs but avoids PCIe transfer latency.

---

## 5. Batching, Scheduling, Disaggregation

**Q: What is continuous batching?**
A: Each step, the scheduler decides which sequences run from a pool of in-flight ones. New sequences join immediately when there is budget; finished ones leave without disturbing others. Contrast with static batching (fixed cohort waits for slowest to finish): continuous batching achieves much higher GPU utilization when generation lengths vary. Standard in vLLM, SGLang, and TRT-LLM. The key insight: since each decode step produces exactly one token per sequence, the scheduler can make per-step admission decisions.

**Q: What is prefill-decode interference and how do you fix it?**
A: Mixing a compute-bound prefill (processing a long prompt) with bandwidth-bound decodes in the same batch makes the step take prefill's time, hurting decode TPOT (time per output token) for all co-batched sequences. Fixes:
- **Chunked prefill**: split long prompts into chunks (e.g., 512-2048 tokens), interleave chunks with decode steps. Bounds the interference per step.
- **Disaggregation**: run prefill and decode on separate GPU pools. Prefill pool optimized for compute throughput, decode pool for latency. KV cache transferred between them.
Chunked prefill is simpler and sufficient below ~1K RPS; disaggregation pays off at scale with bursty long prompts.

**Q: Walk through the lifecycle of a request in vLLM.**
A: (1) Client submits prompt via API. (2) Tokenizer converts to token IDs. (3) Scheduler checks admission: KV budget sufficient? Prefix cache hit? (4) Request joins the waiting queue. (5) On the next scheduler step, admitted to running set. (6) Prefill phase: process all prompt tokens (chunked if long), populate KV cache. (7) Decode loop: each step generates one token, appends to KV cache, checks stopping criteria. (8) On EOS or max-tokens: emit final output, release KV blocks (refcount-- on shared prefix blocks, free blocks with refcount 0). (9) Detokenize and stream or return to client.

**Q: Why is paging required for continuous batching?**
A: Continuous batching needs each sequence to grow independently with arbitrary length. With contiguous allocation, you'd need to either (a) reserve max-length for every sequence (massive waste) or (b) frequently relocate KV data as sequences grow (expensive copies). Paging gives each sequence its own block table over a global pool — sequences grow by allocating new blocks from the free pool without touching other sequences' data. Variable-length-friendly, fragmentation-free.

**Q: When does prefill-decode disaggregation pay off?**
A: When the workload mix has bursty long prompts that interfere with decode SLOs, and at scales where the KV-transfer overhead (typically via RDMA or NVLink across pools) is paid back by 30%+ utilization gain. Below ~1K RPS, simpler coupled + chunked prefill is usually sufficient. Key calculation: if KV transfer takes $T_{\text{xfer}}$ ms and you save $T_{\text{interference}} \times N_{\text{decode}}$ ms per step, disaggregation wins when $T_{\text{interference}} \times N_{\text{decode}} > T_{\text{xfer}}$.

**Q: How does speculative decoding work and when does it help?**
A: A cheap draft model (or draft head) generates $K$ future tokens autoregressively. The target model verifies all $K$ in one forward pass via rejection sampling: for each position, accept if the target agrees (or with probability proportional to target/draft likelihood ratio). Each accepted token is "free" — its compute is hidden in the bandwidth-bound decode step. Helps when acceptance rate $\alpha > 0.5$ and $T_{\text{draft}} / T_{\text{target}} < 1/K$. Hurts otherwise (wasted draft compute + wasted verification compute on rejections).

**Q: Compare Medusa and EAGLE.**
A: Medusa: $K$ parallel heads on top of the target's hidden state, each predicting a future token independently (non-autoregressive). Simple, no extra model, but limited by the non-autoregressive assumption — can't capture dependencies between predicted tokens. EAGLE: a small autoregressive head fed by the target's hidden state that predicts future **embeddings** (not tokens directly), decoded into tokens. Captures inter-token dependencies, achieving higher acceptance rates (0.85-0.95 vs 0.70-0.85 for Medusa) with moderate extra compute. EAGLE-2/3 add dynamic tree structures for even better throughput.

---

## 6. Quantization

**Q: What's the difference between W4A16 and W8A8 quantization?**
A: W4A16: weights quantized to INT4, activations stay FP16, dequantization happens in-kernel during matmul. Easier on accuracy (activations untouched), ideal for bandwidth-bound decode (halves weight reads vs FP8). W8A8: both weights and activations quantized to INT8 or FP8, enabling full Tensor Core INT8/FP8 throughput. Higher accuracy risk from activation quantization (outlier channels), but 2x compute speedup over FP16 on supported hardware. Choose W4A16 for decode-dominated workloads, W8A8 for prefill-heavy or balanced workloads.

**Q: What's GPTQ?**
A: Post-training weight quantization that uses **Optimal Brain Quantization** principles. For each column of the weight matrix, it quantizes the weight and propagates the rounding error to all remaining (not-yet-quantized) columns via the inverse Hessian of the layer's input. The Hessian captures input correlations, so error is distributed to dimensions the model cares less about. Requires a small calibration dataset (~128 samples). Achieves INT4 weights with typically sub-1 perplexity-point loss on large models (70B+).

**Q: AWQ vs GPTQ?**
A: AWQ (Activation-Aware Weight Quantization) identifies salient weight channels — those multiplied by large activations — and scales them up before quantization so the quantization noise lands on less-important channels. Much simpler than GPTQ (no Hessian computation, ~10x faster), and often better for chat/instruction-tuned models. GPTQ has a slight edge on base-model perplexity and non-chat distributions. In practice, AWQ has become the default for production INT4 serving.

**Q: SmoothQuant?**
A: Migrates quantization difficulty from activations to weights via per-channel scaling. The insight: activation outlier channels make activations hard to quantize, but those same channels in the weight matrix are easy to quantize. By scaling:

$$Y = (X \cdot \text{diag}(s)^{-1}) \cdot (\text{diag}(s) \cdot W)$$

activations become smoother (easier for INT8) and weights absorb the scale (still per-channel quantizable). The scale $s$ is chosen per channel based on activation magnitude statistics from calibration. Enables W8A8 with minimal quality loss.

**Q: Why is FP8 better than INT8 for activations?**
A: FP8's exponent bits handle wide dynamic range — outliers fit naturally without dominating the scale. INT8's uniform spacing means one large outlier forces a wide scale, leaving most values quantized to just a few levels. FP8 E4M3 has 4 exponent bits covering a range from $2^{-9}$ to 448, vs INT8's fixed -128 to 127. Hopper has hardware-fused FP8 scaling (per-tensor or per-block scale applied in the Tensor Core pipeline), making FP8 essentially free in terms of overhead.

**Q: How does FP8 KV cache impact quality?**
A: Typically <1% MMLU drop, <0.05 perplexity increase on standard benchmarks. Long-context retrieval (Ruler, NIAH/needle-in-a-haystack) is more sensitive — attention score precision matters when retrieving from 100K+ contexts. Always benchmark on retrieval tasks before deploying FP8 KV for long-context use cases. Per-head or per-block quantization (vs per-tensor) helps preserve quality on these tasks.

**Q: What should you NOT quantize?**
A: (1) **LM head** (output projection to vocabulary) — small relative to total model, quality-critical for token selection. (2) **LayerNorm/RMSNorm parameters** — tiny (just $d$ params each), no bandwidth benefit from quantizing. (3) **Embedding table** — lookup, not matmul; quantization saves memory but doesn't speed up compute. (4) Sometimes the **first and last transformer layers**, which empirically are more sensitive to quantization noise. The general rule: quantize what's large and bandwidth-bound (attention/FFN weight matrices); leave everything else at higher precision.

---

## 7. Distributed Training

**Q: Compare DP, TP, PP, and EP.**
A: **Data Parallelism (DP)**: replicate model on each rank, split batch across ranks, all-reduce gradients after backward. **Tensor Parallelism (TP)**: split individual layer weight matrices across ranks, all-reduce inside each layer's forward/backward. **Pipeline Parallelism (PP)**: split layers into stages across ranks, send activations point-to-point between stages. **Expert Parallelism (EP)**: split MoE experts across ranks, all-to-all per MoE layer to route tokens to the right rank. Each axis has different communication patterns and hardware requirements. See [[Parallelism_Strategies]].

| Strategy | Communication | Bandwidth need | Typical placement |
|----------|--------------|----------------|-------------------|
| DP | All-reduce (gradients) | Moderate | Across nodes |
| TP | All-reduce (per layer) | Very high | Intra-node (NVLink) |
| PP | Point-to-point (activations) | Low | Across nodes |
| EP | All-to-all (tokens) | High | Intra-NVLink domain |

**Q: What does ZeRO-3 (FSDP) do?**
A: Shards parameters, gradients, and optimizer state across DP ranks — each rank stores only $1/N$ of each. Before a layer's forward pass, all-gather the full parameters from all ranks; after backward, reduce-scatter the gradients. This enables training models that don't fit in a single GPU's memory. Communication cost: total bytes transferred is roughly the same as standard DP all-reduce, but spread across forward and backward passes, enabling overlap with compute. The key trade: more communication events (per-layer instead of per-step), but each is smaller and overlappable.

**Q: When do you use TP vs PP?**
A: TP within an NVLink domain (intra-node, typically 8 GPUs) because every layer requires two all-reduces (forward and backward), demanding the ~900 GB/s NVLink bandwidth. PP across slower fabric (inter-node via InfiniBand) because it only sends activations between stage boundaries — much lower bandwidth requirement, and the point-to-point pattern tolerates higher latency. Combined: TP=8 within each node, PP=N across nodes is the standard layout for frontier training.

**Q: What's the pipeline bubble?**
A: Idle time at pipeline start and end while stages wait to be filled. With $P$ stages and $M$ microbatches, the bubble fraction is approximately:

$$\text{Bubble} \approx \frac{P - 1}{M + P - 1}$$

Reduce via: (1) more microbatches $M$ (diminishing returns past $M \gg P$), (2) interleaved 1F1B schedule (each rank handles multiple non-contiguous stages, reducing effective $P$), (3) zero-bubble PP (fills bubble time with computation from other microbatches via careful scheduling). At $M = 4P$, bubble is ~20%; at $M = 16P$, it's ~6%.

**Q: Why do we use BF16 instead of FP16 for training?**
A: BF16 has the same 8-bit exponent range as FP32, so values never overflow/underflow during training — no loss scaling needed. FP16's 5-bit exponent has a much tighter range ($6 \times 10^{-8}$ to 65504), requiring careful loss scaling to avoid underflow in gradients. BF16 trades mantissa precision (7 bits vs 10 for FP16) for range safety. In practice, BF16 "just works" for training without any hyperparameter tuning for numerical stability. All modern training frameworks default to BF16.

**Q: What's Transformer Engine FP8 doing?**
A: Wraps each Linear layer with FP8 quantization logic. It maintains a moving maximum of activation magnitudes (delayed scaling — uses the max from the previous iteration to set the current scale), picks per-tensor scales to keep values in FP8 range, then runs Hopper FP8 matmul (wgmma) at 2x FP16 throughput. Master weights stay FP32; non-matmul operations (LayerNorm, softmax, residual adds) stay BF16. The "delayed" scaling avoids an extra all-reduce to compute the current iteration's max before the matmul.

**Q: What's selective activation recomputation?**
A: Instead of saving all activations for backward (memory-expensive) or recomputing all (compute-expensive), selectively recompute only the cheap-to-recompute, expensive-to-store activations. In transformers: recompute attention (cheap with FlashAttention — just re-run the fused kernel) but save FFN activations (expensive to recompute, relatively small). ~5% extra compute overhead, but dramatic memory savings (~40-60% reduction in activation memory). Standard in Megatron-LM and all frontier training stacks.

**Q: What's "critical batch size"?**
A: The batch size (in tokens) beyond which adding more tokens per step stops improving loss per total tokens processed — you're wasting compute on redundant gradient signal. Empirically estimated via the gradient noise scale (ratio of gradient variance to gradient magnitude squared). Modern frontier runs use 2-16M tokens per step; going beyond adds wall-clock time without proportional quality improvement. The critical batch size tends to increase during training as the loss landscape smooths out.

**Q: How big are checkpoints for 70B?**
A: Mixed-precision training state per parameter: 2 bytes (BF16 param) + 4 bytes (FP32 master) + 4 bytes (FP32 grad) + 8 bytes (Adam first + second moment) = 18 bytes/param. Total: $70 \times 10^9 \times 18 \approx 1.26\;\text{TB}$. Saved asynchronously (distributed checkpoint, DCP) to avoid blocking training. Typical cadence: every 1-2 hours plus every N training steps. FSDP/ZeRO-3 shards the checkpoint across ranks, so each rank writes only its shard.

---

## 8. Inference Frameworks

**Q: vLLM vs TRT-LLM?**
A: **vLLM**: Python + CUDA, open-source, broad model coverage (new architectures supported within days), fast feature uptake (PagedAttention, chunked prefill, spec decoding), easy ops (pip install, single config). **TRT-LLM**: NVIDIA-tuned C++ engine with an ahead-of-time compilation step, lowest latency on NVIDIA hardware (hand-optimized kernels, FP8 graphs), comprehensive quantization support, but requires per-config build pipelines and lags new architectures by weeks-months. Use vLLM for general serving and rapid iteration; TRT-LLM for latency-critical production where you can invest in the build pipeline.

**Q: What does SGLang add over vLLM?**
A: (1) **RadixAttention**: token-granularity prefix sharing via a radix tree (vs vLLM's block-level hash-based approach) — better cache hit rates for structured/agentic workloads. (2) **First-class structured output** via xgrammar integration — grammar-constrained generation without external tools. (3) **Rust-based scheduler** for lower overhead at high RPS. (4) Strong performance on chat, agentic, and RAG workloads where prefix reuse and structured output matter most.

**Q: What is Dynamo (NVIDIA)?**
A: A multi-node inference serving fabric that sits above the engine layer. Key features: disaggregates prefill and decode across separate GPU pools, routes requests with prefix-locality awareness, manages KV cache transfer between pools via NIXL (NVIDIA's transfer library). Can use TRT-LLM or vLLM as the underlying engine. Targets large-scale multi-node deployments where single-engine serving hits utilization limits.

**Q: How does multi-LoRA serving work?**
A: Each request specifies a LoRA adapter ID. The engine maintains a set of loaded adapters in GPU memory with LRU eviction for fleets with many adapters. Two approaches for execution: (1) **batch by LoRA** — group requests with the same adapter, run standard fused matmul with that adapter's weights; (2) **fused multi-LoRA kernel** (Punica/S-LoRA style) — a single kernel that takes per-row LoRA selection and applies different adapters to different rows in the batch. The fused approach is better for high-adapter-diversity workloads; batching is simpler and faster for low diversity.

**Q: How does structured output (JSON, regex, grammar) work?**
A: At each decode step, compute a set of allowed next tokens based on the current state of the grammar/schema. Mask disallowed tokens by setting their logits to $-\infty$ before sampling. Implementation: compile the constraint (JSON schema, regex, CFG) into a finite-state automaton or pushdown automaton; at each state, precompute the set of valid tokens (token IDs whose string representations are valid continuations). Outlines, xgrammar, and lm-format-enforcer each do this differently but the principle is the same: logit masking as a per-step post-processor.

---

## 9. Networking & Collectives

**Q: What is NVLink? How fast?**
A: NVIDIA's GPU-to-GPU interconnect, bypassing PCIe. On Hopper: ~900 GB/s aggregate bidirectional per GPU (18 links x 50 GB/s/link). NVSwitch connects all 8 GPUs in a DGX node at full bisection bandwidth (any GPU can talk to any other at 900 GB/s simultaneously). NVL72 extends the NVSwitch fabric to 72 GPUs in a single domain — critical for large EP and TP configurations. Compare to PCIe Gen5 x16 at 64 GB/s — NVLink is ~14x faster.

**Q: NVLink vs PCIe vs InfiniBand?**
A: Three tiers of interconnect serving different scopes:

| Interconnect | Bandwidth | Scope | Use |
|-------------|-----------|-------|-----|
| NVLink (Hopper) | 900 GB/s | GPU-to-GPU, intra-node | TP, EP all-to-all |
| PCIe Gen5 x16 | 64 GB/s | GPU-to-CPU/NIC | Weight loading, KV swap |
| InfiniBand NDR | 50 GB/s/port | Inter-node | DP all-reduce, PP activations |
| InfiniBand XDR | 100 GB/s/port | Inter-node (next-gen) | Same, 2x bandwidth |

The bandwidth hierarchy drives parallelism placement: TP on NVLink, PP/DP on InfiniBand.

**Q: What's GPUDirect RDMA?**
A: Allows the network interface card (NIC) to read/write GPU HBM directly, bypassing the host CPU and host memory entirely. Without GPUDirect RDMA: GPU -> host memory -> NIC (two copies, CPU involved). With GPUDirect RDMA: GPU -> NIC (one copy, zero CPU). Saves a PCIe round-trip and CPU overhead; cuts cross-node collective latency and reduces PCIe bandwidth pressure. Essential for high-performance distributed training and disaggregated serving (KV transfer).

**Q: What's NIXL?**
A: NVIDIA Inference Xfer Library. Provides a unified API for GPU-to-GPU, GPU-to-CPU, and GPU-to-storage data moves, automatically selecting the best transport (NVLink, RDMA, PCIe) based on topology. Primary use case: transferring KV cache between prefill and decode GPU pools in disaggregated serving (used by NVIDIA Dynamo). Abstracts away the complexity of heterogeneous interconnects.

**Q: How does ring all-reduce work?**
A: $N$ ranks arranged in a logical ring. **Phase 1 (reduce-scatter)**: $N-1$ steps, each rank sends $1/N$ of the data to its neighbor, receives $1/N$ from the other neighbor, and sums locally. After $N-1$ steps, each rank has the fully reduced version of $1/N$ of the data. **Phase 2 (all-gather)**: $N-1$ steps, rotate the complete chunks around the ring. Total bytes per rank: $\frac{2(N-1)}{N} \cdot B \approx 2B$ for large $N$. This is **bandwidth-optimal** — no algorithm can transfer fewer bytes. Optimal at large message sizes; for small messages, latency ($O(N)$ steps) dominates.

**Q: When does NCCL pick tree over ring?**
A: At small payload sizes where latency $\alpha \cdot N$ (ring's $N$ steps) dominates over bandwidth. Tree all-reduce has $O(\log N)$ rounds of latency vs ring's $O(N)$, but uses $2\times$ the bandwidth. NCCL automatically picks the better algorithm based on message size and topology. The crossover threshold is tuned per-hardware and can be configured via NCCL environment variables.

**Q: What is NVLink SHARP?**
A: In-fabric reduction by the NVSwitch hardware. Instead of GPUs sending data to each other and reducing locally, the NVSwitch performs the reduction in the switch itself: GPUs send partial data to NVSwitch, NVSwitch reduces, broadcasts result. All-reduce becomes a single round-trip instead of multiple ring/tree steps. Halves GPU-side bandwidth for all-reduce. Enabled with `NCCL_NVLS_ENABLE=1` on supported fabrics (Hopper NVSwitch and later). Particularly beneficial for small all-reduces (TP) where latency matters.

---

## 10. Observability & Reliability

**Q: What metrics matter for an inference service?**
A: Organized by layer:
- **Latency**: TTFT (time to first token), TPOT (time per output token), E2E (end-to-end) — track p50/p95/p99 by tier/model/request-type.
- **Throughput**: requests per second (RPS), tokens generated per second (tok/s).
- **Engine state**: active sequences, queue depth, KV cache occupancy %, prefix cache hit rate, preemption rate, batch utilization.
- **Hardware**: GPU SM utilization, HBM utilization, Tensor Core utilization (via DCGM/nvml).
- **Business**: SLO violation rate (% of requests exceeding latency target), cost per token.

Alert on: KV occupancy >85%, preemption rate >1%, TPOT p99 exceeding SLO, queue depth growing monotonically.

**Q: How do you debug a hung training run?**
A: Systematic approach: (1) `py-spy dump` on each rank to get stack traces — find where each rank is stuck. (2) Compare ranks: which is ahead/behind? A rank stuck in a collective while others are in a different collective suggests a crash or divergence. (3) Set `NCCL_DEBUG=INFO` to see which collective hung and on which rank. (4) Check IB link health (`ibstat`, `ibdiagnet`), DCGM for hardware events (ECC errors, thermal throttling, NVLink CRC errors). (5) Bisect on recent code/config changes. Most common cause: one rank threw an exception silently, all others hang waiting at the next collective.

**Q: How do you find a straggler?**
A: Per-rank step time histograms — any rank consistently >2x the median is a straggler. Tools: custom per-rank timing instrumentation (log step time per rank), DCGM utilization metrics (a straggler often shows lower SM utilization or thermal throttling), hardware event counters (ECC errors, XID errors). Cure: evacuate the workload from the bad node, replace the GPU/node. Prevention: pre-flight health checks on every node before launching a training job.

**Q: How would you triage "TPOT regressed 30% after deploy"?**
A: (1) Diff the deploy — what changed (model version, engine config, infra)? (2) Check engine config (chunk size changed? batch cap changed? new quant config?). (3) Inspect KV occupancy and prefix cache hit rate (did caching regress?). (4) Run `nccl-tests` to verify bus bandwidth (NIC or NVLink degradation?). (5) Check DCGM: SM/HBM utilization — kernel-bound or bandwidth-bound shift? (6) Profile a single request end-to-end with `nsys` or engine-level tracing. (7) If all infra checks pass, canary the old version to confirm the regression is code-related, then bisect.

**Q: What do you put in a runbook for an LLM service?**
A: Structured for on-call use: (1) **Common failure modes** with diagnostic steps: OOM (check batch config, KV occupancy), hung NCCL (check NIC, run nccl-tests), slow disk (checkpoint writes blocking), NIC down (failover procedure). (2) **Restart procedures**: per-replica, per-model-server, full pool restart with drain. (3) **SLO definitions**: TTFT p99 < X ms, TPOT p99 < Y ms, availability > 99.9%. (4) **Alert thresholds** mapped to severity and response time. (5) **Escalation paths**: who to page for hardware vs software vs model issues. (6) **Postmortem template** for incidents exceeding N minutes.

---

## 11. Big-Picture / Behavioral

**Q: Walk me through the most interesting performance optimization you've shipped.**
A: (Tell a story.) Structure: state the problem (specific metric, specific model/workload), what was tried before (and why it didn't work), your measurement/profiling approach (specific tools: nsys, ncu, custom benchmarks), the change you made (specific: kernel modification, config tuning, architectural change), the quantitative result (e.g., "TPOT p99 dropped from 45ms to 28ms, 38% improvement"), and lessons learned. Strong answers cite specific kernels, configs, and numbers — not vague "made it faster."

**Q: How do you decide what to optimize?**
A: Profile first — never guess. Use `nsys` for end-to-end timeline, `ncu` for per-kernel analysis, engine metrics for request-level breakdown. Identify the top 1-2 bottlenecks (Amdahl's law: optimizing a 5% component can't give more than 5% total improvement). Estimate ROI: engineering hours vs % speedup. Pick the highest ROI. Avoid premature micro-optimization on cold paths. Common framework: is it compute-bound or memory-bound? (Roofline.) If memory-bound, reduce data movement. If compute-bound, use faster math (FP8, better tiling).

**Q: How do you stay current in this rapidly evolving field?**
A: Read papers (arXiv daily for cs.LG/cs.CL, key labs' blogs). Track the OSS engines (vLLM, SGLang, TRT-LLM release notes — changes reveal where the field is heading). Follow GTC, OSDI, MLSys talks. Build small benchmarks to verify claims (don't trust paper numbers blindly). Follow specific researchers and teams on social media. Most importantly: actually run experiments — clone repos, reproduce results, understand why things work or don't.

**Q: Tell me about a time you had to debug a really hard distributed-systems issue.**
A: (Story.) Structure: the incident (what broke, what was the impact), initial observations (what you saw in metrics/logs), your hypotheses (ranked by likelihood), how you ruled them out (specific diagnostics), the final root cause (often surprising), the fix (both immediate and long-term), and post-incident hardening (what you changed to prevent recurrence — monitoring, testing, architecture).

**Q: What's one thing you'd improve about a popular framework?**
A: Pick something concrete and grounded. Good examples: "vLLM's V0 scheduler had Python-overhead bottleneck at high RPS; V1 fixed it but I'd push further on prefix-tree caching with LRU eviction policies" or "TRT-LLM's build pipeline is per-config; an in-memory engine cache with warm-start compilation would cut iteration time." Bad answer: anything generic ("better documentation"). Show you've used the framework deeply enough to have opinions.

**Q: Why this team / role?**
A: Be specific to the team's actual work. Examples: "The work on disaggregated serving is the hardest unsolved problem in inference cost — I want to work on KV transfer optimization across NVLink fabrics" or "I want to ship kernels that real production stacks rely on, and this team's contributions to FlashAttention/vLLM are exactly that." Generic answers ("you're doing cool stuff") don't land. Research the team's recent papers, PRs, and blog posts.

**Q: Where do you see GPU computing going in the next 5 years?**
A: Reasonable themes: (a) FP4/MXFP becoming the standard inference precision, (b) bigger NVLink domains (NVL72 -> NVL576+ with Rubin), (c) closer integration of inference engines with hardware features (TMA, async wgmma, in-fabric reduction), (d) MoE + latent attention as the default architecture (dense models only for small/edge), (e) post-Transformer architectures (SSM hybrids, diffusion-LM) gaining production traction, (f) heterogeneous fleets (H100 + B200 + dedicated inference accelerators like Trainium/TPU) requiring portable engine abstractions.

---

## 12. Crisp Answers Cheat Sheet

When you blank, fall back on:

- **"What is X?"** -- definition first, then mechanism, then why it matters for systems/serving.
- **"Compare X and Y."** -- axes (latency / throughput / quality / complexity / cost), say where each wins, give a concrete recommendation.
- **"How would you design ..."** -- clarify requirements, estimate scale (back-of-envelope), lay out tiers/components, walk a request through the system, discuss tradeoffs, address failure modes.
- **"How would you debug ..."** -- top-down (metrics -> system -> kernel), name specific tools at each level.
- **"How do you decide ..."** -- measurable criteria, prefer numbers over vibes, acknowledge uncertainty.

---

## 13. Mock Interview Plan

Two weeks before an onsite:
- Day 1-3: Knowledge gaps. Use this notebook's table of contents, mark weak topics, study each in depth.
- Day 4-7: Practice 4-6 system designs (see [[System_Design_Interview]]). Time yourself to 45 minutes each.
- Day 8-10: Coding patterns (see [[Coding_Patterns]]). Write kernels on a blank editor without references.
- Day 11-13: Mock interviews with peers; verbalize answers out loud — writing is not the same as speaking.
- Day 14: Light review; sleep well.

---

## 14. Further Reading

Each section above links to its dedicated page. For broader context:
- [[Index]] -- full table of contents.
- [[System_Design_Interview]] -- design framework + worked examples.
- [[Coding_Patterns]] -- the algorithmic / kernel-level coding side.

---

**Next:** [[Coding_Patterns]].

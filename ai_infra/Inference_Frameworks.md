# Inference Frameworks

A high-level map of the production LLM inference landscape: vLLM, SGLang, TensorRT-LLM, NVIDIA Dynamo, llm-d, TGI, MLC, llama.cpp, and the orchestration layers above them. Includes feature matrices, when to choose what, and the architectural patterns shared by all.

**Prerequisites**: [KV_Cache](KV_Cache.md), [Batching_and_Scheduling](Batching_and_Scheduling.md), [Parallelism_Strategies](Parallelism_Strategies.md), [Quantization](Quantization.md).

---

## 1. The Anatomy of an Inference Engine

Every serious LLM inference framework has the same core components, even if they call them different names:

```
┌─────────────────────────────────────────────────────────────┐
│  Frontend: HTTP / gRPC / OpenAI-compatible API              │
├─────────────────────────────────────────────────────────────┤
│  Tokenizer (HF tokenizers or custom)                        │
├─────────────────────────────────────────────────────────────┤
│  Scheduler / Engine                                         │
│    - admission control                                      │
│    - continuous batching loop                               │
│    - chunked prefill                                        │
│    - speculative decoding integration                       │
├─────────────────────────────────────────────────────────────┤
│  KV Cache Manager (paged, prefix cache)                     │
├─────────────────────────────────────────────────────────────┤
│  Model Runner                                               │
│    - layer-by-layer execution                               │
│    - parallelism (TP/PP/EP)                                 │
│    - kernel dispatch                                        │
├─────────────────────────────────────────────────────────────┤
│  Custom Kernels (FlashAttention, GEMM, paged attention,     │
│   sampling, quantized matmul)                               │
├─────────────────────────────────────────────────────────────┤
│  Hardware (CUDA / NCCL / drivers)                            │
└─────────────────────────────────────────────────────────────┘
```

Frameworks differ in (a) which kernels they ship vs import, (b) how the scheduler is implemented, (c) what model architectures are supported, (d) how operationally mature they are.

---

## 2. The Major Engines

### 2.1 vLLM

The de facto open-source standard. Originated PagedAttention. Python-heavy engine + C++/CUDA kernels for hot paths.

**Strengths**:
- Massive model coverage: most HF models work out of the box.
- Active community, fast feature uptake (FA-v3, FP8, MoE).
- Simple to deploy: `vllm serve <model>`.
- Best-in-class continuous batching + prefix caching.
- Multi-LoRA, speculative decoding, structured output.

**Weaknesses**:
- Python overhead on the scheduler critical path (improving with the V1 engine).
- Less optimized for very low-latency / low-batch than TRT-LLM.

### 2.2 SGLang

Born from RadixAttention (tree-structured KV cache). Strong on chat workloads with prefix sharing, structured output, and complex prompting.

**Strengths**:
- RadixAttention: token-granularity prefix sharing.
- First-class structured output (JSON, regex, grammar) via xgrammar.
- Fast scheduler in Rust.
- Competitive throughput, sometimes ahead of vLLM on chat workloads.

**Weaknesses**:
- Smaller ecosystem; fewer model integrations than vLLM.
- More complex internals, fewer learning resources.

### 2.3 TensorRT-LLM

NVIDIA's hand-tuned C++ engine. Built on TensorRT compiler stack. Hardware-optimized for NVIDIA GPUs.

**Strengths**:
- Best peak performance on NVIDIA, especially low-latency single-stream.
- Comprehensive quantization (FP8, INT8, INT4, AWQ, SmoothQuant).
- Tight integration with Triton Inference Server.
- Mature production deployments.

**Weaknesses**:
- Heavy build process: each model+config is compiled into an "engine" file.
- Limited model coverage compared to vLLM.
- Closed-source kernels (mostly), harder to customize.
- Slower to support new architectures.

### 2.4 NVIDIA Dynamo

The newer NVIDIA inference fabric (2024+). Disaggregates prefill and decode, manages multi-node fleets, integrates Triton + TRT-LLM as backends.

**Strengths**:
- Production-grade disaggregated serving.
- Built-in NIXL transport for KV.
- Pluggable engine backends.
- Smart router with prefix-locality awareness.

**Weaknesses**:
- Tied to NVIDIA stack.
- Newer; smaller community.

### 2.5 llm-d

Meta's open-source disaggregated serving stack. Conceptually similar to Dynamo but cross-vendor.

### 2.6 Hugging Face TGI (Text Generation Inference)

HF's Rust+Python serving stack. Strong out-of-the-box experience for HF Hub models. Less aggressive optimization than vLLM/TRT-LLM but solid baseline.

### 2.7 MLC-LLM / Apache TVM

Compile-once, run-anywhere (NVIDIA, AMD, Apple, Intel, mobile). Strong for edge deployment and heterogeneous GPU fleets.

### 2.8 llama.cpp

CPU-first (with GPU offload). Quantized weights via GGUF format. The default for local / edge deployments. Not for production data-center serving but ubiquitous.

### 2.9 LMDeploy / TurboMind

Shanghai-AI-Lab's framework. Strong on Chinese open-source models (InternLM, Qwen). Comparable performance to vLLM on supported models.

---

## 3. Feature Matrix (snapshot, late 2025 / 2026)

| Feature                | vLLM | SGLang | TRT-LLM | TGI  | MLC  | llama.cpp |
|------------------------|------|--------|---------|------|------|-----------|
| Continuous batching    | ✓    | ✓      | ✓       | ✓    | ✓    | partial    |
| Paged KV cache         | ✓    | ✓      | ✓       | ✓    | ✓    | -          |
| Prefix caching         | ✓ hash| ✓ radix| ✓       | ✓    | -    | -          |
| Chunked prefill        | ✓    | ✓      | ✓       | ✓    | -    | -          |
| Disaggregated PD       | ✓    | ✓      | ✓ via Dynamo | -| -    | -          |
| FlashAttention v3      | ✓    | ✓      | ✓       | ✓    | -    | -          |
| FP8 weights & KV       | ✓    | ✓      | ✓       | ✓    | partial | -      |
| Speculative decoding   | ✓    | ✓      | ✓       | ✓    | -    | -          |
| Multi-LoRA serving     | ✓    | ✓      | ✓       | ✓    | -    | -          |
| Structured output      | ✓ outlines| ✓ xgrammar| ✓ xgrammar| ✓| -| -      |
| MoE                    | ✓    | ✓      | ✓       | ✓    | partial | partial |
| TP                     | ✓    | ✓      | ✓       | ✓    | ✓    | partial    |
| PP                     | ✓    | ✓      | ✓       | -    | -    | -          |
| EP                     | ✓    | ✓      | ✓       | partial | -| -          |
| OpenAI API             | ✓    | ✓      | via Triton| ✓  | -    | ✓          |
| AMD GPUs               | ✓    | partial| -       | partial| ✓ | ✓          |
| CPU only               | partial | -   | -       | partial | ✓ | ✓          |

---

## 4. Choosing a Framework

### 4.1 Production GPU fleet, OpenAI-style API

**vLLM** for general-purpose, **SGLang** if you need aggressive prefix sharing or structured output, **TRT-LLM** if you need the absolute lowest latency on NVIDIA and can afford the build pipeline.

### 4.2 Multi-node disaggregated serving

**Dynamo** (NVIDIA) or **llm-d** (open-source) on top of vLLM/TRT-LLM as backends.

### 4.3 Edge / heterogeneous

**MLC** for cross-vendor compile-once. **llama.cpp** for CPU-only or Apple Silicon.

### 4.4 Research / quick iteration

**vLLM** offline mode (`LLM(...).generate(...)`) for batch generation; **SGLang** if you need structured generation.

---

## 5. Common Architectural Patterns

### 5.1 Engine Process Model

Most engines run as a single process per inference replica with a worker thread/process per GPU shard:

```
Engine (Python, scheduler) ──► Worker[0] (GPU 0)
                          ──► Worker[1] (GPU 1)
                          ...
                          ──► Worker[N-1] (GPU N-1)
```

Workers communicate via PyTorch distributed (NCCL). The scheduler dispatches a step's input tensors to all workers and they run in lockstep.

### 5.2 The Step Loop

```
while running:
    runnable = scheduler.next_step()        # which seqs run this step
    inputs = build_packed_inputs(runnable)  # tokens + position_ids + block_table
    logits = workers.forward(inputs)        # collective forward across shards
    samples = sampler(logits, params)
    scheduler.update(runnable, samples)
    emit_streaming_outputs(samples)
```

The scheduler handles admission, preemption, prefix-cache lookup, KV allocation/free.

### 5.3 KV Manager

Pages = fixed-size blocks (16 tokens). Per-sequence block table. Allocation: pop from free list, refcount sharing for prefix cache. Eviction policy: LRU on cached prefixes; recompute or swap on preemption.

### 5.4 Sampler

Per-sequence: temperature, top_k, top_p, repetition penalty, frequency/presence penalty, logit bias, grammar mask. Implementation: Triton or CUDA kernels with per-row params. Output: token IDs (and optionally logprobs).

### 5.5 Streaming

Each emitted token goes through the response queue. SSE (Server-Sent Events) or chunked HTTP / gRPC streaming. Tokenizer detokenizes on-the-fly with byte-pair recovery for incomplete UTF-8 sequences.

---

## 6. The vLLM V1 Engine

vLLM's "V1" engine (rolled out 2024–2025) is a major rewrite addressing Python overhead:

- **Async executor**: scheduler decoupled from worker dispatch.
- **C++ scheduler core** for the hot path.
- **Reduced API surface**, opinionated defaults.
- **Improved multi-modal** integration.

If you're new to vLLM today, you're using V1 by default. The mental model is the same as V0 but the perf ceiling is higher.

---

## 7. Triton Inference Server (the orchestration layer)

Distinct from Triton DSL (kernel language) — Triton Inference Server is NVIDIA's model server that hosts multiple models, multi-framework, with HTTP/gRPC API. Engines like TRT-LLM, vLLM (via backend), and ONNX models all plug in.

Capabilities: dynamic batching, ensemble pipelines, model versioning, A/B testing, GPU pooling.

---

## 8. Cost & Latency Reasoning

Same model, same hardware, three frameworks: throughput differences come from kernel quality, scheduler efficiency, and feature support.

Rough order on a Hopper node (Llama-3-70B FP8, 8K context, normal chat workload):

| Framework  | TTFT p50 | TPOT p50 | Tokens/sec/GPU |
|-----------|----------|----------|----------------|
| TRT-LLM   | best (lowest latency, single-stream) | best | high |
| vLLM      | ~10–30% higher TTFT | ~5–15% higher TPOT | high |
| SGLang    | best when prefix cache hits | competitive | high |
| TGI       | ~20–50% higher than vLLM | similar | medium |

These numbers move with releases — always benchmark on your workload.

---

## 9. Multi-Modal Inference

Vision, audio, image-gen extensions follow a similar pattern but add:
- Image/audio encoders (often a separate ViT/Whisper/CLIP forward pass).
- Modality tokens injected into the LLM's input sequence.
- KV cache covers the multimodal tokens plus subsequent text.

vLLM, SGLang, and TRT-LLM all support major VLMs (LLaVA, Qwen-VL, InternVL, Gemini-style models with adjustments).

---

## 10. Common Pitfalls

- **Picking TRT-LLM without a build pipeline**: each model+TP+precision+seq_len combo is its own compiled engine. Plan for build CI.
- **Underestimating scheduler overhead**: at high RPS the scheduler can become CPU-bound. vLLM V1 helped a lot; still profile.
- **No prefix cache**: easy 2–3× throughput left on the table for chat / RAG workloads.
- **Wrong attention impl**: forgetting to enable FA-v3 on Hopper.
- **Ignoring observability**: no per-request TTFT/TPOT histograms = blind to SLO violations.

---

## 11. Common Interview Questions

**Q: Compare vLLM and TensorRT-LLM.**
A: vLLM is open-source Python+CUDA, broad model coverage, easy to deploy, fast feature uptake. TRT-LLM is NVIDIA's compiled C++ engine, lowest peak latency on NVIDIA hardware, best quantization breadth, but requires per-config build pipelines and lags on new model architectures. Pick vLLM for general-purpose, TRT-LLM for latency-critical NVIDIA-only deployments.

**Q: What's special about SGLang?**
A: RadixAttention — a token-granularity prefix tree that shares KV across requests far more flexibly than block-hash schemes. Plus first-class structured output. Best for chat / agentic / RAG workloads with heavy prompt reuse.

**Q: What's the role of NVIDIA Dynamo?**
A: A multi-node serving fabric that disaggregates prefill and decode, routes requests with prefix-locality, manages KV transfer via NIXL, and uses TRT-LLM/vLLM as engine backends. The orchestration layer above a single inference engine.

**Q: How does PagedAttention enable continuous batching?**
A: Each sequence has an independent block table over a global block pool. Different sequences can have different lengths and grow independently. The scheduler can admit and evict sequences without copying or reshaping tensors. Without paging, ragged batching requires expensive padding or per-request memory regions.

**Q: What does "structured output" mean and how is it implemented?**
A: Constrain generation to a regex / JSON schema / grammar. Implementation: at each decode step, mask logits for tokens that would violate the constraint (set to -inf). Libraries: Outlines, xgrammar, lm-format-enforcer. Engines integrate via per-step logit processors.

**Q: Why is multi-LoRA serving non-trivial?**
A: Each LoRA adapter changes the projection weights. Naive impl loads weights per request → slow. Fused multi-LoRA kernels handle a batch with mixed LoRA IDs in one matmul (Punica/S-LoRA). Engine has to track which adapter to use per row of the batch and route tokens accordingly.

**Q: How does the inference engine handle EOS?**
A: Sampler outputs the EOS token ID; scheduler marks sequence finished, emits final output, releases its KV blocks (decrements refcounts on shared prefix blocks), and frees the slot for a new request.

**Q: How would you debug a sudden TPOT regression in production?**
A: (1) Check if input distribution changed (longer prompts? bigger batch?). (2) Inspect KV occupancy and prefix-cache hit rate. (3) GPU utilization from nvidia-smi/Nsight. (4) Compare engine version vs known-good. (5) Check NCCL bus-bw. (6) Look for failure-mode regressions (NaN-handling adding sync points, etc.).

**Q: What's the difference between "Triton" the kernel language and "Triton" the inference server?**
A: Different products, both NVIDIA-adjacent. Triton DSL = OpenAI's GPU programming language for writing custom kernels (later acquired into the OSS world). Triton Inference Server = NVIDIA's model serving platform supporting multiple frameworks. Confusing naming.

**Q: What does the engine do during a single decode step?**
A: (1) Scheduler picks runnable sequences (admit, preempt, advance). (2) Build packed inputs (tokens, positions, block tables). (3) Forward pass across model with TP/PP shards. (4) Sample (with per-seq parameters and constraints). (5) Update KV and per-sequence state. (6) Stream new tokens to clients.

**Q: How does engine choose batch size each step?**
A: Greedy: include as many runnable sequences as fit in the KV budget and don't exceed compute budget (esp. prefill chunks). Optionally cap by max_batch and admission policies. Dynamic — varies step to step.

**Q: When would you NOT use a high-throughput engine like vLLM?**
A: Single-user local chat (llama.cpp / MLC). Edge / mobile (MLC). Very latency-sensitive single-stream where TRT-LLM wins clearly. Research where you need full control over execution (raw PyTorch).

---

## 12. Further Reading

- vLLM paper (SOSP 2023) and GitHub.
- SGLang paper (NeurIPS 2024).
- TensorRT-LLM docs.
- NVIDIA Dynamo announcement and GTC talks.
- llm-d project (Meta).
- Triton Inference Server docs.
- "Deep Dive into LLM Inference" — various blog posts and Anyscale, AnyScale, Mosaic AI, NVIDIA technical blogs.

---

**Next:** [vLLM_Internals](vLLM_Internals.md).
**See also:** [KV_Cache](KV_Cache.md), [Batching_and_Scheduling](Batching_and_Scheduling.md), [Quantization](Quantization.md), [Production_Architecture](Production_Architecture.md).

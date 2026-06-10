# L8 — Inference & Serving

> The capstone layer. Every hardware constraint (L0-L3), every kernel optimization (L5), every architectural decision (L6), and every training choice (L7) converges here: the production inference system that serves millions of requests per day. L8 covers the full serving stack from KV cache management through scheduling, disaggregation, framework internals, orchestration, observability, and production architecture.

---

## Scope of this layer

L8 covers **inference and serving systems** for large language models.

1. **KV cache** — layout, memory math, PagedAttention, prefix caching, radix tree, offloading.
2. **KV compression** — MLA, KIVI, StreamingLLM, H2O, SnapKV, Quest.
3. **Batching and scheduling** — continuous batching, chunked prefill, admission control, preemption.
4. **Speculative decoding** — vanilla, self-speculation, Medusa, EAGLE, acceptance rate math.
5. **Prefill/decode disaggregation** — motivation, architecture, KV transfer, benchmarks.
6. **Long context engineering** — YaRN, LongRoPE, iRoPE, NSA, MoBA, ring attention, chunked prefill.
7. **Multimodal inference** — VLMs, audio, video gen, unified any-to-any.
8. **Inference frameworks** — vLLM, SGLang, TensorRT-LLM, NVIDIA Dynamo, llm-d, TGI feature comparison.
9. **vLLM internals** — engine architecture, scheduler, workers, executor, block manager, APC.
10. **Disaggregated serving 2025** — Mooncake, DistServe, Splitwise, Sarathi-Serve, NVIDIA Dynamo, llm-d.
11. **Kubernetes and orchestration** — GPU operator, device plugin, MIG, topology-aware scheduling, HPA/KEDA.
12. **Observability and debugging** — TTFT/ITL/TPOT metrics, DCGM, Nsight, debugging flow.
13. **Production architecture** — reference stack, capacity planning, cost modeling, failure modes.

---

## Pages (read in order)

1. [KV_Cache](KV_Cache.md)
2. [Modern_KV_Compression](Modern_KV_Compression.md)
3. [Batching_and_Scheduling](Batching_and_Scheduling.md)
4. [Speculative_Decoding](Speculative_Decoding.md)
5. [Prefill_Decode_Disaggregation](Prefill_Decode_Disaggregation.md)
6. [Long_Context_Engineering](Long_Context_Engineering.md)
7. [Multimodal_Inference](Multimodal_Inference.md)
8. [Inference_Frameworks](Inference_Frameworks.md)
9. [vLLM_Internals](vLLM_Internals.md)
10. [Disaggregated_Serving_2025](Disaggregated_Serving_2025.md)
11. [Agentic_Inference](Agentic_Inference.md)
12. [GPU_Multitenancy](GPU_Multitenancy.md)
13. [Kubernetes_and_Orchestration](Kubernetes_and_Orchestration.md)
14. [Observability_and_Debugging](Observability_and_Debugging.md)
15. [Production_Architecture](Production_Architecture.md)

---

## Reading order

Algorithmic foundations first: KV_Cache → Modern_KV_Compression → Batching_and_Scheduling → Speculative_Decoding → Prefill_Decode_Disaggregation → Long_Context_Engineering → Multimodal_Inference.

Then systems: Inference_Frameworks → vLLM_Internals → Disaggregated_Serving → Agentic_Inference → GPU_Multitenancy → Kubernetes → Observability → Production_Architecture.

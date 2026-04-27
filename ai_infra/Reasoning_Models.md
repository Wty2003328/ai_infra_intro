# Reasoning Models and Test-Time Compute

The "thinking" / "reasoning" / "long-CoT" model class (o1, o3, o4-mini, R1, R1-0528, Claude extended-thinking, Gemini-2.5 thinking, Qwen-3 thinking-mode) reshaped what serving infrastructure must handle in 2025. This page covers the algorithmic foundations, the post-training pipelines (R1-style, o1-style), the inference-time mechanics, and the systems implications — long generations, dynamic budgets, KV pressure, tool integration.

**Prerequisites**: [Distributed_Training](Distributed_Training.md), [Frontier_Models_2025_2026](Frontier_Models_2025_2026.md), [Inference_Frameworks](Inference_Frameworks.md), [Speculative_Decoding](Speculative_Decoding.md).

---

## 1. What Changed in 2024–2026

OpenAI's o1 (Sep 2024) demonstrated a new scaling axis: **compute spent at inference time** to "think before answering." DeepSeek-R1 (Jan 2025) reproduced this in open weights and showed the recipe: heavy RL with rule-based rewards on a strong base model, optionally cold-started with a small CoT SFT.

Within a year, every frontier lab shipped a reasoning variant:

| Lab | Reasoning variant | Mechanism notes |
|-----|-------------------|------------------|
| OpenAI | o1, o1-pro, o3, o3-pro, o4-mini | Hidden CoT; user sees only final answer; reasoning tokens billed |
| Anthropic | Claude-3.7 / 4 with "extended thinking" | User-visible reasoning, configurable thinking budget |
| DeepSeek | R1, R1-0528 | Visible CoT, distilled small variants |
| Google | Gemini-2.5 Pro / Flash with "thinking" | Configurable budget |
| Qwen | Qwen-3 with `/think` mode | Toggleable per request |
| xAI | Grok-3 / Grok-4 reasoning | Visible CoT |
| Mistral | Magistral | Visible CoT, smaller reasoning model |

The shared shape: a strong base model + RL post-training that teaches it to produce long internal reasoning that improves answer quality. Inference-time, the model spends K tokens on reasoning before emitting M tokens of answer.

---

## 2. Algorithmic Foundations

### 2.1 Test-Time Scaling Laws

Empirical finding (Brown 2024 et al.): for a fixed model, accuracy on reasoning benchmarks improves roughly logarithmically with the number of generated tokens (or sampled completions). Several techniques exploit this:

- **Single long CoT** (o1, R1): one trace, longer is better up to a point.
- **Self-consistency / majority voting**: sample N independent CoTs, take the modal answer.
- **Best-of-N + verifier**: sample N, score with a process or outcome reward model, return the best.
- **Tree search (MCTS, ToT)**: branch at decision points, expand promising branches.
- **Iterative refinement**: solve, critique, refine.

Production reasoning models lean heavily on the first two; the rest are research-grade or used in research-mode evaluation.

### 2.2 Process vs Outcome Rewards

- **Outcome reward**: only the final answer is graded (correct / incorrect for math; tests pass / fail for code). Easy to construct.
- **Process reward (PRM)**: each step of the reasoning trace gets a score. Better signal but requires either expensive human annotation (e.g., PRM800K) or LLM-judge proxies. Popular for math; less so for general reasoning.

R1 used purely outcome rewards (math correctness via parser, code correctness via test harness) — and that was sufficient.

### 2.3 R1-Zero: Pure RL Emergence

DeepSeek's R1-Zero experiment: take V3 base, run pure RL (GRPO) with rule-based rewards, no SFT. Reasoning behavior **emerges**:

- Self-correction.
- Backtracking ("wait, that's wrong; let me try…").
- Length increases as training progresses.
- Cross-language switching (Chinese-English mixed mid-trace).

The unreadable mixed-language output motivated the R1 cold-start phase: a few thousand SFT examples to anchor the format.

### 2.4 Distillation From Reasoning Teachers

R1-distilled-7B/14B/32B/70B are small models trained to imitate R1's CoTs. Surprisingly capable — outperforms RL'd small models because the reasoning patterns transfer better through distillation than they emerge through small-model RL.

Implication: small reasoning models for cost-sensitive deployments are practical now.

---

## 3. The R1-Style Post-Training Pipeline

The recipe everyone is copying. End-to-end:

```
[1] Strong base model (V3 / Llama-3-405B / Qwen-2.5)
            │
            ▼
[2] Cold-start SFT (few thousand curated CoT)         ← optional but useful
            │
            ▼
[3] RL (GRPO / PPO) with outcome rewards
            - Math: parser-graded
            - Code: test-graded
            - General: LLM-judge or rules
            - Sample K=16+ completions per prompt
            - Long generations (10K+ tokens)
            │
            ▼
[4] Rejection sampling / data curation
            - Use RL'd model to generate large CoT corpus
            - Filter to high-quality
            │
            ▼
[5] Final SFT on curated corpus + safety
            │
            ▼
[6] Optional: distill into smaller models
```

### 3.1 GRPO Recap

For each prompt:
1. Sample K completions from current policy (e.g., K=16).
2. Score each with reward function R.
3. Compute advantages: `A_i = (R_i − mean) / std`.
4. PPO-style clipped policy gradient update on each (token, advantage) pair.
5. KL penalty against reference (the cold-start SFT model).

Memory: 2 model copies (policy + reference). No critic. Simpler than PPO.

### 3.2 The Inference-Inside-Training Bottleneck

Step 3 dominates wall-clock time. Sampling K=16 long traces per prompt, with 10K-100K output tokens each, generates **trillions of tokens** during a single RL run. At billion-parameter scale this saturates the inference engine completely.

Production stacks (veRL, OpenRLHF, NeMo-Aligner-2, Skywork OREAL):
- Dedicate a subset of GPUs as the **inference cluster**, running vLLM/SGLang.
- Training cluster periodically pushes weight updates to inference cluster (every N policy updates).
- Replay buffer between them.

Weight sync mechanisms:
- **NCCL broadcast** of full weights: simple, slow at frontier scale.
- **Sharded incremental sync**: only delta from last sync, plus periodic full resync.
- **Shared-memory weights**: when training and inference share GPUs (rare; complex scheduling).

This is the single hardest engineering challenge for RL at frontier scale.

---

## 4. Inference-Time Mechanics

### 4.1 Variable Thinking Budgets

User-facing API exposes a `thinking_budget` or similar:
```
{
  "model": "claude-4-sonnet-thinking",
  "messages": [...],
  "thinking": {"budget_tokens": 8192}
}
```

Engine must:
- Generate up to `budget_tokens` of reasoning.
- Stop reasoning early if model emits a "stop thinking" token.
- Then begin user-visible answer.

Tracking these as **separate token counts** for billing matters: reasoning tokens are typically priced like output tokens.

### 4.2 Hidden vs Visible CoT

- **Hidden** (o-series, Claude with summary): model generates reasoning, only final answer streamed. May expose a summary.
- **Visible** (R1, Gemini-2.5, Qwen-3 thinking): full reasoning streamed. User can interrupt.
- **Toggleable** (Qwen-3, Claude-4): per-request switch.

Visible CoT changes the streaming UX: "thinking…" indicator vs raw token stream. Many UIs distinguish the two phases.

### 4.3 Stop-Thinking Tokens

A learned token (often a special `</think>` or similar) that signals "I'm done reasoning, answer now." Engine watches for it; on emission, switches stream phase from reasoning to answer.

### 4.4 KV Cache Pressure

A 50K-token CoT generation produces 50K rows of KV cache for that sequence. For Llama-3-70B GQA, that's 50K × 320 KB = 16 GB per concurrent reasoning sequence. Realistic batch sizes for reasoning workloads:

- 8 H100 node, Llama-3-70B FP8, KV at 128K capacity: ~30 concurrent reasoning sequences.
- Same node, MLA model (DeepSeek-V3 distill): ~300 concurrent sequences.

MLA-class models are dramatically better suited to reasoning workloads.

### 4.5 Speculative Decoding for Reasoning

Reasoning workloads benefit even more than chat:
- Long outputs amortize spec-decode overhead.
- Reasoning text is often repetitive/structured (math notation, code patterns) → high acceptance.
- MTP (DeepSeek) and EAGLE-3 work especially well.

Reported speedups: 2-4× on reasoning workloads with EAGLE-3.

### 4.6 Tool-Use During Reasoning

o1, R1, Claude-4, and Qwen-3 all support tool calls inside the reasoning loop:

```
User: solve <hard problem>
[reasoning starts]
... let me try ...
<tool>python: compute...</tool>
[tool result returned]
... that gives ...
<tool>search: ...</tool>
[tool result]
... so the answer is ...
[reasoning ends]
[answer emitted]
```

Engine must:
- Parse structured tool calls in the streaming output.
- Pause generation, dispatch tool, inject result, resume.
- Multiple tool calls per request typical.

This is **agent infrastructure**, integrated with reasoning serving. Frameworks like LangGraph, LlamaIndex agents, OpenAI Responses API expose this directly.

---

## 5. Capacity Planning for Reasoning

### 5.1 Rough Numbers

| Workload | Output / req | Reasoning / req | Total tokens / req |
|----------|--------------|------------------|---------------------|
| Chat     | 200          | 0                | 200                 |
| RAG      | 500          | 0                | 500                 |
| Reasoning (light)  | 500 | 5K            | 5.5K                |
| Reasoning (heavy)  | 1K | 30K            | 31K                 |
| Deep Research      | 5K | 100K+          | 100K+               |

A reasoning request consumes 10-100× the GPU-time of a chat request. Pricing reflects this.

### 5.2 Sizing Reasoning Pools

If λ reasoning RPS at average 30K total tokens, and per-H100 throughput is 1500 tok/s for the model:
```
GPUs needed = λ · 30K / 1500 = λ · 20
```
For λ = 50 reasoning RPS → 1000 H100s. Big.

Mitigations:
- MLA-class model → 5× more concurrent → ~200 H100s.
- Spec decoding 3× → ~70 H100s.
- Disaggregation 30% utilization gain → ~50 H100s.
- Caching reasoning patterns → marginal but real.

### 5.3 Disaggregation for Reasoning

Reasoning amplifies disaggregation's wins:
- Long decodes interfere with prefills more (longer interference window).
- Decode pool size dominates.
- KV transfer cost per request is paid once but amortized over a 30K-token decode.

Most production reasoning deployments disaggregate by default.

### 5.4 SLO Considerations

- **TTFT** matters less when user expects to wait minutes anyway.
- **Total latency** is the relevant SLO: typically 30s-5min budgets.
- **Streaming reasoning** ("thinking…") gives the user feedback during long waits.
- **Cancellation** (user stops a long reasoning trace) must be supported — the engine should free KV immediately.

---

## 6. Systems Stack for Reasoning RL

### 6.1 Open-Source Options

| Stack | Lineage | Notes |
|-------|---------|-------|
| veRL  | ByteDance | Disaggregates inference (vLLM/SGLang) from training; weight-sync utilities; supports GRPO, PPO, DPO. Strong at large scale. |
| OpenRLHF | DeepSpeed-RLHF descendant | PPO + GRPO, supports multi-node, vLLM rollout |
| NeMo-Aligner | NVIDIA | TRT-LLM rollout, Megatron training; production-grade |
| TRL (Hugging Face) | HF | DPO/PPO, integrated with Accelerate; single-node mostly |
| OREAL (Skywork) | Skywork | RL framework focused on long-CoT |
| LMM-R1 | open community | R1-style recipes, multi-modal extensions |

### 6.2 Architectural Patterns

```
                ┌─────────────────────────┐
                │  Training cluster       │
                │  Megatron / FSDP        │
                │  GRPO / PPO update step │
                └─────────┬───────────────┘
                          │ weights pushed every N steps
                          ▼
                ┌─────────────────────────┐
                │  Inference cluster      │
                │  vLLM / SGLang / TRT-LLM│
                │  K samples per prompt   │
                │  long generations       │
                └─────────┬───────────────┘
                          │ trajectories returned
                          ▼
                ┌─────────────────────────┐
                │  Replay buffer + reward │
                │  computation            │
                └─────────────────────────┘
                          │
                          ▼
                  back to training
```

Sharing GPUs vs separate GPU pools:
- **Separate**: simpler, less efficient (idle GPUs at handoff).
- **Time-shared**: training pauses, inference takes over GPUs. Hard to schedule.
- **Co-located on different GPUs in same job**: typical at frontier scale.

### 6.3 Common Bottlenecks

- **Sample throughput** dominates. The training step is fast; generating samples for the next step is slow.
- **Reward computation latency** (especially LLM-judge based). Mitigated by parallel reward models or rule-based rewards.
- **Long-tail samples**: 99% of trajectories are 5K tokens; 1% are 100K. The 1% bottleneck the batch.
- **Weight sync time**: pushing 671B weights between training and inference clusters takes minutes.

---

## 7. Reasoning Patterns You Should Know

### 7.1 Self-Consistency

Sample N independent CoTs, take majority vote of final answers. Inference cost N×; quality usually +5-15% on reasoning benchmarks. Cheap to implement at the API layer.

### 7.2 Best-of-N + Verifier

Sample N CoTs, score each with a learned reward model, take the highest-scoring. Better than majority voting when the verifier is good. Cost is N× inference + N reward evaluations.

### 7.3 Tree-of-Thoughts / MCTS

Branch at decision points; explore the tree; expand promising branches. Used in research benchmarks; production deployment rare due to control complexity. Some agent frameworks (LangGraph, AutoGen) approximate this.

### 7.4 Iterative Refinement / Reflection

Solve → critique → refine. The model is given its own answer back and asked to find errors. ~5-10% improvement on harder benchmarks; cost 2-3×.

### 7.5 Tool-Augmented Reasoning

Heavily used by o3, Claude-4. Reasoning calls Python interpreter, search, file read. Each tool call is an inference pause. Drastically improves math, factual, and code tasks.

---

## 8. Frontier Research Topics (as of early 2026)

- **Process reward models for general reasoning**: hard to scale beyond math/code.
- **Reasoning + retrieval fusion**: model decides when to retrieve mid-CoT.
- **Skills / library learning**: model accumulates and reuses learned subroutines.
- **Multi-agent reasoning**: multiple LLMs critique and improve each other.
- **Diffusion-LM reasoning**: alternative to autoregressive; potential for parallel decoding.
- **Test-time training**: brief gradient updates at inference time on the current task.
- **Verifier-guided beam search at inference**: structured exploration with a learned verifier.

You should be able to discuss any of these at a conceptual level.

---

## 9. Common Pitfalls

- **Underestimating output length distributions** → KV pool exhaustion under reasoning load.
- **Streaming UX confusion** between thinking-tokens and answer-tokens (don't show thinking as final answer).
- **Cancellation race conditions**: client disconnects mid-reasoning, engine must free KV promptly.
- **Tool call parsing mid-stream** — easy to write fragile parsers; use grammar-constrained generation.
- **Weight sync bottleneck** in RL: failing to overlap with rollout halves throughput.
- **Sampling temperature for reasoning**: too low → mode collapse, too high → meaningless. R1 uses ~0.6 with top-p 0.95.

---

## 10. Common Interview Questions

**Q: What is "test-time compute scaling"?**
A: The empirical finding that, for a fixed model, accuracy on reasoning tasks improves roughly logarithmically with tokens generated at inference time (longer CoT, more samples, etc.). Complementary to train-time scaling laws. Underlies o1/R1-style reasoning models.

**Q: How does R1 train its reasoning behavior?**
A: Three stages: cold-start SFT on a few thousand CoT examples → GRPO RL with rule-based rewards (math correctness via parser, code correctness via tests) and KL anchor to the cold-start model → final SFT and safety on curated outputs from the RL'd model. The middle GRPO stage is where reasoning patterns develop.

**Q: What is GRPO and why is it suited to reasoning RL?**
A: Group Relative Policy Optimization. Per prompt, sample K completions, normalize their rewards within the group as advantages, do PPO-style updates. No critic needed → 2× memory savings vs PPO at frontier scale. Scales to long generations and large models.

**Q: Why is reasoning workload so much harder to serve than chat?**
A: Outputs 10-100× longer → KV pool pressure, longer total latency, more decode steps amortizing fixed overhead. Decoders run for minutes per request, increasing prefill–decode interference unless disaggregated. MLA / spec decoding / disaggregation all compound.

**Q: How is the "thinking budget" exposed to users?**
A: Per-request parameter (e.g., `thinking.budget_tokens`). Engine generates reasoning up to the budget or until a stop-thinking token, then answers. Reasoning tokens billed separately at the same rate as output (typically).

**Q: How does spec decoding interact with reasoning?**
A: Reasoning is even more amenable to spec decoding because long structured outputs (math, code) have high token-level predictability. EAGLE-3 / MTP report 2-4× decode speedup on reasoning workloads vs ~1.5-2× on chat.

**Q: How does an inference-engine-inside-RL training loop work?**
A: Dedicated inference cluster (vLLM/SGLang) generates K samples per prompt for the policy. Trajectories returned to a replay buffer; reward computed; training cluster does the policy update. Weights are pushed from training to inference every N updates. The handoff and weight sync are major engineering challenges.

**Q: Why does R1-Zero exhibit reasoning behavior emergent from pure RL?**
A: With strong enough base capability and a clean reward signal, RL can discover that producing intermediate reasoning improves outcome reward — a form of policy improvement through credit assignment over long sequences. The phenomenon is robust enough to emerge without any SFT seeding.

**Q: What's the cost difference between a chat response and a reasoning response?**
A: Roughly 10-100× on output tokens alone. Plus the reasoning is typically longer-context (so KV cache reads more bytes per step). Provider pricing reflects this — o3 reasoning tokens cost the same as output tokens, and the typical request consumes thousands.

**Q: How do you implement "stop thinking" in the engine?**
A: Train a special token that the model emits when reasoning is complete. The engine runs a per-step token check; on emission, transitions internal state from reasoning-mode to answer-mode (e.g., changes streaming endpoint or flushes "thinking" buffer). Handled in the sampler post-step.

**Q: What's the difference between hidden and visible CoT?**
A: Hidden (o-series): user sees only the final answer; reasoning is logged but not exposed (sometimes a summary is). Visible (R1, Gemini, Qwen-3 thinking): user can see the full reasoning trace stream. UX, billing, and safety review work all differ.

**Q: What's distillation from a reasoning model?**
A: Use the large RL'd model (e.g., R1-671B) to generate CoT outputs on a curated prompt set; SFT-train smaller models on those traces. Smaller models inherit reasoning patterns surprisingly well. R1-distilled 7B-70B variants are widely used.

**Q: What's the "long-tail problem" in RL rollouts?**
A: Most sampled trajectories are short, but a few are very long (e.g., model loops or very hard problems). The longest in a batch determines step latency. Mitigations: timeout per sample, long-tail batching (separate pool for long ones), early termination heuristics.

---

## 11. Further Reading

- DeepSeek-R1 paper (Jan 2025).
- DeepSeek-Math paper (origin of GRPO).
- OpenAI o1 system card; o3 announcement materials.
- Anthropic "Extended Thinking" documentation.
- "Let's Verify Step by Step" (Lightman et al., 2023) — process rewards.
- "Star: Self-taught Reasoner" (Zelikman et al., 2022).
- "Self-Refine" (Madaan et al., 2023).
- veRL, OpenRLHF, NeMo-Aligner repositories and papers.
- "Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Model Parameters" (Snell et al., 2024).

---

**Next:** [Modern_Post_Training](Modern_Post_Training.md), [Modern_MoE](Modern_MoE.md).
**See also:** [Frontier_Models_2025_2026](Frontier_Models_2025_2026.md), [Disaggregated_Serving_2025](Disaggregated_Serving_2025.md), [Speculative_Decoding](Speculative_Decoding.md).

# Modern Post-Training — DPO, GRPO, RLHF at Scale

The 2024–2026 post-training stack: SFT, DPO/IPO/KTO/SimPO, RLHF (PPO), GRPO, online RL with inference-engine-inside-the-loop, reward modeling (process & outcome), distillation, and the open-source frameworks (TRL, OpenRLHF, veRL, NeMo-Aligner, OREAL). Heavy on systems implications.

**Prerequisites**: [Distributed_Training](Distributed_Training.md), [Reasoning_Models](../Part_5_Algorithms_and_Quantization/Reasoning_Models.md), [Inference_Frameworks](../Part_6_Inference_Engines_and_Serving/Inference_Frameworks.md).

---

## 1. The Post-Training Pipeline

Pretraining produces a base model. Post-training shapes it into a usable product through several stages:

```
[Pretrained base]
       │
       ▼
[Supervised Fine-Tuning (SFT)]    ← teach format, instruction following
       │
       ▼
[Preference learning]              ← align with human / model preferences
   - DPO / IPO / KTO / SimPO       ← offline, no RL loop
   - or PPO / GRPO                 ← online RL with sampling
       │
       ▼
[Reasoning RL]                     ← (for o-series / R1-style)
   - GRPO with rule rewards
   - long-CoT generation
       │
       ▼
[Final SFT / Safety]               ← harmonize outputs, refusal training
       │
       ▼
[Optionally: Distillation]         ← compress to smaller model
       │
       ▼
[Production model]
```

Different products take different paths through this. Chat models often stop at DPO. Reasoning models go through full GRPO. Safety-critical products have multi-pass safety SFT.

---

## 2. SFT — Still the Foundation

Supervised Fine-Tuning on (instruction, response) pairs is the universal first step.

### 2.1 Data

- **Curated**: synthetic via larger model + filtering (Distillation). Most production data.
- **Human-written**: expensive but high-signal. Used for tone, persona, edge cases.
- **Mixed**: synthetic bulk + human refinement.

### 2.2 Training Recipe

- Standard cross-entropy with masked loss on the response tokens only.
- Multi-turn: full conversation in one sample, mask user turns from loss.
- Packed sequences (concat with EOS, attention mask) for efficiency.
- LR: 1-5e-5, much lower than pretraining. Cosine decay over 1-3 epochs.
- Batch: 1-4M tokens global. Smaller than pretraining.

### 2.3 Systems

SFT at scale is much cheaper than pretraining:
- Llama-3.3-70B SFT: a few thousand GPU-hours typically.
- Standard FSDP / Megatron-LM with selective activation recomputation.
- Multi-LoRA SFT enables many fine-tunes on the same base.

---

## 3. DPO and Friends (Offline Preference Learning)

### 3.1 Direct Preference Optimization (DPO)

Rafailov et al., 2023. Skip the explicit reward model. Given pairwise preferences `(prompt, chosen, rejected)`, optimize directly:

```
L_DPO = -log σ(β · (log π_θ(chosen|p) - log π_ref(chosen|p)
                  - log π_θ(rejected|p) + log π_ref(rejected|p)))
```

Where π_ref is the SFT model, π_θ is being trained, β is a temperature (~0.1).

Implicit reward = `β · log(π_θ / π_ref)`. The model learns to assign higher probability to chosen vs rejected — equivalent to PPO with a Bradley-Terry reward but without the RL machinery.

### 3.2 Variants

- **IPO** (Identity Preference Optimization): replaces the log-sigmoid with a squared-loss form, more robust to over-optimization.
- **KTO** (Kahneman-Tversky Optimization): uses unpaired data (just (prompt, response, good/bad) flag). Easier to collect data.
- **SimPO** (Simple Preference Optimization): drops the reference model from the loss; uses average log-prob length-normalized. Slightly worse quality, much cheaper memory.
- **ORPO** (Odds Ratio Preference Optimization): combines SFT and preference loss in one objective. Single-stage instead of SFT-then-DPO.
- **NCA, BCO, RPO** etc.: various refinements with different trade-offs. The space evolves quickly.

### 3.3 Systems Cost

DPO is much cheaper than RL because:
- **Offline**: no online sampling needed.
- **Two model copies**: training (θ) and reference (ref). Ref is frozen — can be a sharded inference instance, or batched into the same forward as θ.
- **No critic**, no reward model.

For 70B with FSDP: ~1.5-2× SFT memory cost (the extra ref copy). Wall-clock similar to SFT.

### 3.4 Failure Modes

- **Reward hacking via length**: models exploit length to satisfy preferences. Length-normalized losses (SimPO) help.
- **Distribution drift**: π_θ moves far from π_ref → unreliable reward signal. β controls this; KL clipping helps.
- **Preference data noise**: noisy labels propagate; data quality matters.

---

## 4. PPO — RLHF Classic

The OG online RL for LLMs (Ouyang et al., InstructGPT 2022).

### 4.1 Components

- **Policy** π_θ: the model being trained.
- **Reference** π_ref: frozen SFT model, for KL penalty.
- **Reward model** R(prompt, response): scores completed responses.
- **Critic** V(prompt, prefix): estimates expected future reward; learned alongside policy.

Four models in memory simultaneously. At 70B scale, this is 4 × 140GB = 560 GB just for FP16 weights — needs FSDP/ZeRO-3 + careful memory management.

### 4.2 Training Loop

```
For each step:
  1. Sample N responses from π_θ given prompts (online).
  2. Compute rewards R, values V on each.
  3. Compute advantages (GAE or similar).
  4. Compute PPO loss with KL penalty against π_ref.
  5. Update θ and V parameters.
```

Step 1 is the bottleneck. Sampling N completions of length 1K-100K tokens consumes most wall-clock. Modern stacks dedicate inference GPUs for this.

### 4.3 Stability Issues

PPO is notoriously fiddly:
- KL coefficient tuning.
- Reward model overfitting / hacking.
- Critic convergence.
- Entropy collapse (mode collapse).

Frontier labs spend months stabilizing PPO runs. Most open-source teams now prefer DPO or GRPO for simplicity.

---

## 5. GRPO — Group Relative Policy Optimization

DeepSeek's algorithm (DeepSeekMath, 2024; canonized in R1, 2025). Now the dominant choice for reasoning RL.

### 5.1 Algorithm

```
For each prompt:
  1. Sample K completions (e.g., K=16) from π_θ.
  2. Compute reward R_i for each.
  3. Group-normalize: A_i = (R_i - mean(R)) / std(R).
  4. PPO-style clipped policy gradient on each (token, A_i) pair.
  5. KL penalty against π_ref.
```

No critic — advantages come from within-group normalization. This is the key simplification.

### 5.2 Memory Savings

PPO requires 4 model copies; GRPO requires only 2 (policy + reference). At 70B: 280 GB → 560 GB savings, allowing significantly larger batches or models.

### 5.3 Why It Works for Reasoning

- Many reasoning prompts have high reward variance (some attempts succeed, most fail). Group normalization captures this signal cleanly.
- Long generations make critic learning hard; eliminating the critic dodges the issue.
- Rule-based rewards (math correctness, code passes) plug in naturally — no learned reward model needed.

### 5.4 Variants

- **DAPO** (alibaba, 2025): refines GRPO with clip-higher and dynamic-sampling tricks for better long-CoT training.
- **REINFORCE++** (2024): adds variance-reduction baselines.
- **VinePPO** etc.: hybrid approaches.

### 5.5 Hyperparameters

- K (group size): 8-64; bigger group → lower variance, more compute per step.
- KL β: 0.001-0.04; tighter to ref means slower learning, looser means more drift.
- Clip ε: 0.1-0.3.
- LR: 1e-6 to 3e-6 typical.

---

## 6. Reward Modeling

### 6.1 Outcome Reward Models (ORM)

Score the final answer. Train as a classifier or regression on (prompt, response, label) triples.

For math/code, replace ORM with rule-based:
- **Math**: parse final number; compare to ground truth.
- **Code**: run the test suite; pass/fail.
- **Function calls**: validate against schema.

Rule-based avoids reward hacking — the model can't fool a parser.

### 6.2 Process Reward Models (PRM)

Score each step of the reasoning trace. Train on annotated traces (PRM800K dataset, ~800K human-annotated steps for math).

PRMs give denser feedback but require expensive annotation. Used heavily in math reasoning (Lightman et al. 2023, "Let's Verify Step by Step"), less for general reasoning.

### 6.3 LLM-as-Judge Rewards

Use a strong model (e.g., GPT-4 / Claude / a fine-tuned judge model) to score completions. Cheap to set up, but biases (verbosity, sycophancy) leak in. Fine for early-stage RL; less reliable for final tuning.

### 6.4 Constitutional AI / RLAIF (Anthropic)

The reward signal comes from an AI critic following a written "constitution." Cheaper than human labels at scale. Used for some safety training.

### 6.5 Reward Hacking — The Permanent Risk

The model finds shortcuts that maximize reward without satisfying intent:
- Verbosity inflation.
- Refusal to answer hard questions.
- Format-template gaming.
- Adversarial CoTs that fool the parser.

Mitigations: KL constraint to ref, regular eval against frozen baselines, mixing in supervised data, careful reward design.

---

## 7. Online RL Infrastructure

The hardest engineering challenge in modern post-training.

### 7.1 The Inference-Inside-Training Pattern

```
                  ┌──────────────────────┐
                  │  Training cluster    │
                  │  (FSDP/Megatron)     │
                  │  Loss + grad + step  │
                  └──────────┬───────────┘
                             │ weight sync every N updates
                             ▼
                  ┌──────────────────────┐
                  │  Inference cluster   │
                  │  (vLLM / SGLang)     │
                  │  K samples / prompt  │
                  └──────────┬───────────┘
                             │ trajectories
                             ▼
                  ┌──────────────────────┐
                  │  Replay buffer +     │
                  │  reward computation  │
                  └──────────┬───────────┘
                             │
                             ▼
                       back to training
```

The two clusters can be:
- **Co-located on different GPUs**: typical at frontier scale. Training uses some GPUs, inference uses others, statically split.
- **Time-shared on same GPUs**: training pauses, inference takes over, swap weights. Hard to schedule but better utilization.
- **Fully separate**: training and inference on entirely different machines. Simpler, less efficient.

### 7.2 Weight Synchronization

When training updates θ, inference cluster needs the new weights. Options:

- **NCCL broadcast**: simplest. Stop everything, broadcast full weights, resume. Slow at frontier scale (671B in FP8 = 670 GB; broadcast across NDR IB ≈ minutes).
- **Sharded incremental**: only the changed shards (after FSDP sharding) sync. Faster, complex bookkeeping.
- **CPU-resident shared weights**: training writes to host RAM; inference reads from same buffer. Avoids GPU-GPU broadcast but adds PCIe overhead.
- **NIXL-based**: leverage GPU-direct transports.

Sync cadence: every N policy updates, where N is tuned for latency-vs-staleness. Typical: every 1-10 steps.

### 7.3 Sample Throughput Matters

For GRPO with K=16 samples per prompt and average 10K output tokens:
- 1 prompt = 160K tokens generated.
- For 1000 prompts/step: 160M tokens.
- At 100K tok/s on a 64-GPU inference cluster: 1600 seconds per step = 27 min/step.

This dominates wall-clock. Optimizations:
- Larger inference cluster (more parallel sampling).
- Spec decoding inside RL (2-3× sampling speedup).
- Truncating samples that are clearly diverging.
- KV cache reuse across the K samples for one prompt (the prefix is shared).

### 7.4 Long-Tail Sample Lengths

Most samples finish in 5K tokens; 1% run to 100K. The longest sample gates step latency.
- Timeout cap (e.g., 32K).
- Dedicated long-tail batch / pool.
- Async returning of finished samples while long ones continue.

---

## 8. Frameworks

| Framework | Origin | Strengths | Status |
|-----------|--------|-----------|--------|
| TRL (Hugging Face) | HF | Easy to start; SFT/DPO/PPO/GRPO support; integrated with Accelerate | Mature |
| OpenRLHF | Open community | Multi-node, vLLM rollout, Ray-based; PPO + GRPO + DPO | Production |
| veRL | ByteDance | Disaggregates training and inference cleanly; weight-sync utilities | Production-grade for large scale |
| NeMo-Aligner | NVIDIA | TRT-LLM rollout, Megatron training; integrated with NeMo | Production |
| OREAL | Skywork | Long-CoT focus | Niche |
| LMFlow | OSU | Friendly entry point | Education |
| LMM-R1 | Open community | R1-style multimodal | Niche |
| trlX (CarperAI) | CarperAI | PPO at scale | Some-what dated |

For frontier-scale GRPO: veRL or NeMo-Aligner. For research / quick iteration: OpenRLHF or TRL.

---

## 9. Distillation From Reasoning Teachers

After R1 was released, the open community demonstrated: **distill** R1's CoT outputs onto smaller dense models (1.5B, 7B, 14B, 32B) → small reasoning models.

### 9.1 Method

1. Use R1 (or your reasoning teacher) to generate long-CoT answers on a curated prompt set.
2. Filter for correctness and quality.
3. SFT a smaller base model on (prompt, R1_response) pairs.
4. Optional: brief RL phase to refine.

### 9.2 Results

- R1-Distill-Qwen-32B competitive with GPT-4o on math benchmarks.
- R1-Distill-Llama-70B beats baseline 70Bs by large margins.
- Surprising: distilled models often *exceed* what RL'd small models achieve directly.

### 9.3 Why It Works

Reasoning patterns are transferrable through tokens; the smaller model learns the shape of long-CoT reasoning even though it couldn't discover it on its own. This is the classic "knowledge transfer via outputs" pattern at very large scale.

### 9.4 Cost

- Teacher inference: huge (millions of long generations). Run on a dedicated cluster.
- Student SFT: cheap relative to pretraining; standard FSDP.
- Total: ~1% of pretraining cost.

---

## 10. Safety Post-Training

Often a final stage in production pipelines:
- Refusal SFT on carefully curated harmful-prompt → refusal pairs.
- Red-teaming generation + adversarial training.
- Constitutional rules + RLAIF on safety axes.
- Sandboxed eval harnesses to verify safety properties.

Architecturally same systems as SFT/DPO; the work is in the data and eval.

---

## 11. Common Pitfalls

- **Mixing data distributions** between SFT and DPO: model forgets format; mixed-loss training (ORPO) or staged training fixes this.
- **Reward hacking** in PPO/GRPO that's invisible in training metrics but obvious in eval; always run held-out qual evals.
- **β (KL) too tight**: model can't move; flat training loss but no improvement.
- **β too loose**: model drifts; quality regressions everywhere.
- **Inference cluster too small**: training waits on samples; rollout becomes the bottleneck.
- **Weight sync latency** dominating: staleness hurts learning; tune cadence carefully.
- **Reference model on outdated checkpoint**: in long runs, ref vs θ diverge so much that KL becomes meaningless. Periodically refresh ref.
- **Inadequate eval frequency**: silent quality degradations during long RL runs; eval every few hours.

---

## 12. Common Interview Questions

**Q: Compare SFT, DPO, and PPO.**
A: SFT trains on (prompt, response) with cross-entropy. DPO trains on preferences (chosen, rejected) directly with a Bradley-Terry-style loss — no RL loop, no reward model. PPO is online RL with a learned reward model and critic — more powerful but ~5× more memory and engineering complexity. Pick SFT first, DPO second, PPO/GRPO if needed.

**Q: What is GRPO and why is it preferred over PPO for reasoning RL?**
A: Group Relative Policy Optimization. Sample K completions per prompt; advantages = (reward − group mean) / group std; PPO-style clipped update. Removes the critic — 2× memory savings (4 model copies → 2). Works especially well for reasoning where reward variance within a prompt is high (some samples right, most wrong). DeepSeek's R1 popularized it.

**Q: What's the inference-inside-training bottleneck?**
A: Online RL (PPO, GRPO) requires sampling K completions from the current policy at every step. For reasoning with K=16 and 10K-token avg outputs, this generates millions of tokens per step. Sampling typically dominates wall-clock unless the inference cluster is sized accordingly. Optimizations: bigger cluster, spec decoding, KV reuse across K samples, sample-timeout truncation.

**Q: How does weight synchronization work between training and inference clusters?**
A: After every N policy updates, push the new weights from training to inference. Methods: NCCL broadcast (simple, slow), sharded incremental (fast, complex), CPU-resident shared (avoids GPU broadcast), NIXL transport. Frontier-scale runs use sharded incremental over IB/NVLink with careful overlap to hide sync latency.

**Q: Walk me through running R1-style reasoning RL on a 70B model.**
A: (1) Cold-start SFT on a few thousand curated CoT examples. (2) GRPO with K=16, prompt set of math/code, rule-based rewards, β=0.04. (3) Run for 100K-1M optimization steps; sample 32K-token completions; reward via parser/test-runner. (4) Final SFT pass on curated outputs. (5) Optional: distill into smaller models. Infrastructure: training cluster (FSDP) + inference cluster (vLLM with speculative decoding). Weight sync every few steps via sharded RDMA.

**Q: What's reward hacking and how do you detect it?**
A: The model finds shortcuts that maximize reward without satisfying intent — e.g., gaming length, refusing hard questions, producing format-correct but content-wrong CoT. Detection: held-out qualitative evals run every few hours; check for distribution shifts in output length, refusal rate, format compliance; A/B vs frozen baseline; LLM-judge analyses.

**Q: Why not just always use PPO if it's more powerful?**
A: PPO requires 4 model copies, a learned reward model that can be hacked, careful KL tuning, critic stability issues, and significant engineering. DPO eliminates 3 of those (reward model, critic, online sampling); GRPO eliminates the critic. For most production tuning, DPO/GRPO match PPO quality at much lower cost.

**Q: What's distillation from a reasoning teacher?**
A: Use a frontier reasoning model (R1, o3) to generate long-CoT answers on a curated prompt set. SFT a smaller model on (prompt, teacher_response) pairs. The smaller model inherits reasoning patterns surprisingly well; often exceeds what direct RL on the small model achieves. R1-Distill-Qwen-32B matched or beat GPT-4o on math.

**Q: How do you handle the long-tail of generation lengths during RL sampling?**
A: Per-sample timeout (e.g., 32K cap); dedicated long-tail pool with separate scheduling; async return of finished samples while long ones continue; truncation if early indicators show divergence. Without these, the longest sample in a batch gates step latency.

**Q: What's the role of the reference model in DPO/GRPO?**
A: Anchors the policy via KL: the trained model shouldn't drift too far from the reference (typically the SFT-stage model). KL penalty / log-ratio. Without it, the policy can drift into reward-hacking solutions or mode collapse. β controls the KL strength.

**Q: What's a "rule-based reward" and why is it valuable?**
A: A reward computed by a deterministic checker, not a learned model — e.g., math answer correctness via parser, code correctness via test execution. Cannot be hacked the way learned reward models can. Used in R1's GRPO for math/code; key reason R1 trained without a learned reward model.

**Q: How does online RL affect the inference-engine design?**
A: Inference engine must support: (1) extreme load spikes from RL sampling phases, (2) frequent weight updates without full restart, (3) deterministic / seeded sampling for reproducibility, (4) very long generations (100K+) gracefully, (5) ragged batching across K samples per prompt. Most production engines (vLLM, SGLang) added these in 2024–2025.

---

## 13. Further Reading

- "Direct Preference Optimization" (Rafailov et al., 2023).
- "Training language models to follow instructions with human feedback" (Ouyang et al., 2022) — InstructGPT.
- "DeepSeekMath: Pushing the Limits of Mathematical Reasoning" (DeepSeek, 2024) — GRPO origin.
- "DeepSeek-R1" technical report (Jan 2025).
- "DAPO: An Open-Source LLM Reinforcement Learning System at Scale" (2025).
- "ORPO: Monolithic Preference Optimization without Reference Model" (2024).
- "SimPO: Simple Preference Optimization with a Reference-Free Reward" (2024).
- "Let's Verify Step by Step" (Lightman et al., 2023) — process reward models.
- veRL, OpenRLHF, NeMo-Aligner, TRL repositories and their papers.

---

**Next:** [Multimodal_Inference](../Part_6_Inference_Engines_and_Serving/Multimodal_Inference.md), [Cutting_Edge_Kernels](../Part_4_GPU_Kernel_Engineering/Cutting_Edge_Kernels.md).
**See also:** [Distributed_Training](Distributed_Training.md), [Reasoning_Models](../Part_5_Algorithms_and_Quantization/Reasoning_Models.md), [Inference_Frameworks](../Part_6_Inference_Engines_and_Serving/Inference_Frameworks.md).

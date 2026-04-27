# Speculative Decoding

Speculative decoding is the cheapest way to lift decode throughput by 1.5–3× without retraining or hardware changes. It works because decode is bandwidth-bound: a single forward pass moves the same weights whether you check 1 token or 8. This page covers the algorithm, the math of acceptance, the major variants (vanilla, Medusa, EAGLE, Lookahead, multi-token), and the engineering integration with continuous batching.

**Prerequisites**: [Batching_and_Scheduling](Batching_and_Scheduling.md), [KV_Cache](KV_Cache.md).

---

## 1. Why It Works

Decode of a 70B model is BW-bound: 140 GB / 3.35 TB/s ≈ 42 ms per step regardless of how many tokens you score in that step. If we can produce a guess for the next K tokens cheaply, we can verify them in **one forward pass** that costs essentially the same as a normal step. Every accepted token is "free."

The key constraint: the verifier output distribution must equal the target model's output distribution. The classic Leviathan/Chen 2023 algorithm achieves exact equivalence via rejection sampling.

---

## 2. Vanilla Speculative Decoding (Leviathan/Chen)

### 2.1 Algorithm

Let `q` = draft model's distribution at step t, `p` = target model's distribution.

```
draft K tokens autoregressively from q:
   x_1 ~ q(·| context)
   x_2 ~ q(·| context, x_1)
   ...
   x_K ~ q(·| context, x_1..x_{K-1})

verifier (target) forward pass on (context, x_1, ..., x_K) → returns p_1, ..., p_K, p_{K+1}

for i in 1..K:
   r ~ Uniform(0, 1)
   if r < min(1, p_i(x_i) / q_i(x_i)):
       accept x_i
   else:
       break  # reject, sample correction
```

If we rejected at position i, sample a fresh token from the **adjusted distribution** `(p_i - q_i)_+ / sum`. If all K accepted, sample from `p_{K+1}`. Either way, we always emit at least one token per verifier pass and at most K+1 tokens.

### 2.2 Correctness

Theorem (Leviathan): the resulting tokens are distributed exactly as if drawn from `p`. Sketch: rejection sampling on the conditional `p(x | accept) ∝ q(x) · min(1, p/q)`. The renormalized residual `(p - q)_+` covers the shortfall when q is rejected.

### 2.3 Acceptance Rate Math

Let `α_i = E_{x ~ q}[ min(1, p(x)/q(x)) ]` — single-step acceptance probability. The expected accepted prefix length is:
```
E[len] = sum_{i=0..K} (product_{j=1..i} α_j)
```
For nearly identical distributions (α ≈ 0.9, K=4): `E[len] ≈ 1 + 0.9 + 0.81 + 0.73 + 0.66 ≈ 4.1` tokens per verifier pass.

Speedup vs vanilla decode (ignoring draft cost): roughly `E[len]` times.

### 2.4 Cost Model

```
cost_per_emitted_token ≈ (T_verifier + K · T_draft) / E[len]
```
For `T_verifier = 40 ms`, `T_draft = 5 ms`, `K = 5`, `E[len] = 4`:
```
≈ (40 + 25) / 4 = 16.25 ms / token  vs 40 ms baseline → 2.5× speedup
```

The draft must be **much cheaper** than the target. Rule of thumb: `T_draft / T_target < 1/K`.

---

## 3. Choosing the Draft Model

| Draft               | Acceptance α | T_draft | Notes                                  |
|---------------------|--------------|---------|----------------------------------------|
| Smaller of same family (e.g. Llama-1B drafting 70B) | 0.6–0.8 | low | Easy; standard baseline |
| Self-speculation (Medusa, EAGLE)                    | 0.7–0.9 | very low | Same model w/ extra heads; integrated |
| n-gram / lookup (PLD, Lookahead)                    | 0.3–0.7 | nearly 0 | Heuristic; great for repetitive text |
| Distillation-trained tiny draft                     | 0.7–0.9 | low | Best quality; needs training pipeline |

For chat / code / repetitive output, simple n-gram lookup wins surprisingly often.

---

## 4. Medusa: Self-Speculation Heads

Medusa adds K parallel "Medusa heads" to the target model that predict tokens 1, 2, …, K-ahead in one forward pass. The base model emits a candidate tree; each path is verified.

### 4.1 Tree Verification

Heads produce top-k candidates each. Pruning yields a tree of candidates. The verifier scores all tree nodes in **one** forward pass using a tree-attention mask (each candidate attends only to its ancestors).

### 4.2 Pros / Cons

Pros:
- No separate draft model.
- Heads are tiny (extra ~5% params).
- Very high token output rate when tree paths align.

Cons:
- Distributional drift: heads are not the same as target → acceptance can be lower than ideal rejection sampling.
- Doesn't preserve target distribution exactly without extra work.
- Requires fine-tuning the heads.

---

## 5. EAGLE / EAGLE-2 / EAGLE-3

EAGLE drafts using the **target model's own hidden state** plus a small autoregressive head, predicting next embeddings rather than next tokens. The next embedding is decoded into a token, fed back, etc.

Why it's effective: drafting from the target's hidden representation correlates strongly with the target's output distribution → high acceptance (0.85–0.95).

EAGLE-2 adds a dynamic tree (paths grow based on confidence). EAGLE-3 fuses multiple layers' hidden states.

Reported speedups: 3–4× on chat benchmarks. Industry standard for self-speculation in 2025–2026.

---

## 6. Lookahead Decoding

No draft model; uses the target model itself in a clever way. The model generates K candidate "n-gram completions" by looking at past output. Verifies them in parallel like spec decode. Works extremely well on code (lots of repetition).

---

## 7. Multi-Token Prediction (DeepSeek-V3)

DeepSeek-V3 trains the model with a multi-token prediction loss head — the model natively predicts multiple future tokens per step, then verifies them. Treated as part of the model architecture rather than a serving-layer trick. Reported 1.8× decode throughput improvement.

---

## 8. Engineering Integration

### 8.1 Inside Continuous Batching

The "step" function changes:
```
1. Each in-flight sequence has a draft K-token candidate (and tree shape if Medusa/EAGLE).
2. Build a packed batch of (context + candidate tokens) per sequence.
3. Run target forward — verifier kernel returns logits for every candidate position.
4. Run rejection-sampling kernel per sequence; emit accepted tokens.
5. Update KV cache to reflect accepted prefix only (rollback rejected portion).
```
KV update is the subtle part: we wrote KV for K+1 positions but only kept up to acceptance length. Need to truncate cache to that length without copying — paged cache makes this a refcount adjustment.

### 8.2 Variable Acceptance ⇒ Ragged Step

Different sequences accept different prefix lengths each step. The next step's input lengths differ. Continuous batching already supports ragged sequences (paged KV, packed attention) so this drops in naturally — but the scheduler must handle non-1 token throughput per step per sequence.

### 8.3 Tree-Attention Kernel

For Medusa/EAGLE trees, attention must respect the tree structure (each candidate attends to its ancestors only). Implementations: a tree mask passed to FlashAttention, or a tree-aware kernel like `flash-attn-tree`.

### 8.4 Compatibility With Disaggregation

Speculative decoding lives in the **decode pool**. Draft model and target both live there. No interaction with prefill. KV cache transfer remains target-only.

---

## 9. Failure Modes

- **Draft mismatch**: low acceptance (α < 0.4) → spec decode is slower than baseline because of wasted draft and verifier cost.
- **Long-tail samples**: high temperature, very flat distributions → acceptance drops. Sometimes turn off spec decode at temp > ~1.
- **Mode interactions**: structured output / grammar masks change the distribution; spec decode must constrain draft to the same mask.
- **Tree explosion**: greedy tree expansion with too many paths burns verifier compute. Cap branches.

---

## 10. When To Use It

Almost always, for chat and code:
- Add a tiny draft (Medusa or EAGLE heads, or ngram).
- Cap K at 4–6.
- Fall back to vanilla decode if α drops below threshold or temp > 1.

Less benefit:
- Very high temperature sampling (diversity workloads).
- Already throughput-saturated on tensor cores (rare in decode).
- Latency-critical with extremely strict TTFT (the verifier batch shape changes can add jitter).

---

## 11. Common Interview Questions

**Q: Why is speculative decoding effectively free per accepted token?**
A: Decode is HBM-bandwidth-bound. Reading the model weights once costs the same whether you score 1 or K tokens. So K verified tokens take roughly the same time as 1 baseline token. Each accepted token amortizes the cost.

**Q: How does rejection sampling preserve the target distribution?**
A: For each drafted token, accept with probability `min(1, p/q)`. If rejected, sample the residual `(p − q)_+ / Z`. The mixture is exactly p (proof: condition on accept/reject and combine).

**Q: What is the acceptance rate and how does it determine speedup?**
A: α = expected probability that a single drafted token is accepted. Expected accepted prefix length is roughly `(1 - α^{K+1}) / (1 - α)`. Speedup ≈ E[len] · T_target / (T_target + K·T_draft). High α and cheap draft = big speedup.

**Q: What's the difference between Medusa and EAGLE?**
A: Medusa adds K parallel heads predicting future tokens directly from the target's last hidden state. EAGLE uses the target's hidden state to drive a small autoregressive head that predicts future embeddings, decoded back into tokens. EAGLE achieves higher acceptance (0.85–0.95 vs 0.7–0.85) at slightly more draft cost.

**Q: Why might speculative decoding hurt throughput?**
A: If draft is too expensive or acceptance is too low, the wasted compute from rejected tokens plus the draft cost exceeds the gain. Need T_draft / T_target < ~1/K and α > ~0.5 for a positive ROI.

**Q: How does KV cache handle rollback when tokens are rejected?**
A: We write KV for all K+1 positions during the verifier pass, but only the accepted prefix is kept. With paged KV, we just update the per-sequence "valid length" / partial-block slot. Future steps overwrite the rejected slots.

**Q: How does spec decode interact with continuous batching?**
A: Each sequence in the batch produces a variable number of accepted tokens per step. Paged KV and packed attention support ragged sequences. The scheduler tracks per-sequence progress and admits new requests as before.

**Q: What about constrained decoding (JSON/grammar)?**
A: The grammar mask must be applied to BOTH the draft and the verifier. Otherwise the draft proposes invalid tokens that the verifier rejects, killing acceptance. Some implementations bypass this by drafting only from the masked distribution.

**Q: Tree-based vs sequential drafting — what's the tradeoff?**
A: Sequential: K tokens, one shot. Tree: branches at each position, more candidates, higher chance one path is right. Tree pays more verifier compute (more positions in batch). Net win when α is moderate; loss when α is already very high.

**Q: How do you measure spec-decode quality in production?**
A: Track `mean_accepted_tokens_per_step` and `verifier_step_latency`. Effective TPOT = T_step / accepted. Watch for drift (e.g. when temperature changes, or new fine-tunes shift distributions) and disable spec decode if effective TPOT exceeds baseline.

**Q: Can spec decode help prefill?**
A: No — prefill processes many tokens per step already (compute-bound). Speculation gains nothing.

**Q: Why is n-gram drafting often surprisingly good?**
A: Code and repetitive text have very predictable next-token distributions where the recent context is almost the answer. A simple n-gram match gets 30–50% acceptance with essentially zero draft cost.

**Q: What's "multi-token prediction" in DeepSeek-V3?**
A: The model is trained with extra heads that predict tokens t+1, t+2, … at once. Inference uses these heads as a built-in spec mechanism — no separate draft, high acceptance because heads are co-trained.

---

## 12. Further Reading

- Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (2023).
- Chen et al., "Accelerating Large Language Model Decoding with Speculative Sampling" (2023).
- Cai et al., "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads" (2024).
- Li et al., "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty" (2024); EAGLE-2 / EAGLE-3.
- Fu et al., "Lookahead Decoding" (2024).
- DeepSeek-V3 technical report — multi-token prediction.

---

**Next:** [Quantization](Quantization.md).
**See also:** [Batching_and_Scheduling](Batching_and_Scheduling.md), [Inference_Frameworks](Inference_Frameworks.md), [vLLM_Internals](vLLM_Internals.md).

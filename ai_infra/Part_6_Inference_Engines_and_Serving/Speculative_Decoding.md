# Speculative Decoding

Speculative decoding is the cheapest way to lift decode throughput by 1.5--3$\times$ without retraining or hardware changes. It works because decode is bandwidth-bound: a single forward pass moves the same weights whether you check 1 token or 8. This page covers the algorithm, the math of acceptance, the major variants (vanilla, Medusa, EAGLE, Lookahead, multi-token), and the engineering integration with continuous batching.

**Prerequisites**: [[Batching_and_Scheduling]], [[KV_Cache]].

---

## 1. Why It Works

Decode of a 70B model is BW-bound: $140\;\text{GB} \;/\; 3.35\;\text{TB/s} \approx 42\;\text{ms}$ per step regardless of how many tokens you score in that step. If we can produce a guess for the next $K$ tokens cheaply, we can verify them in **one forward pass** that costs essentially the same as a normal step. Every accepted token is "free."

The key constraint: the verifier output distribution must equal the target model's output distribution. The classic Leviathan/Chen 2023 algorithm achieves exact equivalence via rejection sampling.

**Why bandwidth, not compute, is the bottleneck.** During decode each step processes a single token (batch size 1 on the sequence dimension). The arithmetic intensity is roughly:

$$\text{AI} = \frac{2 \cdot P}{2 \cdot P} = 1 \;\text{FLOP/byte}$$

where $P$ is the parameter count and each parameter is read once (2 bytes in FP16) to perform one multiply-accumulate (2 FLOPs). The H100's roofline crossover is around 200 FLOP/byte, so decode sits deep in the memory-bound regime. Adding $K$ extra tokens to the verification batch raises FLOPs by $\sim K\times$ but does not increase the bytes read from HBM (same weights), so those extra tokens come essentially for free.

---

## 2. Vanilla Speculative Decoding (Leviathan/Chen)

### 2.1 Algorithm

Let $q$ = draft model's distribution at step $t$, $p$ = target model's distribution.

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

If we rejected at position $i$, sample a fresh token from the **adjusted distribution**:

$$x' \sim \frac{(p_i - q_i)_{+}}{\sum_v (p_i(v) - q_i(v))_{+}}$$

If all $K$ tokens are accepted, sample one bonus token from $p_{K+1}$. Either way, we always emit at least one token per verifier pass and at most $K+1$ tokens.

### 2.2 Correctness

**Theorem (Leviathan et al.):** The resulting tokens are distributed exactly as if drawn from $p$.

*Proof sketch.* For a single position, the token $x$ is drawn from $q(x)$ and accepted with probability $\min\!\bigl(1,\; p(x)/q(x)\bigr)$. The probability of outputting $x$ via the acceptance path is:

$$\Pr[\text{accept } x] = q(x) \cdot \min\!\Bigl(1,\;\frac{p(x)}{q(x)}\Bigr) = \min\bigl(q(x),\; p(x)\bigr)$$

The total acceptance probability is:

$$\alpha = \sum_x \min\bigl(q(x),\; p(x)\bigr) = 1 - \frac{1}{2}\|p - q\|_1$$

On rejection (probability $1 - \alpha$), we draw from the residual:

$$r(x) = \frac{(p(x) - q(x))_{+}}{\sum_v (p(v) - q(v))_{+}} = \frac{(p(x) - q(x))_{+}}{1 - \alpha}$$

Combining the two cases, the marginal probability of outputting $x$ is:

$$\alpha \cdot \frac{\min(q(x), p(x))}{\alpha} + (1-\alpha) \cdot \frac{(p(x)-q(x))_{+}}{1-\alpha} = \min(q(x), p(x)) + (p(x)-q(x))_{+} = p(x)$$

Hence the output distribution is exactly $p$. $\square$

### 2.3 Acceptance Rate Math

Let $\alpha_i = \mathbb{E}_{x \sim q}\bigl[\min(1,\; p(x)/q(x))\bigr]$ be the single-step acceptance probability. For a draft of length $K$, the expected number of accepted tokens (including the guaranteed bonus token on full acceptance) is:

$$\mathbb{E}[\text{len}] = \sum_{i=0}^{K} \prod_{j=1}^{i} \alpha_j$$

When all steps share a common $\alpha$, this simplifies to:

$$\mathbb{E}[\text{len}] = \frac{1 - \alpha^{K+1}}{1 - \alpha}$$

**Worked example.** For $\alpha = 0.9$, $K = 4$:

$$\mathbb{E}[\text{len}] = 1 + 0.9 + 0.81 + 0.729 + 0.6561 = \frac{1 - 0.9^5}{1 - 0.9} = \frac{1 - 0.59049}{0.1} \approx 4.10$$

tokens per verifier pass. Speedup vs vanilla decode (ignoring draft cost): roughly $\mathbb{E}[\text{len}]$ times.

**How $\alpha$ relates to distribution similarity.** Since $\alpha = 1 - \tfrac{1}{2}\|p - q\|_1$, the closer the draft model approximates the target, the higher $\alpha$. For identical distributions $\alpha = 1$ and every token is accepted.

### 2.4 Cost Model

$$\text{cost\_per\_emitted\_token} \approx \frac{T_{\text{verifier}} + K \cdot T_{\text{draft}}}{\mathbb{E}[\text{len}]}$$

**Worked example.** For $T_{\text{verifier}} = 40\;\text{ms}$, $T_{\text{draft}} = 5\;\text{ms}$, $K = 5$, $\mathbb{E}[\text{len}] = 4$:

$$\frac{40 + 5 \times 5}{4} = \frac{65}{4} = 16.25\;\text{ms/token}$$

versus $40\;\text{ms}$ baseline, yielding a $\approx 2.5\times$ speedup.

The draft must be **much cheaper** than the target. Rule of thumb: $T_{\text{draft}} / T_{\text{target}} < 1/K$.

**Break-even analysis.** Speculative decoding helps when:

$$\frac{T_{\text{verifier}} + K \cdot T_{\text{draft}}}{\mathbb{E}[\text{len}]} < T_{\text{verifier}}$$

Rearranging:

$$\mathbb{E}[\text{len}] > 1 + \frac{K \cdot T_{\text{draft}}}{T_{\text{verifier}}}$$

For $K = 5$, $T_{\text{draft}}/T_{\text{verifier}} = 0.1$: we need $\mathbb{E}[\text{len}] > 1.5$, i.e. $\alpha > 0.37$. Even moderately aligned drafts pay off.

---

## 3. Choosing the Draft Model

| Draft               | Acceptance $\alpha$ | $T_{\text{draft}}$ | Notes                                  |
|---------------------|--------------|---------|----------------------------------------|
| Smaller of same family (e.g. Llama-1B drafting 70B) | 0.6--0.8 | low | Easy; standard baseline |
| Self-speculation (Medusa, EAGLE)                    | 0.7--0.9 | very low | Same model w/ extra heads; integrated |
| n-gram / lookup (PLD, Lookahead)                    | 0.3--0.7 | nearly 0 | Heuristic; great for repetitive text |
| Distillation-trained tiny draft                     | 0.7--0.9 | low | Best quality; needs training pipeline |

For chat / code / repetitive output, simple n-gram lookup wins surprisingly often.

**How to select $K$.** Larger $K$ gives more potential accepted tokens but also more wasted draft cost on rejection. The optimal $K$ maximizes throughput:

$$K^{*} = \arg\max_K \frac{\mathbb{E}[\text{len}(K)]}{T_{\text{verifier}} + K \cdot T_{\text{draft}}}$$

In practice, $K = 3$--$6$ is the sweet spot. Profile on representative workloads to tune.

---

## 4. Medusa: Self-Speculation Heads

Medusa adds $K$ parallel "Medusa heads" to the target model that predict tokens 1, 2, ..., $K$-ahead in one forward pass. The base model emits a candidate tree; each path is verified.

### 4.1 Tree Verification

Heads produce top-$k$ candidates each. With $K$ heads and top-$k$ per head, the naive Cartesian product is $k^K$ paths. Pruning (keeping only high-confidence paths) yields a tree of candidates. The verifier scores all tree nodes in **one** forward pass using a tree-attention mask (each candidate attends only to its ancestors).

**Tree-attention mask construction.** For a tree with $T$ nodes (including the root), build a $T \times T$ boolean mask where $M[i,j] = 1$ iff node $j$ is an ancestor of node $i$ (or $i = j$). This mask is passed to the FlashAttention kernel in place of the standard causal mask. The KV cache stores entries for all $T$ positions, but only the accepted prefix is retained after verification.

### 4.2 Pros / Cons

Pros:
- No separate draft model.
- Heads are tiny (extra $\sim 5\%$ params).
- Very high token output rate when tree paths align.

Cons:
- Distributional drift: heads are not the same as target so acceptance can be lower than ideal rejection sampling.
- Doesn't preserve target distribution exactly without extra work (needs the modified rejection sampling with a residual correction).
- Requires fine-tuning the heads on representative data (typically a few thousand steps).

---

## 5. EAGLE / EAGLE-2 / EAGLE-3

EAGLE drafts using the **target model's own hidden state** plus a small autoregressive head, predicting next embeddings rather than next tokens. The next embedding is decoded into a token, fed back, etc.

Why it's effective: drafting from the target's hidden representation correlates strongly with the target's output distribution, yielding high acceptance ($0.85$--$0.95$).

**Architecture detail.** The EAGLE draft head takes the target model's last-layer hidden state $h_t \in \mathbb{R}^d$ and feeds it through a lightweight transformer block (typically 1--2 layers, same hidden dim $d$) that autoregressively predicts $\hat{h}_{t+1}, \hat{h}_{t+2}, \ldots$ These predicted hidden states are projected to vocabulary logits via the target model's existing LM head (shared, no extra parameters). This sharing is key: the draft and target use the same unembedding matrix, so the draft's token probabilities are naturally calibrated to the target's.

EAGLE-2 adds a **dynamic tree** (paths grow based on per-token confidence scores; low-confidence branches are pruned early, and high-confidence branches are expanded). EAGLE-3 fuses multiple layers' hidden states via a cross-layer attention mechanism to produce richer draft representations.

Reported speedups: 3--4$\times$ on chat benchmarks. Industry standard for self-speculation in 2025--2026.

---

## 6. Lookahead Decoding

No draft model; uses the target model itself in a clever way. The model generates $K$ candidate "n-gram completions" by looking at past output. Verifies them in parallel like spec decode. Works extremely well on code (lots of repetition).

**How it works in detail.** Maintain a pool of $n$-gram candidates extracted from the generated text so far. At each step, look up the most recent $n$-gram in the pool; if a continuation exists, propose it as the draft. Multiple $n$-gram matches can be batched into a tree of candidates. Because the target model is used for verification (not a separate draft), the output is exact. The "draft cost" is essentially zero since it is just a hash-table lookup.

---

## 7. Multi-Token Prediction (DeepSeek-V3)

DeepSeek-V3 trains the model with a multi-token prediction loss head -- the model natively predicts multiple future tokens per step, then verifies them. Treated as part of the model architecture rather than a serving-layer trick. Reported $1.8\times$ decode throughput improvement.

**Training objective.** In addition to the standard next-token loss $\mathcal{L}_1 = -\log p(x_{t+1} \mid x_{\leq t})$, the model is trained with auxiliary heads:

$$\mathcal{L}_k = -\log p_k(x_{t+k} \mid x_{\leq t}), \quad k = 1, 2, \ldots, M$$

The total loss is $\mathcal{L} = \sum_{k=1}^{M} \lambda_k \mathcal{L}_k$ with $\lambda_1 = 1$ and $\lambda_{k>1}$ typically in the range $0.1$--$0.5$. At inference time, heads $k = 2, \ldots, M$ serve as a built-in draft mechanism with high acceptance because they were co-trained with the base model.

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
KV update is the subtle part: we wrote KV for $K+1$ positions but only kept up to acceptance length. Need to truncate cache to that length without copying -- paged cache makes this a refcount adjustment.

### 8.2 Variable Acceptance => Ragged Step

Different sequences accept different prefix lengths each step. The next step's input lengths differ. Continuous batching already supports ragged sequences (paged KV, packed attention) so this drops in naturally -- but the scheduler must handle non-1 token throughput per step per sequence.

**Scheduling detail.** The scheduler tracks `accepted_len[seq_id]` per step and updates each sequence's `gen_len` counter accordingly. Token budget accounting must sum accepted tokens across all sequences rather than assuming exactly 1 token per sequence per step. When a sequence finishes (hits EOS or max length), the scheduler can immediately admit a new request into the freed slot, just as with standard continuous batching.

### 8.3 Tree-Attention Kernel

For Medusa/EAGLE trees, attention must respect the tree structure (each candidate attends to its ancestors only). Implementations: a tree mask passed to FlashAttention, or a tree-aware kernel like `flash-attn-tree`.

### 8.4 Compatibility With Disaggregation

Speculative decoding lives in the **decode pool**. Draft model and target both live there. No interaction with prefill. KV cache transfer remains target-only.

---

## 9. Failure Modes

- **Draft mismatch**: low acceptance ($\alpha < 0.4$) means spec decode is slower than baseline because of wasted draft and verifier cost.
- **Long-tail samples**: high temperature, very flat distributions cause acceptance to drop. Sometimes turn off spec decode at $\text{temp} > \sim 1$.
- **Mode interactions**: structured output / grammar masks change the distribution; spec decode must constrain draft to the same mask.
- **Tree explosion**: greedy tree expansion with too many paths burns verifier compute. Cap branches.

**Temperature sensitivity derivation.** At temperature $\tau$, the target distribution becomes $p_\tau(x) \propto p(x)^{1/\tau}$ and the draft becomes $q_\tau(x) \propto q(x)^{1/\tau}$. As $\tau \to \infty$, both approach uniform, so $\alpha \to 1$. But at moderate $\tau > 1$, the distributions flatten differently and $\|p_\tau - q_\tau\|_1$ can increase, lowering $\alpha$. At $\tau \to 0$ (greedy), $\alpha \to \mathbf{1}[{\arg\max p = \arg\max q}]$ -- either perfect acceptance or total rejection.

---

## 10. When To Use It

Almost always, for chat and code:
- Add a tiny draft (Medusa or EAGLE heads, or ngram).
- Cap $K$ at 4--6.
- Fall back to vanilla decode if $\alpha$ drops below threshold or temp $> 1$.

Less benefit:
- Very high temperature sampling (diversity workloads).
- Already throughput-saturated on tensor cores (rare in decode).
- Latency-critical with extremely strict TTFT (the verifier batch shape changes can add jitter).

---

## 11. Common Interview Questions

**Q: Why is speculative decoding effectively free per accepted token?**
A: Decode is HBM-bandwidth-bound. Reading the model weights once costs the same whether you score 1 or $K$ tokens. So $K$ verified tokens take roughly the same time as 1 baseline token. Each accepted token amortizes the cost. Quantitatively, the arithmetic intensity of single-token decode is $\sim 1$ FLOP/byte, far below the roofline crossover ($\sim 200$ on H100), so adding tokens to the batch increases FLOPs but not memory traffic.

**Q: How does rejection sampling preserve the target distribution?**
A: For each drafted token $x$, accept with probability $\min(1, \; p(x)/q(x))$. If rejected, sample the residual $(p - q)_{+} / Z$. The mixture of the acceptance branch and the rejection branch yields exactly $p$:

$$\underbrace{\min(p(x), q(x))}_{\text{accept}} + \underbrace{(p(x) - q(x))_{+}}_{\text{reject}} = p(x)$$

**Q: What is the acceptance rate and how does it determine speedup?**
A: $\alpha = \mathbb{E}_{x \sim q}[\min(1, p(x)/q(x))]$. Expected accepted prefix length is $\frac{1 - \alpha^{K+1}}{1 - \alpha}$. Speedup $\approx \mathbb{E}[\text{len}] \cdot T_{\text{target}} / (T_{\text{target}} + K \cdot T_{\text{draft}})$. High $\alpha$ and cheap draft = big speedup.

**Q: What's the difference between Medusa and EAGLE?**
A: Medusa adds $K$ parallel heads predicting future tokens directly from the target's last hidden state. EAGLE uses the target's hidden state to drive a small autoregressive head that predicts future embeddings, decoded back into tokens via the shared LM head. EAGLE achieves higher acceptance ($0.85$--$0.95$ vs $0.7$--$0.85$) at slightly more draft cost because its autoregressive structure captures inter-token dependencies that Medusa's independent heads miss.

**Q: Why might speculative decoding hurt throughput?**
A: If draft is too expensive or acceptance is too low, the wasted compute from rejected tokens plus the draft cost exceeds the gain. Need $T_{\text{draft}} / T_{\text{target}} < \sim 1/K$ and $\alpha > \sim 0.5$ for a positive ROI. The break-even condition is $\mathbb{E}[\text{len}] > 1 + K \cdot T_{\text{draft}} / T_{\text{verifier}}$.

**Q: How does KV cache handle rollback when tokens are rejected?**
A: We write KV for all $K+1$ positions during the verifier pass, but only the accepted prefix is kept. With paged KV, we just update the per-sequence "valid length" / partial-block slot. Future steps overwrite the rejected slots. No data copying is needed -- it is a metadata update on the block table.

**Q: How does spec decode interact with continuous batching?**
A: Each sequence in the batch produces a variable number of accepted tokens per step. Paged KV and packed attention support ragged sequences. The scheduler tracks per-sequence progress and admits new requests as before. The token budget per step is now variable (sum of accepted tokens across sequences), which the scheduler accounts for when deciding admission.

**Q: What about constrained decoding (JSON/grammar)?**
A: The grammar mask must be applied to BOTH the draft and the verifier. Otherwise the draft proposes invalid tokens that the verifier rejects, killing acceptance. Some implementations bypass this by drafting only from the masked distribution. The masks must be kept in sync at each position, which requires the grammar FSM state to be advanced speculatively along with the draft tokens and rolled back on rejection.

**Q: Tree-based vs sequential drafting -- what's the tradeoff?**
A: Sequential: $K$ tokens, one shot. Tree: branches at each position, more candidates, higher chance one path is right. Tree pays more verifier compute (more positions in batch). Net win when $\alpha$ is moderate (the branching hedges against rejection); net loss when $\alpha$ is already very high (sequential rarely rejects, so extra branches are wasted compute).

**Q: How do you measure spec-decode quality in production?**
A: Track `mean_accepted_tokens_per_step` and `verifier_step_latency`. Effective TPOT $= T_{\text{step}} / \text{accepted}$. Watch for drift (e.g. when temperature changes, or new fine-tunes shift distributions) and disable spec decode if effective TPOT exceeds baseline.

**Q: Can spec decode help prefill?**
A: No -- prefill processes many tokens per step already (compute-bound). Speculation gains nothing because the tensor cores are fully utilized and adding more tokens would increase actual compute time, not hide latency.

**Q: Why is n-gram drafting often surprisingly good?**
A: Code and repetitive text have very predictable next-token distributions where the recent context is almost the answer. A simple n-gram match gets 30--50% acceptance with essentially zero draft cost. Since $T_{\text{draft}} \approx 0$, even modest $\alpha$ yields a speedup: any accepted token is pure gain.

**Q: What's "multi-token prediction" in DeepSeek-V3?**
A: The model is trained with extra heads that predict tokens $t+1, t+2, \ldots$ at once. Inference uses these heads as a built-in spec mechanism -- no separate draft, high acceptance because heads are co-trained. The auxiliary loss $\mathcal{L}_k = -\log p_k(x_{t+k} \mid x_{\leq t})$ ensures the heads learn the target's distribution directly.

**Q: How does speculative decoding interact with batched inference serving multiple users?**
A: In a multi-user continuous-batching scenario, each sequence independently maintains its own draft and acceptance state. The verifier forward pass processes all sequences' candidates in a single packed batch. Different sequences may accept different numbers of tokens, producing a "ragged" output. The scheduler must track per-sequence accepted counts rather than assuming uniform 1-token-per-step progress.

---

## 12. Further Reading

- Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (2023).
- Chen et al., "Accelerating Large Language Model Decoding with Speculative Sampling" (2023).
- Cai et al., "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads" (2024).
- Li et al., "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty" (2024); EAGLE-2 / EAGLE-3.
- Fu et al., "Lookahead Decoding" (2024).
- DeepSeek-V3 technical report -- multi-token prediction.

---

**Next:** [[Quantization]].
**See also:** [[Batching_and_Scheduling]], [[Inference_Frameworks]], [[vLLM_Internals]].

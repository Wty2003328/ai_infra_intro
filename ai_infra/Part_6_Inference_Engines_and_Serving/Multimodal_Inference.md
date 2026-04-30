# Multimodal Inference — VLMs, Audio, Video, Unified

What it takes to serve vision-language models (Qwen-VL, InternVL, Llama-4-multimodal, Gemini, GPT-4o), audio models (Whisper, Moshi, Voxtral), and video understanding / generation models in production. Covers the architectural patterns, the encoder-LLM coupling, KV cache for multimodal tokens, real-time audio (full-duplex), and serving challenges unique to each modality.

**Prerequisites**: [Frontier_Models_2025_2026](../Part_5_Algorithms_and_Quantization/Frontier_Models_2025_2026.md), [KV_Cache](../Part_6_Inference_Engines_and_Serving/KV_Cache.md), [Inference_Frameworks](Inference_Frameworks.md).

---

## 1. The Multimodal Architecture Spectrum

There are three main coupling patterns:

### 1.1 Late-Fusion (Encoder + LLM)

```
Image → ViT encoder → projected tokens → injected into LLM context → text out
```

Examples: LLaVA family, Qwen-VL, InternVL. Two separate models composed.
- Pros: modular; reuse existing LLM weights; cheap to add new modality.
- Cons: encoder sees image only once; no late integration.

### 1.2 Early-Fusion (Native Multimodal)

```
Image / audio → patches / tokens → fed alongside text tokens through one unified backbone
```

Examples: Llama-4, Chameleon, Janus, Gemini-2.x, GPT-4o native.
- Pros: deeper integration; modality-agnostic backbone; supports any-to-any generation.
- Cons: pretraining is more expensive; harder to add modalities post-hoc.

### 1.3 Adapter / Cross-Attention

```
LLM with frozen weights + cross-attention layers attending to encoder outputs
```

Examples: Flamingo, Idefics, BLIP-2.
- Pros: can be added to a strong LLM with limited training compute.
- Cons: heavyweight at inference; less common at the frontier in 2026.

---

## 2. Vision-Language Models in Production

### 2.1 Architecture: Late-Fusion VLM

```
Image (HxW pixels)
  │
  ▼
ViT encoder (e.g., SigLIP-400M, OpenCLIP, InternViT)
  │ output: N image patches × D_vit
  ▼
MLP projector (small, often 2-layer)
  │ output: N tokens × D_llm
  ▼
LLM context: [<system>] [<user>] <image_tokens> <text_tokens> [<assistant>] …
  │
  ▼
LLM forward → text out
```

Image token count N varies:
- Fixed (e.g., 256 or 729 tokens per image).
- Dynamic (e.g., AnyRes, NaViT, Qwen2-VL "M-RoPE"): tokens proportional to image size.

For dynamic schemes, an HD image becomes 2K-10K tokens, dramatically increasing context length and KV pressure.

### 2.2 Frontier VLMs (2025–2026)

| Model | Open? | Encoder | LLM backbone | Notes |
|-------|-------|---------|--------------|-------|
| Qwen2.5-VL, Qwen3-VL | ✓ | InternViT-style | Qwen-2.5/3 | M-RoPE for image+video; very strong |
| InternVL-2.5 / 3.5 | ✓ | InternViT-6B | InternLM-2.5 / Qwen | Largest open VLM |
| Llama-4 (multimodal) | ✓ | early-fusion | Llama-4 | Native multimodal, image+video |
| Pixtral-12B / Pixtral-Large | ✓ | custom | Mistral | Strong on doc / chart tasks |
| Gemma-3 (multimodal) | ✓ | SigLIP | Gemma-3 | Compact, on-device target |
| Phi-4-multimodal | ✓ | custom | Phi-4 | Includes audio + vision |
| GPT-4o, GPT-5 | ✗ | early-fusion | – | Native any-to-any |
| Claude-4 (Opus / Sonnet) | ✗ | encoder-LLM | – | Strong on doc + chart |
| Gemini-2.5 (Pro / Flash) | ✗ | early-fusion | – | Native, very long video context |

### 2.3 Inference Engineering for VLMs

- **Two-stage forward**: encoder (small, often a single GPU is enough) → LLM (the heavy part). Pipeline them.
- **Encoder caching**: if the same image is sent multiple times (rare in chat, common in agentic), cache the encoder output indexed by image hash.
- **Image token prefix caching**: image tokens are deterministic given the image + projector → can be prefix-cached in the LLM's KV cache. Big win for chat workflows revisiting attached images.
- **Pre-image prompt cache**: system prompts before the image still cache normally.
- **Dynamic resolution**: Qwen2-VL's variable token count helps long-document images but blows up KV memory; production sets a per-request token cap.

---

## 3. Long-Context Visual Workloads

Document and video understanding push context lengths into the millions:
- A 50-page PDF at high-res ≈ 50 × 1000 tokens = 50K tokens.
- A 1-hour video at 1 fps with 256 tokens/frame ≈ 920K tokens.
- A 10-min screen recording at 5 fps ≈ 760K tokens.

Mitigations:
- **Frame sampling**: skip frames; use dense at key moments.
- **Token compression**: average multiple patches; "video frames as 1 token" extreme compression in some Llama-4 / Gemini configurations.
- **Streaming long-video** ingestion: process chunks, summarize, attend to summaries.
- **MLA / KV compression**: as in pure-text long context, compress the KV.

For 1M+ video understanding, the same long-context engineering applies: sliding-window or chunked attention, prefix caching, NSA-style sparse attention.

---

## 4. Audio Models

### 4.1 ASR (Speech-to-Text)

- **Whisper** (OpenAI): encoder-decoder transformer, all sizes (tiny → large-v3). Standard production ASR.
- **Faster-Whisper / WhisperX**: optimized inference (CTranslate2 backend, batching, alignment).
- **Distil-Whisper**: distilled smaller variants.
- **NeMo Parakeet, Canary** (NVIDIA): production ASR with streaming support.

Inference patterns:
- Batched chunks of 30-second audio.
- Streaming: feed audio incrementally, emit transcript with low latency.
- VAD (voice-activity detection) to chunk efficiently.

### 4.2 TTS (Text-to-Speech)

- **XTTS, OpenVoice, F5-TTS** (open-source).
- **ElevenLabs**, **OpenAI TTS** (closed).
- Architecture: typically autoregressive token model + vocoder (HiFi-GAN, BigVGAN), or non-autoregressive (e.g., flow-matching).

Real-time streaming TTS is the goal for voice agents.

### 4.3 Audio-LLM (Voice Models)

The frontier:
- **Moshi** (Kyutai, 2024): full-duplex audio LLM. Speaks and listens simultaneously. Sub-second turn latency.
- **GPT-4o voice mode**: similar full-duplex; details closed.
- **Voxtral** (Mistral): open audio LLM.
- **Qwen-Audio**: audio-input chat model.
- **Phi-4-multimodal**: includes audio.

Architecturally: audio in token stream alongside text. RVQ (residual vector quantization) tokenizes audio at e.g. 12.5 Hz × 8 codebooks.

### 4.4 Real-Time Voice Engineering

A live voice agent has hard constraints:
- **End-to-end turn latency < 500ms** for natural feel.
- **Streaming everywhere**: audio in (chunked), tokens generated streaming, audio out (streaming TTS).
- **Interrupt handling**: user speaks while model speaks → cancel current generation.
- **VAD + endpoint detection**: when did the user stop talking?
- **Echo cancellation, noise suppression**: at audio I/O level.

The systems stack runs ASR + LLM + TTS in parallel pipelines. Modern voice modes run audio-LLM end-to-end (Moshi) for lowest latency.

---

## 5. Image and Video Generation

### 5.1 Diffusion Models

- **SDXL, SD3, Flux**: image gen.
- **Sora, Veo-2/3, Kling, MovieGen, Wan-2.1**: video gen.

Architecture: U-Net or transformer-based denoiser + VAE encoder/decoder + text encoder (CLIP / T5).

Inference:
- **N denoising steps** (typically 20-50 for image, 40-200 for video).
- Each step is a forward pass through a heavy model.
- Compute-bound; latency dominated by step count × per-step cost.

Optimizations:
- **Distillation** (DMD, Hyper-SD, LCM): reduce denoising steps to 1-4.
- **Quantization**: FP8 / FP4 of the denoiser.
- **Flash attention** in the denoiser.
- **Cache step outputs** for similar prompts.

### 5.2 Inference Frameworks

- **xDiT**: distributed diffusion inference framework with TP/SP/PP for video models.
- **Diffusers** (HF): the de facto framework for image gen.
- **ComfyUI**: node-based pipeline; popular for stable-diffusion-family work.

For video gen at scale (Sora-class), inference requires multi-GPU TP within a node, possibly multi-node CP for very long videos.

### 5.3 Auto-Regressive Image Models

- **Janus / Janus-Pro** (DeepSeek), **Show-o** (Show Lab), **LWM**, **Chameleon**: AR token-by-token image gen.
- Slower than diffusion per-image but unified with text in one model.

Niche but growing.

---

## 6. Unified Any-to-Any Models

GPT-4o, Gemini-2.x, Janus-Pro represent the unified direction:
- Single backbone takes any modality token stream.
- Outputs any modality.
- Image, audio, video, text — all in one context.

Engineering: each modality has its own tokenizer / detokenizer (image VQ-VAE, audio RVQ, text BPE). The serving engine handles each, then interleaves into one token stream for the LLM.

---

## 7. Multimodal Frameworks and Engines

| Framework | Multimodal support | Notes |
|-----------|--------------------|-------|
| vLLM | VLMs, audio (Phi-4), some video | Mature for VLMs, growing audio |
| SGLang | VLMs (Qwen-VL, InternVL) | Multi-modal RadixAttention |
| TRT-LLM | VLMs (LLaVA, Qwen-VL) | Strong on quantized inference |
| LMDeploy / TurboMind | VLMs (InternVL series) | Strong on InternVL/Qwen-VL |
| Diffusers (HF) | Image / video gen | Standard for diffusion |
| xDiT | Video gen distributed | TP/SP/PP for diffusion |
| NVIDIA NIM | Multimodal containers | Production deployments |

The 2025 trend: VLM serving rapidly converging on the same engines as text-only LLMs (vLLM, SGLang). Audio and video remain in their own ecosystems but adopting the same patterns.

---

## 8. Multi-Modal Specific Optimizations

### 8.1 Encoder-LLM Pipelining

Encoder runs on a small dedicated pool (1-4 GPUs). LLM runs on a larger pool. The frontend dispatches images to the encoder pool, waits for image tokens, then dispatches the full request (text + image tokens) to the LLM pool. Pipelining hides encoder latency.

### 8.2 Image Token Caching

A given image processed by a fixed encoder + projector produces deterministic tokens. Cache by image hash → projector output. Saves the encoder pass on repeats.

### 8.3 KV Cache for Multimodal

KV cache works the same for image tokens as text tokens (they're just embeddings). Two implications:
- Image-prefix caching is automatic.
- KV memory growth includes image tokens — at high-res with N=2000 image tokens, a 4-image prompt is 8K tokens of KV. Plan capacity.

### 8.4 Streaming Multimodal Output

For voice / video output, output streams byte-by-byte. Engine generates token, vocoder/decoder converts to audio/image chunk, sent immediately. Lower TTFT critical.

### 8.5 Multimodal Routing

Frontend may route based on modality:
- Image+text request → VLM-capable replica.
- Audio request → audio-capable replica.
- Pure text → text-LLM replica (cheaper).

Smart routing reduces cost.

---

## 9. Common Pitfalls

- **Underestimating image token counts at high-res**: dynamic-resolution VLMs can produce 5K+ tokens per image; KV explodes.
- **Cold-start of encoders**: ViT loading takes seconds; warm-pool an encoder pool separately.
- **Ignoring video memory cost**: hour-long video inputs at any reasonable resolution are massive; cap input length.
- **Audio streaming buffer-bloat**: too-large audio chunks add latency; 50-100ms chunks are typical.
- **Text-token KV-cache mismatch with image tokens**: some engines store them separately or incorrectly; verify on a test.
- **TTS step-count too high**: vocoder's 20+ steps can dominate latency for short utterances.
- **Multimodal prompt cache invalidation**: small image variations produce different hashes; expect lower hit rates than text.

---

## 10. Common Interview Questions

**Q: How does a vision-language model serve a request with an image?**
A: Image is encoded by a ViT (separate model, often on its own pool); output projected to LLM token-space dimension; injected into the LLM's context as N "image tokens"; LLM runs forward as usual on the combined sequence. Two stages, often pipelined.

**Q: What's the difference between late-fusion and early-fusion multimodal?**
A: Late-fusion: separate encoder + LLM, modality embedding injected into context. Modular and cheap to extend. Early-fusion: one unified backbone trained on all modalities from scratch. Tighter integration, supports any-to-any generation, but more expensive to train. Llama-4 / GPT-4o / Gemini are early-fusion.

**Q: How does dynamic-resolution image tokenization affect serving?**
A: Models like Qwen2-VL produce variable image token counts (256 → 5000+) based on image size. This dramatically affects KV cache pressure and TTFT. Production deployments set per-request caps or downsample large images before encoding.

**Q: What's "image-token prefix caching"?**
A: Image tokens output by a fixed encoder are deterministic given the image. The LLM's KV cache treats them like any other tokens, so chats that revisit the same attached image hit the prefix cache for those tokens. Saves the encoder pass + the LLM prefill of image tokens.

**Q: How would you serve a 1-hour video QA workload?**
A: Encode at 1 fps → ~3600 frames × ~50-256 tokens = 200K-900K context. Use a long-context-capable VLM (Llama-4, Gemini-2.x) with chunked attention. Apply frame-sampling heuristics to compress to ~50K tokens. Use FP8 KV. Disaggregate prefill from decode (long video prompts have heavy prefill). Cap concurrent video sessions.

**Q: What's special about Moshi-style audio LLMs?**
A: Full-duplex: model listens and speaks simultaneously via interleaved audio token streams (RVQ tokenization, ~12.5 Hz × 8 codebooks). Sub-second turn latency. Architecturally a transformer LLM with audio tokens; serving requires bidirectional streaming and VAD-style interrupt handling.

**Q: Why is video diffusion inference so expensive?**
A: Each generation step is a forward pass through a large transformer / U-Net denoiser; videos need 40-200 steps; each step processes the full N-frame latent. Total FLOPs scale with frames × steps × model size. Optimizations: distilled few-step models, FP8 quantization, distributed parallelism (xDiT), CP for long videos.

**Q: How does Whisper handle real-time streaming?**
A: WhisperX / Faster-Whisper / NeMo Canary use a sliding window over the audio stream; emit partial transcripts as 30-second windows complete; finalize after VAD detects end-of-speech. Latency is dominated by the chunk size; smaller chunks → more work, lower latency.

**Q: What infrastructure does a real-time voice agent need?**
A: ASR (or audio-LLM) → LLM → TTS (or audio-LLM speaks). All streaming. Sub-500ms turn latency target. Interrupt handling. VAD for endpointing. WebRTC or similar for client transport. If using audio-LLM (Moshi), all-in-one — much simpler latency picture.

**Q: How do you cache image preprocessing at inference?**
A: Hash the image bytes (or a perceptual hash); cache the (image_hash → encoder_output_tokens) mapping. Serves repeated images without re-encoding. Useful for agentic flows that revisit images and document chats.

**Q: What's xDiT and when do you need it?**
A: Distributed diffusion inference framework. Implements TP / SP / CP for video diffusion models. Needed when one denoiser step doesn't fit on a single GPU (large video models), or when latency requires splitting steps across GPUs.

**Q: How do unified any-to-any models change the serving stack?**
A: One model handles all modalities → simplifies routing (no per-modality replica selection at runtime). But context can include arbitrary modalities → larger and more variable KV requirements. Tokenizer / detokenizer plumbing per modality. Output streaming needs to handle modality switches.

**Q: How does Llama-4's native multimodal differ from Qwen-VL?**
A: Llama-4 is early-fusion: image and video tokens flow through the same backbone from pretraining. Qwen-VL is late-fusion: a Qwen LLM with a separate ViT encoder and projector. Llama-4 should integrate modalities more deeply; Qwen-VL is easier to extend post-hoc.

---

## 11. Further Reading

- "Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution" (2024).
- "InternVL 1.5 / 2 / 2.5" technical reports (Shanghai AI Lab).
- "Llama-4" release notes (Meta, 2025).
- "Gemini 1.5 / 2.5" technical papers (Google).
- "GPT-4o System Card" (OpenAI, 2024).
- "Moshi" technical report (Kyutai, 2024).
- "Sora" technical report (OpenAI, 2024).
- "Movie Gen" (Meta, 2024).
- "Veo 2 / 3" overviews (Google DeepMind).
- "Janus-Pro" (DeepSeek, 2025) — unified any-to-any.
- "Phi-4-multimodal" technical report (Microsoft, 2025).
- xDiT, Diffusers, Faster-Whisper repositories.

---

**Next:** [Cutting_Edge_Kernels](../Part_4_GPU_Kernel_Engineering/Cutting_Edge_Kernels.md).
**See also:** [Frontier_Models_2025_2026](../Part_5_Algorithms_and_Quantization/Frontier_Models_2025_2026.md), [KV_Cache](../Part_6_Inference_Engines_and_Serving/KV_Cache.md), [Long_Context_Engineering](../Part_6_Inference_Engines_and_Serving/Long_Context_Engineering.md).

# Gemma 4 — Optimized Inference Engine

High-performance inference for Google's Gemma 4 model family, with two deployment targets:

| Target | Model | Hardware | Peak Speed | Approach |
|--------|-------|----------|------------|----------|
| **GPU** | Gemma 4 E4B (4B) | RTX 3090 | 66.9 tok/s | Custom Python engine + torch.compile |
| **APU** | Gemma 4 31B-it | AMD Ryzen AI MAX+ 395 | 49.3 tok/s | TurboQuant + lookup decoding (llama.cpp) |

Both deployments push their respective hardware to the memory bandwidth wall — the fundamental speed limit for LLM inference.

---

## Gemma 4 31B on AMD APU

**49.3 tok/s effective on a 31B model using a mini PC.** No discrete GPU required.

The AMD Ryzen AI MAX+ 395 is an APU with 40 RDNA 3.5 compute units sharing 64GB of LPDDR5-8000 with the CPU. We combine three techniques to maximize throughput:

```
Stock Q4_K_M:              10.6 tok/s  (1.0x)  baseline
TurboQuant:                13.9 tok/s  (1.3x)  mixed-precision quantization
TurboQuant + Lookup:       49.3 tok/s  (4.7x)  n-gram self-speculation
Theoretical max:           19.7 tok/s  (bandwidth limit at TurboQuant size)
```

### Quick Start

```bash
# Prerequisites: llama.cpp built with ROCm/HIP for your GPU
cd llama.cpp

# 1. Download model (Q4_K_M from HuggingFace)
huggingface-cli download bartowski/gemma-4-31B-it-GGUF \
  gemma-4-31B-it-Q4_K_M.gguf --local-dir ~/models/

# 2. Create TurboQuant (mixed-precision: q2K FFN + q4K attention)
bash apu/create_turbo_quant.sh ~/models/gemma-4-31B-it-Q4_K_M.gguf ~/models/gemma-4-31B-it-TURBO.gguf

# 3a. Run as API server (~14 tok/s, OpenAI-compatible)
bash apu/gemma_server.sh

# 3b. Run interactive chat (~49 tok/s with lookup decoding)
bash apu/gemma_chat.sh

# 4. Install as systemd service (auto-start on boot)
sed "s/YOUR_USER/$(whoami)/g" apu/gemma-server.service | sudo tee /etc/systemd/system/gemma-server.service
sudo systemctl daemon-reload && sudo systemctl enable --now gemma-server
```

### Using with Claude Code / Claw Code

The 31B API server is OpenAI-compatible, so you can use it as a backend for Claude Code, Claw Code, or any tool that speaks the OpenAI chat completions API:

```bash
# Start the server (or use systemd)
bash apu/gemma_server.sh

# Use with Claude Code / Claw Code:
OPENAI_BASE_URL=http://localhost:8082/v1 \
OPENAI_API_KEY=local \
  claw "your prompt here"

# Or configure as a permanent backend in your client's settings
```

The E4B server (`server.py`) additionally provides an **Anthropic Messages API** with full tool-calling translation — it converts between Anthropic's tool format and Gemma 4's native `<|tool_call>` tokens:

```bash
python server.py   # starts on http://localhost:8642

ANTHROPIC_BASE_URL=http://localhost:8642 ANTHROPIC_API_KEY=local \
  claw "your prompt"
```

> **Note:** The lookup decoding speedup (4.7x) is only available in interactive/CLI mode via `gemma_chat.sh`. The API server runs at standard generation speed (~14 tok/s) because llama-server's ngram speculation doesn't achieve meaningful acceptance rates for diverse requests. For batch/API workloads, the speed comes from TurboQuant alone (+30%).

### TurboQuant: Adaptive Mixed-Precision Quantization

Standard quantization applies the same bit-width uniformly across all tensors. TurboQuant applies different precision based on each tensor's impact on quality vs bandwidth:

```
Gemma 4 31B weight distribution (per-token memory read):

  FFN layers (60 layers):          10.77 GB   67.5%   ← bandwidth hog
  SWA Attention (50 layers):        3.42 GB   21.4%   ← quality-critical
  Full Attention (10 layers):       1.03 GB    6.4%   ← quality-critical
  Embeddings/Output:                0.73 GB    4.6%   ← small, keep high precision
```

FFN layers dominate bandwidth (67.5%) but are less sensitive to quantization than attention layers. TurboQuant exploits this:

| Tensor | Stock Q4_K_M | TurboQuant | Rationale |
|--------|-------------|------------|-----------|
| FFN gate/up | q4_K (4.5 bpw) | **q2_K (2.6 bpw)** | Tolerates compression, saves most bandwidth |
| FFN down | q4_K-q6_K | **q3_K (3.4 bpw)** | Slightly more sensitive than gate/up |
| Attention Q/K/V/O | q4_K-q6_K | q4_K-q6_K | Preserved — quality-critical |
| Embeddings | q6_K | q6_K | Preserved — small percentage |

**Result: 12.7 GB (3.55 bpw) vs 17.1 GB (4.77 bpw) — 26% smaller, 30% faster, attention quality preserved.**

### Lookup Decoding: Breaking the Bandwidth Wall

Normal autoregressive generation reads the entire model for every single token:

```
Standard:  token₁ → [read 13GB] → token₂ → [read 13GB] → token₃ → ...
           Each token costs one full model read = 13 GB / 180 GB/s = 72ms = 13.9 tok/s
```

Lookup decoding uses n-gram pattern matching to **predict** the next several tokens, then batch-verifies them all at once:

```
Lookup:    predict tokens₁₋₁₆ → [read 13GB ONCE, verify all 16] → accept 15 → done
           16 tokens for one model read = 72ms / 16 = 4.5ms/tok = 49+ tok/s
```

The n-gram predictor builds a lookup table from the model's own output as it generates. When it sees a token sequence it's seen before (e.g., `self.left = None\n        self.right = None`), it predicts the same continuation. The model then verifies the entire batch at prompt-eval speed (~52 tok/s) instead of generating one-by-one.

### Comprehensive Benchmarks

**Hardware:** AMD Ryzen AI MAX+ 395, 40 CUs @ 2900 MHz, 64GB LPDDR5-8000 (8 channels, 256 GB/s theoretical)

#### Raw Generation Speed

| Quantization | Size | BPW | Generation | Prompt Eval | Bandwidth Used | Efficiency |
|-------------|------|-----|-----------|-------------|---------------|------------|
| Q4_K_M | 17.1 GB | 4.77 | 10.6 tok/s | 293.8 tok/s | 182 GB/s | 71% |
| Q3_K_M | 14.8 GB | 4.14 | 12.0 tok/s | 299.2 tok/s | 179 GB/s | 70% |
| **TurboQuant** | **12.7 GB** | **3.55** | **13.9 tok/s** | **316.9 tok/s** | **180 GB/s** | **70%** |
| Q2_K | 11.8 GB | 3.00 | 15.0 tok/s | 324.7 tok/s | 182 GB/s | 71% |

All quantizations converge to ~180 GB/s actual memory bandwidth — the hardware ceiling.

#### Prompt Eval Scaling (TurboQuant)

| Batch Size | Throughput | Speedup vs pp1 |
|-----------|-----------|----------------|
| pp1 | 12.8 tok/s | 1.0x |
| pp32 | 100.4 tok/s | 7.8x |
| pp128 | 206.9 tok/s | 16.2x |
| pp512 | 331.3 tok/s | 25.9x |

Batch processing amortizes the bandwidth cost across multiple tokens. This is why lookup decoding is fast — it converts generation into batch verification.

#### Lookup Decoding by Content Type

| Content | Accept Rate | Eval Runs / 512 tok | Effective tok/s |
|---------|------------|---------------------|----------------|
| Code | 100% | 3 | 46.5 |
| Prose | 100% | 2 | 46.9 |
| JSON/Structured | 100% | 2 | 46.9 |
| Math/Reasoning | 96% | 11 | 42.0 |

Lookup decoding achieves near-100% acceptance across all content types — not just code.

#### Lookup Decoding vs Generation Length

| Tokens | Time | Effective tok/s | Accept Rate |
|--------|------|----------------|------------|
| 64 | 2.0s | 44.0 | 100% |
| 128 | 3.5s | 44.8 | 92% |
| 256 | 5.9s | 49.6 | 95% |
| 512 | 11.0s | 49.3 | 98% |

Speed **increases** with longer outputs as the lookup cache accumulates more patterns.

#### Generation Timing Stability

| Context Length | tok/s |
|---------------|-------|
| tg32 | 12.92 |
| tg64 | 12.93 |
| tg128 | 12.94 |
| tg256 | 12.93 |

KV cache growth has zero measurable impact within 4K context (variance: +/- 0.01 tok/s).

### Bandwidth Analysis

```
The fundamental equation:   tokens/sec = memory_bandwidth / model_size

  180 GB/s / 12.7 GB (TurboQuant) = 14.2 tok/s   ← matches measured 13.9 tok/s
  180 GB/s / 17.1 GB (Q4_K_M)     = 10.5 tok/s   ← matches measured 10.6 tok/s

Hardware theoretical max:   256 GB/s
Achieved:                   180 GB/s  (70%)
Gap (76 GB/s) breakdown:
  - Memory controller overhead (bank conflicts, refresh)   ~30 GB/s
  - GPU kernel scheduling and launch latency               ~20 GB/s
  - KV cache reads/writes                                  ~10 GB/s
  - Norms, softmax, activation functions                   ~5 GB/s
  - Cache line waste and alignment padding                 ~10 GB/s
```

Lookup decoding sidesteps this equation entirely by converting sequential generation (1 token per model read) into batch verification (16+ tokens per model read).

### What We Tested (Exhaustive)

| Technique | Result | Status |
|-----------|--------|--------|
| TurboQuant (mixed-precision) | +30% generation speed | **Deployed** |
| Lookup decoding (n-gram self-speculation) | +370% effective speed | **Deployed** |
| `--reasoning off` (skip thinking tokens) | 30-70% fewer wasted tokens | **Deployed** |
| Q3_K_M uniform quant | +14% generation speed | Tested |
| Q2_K uniform quant | +42% generation speed | Tested |
| Speculative decoding (Q2_K draft model) | -32% (shared memory bus) | Rejected |
| CPU-GPU hybrid layers | -10% (shared memory bus) | Rejected |
| HIP graphs + WMMA flash attention build | 0% improvement | Rejected |
| mlock / direct-io / realtime priority | 0% improvement | Rejected |
| HSA_ENABLE_SDMA=0 | 0% improvement | Rejected |
| KV cache quantization (q4_0) | -3% | Rejected |
| ngram speculation in server mode | 0% acceptance rate | Rejected |
| Prompt caching (--swa-full) | Incompatible with SWA architecture | Broken |
| Layer pruning via llama-quantize | Re-quantization produces NaN | Broken |

### Architecture: Where the Bandwidth Goes

```
Gemma 4 31B (60 layers, 5376 hidden, 21504 FFN)

Per-token memory read at TurboQuant (12.7 GB total):
┌───────────────────────────────────────────────────────────────┐
│  FFN gate+up (q2_K, 60 layers)      3.8 GB   30%   ████████ │
│  FFN down (q3_K, 60 layers)         2.8 GB   22%   ██████   │
│  Attention Q (q4_K-q6_K, 60 layers) 1.9 GB   15%   ████     │
│  Attention O (q4_K, 60 layers)      1.7 GB   14%   ████     │
│  Attention V (q4_K-q5_K, 50 layers) 0.7 GB    5%   ██       │
│  Attention K (q3_K, 60 layers)      0.5 GB    4%   █        │
│  Token embeddings (q6_K)            1.2 GB    9%   ███      │
│  Norms, scalars (f32)               0.005 GB  0%            │
└───────────────────────────────────────────────────────────────┘

Sliding window attention:  50 of 60 layers (window=1024, head_dim=256, 16 KV heads)
Full attention:            10 of 60 layers (head_dim=512, 4 KV heads)
Pattern: [SWA, SWA, SWA, SWA, SWA, Full] x 10
```

---

## Gemma 4 E4B on NVIDIA GPU

**3.8x faster than HuggingFace on a single RTX 3090. 1,742 lines of Python. No custom CUDA.**

A from-scratch inference engine for Google's [Gemma 4 E4B](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/) (4B parameter model) that bypasses all framework overhead and hits 66% of the GPU's theoretical memory bandwidth limit.

```
HuggingFace baseline:   17.5 tok/s  (1.0x)
Custom engine:           22.6 tok/s  (1.3x)
+ torch.compile:         66.9 tok/s  (3.8x)  <-- peak
+ TurboQuant (16K ctx):  22.7 tok/s  (1.3x)  + 4x context
Theoretical max:        101.0 tok/s  (RTX 3090 bandwidth limit)
```

### Quick Start

```bash
# Install
git clone https://github.com/YOUR_USER/gemma4-engine.git
cd gemma4-engine
pip install -r requirements.txt

# Run (downloads model automatically on first run)
python engine.py                  # ~22 tok/s, instant start
python engine.py --compile        # ~67 tok/s, ~100s compile warmup

# With 16K context (TurboQuant KV compression)
python engine_turbo.py            # ~22 tok/s, 16K context
python engine_turbo.py --compile  # ~67 tok/s, 16K context

# Benchmark
python benchmark.py               # Full comparison (HF vs custom vs compiled)
python benchmark.py --quick       # Quick run (skip HF + compile)
```

### Anthropic-Compatible API Server

Serves Gemma4 as a local drop-in replacement for the Claude API, with full tool-calling translation:

```bash
python server.py   # starts on http://localhost:8642

# Use with any Anthropic-compatible client:
ANTHROPIC_BASE_URL=http://localhost:8642 ANTHROPIC_API_KEY=local \
  your-tool-here "your prompt"
```

### System Requirements

| Component | Minimum | Tested |
|-----------|---------|--------|
| GPU | 20GB+ VRAM (RTX 3090, 4090, A5000+) | RTX 3090 24GB |
| RAM | 16GB | 46GB |
| CUDA | 12.0+ | 12.1 |
| PyTorch | 2.4+ | 2.10.0 |
| Python | 3.10+ | 3.13.5 |

Model weights are ~15GB in bfloat16 and download automatically from HuggingFace on first run.

---

## Architecture Deep Dive

### Gemma 4 E4B Model Architecture

Gemma 4 E4B is not a standard transformer. It has several unique features that required careful handling:

```
Input tokens
    |
    v
[Embedding] ─────── embed_tokens: [262144, 2560]  (1.3 GB)
    |
    ├── [Per-Layer Embedding] ── embed_tokens_per_layer: [262144, 10752]  (5.6 GB!)
    |                            Each of 42 layers gets its own 256-dim token embedding.
    |                            This is the "E4B" innovation — 38% of total model weight.
    |
    v
[42 Transformer Layers] ──── Mixed attention pattern:
    |                         - 35 sliding-window layers (window=512, head_dim=256)
    |                         -  7 full-attention layers  (head_dim=512)
    |                         Pattern repeats: [S,S,S,S,S,F] x 7
    |
    |   Each layer:
    |   ┌──────────────────────────────────────────────────────────┐
    |   │  RMSNorm → Attention → RMSNorm → Residual              │
    |   │  RMSNorm → GeGLU MLP → RMSNorm → Residual              │
    |   │  Per-Layer Input Gate (GELU + multiply + project)       │
    |   │  Layer Scalar (learned, range 0.06 to 0.89)             │
    |   └──────────────────────────────────────────────────────────┘
    |
    |   Attention details:
    |   - GQA 8:2 (8 query heads, 2 KV heads, 4x grouping)
    |   - Q/K RMSNorm before RoPE
    |   - V RMSNorm (no learnable scale)
    |   - Scaling factor = 1.0 (not 1/sqrt(d) — norms handle it)
    |   - KV Sharing: Layers 24-41 reuse KV from layers 22/23
    |
    |   RoPE config:
    |   - Sliding: theta=10,000, full rotation on 256 dims
    |   - Full:    theta=1,000,000, partial rotation (25% of 512 dims)
    |
    v
[RMSNorm → LM Head] ─── Tied with embed_tokens (no separate weight)
    |                     Logit softcapping: tanh(logits/30) * 30
    v
Output logits [262144]
```

### Weight Distribution (E4B)

```
                    What the GPU reads every token
┌──────────────────────────────────────────────────────────┐
│  MLP weights (42 layers)              6.61 GB    71%     │ ████████████████████████████████████
│  LM head (262K vocab)                 1.34 GB    14%     │ ███████
│  Attention projections                1.17 GB    13%     │ ██████
│  Per-layer input gate                 0.11 GB     1%     │ █
│  Embedding lookups                    ~0 GB      ~0%     │
├──────────────────────────────────────────────────────────┤
│  TOTAL per decode step                9.29 GB            │
└──────────────────────────────────────────────────────────┘

RTX 3090 memory bandwidth: 936 GB/s
Theoretical max: 936 / 9.29 = 101 tok/s
Achieved: 66.9 tok/s = 66% utilization
```

**The MLP is 71% of all weight reads.** Each of the 42 layers reads a fused gate+up projection (20480 x 2560) and a down projection (2560 x 10240). This is the single biggest bottleneck and cannot be reduced without quantization.

### Where Time Goes (Per Decode Step)

Profiled at cache length 100, uncompiled:

```
Component               Time (ms)    % of total
─────────────────────── ─────────── ──────────
Python/dispatch overhead    19.7 ms      42%    <-- torch.compile eliminates this
MLP (42 layers)             9.0 ms      19%
RMSNorm (168 calls)        11.1 ms      24%
Attention projections       2.6 ms       6%
LM head + softcap          1.5 ms       3%
SDPA attention              0.7 ms       1%
Per-layer input gate        1.9 ms       4%
Embeddings                  0.1 ms       0%
─────────────────────── ─────────── ──────────
TOTAL                      46.6 ms     100%    = 21.5 tok/s (uncompiled)

After torch.compile:       14.9 ms            = 67.0 tok/s (compiled)
```

**The #1 bottleneck is Python dispatch overhead (42%).** Each decode step launches ~630 CUDA kernels through PyTorch's Python dispatch layer. `torch.compile` fuses these into fewer optimized kernels, eliminating most of the overhead and achieving a 3.1x speedup from compilation alone.

---

## Optimizations Explained

### 1. Direct Weight Loading

**Problem:** HuggingFace's `from_pretrained()` creates Python model classes, initializes buffers, moves to device, and runs post-processing. This adds seconds of startup and ongoing dispatch overhead.

**Solution:** Load safetensor files directly to GPU, skipping all model class construction:
```python
with safe_open(path, framework="pt", device="cuda") as f:
    for key in f.keys():
        weights[key] = f.get_tensor(key).to(torch.bfloat16)
```

### 2. Pre-Resolved Weight References

**Problem:** Looking up `self.w["layers.5.self_attn.q_proj.weight"]` in a Python dict requires string hashing and comparison — done ~15 times per layer, 42 layers per token.

**Solution:** Resolve all weight tensor references at init time into a `LayerWeights` struct with `__slots__`:
```python
class LayerWeights:
    __slots__ = ['q_proj', 'kv_proj', 'o_proj', 'gate_up_proj', 'down_proj', ...]
```
Every weight is a direct attribute access — zero dict lookups in the decode loop.

### 3. Fused Projections

**Problem:** Gate and up projections in the MLP are two separate matmuls. K and V projections are two separate matmuls. Each kernel launch has ~10-20us of overhead.

**Solution:** Concatenate weights at init, split output after:
```python
# At init: gate_up = cat([gate_proj, up_proj], dim=0)  # [20480, 2560]
# At runtime: gate, up = F.linear(x, gate_up).chunk(2, dim=-1)
```
Saves 42 MLP launches + 24 KV launches = 66 fewer kernel launches per token.

### 4. Static KV Cache

**Problem:** Dynamic KV cache uses `torch.cat()` to grow, which allocates new memory every token and prevents CUDA graph capture.

**Solution:** Pre-allocate the full cache at init:
```python
self.k = [torch.zeros(1, kv_heads, max_seq, head_dim, device="cuda") for ...]
```
Writing new KV is just a slice assignment — no allocation, no copy, no graph break.

### 5. torch.compile

**Problem:** 630+ individual CUDA kernel launches per decode step, each with Python dispatch overhead.

**Solution:** `torch.compile(mode="default")` traces the forward pass and fuses operations:
- Multiple small matmuls → batched operations
- RMSNorm + matmul chains → fused kernels
- Element-wise chains (GELU, multiply, add) → single kernels

Result: **3.1x speedup** (21.5 → 66.9 tok/s) from compilation alone.

### 6. Last-Token-Only Logits

**Problem:** During prefill, `F.linear(hidden, embed_tokens)` computes logits for ALL input tokens. With 262K vocab and 2000+ token prefill, this creates a 2GB+ intermediate tensor.

**Solution:** Only compute logits for the last token:
```python
hidden_last = hidden[:, -1:, :]
logits = F.linear(rms_norm(hidden_last, self.final_norm), self.embed_tokens)
```
Saves 2GB+ VRAM on long prefills, enabling larger context windows.

---

## TurboQuant KV-Cache Compression (E4B)

`engine_turbo.py` extends the base engine with compressed KV storage for full-attention layers, enabling 16K+ context in the same VRAM.

### How It Works

**Key insight:** Only the 7 full-attention layers (of 42 total) have unbounded KV cache. The 35 sliding-window layers are already capped at 512 tokens. TurboQuant targets only the full-attention layers.

**Algorithm: Lloyd-Max quantization with random orthogonal rotation**

```
Original KV vector: x ∈ R^512 (bf16, 1024 bytes)

1. Save norm:     ||x||₂
2. Normalize:     x_unit = x / ||x||₂  
3. Rotate:        y = x_unit @ Pi^T     (Pi = random orthogonal matrix)
4. Quantize:      indices = Lloyd-Max(y)  (per-coordinate, optimal for post-rotation distribution)

Compressed: indices (uint8) + norm (fp16) = 514 bytes for K4, 130 bytes for V2
```

After rotation, coordinates follow a Beta distribution (Gaussian for d >= 64). Lloyd-Max quantization is optimal for this distribution at each bit-width.

### Cache Layout

```
Full-attention layer KV cache:

[Sinks (bf16)]  [Decompressed buffer (bf16)]  [Recent (bf16)]
 first 4 tokens   compressed tokens written     last 64 tokens
 always bf16       once at compression time      current window

Compression triggers when recent buffer overflows (every ~64 tokens).
get_full_kv() is just a tensor slice — zero overhead per decode step.
```

### Compression Results

| Setting | Keys | Values | Recent Window | Context | Compression |
|---------|------|--------|---------------|---------|-------------|
| Baseline | 16-bit | 16-bit | all | 4K | 1.0x |
| TurboQuant | 4-bit | 2-bit | 64 tokens | **16K** | ~3x on KV |

---

## API Server & Tool Calling

`server.py` implements an Anthropic Messages API compatible server that translates between Anthropic's tool-calling format and Gemma 4's native `<|tool_call>` format.

### Translation Layer

```
Anthropic format (what clients send):
  tools: [{"name": "bash", "input_schema": {...}}]
  
Gemma4 native format (what the model sees):
  <|tool>declaration:bash{description:...,parameters:{...}}<tool|>
  
Model outputs:
  <|tool_call>call:bash{command:<|"|>ls /tmp<|"|>}<tool_call|>
  
Server translates back to Anthropic:
  {"type": "tool_use", "name": "bash", "input": {"command": "ls /tmp"}}
```

### Small-Model Optimization

The server includes a compaction layer for the 4B model:
- **48 client tools → 6 core tools**: `bash`, `read_file`, `write_file`, `edit_file`, `glob_search`, `grep_search`
- **550 token system prompt → 100 tokens**: Focused instructions that prevent the model from refusing tool use
- **Native format**: Uses Gemma4's trained `<|tool_call>` special tokens, not prompt engineering

---

## The Bandwidth Wall

LLM inference is **memory-bandwidth bound**, not compute bound. Every token requires reading the entire model from memory:

```
                   Bandwidth    Model Size    Max tok/s    Achieved
RTX 3090 (E4B):    936 GB/s     9.29 GB       101          66.9 (66%)
APU (31B Turbo):   256 GB/s    12.67 GB        20          13.9 (70%)
```

Both platforms hit 66-70% of their theoretical bandwidth limit. The remaining 30% is irreducible overhead from memory controller scheduling, kernel launch latency, and non-matmul operations.

**Lookup decoding is the only technique that breaks this limit** — by converting sequential generation into batch verification, multiple tokens share a single bandwidth pass.

---

## File Structure

```
gemma4-engine/
  engine.py            # E4B core inference engine (457 lines)
  engine_turbo.py      # E4B + TurboQuant KV compression (781 lines)
  server.py            # E4B Anthropic API translation server (504 lines)
  benchmark.py         # E4B benchmark suite
  requirements.txt     # Python dependencies (E4B)
  apu/
    create_turbo_quant.sh   # Build mixed-precision GGUF for 31B
    gemma_server.sh         # API server startup script
    gemma_chat.sh           # Interactive chat with lookup decoding
    gemma-server.service    # systemd unit for auto-start
```

---

## Cost Comparison

| Metric | Claude API (Opus) | Local Gemma4 E4B | Local Gemma4 31B |
|--------|-------------------|-----------------|-----------------|
| 50-turn coding session | $3.38 | $0.02 (electricity) | $0.02 (electricity) |
| First token latency | ~500ms (network) | ~15ms | ~72ms |
| Generation speed | ~20-30 tok/s | ~67 tok/s | ~49 tok/s (lookup) |
| Privacy | Data sent to cloud | Fully local | Fully local |
| Context limit | 200K | 16K (TurboQuant) | 4K |
| Model quality | Frontier | 4B (lightweight) | 31B (strong) |

---

## License

MIT

## Acknowledgments

- [Google DeepMind](https://deepmind.google/) for the Gemma 4 model family
- [llama.cpp](https://github.com/ggml-org/llama.cpp) for the GGUF quantization and inference runtime
- TurboQuant KV compression based on Lloyd-Max quantization with random orthogonal rotation

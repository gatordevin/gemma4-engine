# Gemma 4 E4B — Bare-Metal Inference Engine

**3.8x faster than HuggingFace on a single RTX 3090. 1,742 lines of Python. No custom CUDA.**

A from-scratch inference engine for Google's [Gemma 4 E4B](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/) (4B parameter model) that bypasses all framework overhead and hits 66% of the GPU's theoretical memory bandwidth limit.

```
HuggingFace baseline:   17.5 tok/s  (1.0x)
Custom engine:           22.6 tok/s  (1.3x)
+ torch.compile:         66.9 tok/s  (3.8x)  <-- peak
+ TurboQuant (16K ctx):  22.7 tok/s  (1.3x)  + 4x context
Theoretical max:        101.0 tok/s  (RTX 3090 bandwidth limit)
```

## Quick Start

```bash
# Install
git clone https://github.com/gatordevin/gemma4-engine.git
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

## System Requirements

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

### Weight Distribution

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

## TurboQuant KV-Cache Compression

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

## Bandwidth Analysis

LLM inference is **memory-bandwidth bound**, not compute bound. Every token requires reading the entire model from VRAM:

```
RTX 3090:  936 GB/s bandwidth,  35.6 TFLOPS bf16 compute
Model:     9.29 GB weights read per token
Compute:   ~0.5 GFLOP per token (for batch=1)

Time to read weights:  9.29 GB / 936 GB/s  = 9.9 ms  → 101 tok/s max
Time to compute:       0.5 GFLOP / 35.6 TFLOPS = 0.01 ms  (negligible)

We achieve: 14.9 ms per token → 66.9 tok/s → 66% of bandwidth limit
```

The remaining 34% is irreducible overhead: CUDA kernel launch latency, Python runtime, memory controller inefficiency, and torch.compile's generated code not perfectly saturating bandwidth.

---

## Cost Comparison

| Metric | Claude API (Opus) | Local Gemma4 |
|--------|-------------------|-------------|
| 50-turn coding session | $3.38 | $0.02 (electricity) |
| First token latency | ~500ms (network) | ~15ms |
| Generation speed | ~20-30 tok/s | ~67 tok/s |
| Privacy | Data sent to cloud | Fully local |
| Context limit | 200K | 16K (TurboQuant) |

---

## File Structure

```
gemma4-engine/
  engine.py          # Core inference engine (457 lines)
  engine_turbo.py    # + TurboQuant KV compression (781 lines)
  server.py          # Anthropic API translation server (504 lines)
  benchmark.py       # Benchmark suite
  requirements.txt   # Python dependencies
  README.md          # This file
```

Total: **1,742 lines of Python**. No C++, no custom CUDA kernels — pure PyTorch.

---

## License

MIT

## Acknowledgments

- [Google DeepMind](https://deepmind.google/) for the Gemma 4 model family
- [Claw Code](https://github.com/instructkr/claw-code) for the open-source Claude Code implementation
- TurboQuant KV compression based on Lloyd-Max quantization with random orthogonal rotation

"""
Gemma 4 E4B — Custom bare-metal inference engine v3.
Bypasses all HuggingFace overhead. Direct weight loading + manual forward pass.

Architecture: 42 layers (35 sliding + 7 full attention), GQA 4:1, per-layer input gating.
KV sharing: layers 24-41 reuse KV from layers 22 (sliding) and 23 (full).

Optimizations v3:
  - Static pre-allocated KV cache
  - Fused gate+up MLP and K+V projections (fewer kernel launches)
  - Pre-resolved weight tensors (zero dict lookups in hot loop)
  - Unrolled layer config (no string formatting in decode)
  - torch.compile with max-autotune

Usage:
  python gemma4_fast.py                  # ~21 tok/s, instant start
  python gemma4_fast.py --compile        # ~70+ tok/s, ~100s warmup
"""

import torch
import torch.nn.functional as F
import time
import sys
from pathlib import Path
from safetensors import safe_open
from transformers import AutoTokenizer

# ── Config ──────────────────────────────────────────────────────────────────────
DEVICE = "cuda"
DTYPE = torch.bfloat16
MODEL_ID = "google/gemma-4-E4B-it"

NUM_LAYERS = 42
HIDDEN = 2560
INTERMEDIATE = 10240
NUM_HEADS = 8
NUM_KV_HEADS = 2
GQA_GROUPS = NUM_HEADS // NUM_KV_HEADS  # 4
HEAD_DIM_SLIDING = 256
HEAD_DIM_FULL = 512
VOCAB_SIZE = 262144
HIDDEN_PER_LAYER = 256
SLIDING_WINDOW = 512
SOFTCAP = 30.0
RMS_EPS = 1e-6
NUM_KV_SHARED = 18
FIRST_SHARED = NUM_LAYERS - NUM_KV_SHARED  # 24
EMBED_SCALE = HIDDEN ** 0.5
PLI_EMBED_SCALE = HIDDEN_PER_LAYER ** 0.5
PLI_PROJ_SCALE = HIDDEN ** -0.5
PLI_COMBINE_SCALE = 2.0 ** -0.5

IS_SLIDING = [lt == "sliding" for lt in [
    "sliding","sliding","sliding","sliding","sliding","full",
    "sliding","sliding","sliding","sliding","sliding","full",
    "sliding","sliding","sliding","sliding","sliding","full",
    "sliding","sliding","sliding","sliding","sliding","full",
    "sliding","sliding","sliding","sliding","sliding","full",
    "sliding","sliding","sliding","sliding","sliding","full",
    "sliding","sliding","sliding","sliding","sliding","full",
]]

# KV sharing
KV_SHARE_MAP = {}
prev = IS_SLIDING[:FIRST_SHARED]
for i in range(FIRST_SHARED, NUM_LAYERS):
    KV_SHARE_MAP[i] = len(prev) - 1 - prev[::-1].index(IS_SLIDING[i])

STORE_FULL_KV = set()
for i in range(FIRST_SHARED):
    last = len(prev) - 1 - prev[::-1].index(IS_SLIDING[i])
    if i == last:
        STORE_FULL_KV.add(i)


# ── Ops ─────────────────────────────────────────────────────────────────────────

def rms_norm(x, weight):
    x_f = x.float()
    normed = x_f * torch.pow(x_f.pow(2).mean(-1, keepdim=True) + RMS_EPS, -0.5)
    return (normed * weight.float()).to(x.dtype)


def rms_norm_no_weight(x):
    x_f = x.float()
    return (x_f * torch.pow(x_f.pow(2).mean(-1, keepdim=True) + RMS_EPS, -0.5)).to(x.dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)


# ── Pre-resolved Layer Weights ─────────────────────────────────────────────────

class LayerWeights:
    """All weight tensors for one transformer layer, pre-resolved."""
    __slots__ = [
        'idx', 'is_sliding', 'is_shared', 'head_dim', 'kv_share_src',
        'q_proj', 'kv_proj', 'o_proj',
        'gate_up_proj', 'down_proj',
        'input_ln', 'post_attn_ln', 'pre_ff_ln', 'post_ff_ln',
        'q_norm', 'k_norm',
        'pli_gate', 'pli_proj', 'pli_norm',
        'layer_scalar',
    ]


# ── Static KV Cache ─────────────────────────────────────────────────────────────

class StaticKVCache:
    def __init__(self, max_seq_len):
        self.max_seq_len = max_seq_len
        self.seq_len = 0
        self.k = [None] * FIRST_SHARED
        self.v = [None] * FIRST_SHARED
        for i in range(FIRST_SHARED):
            hd = HEAD_DIM_FULL if not IS_SLIDING[i] else HEAD_DIM_SLIDING
            self.k[i] = torch.zeros(1, NUM_KV_HEADS, max_seq_len, hd, dtype=DTYPE, device=DEVICE)
            self.v[i] = torch.zeros(1, NUM_KV_HEADS, max_seq_len, hd, dtype=DTYPE, device=DEVICE)

    def reset(self):
        self.seq_len = 0


# ── Engine ──────────────────────────────────────────────────────────────────────

class Gemma4Engine:
    def __init__(self, model_path=None, max_seq_len=4096):
        self.max_seq_len = max_seq_len

        if model_path is None:
            from huggingface_hub import snapshot_download
            model_path = snapshot_download(MODEL_ID, allow_patterns=["*.safetensors", "*.json"])

        self.model_path = Path(model_path)
        print(f"Loading from {self.model_path}")

        t0 = time.time()
        w = {}
        self._load_weights_raw(w)
        print(f"Weights loaded in {time.time()-t0:.1f}s, VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

        # Pre-resolve into structured layer weights
        self.layers = self._build_layers(w)
        self.embed_tokens = w["embed_tokens.weight"]
        self.embed_pli = w["embed_tokens_per_layer.weight"]
        self.pli_model_proj = w["per_layer_model_projection.weight"]
        self.pli_proj_norm = w["per_layer_projection_norm.weight"]
        self.final_norm = w["norm.weight"]

        del w  # Free the dict
        print(f"After setup: VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

        self._build_rope()
        self.cache = StaticKVCache(max_seq_len)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self._compiled_decode = None

    def _load_weights_raw(self, w):
        st_files = sorted(self.model_path.glob("*.safetensors"))
        for sf in st_files:
            with safe_open(str(sf), framework="pt", device=str(DEVICE)) as f:
                for key in f.keys():
                    if key.startswith("model.language_model.") or key.startswith("language_model."):
                        short = key.replace("model.language_model.", "").replace("language_model.", "")
                        w[short] = f.get_tensor(key).to(DTYPE)

    def _build_layers(self, w):
        """Pre-resolve all weight references into LayerWeights structs."""
        layers = []
        for i in range(NUM_LAYERS):
            lw = LayerWeights()
            lw.idx = i
            lw.is_sliding = IS_SLIDING[i]
            lw.is_shared = i >= FIRST_SHARED
            lw.head_dim = HEAD_DIM_SLIDING if lw.is_sliding else HEAD_DIM_FULL
            lw.kv_share_src = KV_SHARE_MAP.get(i)

            p = f"layers.{i}."
            ap = p + "self_attn."
            mp = p + "mlp."

            lw.q_proj = w[ap + "q_proj.weight"]
            lw.o_proj = w[ap + "o_proj.weight"]

            # Fused K+V for non-shared layers
            if not lw.is_shared:
                k = w.pop(ap + "k_proj.weight")
                v = w.pop(ap + "v_proj.weight")
                lw.kv_proj = torch.cat([k, v], dim=0)
            else:
                lw.kv_proj = None
                # Remove unused K/V weights for shared layers
                w.pop(ap + "k_proj.weight", None)
                w.pop(ap + "v_proj.weight", None)

            # Fused gate+up MLP
            gate = w.pop(mp + "gate_proj.weight")
            up = w.pop(mp + "up_proj.weight")
            lw.gate_up_proj = torch.cat([gate, up], dim=0)
            lw.down_proj = w[mp + "down_proj.weight"]

            lw.input_ln = w[p + "input_layernorm.weight"]
            lw.post_attn_ln = w[p + "post_attention_layernorm.weight"]
            lw.pre_ff_ln = w[p + "pre_feedforward_layernorm.weight"]
            lw.post_ff_ln = w[p + "post_feedforward_layernorm.weight"]
            lw.q_norm = w[ap + "q_norm.weight"]
            lw.k_norm = w.get(ap + "k_norm.weight")

            lw.pli_gate = w[p + "per_layer_input_gate.weight"]
            lw.pli_proj = w[p + "per_layer_projection.weight"]
            lw.pli_norm = w[p + "post_per_layer_input_norm.weight"]
            lw.layer_scalar = w[p + "layer_scalar"]

            layers.append(lw)
        return layers

    def _build_rope(self):
        positions = torch.arange(self.max_seq_len, device=DEVICE, dtype=torch.float32)

        dim_s = HEAD_DIM_SLIDING
        inv_freq_s = 1.0 / (10000.0 ** (torch.arange(0, dim_s, 2, device=DEVICE, dtype=torch.float32) / dim_s))
        emb_s = torch.cat([torch.outer(positions, inv_freq_s)] * 2, dim=-1)
        self.cos_s = emb_s.cos().to(DTYPE)
        self.sin_s = emb_s.sin().to(DTYPE)

        dim_f = HEAD_DIM_FULL
        ra = int(0.25 * dim_f // 2)
        na = dim_f // 2 - ra
        inv_freq_r = 1.0 / (1e6 ** (torch.arange(0, 2*ra, 2, device=DEVICE, dtype=torch.float32) / dim_f))
        inv_freq_f = torch.cat([inv_freq_r, torch.zeros(na, device=DEVICE, dtype=torch.float32)])
        emb_f = torch.cat([torch.outer(positions, inv_freq_f)] * 2, dim=-1)
        self.cos_f = emb_f.cos().to(DTYPE)
        self.sin_f = emb_f.sin().to(DTYPE)

    def _run_layer(self, hidden, lw, cos, sin, pli, seq_end):
        """Forward one layer. All weight tensors are pre-resolved in lw."""
        B = hidden.shape[0]
        hd = lw.head_dim

        # --- Attention ---
        residual = hidden
        h = rms_norm(hidden, lw.input_ln)

        # Q
        q = F.linear(h, lw.q_proj).view(B, -1, NUM_HEADS, hd)
        q = rms_norm(q, lw.q_norm)
        q = apply_rope(q, cos, sin)
        q = q.transpose(1, 2)

        if lw.is_shared:
            src = lw.kv_share_src
            k = self.cache.k[src][:, :, :seq_end]
            v = self.cache.v[src][:, :, :seq_end]
        else:
            # Fused K+V
            kv = F.linear(h, lw.kv_proj)
            kv_dim = NUM_KV_HEADS * hd
            k_new = kv[..., :kv_dim].reshape(B, -1, NUM_KV_HEADS, hd)
            v_new = kv[..., kv_dim:].reshape(B, -1, NUM_KV_HEADS, hd)

            k_new = rms_norm(k_new, lw.k_norm)
            k_new = apply_rope(k_new, cos, sin)
            k_new = k_new.transpose(1, 2)

            v_new = rms_norm_no_weight(v_new)
            v_new = v_new.transpose(1, 2)

            layer_idx = lw.idx
            S = k_new.shape[2]
            sl = self.cache.seq_len
            self.cache.k[layer_idx][:, :, sl:sl+S] = k_new
            self.cache.v[layer_idx][:, :, sl:sl+S] = v_new

            k = self.cache.k[layer_idx][:, :, :seq_end]
            v = self.cache.v[layer_idx][:, :, :seq_end]

        # GQA expand + SDPA
        k = k.repeat_interleave(GQA_GROUPS, dim=1)
        v = v.repeat_interleave(GQA_GROUPS, dim=1)
        if lw.is_sliding and k.shape[2] > SLIDING_WINDOW:
            k = k[:, :, -SLIDING_WINDOW:]
            v = v[:, :, -SLIDING_WINDOW:]

        S = q.shape[2]
        a = F.scaled_dot_product_attention(q, k, v, is_causal=(S > 1), scale=1.0)
        a = a.transpose(1, 2).reshape(B, S, -1)
        a = F.linear(a, lw.o_proj)

        hidden = residual + rms_norm(a, lw.post_attn_ln)

        # --- MLP ---
        residual = hidden
        h = rms_norm(hidden, lw.pre_ff_ln)
        gu = F.linear(h, lw.gate_up_proj)
        gate, up = gu.chunk(2, dim=-1)
        m = F.linear(F.gelu(gate, approximate="tanh") * up, lw.down_proj)
        hidden = residual + rms_norm(m, lw.post_ff_ln)

        # --- Per-layer input gate ---
        res = hidden
        g = F.gelu(F.linear(hidden, lw.pli_gate), approximate="tanh") * pli
        hidden = res + rms_norm(F.linear(g, lw.pli_proj), lw.pli_norm)

        return hidden * lw.layer_scalar

    def forward(self, input_ids, pos_ids):
        B, S = input_ids.shape
        seq_start = self.cache.seq_len
        seq_end = seq_start + S

        # Embeddings
        hidden = F.embedding(input_ids, self.embed_tokens) * EMBED_SCALE

        # Per-layer inputs
        pli_raw = F.embedding(input_ids, self.embed_pli) * PLI_EMBED_SCALE
        pli_raw = pli_raw.reshape(B, S, NUM_LAYERS, HIDDEN_PER_LAYER)
        proj = F.linear(hidden, self.pli_model_proj) * PLI_PROJ_SCALE
        proj = proj.reshape(B, S, NUM_LAYERS, HIDDEN_PER_LAYER)
        proj = rms_norm(proj, self.pli_proj_norm)
        pli_all = (proj + pli_raw) * PLI_COMBINE_SCALE

        # RoPE
        pf = pos_ids[0]
        cos_s = self.cos_s[pf].unsqueeze(0).unsqueeze(2)  # [1, S, 1, hd]
        sin_s = self.sin_s[pf].unsqueeze(0).unsqueeze(2)
        cos_f = self.cos_f[pf].unsqueeze(0).unsqueeze(2)
        sin_f = self.sin_f[pf].unsqueeze(0).unsqueeze(2)

        # Run all layers
        for i in range(NUM_LAYERS):
            lw = self.layers[i]
            cos = cos_s if lw.is_sliding else cos_f
            sin = sin_s if lw.is_sliding else sin_f
            hidden = self._run_layer(hidden, lw, cos, sin, pli_all[:, :, i, :], seq_end)

        self.cache.seq_len = seq_end

        # LM head
        hidden = rms_norm(hidden, self.final_norm)
        logits = F.linear(hidden, self.embed_tokens)
        return torch.tanh(logits / SOFTCAP) * SOFTCAP

    def decode_one_token(self, token_id, pos_id):
        return self.forward(token_id.view(1, 1), pos_id.view(1, 1)).squeeze(0)

    def get_compiled_decode(self):
        if self._compiled_decode is None:
            print("Compiling decode step...")
            self._compiled_decode = torch.compile(
                self.decode_one_token,
                mode="default",
                fullgraph=False,
            )
        return self._compiled_decode

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=256, temperature=0.7, top_k=50,
                 use_compile=False):
        self.cache.reset()

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(DEVICE)
        B, S = input_ids.shape

        pos_ids = torch.arange(S, device=DEVICE).unsqueeze(0)
        logits = self.forward(input_ids, pos_ids)

        decode_fn = self.get_compiled_decode() if use_compile else self.decode_one_token

        next_logits = logits[:, -1, :]
        token_ids = []

        t0 = time.time()
        for _ in range(max_new_tokens):
            if temperature > 0:
                probs = F.softmax(next_logits / temperature, dim=-1)
                if top_k > 0:
                    v, idx = torch.topk(probs, top_k)
                    probs = torch.zeros_like(probs).scatter_(1, idx, v)
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                next_token = torch.multinomial(probs, 1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            tid = next_token.item()
            token_ids.append(tid)

            if tid in (1, 106):
                break

            next_logits = decode_fn(
                torch.tensor([tid], device=DEVICE),
                torch.tensor([self.cache.seq_len], device=DEVICE),
            )

        dt = time.time() - t0
        return self.tokenizer.decode(token_ids, skip_special_tokens=True), len(token_ids), dt


def main():
    use_compile = "--compile" in sys.argv

    print("=== Gemma 4 E4B Custom Engine v3 ===")
    engine = Gemma4Engine()

    if use_compile:
        print("\nCompile warmup (~100s first run)...")
        _, n, t = engine.generate("Hi", max_new_tokens=32, use_compile=True)
        print(f"  warmup1 (compile): {n} tok, {t:.1f}s, {n/t:.1f} tok/s")
        _, n, t = engine.generate("Hello there", max_new_tokens=64, use_compile=True)
        print(f"  warmup2: {n} tok, {t:.1f}s, {n/t:.1f} tok/s")
        _, n, t = engine.generate("Hey", max_new_tokens=32, use_compile=True)
        print(f"  warmup3: {n} tok, {t:.1f}s, {n/t:.1f} tok/s")
    else:
        print("\nWarmup...")
        _, n, t = engine.generate("Hi", max_new_tokens=32)
        print(f"  warmup: {n} tok, {t:.1f}s, {n/t:.1f} tok/s")

    print("\n--- Benchmark ---")
    text, n, t = engine.generate("What are you? Respond in 2 sentences.",
                                  max_new_tokens=256, temperature=0.0,
                                  use_compile=use_compile)
    print(f"{n} tokens, {t:.1f}s, {n/t:.1f} tok/s")
    print(text)

    text, n, t = engine.generate("Explain quantum computing in simple terms.",
                                  max_new_tokens=256, temperature=0.0,
                                  use_compile=use_compile)
    print(f"\n{n} tokens, {t:.1f}s, {n/t:.1f} tok/s")

    mode = "compiled max-autotune" if use_compile else "uncompiled"
    print(f"\n{'='*60}\nGemma 4 E4B ({mode})\nType prompts (Ctrl+C to exit):\n{'='*60}")
    while True:
        try:
            prompt = input("\n> ")
            if not prompt.strip():
                continue
            text, n, t = engine.generate(prompt, max_new_tokens=1024, use_compile=use_compile)
            print(f"\n[{n} tok, {t:.1f}s, {n/t:.1f} tok/s]")
            print(text)
        except KeyboardInterrupt:
            print("\nBye!")
            break


if __name__ == "__main__":
    main()

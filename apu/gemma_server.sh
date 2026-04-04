#!/bin/bash
# Gemma 4 31B Optimized Server for AMD APU
# Hardware: AMD Ryzen AI MAX+ 395 (gfx1151), 64GB LPDDR5-8000
#
# TURBO quant: FFN at q2_K, Attention at q4_K, Embeddings at q6_K
# 30% faster than stock Q4_K_M with attention quality preserved
#
# Speed: ~14 t/s generation, ~60 t/s prompt eval
# For interactive use with 4x speed (lookup decoding), use gemma_chat.sh

set -e

LLAMA_CPP="${LLAMA_CPP:-$HOME/llama.cpp}"
MODEL="${MODEL:-$HOME/models/gemma-4-31B-it-TURBO.gguf}"
PORT="${PORT:-8082}"

exec "${LLAMA_CPP}/build/bin/llama-server" \
  -m "$MODEL" \
  -ngl 99 \
  -t 16 \
  -c 4096 \
  --reasoning off \
  --parallel 1 \
  --port "$PORT" \
  --host 0.0.0.0

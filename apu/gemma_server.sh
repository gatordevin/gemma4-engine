#!/bin/bash
# Gemma 4 31B Optimized Server for AMD APU
# Hardware: AMD Ryzen AI MAX+ 395 (gfx1151), 64GB LPDDR5-8000
#
# Default: A4B TurboQuant (MoE, 128 experts, 8 active per token)
# Speed: ~65 t/s generation, ~166 t/s prompt eval, ~230 t/s with lookup decoding
# Vision: auto-enabled if mmproj file exists
# For interactive use with lookup decoding (~230 t/s), use gemma_chat.sh

set -e

LLAMA_CPP="${LLAMA_CPP:-$HOME/llama.cpp}"
MODEL="${MODEL:-$HOME/models/gemma-4-26B-A4B-it-TURBO.gguf}"
MMPROJ="${MMPROJ:-$HOME/models/mmproj-gemma-4-26B-A4B-f16.gguf}"
PORT="${PORT:-8082}"

# Vision support: add --mmproj if projector exists
VISION_FLAG=""
if [ -f "$MMPROJ" ]; then
  VISION_FLAG="--mmproj $MMPROJ"
fi

exec "${LLAMA_CPP}/build/bin/llama-server" \
  -m "$MODEL" \
  $VISION_FLAG \
  -ngl 99 \
  -t 16 \
  -c 4096 \
  --parallel 1 \
  --port "$PORT" \
  --host 0.0.0.0

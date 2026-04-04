#!/bin/bash
# Gemma 4 31B Interactive Chat — Lookup Decoding Mode
# ~230 t/s effective with A4B TurboQuant + lookup decoding
#
# Uses n-gram self-speculation: predicts next tokens from patterns
# in the current generation and batch-verifies them at prompt-eval speed.
# Lookup cache persists between sessions for faster warmup.

set -e

LLAMA_CPP="${LLAMA_CPP:-$HOME/llama.cpp}"
MODEL="${MODEL:-$HOME/models/gemma-4-26B-A4B-it-TURBO.gguf}"

exec "${LLAMA_CPP}/build/bin/llama-lookup" \
  -m "$MODEL" \
  -ngl 99 \
  -t 16 \
  -c 4096 \
  --temp 0.7 \
  -lcd "$HOME/models/lookup_cache_dynamic.bin" \
  --draft 16 \
  --conversation \
  "$@"

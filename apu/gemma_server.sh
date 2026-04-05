#!/bin/bash
# Gemma 4 A4B Server — optimized for multi-turn chat
# 52 t/s consistent generation for diverse conversations
# Ngram speculation available via --spec-type ngram-cache for repetitive workloads

LLAMA_CPP="${LLAMA_CPP:-$HOME/llama.cpp}"
MODEL="${MODEL:-$HOME/models/gemma-4-26B-A4B-it-Q4_K_M.gguf}"
PORT="${PORT:-8082}"

# Use SPECULATE=1 to enable ngram speculation (best for repetitive code gen)
if [ "${SPECULATE:-0}" = "1" ]; then
  SPEC_FLAGS="--spec-type ngram-cache -lcd ${LLAMA_CPP}/../models/ngram_server_cache.bin --draft 16"
else
  SPEC_FLAGS=""
fi

exec "${LLAMA_CPP}/build/bin/llama-server" \
  -m "$MODEL" \
  -ngl 99 \
  -t 16 \
  -c 4096 \
  $SPEC_FLAGS \
  --parallel 1 \
  --port "$PORT" \
  --host 0.0.0.0

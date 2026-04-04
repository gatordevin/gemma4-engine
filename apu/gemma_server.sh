#!/bin/bash
# Gemma 4 A4B Server — ngram speculation + native tool calling
# Requires patched llama.cpp (see llama-cpp-ngram-fix.patch)
#
# The ngram cache warms up over the session. Acceptance rate climbs
# from ~10% (cold) to 40%+ (warm), accelerating batch verification.

LLAMA_CPP="${LLAMA_CPP:-$HOME/llama.cpp}"
MODEL="${MODEL:-$HOME/models/gemma-4-26B-A4B-it-Q4_K_M.gguf}"
NGRAM_CACHE="${NGRAM_CACHE:-$HOME/models/ngram_server_cache.bin}"
PORT="${PORT:-8082}"

touch "$NGRAM_CACHE"

exec "${LLAMA_CPP}/build/bin/llama-server" \
  -m "$MODEL" \
  -ngl 99 \
  -t 16 \
  -c 4096 \
  --spec-type ngram-cache \
  -lcd "$NGRAM_CACHE" \
  --draft 16 \
  --parallel 1 \
  --port "$PORT" \
  --host 0.0.0.0

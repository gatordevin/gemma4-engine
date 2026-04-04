#!/bin/bash
# Create TurboQuant: mixed-precision GGUF for Gemma 4 31B
#
# Strategy: FFN layers consume 67.5% of memory bandwidth but are less
# quality-sensitive than attention layers. By crushing FFN to q2_K while
# keeping attention at q4_K, we get Q2_K-level speed with Q4_K attention quality.
#
# Result: 3.55 BPW, 13 GB, ~14 t/s (vs 10.7 t/s for stock Q4_K_M)
#
# Requires: llama-quantize binary and a Q4_K_M source model

set -e

LLAMA_CPP="${LLAMA_CPP:-$HOME/llama.cpp}"
QUANTIZE="${LLAMA_CPP}/build/bin/llama-quantize"
SOURCE="${1:-$HOME/models/gemma-4-31B-it-Q4_K_M.gguf}"
OUTPUT="${2:-$HOME/models/gemma-4-31B-it-TURBO.gguf}"

if [ ! -f "$SOURCE" ]; then
    echo "Source model not found: $SOURCE"
    echo "Usage: $0 [source_q4km.gguf] [output_turbo.gguf]"
    exit 1
fi

echo "=== Creating TurboQuant ==="
echo "Source: $SOURCE"
echo "Output: $OUTPUT"
echo ""
echo "Quantization map:"
echo "  FFN gate/up:     q2_K  (crushed — 67% of bandwidth, tolerates compression)"
echo "  FFN down:        q3_K  (slightly higher — most sensitive FFN tensor)"
echo "  Attention Q/K/V/O: q4_K  (preserved — quality-critical)"
echo "  Embeddings:      q6_K  (preserved — small, worth keeping)"
echo ""

"$QUANTIZE" \
  --allow-requantize \
  --tensor-type ffn_gate=q2_K \
  --tensor-type ffn_up=q2_K \
  --tensor-type ffn_down=q3_K \
  "$SOURCE" \
  "$OUTPUT" \
  Q4_K_M 16

echo ""
echo "Done! Output: $OUTPUT"
ls -lh "$OUTPUT"

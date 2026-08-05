#!/usr/bin/env bash
# Download the two ONNX models Krino needs:
#   - NLI:       roberta-large-mnli-static-int8     → models/roberta-large-mnli-static-int8/
#   - Embedding: all-MiniLM-L6-v2-onnx-quantized    → models/all-MiniLM-L6-v2-onnx-quantized/
#
# These paths match the defaults in `krino-api.toml.example`. If you have a
# different path layout, edit your `krino-api.toml` to point at the right
# directories.
#
# Until pre-built quantized weights are published to Hugging Face under the
# Krino organization, this script:
#   1. Downloads the FP32 source models from Hugging Face.
#   2. Calls the export/quantize scripts under `scripts/` to produce the
#      INT8 variants.
#
# Requires: bash, wget, python3 with `uv` (or `pip` + the dependencies in
# `scripts/pyproject.toml`).

set -euo pipefail

MODELS_DIR="${MODELS_DIR:-models}"
mkdir -p "$MODELS_DIR"

echo "Krino model download → $MODELS_DIR"
echo

# 1. Embedding model: sentence-transformers/all-MiniLM-L6-v2
if [[ ! -d "$MODELS_DIR/all-MiniLM-L6-v2-onnx-quantized" ]]; then
    echo "[1/2] Building all-MiniLM-L6-v2 quantized ONNX..."
    cd scripts && uv run export_embedding_onnx.py && cd ..
    echo "    → $MODELS_DIR/all-MiniLM-L6-v2-onnx-quantized"
else
    echo "[1/2] all-MiniLM-L6-v2-onnx-quantized exists, skipping."
fi

# 2. NLI model: roberta-large-mnli (static INT8)
if [[ ! -d "$MODELS_DIR/roberta-large-mnli-static-int8" ]]; then
    echo "[2/2] Building roberta-large-mnli static-INT8 ONNX..."
    cd scripts && uv run export_roberta_mnli_onnx.py && cd ..
    cd scripts && uv run static_quantize_nli.py && cd ..
    echo "    → $MODELS_DIR/roberta-large-mnli-static-int8"
else
    echo "[2/2] roberta-large-mnli-static-int8 exists, skipping."
fi

echo
echo "Done. Verify paths in krino-api.toml point at:"
echo "  models/all-MiniLM-L6-v2-onnx-quantized"
echo "  models/roberta-large-mnli-static-int8"

#!/bin/bash
# Download sentence-transformers/all-MiniLM-L6-v2 for groundedness pre-filtering
#
# This model is required for production use of the groundedness checker.
# Without it, MockEmbedding is used which produces garbage similarity scores,
# forcing top_k_context=0 (disabled pre-filtering) and causing slow NLI calls.

set -e

MODEL_DIR="models/all-MiniLM-L6-v2"

echo "📥 Downloading all-MiniLM-L6-v2 embedding model..."
echo "   Size: ~22MB"
echo "   Target: $MODEL_DIR"
echo

mkdir -p "$MODEL_DIR"

echo "Downloading model.safetensors..."
wget -q --show-progress \
    https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/model.safetensors \
    -O "$MODEL_DIR/model.safetensors"

echo "Downloading tokenizer.json..."
wget -q --show-progress \
    https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json \
    -O "$MODEL_DIR/tokenizer.json"

echo "Downloading config.json..."
wget -q --show-progress \
    https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config.json \
    -O "$MODEL_DIR/config.json"

echo
echo "✅ Model downloaded successfully!"
echo "   Location: $MODEL_DIR"
echo

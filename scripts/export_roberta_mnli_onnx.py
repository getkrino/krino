#!/usr/bin/env python3
"""Export roberta-large-mnli to ONNX format.

Why this model: SummaC-style sentence-to-sentence groundedness (which is what
GroundednessChecker implements) requires a real 3-class MNLI classifier — one
that emits a meaningful *neutral* signal for off-topic context sentences.
MiniCheck (binary supported/unsupported) was tried in PR #33 and failed:
off-topic sentences came back with p_contradiction = 0.8-0.9, overwhelming
the on-topic entailment signal in the verdict tracker. See
project_minicheck_quantization memory.

roberta-large-mnli is RoBERTa-large fine-tuned on MNLI — same architecture as
MiniCheck-RoBERTa-Large (so static INT8 quantization should work cleanly per
the memory's architecture rule), but with the 3-class output SummaC was
designed for.

This script only does the FP32 export. Static INT8 quantization is a separate
step via scripts/static_quantize_nli.py with env-var overrides:

    uv run --with optimum[onnxruntime] --with transformers --with torch \\
        scripts/export_roberta_mnli_onnx.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

MODEL_NAME = "FacebookAI/roberta-large-mnli"
# Path resolves relative to repo root when invoked from there. The static
# quantize script defaults KRINO_FP32_DIR to "../models/..." (it lives in
# scripts/) so this path keeps the two scripts consistent.
OUTPUT_DIR = Path("models/roberta-large-mnli-onnx")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Exporting {MODEL_NAME} → {OUTPUT_DIR}")
    model = ORTModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        export=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    # RoBERTa-large-mnli's canonical id2label is
    # {0: CONTRADICTION, 1: NEUTRAL, 2: ENTAILMENT} — reversed from DeBERTa-MNLI.
    # The Rust backend handles the remap via a label permutation it builds
    # from id2label names at load time, so we just print the order here for
    # diagnostic confirmation, no rewrite needed.
    import json
    cfg = json.loads((OUTPUT_DIR / "config.json").read_text())
    print(f"  id2label = {cfg.get('id2label')}")

    print("✓ Export complete")
    print(f"  model.onnx     : {(OUTPUT_DIR / 'model.onnx').stat().st_size / 1e6:.0f} MB")
    print(f"  tokenizer.json : {(OUTPUT_DIR / 'tokenizer.json').stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())

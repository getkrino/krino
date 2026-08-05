#!/usr/bin/env python3
"""
MiniCheck FP32 smoke test.

Runs the exported MiniCheck-DeBERTa-v3-Large ONNX model on the Android example
from the Krino demo UI and prints per-(context, claim) support scores. The
purpose is to validate Phase 1 of the MiniCheck pivot:

  - Does the model load and run end-to-end via plain onnxruntime?
  - On the Android example that DistilBERT-NLI gets wrong (claim:
    "first commercial Android device was the iPhone"), does MiniCheck
    correctly mark the iPhone claim as unsupported?
  - Are the probability distributions calibrated (high confidence on the
    obvious entailments and contradictions)?

If yes, we proceed to Phase 2 (wire into Rust). If no, we stop here and
investigate before touching any production code.

Usage:
    uv run --with onnxruntime --with transformers --with numpy \\
        scripts/minicheck_smoke.py

The model is loaded from ../models/minicheck-deberta-v3-large-onnx/.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import os
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

# Override with MINICHECK_MODEL_DIR to point at a different MiniCheck variant
# (e.g. RoBERTa, quantized builds). Defaults to the DeBERTa FP32 export.
_default = Path(__file__).resolve().parent.parent / "models" / "minicheck-deberta-v3-large-onnx"
MODEL_DIR = Path(os.environ.get("MINICHECK_MODEL_DIR", str(_default)))
MAX_SEQ_LENGTH = 512  # MiniCheck supports up to 2048 but our context fits in 512

# The Android example from krino-ui/src/components/eval_pane.rs:368-370
CONTEXT = (
    "Android is a mobile operating system based on a modified version of the Linux kernel "
    "and other open-source software, designed primarily for touchscreen mobile devices such "
    "as smartphones and tablets. "
    "Android is developed by a consortium of developers known as the Open Handset Alliance, "
    "though its most widely used version is primarily developed by Google. "
    "It was unveiled in November 2007, with the first commercial Android device, the HTC Dream, "
    "launched in September 2008. "
    "Most versions of Android are proprietary, although some components are derived from and "
    "released under open-source licenses such as the Apache License. "
    "The source code has been used to develop variants of Android on a range of other electronics. "
    "As of 2024, Android has the largest installed base of any operating system in the world, "
    "with over 3 billion monthly active users. "
    "Android applications are typically written in Kotlin or Java and run on the Android Runtime."
)

# Each claim from the demo's LLM output, with expected MiniCheck verdict.
# "supported" means MiniCheck should return p_supported > 0.5.
CLAIMS = [
    (
        "Android is a mobile operating system built on top of the Linux kernel.",
        "supported",
        "paraphrase of context — distilbert wrongly contradicts at 81%",
    ),
    (
        "It was first unveiled in November 2007.",
        "supported",
        "verbatim entity match",
    ),
    (
        "The first commercial Android device was the iPhone, released in September 2008.",
        "unsupported",
        "FAILURE MODE: context says HTC Dream, not iPhone — distilbert misses this",
    ),
    (
        "Android is developed entirely by Apple Inc. and uses the Swift programming language.",
        "unsupported",
        "directly contradicted by context (Google / Kotlin/Java)",
    ),
    (
        "As of 2024, Android has over 3 billion monthly active users.",
        "supported",
        "verbatim statistic",
    ),
    (
        "Android apps can be written in Kotlin.",
        "supported",
        "verbatim statement",
    ),
]


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along the last axis."""
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def main() -> int:
    if not MODEL_DIR.exists():
        print(f"ERROR: model dir not found: {MODEL_DIR}", file=sys.stderr)
        print(
            "Run the optimum-cli export first — see scripts/minicheck_smoke.py "
            "module docstring for the command.",
            file=sys.stderr,
        )
        return 1

    print(f"Loading tokenizer from {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))

    print(f"Loading ONNX session...")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    model_path = MODEL_DIR / "model_quantized.onnx"
    if not model_path.exists():
        model_path = MODEL_DIR / "model.onnx"
    print(f"Loading model: {model_path.name}")
    sess = ort.InferenceSession(str(model_path), opts)
    input_names = {inp.name for inp in sess.get_inputs()}
    print(f"  inputs: {sorted(input_names)}")
    print(f"  outputs: {[out.name for out in sess.get_outputs()]}")

    print()
    print("=" * 100)
    print(f"{'Claim':<70} {'p_sup':>8} {'verdict':>12} {'expected':>12}")
    print("=" * 100)

    latencies: list[float] = []
    mismatches = 0
    confident_correct = 0

    for claim_text, expected, note in CLAIMS:
        encoded = tokenizer(
            CONTEXT,
            claim_text,
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="np",
        )
        feed = {k: v.astype(np.int64) for k, v in encoded.items() if k in input_names}

        t0 = time.perf_counter()
        logits = sess.run(None, feed)[0]
        latencies.append((time.perf_counter() - t0) * 1000)

        probs = softmax(logits)[0]
        # MiniCheck label space: 0 = unsupported, 1 = supported
        p_unsupported, p_supported = float(probs[0]), float(probs[1])
        verdict = "supported" if p_supported > 0.5 else "unsupported"

        match_marker = "OK" if verdict == expected else "FAIL"
        if verdict != expected:
            mismatches += 1
        elif max(p_supported, p_unsupported) > 0.8:
            confident_correct += 1

        truncated = claim_text if len(claim_text) <= 66 else claim_text[:63] + "..."
        print(
            f"{truncated:<70} {p_supported:>8.3f} {verdict:>12} {expected:>12}  [{match_marker}]"
        )
        print(f"  └─ {note}")

    print()
    print("=" * 100)
    print(f"Correct: {len(CLAIMS) - mismatches} / {len(CLAIMS)}")
    print(f"Confident-correct (>0.8 on right verdict): {confident_correct} / {len(CLAIMS)}")
    print(
        f"Latency per claim — mean={np.mean(latencies):.1f}ms  "
        f"p50={np.percentile(latencies, 50):.1f}ms  "
        f"p95={np.percentile(latencies, 95):.1f}ms"
    )
    print()

    if mismatches > 0:
        print(f"⚠  {mismatches} claim(s) got the wrong verdict — see [FAIL] rows above.")
        return 2
    print("All claims verdicted correctly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

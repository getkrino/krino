#!/usr/bin/env python3
"""
MiniCheck FP32 smoke test — c8a.2xlarge variant.

Designed to run on the production EC2 instance via SSM. Differences from the local
smoke test:

  - 8 intra-op threads (matches c8a.2xlarge's 8 vCPUs)
  - Model path is /tmp/minicheck-smoke/model/ (where the SSM bootstrap
    drops the synced S3 contents)
  - Warm-up pass before timing (kernels JIT, threadpool spins up — first
    call's latency is not representative)
  - Per-call breakdown of tokenize vs. session-run so we see where the
    time actually goes

The Android example is identical to scripts/minicheck_smoke.py — same
context, same six claims, same expected verdicts. We want a directly
comparable apples-to-apples latency reading.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

MODEL_DIR = Path("/tmp/minicheck-smoke/model")
MAX_SEQ_LENGTH = 512
INTRA_OP_THREADS = 8  # match c8a.2xlarge core count

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

CLAIMS = [
    ("Android is a mobile operating system built on top of the Linux kernel.", "supported"),
    ("It was first unveiled in November 2007.", "supported"),
    ("The first commercial Android device was the iPhone, released in September 2008.", "unsupported"),
    ("Android is developed entirely by Apple Inc. and uses the Swift programming language.", "unsupported"),
    ("As of 2024, Android has over 3 billion monthly active users.", "supported"),
    ("Android apps can be written in Kotlin.", "supported"),
]


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def cpu_info() -> str:
    """Best-effort CPU summary for the report header."""
    try:
        with open("/proc/cpuinfo") as f:
            data = f.read()
        cores = data.count("processor\t:")
        model = next(
            (line.split(":", 1)[1].strip() for line in data.splitlines() if line.startswith("model name")),
            "unknown",
        )
        flags_line = next(
            (line.split(":", 1)[1] for line in data.splitlines() if line.startswith("flags")),
            "",
        )
        flags = set(flags_line.split())
        avx_summary = ", ".join(
            f for f in ("avx2", "avx512f", "avx512vnni", "fma") if f in flags
        ) or "none"
        return f"{cores}x {model} [{avx_summary}]"
    except OSError:
        return "unknown"


def main() -> int:
    print(f"CPU: {cpu_info()}")
    print(f"ORT version: {ort.__version__}")
    print(f"Model dir: {MODEL_DIR}")
    print()

    if not MODEL_DIR.exists():
        print(f"ERROR: model dir not found: {MODEL_DIR}", file=sys.stderr)
        return 1

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    print(f"Tokenizer loaded in {(time.perf_counter() - t0) * 1000:.0f}ms")

    t0 = time.perf_counter()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = INTRA_OP_THREADS
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # Quantized builds use model_quantized.onnx; FP32 uses model.onnx.
    model_path = MODEL_DIR / "model_quantized.onnx"
    if not model_path.exists():
        model_path = MODEL_DIR / "model.onnx"
    print(f"Loading model: {model_path.name}")
    sess = ort.InferenceSession(str(model_path), opts)
    input_names = {inp.name for inp in sess.get_inputs()}
    print(f"Session loaded in {(time.perf_counter() - t0) * 1000:.0f}ms  inputs={sorted(input_names)}")
    print(f"intra_op_num_threads={INTRA_OP_THREADS}")
    print()

    # Warm-up pass — JIT, kernel selection, threadpool init.
    warm_in = tokenizer(
        CONTEXT, CLAIMS[0][0],
        padding="max_length", truncation=True,
        max_length=MAX_SEQ_LENGTH, return_tensors="np",
    )
    warm_feed = {k: v.astype(np.int64) for k, v in warm_in.items() if k in input_names}
    t0 = time.perf_counter()
    sess.run(None, warm_feed)
    warmup_ms = (time.perf_counter() - t0) * 1000
    print(f"Warm-up call: {warmup_ms:.1f}ms (excluded from stats)")
    print()

    print("=" * 100)
    print(f"{'Claim':<70} {'p_sup':>8} {'tok_ms':>8} {'run_ms':>8}")
    print("=" * 100)

    tok_latencies: list[float] = []
    run_latencies: list[float] = []
    mismatches = 0

    for claim_text, expected in CLAIMS:
        t0 = time.perf_counter()
        encoded = tokenizer(
            CONTEXT, claim_text,
            padding="max_length", truncation=True,
            max_length=MAX_SEQ_LENGTH, return_tensors="np",
        )
        feed = {k: v.astype(np.int64) for k, v in encoded.items() if k in input_names}
        tok_ms = (time.perf_counter() - t0) * 1000
        tok_latencies.append(tok_ms)

        t0 = time.perf_counter()
        logits = sess.run(None, feed)[0]
        run_ms = (time.perf_counter() - t0) * 1000
        run_latencies.append(run_ms)

        probs = softmax(logits)[0]
        p_supported = float(probs[1])
        verdict = "supported" if p_supported > 0.5 else "unsupported"
        if verdict != expected:
            mismatches += 1

        truncated = claim_text if len(claim_text) <= 66 else claim_text[:63] + "..."
        print(f"{truncated:<70} {p_supported:>8.3f} {tok_ms:>8.1f} {run_ms:>8.1f}")

    print()
    print("=" * 100)
    print(f"Correct: {len(CLAIMS) - mismatches} / {len(CLAIMS)}")
    print()
    print("Latency stats (ms, excludes warm-up):")
    print(
        f"  tokenize:  mean={np.mean(tok_latencies):6.1f}  "
        f"p50={np.percentile(tok_latencies, 50):6.1f}  "
        f"p95={np.percentile(tok_latencies, 95):6.1f}"
    )
    print(
        f"  session:   mean={np.mean(run_latencies):6.1f}  "
        f"p50={np.percentile(run_latencies, 50):6.1f}  "
        f"p95={np.percentile(run_latencies, 95):6.1f}"
    )
    totals = np.array(tok_latencies) + np.array(run_latencies)
    print(
        f"  TOTAL:     mean={np.mean(totals):6.1f}  "
        f"p50={np.percentile(totals, 50):6.1f}  "
        f"p95={np.percentile(totals, 95):6.1f}"
    )

    return 0 if mismatches == 0 else 2


if __name__ == "__main__":
    sys.exit(main())

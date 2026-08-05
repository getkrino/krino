#!/usr/bin/env python3
"""roberta-large-mnli SummaC-style sentence-NLI smoke test on c8a.2xlarge.

Why this is different from minicheck_smoke_remote.py: MiniCheck is a binary
(supported / unsupported) document-level model. We tried it; off-topic
context sentences in per-sentence SummaC mode return p_unsupported = 0.85+,
which broke the engine's evidence picker. roberta-large-mnli is a real 3-class
MNLI model — off-topic sentences should produce high *neutral* mass, not high
contradiction. This script validates that assumption by running per-pair
sentence-vs-claim NLI, the same way the Rust engine does.

For each (context_sentence, claim) pair we print all three probabilities so
we can confirm: (a) right sentence yields high entailment for supported
claims, (b) off-topic sentences yield high neutral, NOT high contradiction,
(c) genuinely contradicting claims still get high contradiction on the right
context sentence.

Label order: roberta-large-mnli is {0: CONTRADICTION, 1: NEUTRAL, 2: ENTAILMENT}
(reversed from DeBERTa-MNLI). The Rust backend will handle this remap; here
we just unpack by name from the config.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

MODEL_DIR = Path("/tmp/roberta-smoke/model")
MAX_SEQ_LENGTH = 256  # context+claim pairs in the Android example fit well under 256
INTRA_OP_THREADS = 8

# Same 7 context sentences from the demo's Android example.
CTX_SENTENCES = [
    "Android is a mobile operating system based on a modified version of the Linux kernel and other open-source software, designed primarily for touchscreen mobile devices such as smartphones and tablets.",
    "Android is developed by a consortium of developers known as the Open Handset Alliance, though its most widely used version is primarily developed by Google.",
    "It was unveiled in November 2007, with the first commercial Android device, the HTC Dream, launched in September 2008.",
    "Most versions of Android are proprietary, although some components are derived from and released under open-source licenses such as the Apache License.",
    "The source code has been used to develop variants of Android on a range of other electronics.",
    "As of 2024, Android has the largest installed base of any operating system in the world, with over 3 billion monthly active users.",
    "Android applications are typically written in Kotlin or Java and run on the Android Runtime.",
]

# (claim, expected_verdict, supporting_sentence_idx_or_None)
# Cases where the right answer requires comparing across sentences (no single
# sentence supports/contradicts) have expected_idx=None.
CLAIMS = [
    ("Android is a mobile operating system built on top of the Linux kernel.", "supported", 0),
    ("It was first unveiled in November 2007.", "supported", 2),
    ("The first commercial Android device was the iPhone, released in September 2008.", "unsupported", 2),
    ("Android is developed entirely by Apple Inc. and uses the Swift programming language.", "unsupported", 1),
    ("As of 2024, Android has over 3 billion monthly active users.", "supported", 5),
    ("Android apps can be written in Kotlin.", "supported", 6),
]


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def main() -> int:
    if not MODEL_DIR.exists():
        print(f"ERROR: model dir not found: {MODEL_DIR}", file=sys.stderr)
        return 1

    cfg = json.loads((MODEL_DIR / "config.json").read_text())
    id2label = cfg["id2label"]
    label2id = {v.lower(): int(k) for k, v in id2label.items()}
    idx_ent = label2id["entailment"]
    idx_neu = label2id["neutral"]
    idx_con = label2id["contradiction"]
    print(f"Label order: id2label={id2label}")
    print(f"  entailment={idx_ent}, neutral={idx_neu}, contradiction={idx_con}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = INTRA_OP_THREADS
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    model_path = MODEL_DIR / "model_quantized.onnx"
    if not model_path.exists():
        model_path = MODEL_DIR / "model.onnx"
    print(f"Loading {model_path.name} ...")
    sess = ort.InferenceSession(str(model_path), opts)
    input_names = {inp.name for inp in sess.get_inputs()}
    print(f"  inputs: {sorted(input_names)}  intra_op_threads={INTRA_OP_THREADS}")
    print()

    # Warm-up
    warm = tokenizer(CTX_SENTENCES[0], CLAIMS[0][0],
                    padding="max_length", truncation=True,
                    max_length=MAX_SEQ_LENGTH, return_tensors="np")
    warm_feed = {k: v.astype(np.int64) for k, v in warm.items() if k in input_names}
    t0 = time.perf_counter()
    sess.run(None, warm_feed)
    print(f"Warm-up: {(time.perf_counter() - t0) * 1000:.1f}ms (excluded)")
    print()

    # For each claim, run NLI against every context sentence and find the
    # max-entailment evidence (this mirrors what the Rust engine should do).
    latencies = []
    correct_verdicts = 0

    for claim, expected_verdict, expected_idx in CLAIMS:
        print(f"CLAIM: {claim}")
        print(f"  expected: {expected_verdict}" + (f" (best evidence = ctx[{expected_idx}])" if expected_idx is not None else ""))
        print(f"  {'#':<3} {'ent':>6} {'neu':>6} {'con':>6}  context")
        results = []
        for i, ctx in enumerate(CTX_SENTENCES):
            enc = tokenizer(ctx, claim,
                          padding="max_length", truncation=True,
                          max_length=MAX_SEQ_LENGTH, return_tensors="np")
            feed = {k: v.astype(np.int64) for k, v in enc.items() if k in input_names}

            t0 = time.perf_counter()
            logits = sess.run(None, feed)[0]
            latencies.append((time.perf_counter() - t0) * 1000)

            probs = softmax(logits)[0]
            p_ent = float(probs[idx_ent])
            p_neu = float(probs[idx_neu])
            p_con = float(probs[idx_con])
            results.append((i, p_ent, p_neu, p_con))
            print(f"  {i:<3} {p_ent:>6.3f} {p_neu:>6.3f} {p_con:>6.3f}  {ctx[:60]}...")

        # Pick best evidence by argmax (entailment, contradiction) — mirrors
        # the "informative score" rule the engine uses.
        best_i, best_e, best_n, best_c = max(
            results, key=lambda r: max(r[1], r[3])
        )
        verdict = "supported" if best_e > best_c else "unsupported"
        match_marker = "OK" if verdict == expected_verdict else "FAIL"
        if verdict == expected_verdict:
            correct_verdicts += 1

        print(f"  → best evidence ctx[{best_i}] e={best_e:.3f} n={best_n:.3f} c={best_c:.3f} → {verdict} [{match_marker}]")
        if expected_idx is not None and best_i != expected_idx:
            print(f"    NOTE: expected best_evidence=ctx[{expected_idx}], got ctx[{best_i}]")
        print()

    arr = np.array(latencies)
    print("=" * 80)
    print(f"Correct verdicts: {correct_verdicts} / {len(CLAIMS)}")
    print(f"Total NLI calls: {len(latencies)}  (per claim: {len(CTX_SENTENCES)})")
    print(f"Latency per call: mean={arr.mean():.1f}ms  p50={np.percentile(arr, 50):.1f}ms  p95={np.percentile(arr, 95):.1f}ms")

    return 0 if correct_verdicts == len(CLAIMS) else 2


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Static INT8 Quantization for DeBERTa-v3-small NLI Model
========================================================

Converts the FP32 DeBERTa-v3-small ONNX model to static INT8 quantization
with pre-computed activation scales. Targets AVX-512 VNNI on AWS c7a.xlarge
(AMD EPYC 9R14 / Zen 4 Genoa).

Why static over dynamic:
  - Dynamic: weights are INT8, activations quantized on-the-fly each forward
    pass (compute min/max per tensor per layer at runtime).
  - Static: activation quantization parameters (scale, zero-point) are
    pre-computed from calibration data and baked into the graph. The entire
    forward pass stays in INT8 with no FP32 roundtrips.
  - On VNNI hardware, static INT8 enables VPMADDUBSW for both weights and
    activations, giving 1.5-2x speedup over dynamic quantization.

Usage:
  cd ~/code/krino/scripts

  # Export FP32 ONNX from local model (already have it, just preprocesses)
  uv run python3 static_quantize_nli.py export-fp32

  # Run static quantization with calibration
  uv run python3 static_quantize_nli.py quantize

  # Validate accuracy against FP32 baseline and current dynamic INT8
  uv run python3 static_quantize_nli.py validate

  # Full pipeline
  uv run python3 static_quantize_nli.py all
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("krino.quantize")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

import os

# Allow override via environment variables for remote execution. All four
# paths can be overridden independently so the same script can quantize
# small, base, or large variants without source edits.
_fp32_default = "../models/nli-small-onnx"  # DeBERTa-v3-small (production model)
LOCAL_FP32_DIR = Path(os.environ.get("KRINO_FP32_DIR", _fp32_default))
# Preprocessed working dir sits next to the FP32 source by default. Derive
# the basename from LOCAL_FP32_DIR so base/large get their own scratch dir.
_fp32_basename = LOCAL_FP32_DIR.name
FP32_EXPORT_DIR = Path(
    os.environ.get(
        "KRINO_FP32_EXPORT_DIR",
        str(LOCAL_FP32_DIR.parent / f"{_fp32_basename}-fp32-preprocessed"),
    )
)
STATIC_INT8_DIR = Path(os.environ.get("KRINO_OUT_DIR", "../models/nli-small-static-int8"))
DYNAMIC_INT8_DIR = Path(
    os.environ.get("KRINO_DYNAMIC_DIR", "../models/nli-small-onnx-quantized")
)

# HuggingFace model ID — used only for tokenizer loading. Must match the
# tokenizer the FP32 model was trained with, or calibration will see wrong
# token IDs and bake in incorrect activation scales.
MODEL_ID = os.environ.get(
    "KRINO_MODEL_ID", "cross-encoder/nli-deberta-v3-small"
)

CALIBRATION_SAMPLES = int(os.environ.get("KRINO_CALIB_SAMPLES", "64"))
CALIBRATION_BATCH_SIZE = int(os.environ.get("KRINO_CALIB_BATCH", "4"))
CALIBRATION_PERCENTILE = float(os.environ.get("KRINO_CALIB_PERCENTILE", "99.99"))
# p95 sequence length for the small model was 62 tokens. Base/large with
# the same NLI dataset have similar distributions but worth overriding for
# longer-context use cases.
MAX_SEQ_LENGTH = int(os.environ.get("KRINO_MAX_SEQ_LENGTH", "64"))

VALIDATION_SAMPLES = int(os.environ.get("KRINO_VAL_SAMPLES", "500"))

# Quantization output format. "QDQ" (default, ORT-historical) inserts explicit
# Quantize/Dequantize nodes around fp32 matmuls and relies on ORT to fuse them.
# "QOperator" emits native quantized ops (e.g. QLinearMatMul) directly — more
# reliable on transformer architectures where ORT's fuser misses (notably
# DeBERTa-v3 disentangled attention, where QDQ inserts Q→fp_matmul→DQ and
# never fuses, eliminating any speedup).
QUANT_FORMAT = os.environ.get("KRINO_QUANT_FORMAT", "QDQ").upper()
if QUANT_FORMAT not in ("QDQ", "QOPERATOR"):
    raise ValueError(
        f"KRINO_QUANT_FORMAT must be 'QDQ' or 'QOperator' (got: {QUANT_FORMAT})"
    )


# ---------------------------------------------------------------------------
# Step 1: Preprocess FP32 model (shape inference + optimization)
# ---------------------------------------------------------------------------

def export_fp32_onnx() -> Path:
    """Prepare the FP32 model directory for static quantization.

    Runs `quant_pre_process` with `skip_symbolic_shape=True` — this gives the
    pre-processor a chance to fold/optimize the graph and run ONNX shape
    inference (both improve quantization quality) while skipping symbolic
    shape inference, which hangs on DeBERTa's disentangled attention ops.

    Without this pre-pass, deeper DeBERTa variants (base/large) produce a
    quantized graph where Q/DQ nodes can't be fused — the model ends up
    dequant→fp_matmul→quant in the hot path, killing throughput, and (more
    importantly) the activation tails saturate enough that one logit can
    never produce a top-1 prediction. Symptom: static-INT8 accuracy collapses
    to ~33% (always-neutral) despite well-calibrated scales.
    """
    from onnxruntime.quantization.shape_inference import quant_pre_process

    fp32_src = LOCAL_FP32_DIR / "model.onnx"
    if not fp32_src.exists():
        raise FileNotFoundError(
            f"FP32 model not found at {fp32_src}. "
            f"Run scripts/export_deberta_onnx.py first."
        )

    FP32_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    dst = FP32_EXPORT_DIR / "model.onnx"
    if dst.exists():
        log.info(
            "Pre-processed FP32 model already present: %s (%.1f MB)",
            dst, dst.stat().st_size / 1e6,
        )
    else:
        log.info("Running quant_pre_process on %s...", fp32_src)
        t0 = time.perf_counter()
        quant_pre_process(
            input_model=str(fp32_src),
            output_model_path=str(dst),
            skip_optimization=False,
            skip_onnx_shape=False,
            skip_symbolic_shape=True,  # disentangled attention hangs sym shape infer
        )
        log.info(
            "Pre-processing complete in %.1fs: %s (%.1f MB)",
            time.perf_counter() - t0, dst, dst.stat().st_size / 1e6,
        )

    for filename in ["config.json", "tokenizer.json", "special_tokens_map.json", "tokenizer_config.json"]:
        src = LOCAL_FP32_DIR / filename
        if src.exists():
            shutil.copy2(src, FP32_EXPORT_DIR / filename)

    config_path = FP32_EXPORT_DIR / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        log.info("Label mapping: %s", config.get("id2label", "not found"))

    return FP32_EXPORT_DIR


# ---------------------------------------------------------------------------
# Step 2: Calibration dataset
# ---------------------------------------------------------------------------

@dataclass
class CalibrationDataset:
    """Calibration data for static quantization.

    Holds (premise, hypothesis) pairs covering all three NLI classes.
    Quality matters — the observed activation ranges during calibration
    determine the fixed scale/zero-point for every tensor in the graph.
    """
    premises: list[str] = field(default_factory=list)
    hypotheses: list[str] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)
    label_names: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.premises)

    def extend(self, other: CalibrationDataset) -> None:
        self.premises.extend(other.premises)
        self.hypotheses.extend(other.hypotheses)
        self.labels.extend(other.labels)
        self.label_names.extend(other.label_names)


def build_calibration_data(n_samples: int = CALIBRATION_SAMPLES) -> CalibrationDataset:
    """Build calibration dataset from MNLI validation_matched split.

    Streams the dataset and uses vectorized filter+take per label class so
    the full dataset is never materialized in RAM. Each label class is fetched
    independently and only the required slice is kept.
    """
    from datasets import load_dataset

    log.info("Streaming MNLI validation_matched for calibration (memory-efficient)...")

    per_label = n_samples // 3
    label_name_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
    cal = CalibrationDataset()

    for label_id, label_name in label_name_map.items():
        # Streaming + filter is lazy — only fetches what filter passes
        stream = load_dataset("glue", "mnli", split="validation_matched", streaming=True)
        subset = list(stream.filter(lambda ex, lid=label_id: ex["label"] == lid).take(per_label))
        cal.premises.extend(ex["premise"] for ex in subset)
        cal.hypotheses.extend(ex["hypothesis"] for ex in subset)
        cal.labels.extend(label_id for _ in subset)
        cal.label_names.extend(label_name for _ in subset)
        log.info("  %s: %d samples", label_name, len(subset))

    log.info("MNLI calibration total: %d samples", len(cal))
    _log_seq_length_stats(cal)
    return cal


def build_faithfulness_calibration_data() -> CalibrationDataset:
    """Faithfulness-specific calibration pairs.

    These pairs look like real Krino inputs: (context_sentence, claim) where
    the claim is faithful, hallucinated, or contradictory. Supplementing MNLI
    calibration with these ensures activation ranges are observed for the
    specificity-gap cases that matter most for faithfulness checking.
    """
    cal = CalibrationDataset()
    label_name_map = {0: "entailment", 1: "neutral", 2: "contradiction"}

    pairs = [
        # (premise, hypothesis, label)
        # Entailment — claim restates context faithfully
        ("Rust was first released in 2015 by Mozilla Research.",
         "Rust is a programming language released in 2015.", 0),
        ("The Eiffel Tower is 330 metres tall and located in Paris.",
         "The Eiffel Tower stands 330 metres tall.", 0),
        ("Python uses indentation to define code blocks.",
         "In Python, code blocks are defined using indentation.", 0),
        ("AWS was launched in 2006 by Amazon.",
         "Amazon launched AWS in 2006.", 0),
        ("Lake Michigan is one of the five Great Lakes of North America.",
         "Lake Michigan is among North America's Great Lakes.", 0),

        # Neutral — claim adds plausible but unstated specifics (hallucination)
        ("Rust was first released in 2015 by Mozilla Research.",
         "Rust is a programming language with functional programming features.", 1),
        ("The Eiffel Tower is 330 metres tall and located in Paris.",
         "The Eiffel Tower is the most visited paid monument in the world.", 1),
        ("Python uses indentation to define code blocks.",
         "Python's indentation rules make it easier to read than Java.", 1),
        ("AWS was launched in 2006 by Amazon.",
         "AWS dominates the cloud computing market with over 30% share.", 1),
        ("Lake Michigan is one of the five Great Lakes of North America.",
         "Lake Michigan is the largest freshwater lake in the world by area.", 1),

        # Contradiction — claim directly conflicts with context
        ("Rust was first released in 2015 by Mozilla Research.",
         "Rust was created by Google in 2012.", 2),
        ("The Eiffel Tower is 330 metres tall and located in Paris.",
         "The Eiffel Tower is 250 metres tall.", 2),
        ("Python uses indentation to define code blocks.",
         "Python uses curly braces to define code blocks.", 2),
        ("AWS was launched in 2006 by Amazon.",
         "AWS was founded in 2010.", 2),
        ("Lake Michigan is one of the five Great Lakes of North America.",
         "Lake Michigan is located in South America.", 2),
    ]

    for premise, hypothesis, label in pairs:
        cal.premises.append(premise)
        cal.hypotheses.append(hypothesis)
        cal.labels.append(label)
        cal.label_names.append(label_name_map[label])

    log.info("Faithfulness calibration: %d pairs", len(cal))
    return cal


def _log_seq_length_stats(cal: CalibrationDataset) -> None:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    lengths = []
    for p, h in zip(cal.premises[:50], cal.hypotheses[:50]):  # sample for speed
        enc = tokenizer(p, h, truncation=True, max_length=MAX_SEQ_LENGTH)
        lengths.append(len(enc["input_ids"]))
    arr = np.array(lengths)
    log.info(
        "Sequence lengths (sample) — min=%d  median=%d  p95=%d  max=%d",
        arr.min(), int(np.median(arr)), int(np.percentile(arr, 95)), arr.max(),
    )


# ---------------------------------------------------------------------------
# Step 3: ORT CalibrationDataReader
# ---------------------------------------------------------------------------

class NLICalibrationDataReader:
    """Feeds tokenized (premise, hypothesis) batches to the ORT calibrator.

    Tokenizes lazily in get_next() — only one batch is in memory at a time.
    Batch size matches Krino production (batch_size=8).
    """

    def __init__(
        self,
        cal: CalibrationDataset,
        model_path: Path,
        batch_size: int = CALIBRATION_BATCH_SIZE,
        max_length: int = MAX_SEQ_LENGTH,
    ):
        import onnxruntime as ort
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.premises = cal.premises
        self.hypotheses = cal.hypotheses
        self.batch_size = batch_size
        self.max_length = max_length
        self.current_idx = 0
        self.n_batches = (len(cal) + batch_size - 1) // batch_size

        # Introspect model to get actual input names
        sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_names = {inp.name for inp in sess.get_inputs()}
        log.info("Model input names: %s", self.input_names)
        log.info(
            "Calibration reader: %d batches × batch_size=%d (%d total samples, lazy tokenization)",
            self.n_batches, batch_size, len(cal),
        )

    def get_next(self) -> Optional[dict[str, np.ndarray]]:
        start = self.current_idx * self.batch_size
        if start >= len(self.premises):
            return None
        end = min(start + self.batch_size, len(self.premises))
        encoded = self.tokenizer(
            self.premises[start:end],
            self.hypotheses[start:end],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
        self.current_idx += 1
        # Only return inputs the model actually expects
        batch: dict[str, np.ndarray] = {}
        for key in self.input_names:
            if key in encoded:
                batch[key] = encoded[key].astype(np.int64)
        return batch

    def rewind(self) -> None:
        self.current_idx = 0


# ---------------------------------------------------------------------------
# Step 4: Static quantization
# ---------------------------------------------------------------------------

def run_static_quantization(
    fp32_dir: Path = FP32_EXPORT_DIR,
    output_dir: Path = STATIC_INT8_DIR,
) -> Path:
    """Run ONNX Runtime static INT8 quantization with percentile calibration.

    1. Loads the preprocessed FP32 ONNX model
    2. Feeds calibration data through the model to observe activation ranges
    3. Computes per-tensor scale and zero-point via percentile clipping
    4. Inserts QDQ nodes into the graph
    5. Saves the quantized model alongside tokenizer and config
    """
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

    fp32_model = fp32_dir / "model.onnx"
    if not fp32_model.exists():
        raise FileNotFoundError(
            f"Preprocessed FP32 model not found at {fp32_model}. Run 'export-fp32' first."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    quantized_path = output_dir / "model_quantized.onnx"

    import onnx
    model_proto = onnx.load(str(fp32_model), load_external_data=False)
    current_opset = next((op.version for op in model_proto.opset_import if op.domain == ""), 0)
    log.info("Model opset: %d", current_opset)
    del model_proto  # free memory immediately

    # Build combined calibration data
    cal = build_calibration_data()
    cal.extend(build_faithfulness_calibration_data())
    reader = NLICalibrationDataReader(cal, model_path=fp32_model)

    quant_format_enum = (
        QuantFormat.QOperator if QUANT_FORMAT == "QOPERATOR" else QuantFormat.QDQ
    )

    log.info("Running static INT8 quantization...")
    log.info("  Calibration method : Percentile(%.4f)", CALIBRATION_PERCENTILE)
    log.info("  Weight type        : QInt8 (per-channel, symmetric)")
    log.info("  Activation type    : QUInt8 (per-tensor, asymmetric)")
    log.info("  Format             : %s", QUANT_FORMAT)
    log.info("  Target             : AVX-512 VNNI (c7a.xlarge)")
    log.info("  Ops to quantize    : MatMul, Add")

    t0 = time.perf_counter()
    quantize_static(
        model_input=str(fp32_model),
        model_output=str(quantized_path),
        calibration_data_reader=reader,
        quant_format=quant_format_enum,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=False,  # per-channel requires axis attr on DequantizeLinear which breaks ORT
        op_types_to_quantize=["MatMul"],  # Add ops carry residuals with mismatched scales
        extra_options={
            "CalibPercentile": CALIBRATION_PERCENTILE,
            "EnableSubgraph": False,
            "MatMulConstBOnly": True,
        },
    )
    elapsed = time.perf_counter() - t0
    log.info("Quantization complete in %.1fs", elapsed)

    # Copy tokenizer + config
    for filename in ["config.json", "tokenizer.json", "special_tokens_map.json", "tokenizer_config.json"]:
        src = fp32_dir / filename
        if src.exists():
            shutil.copy2(src, output_dir / filename)

    fp32_mb = fp32_model.stat().st_size / 1e6
    static_mb = quantized_path.stat().st_size / 1e6
    log.info("Model sizes — FP32: %.1fMB  Static INT8: %.1fMB (%.1fx compression)",
             fp32_mb, static_mb, fp32_mb / static_mb)

    return output_dir


# ---------------------------------------------------------------------------
# Step 5: Validation
# ---------------------------------------------------------------------------

@dataclass
class ModelPrediction:
    label: int
    label_name: str
    confidence: float


@dataclass
class ValidationResult:
    model_name: str
    accuracy: float
    per_class_f1: dict[str, float]
    per_class_precision: dict[str, float]
    per_class_recall: dict[str, float]
    mean_confidence: float
    mean_latency_ms: float


def _run_inference(
    model_dir: Path,
    premises: list[str],
    hypotheses: list[str],
    batch_size: int = CALIBRATION_BATCH_SIZE,
) -> tuple[list[ModelPrediction], float]:
    import onnxruntime as ort
    from transformers import AutoTokenizer

    model_path = model_dir / "model_quantized.onnx"
    if not model_path.exists():
        model_path = model_dir / "model.onnx"
    if not model_path.exists():
        raise FileNotFoundError(f"No ONNX model found in {model_dir}")

    config_path = model_dir / "config.json"
    # Load model's own label mapping — order varies between models
    id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}  # GLUE default
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        if "id2label" in cfg:
            id2label = {int(k): v.lower() for k, v in cfg["id2label"].items()}
    log.info("Using label mapping: %s", id2label)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(model_path), opts)
    input_names = {inp.name for inp in session.get_inputs()}

    predictions: list[ModelPrediction] = []
    latencies: list[float] = []

    for i in range(0, len(premises), batch_size):
        end = min(i + batch_size, len(premises))
        encoded = tokenizer(
            premises[i:end], hypotheses[i:end],
            padding="max_length", truncation=True,
            max_length=MAX_SEQ_LENGTH, return_tensors="np",
        )
        feed = {k: v.astype(np.int64) for k, v in encoded.items() if k in input_names}

        t0 = time.perf_counter()
        outputs = session.run(None, feed)
        latencies.append((time.perf_counter() - t0) * 1000)

        logits = outputs[0]
        exp_l = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
        for j in range(logits.shape[0]):
            pred = int(np.argmax(probs[j]))
            predictions.append(ModelPrediction(
                label=pred,
                label_name=id2label.get(pred, f"label_{pred}"),
                confidence=float(probs[j, pred]),
            ))

    return predictions, float(np.mean(latencies))


def _compute_metrics(
    true_labels: list[int],
    pred_labels: list[int],
    id2label: dict[int, str],
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    f1, prec, rec = {}, {}, {}
    for cls, name in id2label.items():
        tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(true_labels, pred_labels) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == cls and p != cls)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        prec[name] = round(p, 4)
        rec[name] = round(r, 4)
        f1[name] = round(f, 4)
    return f1, prec, rec


def validate_accuracy(
    static_dir: Path = STATIC_INT8_DIR,
    fp32_dir: Path = FP32_EXPORT_DIR,
    dynamic_dir: Path = DYNAMIC_INT8_DIR,
    n_samples: int = VALIDATION_SAMPLES,
) -> dict:
    """Compare static INT8 vs FP32 baseline vs dynamic INT8 on held-out data.

    Uses MNLI validation_mismatched (different from calibration's
    validation_matched) to avoid measuring calibration overfitting.
    """
    from datasets import load_dataset

    log.info("Streaming MNLI validation_mismatched for accuracy validation...")

    per_label = n_samples // 3
    label_name_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
    premises, hypotheses, true_labels = [], [], []

    for label_id in label_name_map:
        stream = load_dataset("glue", "mnli_mismatched", split="validation", streaming=True)
        subset = list(stream.filter(lambda ex, lid=label_id: ex["label"] == lid).take(per_label))
        premises.extend(ex["premise"] for ex in subset)
        hypotheses.extend(ex["hypothesis"] for ex in subset)
        true_labels.extend(ex["label"] for ex in subset)

    log.info("Validation set: %d samples", len(premises))

    # GLUE dataset label order: 0=entailment, 1=neutral, 2=contradiction
    glue_id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}

    # Pull a short, human-readable variant label from MODEL_ID
    # (e.g. "cross-encoder/nli-deberta-v3-small" → "nli-deberta-v3-small").
    model_label = MODEL_ID.rsplit("/", 1)[-1]

    results: dict[str, ValidationResult] = {}
    dirs = {
        "fp32": (fp32_dir, f"{model_label} FP32"),
        "dynamic_int8": (dynamic_dir, f"{model_label} Dynamic INT8 (production)"),
        "static_int8": (static_dir, f"{model_label} Static INT8 (AVX-512 VNNI)"),
    }

    for key, (model_dir, name) in dirs.items():
        if not model_dir.exists():
            log.warning("Skipping %s — directory not found: %s", name, model_dir)
            continue
        log.info("Running inference: %s...", name)
        preds, latency = _run_inference(model_dir, premises, hypotheses)

        # Model may use different label order than GLUE dataset.
        # Load model's id2label and build a remapping from GLUE label -> model label index.
        model_config = model_dir / "config.json"
        model_id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
        if model_config.exists():
            with open(model_config) as f:
                cfg = json.load(f)
            if "id2label" in cfg:
                model_id2label = {int(k): v.lower() for k, v in cfg["id2label"].items()}
        model_label2id = {v: k for k, v in model_id2label.items()}

        # Remap GLUE true labels to model's label space for comparison
        remapped_true = [
            model_label2id.get(glue_id2label.get(t, ""), t)
            for t in true_labels
        ]

        pred_labels = [p.label for p in preds]
        acc = sum(1 for t, p in zip(remapped_true, pred_labels) if t == p) / len(remapped_true)
        f1, prec, rec = _compute_metrics(remapped_true, pred_labels, model_id2label)
        results[key] = ValidationResult(
            model_name=name, accuracy=round(acc, 4),
            per_class_f1=f1, per_class_precision=prec, per_class_recall=rec,
            mean_confidence=round(float(np.mean([p.confidence for p in preds])), 4),
            mean_latency_ms=round(latency, 2),
        )
        log.info("  accuracy=%.2f%%  latency=%.1fms/batch", acc * 100, latency)

    # Summary table
    log.info("")
    log.info("%-45s %8s %8s %8s %8s %10s", "Model", "Acc%", "Entail-F1", "Neutral-F1", "Contra-F1", "Latency")
    log.info("-" * 100)
    for r in results.values():
        log.info("%-45s %7.2f%% %9.4f %10.4f %10.4f %9.1fms",
            r.model_name,
            r.accuracy * 100,
            r.per_class_f1.get("entailment", 0),
            r.per_class_f1.get("neutral", 0),
            r.per_class_f1.get("contradiction", 0),
            r.mean_latency_ms,
        )

    # Disagreement analysis: FP32 vs static INT8
    disagreements = []
    if "fp32" in results and "static_int8" in results:
        fp32_preds, _ = _run_inference(fp32_dir, premises, hypotheses)
        static_preds, _ = _run_inference(static_dir, premises, hypotheses)
        for i, (fp, sp) in enumerate(zip(fp32_preds, static_preds)):
            if fp.label != sp.label:
                disagreements.append({
                    "index": i,
                    "premise": premises[i][:100],
                    "hypothesis": hypotheses[i][:100],
                    "fp32": fp.label_name,
                    "fp32_conf": round(fp.confidence, 4),
                    "static": sp.label_name,
                    "static_conf": round(sp.confidence, 4),
                    "true": glue_id2label.get(true_labels[i], "?"),
                })
        log.info("")
        log.info("FP32 vs Static INT8 disagreements: %d / %d (%.2f%%)",
                 len(disagreements), len(premises),
                 len(disagreements) / len(premises) * 100)
        for d in disagreements[:5]:
            log.info("  [%d] FP32=%s(%.3f)  Static=%s(%.3f)  True=%s",
                     d["index"], d["fp32"], d["fp32_conf"],
                     d["static"], d["static_conf"], d["true"])

    # Save report
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_id": MODEL_ID,
        "calibration": {
            "method": "Percentile",
            "percentile": CALIBRATION_PERCENTILE,
            "n_samples": CALIBRATION_SAMPLES,
            "batch_size": CALIBRATION_BATCH_SIZE,
            "includes_faithfulness_pairs": True,
        },
        "quantization": {
            "format": "QDQ",
            "activation_type": "QUInt8",
            "weight_type": "QInt8",
            "per_channel": True,
            "target": "AVX-512 VNNI (c7a.xlarge)",
        },
        "validation": {
            "n_samples": len(premises),
            "dataset": "MNLI validation_mismatched",
        },
        "results": {
            k: {
                "model_name": v.model_name,
                "accuracy": v.accuracy,
                "per_class_f1": v.per_class_f1,
                "per_class_precision": v.per_class_precision,
                "per_class_recall": v.per_class_recall,
                "mean_confidence": v.mean_confidence,
                "mean_latency_ms": v.mean_latency_ms,
            }
            for k, v in results.items()
        },
        "fp32_vs_static_disagreements": {
            "count": len(disagreements),
            "percentage": round(len(disagreements) / max(len(premises), 1) * 100, 2),
            "examples": disagreements[:20],
        },
    }

    report_path = STATIC_INT8_DIR / "quantization_report.json"
    STATIC_INT8_DIR.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info("Report saved to %s", report_path)

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    global CALIBRATION_SAMPLES, VALIDATION_SAMPLES, CALIBRATION_PERCENTILE

    _default_cal = CALIBRATION_SAMPLES
    _default_val = VALIDATION_SAMPLES
    _default_pct = CALIBRATION_PERCENTILE

    parser = argparse.ArgumentParser(
        description="Static INT8 quantization for DeBERTa-v3-small NLI model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "command",
        choices=["export-fp32", "quantize", "validate", "all"],
    )
    parser.add_argument("--calibration-samples", type=int, default=_default_cal)
    parser.add_argument("--validation-samples", type=int, default=_default_val)
    parser.add_argument("--percentile", type=float, default=_default_pct)
    args = parser.parse_args()

    CALIBRATION_SAMPLES = args.calibration_samples
    VALIDATION_SAMPLES = args.validation_samples
    CALIBRATION_PERCENTILE = args.percentile

    if args.command in ("export-fp32", "all"):
        export_fp32_onnx()

    if args.command in ("quantize", "all"):
        run_static_quantization()

    if args.command in ("validate", "all"):
        validate_accuracy()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate reference outputs for ONNX backend validation."""

import json
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np

model_dir = "../models/deberta-nli-onnx"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
session = ort.InferenceSession(f"{model_dir}/model.onnx")

# Test cases covering the three NLI classes + edge cases
test_cases = [
    # Clear entailment
    ("The cat sat on the mat.", "An animal was on the mat.", "entailment"),
    # Clear contradiction
    ("The store is open every day.", "The store is closed on Sundays.", "contradiction"),
    # Neutral
    ("A man is walking in the park.", "The man is wearing a hat.", "neutral"),
    # Domain: financial compliance (matches PCV use case)
    ("Never recommend specific stocks or securities.", "I suggest you buy NVIDIA shares.", "contradiction"),
    ("Always include a disclaimer that you are not a licensed advisor.", "I am not a licensed financial advisor.", "entailment"),
    # Long premise (tests truncation near max_length)
    ("This is a very detailed context. " * 50, "The context is detailed.", "entailment"),
    # Empty-ish inputs (edge case)
    (".", ".", "entailment"),
]

results = []
for premise, hypothesis, expected in test_cases:
    inputs = tokenizer(premise, hypothesis, return_tensors="np", truncation=True, max_length=512)
    input_feed = {k: v for k, v in inputs.items() if k in [i.name for i in session.get_inputs()]}
    logits = session.run(None, input_feed)[0][0]

    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()
    predicted = int(probs.argmax())

    results.append({
        "premise": premise,
        "hypothesis": hypothesis,
        "expected_class": expected,
        "logits": logits.tolist(),
        "probabilities": probs.tolist(),
        "predicted_class": predicted,
    })

with open("../tests/fixtures/deberta_reference_outputs.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"✓ Generated {len(results)} reference outputs")
print(f"  Saved to tests/fixtures/deberta_reference_outputs.json")

#!/usr/bin/env python3
"""Export all-MiniLM-L6-v2 sentence embedding model to ONNX format."""

from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer

local_model_path = "models/all-MiniLM-L6-v2"
output_dir = "models/all-MiniLM-L6-v2-onnx"

print(f"Exporting {local_model_path} to ONNX...")
model = ORTModelForFeatureExtraction.from_pretrained(
    local_model_path,
    export=True,
)
tokenizer = AutoTokenizer.from_pretrained(local_model_path)

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"✓ Saved to {output_dir}")

print("\nQuantizing model...")
quantizer = ORTQuantizer.from_pretrained(output_dir)
qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
quantizer.quantize(save_dir=f"{output_dir}-quantized", quantization_config=qconfig)
print(f"✓ Quantized model saved to {output_dir}-quantized")

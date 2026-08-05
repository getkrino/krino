#!/usr/bin/env python3
"""Export DeBERTa-v3-large-mnli to ONNX format."""

from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer

model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
output_dir = "models/deberta-nli-onnx"

# Export with Optimum (handles opset selection, dynamic axes, etc.)
print(f"Exporting {model_name} to ONNX...")
model = ORTModelForSequenceClassification.from_pretrained(
    model_name,
    export=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"✓ Saved to {output_dir}")

# Quantize for faster inference
print("\nQuantizing model...")
quantizer = ORTQuantizer.from_pretrained(output_dir)
qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
quantizer.quantize(save_dir=f"{output_dir}-quantized", quantization_config=qconfig)
print(f"✓ Quantized model saved to {output_dir}-quantized")

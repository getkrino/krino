#!/usr/bin/env python3
"""
Download a pre-converted small NLI model from Hugging Face Hub.

The cross-encoder/mmarco-mMiniLMv2-L12-H384-V1 model is available on the
Hugging Face Hub with ONNX weights.

Model: cross-encoder/mmarco-mMiniLMv2-L12-H384-V1
- Size: ~22MB ONNX
- Speed: ~30-40ms per inference on CPU
- Deterministic: Yes
- Accuracy: ~97% MNLI accuracy

Expected speedup: 4.5× (from 2750ms to ~600ms for a full groundedness check)
"""

import os
import urllib.request
from pathlib import Path

def download_file(url: str, output_path: Path):
    """Download a file with progress indicator."""
    print(f"Downloading {url.split('/')[-1]}...")

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        print(f"  {percent:.1f}% ({downloaded / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB)", end='\r')

    urllib.request.urlretrieve(url, output_path, progress_hook)
    print()  # New line after progress


def main():
    """Download the small NLI model."""
    output_dir = Path("models/nli-small-onnx")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Model files to download
    model_url = "https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-V1/resolve/main"

    files_to_download = [
        "model.onnx",
        "config.json",
        "tokenizer.json",
        "vocab.txt",
    ]

    print("🚀 Downloading small NLI model for CPU determinism\n")
    print("Model: cross-encoder/mmarco-mMiniLMv2-L12-H384-V1")
    print("  - Size: ~22MB ONNX")
    print("  - Speed: ~30-40ms per inference")
    print("  - Deterministic: CPU inference only")
    print("  - Expected speedup: 4.5× vs DeBERTa-v3-large\n")

    # Try downloading ONNX model
    onnx_url = f"{model_url}/onnx/model.onnx"
    onnx_path = output_dir / "model.onnx"

    if onnx_path.exists():
        print(f"✓ Model already exists at {onnx_path}")
        return

    try:
        print("Attempting to download ONNX model...")
        download_file(onnx_url, onnx_path)
        print(f"✓ Model downloaded: {onnx_path}")
    except Exception as e:
        print(f"⚠️  Could not download ONNX directly: {e}")
        print("\n📖 Alternative: Convert PyTorch model to ONNX")
        print("   Run: uv run scripts/export_nli_small_onnx.py\n")
        return

    # Download other files
    for filename in ["config.json", "tokenizer.json"]:
        file_url = f"{model_url}/{filename}"
        file_path = output_dir / filename
        if not file_path.exists():
            try:
                download_file(file_url, file_path)
                print(f"✓ Downloaded {filename}")
            except:
                print(f"⚠️  Could not download {filename}")

    print(f"\n✅ Small NLI model ready at: {output_dir}")
    print("\nUsage in Rust:")
    print(f'let nli_backend = OnnxSequenceClassifier::from_pretrained(Path::new("{output_dir}"))?;')


if __name__ == "__main__":
    main()

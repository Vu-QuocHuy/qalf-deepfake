#!/usr/bin/env python3
"""Quantize an ONNX TextureSBI model to INT8 for optimized ARM NEON CPU inference."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize ONNX model to INT8 for Raspberry Pi 4 ARM CPU")
    parser.add_argument("--input", required=True, help="Path to input FP32 ONNX model (e.g. models/qalf.onnx)")
    parser.add_argument("--output", default=None, help="Path to output INT8 ONNX model (default: models/qalf_int8.onnx)")
    parser.add_argument("--per-channel", action="store_true", default=True, help="Per-channel weight quantization")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input ONNX model not found: {input_path}")

    output_path = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}_int8.onnx")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as e:
        raise ImportError("onnxruntime is required for quantization. Run: pip install onnxruntime") from e

    print(f"[*] Quantizing FP32 model '{input_path}' to INT8...")
    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
        per_channel=args.per_channel,
        reduce_range=False,
    )

    orig_size_mb = input_path.stat().st_size / (1024 * 1024)
    quant_size_mb = output_path.stat().st_size / (1024 * 1024)
    reduction = (1.0 - quant_size_mb / orig_size_mb) * 100.0

    print(f"[+] Quantization successful!")
    print(f"    - Original FP32 Model : {orig_size_mb:.2f} MB")
    print(f"    - Quantized INT8 Model: {quant_size_mb:.2f} MB ({reduction:.1f}% size reduction)")
    print(f"    - Output saved to     : {output_path}")

    # Copy companion JSON metadata if present
    input_json = input_path.with_suffix(".json")
    output_json = output_path.with_suffix(".json")
    if input_json.is_file():
        with open(input_json, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["precision"] = "INT8"
        meta["output_onnx"] = str(output_path)
        meta["bytes"] = output_path.stat().st_size
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"[+] Companion metadata updated: {output_json}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Export a CPU TorchScript artifact with dynamic INT8 quantization of Linear layers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.export_onnx import ONNXTextureSBIWrapper, build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    data = checkpoint["config"]["data"]
    texture_channels = 3
    model = build_model(checkpoint)
    quantized = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    wrapper = ONNXTextureSBIWrapper(quantized).eval()
    examples = (
        torch.zeros(
            1,
            int(data["texture_frames"]),
            texture_channels,
            int(data["image_size"]),
            int(data["image_size"]),
        ),
    )
    with torch.inference_mode():
        traced = torch.jit.trace(wrapper, examples, strict=False)
        traced = torch.jit.freeze(traced)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(traced, output)
    print(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "output": str(output),
                "bytes": output.stat().st_size,
                "quantization": "dynamic_int8_linear_only",
                "warning": "Convolution layers remain floating point; this is not full INT8 PTQ.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

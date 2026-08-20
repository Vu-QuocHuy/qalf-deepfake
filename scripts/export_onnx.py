#!/usr/bin/env python3
"""Export a TextureSBI / QALF checkpoint to ONNX with embedded metadata and companion JSON config."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qalf.models import QALFModel, build_model_from_checkpoint


class ONNXTextureSBIWrapper(nn.Module):
    """Clean wrapper that outputs the single classification logit for TextureSBI."""

    def __init__(self, model: QALFModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, texture: torch.Tensor) -> torch.Tensor:
        return self.model({"texture": texture})["logit"]


def build_model(checkpoint: dict[str, object]) -> QALFModel:
    return build_model_from_checkpoint(checkpoint).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export TextureSBI checkpoint to ONNX with metadata")
    parser.add_argument("--checkpoint", required=True, help="Path to PyTorch checkpoint (.pt)")
    parser.add_argument("--output", required=True, help="Path to output .onnx file")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (default 17)")
    parser.add_argument("--verify", action="store_true", help="Verify exported ONNX graph")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint.get("config", {})
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})

    image_size = int(data_cfg.get("image_size", 160))
    texture_frames = int(data_cfg.get("texture_frames", 8))
    target_fps = float(data_cfg.get("target_fps", 10.0))
    threshold = float(checkpoint.get("threshold", checkpoint.get("optimal_threshold", 0.5)))
    backbone = model_cfg.get("texture_backbone", "efficientnet_b0")

    wrapper = ONNXTextureSBIWrapper(build_model(checkpoint)).eval()
    example = torch.zeros(1, texture_frames, 3, image_size, image_size, dtype=torch.float32)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        torch.onnx.export(
            wrapper,
            example,
            output,
            input_names=("texture",),
            output_names=("logit",),
            dynamic_axes={"texture": {0: "batch", 1: "texture_frames"}, "logit": {0: "batch"}},
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
        )
    except TypeError:
        # For older PyTorch versions without dynamo argument
        torch.onnx.export(
            wrapper,
            example,
            output,
            input_names=("texture",),
            output_names=("logit",),
            dynamic_axes={"texture": {0: "batch", 1: "texture_frames"}, "logit": {0: "batch"}},
            opset_version=args.opset,
            do_constant_folding=True,
        )

    metadata = {
        "architecture": "TextureSBIModel",
        "checkpoint": str(args.checkpoint),
        "output_onnx": str(output),
        "texture_backbone": backbone,
        "texture_frames": texture_frames,
        "image_size": image_size,
        "target_fps": target_fps,
        "optimal_threshold": threshold,
        "bytes": output.stat().st_size,
        "opset": args.opset,
        "optimal_threshold": float(checkpoint.get("threshold", 0.5)),
        "verified": False,
    }

    # Save companion metadata JSON alongside ONNX file (e.g. models/qalf.json)
    json_path = output.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if args.verify:
        try:
            import onnx
            exported = onnx.load(str(output))
            onnx.checker.check_model(exported)
            metadata["verified"] = True
        except Exception as e:
            print(f"[Warning] ONNX check skipped/failed: {e}")

    print(f"[+] Successfully exported ONNX model to: {output}")
    print(f"[+] Metadata and optimal threshold saved to: {json_path}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

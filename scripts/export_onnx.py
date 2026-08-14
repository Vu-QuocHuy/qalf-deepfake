#!/usr/bin/env python3
"""Export a texture-only QALF checkpoint to ONNX."""

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


class ONNXQALFWrapper(nn.Module):
    def __init__(self, model: QALFModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, texture: torch.Tensor) -> torch.Tensor:
        return self.model({"texture": texture})["logit"]


def build_model(checkpoint: dict[str, object]) -> QALFModel:
    return build_model_from_checkpoint(checkpoint).eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    data = checkpoint["config"]["data"]
    wrapper = ONNXQALFWrapper(build_model(checkpoint)).eval()
    example = torch.zeros(
        1,
        int(data["texture_frames"]),
        3,
        int(data["image_size"]),
        int(data["image_size"]),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
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
    report = {
        "checkpoint": str(args.checkpoint),
        "output": str(output),
        "architecture": "texture_only",
        "texture_backbone": checkpoint["config"]["model"].get(
            "texture_backbone", "efficientnet_b0"
        ),
        "texture_temporal_pooling": checkpoint["config"]["model"].get(
            "temporal_pooling", "mean"
        ),
        "bytes": output.stat().st_size,
        "opset": args.opset,
        "verified": False,
    }
    if args.verify:
        import onnx

        exported = onnx.load(output)
        onnx.checker.check_model(exported)
        report["verified"] = True
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

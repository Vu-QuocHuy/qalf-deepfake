#!/usr/bin/env python3
"""Measure one TextureSBI checkpoint's static FLOPs and GMACs with fvcore.

The reported value is for one forward pass of one clip, excluding cached-frame
I/O, landmark alignment, clip aggregation, and optional horizontal-flip TTA.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qalf.models import build_model_from_checkpoint


class _TextureWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, texture: torch.Tensor) -> torch.Tensor:
        return self.model({"texture": texture})["logit"]


def _gmacs_from_flops(flops: float) -> float:
    # fvcore calls one fused multiply-add one flop, i.e. this is a MAC count.
    return flops / 1_000_000_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    try:
        from fvcore.nn import FlopCountAnalysis, flop_count_table
    except ImportError as error:
        raise SystemExit("fvcore is required: python -m pip install fvcore") from error

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    data = checkpoint["config"]["data"]
    texture_frames = int(data["texture_frames"])
    image_size = int(data["image_size"])
    model = _TextureWrapper(build_model_from_checkpoint(checkpoint).eval())
    dummy = torch.zeros(1, texture_frames, 3, image_size, image_size)
    analysis = FlopCountAnalysis(model, dummy)
    flops = float(analysis.total())
    report = {
        "checkpoint": str(args.checkpoint), "batch_size": 1,
        "texture_frames": texture_frames, "image_size": image_size,
        "fvcore_fma_ops_per_clip": flops,
        "flops_per_clip": 2 * flops, "gflops_per_clip": 2 * flops / 1_000_000_000,
        "gmacs_per_clip": _gmacs_from_flops(flops),
        "scope": "one neural forward pass; excludes preprocessing, clip aggregation, and flip TTA",
        "counting_convention": "fvcore counts one fused multiply-add as one operation; mathematical FLOPs = 2 × fvcore count",
    }
    print(json.dumps(report, indent=2))
    print("\nPer-module table:")
    print(flop_count_table(analysis))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

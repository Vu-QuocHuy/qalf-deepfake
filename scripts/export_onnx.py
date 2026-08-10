#!/usr/bin/env python3
"""Export a trained QALF checkpoint to ONNX with explicit tensor inputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qalf.models import QALFModel


class ONNXQALFWrapper(nn.Module):
    def __init__(self, model: QALFModel) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        geometry: torch.Tensor,
        texture: torch.Tensor,
        geometry_quality: torch.Tensor,
        texture_quality: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.model(
            {
                "geometry": geometry,
                "texture": texture,
                "geometry_quality": geometry_quality,
                "texture_quality": texture_quality,
            }
        )
        return (
            outputs["logit"],
            outputs["geometry_logit"],
            outputs["texture_logit"],
            outputs["fusion_weights"],
        )


def build_model(checkpoint: dict[str, object]) -> QALFModel:
    config = checkpoint["config"]
    model_config = config["model"]
    model = QALFModel(
        geometry_input_dim=int(checkpoint["geometry_input_dim"]),
        geometry_hidden=int(model_config.get("geometry_hidden", 96)),
        geometry_layers=int(model_config.get("geometry_layers", 3)),
        embedding_dim=int(model_config.get("embedding_dim", 128)),
        dropout=float(model_config.get("dropout", 0.2)),
        texture_pretrained=False,
        texture_backbone=str(model_config.get("texture_backbone", "efficientnet_b0")),
        geometry_quality_dim=int(checkpoint.get("geometry_quality_dim", 5)),
        texture_quality_dim=int(checkpoint.get("texture_quality_dim", 5)),
        fusion_mode=str(model_config.get("fusion_mode", "quality")),
        use_srm=bool(model_config.get("use_srm", False)),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.eval()


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
    examples = (
        torch.zeros(1, int(data["num_frames"]), int(checkpoint["geometry_input_dim"])),
        torch.zeros(
            1,
            int(data["texture_frames"]),
            3,
            int(data["image_size"]),
            int(data["image_size"]),
        ),
        torch.zeros(1, int(checkpoint.get("geometry_quality_dim", 5))),
        torch.zeros(1, int(checkpoint.get("texture_quality_dim", 5))),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        examples,
        output,
        input_names=("geometry", "texture", "geometry_quality", "texture_quality"),
        output_names=("logit", "geometry_logit", "texture_logit", "fusion_weights"),
        dynamic_axes={
            "geometry": {0: "batch", 1: "geometry_frames"},
            "texture": {0: "batch", 1: "texture_frames"},
            "geometry_quality": {0: "batch"},
            "texture_quality": {0: "batch"},
            "logit": {0: "batch"},
            "geometry_logit": {0: "batch"},
            "texture_logit": {0: "batch"},
            "fusion_weights": {0: "batch"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )
    report = {
        "checkpoint": str(args.checkpoint),
        "output": str(output),
        "texture_backbone": str(
            checkpoint["config"]["model"].get(
                "texture_backbone", "efficientnet_b0"
            )
        ),
        "srm_enabled": bool(checkpoint["config"]["model"].get("use_srm", False)),
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

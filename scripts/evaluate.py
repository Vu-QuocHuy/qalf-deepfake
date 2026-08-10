#!/usr/bin/env python3
"""Evaluate an FF++-trained QALF checkpoint on Celeb-DF or another manifest."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qalf.data.dataset import QALFVideoDataset
from qalf.engine import aggregate_predictions, predict
from qalf.metrics import compute_metrics
from qalf.models import QALFModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--frame-root", required=True)
    parser.add_argument("--landmark-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--clips-per-video", type=int)
    parser.add_argument("--aggregation", choices=("mean", "median", "topk"))
    parser.add_argument("--top-k", type=int)
    parser.add_argument(
        "--geometry-corruption-json",
        help="Optional deterministic geometry corruption config for robustness evaluation.",
    )
    parser.add_argument("--geometry-corruption-seed", type=int, default=12345)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    data, model_config = config["data"], config["model"]
    geometry_corruption = {}
    if args.geometry_corruption_json:
        with Path(args.geometry_corruption_json).open("r", encoding="utf-8") as handle:
            geometry_corruption = json.load(handle)
        if not isinstance(geometry_corruption, dict):
            parser.error("--geometry-corruption-json must contain a JSON object")
    dataset = QALFVideoDataset(
        args.manifest,
        args.frame_root,
        args.landmark_root,
        num_frames=int(data["num_frames"]),
        texture_frames=int(data["texture_frames"]),
        image_size=int(data["image_size"]),
        geometry_mode=str(data.get("geometry_mode", "aligned_motion_3d")),
        texture_mode=str(data.get("texture_mode", "canonical_skin")),
        training=False,
        clips_per_video=int(
            args.clips_per_video
            if args.clips_per_video is not None
            else data.get("eval_clips_per_video", 1)
        ),
        geometry_corruption=geometry_corruption,
        geometry_corruption_seed=args.geometry_corruption_seed,
    )
    if dataset.geometry_input_dim != int(checkpoint["geometry_input_dim"]):
        raise ValueError("Evaluation geometry feature dimension differs from checkpoint")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    model = QALFModel(
        geometry_input_dim=int(checkpoint["geometry_input_dim"]),
        geometry_hidden=int(model_config.get("geometry_hidden", 96)),
        geometry_layers=int(model_config.get("geometry_layers", 3)),
        embedding_dim=int(model_config.get("embedding_dim", 128)),
        dropout=float(model_config.get("dropout", 0.2)),
        texture_pretrained=False,
        geometry_quality_dim=int(checkpoint.get("geometry_quality_dim", 5)),
        texture_quality_dim=int(checkpoint.get("texture_quality_dim", 5)),
        fusion_mode=str(model_config.get("fusion_mode", "quality")),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    clip_predictions = predict(model, loader, device)
    predictions = aggregate_predictions(
        clip_predictions,
        method=str(args.aggregation or data.get("video_aggregation", "mean")),
        top_k=int(args.top_k if args.top_k is not None else data.get("top_k", 1)),
    )
    metrics = compute_metrics(
        np.asarray(predictions["label"], dtype=np.int64),
        np.asarray(predictions["score"], dtype=np.float64),
        float(checkpoint["threshold"]),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    with (output_dir / "evaluation_protocol.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "threshold": float(checkpoint["threshold"]),
                "threshold_selection": {"datasets": ["ffpp"], "splits": ["val"]},
                "geometry_corruption": geometry_corruption,
                "geometry_corruption_seed": args.geometry_corruption_seed,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    for filename, rows in (
        ("clip_predictions.csv", clip_predictions),
        ("predictions.csv", predictions),
    ):
        columns = list(rows)
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in zip(*(rows[column] for column in columns), strict=True):
                writer.writerow(dict(zip(columns, row, strict=True)))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

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
from qalf.data.geometry import DEFAULT_GEOMETRY_FEATURE_MODE
from qalf.engine import aggregate_predictions, predict
from qalf.metrics import compute_metrics, select_threshold
from qalf.models import QALFModel


def _format_metrics(metrics: dict[str, float]) -> str:
    lines = ["QALF evaluation metrics"]
    lines.extend(f"{name}: {value:.4f}" for name, value in metrics.items())
    return "\n".join(lines) + "\n"


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
        "--threshold-manifest",
        help="FF++ validation manifest used to calibrate the decision threshold.",
    )
    parser.add_argument("--threshold-frame-root")
    parser.add_argument("--threshold-landmark-root")
    parser.add_argument("--threshold-clips-per-video", type=int)
    parser.add_argument(
        "--texture-flip-tta",
        action="store_true",
        help="Average predictions from original and horizontally flipped texture inputs.",
    )
    parser.add_argument(
        "--geometry-corruption-json",
        help="Optional deterministic geometry corruption config for robustness evaluation.",
    )
    parser.add_argument("--geometry-corruption-seed", type=int, default=12345)
    args = parser.parse_args()

    threshold_paths = (
        args.threshold_manifest,
        args.threshold_frame_root,
        args.threshold_landmark_root,
    )
    if any(threshold_paths) and not all(threshold_paths):
        parser.error(
            "--threshold-manifest, --threshold-frame-root, and "
            "--threshold-landmark-root must be provided together"
        )

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
        geometry_mode=str(data.get("geometry_mode", DEFAULT_GEOMETRY_FEATURE_MODE)),
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
        texture_backbone=str(model_config.get("texture_backbone", "mobilenet_v3_small")),
        geometry_quality_dim=int(checkpoint.get("geometry_quality_dim", 5)),
        texture_quality_dim=int(checkpoint.get("texture_quality_dim", 5)),
        fusion_mode=str(model_config.get("fusion_mode", "quality")),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    threshold = float(checkpoint["threshold"])
    threshold_source = "checkpoint_ffpp_validation"
    threshold_calibration_metrics: dict[str, float] | None = None
    if args.threshold_manifest:
        threshold_dataset = QALFVideoDataset(
            args.threshold_manifest,
            args.threshold_frame_root,
            args.threshold_landmark_root,
            num_frames=int(data["num_frames"]),
            texture_frames=int(data["texture_frames"]),
            image_size=int(data["image_size"]),
            geometry_mode=str(data.get("geometry_mode", DEFAULT_GEOMETRY_FEATURE_MODE)),
            texture_mode=str(data.get("texture_mode", "canonical_skin")),
            training=False,
            clips_per_video=int(
                args.threshold_clips_per_video
                if args.threshold_clips_per_video is not None
                else data.get("eval_clips_per_video", 1)
            ),
            fake_methods=data.get("fake_methods"),
        )
        if threshold_dataset.geometry_input_dim != int(checkpoint["geometry_input_dim"]):
            raise ValueError("Threshold calibration geometry dimension differs from checkpoint")
        threshold_protocol = (
            sorted({record.dataset for record in threshold_dataset.records}),
            sorted({record.split for record in threshold_dataset.records}),
        )
        if threshold_protocol != (["ffpp"], ["val"]):
            raise ValueError(
                "Threshold calibration is restricted to the official FF++ validation split; "
                f"got datasets={threshold_protocol[0]}, splits={threshold_protocol[1]}"
            )
        threshold_loader = DataLoader(
            threshold_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=False,
        )
        threshold_clip_predictions = predict(
            model,
            threshold_loader,
            device,
            texture_flip_tta=args.texture_flip_tta,
        )
        threshold_predictions = aggregate_predictions(
            threshold_clip_predictions,
            method=str(args.aggregation or data.get("video_aggregation", "mean")),
            top_k=int(args.top_k if args.top_k is not None else data.get("top_k", 1)),
        )
        threshold_labels = np.asarray(threshold_predictions["label"], dtype=np.int64)
        threshold_scores = np.asarray(threshold_predictions["score"], dtype=np.float64)
        threshold = select_threshold(threshold_labels, threshold_scores)
        threshold_calibration_metrics = compute_metrics(
            threshold_labels,
            threshold_scores,
            threshold,
        )
        threshold_source = str(args.threshold_manifest)

    clip_predictions = predict(
        model,
        loader,
        device,
        texture_flip_tta=args.texture_flip_tta,
    )
    predictions = aggregate_predictions(
        clip_predictions,
        method=str(args.aggregation or data.get("video_aggregation", "mean")),
        top_k=int(args.top_k if args.top_k is not None else data.get("top_k", 1)),
    )
    labels = np.asarray(predictions["label"], dtype=np.int64)
    fused_scores = np.asarray(predictions["score"], dtype=np.float64)
    geometry_scores = np.asarray(predictions["geometry_score"], dtype=np.float64)
    texture_scores = np.asarray(predictions["texture_score"], dtype=np.float64)
    metrics = compute_metrics(labels, fused_scores, threshold)
    metrics["geometry_auc"] = compute_metrics(labels, geometry_scores, threshold)["auc"]
    metrics["texture_auc"] = compute_metrics(labels, texture_scores, threshold)["auc"]
    metrics["fixed_average_auc"] = compute_metrics(
        labels,
        0.5 * (geometry_scores + texture_scores),
        threshold,
    )["auc"]
    metrics["mean_geometry_weight"] = float(np.mean(predictions["geometry_weight"]))
    metrics["mean_texture_weight"] = float(np.mean(predictions["texture_weight"]))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_text = _format_metrics(metrics)
    (output_dir / "metrics.txt").write_text(metrics_text, encoding="utf-8")
    if threshold_calibration_metrics is not None:
        (output_dir / "threshold_calibration_metrics.txt").write_text(
            _format_metrics(threshold_calibration_metrics),
            encoding="utf-8",
        )
    (output_dir / "evaluation_protocol.txt").write_text(
        "\n".join(
            (
                "QALF evaluation protocol",
                f"checkpoint: {args.checkpoint}",
                f"manifest: {args.manifest}",
                f"texture_backbone: {model_config.get('texture_backbone', 'mobilenet_v3_small')}",
                f"num_frames: {int(data['num_frames'])}",
                f"texture_frames: {int(data['texture_frames'])}",
                f"image_size: {int(data['image_size'])}",
                f"clips_per_video: {dataset.clips_per_video}",
                f"aggregation: {args.aggregation or data.get('video_aggregation', 'mean')}",
                f"texture_flip_tta: {args.texture_flip_tta}",
                f"threshold: {threshold:.4f}",
                f"checkpoint_threshold: {float(checkpoint['threshold']):.4f}",
                f"threshold_source: {threshold_source}",
                f"threshold_clips_per_video: {int(args.threshold_clips_per_video if args.threshold_clips_per_video is not None else data.get('eval_clips_per_video', 1))}",
                f"geometry_corruption: {json.dumps(geometry_corruption, ensure_ascii=False)}",
                f"geometry_corruption_seed: {args.geometry_corruption_seed}",
            )
        )
        + "\n",
        encoding="utf-8",
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
    print(metrics_text, end="")


if __name__ == "__main__":
    main()

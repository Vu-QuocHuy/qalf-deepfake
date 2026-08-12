#!/usr/bin/env python3
"""Evaluate a texture-only QALF checkpoint at video level."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qalf.config import save_json
from qalf.data.dataset import QALFVideoDataset
from qalf.engine import aggregate_predictions, predict
from qalf.metrics import compute_metrics, select_threshold
from qalf.models import build_model_from_checkpoint
from qalf.reporting import collect_run_metadata, format_evaluation_report, save_evaluation_plots


def _create_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("qalf.evaluate")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def _dataset(
    manifest: str,
    frame_root: str,
    landmark_root: str,
    data: dict[str, object],
    texture_frames: int,
    clips_per_video: int,
    *,
    fake_methods: list[str] | None = None,
) -> QALFVideoDataset:
    return QALFVideoDataset(
        manifest,
        frame_root,
        landmark_root,
        num_frames=int(data["num_frames"]),
        texture_frames=texture_frames,
        image_size=int(data["image_size"]),
        texture_mode=str(data.get("texture_mode", "full_face")),
        training=False,
        clips_per_video=clips_per_video,
        fake_methods=fake_methods,
    )


def _loader(dataset: QALFVideoDataset, batch_size: int, workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def _write_csv(path: Path, rows: dict[str, object]) -> None:
    columns = list(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in zip(*(rows[column] for column in columns), strict=True):
            writer.writerow(dict(zip(columns, row, strict=True)))


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
    parser.add_argument("--texture-frames", type=int)
    parser.add_argument("--aggregation", choices=("mean", "median", "topk"))
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--threshold-manifest")
    parser.add_argument("--threshold-frame-root")
    parser.add_argument("--threshold-landmark-root")
    parser.add_argument("--threshold-clips-per-video", type=int)
    parser.add_argument("--texture-flip-tta", action="store_true")
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = _create_logger(output_dir / "eval.log")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if checkpoint.get("architecture", "texture_only") != "texture_only":
        raise ValueError("This cleaned evaluator only supports texture-only checkpoints")
    config = checkpoint["config"]
    data, model_config = config["data"], config["model"]
    save_json(collect_run_metadata(sys.argv, config, PROJECT_ROOT), output_dir / "run_metadata.json")
    training_texture_frames = int(data["texture_frames"])
    texture_frames = int(args.texture_frames or training_texture_frames)
    if not 1 <= texture_frames <= int(data["num_frames"]):
        parser.error("--texture-frames must be in [1, checkpoint num_frames]")
    clips_per_video = int(args.clips_per_video or data.get("eval_clips_per_video", 1))
    aggregation = str(args.aggregation or data.get("video_aggregation", "mean"))
    top_k = int(args.top_k or data.get("top_k", 1))
    logger.info("=" * 72)
    logger.info("QALF TEXTURE-ONLY EVALUATION")
    logger.info(
        "  model | backbone=%s input=full_face frames=%d/%d image_size=%d pooling=%s "
        "weights=%s ema_decay=%.4f",
        model_config.get("texture_backbone", "efficientnet_b0"),
        texture_frames,
        int(data["num_frames"]),
        int(data["image_size"]),
        model_config.get("temporal_pooling", "mean"),
        checkpoint.get("model_weights", "raw"),
        float(checkpoint.get("ema_decay", 0.0)),
    )
    logger.info(
        "  video | clips=%d aggregation=%s flip_tta=%s",
        clips_per_video,
        aggregation,
        args.texture_flip_tta,
    )
    logger.info("=" * 72)

    dataset = _dataset(
        args.manifest,
        args.frame_root,
        args.landmark_root,
        data,
        texture_frames,
        clips_per_video,
    )
    model = build_model_from_checkpoint(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    threshold = float(checkpoint["threshold"])
    threshold_source = "checkpoint_ffpp_validation"
    threshold_metrics = None
    threshold_predictions = None
    if args.threshold_manifest:
        threshold_clips = int(
            args.threshold_clips_per_video or data.get("eval_clips_per_video", 1)
        )
        threshold_dataset = _dataset(
            args.threshold_manifest,
            args.threshold_frame_root,
            args.threshold_landmark_root,
            data,
            texture_frames,
            threshold_clips,
            fake_methods=data.get("fake_methods"),
        )
        protocol = (
            sorted({record.dataset for record in threshold_dataset.records}),
            sorted({record.split for record in threshold_dataset.records}),
        )
        if protocol != (["ffpp"], ["val"]):
            raise ValueError(
                "Threshold calibration is restricted to the official FF++ validation split; "
                f"got datasets={protocol[0]}, splits={protocol[1]}"
            )
        threshold_predictions = aggregate_predictions(
            predict(
                model,
                _loader(threshold_dataset, args.batch_size, args.num_workers),
                device,
                texture_flip_tta=args.texture_flip_tta,
            ),
            method=aggregation,
            top_k=top_k,
        )
        threshold_labels = np.asarray(threshold_predictions["label"], dtype=np.int64)
        threshold_scores = np.asarray(threshold_predictions["score"], dtype=np.float64)
        threshold = select_threshold(threshold_labels, threshold_scores)
        threshold_metrics = compute_metrics(threshold_labels, threshold_scores, threshold)
        threshold_source = str(args.threshold_manifest)

    clip_predictions = predict(
        model,
        _loader(dataset, args.batch_size, args.num_workers),
        device,
        texture_flip_tta=args.texture_flip_tta,
    )
    predictions = aggregate_predictions(clip_predictions, method=aggregation, top_k=top_k)
    labels = np.asarray(predictions["label"], dtype=np.int64)
    scores = np.asarray(predictions["score"], dtype=np.float64)
    metrics = compute_metrics(labels, scores, threshold)
    datasets = sorted({str(value) for value in predictions["dataset"]})
    context = {
        "Dataset": ", ".join(datasets),
        "Videos": (
            f"{int(metrics['sample_count'])} "
            f"(real={int(metrics['real_count'])}, fake={int(metrics['fake_count'])})"
        ),
        "Model": (
            "EfficientNet-B0 texture-only SBI "
            f"(weights={checkpoint.get('model_weights', 'raw')}, "
            f"ema_decay={float(checkpoint.get('ema_decay', 0.0)):.4f})"
        ),
        "Inference": (
            f"clips={clips_per_video}, aggregation={aggregation}, "
            f"texture_frames={texture_frames}, flip_tta={args.texture_flip_tta}"
        ),
        "Threshold source": (
            "FF++ validation calibration (Youden-J)"
            if args.threshold_manifest
            else "checkpoint FF++ validation (Youden-J)"
        ),
    }
    report = format_evaluation_report(metrics, context=context)
    (output_dir / "metrics.txt").write_text(report, encoding="utf-8")
    print("\n" + report)
    if threshold_metrics is not None:
        (output_dir / "threshold_calibration_metrics.txt").write_text(
            format_evaluation_report(
                threshold_metrics,
                title="FF++ THRESHOLD CALIBRATION",
                context={"Manifest": args.threshold_manifest},
            ),
            encoding="utf-8",
        )
    protocol = {
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "datasets": datasets,
        "label_convention": str(checkpoint.get("label_convention", "real=0,fake=1")),
        "score_target": str(checkpoint.get("score_target", "fake")),
        "model": {
            "architecture": "texture_only",
            "texture_backbone": model_config.get("texture_backbone", "efficientnet_b0"),
            "texture_temporal_pooling": model_config.get("temporal_pooling", "mean"),
            "texture_mode": data.get("texture_mode", "full_face"),
            "model_weights": checkpoint.get("model_weights", "raw"),
            "ema_decay": float(checkpoint.get("ema_decay", 0.0)),
        },
        "training_data": {
            "num_frames": int(data["num_frames"]),
            "texture_frames": training_texture_frames,
            "image_size": int(data["image_size"]),
            "sbi": data.get("sbi", {"enabled": False}),
        },
        "inference": {
            "num_frames": int(data["num_frames"]),
            "texture_frames": texture_frames,
            "clips_per_video": clips_per_video,
            "aggregation": aggregation,
            "top_k": top_k,
            "texture_flip_tta": args.texture_flip_tta,
        },
        "threshold": {
            "value": threshold,
            "source": threshold_source,
            "checkpoint_value": float(checkpoint["threshold"]),
            "selection": checkpoint.get("threshold_selection", "youden_j_ffpp_validation"),
        },
    }
    save_json({"metrics": metrics, "protocol": protocol}, output_dir / "metrics.json")
    (output_dir / "evaluation_protocol.txt").write_text(
        "\n".join(
            (
                "QALF TEXTURE-ONLY EVALUATION PROTOCOL",
                "=" * 72,
                f"checkpoint: {protocol['checkpoint']}",
                f"manifest: {protocol['manifest']}",
                f"datasets: {', '.join(protocol['datasets'])}",
                f"model: {json.dumps(protocol['model'], ensure_ascii=False)}",
                f"training_data: {json.dumps(protocol['training_data'], ensure_ascii=False)}",
                f"inference: {json.dumps(protocol['inference'], ensure_ascii=False)}",
                f"threshold: {json.dumps(protocol['threshold'], ensure_ascii=False)}",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "clip_predictions.csv", clip_predictions)
    _write_csv(output_dir / "predictions.csv", predictions)
    if threshold_predictions is not None:
        _write_csv(output_dir / "threshold_predictions.csv", threshold_predictions)
    try:
        plots = save_evaluation_plots(labels, scores, threshold, output_dir / "plots")
    except Exception as error:
        logger.warning("Evaluation plots disabled: %s", error)
        plots = []
    logger.info(
        "evaluation_complete metrics=%s predictions=%s plots=%s",
        output_dir / "metrics.json",
        output_dir / "predictions.csv",
        ",".join(plots) if plots else "disabled",
    )


if __name__ == "__main__":
    main()

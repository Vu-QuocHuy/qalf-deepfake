#!/usr/bin/env python3
"""Evaluate an FF++-trained QALF checkpoint on Celeb-DF or another manifest."""

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
from qalf.data.dataset import QALFVideoDataset, texture_view_count
from qalf.data.geometry import DEFAULT_GEOMETRY_FEATURE_MODE
from qalf.engine import aggregate_predictions, predict
from qalf.metrics import compute_metrics, select_threshold
from qalf.models import QALFModel
from qalf.reporting import (
    collect_run_metadata,
    format_evaluation_report,
    save_evaluation_plots,
)


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
    parser.add_argument(
        "--texture-frames",
        type=int,
        help="Override the number of uniformly sampled texture frames at evaluation.",
    )
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = _create_logger(output_dir / "eval.log")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if "label_convention" not in checkpoint or "score_target" not in checkpoint:
        logger.warning(
            "Legacy checkpoint has no explicit label metadata; assuming real=0, fake=1, "
            "score=P(fake)."
        )
    label_convention = str(checkpoint.get("label_convention", "real=0,fake=1"))
    score_target = str(checkpoint.get("score_target", "fake"))
    if label_convention != "real=0,fake=1" or score_target != "fake":
        raise ValueError(
            "Checkpoint label convention is incompatible: "
            f"label_convention={label_convention}, score_target={score_target}"
        )
    config = checkpoint["config"]
    save_json(
        collect_run_metadata(sys.argv, config, PROJECT_ROOT),
        output_dir / "run_metadata.json",
    )
    data, model_config = config["data"], config["model"]
    texture_frames = int(
        args.texture_frames if args.texture_frames is not None else data["texture_frames"]
    )
    if not 1 <= texture_frames <= int(data["num_frames"]):
        parser.error("--texture-frames must be in [1, checkpoint num_frames]")
    logger.info("=" * 72)
    logger.info("QALF EVALUATION RUN")
    logger.info(
        "  model | "
        f"backbone={model_config.get('texture_backbone', 'efficientnet_b0')} "
        f"texture_mode={data.get('texture_mode', 'canonical_skin')} "
        f"texture_pooling={model_config.get('texture_temporal_pooling', 'mean')} "
        f"mixstyle_probability={float(model_config.get('texture_mixstyle_probability', 0.0)):.3f} "
        f"texture_frames={texture_frames} "
        f"image_size={int(data['image_size'])} "
        f"weights={checkpoint.get('model_weights', 'raw')} "
        f"ema_decay={float(checkpoint.get('ema_decay', 0.0)):.4f}"
    )
    sbi_config = data.get("sbi", {"enabled": False})
    logger.info(
        "  train | sbi_enabled=%s sbi_mixture=%s",
        bool(sbi_config.get("enabled", False)),
        sbi_config.get("mixture", "legacy/default"),
    )
    logger.info("  input | checkpoint=%s", args.checkpoint)
    logger.info("  input | manifest=%s", args.manifest)
    logger.info("=" * 72)
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
        texture_frames=texture_frames,
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
        texture_backbone=str(model_config.get("texture_backbone", "efficientnet_b0")),
        texture_temporal_pooling=str(model_config.get("texture_temporal_pooling", "mean")),
        texture_views=texture_view_count(
            str(data.get("texture_mode", "canonical_skin"))
        ),
        texture_mixstyle_probability=float(
            model_config.get("texture_mixstyle_probability", 0.0)
        ),
        texture_mixstyle_alpha=float(model_config.get("texture_mixstyle_alpha", 0.1)),
        texture_mixstyle_layers=tuple(
            int(index) for index in model_config.get("texture_mixstyle_layers", [])
        ),
        geometry_quality_dim=int(checkpoint.get("geometry_quality_dim", 5)),
        texture_quality_dim=int(checkpoint.get("texture_quality_dim", 5)),
        fusion_mode=str(model_config.get("fusion_mode", "quality")),
        texture_gate_bias=float(model_config.get("texture_gate_bias", 0.0)),
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
            texture_frames=texture_frames,
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
    dataset_names = sorted({str(value) for value in predictions["dataset"]})
    aggregation = str(args.aggregation or data.get("video_aggregation", "mean"))
    threshold_context = (
        "FF++ validation calibration (Youden-J)"
        if args.threshold_manifest
        else "checkpoint FF++ validation (Youden-J)"
    )
    report_context = {
        "Dataset": ", ".join(dataset_names),
        "Videos": (
            f"{int(metrics['sample_count'])} "
            f"(real={int(metrics['real_count'])}, fake={int(metrics['fake_count'])})"
        ),
        "Inference": (
            f"clips={dataset.clips_per_video}, aggregation={aggregation}, "
            f"texture_frames={texture_frames}, flip_tta={args.texture_flip_tta}"
        ),
        "Threshold source": threshold_context,
        "Model weights": checkpoint.get("model_weights", "raw"),
    }
    metrics_text = format_evaluation_report(metrics, context=report_context)
    (output_dir / "metrics.txt").write_text(metrics_text, encoding="utf-8")
    if threshold_calibration_metrics is not None:
        (output_dir / "threshold_calibration_metrics.txt").write_text(
            format_evaluation_report(
                threshold_calibration_metrics,
                title="FF++ THRESHOLD CALIBRATION",
                context={"Manifest": args.threshold_manifest},
            ),
            encoding="utf-8",
        )
    protocol = {
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "datasets": dataset_names,
        "label_convention": label_convention,
        "score_target": score_target,
        "model": {
            "texture_backbone": model_config.get("texture_backbone", "efficientnet_b0"),
            "texture_temporal_pooling": model_config.get(
                "texture_temporal_pooling", "mean"
            ),
            "texture_mode": data.get("texture_mode", "canonical_skin"),
            "texture_mixstyle_probability": float(
                model_config.get("texture_mixstyle_probability", 0.0)
            ),
            "texture_mixstyle_alpha": float(
                model_config.get("texture_mixstyle_alpha", 0.1)
            ),
            "texture_mixstyle_layers": model_config.get(
                "texture_mixstyle_layers", []
            ),
            "model_weights": checkpoint.get("model_weights", "raw"),
            "ema_decay": float(checkpoint.get("ema_decay", 0.0)),
        },
        "training_data": {
            "sbi": sbi_config,
        },
        "inference": {
            "num_frames": int(data["num_frames"]),
            "texture_frames": texture_frames,
            "image_size": int(data["image_size"]),
            "clips_per_video": dataset.clips_per_video,
            "aggregation": aggregation,
            "top_k": int(args.top_k if args.top_k is not None else data.get("top_k", 1)),
            "texture_flip_tta": args.texture_flip_tta,
        },
        "threshold": {
            "value": threshold,
            "source": threshold_source,
            "checkpoint_value": float(checkpoint["threshold"]),
            "selection": checkpoint.get(
                "threshold_selection", "youden_j_ffpp_validation"
            ),
            "calibration_clips_per_video": int(
                args.threshold_clips_per_video
                if args.threshold_clips_per_video is not None
                else data.get("eval_clips_per_video", 1)
            ),
        },
        "geometry_corruption": geometry_corruption,
        "geometry_corruption_seed": args.geometry_corruption_seed,
    }
    save_json({"metrics": metrics, "protocol": protocol}, output_dir / "metrics.json")
    (output_dir / "evaluation_protocol.txt").write_text(
        "\n".join(
            (
                "QALF EVALUATION PROTOCOL",
                "=" * 72,
                f"checkpoint: {protocol['checkpoint']}",
                f"manifest: {protocol['manifest']}",
                f"datasets: {', '.join(protocol['datasets'])}",
                f"label_convention: {protocol['label_convention']}",
                f"score_target: {protocol['score_target']}",
                f"model: {json.dumps(protocol['model'], ensure_ascii=False)}",
                "training_data: "
                f"{json.dumps(protocol['training_data'], ensure_ascii=False)}",
                f"inference: {json.dumps(protocol['inference'], ensure_ascii=False)}",
                f"threshold: {json.dumps(protocol['threshold'], ensure_ascii=False)}",
                "geometry_corruption: "
                f"{json.dumps(protocol['geometry_corruption'], ensure_ascii=False)}",
                f"geometry_corruption_seed: {protocol['geometry_corruption_seed']}",
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
    try:
        plot_files = save_evaluation_plots(
            labels,
            fused_scores,
            threshold,
            output_dir / "plots",
        )
    except Exception as error:
        plot_files = []
        logger.warning("Could not generate evaluation plots: %s", error)
    logger.info("\n%s", metrics_text.rstrip())
    logger.info(
        "evaluation_complete metrics=%s predictions=%s plots=%s",
        output_dir / "metrics.json",
        output_dir / "predictions.csv",
        ",".join(plot_files),
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate cached, budget-adaptive texture inference over ordered video clips."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qalf.data.dataset import QALFVideoDataset
from qalf.data.geometry import DEFAULT_GEOMETRY_FEATURE_MODE
from qalf.engine import aggregate_predictions, move_batch
from qalf.metrics import compute_metrics, select_threshold
from qalf.models import QALFModel, TextureRefreshPolicy


def _write_csv(path: Path, rows: dict[str, object]) -> None:
    columns = list(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in zip(*(rows[column] for column in columns), strict=True):
            writer.writerow(dict(zip(columns, row, strict=True)))


def _load_threshold(path: str | Path) -> tuple[float, dict[str, list[str]]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    provenance = payload.get("threshold_selection")
    if not isinstance(provenance, dict):
        raise ValueError(f"Threshold JSON lacks validated threshold_selection provenance: {path}")
    datasets = sorted(str(value) for value in provenance.get("datasets", []))
    splits = sorted(str(value) for value in provenance.get("splits", []))
    if datasets != ["ffpp"] or splits != ["val"]:
        raise ValueError(
            f"Threshold must originate from FF++ validation, got datasets={datasets}, splits={splits}"
        )
    if "threshold" in payload:
        threshold = float(payload["threshold"])
    elif "metrics" in payload and "threshold" in payload["metrics"]:
        threshold = float(payload["metrics"]["threshold"])
    else:
        raise ValueError(f"No threshold found in: {path}")
    return threshold, {"datasets": datasets, "splits": splits}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--frame-root", required=True)
    parser.add_argument("--landmark-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--policy", choices=("always", "periodic", "rule", "geometry"), default="rule"
    )
    parser.add_argument("--clips-per-video", type=int)
    parser.add_argument("--aggregation", choices=("mean", "median", "topk"))
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--refresh-interval", type=int, default=2)
    parser.add_argument("--max-cache-age", type=int, default=4)
    parser.add_argument("--uncertainty-threshold", type=float, default=0.65)
    parser.add_argument("--min-landmark-ratio", type=float, default=0.80)
    parser.add_argument("--threshold-json")
    parser.add_argument(
        "--budget-reference",
        help="Rule-policy report used to require a matched periodic texture-call budget.",
    )
    parser.add_argument("--budget-tolerance", type=float, default=0.02)
    parser.add_argument(
        "--select-threshold",
        action="store_true",
        help="Select threshold on this manifest. Use only on FF++ validation, never target test.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    if args.select_threshold and args.threshold_json:
        parser.error("--select-threshold and --threshold-json are mutually exclusive")
    if args.policy != "always" and not (args.select_threshold or args.threshold_json):
        parser.error(
            "Adaptive policies require --select-threshold on FF++ validation or "
            "--threshold-json from that validation run"
        )
    if args.refresh_interval < 1 or args.max_cache_age < 1:
        parser.error("refresh interval and max cache age must be positive")
    if args.budget_reference and args.policy != "periodic":
        parser.error("--budget-reference is only valid for the periodic baseline")
    if args.budget_tolerance < 0:
        parser.error("--budget-tolerance must be non-negative")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    data, model_config = config["data"], config["model"]
    clips_per_video = int(
        args.clips_per_video
        if args.clips_per_video is not None
        else data.get("eval_clips_per_video", 2)
    )
    aggregation = str(args.aggregation or data.get("video_aggregation", "mean"))
    top_k = int(args.top_k if args.top_k is not None else data.get("top_k", 1))
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
        clips_per_video=clips_per_video,
    )
    manifest_datasets = sorted({record.dataset for record in dataset.records})
    manifest_splits = sorted({record.split for record in dataset.records})
    if args.select_threshold and (manifest_datasets != ["ffpp"] or manifest_splits != ["val"]):
        parser.error(
            "--select-threshold is hard-limited to an FF++ validation manifest; "
            f"got datasets={manifest_datasets}, splits={manifest_splits}"
        )
    if dataset.geometry_input_dim != int(checkpoint["geometry_input_dim"]):
        raise ValueError("Evaluation geometry feature dimension differs from checkpoint")
    loader = DataLoader(
        dataset,
        batch_size=1,
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
        geometry_quality_dim=int(checkpoint.get("geometry_quality_dim", 5)),
        texture_quality_dim=int(checkpoint.get("texture_quality_dim", 5)),
        fusion_mode=str(model_config.get("fusion_mode", "quality")),
        texture_gate_bias=float(model_config.get("texture_gate_bias", 0.0)),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    refresh_policy = TextureRefreshPolicy(
        max_cache_age=args.max_cache_age,
        geometry_uncertainty_threshold=args.uncertainty_threshold,
        min_landmark_detection_ratio=args.min_landmark_ratio,
    )

    clip_rows: dict[str, list] = {
        key: []
        for key in (
            "label",
            "score",
            "geometry_score",
            "texture_score",
            "geometry_weight",
            "texture_weight",
            "clip_index",
            "texture_invoked",
            "cache_age",
            "video_id",
            "method",
            "dataset",
        )
    }
    current_key: tuple[str, str, str] | None = None
    cached_embedding: torch.Tensor | None = None
    cached_logit: torch.Tensor | None = None
    cached_quality: torch.Tensor | None = None
    cache_age = 0
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"adaptive:{args.policy}"):
            batch = move_batch(batch, device)
            key = (batch["dataset"][0], batch["method"][0], batch["video_id"][0])
            if key != current_key:
                current_key = key
                cached_embedding = cached_logit = cached_quality = None
                cache_age = 0
            elif cached_embedding is not None:
                cache_age += 1

            geometry_embedding, geometry_logit = model.forward_geometry(batch["geometry"])
            invoked = False
            if args.policy != "geometry":
                if cached_embedding is None or args.policy == "always":
                    invoked = True
                elif args.policy == "periodic":
                    invoked = cache_age >= args.refresh_interval
                else:
                    invoked = bool(
                        refresh_policy.should_refresh(
                            geometry_logit, batch["geometry_quality"], cache_age
                        ).item()
                    )

            age_used = cache_age
            if invoked:
                cached_embedding, cached_logit = model.forward_texture(batch["texture"])
                cached_quality = batch["texture_quality"]
                cache_age = 0
                age_used = 0

            if args.policy == "geometry":
                fused_logit = geometry_logit
                texture_logit = torch.zeros_like(geometry_logit)
                weights = torch.tensor([1.0, 0.0], device=device).repeat(1, 1)
            else:
                assert cached_embedding is not None
                assert cached_logit is not None
                assert cached_quality is not None
                texture_logit = cached_logit
                fused_logit, weights = model.fuse_precomputed(
                    geometry_embedding,
                    geometry_logit,
                    cached_embedding,
                    cached_logit,
                    batch["geometry_quality"],
                    cached_quality,
                )

            clip_rows["label"].append(float(batch["label"].item()))
            clip_rows["score"].append(float(torch.sigmoid(fused_logit).item()))
            clip_rows["geometry_score"].append(float(torch.sigmoid(geometry_logit).item()))
            clip_rows["texture_score"].append(float(torch.sigmoid(texture_logit).item()))
            clip_rows["geometry_weight"].append(float(weights[0, 0].item()))
            clip_rows["texture_weight"].append(float(weights[0, 1].item()))
            clip_rows["clip_index"].append(int(batch["clip_index"].item()))
            clip_rows["texture_invoked"].append(int(invoked))
            clip_rows["cache_age"].append(age_used)
            clip_rows["video_id"].append(key[2])
            clip_rows["method"].append(key[1])
            clip_rows["dataset"].append(key[0])

    predictions = aggregate_predictions(clip_rows, aggregation, top_k)
    labels = np.asarray(predictions["label"], dtype=np.int64)
    scores = np.asarray(predictions["score"], dtype=np.float64)
    if args.select_threshold:
        threshold = select_threshold(labels, scores)
        threshold_source = "selected_on_current_manifest"
        threshold_selection = {
            "datasets": manifest_datasets,
            "splits": manifest_splits,
        }
    elif args.threshold_json:
        threshold, threshold_selection = _load_threshold(args.threshold_json)
        threshold_source = str(args.threshold_json)
    else:
        threshold = float(checkpoint["threshold"])
        threshold_source = "always_on_ffpp_validation_checkpoint"
        threshold_selection = {"datasets": ["ffpp"], "splits": ["val"]}
    metrics = compute_metrics(labels, scores, threshold)
    texture_calls = int(sum(clip_rows["texture_invoked"]))
    report = {
        "metrics": metrics,
        "threshold_source": threshold_source,
        "threshold_selection": threshold_selection,
        "policy": args.policy,
        "clips_per_video": clips_per_video,
        "aggregation": aggregation,
        "top_k": top_k,
        "texture_calls": texture_calls,
        "total_clips": len(clip_rows["score"]),
        "texture_invocation_ratio": texture_calls / max(len(clip_rows["score"]), 1),
        "refresh_interval": args.refresh_interval,
        "max_cache_age": args.max_cache_age,
        "uncertainty_threshold": args.uncertainty_threshold,
        "min_landmark_ratio": args.min_landmark_ratio,
    }
    if args.budget_reference:
        with Path(args.budget_reference).open("r", encoding="utf-8") as handle:
            reference_report = json.load(handle)
        if reference_report.get("policy") != "rule":
            raise ValueError("--budget-reference must point to a rule-policy report")
        if int(reference_report.get("total_clips", -1)) != len(clip_rows["score"]):
            raise ValueError("Budget reference and periodic run contain different numbers of clips")
        reference_ratio = float(reference_report["texture_invocation_ratio"])
        budget_delta = abs(report["texture_invocation_ratio"] - reference_ratio)
        report["budget_reference"] = str(args.budget_reference)
        report["budget_reference_ratio"] = reference_ratio
        report["budget_absolute_delta"] = budget_delta
        report["budget_matched"] = budget_delta <= args.budget_tolerance
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "adaptive_clip_predictions.csv", clip_rows)
    _write_csv(output_dir / "adaptive_predictions.csv", predictions)
    with (output_dir / "adaptive_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.budget_reference and not report["budget_matched"]:
        raise RuntimeError(
            "Periodic baseline does not match the rule-policy texture-call budget: "
            f"absolute delta={report['budget_absolute_delta']:.4f}, "
            f"tolerance={args.budget_tolerance:.4f}. Adjust --refresh-interval."
        )


if __name__ == "__main__":
    main()

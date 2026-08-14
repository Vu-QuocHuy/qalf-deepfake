#!/usr/bin/env python3
"""Bootstrap video-level confidence intervals from aggregated predictions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qalf.metrics import compute_metrics


METRIC_NAMES = (
    "auc",
    "average_precision",
    "eer",
    "accuracy",
    "balanced_accuracy",
    "f1_fake",
    "f1_real",
    "f1_macro",
    "apcer",
    "bpcer",
    "acer",
)


def _load_predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Prediction file is empty: {path}")
    required = {"label", "score"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Prediction file is missing columns: {sorted(missing)}")
    labels = np.asarray([int(float(row["label"])) for row in rows], dtype=np.int64)
    scores = np.asarray([float(row["score"]) for row in rows], dtype=np.float64)
    if np.unique(labels).size != 2:
        raise ValueError("Bootstrap CI requires both real and fake videos")
    return labels, scores


def _point_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    metrics = compute_metrics(labels, scores, threshold)
    return {name: float(metrics[name]) for name in METRIC_NAMES}


def _bootstrap(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    repetitions: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    samples: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}
    rng = np.random.default_rng(seed)
    size = labels.size
    for _ in range(repetitions):
        indices = rng.integers(0, size, size=size)
        sampled_labels = labels[indices]
        if np.unique(sampled_labels).size != 2:
            continue
        sampled_scores = scores[indices]
        sampled_metrics = compute_metrics(sampled_labels, sampled_scores, threshold)
        for name in METRIC_NAMES:
            samples[name].append(float(sampled_metrics[name]))

    result: dict[str, dict[str, float]] = {}
    for name, values in samples.items():
        if not values:
            raise RuntimeError(f"No valid bootstrap samples for {name}")
        result[name] = {
            "mean": float(np.mean(values)),
            "ci95_low": float(np.percentile(values, 2.5)),
            "ci95_high": float(np.percentile(values, 97.5)),
            "valid_repetitions": float(len(values)),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.repetitions < 100:
        parser.error("--repetitions must be at least 100")

    prediction_path = Path(args.predictions)
    labels, scores = _load_predictions(prediction_path)
    metrics_path = Path(args.metrics) if args.metrics else prediction_path.parent / "metrics.json"
    threshold = args.threshold
    protocol: dict[str, object] = {}
    if metrics_path.is_file():
        with metrics_path.open(encoding="utf-8") as handle:
            metrics_payload = json.load(handle)
        threshold = float(threshold if threshold is not None else metrics_payload["metrics"]["threshold"])
        protocol = dict(metrics_payload.get("protocol", {}))
    if threshold is None:
        raise ValueError("Provide --threshold or a metrics.json beside predictions.csv")

    point = _point_metrics(labels, scores, threshold)
    intervals = _bootstrap(labels, scores, threshold, args.repetitions, args.seed)
    payload = {
        "protocol": {
            "predictions": str(prediction_path),
            "metrics": str(metrics_path),
            "unit": "video_after_clip_aggregation",
            "sample_count": int(labels.size),
            "threshold": float(threshold),
            "bootstrap_repetitions": args.repetitions,
            "bootstrap_seed": args.seed,
            "source_protocol": protocol,
        },
        "point_estimate": point,
        "bootstrap_95_ci": intervals,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# QALF bootstrap confidence intervals",
        "",
        "Bootstrap resampling is performed at the video level after clip aggregation.",
        "The threshold is fixed from FF++ validation and is not reselected per sample.",
        "",
        f"- videos: `{labels.size}`",
        f"- threshold: `{threshold:.6f}`",
        f"- repetitions: `{args.repetitions}`",
        f"- seed: `{args.seed}`",
        "",
        "| Metric | Point estimate | 95% CI |",
        "| --- | ---: | ---: |",
    ]
    for name, value in point.items():
        interval = intervals[name]
        lines.append(
            f"| {name} | {value:.4f} | [{interval['ci95_low']:.4f}, {interval['ci95_high']:.4f}] |"
        )
    output_path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Bootstrap CI JSON: {output_path}")
    print(f"Bootstrap CI Markdown: {output_path.with_suffix('.md')}")


if __name__ == "__main__":
    main()

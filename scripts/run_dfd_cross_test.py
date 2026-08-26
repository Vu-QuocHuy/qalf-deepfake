#!/usr/bin/env python3
"""Evaluate the five baseline seeds on DFD and summarize their metrics."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = (0, 17, 42, 73, 123)
METRICS = (
    "auc",
    "average_precision",
    "accuracy",
    "balanced_accuracy",
    "f1_macro",
    "eer",
    "apcer",
    "bpcer",
    "acer",
)
PRIMARY_METRICS = ("auc", "accuracy", "balanced_accuracy", "f1_macro", "eer")


def _mean_std(values: Iterable[float]) -> tuple[float, float]:
    items = list(values)
    if not items:
        raise ValueError("Cannot summarize an empty metric list")
    return statistics.mean(items), statistics.stdev(items) if len(items) > 1 else 0.0


def _output_directory(root: Path, seed: int) -> Path:
    return root / f"baseline_seed{seed}_to_dfd_8f_3clips_mean_eer_tta"


def _load_run(metrics_path: Path, seed: int) -> dict[str, Any]:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics")
    protocol = payload.get("protocol")
    if not isinstance(metrics, dict) or not isinstance(protocol, dict):
        raise ValueError(f"Malformed evaluation payload: {metrics_path}")
    if protocol.get("datasets") != ["dfd"]:
        raise ValueError(
            f"Unexpected dataset protocol in {metrics_path}: {protocol.get('datasets')}"
        )
    missing = [name for name in METRICS if name not in metrics]
    if missing:
        raise ValueError(f"Missing metrics {missing} in {metrics_path}")
    row: dict[str, Any] = {
        "seed": seed,
        "metrics_path": str(metrics_path),
        "sample_count": int(metrics["sample_count"]),
        "real_count": int(metrics["real_count"]),
        "fake_count": int(metrics["fake_count"]),
        "threshold": float(metrics["threshold"]),
    }
    row.update({name: float(metrics[name]) for name in METRICS})
    return row


def _summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        raise ValueError("No DFD evaluation runs were found")
    runs = sorted(runs, key=lambda row: int(row["seed"]))
    counts = {
        (int(row["sample_count"]), int(row["real_count"]), int(row["fake_count"]))
        for row in runs
    }
    if len(counts) != 1:
        raise ValueError(f"Inconsistent DFD class counts across seeds: {sorted(counts)}")
    sample_count, real_count, fake_count = counts.pop()
    summary: dict[str, Any] = {
        "n_seeds": len(runs),
        "seeds": [int(row["seed"]) for row in runs],
        "sample_count_per_seed": sample_count,
        "real_count_per_seed": real_count,
        "fake_count_per_seed": fake_count,
    }
    for name in METRICS:
        mean, std = _mean_std(float(row[name]) for row in runs)
        summary[f"{name}_mean"] = mean
        summary[f"{name}_std"] = std
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _percent(mean: float, std: float) -> str:
    return f"{100.0 * mean:.2f} ± {100.0 * std:.2f}"


def _render_markdown(summary: dict[str, Any], runs: list[dict[str, Any]]) -> str:
    lines = [
        "# DFD cross-dataset summary",
        "",
        "Metrics are mean ± sample standard deviation across baseline training seeds.",
        "The checkpoint is trained on FF++ and the decision threshold is calibrated on FF++ validation.",
        "",
        "## Per-seed results",
        "",
        "| Seed | Videos | Real | Fake | AUC (%) | Accuracy (%) | Balanced Acc. (%) | Macro-F1 (%) | EER (%) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in runs:
        lines.append(
            "| {seed} | {sample_count} | {real_count} | {fake_count} | "
            "{auc:.2f} | {accuracy:.2f} | {balanced:.2f} | {f1:.2f} | {eer:.2f} |".format(
                seed=row["seed"],
                sample_count=row["sample_count"],
                real_count=row["real_count"],
                fake_count=row["fake_count"],
                auc=100.0 * row["auc"],
                accuracy=100.0 * row["accuracy"],
                balanced=100.0 * row["balanced_accuracy"],
                f1=100.0 * row["f1_macro"],
                eer=100.0 * row["eer"],
            )
        )
    lines.extend(
        [
            "",
            "## Mean ± sample standard deviation",
            "",
            f"- Videos/seed: {summary['sample_count_per_seed']} "
            f"({summary['real_count_per_seed']} real, {summary['fake_count_per_seed']} fake)",
            f"- AUC: {_percent(summary['auc_mean'], summary['auc_std'])}",
            f"- Accuracy: {_percent(summary['accuracy_mean'], summary['accuracy_std'])}",
            f"- Balanced Accuracy: {_percent(summary['balanced_accuracy_mean'], summary['balanced_accuracy_std'])}",
            f"- Macro-F1: {_percent(summary['f1_macro_mean'], summary['f1_macro_std'])}",
            f"- EER: {_percent(summary['eer_mean'], summary['eer_std'])}",
            "",
            "Average precision is retained in summary.json/summary.csv but is not used as the headline metric because DFD is class-imbalanced.",
        ]
    )
    return "\n".join(lines) + "\n"


def _evaluate_seed(args: argparse.Namespace, seed: int, output_root: Path) -> Path:
    checkpoint = Path(args.ablation_root) / f"baseline_seed{seed}" / "best.pt"
    output_dir = _output_directory(output_root, seed)
    metrics_path = output_dir / "metrics.json"
    if metrics_path.is_file() and not args.force_eval:
        print(f"[{seed}] evaluation exists; skipping")
        return metrics_path
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found for seed {seed}: {checkpoint}")
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "evaluate.py"),
        "--checkpoint",
        str(checkpoint),
        "--manifest",
        str(args.dfd_manifest),
        "--frame-root",
        str(args.dfd_root),
        "--landmark-root",
        str(args.dfd_root),
        "--output-dir",
        str(output_dir),
        "--batch-size",
        "8",
        "--num-workers",
        "4",
        "--clips-per-video",
        "3",
        "--aggregation",
        "mean",
        "--top-k",
        "1",
        "--texture-frames",
        "8",
        "--threshold-manifest",
        str(args.ffpp_val_manifest),
        "--threshold-frame-root",
        str(args.ffpp_frame_root),
        "--threshold-landmark-root",
        str(args.ffpp_landmark_root),
        "--threshold-clips-per-video",
        "3",
        "--threshold-selection",
        "eer",
    ]
    if not args.no_flip_tta:
        command.append("--texture-flip-tta")
    print(f"[{seed}] evaluating DFD")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Evaluation did not produce metrics.json: {metrics_path}")
    return metrics_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-root", required=True)
    parser.add_argument("--dfd-manifest", required=True)
    parser.add_argument("--dfd-root", required=True)
    parser.add_argument("--ffpp-val-manifest", required=True)
    parser.add_argument("--ffpp-frame-root", required=True)
    parser.add_argument("--ffpp-landmark-root", required=True)
    parser.add_argument("--output-root")
    parser.add_argument("--summary-dir")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--force-eval", action="store_true")
    parser.add_argument("--no-flip-tta", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds contains duplicates")
    args.ablation_root = Path(args.ablation_root)
    args.dfd_manifest = Path(args.dfd_manifest)
    args.dfd_root = Path(args.dfd_root)
    args.ffpp_val_manifest = Path(args.ffpp_val_manifest)
    args.ffpp_frame_root = Path(args.ffpp_frame_root)
    args.ffpp_landmark_root = Path(args.ffpp_landmark_root)
    output_root = Path(args.output_root) if args.output_root else args.ablation_root
    summary_dir = Path(args.summary_dir) if args.summary_dir else output_root / "dfd_summary"
    for required in (
        args.dfd_manifest,
        args.dfd_root,
        args.ffpp_val_manifest,
        args.ffpp_frame_root,
        args.ffpp_landmark_root,
    ):
        if not required.exists():
            raise FileNotFoundError(f"Required DFD evaluation input is missing: {required}")

    metrics_paths = [_evaluate_seed(args, seed, output_root) for seed in args.seeds]
    runs = [_load_run(path, seed) for path, seed in zip(metrics_paths, args.seeds, strict=True)]
    summary = _summarize(runs)
    summary_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(summary_dir / "runs.csv", runs)
    _write_csv(summary_dir / "summary.csv", [summary])
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "dataset": "dfd",
            "seeds": list(args.seeds),
            "texture_frames": 8,
            "clips_per_video": 3,
            "aggregation": "mean",
            "threshold_selection": "eer_ffpp_validation",
            "flip_tta": not args.no_flip_tta,
        },
        "summary": summary,
        "runs": runs,
    }
    (summary_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (summary_dir / "summary.md").write_text(
        _render_markdown(summary, runs), encoding="utf-8"
    )
    print(f"DFD cross-test complete: {len(runs)} seeds")
    print(f"Summary: {summary_dir / 'summary.md'}")


if __name__ == "__main__":
    main()

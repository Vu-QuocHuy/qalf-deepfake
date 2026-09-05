#!/usr/bin/env python3
"""Run the cached-input PyTorch leg of Table IV(a) for baseline seeds.

For each checkpoint this script evaluates the server PyTorch model on the
supplied cached Celeb-DF frames and landmarks, exports the exact checkpoint to
ONNX, and writes a reproducible AUC summary.  Copy the ONNX files and the same
cached input to Pi4 for the companion ONNX leg.
"""

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


def _mean_std(values: Iterable[float]) -> tuple[float, float]:
    values = list(values)
    if not values:
        raise ValueError("Cannot summarize an empty result set")
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def _load_run(metrics_path: Path, seed: int) -> dict[str, Any]:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"Malformed metrics payload: {metrics_path}")
    required = ("auc", "sample_count", "real_count", "fake_count")
    missing = [key for key in required if key not in metrics]
    if missing:
        raise ValueError(f"Missing {missing} in {metrics_path}")
    return {
        "seed": seed,
        "auc": float(metrics["auc"]),
        "sample_count": int(metrics["sample_count"]),
        "real_count": int(metrics["real_count"]),
        "fake_count": int(metrics["fake_count"]),
        "metrics_path": str(metrics_path),
    }


def _summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        raise ValueError("No seed results to summarize")
    runs = sorted(runs, key=lambda row: int(row["seed"]))
    counts = {(row["sample_count"], row["real_count"], row["fake_count"]) for row in runs}
    if len(counts) != 1:
        raise ValueError(f"Inconsistent video/class counts across seeds: {sorted(counts)}")
    sample_count, real_count, fake_count = counts.pop()
    auc_mean, auc_std = _mean_std(float(row["auc"]) for row in runs)
    return {
        "n_seeds": len(runs),
        "seeds": [int(row["seed"]) for row in runs],
        "sample_count_per_seed": sample_count,
        "real_count_per_seed": real_count,
        "fake_count_per_seed": fake_count,
        "auc_mean": auc_mean,
        "auc_std": auc_std,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _evaluate_seed(args: argparse.Namespace, seed: int) -> tuple[Path, Path]:
    checkpoint = args.ablation_root / f"baseline_seed{seed}" / "best.pt"
    output_dir = args.output_root / f"baseline_seed{seed}_table_iva_server"
    metrics_path = output_dir / "metrics.json"
    onnx_path = args.onnx_output_dir / f"baseline_seed{seed}.onnx"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not onnx_path.is_file() or args.force_export:
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [sys.executable, "scripts/export_onnx.py", "--checkpoint", str(checkpoint),
             "--output", str(onnx_path), "--verify"],
            cwd=PROJECT_ROOT, check=True,
        )
    if not metrics_path.is_file() or args.force_eval:
        output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable, "scripts/evaluate.py", "--checkpoint", str(checkpoint),
            "--manifest", str(args.celebdf_manifest), "--frame-root", str(args.celebdf_frame_root),
            "--landmark-root", str(args.celebdf_landmark_root), "--output-dir", str(output_dir),
            "--batch-size", str(args.batch_size), "--num-workers", str(args.num_workers),
            "--clips-per-video", "3", "--aggregation", "mean", "--top-k", "1",
            "--texture-frames", "8", "--threshold-manifest", str(args.ffpp_val_manifest),
            "--threshold-frame-root", str(args.ffpp_frame_root),
            "--threshold-landmark-root", str(args.ffpp_landmark_root),
            "--threshold-clips-per-video", "3", "--threshold-selection", "eer",
        ]
        if not args.no_flip_tta:
            command.append("--texture-flip-tta")
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Evaluation did not produce: {metrics_path}")
    return metrics_path, onnx_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-root", required=True, type=Path)
    parser.add_argument("--celebdf-manifest", required=True, type=Path)
    parser.add_argument("--celebdf-frame-root", required=True, type=Path)
    parser.add_argument("--celebdf-landmark-root", required=True, type=Path)
    parser.add_argument("--ffpp-val-manifest", required=True, type=Path)
    parser.add_argument("--ffpp-frame-root", required=True, type=Path)
    parser.add_argument("--ffpp-landmark-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--onnx-output-dir", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-flip-tta", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    parser.add_argument("--force-export", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds contains duplicates")
    for required in (args.celebdf_manifest, args.celebdf_frame_root, args.celebdf_landmark_root,
                     args.ffpp_val_manifest, args.ffpp_frame_root, args.ffpp_landmark_root):
        if not required.exists():
            raise FileNotFoundError(f"Required input is missing: {required}")
    metrics_paths, onnx_paths = zip(*(_evaluate_seed(args, seed) for seed in args.seeds), strict=True)
    runs = [_load_run(path, seed) for path, seed in zip(metrics_paths, args.seeds, strict=True)]
    summary = _summarize(runs)
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_root / "table_iva_server_runs.csv", runs)
    payload = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "protocol": {
        "seeds": args.seeds, "input": "cached frames + landmarks", "texture_frames": 8,
        "clips_per_video": 3, "aggregation": "mean", "flip_tta": not args.no_flip_tta,
        "threshold_selection": "eer_ffpp_validation", "onnx_paths": [str(path) for path in onnx_paths],
    }, "summary": summary, "runs": runs}
    (args.output_root / "table_iva_server_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Server Table IV(a) complete: AUC = {100 * summary['auc_mean']:.2f} ± {100 * summary['auc_std']:.2f}%")


if __name__ == "__main__":
    main()

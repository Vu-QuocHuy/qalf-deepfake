#!/usr/bin/env python3
"""Run the cached-input ONNX/Pi4 leg of Table IV(a) for baseline seeds."""

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
    return {"seed": seed, "auc": float(metrics["auc"]),
            "sample_count": int(metrics["sample_count"]), "real_count": int(metrics["real_count"]),
            "fake_count": int(metrics["fake_count"]), "metrics_path": str(metrics_path)}


def _summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        raise ValueError("No seed results to summarize")
    runs = sorted(runs, key=lambda row: int(row["seed"]))
    counts = {(row["sample_count"], row["real_count"], row["fake_count"]) for row in runs}
    if len(counts) != 1:
        raise ValueError(f"Inconsistent video/class counts across seeds: {sorted(counts)}")
    sample_count, real_count, fake_count = counts.pop()
    auc_mean, auc_std = _mean_std(float(row["auc"]) for row in runs)
    return {"n_seeds": len(runs), "seeds": [int(row["seed"]) for row in runs],
            "sample_count_per_seed": sample_count, "real_count_per_seed": real_count,
            "fake_count_per_seed": fake_count, "auc_mean": auc_mean, "auc_std": auc_std}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _evaluate_seed(args: argparse.Namespace, seed: int) -> Path:
    onnx_path = args.onnx_dir / f"baseline_seed{seed}.onnx"
    output_dir = args.output_root / f"baseline_seed{seed}_table_iva_pi4"
    metrics_path = output_dir / "metrics.json"
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    if metrics_path.is_file() and not args.force_eval:
        print(f"[{seed}] result exists; skipping")
        return metrics_path
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "scripts/evaluate_pi4.py", "--manifest", str(args.manifest),
        "--frame-root", str(args.frame_root), "--landmark-root", str(args.landmark_root),
        "--onnx", str(onnx_path), "--output-dir", str(output_dir), "--num-frames", "32",
        "--texture-frames", "8", "--clips-per-video", "3", "--aggregation", "mean",
        "--top-k", "1", "--image-size", "160", "--cpu-threads", str(args.cpu_threads),
    ]
    if args.no_flip_tta:
        command.append("--no-flip-tta")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Evaluation did not produce: {metrics_path}")
    return metrics_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--frame-root", required=True, type=Path)
    parser.add_argument("--landmark-root", required=True, type=Path)
    parser.add_argument("--onnx-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--no-flip-tta", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds contains duplicates")
    for required in (args.manifest, args.frame_root, args.landmark_root, args.onnx_dir):
        if not required.exists():
            raise FileNotFoundError(f"Required input is missing: {required}")
    metrics_paths = [_evaluate_seed(args, seed) for seed in args.seeds]
    runs = [_load_run(path, seed) for path, seed in zip(metrics_paths, args.seeds, strict=True)]
    summary = _summarize(runs)
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_root / "table_iva_pi4_runs.csv", runs)
    payload = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "protocol": {
        "seeds": args.seeds, "input": "cached frames + landmarks", "texture_frames": 8,
        "clips_per_video": 3, "aggregation": "mean", "flip_tta": not args.no_flip_tta,
        "cpu_threads": args.cpu_threads,
    }, "summary": summary, "runs": runs}
    (args.output_root / "table_iva_pi4_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Pi4 Table IV(a) complete: AUC = {100 * summary['auc_mean']:.2f} ± {100 * summary['auc_std']:.2f}%")


if __name__ == "__main__":
    main()

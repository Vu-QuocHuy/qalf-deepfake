#!/usr/bin/env python3
"""Ablation study automation script.

Generates commands or automatically runs the training for all 2^3 = 8
ablation configurations of QALF v2:

  A0  baseline       (no modules)
  A1  freq_only      (+Frequency)
  A2  ms_only        (+MultiScale)
  A3  ta_only        (+TemporalAttention)
  A4  freq_ms        (+Freq +MS)
  A5  freq_ta        (+Freq +TA)
  A6  ms_ta          (+MS +TA)
  A7  full_v2        (+Freq +MS +TA)

All hyper-parameters besides the three architecture flags are kept constant
so the comparison is a controlled experiment.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


# Full factorial 2^3 design ---------------------------------------------------

ABLATIONS: list[dict[str, object]] = [
    {"name": "baseline",  "flags": ["--no-frequency-preprocess", "--no-multiscale", "--no-temporal-attention"]},
    {"name": "freq_only", "flags": ["--frequency-preprocess", "--no-multiscale", "--no-temporal-attention"]},
    {"name": "ms_only",   "flags": ["--no-frequency-preprocess", "--multiscale", "--no-temporal-attention"]},
    {"name": "ta_only",   "flags": ["--no-frequency-preprocess", "--no-multiscale", "--temporal-attention"]},
    # {"name": "freq_ms",   "flags": ["--frequency-preprocess", "--multiscale", "--no-temporal-attention"]},
    # {"name": "freq_ta",   "flags": ["--frequency-preprocess", "--no-multiscale", "--temporal-attention"]},
    # {"name": "ms_ta",     "flags": ["--no-frequency-preprocess", "--multiscale", "--temporal-attention"]},
    # {"name": "full_v2",   "flags": ["--frequency-preprocess", "--multiscale", "--temporal-attention"]},
]


def _is_completed(output_dir: Path) -> bool:
    """Check whether a previous run already finished successfully."""
    summary = output_dir / "training_summary.json"
    if not summary.is_file():
        return False
    try:
        with summary.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return str(data.get("status", "")).lower() == "complete"
    except (json.JSONDecodeError, OSError):
        return False


def _print_summary(root: Path) -> None:
    """Print a Markdown comparison table from completed ablation results."""
    rows: list[dict[str, object]] = []
    for ablation in ABLATIONS:
        name = str(ablation["name"])
        summary_path = root / f"ablation_{name}" / "training_summary.json"
        if not summary_path.is_file():
            continue
        try:
            with summary_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (json.JSONDecodeError, OSError):
            continue
        metrics = data.get("best_validation_metrics", {})
        rows.append({
            "name": name,
            "auc": metrics.get("auc", float("nan")),
            "ap": metrics.get("average_precision", float("nan")),
            "eer": metrics.get("eer", float("nan")),
            "bal_acc": metrics.get("balanced_accuracy", float("nan")),
            "best_epoch": data.get("best_epoch", "?"),
            "epochs": data.get("completed_epochs", "?"),
            "duration_m": data.get("duration_seconds", 0) / 60.0,
        })

    if not rows:
        print("\nNo completed ablation results found.")
        return

    baseline_auc = next(
        (float(r["auc"]) for r in rows if r["name"] == "baseline"), float("nan")
    )

    print("\n" + "=" * 90)
    print("ABLATION STUDY RESULTS")
    print("=" * 90)
    header = (
        f"{'Config':<12} {'AUC':>7} {'AP':>7} {'EER':>7} "
        f"{'BalAcc':>7} {'ΔAUC':>7} {'Epoch':>6} {'Time':>8}"
    )
    print(header)
    print("-" * 90)
    for row in rows:
        auc = float(row["auc"])
        delta = auc - baseline_auc if baseline_auc == baseline_auc else float("nan")
        print(
            f"{row['name']:<12} {auc:7.4f} {float(row['ap']):7.4f} "
            f"{float(row['eer']):7.4f} {float(row['bal_acc']):7.4f} "
            f"{delta:+7.4f} {row['best_epoch']:>3}/{row['epochs']:<3} "
            f"{float(row['duration_m']):6.1f}m"
        )
    print("=" * 90)


def run_command(cmd: list[str], env: dict[str, str]) -> None:
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run QALF v2 ablation study (full 2^3 factorial design)."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without running")
    parser.add_argument("--summary-only", action="store_true",
                        help="Only print the summary table from existing results")
    parser.add_argument("--resume", action="store_true",
                        help="Skip ablations whose training_summary.json shows 'complete'")
    parser.add_argument("--python", default="python",
                        help="Path to Python executable")
    parser.add_argument("--config", required=True,
                        help="Base config JSON path")
    parser.add_argument("--output-dir", required=True,
                        help="Root directory for ablation outputs")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--frame-root", required=True)
    parser.add_argument("--landmark-root", required=True)
    # Shared hyper-parameters (forwarded to train.py)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.0002)
    parser.add_argument("--backbone-learning-rate", type=float, default=0.00002)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--early-stop-patience", type=int, default=7)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--texture-frames", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--embedding-dim", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    args = parser.parse_args()

    root = Path(args.output_dir)

    if args.summary_only:
        _print_summary(root)
        return

    env = os.environ.copy()
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # Shared flags that are identical across all ablation runs.
    shared_flags = [
        "--seed", str(args.seed),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--learning-rate", str(args.learning_rate),
        "--backbone-learning-rate", str(args.backbone_learning_rate),
        "--weight-decay", str(args.weight_decay),
        "--early-stop-patience", str(args.early_stop_patience),
        "--ema-decay", str(args.ema_decay),
        "--validation-weights", "ema",
        "--num-frames", str(args.num_frames),
        "--texture-frames", str(args.texture_frames),
        "--image-size", str(args.image_size),
        "--eval-clips-per-video", "3",
        "--fake-methods", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures",
        "--texture-backbone", "efficientnet_b0",
        "--texture-mode", "full_face",
        "--embedding-dim", str(args.embedding_dim),
        "--dropout", str(args.dropout),
        "--label-smoothing", str(args.label_smoothing),
        "--sbi",
        "--deterministic",
    ]

    completed = 0
    skipped = 0
    failed: list[str] = []

    for run_index, ablation in enumerate(ABLATIONS, 1):
        name = str(ablation["name"])
        flags: list[str] = list(ablation["flags"])  # type: ignore[arg-type]
        out_dir = root / f"ablation_{name}"

        if args.resume and _is_completed(out_dir):
            print(f"\n=== Ablation: {name} — SKIPPED (already complete) ===")
            skipped += 1
            continue

        train_cmd = [
            args.python, "scripts/train.py",
            "--config", args.config,
            "--train-manifest", args.train_manifest,
            "--val-manifest", args.val_manifest,
            "--frame-root", args.frame_root,
            "--landmark-root", args.landmark_root,
            "--output-dir", str(out_dir),
        ] + shared_flags + flags

        print(f"\n{'=' * 72}")
        print(f"  Ablation [{run_index}/{len(ABLATIONS)}]: {name}")
        print(f"  Modules: {' '.join(flags) if flags else '(none — baseline)'}")
        print(f"  Output:  {out_dir}")
        print(f"{'=' * 72}")

        if args.dry_run:
            print("CMD:", " \\\n     ".join(train_cmd))
        else:
            try:
                run_command(train_cmd, env)
                completed += 1
            except subprocess.CalledProcessError as error:
                print(f"ERROR: Ablation '{name}' failed with exit code {error.returncode}")
                failed.append(name)

    print(f"\n{'=' * 72}")
    print(f"Ablation study finished: {completed} completed, {skipped} skipped, {len(failed)} failed")
    if failed:
        print(f"Failed runs: {', '.join(failed)}")
    print(f"{'=' * 72}")

    if not args.dry_run:
        _print_summary(root)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Ablation study automation script.

Generates commands or automatically runs the training and evaluation for
the different ablation configurations of QALF v2:
1. Baseline (v1)
2. +Frequency Preprocess
3. +Multi-Scale Aggregation
4. +Temporal Attention
5. Full V2 (+Freq +MS +TA)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd: list[str], env: dict[str, str]) -> None:
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, env=env, check=True)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    parser.add_argument("--python", default="python", help="Path to Python executable")
    parser.add_argument("--config", required=True, help="Base config JSON path")
    parser.add_argument("--output-dir", required=True, help="Root directory for ablation outputs")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--frame-root", required=True)
    parser.add_argument("--landmark-root", required=True)
    args = parser.parse_args()

    ablations = [
        {"name": "baseline", "flags": []},
        {"name": "freq_only", "flags": ["--frequency-preprocess"]},
        {"name": "ms_only", "flags": ["--multiscale"]},
        {"name": "ta_only", "flags": ["--temporal-attention"]},
        {"name": "full_v2", "flags": ["--frequency-preprocess", "--multiscale", "--temporal-attention"]},
    ]

    env = os.environ.copy()
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    for config in ablations:
        name = config["name"]
        flags = config["flags"]
        
        out_dir = Path(args.output_dir) / f"ablation_{name}"
        
        train_cmd = [
            args.python, "scripts/train.py",
            "--config", args.config,
            "--train-manifest", args.train_manifest,
            "--val-manifest", args.val_manifest,
            "--frame-root", args.frame_root,
            "--landmark-root", args.landmark_root,
            "--output-dir", str(out_dir),
            "--batch-size", "6",
            "--num-workers", "4",
            "--sbi",
            "--deterministic"
        ] + flags

        print(f"\n=== Ablation: {name} ===")
        if args.dry_run:
            print("Train cmd:", " ".join(train_cmd))
        else:
            run_command(train_cmd, env)

if __name__ == "__main__":
    main()

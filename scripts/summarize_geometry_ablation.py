#!/usr/bin/env python3
"""Collect retained QALF and texture-only control metrics into one table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

EXPERIMENTS = (
    ("SBI baseline", "qalf_ffpp4_effb0_160_8f_full_face_sbi"),
    (
        "Geometry candidate",
        "qalf_ffpp4_effb0_160_8f_sbi_geometry_i3_attentive_reliability",
    ),
    (
        "Texture-only SBI control",
        "qalf_ffpp4_effb0_160_8f_full_face_sbi_texture_only",
    ),
)
EVALUATION_SUFFIX = "_to_celebdf_12f_3clips_mean_tta_ffpp_threshold"
FIELDS = (
    "ffpp_val_auc",
    "domain_gap",
    "auc",
    "average_precision",
    "eer",
    "balanced_accuracy",
    "acer",
    "geometry_auc",
    "texture_auc",
    "fixed_average_auc",
    "mean_geometry_weight",
    "zero_geometry_auc",
    "auc_gain_over_zero_geometry",
)
DERIVED_FIELDS = (
    "fusion_gain_over_texture",
    "fixed_average_gap",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-root", required=True)
    parser.add_argument("--output-prefix")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be non-negative")

    root = Path(args.experiments_root)
    experiment_suffix = "" if args.seed == 42 else f"_seed{args.seed}"
    rows: list[dict[str, str | float]] = []
    missing: list[Path] = []
    for profile, base_experiment in EXPERIMENTS:
        experiment = f"{base_experiment}{experiment_suffix}"
        metrics_path = root / f"{experiment}{EVALUATION_SUFFIX}" / "metrics.json"
        if not metrics_path.is_file():
            missing.append(metrics_path)
            continue
        with metrics_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        metrics = payload.get("metrics", payload)
        summary_path = root / experiment / "training_summary.json"
        ffpp_val_auc = float("nan")
        if summary_path.is_file():
            with summary_path.open("r", encoding="utf-8") as handle:
                training_summary = json.load(handle)
            ffpp_val_auc = float(training_summary.get("best_value", float("nan")))
        row: dict[str, str | float] = {
            "profile": profile,
            "experiment": experiment,
            "ffpp_val_auc": ffpp_val_auc,
            "domain_gap": ffpp_val_auc - float(metrics["auc"]),
        }
        for field in FIELDS:
            if field not in row:
                row[field] = float(metrics.get(field, float("nan")))
        row["fusion_gain_over_texture"] = float(metrics["auc"]) - float(metrics["texture_auc"])
        row["fixed_average_gap"] = float(metrics["fixed_average_auc"]) - float(
            metrics["texture_auc"]
        )
        rows.append(row)

    baseline_path = (
        root / f"{EXPERIMENTS[0][1]}{experiment_suffix}{EVALUATION_SUFFIX}" / "metrics.json"
    )
    if not baseline_path.is_file():
        raise SystemExit(
            f"SBI baseline evaluation is required before computing deltas: {baseline_path}"
        )
    if not rows:
        paths = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(f"No completed evaluation metrics found:\n{paths}")

    baseline_auc = float(rows[0]["auc"])
    for row in rows:
        row["auc_delta"] = float(row["auc"]) - baseline_auc

    default_report = f"p1_texture_control_comparison{experiment_suffix}"
    prefix = Path(args.output_prefix) if args.output_prefix else root / default_report
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["profile", "experiment", *FIELDS, *DERIVED_FIELDS, "auc_delta"]
    with prefix.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    header = (
        "| Profile | FF++ val AUC | Celeb-DF AUC | Delta | Domain gap | "
        "Geometry AUC | Texture AUC | Fusion gain | Fixed avg | Fixed avg gap | AP | EER | "
        "Balanced acc | ACER | Geometry weight | Zero-geometry AUC | Counterfactual gain |"
    )
    lines = [
        "# QALF P1 texture-control comparison",
        "",
        header,
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['profile']} | {float(row['ffpp_val_auc']):.4f} | "
            f"{float(row['auc']):.4f} | {float(row['auc_delta']):+.4f} | "
            f"{float(row['domain_gap']):.4f} | {float(row['geometry_auc']):.4f} | "
            f"{float(row['texture_auc']):.4f} | "
            f"{float(row['fusion_gain_over_texture']):+.4f} | "
            f"{float(row['fixed_average_auc']):.4f} | "
            f"{float(row['fixed_average_gap']):+.4f} | "
            f"{float(row['average_precision']):.4f} | {float(row['eer']):.4f} | "
            f"{float(row['balanced_accuracy']):.4f} | {float(row['acer']):.4f} | "
            f"{float(row['mean_geometry_weight']):.4f} | "
            f"{float(row['zero_geometry_auc']):.4f} | "
            f"{float(row['auc_gain_over_zero_geometry']):+.4f} |"
        )
    if missing:
        lines.extend(
            [
                "",
                "## Missing evaluations",
                "",
                *(f"- `{path}`" for path in missing),
            ]
        )
    markdown = "\n".join(lines) + "\n"
    prefix.with_suffix(".md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"CSV: {prefix.with_suffix('.csv')}")
    print(f"Markdown: {prefix.with_suffix('.md')}")


if __name__ == "__main__":
    main()

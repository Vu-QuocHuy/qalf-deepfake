#!/usr/bin/env python3
"""Summarize the locked texture, geometry, and SRM comparison."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

EVALUATION_SUFFIX = "_to_celebdf_12f_3clips_mean_tta_ffpp_threshold"
PROFILES = (
    ("Texture-only SBI", "qalf_ffpp4_effb0_160_8f_full_face_sbi_texture_only", "none"),
    ("Geometry baseline A", "qalf_ffpp4_effb0_160_8f_full_face_sbi", "geometry"),
    ("SRM candidate", "qalf_ffpp4_effb0_160_8f_full_face_sbi_srm", "srm"),
)


def _load(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _metric(metrics: dict[str, object], current: str, legacy: str | None = None) -> float:
    if current in metrics:
        return float(metrics[current])
    if legacy is not None and legacy in metrics:
        return float(metrics[legacy])
    raise KeyError(f"Missing required metric: {current}")


def _branch_from_protocol(protocol: dict[str, object]) -> str:
    model = protocol.get("model", {})
    if "auxiliary_branch" in model:
        return str(model["auxiliary_branch"])
    return "none" if model.get("fusion_mode") == "texture" else "geometry"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.experiments_root)
    suffix = "" if args.seed == 42 else f"_seed{args.seed}"
    rows: list[dict[str, str | float]] = []
    for profile, base_experiment, expected_branch in PROFILES:
        experiment = f"{base_experiment}{suffix}"
        summary_path = root / experiment / "training_summary.json"
        metrics_path = root / f"{experiment}{EVALUATION_SUFFIX}" / "metrics.json"
        missing = [path for path in (summary_path, metrics_path) if not path.is_file()]
        if missing:
            raise SystemExit(
                "Missing required SRM comparison inputs:\n"
                + "\n".join(f"  - {path}" for path in missing)
            )
        summary = _load(summary_path)
        payload = _load(metrics_path)
        metrics = payload.get("metrics", payload)
        protocol = payload.get("protocol", {})
        branch = _branch_from_protocol(protocol)
        if branch != expected_branch:
            raise SystemExit(f"{profile}: expected auxiliary={expected_branch}, got {branch}")
        auc = _metric(metrics, "auc")
        texture_auc = _metric(metrics, "texture_auc")
        auxiliary_auc = 0.5 if branch == "none" else _metric(
            metrics, "auxiliary_auc", "geometry_auc"
        )
        auxiliary_weight = 0.0 if branch == "none" else _metric(
            metrics, "mean_auxiliary_weight", "mean_geometry_weight"
        )
        zero_auxiliary_auc = auc if branch == "none" else _metric(
            metrics, "zero_auxiliary_auc", "zero_geometry_auc"
        )
        counterfactual_gain = auc - zero_auxiliary_auc
        rows.append(
            {
                "profile": profile,
                "experiment": experiment,
                "auxiliary_branch": branch,
                "ffpp_val_auc": float(summary["best_value"]),
                "auc": auc,
                "average_precision": _metric(metrics, "average_precision"),
                "eer": _metric(metrics, "eer"),
                "balanced_accuracy": _metric(metrics, "balanced_accuracy"),
                "auxiliary_auc": auxiliary_auc,
                "texture_auc": texture_auc,
                "fusion_gain_over_texture": auc - texture_auc,
                "mean_auxiliary_weight": auxiliary_weight,
                "zero_auxiliary_auc": zero_auxiliary_auc,
                "counterfactual_gain": counterfactual_gain,
            }
        )

    texture_auc = float(rows[0]["auc"])
    for row in rows:
        row["domain_gap"] = float(row["ffpp_val_auc"]) - float(row["auc"])
        row["auc_gain_over_texture_control"] = float(row["auc"]) - texture_auc
    srm = rows[2]
    passes = (
        float(srm["auc_gain_over_texture_control"]) >= 0.005
        and float(srm["counterfactual_gain"]) >= 0.003
        and float(srm["mean_auxiliary_weight"]) >= 0.05
        and float(srm["fusion_gain_over_texture"]) > 0.0
    )
    if args.seed == 42:
        decision = (
            "SRM PASSES seed-42 gate: run seeds 17 and 73."
            if passes
            else "SRM FAILS seed-42 gate: do not run more seeds; inspect residual signal first."
        )
    else:
        decision = (
            f"Replication seed {args.seed} recorded. Apply the final claim to the combined "
            "three-seed result, not this row alone."
        )

    prefix = root / f"srm_ablation_seed{args.seed}"
    fieldnames = sorted({key for row in rows for key in row})
    with prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"# QALF SRM ablation — seed {args.seed}",
        "",
        "| Profile | FF++ val AUC | Celeb-DF AUC | vs texture control | Domain gap | "
        "Auxiliary AUC | Texture AUC | Fusion gain | Aux weight | Zero-aux AUC | "
        "Counterfactual gain | AP | EER | Bal acc |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['profile']} | {float(row['ffpp_val_auc']):.4f} | "
            f"{float(row['auc']):.4f} | "
            f"{float(row['auc_gain_over_texture_control']):+.4f} | "
            f"{float(row['domain_gap']):.4f} | "
            f"{float(row['auxiliary_auc']):.4f} | "
            f"{float(row['texture_auc']):.4f} | "
            f"{float(row['fusion_gain_over_texture']):+.4f} | "
            f"{float(row['mean_auxiliary_weight']):.4f} | "
            f"{float(row['zero_auxiliary_auc']):.4f} | "
            f"{float(row['counterfactual_gain']):+.4f} | "
            f"{float(row['average_precision']):.4f} | {float(row['eer']):.4f} | "
            f"{float(row['balanced_accuracy']):.4f} |"
        )
    lines.extend(("", "## Decision", "", decision))
    markdown = "\n".join(lines) + "\n"
    prefix.with_suffix(".md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"CSV: {prefix.with_suffix('.csv')}")
    print(f"Markdown: {prefix.with_suffix('.md')}")


if __name__ == "__main__":
    main()

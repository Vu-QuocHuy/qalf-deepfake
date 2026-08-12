#!/usr/bin/env python3
"""Summarize the locked A-C-D-E geometry gate failure diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

EXPERIMENTS = (
    ("A — SBI baseline", "qalf_ffpp4_effb0_160_8f_full_face_sbi", 0.0, 0.0, False),
    (
        "C — dropout only",
        "qalf_ffpp4_effb0_160_8f_sbi_geometry_dropout_only",
        0.15,
        0.0,
        False,
    ),
    (
        "D — dropout + reliability",
        "qalf_ffpp4_effb0_160_8f_sbi_geometry_i2_reliability",
        0.15,
        0.10,
        False,
    ),
    (
        "E — SBI-aware reliability",
        "qalf_ffpp4_effb0_160_8f_sbi_geometry_sbi_aware_reliability",
        0.15,
        0.10,
        True,
    ),
)
EVALUATION_SUFFIX = "_to_celebdf_12f_3clips_mean_tta_ffpp_threshold"


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _training_auc(root: Path, experiment: str) -> float:
    payload = _load_json(root / experiment / "training_summary.json")
    return float(payload["best_value"])


def _metric(metrics: dict[str, object], name: str) -> float:
    if name not in metrics:
        raise KeyError(f"evaluation is missing required diagnostic '{name}'")
    return float(metrics[name])


def _interpret(dropout_weight: float, dropout_gain: float) -> str:
    if dropout_weight < 0.01 and abs(dropout_gain) < 0.001:
        return (
            "DROP-ONLY COLLAPSE: modality dropout alone reproduces the failure. "
            "Do not add reliability; test SBI fusion isolation next."
        )
    if dropout_weight >= 0.05 and dropout_gain >= 0.003:
        return (
            "GEOMETRY PRESERVED: modality dropout alone does not reproduce the failure. "
            "Reliability supervision is the proximate suspect; inspect E below."
        )
    return (
        "MIXED RESULT: dropout weakens geometry but does not cleanly reproduce or avoid collapse. "
        "Inspect gate distributions and test SBI-aware loss routing before any three-seed run."
    )


def _interpret_sbi_aware(
    baseline_auc: float,
    sbi_aware_auc: float,
    geometry_weight: float,
    counterfactual_gain: float,
) -> str:
    auc_floor = baseline_auc - 0.005
    if (
        geometry_weight >= 0.05
        and counterfactual_gain >= 0.003
        and sbi_aware_auc >= auc_floor
    ):
        return (
            "E PASSES: SBI-aware routing restores usable geometry without an unacceptable "
            "clean-AUC loss. Confirm E on seeds 17 and 73 before making a claim."
        )
    return (
        "E FAILS: SBI-aware routing does not satisfy the pre-registered gate "
        f"(geometry weight >=0.05, counterfactual gain >=0.003, AUC >={auc_floor:.4f}). "
        "Reject reliability routing and continue with baseline A for robustness/SRM work."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-root", required=True)
    parser.add_argument("--output-prefix")
    args = parser.parse_args()

    root = Path(args.experiments_root)
    rows: list[dict[str, str | float]] = []
    for (
        profile,
        experiment,
        expected_dropout,
        expected_reliability,
        expected_sbi_aware,
    ) in EXPERIMENTS:
        metrics_path = root / f"{experiment}{EVALUATION_SUFFIX}" / "metrics.json"
        summary_path = root / experiment / "training_summary.json"
        missing = [path for path in (metrics_path, summary_path) if not path.is_file()]
        if missing:
            formatted = "\n".join(f"  - {path}" for path in missing)
            raise SystemExit(f"Missing required diagnostic inputs:\n{formatted}")

        payload = _load_json(metrics_path)
        metrics = payload.get("metrics", payload)
        protocol = payload.get("protocol", {})
        model_protocol = protocol.get("model", {})
        training_protocol = protocol.get("training_data", {})
        actual_dropout = float(model_protocol.get("modality_dropout_probability", 0.0))
        actual_reliability = float(training_protocol.get("reliability_gate_loss_weight", 0.0))
        actual_sbi_aware = bool(
            model_protocol.get("exclude_sbi_from_modality_dropout", False)
        )
        geometry_architecture = str(model_protocol.get("geometry_architecture", "tcn_mean"))
        fusion_mode = str(model_protocol.get("fusion_mode", "quality"))
        if abs(actual_dropout - expected_dropout) > 1e-8:
            raise SystemExit(
                f"{profile}: expected dropout={expected_dropout}, got {actual_dropout}"
            )
        if abs(actual_reliability - expected_reliability) > 1e-8:
            raise SystemExit(
                f"{profile}: expected reliability={expected_reliability}, got {actual_reliability}"
            )
        if actual_sbi_aware != expected_sbi_aware:
            raise SystemExit(
                f"{profile}: expected SBI-aware routing={expected_sbi_aware}, "
                f"got {actual_sbi_aware}"
            )
        if geometry_architecture != "tcn_mean" or fusion_mode != "quality":
            raise SystemExit(
                f"{profile}: expected tcn_mean/quality, got "
                f"{geometry_architecture}/{fusion_mode}"
            )

        row: dict[str, str | float] = {
            "profile": profile,
            "experiment": experiment,
            "modality_dropout_probability": actual_dropout,
            "reliability_gate_loss_weight": actual_reliability,
            "exclude_sbi_from_modality_dropout": actual_sbi_aware,
            "ffpp_val_auc": _training_auc(root, experiment),
        }
        for name in (
            "auc",
            "average_precision",
            "geometry_auc",
            "texture_auc",
            "mean_geometry_weight",
            "median_geometry_weight",
            "p90_geometry_weight",
            "mean_geometry_weight_real",
            "mean_geometry_weight_fake",
            "zero_geometry_auc",
            "auc_gain_over_zero_geometry",
            "mean_abs_zero_geometry_score_shift",
        ):
            row[name] = _metric(metrics, name)
        row["fusion_gain_over_texture"] = float(row["auc"]) - float(row["texture_auc"])
        rows.append(row)

    baseline_auc = float(rows[0]["auc"])
    for row in rows:
        row["auc_delta"] = float(row["auc"]) - baseline_auc

    dropout_row = rows[1]
    interpretation = _interpret(
        float(dropout_row["mean_geometry_weight"]),
        float(dropout_row["auc_gain_over_zero_geometry"]),
    )
    sbi_aware_row = rows[3]
    sbi_aware_interpretation = _interpret_sbi_aware(
        baseline_auc,
        float(sbi_aware_row["auc"]),
        float(sbi_aware_row["mean_geometry_weight"]),
        float(sbi_aware_row["auc_gain_over_zero_geometry"]),
    )
    prefix = (
        Path(args.output_prefix)
        if args.output_prefix
        else root / "geometry_failure_diagnostic_seed42"
    )
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with prefix.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# QALF geometry failure diagnostic — seed 42",
        "",
        "| Profile | Dropout | Reliability | SBI-aware | FF++ val AUC | Celeb-DF AUC | Delta | "
        "Geometry AUC | Texture AUC | Fusion gain | Geo weight | Median | P90 | "
        "Zero-geo AUC | Counterfactual gain | Mean score shift |",
        "| --- | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['profile']} | {float(row['modality_dropout_probability']):.2f} | "
            f"{float(row['reliability_gate_loss_weight']):.2f} | "
            f"{'yes' if bool(row['exclude_sbi_from_modality_dropout']) else 'no'} | "
            f"{float(row['ffpp_val_auc']):.4f} | {float(row['auc']):.4f} | "
            f"{float(row['auc_delta']):+.4f} | {float(row['geometry_auc']):.4f} | "
            f"{float(row['texture_auc']):.4f} | "
            f"{float(row['fusion_gain_over_texture']):+.4f} | "
            f"{float(row['mean_geometry_weight']):.4f} | "
            f"{float(row['median_geometry_weight']):.4f} | "
            f"{float(row['p90_geometry_weight']):.4f} | "
            f"{float(row['zero_geometry_auc']):.4f} | "
            f"{float(row['auc_gain_over_zero_geometry']):+.4f} | "
            f"{float(row['mean_abs_zero_geometry_score_shift']):.4f} |"
        )
    lines.extend(
        (
            "",
            "## Dropout diagnosis",
            "",
            interpretation,
            "",
            "## SBI-aware routing decision",
            "",
            sbi_aware_interpretation,
        )
    )
    markdown = "\n".join(lines) + "\n"
    prefix.with_suffix(".md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"CSV: {prefix.with_suffix('.csv')}")
    print(f"Markdown: {prefix.with_suffix('.md')}")


if __name__ == "__main__":
    main()

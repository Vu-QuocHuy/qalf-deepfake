#!/usr/bin/env python3
"""Summarize FF++ in-domain evaluation runs for registered ablation profiles."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any


METRICS = (
    "auc",
    "average_precision",
    "eer",
    "balanced_accuracy",
    "accuracy",
    "f1_fake",
    "f1_real",
    "f1_macro",
    "acer",
)


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _parse_run(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Expected PROFILE=EVAL_DIR, got: {spec!r}")
    profile, directory = spec.split("=", 1)
    if not profile or not directory:
        raise ValueError(f"Expected PROFILE=EVAL_DIR, got: {spec!r}")
    return profile, Path(directory)


def _method_seed(profile: str) -> tuple[str, str]:
    match = re.fullmatch(r"(.+)_seed(\d+)", profile)
    if match is None:
        return profile, "NA"
    return match.group(1), match.group(2)


def _fmt(value: object) -> str:
    return "NA" if value is None else f"{float(value):.4f}"


def _mean_std(rows: list[dict[str, Any]], key: str) -> tuple[float | None, float | None]:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None, None
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def _discover_runs(root: Path, threshold_selection: str | None) -> list[str]:
    """Find existing FF++ metrics without assuming the current output suffix."""
    specs: list[str] = []
    for metrics_path in sorted(root.rglob("metrics.json")):
        try:
            payload = _load(metrics_path)
        except (OSError, json.JSONDecodeError):
            continue
        protocol = payload.get("protocol", {})
        if "ffpp" not in {str(value) for value in protocol.get("datasets", [])}:
            continue
        selection = str(protocol.get("threshold", {}).get("selection", ""))
        if threshold_selection and threshold_selection not in selection:
            continue
        directory_name = metrics_path.parent.name
        match = re.match(r"(.+?)(?:_to)?_?ffpp_test(?:_|$)", directory_name)
        profile = match.group(1) if match else directory_name
        specs.append(f"{profile}={metrics_path.parent}")
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", action="append", metavar="PROFILE=DIR")
    source.add_argument("--discover-root", type=Path)
    parser.add_argument("--output-stem", required=True)
    parser.add_argument("--threshold-selection", choices=("youden_j", "eer"))
    args = parser.parse_args()

    specs = args.run or _discover_runs(args.discover_root, args.threshold_selection)
    if not specs:
        raise SystemExit(f"No FF++ metrics.json found under {args.discover_root}")

    rows: list[dict[str, Any]] = []
    for spec in specs:
        profile, eval_dir = _parse_run(spec)
        method, seed = _method_seed(profile)
        metrics_path = eval_dir / "metrics.json"
        row: dict[str, Any] = {
            "method": method,
            "seed": seed,
            "profile": profile,
            "eval_dir": str(eval_dir),
            "status": "complete",
        }
        if not metrics_path.is_file():
            row["status"] = "missing_metrics"
            rows.append(row)
            continue
        payload = _load(metrics_path)
        metrics = payload.get("metrics", {})
        protocol = payload.get("protocol", {})
        row.update({key: metrics.get(key) for key in METRICS})
        row["threshold"] = metrics.get("threshold")
        row["fake_method_filter"] = ",".join(protocol.get("fake_method_filter") or [])
        checkpoint = Path(protocol.get("checkpoint", ""))
        summary_path = checkpoint.parent / "training_summary.json"
        if summary_path.is_file():
            training = _load(summary_path)
            row["ffpp_val_auc"] = training.get("best_value")
            row["best_epoch"] = training.get("best_epoch")
            if row.get("ffpp_val_auc") is not None and row.get("auc") is not None:
                row["domain_gap"] = float(row["ffpp_val_auc"]) - float(row["auc"])
        rows.append(row)

    stem = Path(args.output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "method", "seed", "profile", "status", "best_epoch", "ffpp_val_auc", "auc", "domain_gap",
        "average_precision", "eer", "balanced_accuracy", "accuracy", "f1_fake", "f1_real", "f1_macro",
        "acer", "threshold", "fake_method_filter", "eval_dir",
    ]
    with stem.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# QALF FF++ in-domain ablation summary",
        "",
        "The evaluation split is the official FF++ test split. Thresholds are calibrated on FF++ validation only.",
        "FaceShifter is excluded; fake methods are listed per row.",
        "",
        "## Per-seed results",
        "",
        "| Method | Seed | Status | FF++ val AUC | FF++ test AUC | Gap | AP | EER | Accuracy | Balanced acc | F1 fake | F1 real | F1 macro | ACER |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['seed']} | {row['status']} | {_fmt(row.get('ffpp_val_auc'))} | "
            f"{_fmt(row.get('auc'))} | {_fmt(row.get('domain_gap'))} | "
            f"{_fmt(row.get('average_precision'))} | {_fmt(row.get('eer'))} | "
            f"{_fmt(row.get('accuracy'))} | {_fmt(row.get('balanced_accuracy'))} | "
            f"{_fmt(row.get('f1_fake'))} | {_fmt(row.get('f1_real'))} | {_fmt(row.get('f1_macro'))} | "
            f"{_fmt(row.get('acer'))} |"
        )

    complete = [row for row in rows if row["status"] == "complete"]
    grouped: list[dict[str, Any]] = []
    lines.extend(
        (
            "",
            "## Mean ± standard deviation by method",
            "",
            "| Method | Seeds | FF++ val AUC | FF++ test AUC | AP | EER | Accuracy | Balanced acc | F1 fake | F1 real | F1 macro | ACER |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        )
    )
    grouped_metrics = (
        "ffpp_val_auc", "auc", "average_precision", "eer", "accuracy",
        "balanced_accuracy", "f1_fake", "f1_real", "f1_macro", "acer",
    )
    for method in sorted({str(row["method"]) for row in complete}):
        selected = [row for row in complete if row["method"] == method]
        grouped_row: dict[str, Any] = {"method": method, "seeds": len(selected)}
        rendered: list[str] = []
        for key in grouped_metrics:
            mean, std = _mean_std(selected, key)
            grouped_row[f"{key}_mean"] = mean
            grouped_row[f"{key}_std"] = std
            rendered.append("NA" if mean is None else f"{mean:.4f} ± {std:.4f}")
        lines.append(
            f"| {method} | {len(selected)} | {rendered[0]} | {rendered[1]} | {rendered[2]} | "
            f"{rendered[3]} | {rendered[4]} | {rendered[5]} | {rendered[6]} | "
            f"{rendered[7]} | {rendered[8]} | {rendered[9]} |"
        )
        grouped.append(grouped_row)

    grouped_fields = ["method", "seeds"]
    for key in grouped_metrics:
        grouped_fields.extend((f"{key}_mean", f"{key}_std"))
    grouped_path = stem.with_name(f"{stem.name}_by_method").with_suffix(".csv")
    with grouped_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=grouped_fields)
        writer.writeheader()
        writer.writerows(grouped)

    lines.extend(
        (
            "",
            f"Per-seed CSV: `{stem.with_suffix('.csv')}`",
            f"Grouped CSV: `{grouped_path}`",
            "",
        )
    )
    stem.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Per-seed CSV: {stem.with_suffix('.csv')}")
    print(f"Grouped CSV: {grouped_path}")
    print(f"Markdown: {stem.with_suffix('.md')}")


if __name__ == "__main__":
    main()

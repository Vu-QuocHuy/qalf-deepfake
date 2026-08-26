#!/usr/bin/env python3
"""Summarize the paired SBI ablation on Celeb-DF and FF++ across seeds."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

DEFAULT_PROFILES = ("baseline", "sbi_frame", "no_sbi")
DEFAULT_SEEDS = (0, 17, 42, 73, 123)
METRICS = (
    "auc",
    "accuracy",
    "balanced_accuracy",
    "f1_macro",
    "eer",
    "apcer",
    "bpcer",
    "acer",
)
PRIMARY_METRICS = ("auc", "accuracy", "f1_macro", "eer")
PROFILE_LABELS = {
    "baseline": "Clip-coherent SBI",
    "sbi_frame": "Frame-independent SBI",
    "no_sbi": "No SBI",
}
DATASET_LABELS = {"celebdf": "Celeb-DF-v2", "ffpp": "FF++ c23"}


def _sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _mean_std(values: Iterable[float]) -> tuple[float, float]:
    items = list(values)
    if not items:
        raise ValueError("Cannot summarize an empty metric list")
    return statistics.mean(items), _sample_std(items)


def _expected_directory_name(
    dataset: str,
    profile: str,
    seed: int,
    *,
    texture_frames: int,
    clips_per_video: int,
    aggregation: str,
    threshold_selection: str,
    flip_tta: bool,
) -> str:
    tta_suffix = "tta" if flip_tta else "no_tta"
    if dataset == "celebdf":
        return (
            f"{profile}_seed{seed}_to_celebdf_{texture_frames}f_"
            f"{clips_per_video}clips_{aggregation}_{threshold_selection}_"
            f"{tta_suffix}_ffpp_threshold"
        )
    if dataset == "ffpp":
        return (
            f"{profile}_seed{seed}_ffpp_test_{texture_frames}f_"
            f"{clips_per_video}clips_{aggregation}_{threshold_selection}_{tta_suffix}"
        )
    raise ValueError(f"Unknown dataset: {dataset}")


def _load_run(
    metrics_path: Path,
    *,
    dataset: str,
    profile: str,
    seed: int,
) -> dict[str, Any]:
    with metrics_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    metrics = payload.get("metrics")
    protocol = payload.get("protocol")
    if not isinstance(metrics, dict) or not isinstance(protocol, dict):
        raise ValueError(f"Malformed metrics payload: {metrics_path}")
    missing_metrics = [name for name in METRICS if name not in metrics]
    if missing_metrics:
        raise ValueError(f"Missing metrics {missing_metrics} in {metrics_path}")

    expected_dataset = "celebdf_v2" if dataset == "celebdf" else "ffpp"
    protocol_datasets = protocol.get("datasets")
    if protocol_datasets != [expected_dataset]:
        raise ValueError(
            f"Unexpected dataset protocol in {metrics_path}: "
            f"expected {[expected_dataset]}, got {protocol_datasets}"
        )

    row: dict[str, Any] = {
        "dataset": dataset,
        "profile": profile,
        "seed": seed,
        "threshold": float(metrics["threshold"]),
        "sample_count": int(metrics["sample_count"]),
        "metrics_path": str(metrics_path),
        "bootstrap_ci_present": (metrics_path.parent / "bootstrap_ci.json").is_file(),
    }
    for name in METRICS:
        row[name] = float(metrics[name])
    return row


def collect_runs(
    *,
    celebdf_root: Path,
    ffpp_root: Path,
    profiles: list[str],
    seeds: list[int],
    texture_frames: int,
    clips_per_video: int,
    aggregation: str,
    threshold_selection: str,
    flip_tta: bool,
    allow_missing: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    runs: list[dict[str, Any]] = []
    missing: list[str] = []
    for dataset, root in (("celebdf", celebdf_root), ("ffpp", ffpp_root)):
        for profile in profiles:
            for seed in seeds:
                directory = _expected_directory_name(
                    dataset,
                    profile,
                    seed,
                    texture_frames=texture_frames,
                    clips_per_video=clips_per_video,
                    aggregation=aggregation,
                    threshold_selection=threshold_selection,
                    flip_tta=flip_tta,
                )
                metrics_path = root / directory / "metrics.json"
                if not metrics_path.is_file():
                    missing.append(str(metrics_path))
                    continue
                runs.append(
                    _load_run(
                        metrics_path,
                        dataset=dataset,
                        profile=profile,
                        seed=seed,
                    )
                )
    if missing and not allow_missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "The SBI summary is incomplete. Missing metrics files:\n"
            f"{formatted}\n"
            "Rerun the corresponding evaluations or pass --allow-missing for a draft."
        )
    return runs, missing


def summarize_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[(str(run["dataset"]), str(run["profile"]))].append(run)

    summary: list[dict[str, Any]] = []
    for (dataset, profile), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row["seed"]))
        sample_counts = {int(row["sample_count"]) for row in rows}
        if len(sample_counts) != 1:
            raise ValueError(
                f"Inconsistent sample counts for {dataset}/{profile}: {sorted(sample_counts)}"
            )
        item: dict[str, Any] = {
            "dataset": dataset,
            "profile": profile,
            "n_seeds": len(rows),
            "seeds": [int(row["seed"]) for row in rows],
            "sample_count_per_seed": sample_counts.pop(),
            "bootstrap_ci_files": sum(bool(row["bootstrap_ci_present"]) for row in rows),
        }
        for metric in METRICS:
            mean, std = _mean_std(float(row[metric]) for row in rows)
            item[f"{metric}_mean"] = mean
            item[f"{metric}_std"] = std
        summary.append(item)
    return summary


def paired_differences(
    runs: list[dict[str, Any]],
    *,
    reference: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    indexed = {
        (str(row["dataset"]), str(row["profile"]), int(row["seed"])): row
        for row in runs
    }
    details: list[dict[str, Any]] = []
    comparisons = sorted({str(row["profile"]) for row in runs} - {reference})
    datasets = sorted({str(row["dataset"]) for row in runs})
    seeds = sorted({int(row["seed"]) for row in runs})
    for dataset in datasets:
        for comparison in comparisons:
            for seed in seeds:
                baseline = indexed.get((dataset, reference, seed))
                candidate = indexed.get((dataset, comparison, seed))
                if baseline is None or candidate is None:
                    continue
                row: dict[str, Any] = {
                    "dataset": dataset,
                    "comparison": f"{comparison}_minus_{reference}",
                    "candidate": comparison,
                    "reference": reference,
                    "seed": seed,
                }
                for metric in METRICS:
                    row[f"{metric}_delta"] = float(candidate[metric]) - float(
                        baseline[metric]
                    )
                details.append(row)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in details:
        grouped[(str(row["dataset"]), str(row["comparison"]))].append(row)
    summary: list[dict[str, Any]] = []
    for (dataset, comparison), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row["seed"]))
        item: dict[str, Any] = {
            "dataset": dataset,
            "comparison": comparison,
            "candidate": rows[0]["candidate"],
            "reference": rows[0]["reference"],
            "n_paired_seeds": len(rows),
            "seeds": [int(row["seed"]) for row in rows],
        }
        for metric in METRICS:
            mean, std = _mean_std(float(row[f"{metric}_delta"]) for row in rows)
            item[f"{metric}_delta_mean"] = mean
            item[f"{metric}_delta_std"] = std
        summary.append(item)
    return details, summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for source in rows:
            row = dict(source)
            for key, value in row.items():
                if isinstance(value, list):
                    row[key] = " ".join(str(item) for item in value)
            writer.writerow(row)


def _percent_mean_std(mean: float, std: float, *, signed: bool = False) -> str:
    prefix = "+" if signed and mean > 0 else ""
    return f"{prefix}{100.0 * mean:.2f} ± {100.0 * std:.2f}"


def render_markdown(
    summary: list[dict[str, Any]],
    paired_summary: list[dict[str, Any]],
    missing: list[str],
) -> str:
    lines = [
        "# SBI ablation summary",
        "",
        "Values are mean ± sample standard deviation across training seeds.",
        "All metric values are rendered as percentages in this report.",
        "",
    ]
    for dataset in ("celebdf", "ffpp"):
        rows = [row for row in summary if row["dataset"] == dataset]
        if not rows:
            continue
        lines.extend(
            [
                f"## {DATASET_LABELS[dataset]}",
                "",
                "| Profile | Seeds | Videos/seed | AUC (%) | Accuracy (%) | Macro-F1 (%) | EER (%) |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        profile_order = {profile: index for index, profile in enumerate(DEFAULT_PROFILES)}
        for row in sorted(rows, key=lambda item: profile_order.get(str(item["profile"]), 99)):
            lines.append(
                "| {label} | {seeds} | {samples} | {auc} | {accuracy} | {f1} | {eer} |".format(
                    label=PROFILE_LABELS.get(str(row["profile"]), str(row["profile"])),
                    seeds=row["n_seeds"],
                    samples=row["sample_count_per_seed"],
                    auc=_percent_mean_std(row["auc_mean"], row["auc_std"]),
                    accuracy=_percent_mean_std(
                        row["accuracy_mean"], row["accuracy_std"]
                    ),
                    f1=_percent_mean_std(row["f1_macro_mean"], row["f1_macro_std"]),
                    eer=_percent_mean_std(row["eer_mean"], row["eer_std"]),
                )
            )
        lines.extend(["", "### Paired change relative to clip-coherent SBI", ""])
        dataset_deltas = [row for row in paired_summary if row["dataset"] == dataset]
        if dataset_deltas:
            lines.extend(
                [
                    "| Comparison | Paired seeds | ΔAUC (pp) | ΔAccuracy (pp) | ΔMacro-F1 (pp) | ΔEER (pp) |",
                    "| --- | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for row in dataset_deltas:
                candidate = PROFILE_LABELS.get(str(row["candidate"]), str(row["candidate"]))
                reference = PROFILE_LABELS.get(str(row["reference"]), str(row["reference"]))
                lines.append(
                    "| {candidate} − {reference} | {seeds} | {auc} | {accuracy} | {f1} | {eer} |".format(
                        candidate=candidate,
                        reference=reference,
                        seeds=row["n_paired_seeds"],
                        auc=_percent_mean_std(
                            row["auc_delta_mean"], row["auc_delta_std"], signed=True
                        ),
                        accuracy=_percent_mean_std(
                            row["accuracy_delta_mean"],
                            row["accuracy_delta_std"],
                            signed=True,
                        ),
                        f1=_percent_mean_std(
                            row["f1_macro_delta_mean"],
                            row["f1_macro_delta_std"],
                            signed=True,
                        ),
                        eer=_percent_mean_std(
                            row["eer_delta_mean"], row["eer_delta_std"], signed=True
                        ),
                    )
                )
        else:
            lines.append("No complete paired seeds are available.")
        lines.append("")

    lines.extend(
        [
            "## Interpretation notes",
            "",
            "- Positive ΔAUC, ΔAccuracy, and ΔMacro-F1 favor the candidate profile.",
            "- Negative ΔEER favors the candidate profile.",
            "- Bootstrap confidence intervals remain in each run directory and quantify test-video sampling uncertainty; they are not a replacement for variation across training seeds.",
        ]
    )
    if missing:
        lines.extend(["", "## Missing runs", ""])
        lines.extend(f"- `{path}`" for path in missing)
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-root", required=True)
    parser.add_argument(
        "--celebdf-root",
        help="Celeb-DF evaluation root (default: --ablation-root)",
    )
    parser.add_argument(
        "--ffpp-root",
        help="FF++ evaluation root (default: <ablation-root>/ffpp_test)",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory (default: <ablation-root>/sbi_summary)",
    )
    parser.add_argument("--profiles", nargs="+", default=list(DEFAULT_PROFILES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--texture-frames", type=int, default=8)
    parser.add_argument("--clips-per-video", type=int, default=3)
    parser.add_argument("--aggregation", choices=("mean", "topk"), default="mean")
    parser.add_argument("--threshold-selection", default="eer")
    parser.add_argument("--no-flip-tta", action="store_true")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Write a clearly marked draft summary even if runs are missing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    unknown_profiles = sorted(set(args.profiles) - set(PROFILE_LABELS))
    if unknown_profiles:
        raise ValueError(f"Unknown SBI profiles: {unknown_profiles}")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds contains duplicates")

    ablation_root = Path(args.ablation_root)
    celebdf_root = Path(args.celebdf_root) if args.celebdf_root else ablation_root
    ffpp_root = Path(args.ffpp_root) if args.ffpp_root else ablation_root / "ffpp_test"
    output_dir = Path(args.output_dir) if args.output_dir else ablation_root / "sbi_summary"

    runs, missing = collect_runs(
        celebdf_root=celebdf_root,
        ffpp_root=ffpp_root,
        profiles=list(args.profiles),
        seeds=list(args.seeds),
        texture_frames=args.texture_frames,
        clips_per_video=args.clips_per_video,
        aggregation=args.aggregation,
        threshold_selection=args.threshold_selection,
        flip_tta=not args.no_flip_tta,
        allow_missing=args.allow_missing,
    )
    if not runs:
        raise RuntimeError("No completed SBI ablation runs were found")

    summary = summarize_runs(runs)
    paired_details, paired_summary = paired_differences(runs, reference="baseline")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "runs.csv", runs)
    _write_csv(output_dir / "summary.csv", summary)
    _write_csv(output_dir / "paired_deltas.csv", paired_details)
    _write_csv(output_dir / "paired_delta_summary.csv", paired_summary)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "profiles": list(args.profiles),
            "seeds": list(args.seeds),
            "texture_frames": args.texture_frames,
            "clips_per_video": args.clips_per_video,
            "aggregation": args.aggregation,
            "threshold_selection": args.threshold_selection,
            "flip_tta": not args.no_flip_tta,
            "strict_complete": not args.allow_missing,
        },
        "summary": summary,
        "paired_delta_summary": paired_summary,
        "runs": runs,
        "paired_deltas": paired_details,
        "missing": missing,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(
        render_markdown(summary, paired_summary, missing), encoding="utf-8"
    )

    print(f"SBI summary runs: {len(runs)}")
    print(f"SBI summary output: {output_dir}")
    for filename in (
        "summary.md",
        "summary.json",
        "summary.csv",
        "runs.csv",
        "paired_delta_summary.csv",
        "paired_deltas.csv",
    ):
        print(f"  - {output_dir / filename}")


if __name__ == "__main__":
    main()

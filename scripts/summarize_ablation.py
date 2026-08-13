#!/usr/bin/env python3
"""Summarize ablation study results into Markdown, LaTeX, and bar-chart.

Reads ``training_summary.json`` from each ablation output directory and
generates:

1. A Markdown comparison table (printed to stdout and saved to file).
2. A LaTeX table fragment ready for copy-paste into a paper.
3. An optional bar-chart PNG comparing AUC across configurations.

Usage:
    python scripts/summarize_ablation.py \\
        --ablation-dir E:/DeepFakeData/experiments/ablation \\
        --output       E:/DeepFakeData/experiments/ablation/ablation_results.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Same order as ablation_study.py
ABLATION_NAMES = [
    "baseline", "freq_only", "ms_only", "ta_only",
    "freq_ms", "freq_ta", "ms_ta", "full_v2",
]

DISPLAY_LABELS = {
    "baseline":  "Baseline (V1)",
    "freq_only": "+Freq",
    "ms_only":   "+MS",
    "ta_only":   "+TA",
    "freq_ms":   "+Freq+MS",
    "freq_ta":   "+Freq+TA",
    "ms_ta":     "+MS+TA",
    "full_v2":   "Full V2",
}

MODULE_FLAGS = {
    "baseline":  ("",  "",  ""),
    "freq_only": ("✓", "",  ""),
    "ms_only":   ("",  "✓", ""),
    "ta_only":   ("",  "",  "✓"),
    "freq_ms":   ("✓", "✓", ""),
    "freq_ta":   ("✓", "",  "✓"),
    "ms_ta":     ("",  "✓", "✓"),
    "full_v2":   ("✓", "✓", "✓"),
}


def _load_results(ablation_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name in ABLATION_NAMES:
        summary_path = ablation_dir / f"ablation_{name}" / "training_summary.json"
        if not summary_path.is_file():
            continue
        try:
            with summary_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (json.JSONDecodeError, OSError):
            continue
        if str(data.get("status", "")).lower() != "complete":
            continue

        metrics = data.get("best_validation_metrics", {})

        # Try to read param count from config.json or run_metadata.json
        config_path = ablation_dir / f"ablation_{name}" / "config.json"
        params_total = "—"
        try:
            if config_path.is_file():
                with config_path.open("r", encoding="utf-8") as handle:
                    config = json.load(handle)
                # params are logged in run_metadata.json but not always;
                # leave as dash if unavailable
        except (json.JSONDecodeError, OSError):
            pass

        rows.append({
            "name": name,
            "label": DISPLAY_LABELS.get(name, name),
            "freq": MODULE_FLAGS[name][0],
            "ms": MODULE_FLAGS[name][1],
            "ta": MODULE_FLAGS[name][2],
            "auc": float(metrics.get("auc", float("nan"))),
            "ap": float(metrics.get("average_precision", float("nan"))),
            "eer": float(metrics.get("eer", float("nan"))),
            "bal_acc": float(metrics.get("balanced_accuracy", float("nan"))),
            "best_epoch": int(data.get("best_epoch", 0)),
            "total_epochs": int(data.get("completed_epochs", 0)),
            "duration_min": float(data.get("duration_seconds", 0)) / 60.0,
        })
    return rows


def _format_markdown(rows: list[dict[str, object]]) -> str:
    baseline_auc = next(
        (float(r["auc"]) for r in rows if r["name"] == "baseline"), float("nan")
    )

    lines = [
        "# QALF v2 Ablation Study Results",
        "",
        "| Config | Freq | MS | TA | AUC ↑ | AP ↑ | EER ↓ | BalAcc | ΔAUC | Best Epoch |",
        "|--------|:----:|:--:|:--:|------:|-----:|------:|-------:|-----:|-----------:|",
    ]
    for row in rows:
        auc = float(row["auc"])
        delta = auc - baseline_auc if baseline_auc == baseline_auc else float("nan")
        delta_str = f"{delta:+.4f}" if delta == delta else "—"
        lines.append(
            f"| {row['label']:<16} | {row['freq']:^4} | {row['ms']:^2} | {row['ta']:^2} "
            f"| {auc:.4f} | {float(row['ap']):.4f} | {float(row['eer']):.4f} "
            f"| {float(row['bal_acc']):.4f} | {delta_str:>7} "
            f"| {row['best_epoch']}/{row['total_epochs']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _format_latex(rows: list[dict[str, object]]) -> str:
    baseline_auc = next(
        (float(r["auc"]) for r in rows if r["name"] == "baseline"), float("nan")
    )

    lines = [
        "% LaTeX table for paper — copy into your tabular environment",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation study on FF++ validation set (video-level metrics).}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{l ccc rrrr r}",
        r"\toprule",
        r"Config & Freq & MS & TA & AUC$\uparrow$ & AP$\uparrow$ & EER$\downarrow$ & BalAcc & $\Delta$AUC \\",
        r"\midrule",
    ]
    for row in rows:
        auc = float(row["auc"])
        delta = auc - baseline_auc if baseline_auc == baseline_auc else float("nan")
        delta_str = f"{delta:+.4f}" if delta == delta else "—"
        freq = r"\checkmark" if row["freq"] else ""
        ms = r"\checkmark" if row["ms"] else ""
        ta = r"\checkmark" if row["ta"] else ""
        label = str(row["label"]).replace("+", r"+")

        # Bold the best row (full_v2)
        if row["name"] == "full_v2":
            label = r"\textbf{" + label + "}"

        lines.append(
            f"{label} & {freq} & {ms} & {ta} "
            f"& {auc:.4f} & {float(row['ap']):.4f} & {float(row['eer']):.4f} "
            f"& {float(row['bal_acc']):.4f} & {delta_str} \\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def _save_bar_chart(rows: list[dict[str, object]], output_path: Path) -> bool:
    """Generate a bar chart comparing AUC. Returns True on success."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping bar chart", file=sys.stderr)
        return False

    labels = [str(r["label"]) for r in rows]
    aucs = [float(r["auc"]) for r in rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#6C757D"] + ["#0D6EFD"] * (len(rows) - 2) + ["#198754"]
    if len(rows) <= 1:
        colors = ["#0D6EFD"] * len(rows)
    bars = ax.bar(labels, aucs, color=colors[:len(rows)], edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Video-level AUC", fontsize=12)
    ax.set_title("QALF v2 Ablation Study — FF++ Validation", fontsize=14)
    ax.set_ylim(min(aucs) - 0.02, min(max(aucs) + 0.02, 1.0))
    for bar, auc in zip(bars, aucs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{auc:.4f}",
            ha="center", va="bottom", fontsize=9,
        )
    plt.xticks(rotation=30, ha="right", fontsize=10)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize QALF v2 ablation study results."
    )
    parser.add_argument("--ablation-dir", required=True,
                        help="Root directory containing ablation_<name>/ subdirectories")
    parser.add_argument("--output", default=None,
                        help="Path for the Markdown output file (default: stdout only)")
    parser.add_argument("--latex", default=None,
                        help="Path for the LaTeX table output file")
    parser.add_argument("--chart", default=None,
                        help="Path for the bar-chart PNG file")
    args = parser.parse_args()

    ablation_dir = Path(args.ablation_dir)
    rows = _load_results(ablation_dir)

    if not rows:
        print("No completed ablation results found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(rows)}/{len(ABLATION_NAMES)} completed ablation runs.\n")

    md = _format_markdown(rows)
    print(md)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(md, encoding="utf-8")
        print(f"\nMarkdown saved to: {output_path}")

    latex = _format_latex(rows)
    print("\n" + latex)
    if args.latex:
        latex_path = Path(args.latex)
        latex_path.parent.mkdir(parents=True, exist_ok=True)
        latex_path.write_text(latex, encoding="utf-8")
        print(f"\nLaTeX saved to: {latex_path}")

    chart_path = Path(args.chart) if args.chart else ablation_dir / "ablation_auc_chart.png"
    if _save_bar_chart(rows, chart_path):
        print(f"\nBar chart saved to: {chart_path}")


if __name__ == "__main__":
    main()

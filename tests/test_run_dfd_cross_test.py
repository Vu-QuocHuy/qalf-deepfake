from __future__ import annotations

from scripts.run_dfd_cross_test import _render_markdown, _summarize


def _run(seed: int, offset: float) -> dict[str, float | int]:
    return {
        "seed": seed,
        "sample_count": 3431,
        "real_count": 363,
        "fake_count": 3068,
        "threshold": 0.489 + offset,
        "auc": 0.90 + offset,
        "average_precision": 0.98 + offset,
        "accuracy": 0.87 + offset,
        "balanced_accuracy": 0.82 + offset,
        "f1_macro": 0.74 + offset,
        "eer": 0.17 - offset,
        "apcer": 0.12 - offset,
        "bpcer": 0.22 + offset,
        "acer": 0.17,
    }


def test_summarize_dfd_runs_and_render_markdown() -> None:
    runs = [_run(0, 0.0), _run(42, 0.01)]
    summary = _summarize(runs)

    assert summary["n_seeds"] == 2
    assert summary["sample_count_per_seed"] == 3431
    assert summary["auc_mean"] == 0.905
    assert summary["auc_std"] > 0
    markdown = _render_markdown(summary, runs)
    assert "DFD cross-dataset summary" in markdown
    assert "Balanced Accuracy" in markdown
    assert "3431" in markdown

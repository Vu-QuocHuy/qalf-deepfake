from __future__ import annotations

from scripts.run_table_iva_server import _summarize


def test_summarize_requires_a_common_complete_protocol() -> None:
    runs = [
        {"seed": 0, "auc": 0.80, "sample_count": 518, "real_count": 178, "fake_count": 340},
        {"seed": 42, "auc": 0.82, "sample_count": 518, "real_count": 178, "fake_count": 340},
    ]

    summary = _summarize(runs)

    assert summary["n_seeds"] == 2
    assert summary["seeds"] == [0, 42]
    assert summary["sample_count_per_seed"] == 518
    assert summary["auc_mean"] == 0.81
    assert summary["auc_std"] > 0

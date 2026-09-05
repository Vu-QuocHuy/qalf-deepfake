from __future__ import annotations

from scripts.compare_table_iva import _compare


def test_compare_reports_seedwise_percentage_point_gap() -> None:
    rows, summary = _compare(
        [{"seed": "0", "auc": "0.80"}], [{"seed": "0", "auc": "0.7999"}]
    )

    assert rows[0]["delta_pp"] == 0.01
    assert summary["delta_pp_mean"] == 0.01

from __future__ import annotations

from scripts.measure_model_complexity import _gmacs_from_flops


def test_gmacs_are_half_of_flops() -> None:
    assert _gmacs_from_flops(4_000_000_000) == 2.0

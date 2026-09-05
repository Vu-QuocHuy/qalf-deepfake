from __future__ import annotations

import pytest

from scripts.evaluate_pi4 import _require_complete_results


def test_require_complete_results_rejects_skipped_videos() -> None:
    with pytest.raises(RuntimeError, match="518 / 519"):
        _require_complete_results(processed=518, expected=519, errors=["bad-video: unreadable"])

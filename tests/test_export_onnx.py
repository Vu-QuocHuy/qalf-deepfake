from __future__ import annotations

from pathlib import Path

from scripts.export_onnx import _metadata


def test_metadata_records_the_post_verification_state() -> None:
    checkpoint = {"threshold": 0.7, "config": {"data": {}, "model": {}}}

    metadata = _metadata(checkpoint, Path("model.onnx"), bytes_size=123, opset=17, verified=True)

    assert metadata["optimal_threshold"] == 0.7
    assert metadata["verified"] is True

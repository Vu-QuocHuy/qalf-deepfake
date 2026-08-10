"""JSON configuration helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qalf.data.geometry import DEFAULT_GEOMETRY_FEATURE_MODE, GEOMETRY_FEATURE_MODES


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    for section in ("data", "model", "training"):
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"Missing configuration section: {section}")
    num_frames = int(config["data"].get("num_frames", 0))
    texture_frames = int(config["data"].get("texture_frames", 0))
    if num_frames < 2:
        raise ValueError("data.num_frames must be >= 2")
    if not 1 <= texture_frames <= num_frames:
        raise ValueError("data.texture_frames must be in [1, data.num_frames]")
    if int(config["data"].get("eval_clips_per_video", 1)) < 1:
        raise ValueError("data.eval_clips_per_video must be >= 1")
    geometry_mode = config["data"].setdefault("geometry_mode", DEFAULT_GEOMETRY_FEATURE_MODE)
    if geometry_mode not in GEOMETRY_FEATURE_MODES:
        raise ValueError(f"Unsupported data.geometry_mode: {geometry_mode}")
    if config["data"].get("texture_mode", "canonical_skin") not in {
        "canonical_skin",
        "full_face",
    }:
        raise ValueError("Unsupported data.texture_mode")
    if config["data"].get("video_aggregation", "mean") not in {"mean", "median", "topk"}:
        raise ValueError("Unsupported data.video_aggregation")
    if int(config["data"].get("top_k", 1)) < 1:
        raise ValueError("data.top_k must be >= 1")
    return config


def save_json(payload: Any, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    temporary.replace(output)

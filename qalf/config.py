"""JSON configuration helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qalf.data.geometry import DEFAULT_GEOMETRY_FEATURE_MODE, GEOMETRY_FEATURE_MODES
from qalf.data.sbi import resolve_sbi_config


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
        "dual_view",
    }:
        raise ValueError("Unsupported data.texture_mode")
    if config["data"].get("video_aggregation", "mean") not in {"mean", "median", "topk"}:
        raise ValueError("Unsupported data.video_aggregation")
    if int(config["data"].get("top_k", 1)) < 1:
        raise ValueError("data.top_k must be >= 1")
    config["data"]["sbi"] = resolve_sbi_config(config["data"].get("sbi"))
    if bool(config["data"]["sbi"]["enabled"]) and config["data"].get(
        "texture_mode", "canonical_skin"
    ) != "full_face":
        raise ValueError("Enabled SBI requires data.texture_mode=full_face")
    fake_methods = config["data"].get("fake_methods")
    if fake_methods is not None:
        if not isinstance(fake_methods, list) or not fake_methods:
            raise ValueError("data.fake_methods must be a non-empty list")
        if not all(isinstance(method, str) and method for method in fake_methods):
            raise ValueError("data.fake_methods must contain non-empty strings")
        if len(set(fake_methods)) != len(fake_methods):
            raise ValueError("data.fake_methods must not contain duplicates")
    model_defaults = {
        "geometry_hidden": 96,
        "geometry_layers": 3,
        "embedding_dim": 128,
        "dropout": 0.2,
        "texture_backbone": "efficientnet_b0",
        "texture_temporal_pooling": "mean",
        "texture_mixstyle_probability": 0.0,
        "texture_mixstyle_alpha": 0.1,
        "texture_mixstyle_layers": [],
    }
    for name, default in model_defaults.items():
        config["model"].setdefault(name, default)
    if config["model"]["texture_backbone"] not in {"efficientnet_b0", "efficientnet_b1"}:
        raise ValueError("model.texture_backbone must be efficientnet_b0 or efficientnet_b1")
    if config["model"]["texture_temporal_pooling"] not in {
        "mean",
        "dynamics",
    }:
        raise ValueError("model.texture_temporal_pooling must be mean or dynamics")
    mixstyle_probability = float(config["model"]["texture_mixstyle_probability"])
    if not 0.0 <= mixstyle_probability <= 1.0:
        raise ValueError("model.texture_mixstyle_probability must be in [0, 1]")
    if float(config["model"]["texture_mixstyle_alpha"]) <= 0.0:
        raise ValueError("model.texture_mixstyle_alpha must be positive")
    mixstyle_layers = config["model"]["texture_mixstyle_layers"]
    if not isinstance(mixstyle_layers, list) or not all(
        isinstance(index, int) and index >= 0 for index in mixstyle_layers
    ):
        raise ValueError("model.texture_mixstyle_layers must contain non-negative integers")
    if len(set(mixstyle_layers)) != len(mixstyle_layers):
        raise ValueError("model.texture_mixstyle_layers must not contain duplicates")
    if mixstyle_probability > 0.0 and not mixstyle_layers:
        raise ValueError(
            "model.texture_mixstyle_layers cannot be empty when MixStyle is enabled"
        )
    for name in ("geometry_hidden", "geometry_layers", "embedding_dim"):
        if int(config["model"].get(name, 0)) < 1:
            raise ValueError(f"model.{name} must be >= 1")
    if not 0.0 <= float(config["model"].get("dropout", 0.0)) < 1.0:
        raise ValueError("model.dropout must be in [0, 1)")
    return config


def save_json(payload: Any, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    temporary.replace(output)

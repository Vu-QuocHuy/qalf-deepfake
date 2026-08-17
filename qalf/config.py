"""JSON configuration helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qalf.data.sbi import resolve_sbi_config


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    for section in ("data", "model", "training"):
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"Missing configuration section: {section}")

    data = config["data"]
    num_frames = int(data.get("num_frames", 0))
    texture_frames = int(data.get("texture_frames", 0))
    if num_frames < 2:
        raise ValueError("data.num_frames must be >= 2")
    if not 1 <= texture_frames <= num_frames:
        raise ValueError("data.texture_frames must be in [1, data.num_frames]")
    if int(data.get("eval_clips_per_video", 1)) < 1:
        raise ValueError("data.eval_clips_per_video must be >= 1")
    if data.get("texture_mode", "full_face") != "full_face":
        raise ValueError("data.texture_mode must be full_face")
    temporal_sampling = str(data.setdefault("temporal_sampling", "uniform"))
    if temporal_sampling not in {"uniform", "paired"}:
        raise ValueError("data.temporal_sampling must be uniform or paired")
    if temporal_sampling == "paired" and (texture_frames < 2 or texture_frames % 2):
        raise ValueError(
            "data.texture_frames must be even and >= 2 for paired temporal sampling"
        )
    coherent_augmentation = data.setdefault("coherent_augmentation", False)
    if not isinstance(coherent_augmentation, bool):
        raise ValueError("data.coherent_augmentation must be boolean")
    if data.get("video_aggregation", "mean") not in {"mean", "median", "topk"}:
        raise ValueError("Unsupported data.video_aggregation")
    if int(data.get("top_k", 1)) < 1:
        raise ValueError("data.top_k must be >= 1")
    data["sbi"] = resolve_sbi_config(data.get("sbi"))
    fake_methods = data.get("fake_methods")
    if fake_methods is not None:
        if not isinstance(fake_methods, list) or not fake_methods:
            raise ValueError("data.fake_methods must be a non-empty list")
        if not all(isinstance(method, str) and method for method in fake_methods):
            raise ValueError("data.fake_methods must contain non-empty strings")
        if len(set(fake_methods)) != len(fake_methods):
            raise ValueError("data.fake_methods must not contain duplicates")

    model = config["model"]
    model.setdefault("embedding_dim", 128)
    model.setdefault("dropout", 0.2)
    model.setdefault("texture_backbone", "efficientnet_b0")
    model.setdefault("temporal_pooling", "mean")
    model.setdefault("temporal_bottleneck", 32)
    model.setdefault("temporal_residual_scale", 0.1)
    if model["texture_backbone"] != "efficientnet_b0":
        raise ValueError("model.texture_backbone must be efficientnet_b0")
    if int(model["embedding_dim"]) < 1:
        raise ValueError("model.embedding_dim must be >= 1")
    if model["temporal_pooling"] not in {"mean", "paired_residual"}:
        raise ValueError("model.temporal_pooling must be mean or paired_residual")
    if int(model["temporal_bottleneck"]) < 8:
        raise ValueError("model.temporal_bottleneck must be >= 8")
    if not 0.0 < float(model["temporal_residual_scale"]) <= 1.0:
        raise ValueError("model.temporal_residual_scale must be in (0, 1]")
    if model["temporal_pooling"] == "paired_residual" and temporal_sampling != "paired":
        raise ValueError(
            "model.temporal_pooling=paired_residual requires data.temporal_sampling=paired"
        )
    if not 0.0 <= float(model["dropout"]) < 1.0:
        raise ValueError("model.dropout must be in [0, 1)")

    training = config["training"]
    if int(training.get("epochs", 0)) < 1:
        raise ValueError("training.epochs must be >= 1")
    ema_decay = float(training.get("ema_decay", 0.0))
    if not 0.0 <= ema_decay < 1.0:
        raise ValueError("training.ema_decay must be in [0, 1)")
    validation_weights = str(
        training.setdefault("validation_weights", "ema" if ema_decay > 0.0 else "raw")
    )
    if validation_weights not in {"raw", "ema"}:
        raise ValueError("training.validation_weights must be raw or ema")
    if validation_weights == "ema" and ema_decay <= 0.0:
        raise ValueError("EMA validation requires training.ema_decay > 0")
    return config


def save_json(payload: Any, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    temporary.replace(output)

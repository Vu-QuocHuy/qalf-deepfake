"""QALF model components."""

from typing import Any

from .qalf import QALFModel
from .texture import SUPPORTED_TEMPORAL_POOLING, SUPPORTED_TEXTURE_BACKBONES


def build_model_from_checkpoint(
    checkpoint: dict[str, Any], *, load_weights: bool = True
) -> QALFModel:
    """Build the texture-only architecture recorded in a current checkpoint."""

    model_config = checkpoint["config"]["model"]
    model = QALFModel(
        embedding_dim=int(model_config.get("embedding_dim", 128)),
        dropout=float(model_config.get("dropout", 0.2)),
        texture_pretrained=False,
        texture_backbone=str(model_config.get("texture_backbone", "efficientnet_b0")),
        temporal_pooling=str(model_config.get("temporal_pooling", "mean")),
        temporal_bottleneck=int(model_config.get("temporal_bottleneck", 32)),
        temporal_residual_scale=float(model_config.get("temporal_residual_scale", 0.1)),
    )
    if load_weights:
        model.load_state_dict(checkpoint["model"], strict=True)
    return model


__all__ = [
    "QALFModel",
    "SUPPORTED_TEMPORAL_POOLING",
    "SUPPORTED_TEXTURE_BACKBONES",
    "build_model_from_checkpoint",
]

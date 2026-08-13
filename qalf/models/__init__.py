"""QALF model components."""

from typing import Any

from .qalf import QALFModel
from .texture import SUPPORTED_TEXTURE_BACKBONES


def build_model_from_checkpoint(
    checkpoint: dict[str, Any], *, load_weights: bool = True
) -> QALFModel:
    """Build the texture-only architecture recorded in a current checkpoint.

    Supports both v1 (mean-pooling only) and v2 (frequency / multiscale /
    temporal-attention) checkpoints.  Missing v2 keys default to ``False``
    for full backward compatibility.
    """

    model_config = checkpoint["config"]["model"]
    model = QALFModel(
        embedding_dim=int(model_config.get("embedding_dim", 128)),
        dropout=float(model_config.get("dropout", 0.2)),
        texture_pretrained=False,
        texture_backbone=str(model_config.get("texture_backbone", "efficientnet_b0")),
        frequency_preprocess=bool(model_config.get("frequency_preprocess", False)),
        multiscale=bool(model_config.get("multiscale", False)),
        temporal_attention=bool(model_config.get("temporal_attention", False)),
    )
    if load_weights:
        model.load_state_dict(checkpoint["model"], strict=True)
    return model


__all__ = [
    "QALFModel",
    "SUPPORTED_TEXTURE_BACKBONES",
    "build_model_from_checkpoint",
]

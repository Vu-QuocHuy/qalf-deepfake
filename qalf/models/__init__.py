"""TextureSBI model components."""

from typing import Any

from .texture_sbi import TextureSBIModel
from .texture import SUPPORTED_TEXTURE_BACKBONES


def build_model_from_checkpoint(
    checkpoint: dict[str, Any], *, load_weights: bool = True
) -> TextureSBIModel:
    """Build the TextureSBI architecture recorded in a checkpoint."""

    model_config = checkpoint["config"]["model"]
    model = TextureSBIModel(
        embedding_dim=int(model_config.get("embedding_dim", 128)),
        dropout=float(model_config.get("dropout", 0.2)),
        texture_pretrained=False,
        texture_backbone=str(model_config.get("texture_backbone", "efficientnet_b0")),
    )
    if load_weights:
        model.load_state_dict(checkpoint["model"], strict=True)
    return model


__all__ = [
    "TextureSBIModel",
    "SUPPORTED_TEXTURE_BACKBONES",
    "build_model_from_checkpoint",
]

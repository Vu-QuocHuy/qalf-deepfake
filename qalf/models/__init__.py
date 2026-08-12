"""QALF model components."""

from typing import Any

from .geometry import SUPPORTED_GEOMETRY_ARCHITECTURES
from .qalf import QALFModel
from .scheduler import TextureRefreshPolicy
from .texture import SUPPORTED_TEXTURE_BACKBONES


def build_model_from_checkpoint(
    checkpoint: dict[str, Any], *, load_weights: bool = True
) -> QALFModel:
    """Build the retained architecture from checkpointed hyperparameters."""

    config = checkpoint["config"]
    model_config = config["model"]
    model = QALFModel(
        geometry_input_dim=int(checkpoint["geometry_input_dim"]),
        geometry_hidden=int(model_config.get("geometry_hidden", 96)),
        geometry_layers=int(model_config.get("geometry_layers", 3)),
        geometry_architecture=str(model_config.get("geometry_architecture", "tcn_mean")),
        embedding_dim=int(model_config.get("embedding_dim", 128)),
        dropout=float(model_config.get("dropout", 0.2)),
        texture_pretrained=False,
        texture_backbone=str(model_config.get("texture_backbone", "efficientnet_b0")),
        geometry_quality_dim=int(checkpoint.get("geometry_quality_dim", 5)),
        texture_quality_dim=int(checkpoint.get("texture_quality_dim", 5)),
        fusion_mode=str(model_config.get("fusion_mode", "quality")),
        texture_gate_bias=float(model_config.get("texture_gate_bias", 0.0)),
        modality_dropout_probability=float(model_config.get("modality_dropout_probability", 0.0)),
    )
    if load_weights:
        model.load_state_dict(checkpoint["model"], strict=True)
    return model


__all__ = [
    "QALFModel",
    "build_model_from_checkpoint",
    "SUPPORTED_GEOMETRY_ARCHITECTURES",
    "SUPPORTED_TEXTURE_BACKBONES",
    "TextureRefreshPolicy",
]

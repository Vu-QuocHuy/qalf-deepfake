"""EfficientNet-B0 encoder with lightweight temporal aggregation."""

from __future__ import annotations

import torch
from torch import nn

SUPPORTED_TEXTURE_BACKBONES = {"efficientnet_b0"}
SUPPORTED_TEMPORAL_POOLINGS = {"mean", "attention"}


def _build_backbone(name: str, pretrained: bool) -> tuple[nn.Module, nn.Module, int]:
    if name != "efficientnet_b0":
        raise ValueError(f"Unsupported texture backbone: {name}")
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)
    feature_dim = int(model.classifier[-1].in_features)
    return model.features, model.avgpool, feature_dim


class TextureEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 128,
        dropout: float = 0.2,
        pretrained: bool = True,
        backbone: str = "efficientnet_b0",
        temporal_pooling: str = "mean",
    ) -> None:
        super().__init__()
        if backbone not in SUPPORTED_TEXTURE_BACKBONES:
            raise ValueError(f"Unsupported texture backbone: {backbone}")
        if temporal_pooling not in SUPPORTED_TEMPORAL_POOLINGS:
            raise ValueError(
                f"Unsupported temporal pooling: {temporal_pooling}; "
                f"choose one of {sorted(SUPPORTED_TEMPORAL_POOLINGS)}"
            )
        self.backbone_name = backbone
        self.temporal_pooling = temporal_pooling
        self.features, self.pool, feature_dim = _build_backbone(backbone, pretrained)
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.Hardswish(),
            nn.Dropout(dropout),
        )
        self.temporal_attention = (
            nn.Linear(embedding_dim, 1) if temporal_pooling == "attention" else None
        )
        self.classifier = nn.Linear(embedding_dim, 1)

    def forward(self, texture: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, frames, channels, height, width = texture.shape
        if channels != 3:
            raise ValueError(f"Texture encoder expects 3 channels, received {channels}")
        output = texture.reshape(batch * frames, channels, height, width)
        for layer in self.features:
            output = layer(output)
        output = self.pool(output).flatten(1)
        output = self.projection(output).reshape(batch, frames, -1)
        if self.temporal_pooling == "mean":
            embedding = output.mean(dim=1)
        else:
            if self.temporal_attention is None:
                raise RuntimeError("Attention pooling layer is not initialized")
            attention_logits = self.temporal_attention(output).squeeze(-1)
            attention = torch.softmax(attention_logits, dim=1).unsqueeze(-1)
            embedding = (output * attention).sum(dim=1)
        return embedding, self.classifier(embedding).squeeze(-1)

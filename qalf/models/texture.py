"""Lightweight RGB encoders for landmark-aligned full-face sequences."""

from __future__ import annotations

import torch
from torch import nn

SUPPORTED_TEXTURE_BACKBONES = {"efficientnet_b0", "mobilenet_v3_large"}


def _build_backbone(name: str, pretrained: bool) -> tuple[nn.Module, nn.Module, int]:
    if name == "efficientnet_b0":
        from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
        feature_dim = int(model.classifier[-1].in_features)
        return model.features, model.avgpool, feature_dim
    if name == "mobilenet_v3_large":
        from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_large(weights=weights)
        # MobileNetV3 has a classifier projection before its final logits;
        # the pooled feature width is the first classifier input (960), not
        # the final classifier input (1280).
        feature_dim = int(model.classifier[0].in_features)
        return model.features, model.avgpool, feature_dim
    else:
        raise ValueError(f"Unsupported texture backbone: {name}")


class TextureEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 128,
        dropout: float = 0.2,
        pretrained: bool = True,
        backbone: str = "efficientnet_b0",
    ) -> None:
        super().__init__()
        if backbone not in SUPPORTED_TEXTURE_BACKBONES:
            raise ValueError(f"Unsupported texture backbone: {backbone}")
        self.backbone_name = backbone
        self.features, self.pool, feature_dim = _build_backbone(backbone, pretrained)
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.Hardswish(),
            nn.Dropout(dropout),
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
        embedding = output.mean(dim=1)
        return embedding, self.classifier(embedding).squeeze(-1)

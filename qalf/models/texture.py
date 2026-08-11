"""Pretrained EfficientNet encoders for landmark-aligned skin maps."""

from __future__ import annotations

import torch
from torch import nn

SUPPORTED_TEXTURE_BACKBONES = {"efficientnet_b0", "efficientnet_b1"}
SUPPORTED_TEXTURE_POOLING = {"mean", "attention"}


def _build_backbone(name: str, pretrained: bool) -> tuple[nn.Module, nn.Module, int]:
    if name == "efficientnet_b0":
        from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
        feature_dim = int(model.classifier[-1].in_features)
    elif name == "efficientnet_b1":
        from torchvision.models import EfficientNet_B1_Weights, efficientnet_b1

        weights = EfficientNet_B1_Weights.DEFAULT if pretrained else None
        model = efficientnet_b1(weights=weights)
        feature_dim = int(model.classifier[-1].in_features)
    else:
        raise ValueError(f"Unsupported texture backbone: {name}")
    return model.features, model.avgpool, feature_dim


class TemporalAttentionPool(nn.Module):
    """Permutation-invariant attention pooling for frame-level texture embeddings."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        hidden_dim = max(32, embedding_dim // 2)
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        # Start as exact mean pooling. Training can then learn which frames matter
        # without introducing a random pooling bias at initialization.
        nn.init.zeros_(self.attention[-1].weight)
        nn.init.zeros_(self.attention[-1].bias)

    def forward(self, frame_embeddings: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.attention(frame_embeddings), dim=1)
        return torch.sum(weights * frame_embeddings, dim=1)


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
        if temporal_pooling not in SUPPORTED_TEXTURE_POOLING:
            raise ValueError(f"Unsupported texture temporal pooling: {temporal_pooling}")
        self.backbone_name = backbone
        self.temporal_pooling = temporal_pooling
        self.features, self.pool, feature_dim = _build_backbone(backbone, pretrained)
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.Hardswish(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(embedding_dim, 1)
        self.temporal_attention = (
            TemporalAttentionPool(embedding_dim) if temporal_pooling == "attention" else None
        )

    def forward(self, texture: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, frames, channels, height, width = texture.shape
        output = texture.reshape(batch * frames, channels, height, width)
        output = self.pool(self.features(output)).flatten(1)
        output = self.projection(output).reshape(batch, frames, -1)
        embedding = (
            output.mean(dim=1)
            if self.temporal_attention is None
            else self.temporal_attention(output)
        )
        return embedding, self.classifier(embedding).squeeze(-1)

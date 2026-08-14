"""EfficientNet-B0 encoder for landmark-aligned full-face sequences."""

from __future__ import annotations

import torch
from torch import nn

SUPPORTED_TEXTURE_BACKBONES = {"efficientnet_b0"}
SUPPORTED_TEMPORAL_POOLING = {"mean", "residual_tcn"}


class _DepthwiseTemporalBlock(nn.Module):
    """Small order-aware block operating on (batch, channels, frames)."""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.norm = nn.GroupNorm(1, channels)
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.activation = nn.GELU()

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        update = self.depthwise(sequence)
        update = self.activation(self.norm(update))
        return sequence + self.pointwise(update)


class TemporalResidualTCN(nn.Module):
    """Lightweight temporal residual over per-frame texture embeddings.

    The zero-initialized output projection makes the initial model exactly
    equivalent to mean pooling. Temporal differences and depthwise convolutions
    then learn local order-aware corrections without adding a second modality.
    """

    def __init__(self, embedding_dim: int, bottleneck_dim: int = 48) -> None:
        super().__init__()
        if bottleneck_dim < 8:
            raise ValueError("Temporal TCN bottleneck_dim must be at least 8")
        self.input_projection = nn.Sequential(
            nn.Linear(embedding_dim * 2, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            _DepthwiseTemporalBlock(bottleneck_dim, dilation=1),
            _DepthwiseTemporalBlock(bottleneck_dim, dilation=2),
        )
        self.output_projection = nn.Linear(bottleneck_dim, embedding_dim)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, frame_embeddings: torch.Tensor) -> torch.Tensor:
        if frame_embeddings.ndim != 3:
            raise ValueError("Temporal TCN expects (batch, frames, embedding_dim)")
        if frame_embeddings.shape[1] < 2:
            return frame_embeddings.mean(dim=1)
        differences = torch.zeros_like(frame_embeddings)
        differences[:, 1:] = frame_embeddings[:, 1:] - frame_embeddings[:, :-1]
        sequence = self.input_projection(
            torch.cat((frame_embeddings, differences), dim=-1)
        )
        sequence = self.blocks(sequence.transpose(1, 2)).transpose(1, 2)
        residual = self.output_projection(sequence.mean(dim=1))
        return frame_embeddings.mean(dim=1) + residual


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
        temporal_bottleneck: int = 48,
    ) -> None:
        super().__init__()
        if backbone not in SUPPORTED_TEXTURE_BACKBONES:
            raise ValueError(f"Unsupported texture backbone: {backbone}")
        if temporal_pooling not in SUPPORTED_TEMPORAL_POOLING:
            raise ValueError(f"Unsupported temporal pooling: {temporal_pooling}")
        self.backbone_name = backbone
        self.temporal_pooling = temporal_pooling
        self.features, self.pool, feature_dim = _build_backbone(backbone, pretrained)
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.Hardswish(),
            nn.Dropout(dropout),
        )
        self.temporal_encoder = (
            TemporalResidualTCN(embedding_dim, int(temporal_bottleneck))
            if temporal_pooling == "residual_tcn"
            else None
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
        embedding = (
            output.mean(dim=1)
            if self.temporal_encoder is None
            else self.temporal_encoder(output)
        )
        return embedding, self.classifier(embedding).squeeze(-1)

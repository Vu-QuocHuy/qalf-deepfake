"""EfficientNet-B0 encoder for landmark-aligned full-face sequences.

Supports three optional enhancements over the v1 baseline (mean-pooling only):
  1. **FrequencyPreprocess** — SRM high-pass filter input layer.
  2. **MultiScaleAggregation** — taps low/mid/high backbone stages.
  3. **TemporalAttentionPooling** — learned frame importance + variance.

All three default to ``False`` so the v1 architecture is recovered exactly
when no flags are set (backward-compatible checkpoint loading).
"""

from __future__ import annotations

import torch
from torch import nn

from .frequency import FrequencyPreprocess
from .multiscale import MultiScaleAggregation
from .temporal import TemporalAttentionPooling

SUPPORTED_TEXTURE_BACKBONES = {"efficientnet_b0"}

# EfficientNet-B0 ``features`` stage boundaries and output channels:
#   Stage low  = features[0:3]  → 24 ch
#   Stage mid  = features[3:6]  → 112 ch
#   Stage high = features[6:9]  → 1280 ch
_LOW_END = 3
_MID_END = 6
_LOW_CH = 24
_MID_CH = 112
_HIGH_CH = 1280


def _build_backbone(name: str, pretrained: bool) -> tuple[nn.Module, nn.Module, int]:
    if name != "efficientnet_b0":
        raise ValueError(f"Unsupported texture backbone: {name}")
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)
    feature_dim = int(model.classifier[-1].in_features)
    return model.features, model.avgpool, feature_dim


class TextureEncoder(nn.Module):
    """Encode a clip of aligned RGB face frames into a single embedding.

    Parameters
    ----------
    embedding_dim:
        Output embedding width (and classifier input width).
    dropout:
        Dropout probability in the projection head.
    pretrained:
        Load ImageNet-pretrained backbone weights.
    backbone:
        Backbone architecture name (currently only ``efficientnet_b0``).
    frequency_preprocess:
        Prepend fixed SRM high-pass filters before the backbone.
    multiscale:
        Aggregate features from three backbone stages instead of the final
        pooled vector only.
    temporal_attention:
        Use learned attention pooling (+ variance) instead of mean pooling.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        dropout: float = 0.2,
        pretrained: bool = True,
        backbone: str = "efficientnet_b0",
        frequency_preprocess: bool = False,
        multiscale: bool = False,
        temporal_attention: bool = False,
    ) -> None:
        super().__init__()
        if backbone not in SUPPORTED_TEXTURE_BACKBONES:
            raise ValueError(f"Unsupported texture backbone: {backbone}")
        self.backbone_name = backbone
        self.use_frequency = frequency_preprocess
        self.use_multiscale = multiscale
        self.use_temporal_attention = temporal_attention

        self.features, self.pool, feature_dim = _build_backbone(backbone, pretrained)

        # Optional frequency preprocessing
        self.frequency: FrequencyPreprocess | None = None
        if self.use_frequency:
            self.frequency = FrequencyPreprocess()

        # Multi-scale or single-scale projection
        self.multiscale_agg: MultiScaleAggregation | None = None
        if self.use_multiscale:
            self.multiscale_agg = MultiScaleAggregation(
                low_channels=_LOW_CH,
                mid_channels=_MID_CH,
                high_channels=_HIGH_CH,
                embedding_dim=embedding_dim,
            )
            # When multiscale is active we do NOT use the single-vector projection.
            self.projection = None
        else:
            self.projection = nn.Sequential(
                nn.Linear(feature_dim, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.Hardswish(),
                nn.Dropout(dropout),
            )

        # Temporal pooling
        self.temporal_pool: TemporalAttentionPooling | None = None
        if self.use_temporal_attention:
            self.temporal_pool = TemporalAttentionPooling(
                embedding_dim=embedding_dim,
                hidden_dim=64,
                dropout=dropout,
            )

        self.classifier = nn.Linear(embedding_dim, 1)

    # ------------------------------------------------------------------
    # Backbone forward helpers
    # ------------------------------------------------------------------

    def _forward_single_scale(self, x: torch.Tensor) -> torch.Tensor:
        """Standard single-pass through the full backbone → pooled vector."""
        for layer in self.features:
            x = layer(x)
        return self.pool(x).flatten(1)  # (N, feature_dim)

    def _forward_multi_scale(self, x: torch.Tensor) -> torch.Tensor:
        """Run backbone in stages and aggregate low/mid/high features."""
        # Low stage: features[0:3]
        for layer in list(self.features)[:_LOW_END]:
            x = layer(x)
        low = x

        # Mid stage: features[3:6]
        for layer in list(self.features)[_LOW_END:_MID_END]:
            x = layer(x)
        mid = x

        # High stage: features[6:9]
        for layer in list(self.features)[_MID_END:]:
            x = layer(x)
        high = x

        assert self.multiscale_agg is not None
        return self.multiscale_agg(low, mid, high)  # (N, embedding_dim)

    # ------------------------------------------------------------------
    # Main forward
    # ------------------------------------------------------------------

    def forward(self, texture: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, frames, channels, height, width = texture.shape
        if channels != 3:
            raise ValueError(f"Texture encoder expects 3 channels, received {channels}")

        x = texture.reshape(batch * frames, channels, height, width)

        # Optional frequency preprocessing
        if self.frequency is not None:
            x = self.frequency(x)

        # Feature extraction
        if self.use_multiscale:
            frame_emb = self._forward_multi_scale(x)  # (B*T, embedding_dim)
        else:
            pooled = self._forward_single_scale(x)  # (B*T, feature_dim)
            assert self.projection is not None
            frame_emb = self.projection(pooled)  # (B*T, embedding_dim)

        frame_emb = frame_emb.reshape(batch, frames, -1)  # (B, T, D)

        # Temporal pooling
        if self.temporal_pool is not None:
            embedding = self.temporal_pool(frame_emb)  # (B, D)
        else:
            embedding = frame_emb.mean(dim=1)  # (B, D)

        return embedding, self.classifier(embedding).squeeze(-1)

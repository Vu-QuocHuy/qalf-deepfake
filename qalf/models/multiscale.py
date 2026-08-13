"""Lightweight multi-scale feature aggregation for EfficientNet-B0.

Instead of using only the final 1280-d pooled feature, this module taps into
three intermediate stages of EfficientNet-B0 and projects each to a compact
embedding.  The concatenated multi-scale descriptor captures both fine-grained
artefacts (early layers) and high-level semantic patterns (late layers).
"""

from __future__ import annotations

import torch
from torch import nn

# EfficientNet-B0 ``features`` is a Sequential of 9 sub-blocks (indices 0–8).
# Channel counts at the output of each sub-block:
#   0: 32,  1: 16,  2: 24,  3: 40,  4: 80,  5: 112,  6: 192,  7: 320,  8: 1280
# We group them into three stages:
#   low  = features[0:3]  → output channels = 24
#   mid  = features[3:6]  → output channels = 112
#   high = features[6:9]  → output channels = 1280


class _ScaleProjection(nn.Module):
    """AdaptiveAvgPool → Linear → LayerNorm → SiLU for one feature scale."""

    def __init__(self, in_channels: int, out_features: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Sequential(
            nn.Linear(in_channels, out_features),
            nn.LayerNorm(out_features),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.pool(x).flatten(1))


class MultiScaleAggregation(nn.Module):
    """Aggregate low / mid / high EfficientNet-B0 features into one embedding.

    Parameters
    ----------
    low_channels:
        Channel count from the low-level stage (default 24 for EffB0 ``features[:3]``).
    mid_channels:
        Channel count from the mid-level stage (default 112 for EffB0 ``features[3:6]``).
    high_channels:
        Channel count from the high-level stage (default 1280 for EffB0 ``features[6:]``).
    embedding_dim:
        Final output embedding dimensionality.
    """

    # Projection widths per scale — keep small for lightweight budget.
    LOW_DIM = 64
    MID_DIM = 64
    HIGH_DIM = 128

    def __init__(
        self,
        low_channels: int = 24,
        mid_channels: int = 112,
        high_channels: int = 1280,
        embedding_dim: int = 192,
    ) -> None:
        super().__init__()
        self.low = _ScaleProjection(low_channels, self.LOW_DIM)
        self.mid = _ScaleProjection(mid_channels, self.MID_DIM)
        self.high = _ScaleProjection(high_channels, self.HIGH_DIM)
        concat_dim = self.LOW_DIM + self.MID_DIM + self.HIGH_DIM  # 256
        self.merge = nn.Sequential(
            nn.Linear(concat_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.Hardswish(),
        )

    def forward(
        self,
        low: torch.Tensor,
        mid: torch.Tensor,
        high: torch.Tensor,
    ) -> torch.Tensor:
        """Merge three feature maps into one embedding vector per sample.

        Args:
            low:  (B, low_channels, H1, W1)
            mid:  (B, mid_channels, H2, W2)
            high: (B, high_channels, H3, W3)

        Returns:
            (B, embedding_dim)
        """
        return self.merge(
            torch.cat([self.low(low), self.mid(mid), self.high(high)], dim=1)
        )

"""Temporal encoders for pose-decoupled landmark features."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

SUPPORTED_GEOMETRY_ARCHITECTURES = {"tcn_mean"}


class CausalDepthwiseBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        kernel_size = 3
        self.left_padding = dilation * (kernel_size - 1)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=channels,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        output = F.pad(inputs, (self.left_padding, 0))
        output = self.depthwise(output)
        output = self.pointwise(output)
        output = self.norm(output)
        output = F.silu(output)
        output = self.dropout(output)
        return residual + output


class GeometryEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 96,
        embedding_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        architecture: str = "tcn_mean",
    ) -> None:
        super().__init__()
        if architecture not in SUPPORTED_GEOMETRY_ARCHITECTURES:
            raise ValueError(f"Unsupported geometry architecture: {architecture}")
        self.architecture = architecture
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        self.temporal = nn.Sequential(
            *(CausalDepthwiseBlock(hidden_dim, 2**layer, dropout) for layer in range(num_layers))
        )
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(embedding_dim, 1)

    def forward(self, geometry: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = self.input_projection(geometry)
        sequence = self.temporal(sequence.transpose(1, 2)).transpose(1, 2)
        pooled = sequence.mean(dim=1)
        embedding = self.projection(pooled)
        return embedding, self.classifier(embedding).squeeze(-1)

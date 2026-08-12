"""Quality-conditioned reliability fusion."""

from __future__ import annotations

import torch
from torch import nn


class QualityAwareFusion(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 128,
        geometry_quality_dim: int = 5,
        texture_quality_dim: int = 5,
        dropout: float = 0.2,
        texture_gate_bias: float = 0.0,
    ) -> None:
        super().__init__()
        gate_input = embedding_dim * 2 + geometry_quality_dim + texture_quality_dim + 2
        self.gate = nn.Sequential(
            nn.Linear(gate_input, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 2),
        )
        with torch.no_grad():
            self.gate[-1].bias[1] += float(texture_gate_bias)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, 1),
        )

    def forward(
        self,
        geometry_embedding: torch.Tensor,
        texture_embedding: torch.Tensor,
        geometry_logit: torch.Tensor,
        texture_logit: torch.Tensor,
        geometry_quality: torch.Tensor,
        texture_quality: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        branch_uncertainty = torch.stack(
            [
                1.0 - torch.abs(torch.sigmoid(geometry_logit) - 0.5) * 2.0,
                1.0 - torch.abs(torch.sigmoid(texture_logit) - 0.5) * 2.0,
            ],
            dim=1,
        ).detach()
        gate_input = torch.cat(
            [
                geometry_embedding,
                texture_embedding,
                geometry_quality,
                texture_quality,
                branch_uncertainty,
            ],
            dim=1,
        )
        weights = torch.softmax(self.gate(gate_input), dim=1)
        fused = weights[:, :1] * geometry_embedding + weights[:, 1:] * texture_embedding
        return self.classifier(fused).squeeze(-1), weights

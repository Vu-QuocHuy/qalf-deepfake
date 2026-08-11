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
        gate_mode: str = "full",
        texture_gate_bias: float = 0.0,
    ) -> None:
        super().__init__()
        if gate_mode not in {"full", "content", "quality"}:
            raise ValueError(f"Unsupported gate mode: {gate_mode}")
        self.gate_mode = gate_mode
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
        if self.gate_mode == "quality":
            geometry_embedding_for_gate = torch.zeros_like(geometry_embedding)
            texture_embedding_for_gate = torch.zeros_like(texture_embedding)
            branch_uncertainty = torch.zeros_like(branch_uncertainty)
        else:
            geometry_embedding_for_gate = geometry_embedding
            texture_embedding_for_gate = texture_embedding
        if self.gate_mode == "content":
            geometry_quality_for_gate = torch.zeros_like(geometry_quality)
            texture_quality_for_gate = torch.zeros_like(texture_quality)
        else:
            geometry_quality_for_gate = geometry_quality
            texture_quality_for_gate = texture_quality
        gate_input = torch.cat(
            [
                geometry_embedding_for_gate,
                texture_embedding_for_gate,
                geometry_quality_for_gate,
                texture_quality_for_gate,
                branch_uncertainty,
            ],
            dim=1,
        )
        weights = torch.softmax(self.gate(gate_input), dim=1)
        fused = weights[:, :1] * geometry_embedding + weights[:, 1:] * texture_embedding
        return self.classifier(fused).squeeze(-1), weights


class ConcatFusion(nn.Module):
    def __init__(self, embedding_dim: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1),
        )

    def forward(
        self, geometry_embedding: torch.Tensor, texture_embedding: torch.Tensor
    ) -> torch.Tensor:
        return self.classifier(torch.cat([geometry_embedding, texture_embedding], dim=1)).squeeze(
            -1
        )

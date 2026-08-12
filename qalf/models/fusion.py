"""Quality-conditioned two-branch feature fusion."""

from __future__ import annotations

import torch
from torch import nn


class QualityAwareFusion(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 128,
        auxiliary_quality_dim: int = 5,
        texture_quality_dim: int = 5,
        dropout: float = 0.2,
        texture_gate_bias: float = 0.0,
    ) -> None:
        super().__init__()
        gate_input = embedding_dim * 2 + auxiliary_quality_dim + texture_quality_dim + 2
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
        auxiliary_embedding: torch.Tensor,
        texture_embedding: torch.Tensor,
        auxiliary_logit: torch.Tensor,
        texture_logit: torch.Tensor,
        auxiliary_quality: torch.Tensor,
        texture_quality: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        branch_uncertainty = torch.stack(
            [
                1.0 - torch.abs(torch.sigmoid(auxiliary_logit) - 0.5) * 2.0,
                1.0 - torch.abs(torch.sigmoid(texture_logit) - 0.5) * 2.0,
            ],
            dim=1,
        ).detach()
        gate_input = torch.cat(
            [
                auxiliary_embedding,
                texture_embedding,
                auxiliary_quality,
                texture_quality,
                branch_uncertainty,
            ],
            dim=1,
        )
        weights = torch.softmax(self.gate(gate_input), dim=1)
        fused = weights[:, :1] * auxiliary_embedding + weights[:, 1:] * texture_embedding
        return self.classifier(fused).squeeze(-1), weights


class ResidualInteractionFusion(nn.Module):
    """Use forensic residual evidence as a bounded correction to the RGB logit.

    Unlike convex feature mixing, the RGB decision has a direct skip path. The
    auxiliary stream learns when and how to correct it through multiplicative
    and absolute-difference interactions.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        auxiliary_quality_dim: int = 6,
        texture_quality_dim: int = 5,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.auxiliary_norm = nn.LayerNorm(embedding_dim)
        self.texture_norm = nn.LayerNorm(embedding_dim)
        input_dim = (
            embedding_dim * 4
            + auxiliary_quality_dim
            + texture_quality_dim
            + 4
        )
        self.interaction = nn.Sequential(
            nn.Linear(input_dim, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
        )
        self.confidence = nn.Linear(embedding_dim, 1)
        self.correction = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1),
        )
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(
        self,
        auxiliary_embedding: torch.Tensor,
        texture_embedding: torch.Tensor,
        auxiliary_logit: torch.Tensor,
        texture_logit: torch.Tensor,
        auxiliary_quality: torch.Tensor,
        texture_quality: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        auxiliary = self.auxiliary_norm(auxiliary_embedding)
        texture = self.texture_norm(texture_embedding)
        probabilities = torch.stack(
            (torch.sigmoid(auxiliary_logit), torch.sigmoid(texture_logit)),
            dim=1,
        )
        uncertainty = (1.0 - torch.abs(probabilities - 0.5) * 2.0).detach()
        interaction = self.interaction(
            torch.cat(
                (
                    auxiliary,
                    texture,
                    auxiliary * texture,
                    torch.abs(auxiliary - texture),
                    auxiliary_quality,
                    texture_quality,
                    auxiliary_logit[:, None],
                    texture_logit[:, None],
                    uncertainty,
                ),
                dim=1,
            )
        )
        confidence = torch.sigmoid(self.confidence(interaction)).squeeze(-1)
        correction = 2.0 * torch.tanh(self.correction(interaction).squeeze(-1))
        fused_logit = texture_logit + confidence * correction
        routing = torch.stack((confidence, 1.0 - confidence), dim=1)
        return fused_logit, routing

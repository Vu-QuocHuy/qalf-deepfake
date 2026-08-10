"""Training-only adversary for suppressing manipulation-specific shortcuts."""

from __future__ import annotations

import torch
from torch import nn


class _ReverseGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, strength: float) -> torch.Tensor:
        ctx.strength = float(strength)
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.strength * gradient, None


def reverse_gradient(inputs: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    """Act as identity in the forward pass and reverse encoder gradients."""

    return _ReverseGradient.apply(inputs, float(strength))


class MethodDiscriminator(nn.Module):
    """Predict the FF++ manipulation method from fake texture embeddings."""

    def __init__(
        self,
        embedding_dim: int,
        num_methods: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if num_methods < 2:
            raise ValueError("MethodDiscriminator requires at least two fake methods")
        hidden = max(32, embedding_dim // 2)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_methods),
        )

    def forward(self, embeddings: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
        return self.classifier(reverse_gradient(embeddings, strength))

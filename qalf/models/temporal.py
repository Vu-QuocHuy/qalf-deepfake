"""Attention-weighted temporal pooling with variance feature.

Replaces naive mean-pooling over frame embeddings with a learned attention
mechanism that lets the model focus on the most informative frames.  A
weighted temporal variance descriptor is appended so the classifier can
detect frame-to-frame inconsistencies — a strong deepfake signature that
simple averaging discards.
"""

from __future__ import annotations

import torch
from torch import nn


class TemporalAttentionPooling(nn.Module):
    """Attend over frame embeddings and capture temporal variance.

    Given ``T`` per-frame embeddings of dimension ``D``, the module:

    1. Computes a scalar importance score per frame via a small MLP.
    2. Produces an attended (weighted-sum) embedding.
    3. Computes the attention-weighted variance across frames.
    4. Concatenates the attended embedding and variance, then projects back
       to ``embedding_dim``.

    Parameters
    ----------
    embedding_dim:
        Dimensionality of each frame embedding (input and output).
    hidden_dim:
        Width of the attention score MLP hidden layer.
    dropout:
        Dropout probability applied after the final projection.
    """

    def __init__(
        self,
        embedding_dim: int = 192,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.attention_mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.Hardswish(),
            nn.Dropout(dropout),
        )

    def forward(self, frame_embeddings: torch.Tensor) -> torch.Tensor:
        """Pool ``T`` frame embeddings into a single clip embedding.

        Args:
            frame_embeddings: ``(B, T, D)`` — per-frame feature vectors.

        Returns:
            ``(B, D)`` — pooled clip-level embedding.
        """
        # Attention scores: (B, T, 1)
        scores = self.attention_mlp(frame_embeddings)
        weights = torch.softmax(scores, dim=1)  # (B, T, 1)

        # Weighted mean embedding: (B, D)
        attended = (weights * frame_embeddings).sum(dim=1)

        # Weighted variance: (B, D)
        diff_sq = (frame_embeddings - attended.unsqueeze(1)) ** 2
        variance = (weights * diff_sq).sum(dim=1)

        # Concatenate and project: (B, 2D) → (B, D)
        return self.projection(torch.cat([attended, variance], dim=1))

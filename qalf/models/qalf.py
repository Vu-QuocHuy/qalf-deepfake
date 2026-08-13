"""Texture-only QALF video classifier."""

from __future__ import annotations

import torch
from torch import nn

from .texture import TextureEncoder


class QALFModel(nn.Module):
    """Classify a video clip from uniformly sampled aligned RGB face frames."""

    def __init__(
        self,
        embedding_dim: int = 128,
        dropout: float = 0.2,
        texture_pretrained: bool = True,
        texture_backbone: str = "efficientnet_b0",
        srm_preprocess: bool = False,
    ) -> None:
        super().__init__()
        self.texture_encoder = TextureEncoder(
            embedding_dim=embedding_dim,
            dropout=dropout,
            pretrained=texture_pretrained,
            backbone=texture_backbone,
            srm_preprocess=srm_preprocess,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        embedding, logit = self.texture_encoder(batch["texture"])
        return {"logit": logit, "embedding": embedding}

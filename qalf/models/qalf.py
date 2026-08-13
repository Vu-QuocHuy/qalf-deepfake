"""Texture-only QALF video classifier (v1 and v2)."""

from __future__ import annotations

import torch
from torch import nn

from .texture import TextureEncoder


class QALFModel(nn.Module):
    """Classify a video clip from uniformly sampled aligned RGB face frames.

    The v2 architecture adds three optional lightweight modules on top of the
    v1 baseline:

    - ``frequency_preprocess`` — fixed SRM high-pass filter input layer.
    - ``multiscale`` — aggregate low/mid/high backbone features.
    - ``temporal_attention`` — learned attention pooling + variance descriptor.

    With all three flags set to ``False`` (the default), the model reproduces
    the original v1 mean-pooling architecture exactly.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        dropout: float = 0.2,
        texture_pretrained: bool = True,
        texture_backbone: str = "efficientnet_b0",
        frequency_preprocess: bool = False,
        multiscale: bool = False,
        temporal_attention: bool = False,
    ) -> None:
        super().__init__()
        self.texture_encoder = TextureEncoder(
            embedding_dim=embedding_dim,
            dropout=dropout,
            pretrained=texture_pretrained,
            backbone=texture_backbone,
            frequency_preprocess=frequency_preprocess,
            multiscale=multiscale,
            temporal_attention=temporal_attention,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        embedding, logit = self.texture_encoder(batch["texture"])
        return {"logit": logit, "embedding": embedding}

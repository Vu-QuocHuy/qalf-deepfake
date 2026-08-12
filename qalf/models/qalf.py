"""QALF model with one optional lightweight auxiliary branch."""

from __future__ import annotations

import torch
from torch import nn

from .fusion import QualityAwareFusion
from .geometry import GeometryEncoder
from .srm import SRMEncoder
from .texture import TextureEncoder

SUPPORTED_AUXILIARY_BRANCHES = {"geometry", "srm", "none"}


class QALFModel(nn.Module):
    def __init__(
        self,
        geometry_input_dim: int,
        geometry_hidden: int = 96,
        geometry_layers: int = 3,
        geometry_architecture: str = "tcn_mean",
        embedding_dim: int = 128,
        dropout: float = 0.2,
        texture_pretrained: bool = True,
        texture_backbone: str = "efficientnet_b0",
        geometry_quality_dim: int = 5,
        texture_quality_dim: int = 5,
        auxiliary_branch: str = "geometry",
        texture_gate_bias: float = 0.0,
    ) -> None:
        super().__init__()
        if auxiliary_branch not in SUPPORTED_AUXILIARY_BRANCHES:
            raise ValueError(f"Unsupported auxiliary branch: {auxiliary_branch}")
        self.auxiliary_branch = auxiliary_branch
        self.embedding_dim = embedding_dim
        self.geometry_encoder = (
            GeometryEncoder(
                geometry_input_dim,
                geometry_hidden,
                embedding_dim,
                geometry_layers,
                dropout,
                architecture=geometry_architecture,
            )
            if auxiliary_branch == "geometry"
            else None
        )
        self.srm_encoder = (
            SRMEncoder(embedding_dim, dropout) if auxiliary_branch == "srm" else None
        )
        self.texture_encoder = TextureEncoder(
            embedding_dim,
            dropout,
            texture_pretrained,
            backbone=texture_backbone,
        )
        auxiliary_quality_dim = (
            SRMEncoder.quality_dim if auxiliary_branch == "srm" else geometry_quality_dim
        )
        self.fusion = (
            None
            if auxiliary_branch == "none"
            else QualityAwareFusion(
                embedding_dim,
                auxiliary_quality_dim,
                texture_quality_dim,
                dropout,
                texture_gate_bias=texture_gate_bias,
            )
        )

    def forward_auxiliary(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.auxiliary_branch == "geometry":
            if self.geometry_encoder is None:
                raise RuntimeError("Geometry encoder is unavailable")
            embedding, logit = self.geometry_encoder(batch["geometry"])
            return embedding, logit, batch["geometry_quality"]
        if self.auxiliary_branch == "srm":
            if self.srm_encoder is None:
                raise RuntimeError("SRM encoder is unavailable")
            return self.srm_encoder(batch["texture"])
        raise RuntimeError("Texture-only mode has no auxiliary branch")

    def forward_texture(self, texture: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.texture_encoder(texture)

    def fuse_precomputed(
        self,
        auxiliary_embedding: torch.Tensor,
        auxiliary_logit: torch.Tensor,
        texture_embedding: torch.Tensor,
        texture_logit: torch.Tensor,
        auxiliary_quality: torch.Tensor,
        texture_quality: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.auxiliary_branch == "none":
            weights = texture_logit.new_tensor([0.0, 1.0]).expand(texture_logit.shape[0], -1)
            return texture_logit, weights
        if self.fusion is None:
            raise RuntimeError("Fusion module is unavailable")
        return self.fusion(
            auxiliary_embedding,
            texture_embedding,
            auxiliary_logit,
            texture_logit,
            auxiliary_quality,
            texture_quality,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | str]:
        texture_embedding, texture_logit = self.forward_texture(batch["texture"])
        if self.auxiliary_branch == "none":
            auxiliary_embedding = torch.zeros_like(texture_embedding)
            auxiliary_logit = torch.zeros_like(texture_logit)
            auxiliary_quality = texture_logit.new_zeros(texture_logit.shape[0], 0)
        else:
            auxiliary_embedding, auxiliary_logit, auxiliary_quality = self.forward_auxiliary(batch)
        fused_logit, weights = self.fuse_precomputed(
            auxiliary_embedding,
            auxiliary_logit,
            texture_embedding,
            texture_logit,
            auxiliary_quality,
            batch["texture_quality"],
        )
        return {
            "logit": fused_logit,
            "auxiliary_logit": auxiliary_logit,
            "texture_logit": texture_logit,
            "fusion_weights": weights,
            "auxiliary_embedding": auxiliary_embedding,
            "texture_embedding": texture_embedding,
            "auxiliary_quality": auxiliary_quality,
            "auxiliary_name": self.auxiliary_branch,
        }

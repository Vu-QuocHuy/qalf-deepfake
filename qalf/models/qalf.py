"""Retained QALF fusion model with an explicit texture-only control."""

from __future__ import annotations

import torch
from torch import nn

from .fusion import QualityAwareFusion
from .geometry import GeometryEncoder
from .texture import TextureEncoder


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
        fusion_mode: str = "quality",
        texture_gate_bias: float = 0.0,
        modality_dropout_probability: float = 0.0,
    ) -> None:
        super().__init__()
        if fusion_mode not in {"quality", "texture"}:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")
        self.fusion_mode = fusion_mode
        self.embedding_dim = embedding_dim
        if not 0.0 <= modality_dropout_probability <= 0.5:
            raise ValueError("modality_dropout_probability must be in [0, 0.5]")
        self.modality_dropout_probability = float(modality_dropout_probability)
        self.geometry_encoder = (
            None
            if fusion_mode == "texture"
            else GeometryEncoder(
                geometry_input_dim,
                geometry_hidden,
                embedding_dim,
                geometry_layers,
                dropout,
                architecture=geometry_architecture,
            )
        )
        self.texture_encoder = TextureEncoder(
            embedding_dim,
            dropout,
            texture_pretrained,
            backbone=texture_backbone,
        )
        self.fusion = (
            QualityAwareFusion(
                embedding_dim,
                geometry_quality_dim,
                texture_quality_dim,
                dropout,
                texture_gate_bias=texture_gate_bias,
            )
            if fusion_mode == "quality"
            else None
        )

    def forward_geometry(self, geometry: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.geometry_encoder is None:
            raise RuntimeError("Geometry encoder is disabled in texture-only mode")
        return self.geometry_encoder(geometry)

    def forward_texture(self, texture: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.texture_encoder(texture)

    def fuse_precomputed(
        self,
        geometry_embedding: torch.Tensor,
        geometry_logit: torch.Tensor,
        texture_embedding: torch.Tensor,
        texture_logit: torch.Tensor,
        geometry_quality: torch.Tensor,
        texture_quality: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse branch outputs, including a cached texture embedding during streaming."""

        if self.fusion_mode == "texture":
            weights = texture_logit.new_tensor([0.0, 1.0]).expand(texture_logit.shape[0], -1)
            return texture_logit, weights
        if self.fusion is None:
            raise RuntimeError(f"No fusion module configured for mode {self.fusion_mode}")
        return self.fusion(
            geometry_embedding,
            texture_embedding,
            geometry_logit,
            texture_logit,
            geometry_quality,
            texture_quality,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.fusion_mode == "texture":
            texture_embedding, texture_logit = self.forward_texture(batch["texture"])
            zeros = torch.zeros_like(texture_embedding)
            return {
                "logit": texture_logit,
                "geometry_logit": torch.zeros_like(texture_logit),
                "texture_logit": texture_logit,
                "fusion_weights": texture_logit.new_tensor([0.0, 1.0]).expand(
                    texture_logit.shape[0], -1
                ),
                "geometry_embedding": zeros,
                "texture_embedding": texture_embedding,
            }
        geometry_embedding, geometry_logit = self.forward_geometry(batch["geometry"])
        texture_embedding, texture_logit = self.forward_texture(batch["texture"])
        fusion_geometry_embedding = geometry_embedding
        fusion_geometry_logit = geometry_logit
        fusion_texture_embedding = texture_embedding
        fusion_texture_logit = texture_logit
        fusion_geometry_quality = batch["geometry_quality"]
        fusion_texture_quality = batch["texture_quality"]
        reliability_target = geometry_logit.new_full(
            (geometry_logit.shape[0],), -1, dtype=torch.long
        )
        if self.training and self.modality_dropout_probability > 0.0 and self.fusion is not None:
            probability = self.modality_dropout_probability
            draws = torch.rand(geometry_logit.shape[0], device=geometry_logit.device)
            geometry_missing = draws < probability
            texture_missing = (draws >= probability) & (draws < 2.0 * probability)
            # SBI has authentic geometry paired with a synthetic fake texture. Dropping its
            # texture would create an impossible fake target for the geometry branch.
            geometry_supervision = batch.get("geometry_loss_mask")
            if geometry_supervision is not None:
                texture_missing &= geometry_supervision.reshape(-1) > 0.5
            fusion_geometry_embedding = geometry_embedding.masked_fill(
                geometry_missing[:, None], 0.0
            )
            fusion_geometry_logit = geometry_logit.masked_fill(geometry_missing, 0.0)
            fusion_geometry_quality = batch["geometry_quality"].masked_fill(
                geometry_missing[:, None], 0.0
            )
            fusion_texture_embedding = texture_embedding.masked_fill(texture_missing[:, None], 0.0)
            fusion_texture_logit = texture_logit.masked_fill(texture_missing, 0.0)
            fusion_texture_quality = batch["texture_quality"].masked_fill(
                texture_missing[:, None], 0.0
            )
            reliability_target[geometry_missing] = 1
            reliability_target[texture_missing] = 0
        fused_logit, weights = self.fuse_precomputed(
            fusion_geometry_embedding,
            fusion_geometry_logit,
            fusion_texture_embedding,
            fusion_texture_logit,
            fusion_geometry_quality,
            fusion_texture_quality,
        )
        return {
            "logit": fused_logit,
            "geometry_logit": geometry_logit,
            "texture_logit": texture_logit,
            "fusion_weights": weights,
            "geometry_embedding": geometry_embedding,
            "texture_embedding": texture_embedding,
            "reliability_target": reliability_target,
        }

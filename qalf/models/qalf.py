"""End-to-end always-on QALF model with separable branch APIs for streaming deployment."""

from __future__ import annotations

import torch
from torch import nn

from .fusion import ConcatFusion, QualityAwareFusion
from .geometry import GeometryEncoder
from .texture import TextureEncoder


class QALFModel(nn.Module):
    def __init__(
        self,
        geometry_input_dim: int,
        geometry_hidden: int = 96,
        geometry_layers: int = 3,
        embedding_dim: int = 128,
        dropout: float = 0.2,
        texture_pretrained: bool = True,
        texture_backbone: str = "efficientnet_b0",
        texture_temporal_pooling: str = "mean",
        texture_mixstyle_probability: float = 0.0,
        texture_mixstyle_alpha: float = 0.1,
        texture_mixstyle_layers: tuple[int, ...] = (),
        geometry_quality_dim: int = 5,
        texture_quality_dim: int = 5,
        fusion_mode: str = "quality",
        texture_gate_bias: float = 0.0,
    ) -> None:
        super().__init__()
        if fusion_mode not in {
            "quality",
            "quality_only",
            "content_gate",
            "concat",
            "average",
            "geometry",
            "texture",
        }:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")
        self.fusion_mode = fusion_mode
        self.embedding_dim = embedding_dim
        self.geometry_encoder = (
            None
            if fusion_mode == "texture"
            else GeometryEncoder(
                geometry_input_dim,
                geometry_hidden,
                embedding_dim,
                geometry_layers,
                dropout,
            )
        )
        self.texture_encoder = (
            None
            if fusion_mode == "geometry"
            else TextureEncoder(
                embedding_dim,
                dropout,
                texture_pretrained,
                backbone=texture_backbone,
                temporal_pooling=texture_temporal_pooling,
                mixstyle_probability=texture_mixstyle_probability,
                mixstyle_alpha=texture_mixstyle_alpha,
                mixstyle_layers=texture_mixstyle_layers,
            )
        )
        gate_modes = {
            "quality": "full",
            "quality_only": "quality",
            "content_gate": "content",
        }
        self.fusion = (
            QualityAwareFusion(
                embedding_dim,
                geometry_quality_dim,
                texture_quality_dim,
                dropout,
                gate_mode=gate_modes[fusion_mode],
                texture_gate_bias=texture_gate_bias,
            )
            if fusion_mode in gate_modes
            else None
        )
        self.concat_fusion = (
            ConcatFusion(embedding_dim, dropout) if fusion_mode == "concat" else None
        )

    def forward_geometry(self, geometry: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.geometry_encoder is None:
            raise RuntimeError("Geometry encoder is disabled in texture-only mode")
        return self.geometry_encoder(geometry)

    def forward_texture(self, texture: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.texture_encoder is None:
            raise RuntimeError("Texture encoder is disabled in geometry-only mode")
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

        if self.fusion_mode == "geometry":
            weights = geometry_logit.new_tensor([1.0, 0.0]).expand(geometry_logit.shape[0], -1)
            return geometry_logit, weights
        if self.fusion_mode == "texture":
            weights = texture_logit.new_tensor([0.0, 1.0]).expand(texture_logit.shape[0], -1)
            return texture_logit, weights
        if self.fusion_mode == "average":
            fused_logit = 0.5 * (geometry_logit + texture_logit)
            weights = fused_logit.new_full((fused_logit.shape[0], 2), 0.5)
            return fused_logit, weights
        if self.fusion_mode == "concat":
            assert self.concat_fusion is not None
            fused_logit = self.concat_fusion(geometry_embedding, texture_embedding)
            weights = fused_logit.new_full((fused_logit.shape[0], 2), 0.5)
            return fused_logit, weights
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
        if self.fusion_mode == "geometry":
            zeros = torch.zeros_like(geometry_embedding)
            return {
                "logit": geometry_logit,
                "geometry_logit": geometry_logit,
                "texture_logit": torch.zeros_like(geometry_logit),
                "fusion_weights": geometry_logit.new_tensor([1.0, 0.0]).expand(
                    geometry_logit.shape[0], -1
                ),
                "geometry_embedding": geometry_embedding,
                "texture_embedding": zeros,
            }
        texture_embedding, texture_logit = self.forward_texture(batch["texture"])
        fused_logit, weights = self.fuse_precomputed(
            geometry_embedding,
            geometry_logit,
            texture_embedding,
            texture_logit,
            batch["geometry_quality"],
            batch["texture_quality"],
        )
        return {
            "logit": fused_logit,
            "geometry_logit": geometry_logit,
            "texture_logit": texture_logit,
            "fusion_weights": weights,
            "geometry_embedding": geometry_embedding,
            "texture_embedding": texture_embedding,
        }

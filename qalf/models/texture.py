"""EfficientNet-B0 encoder for landmark-aligned full-face sequences."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

SUPPORTED_TEXTURE_BACKBONES = {"efficientnet_b0"}
SUPPORTED_TEMPORAL_POOLING = {"mean", "paired_residual", "dual_rate_residual"}


class _LockedTemporalDropout(nn.Module):
    """Apply one feature-dropout mask across every frame in a clip."""

    def __init__(self, probability: float) -> None:
        super().__init__()
        self.probability = float(probability)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability == 0.0:
            return sequence
        mask = torch.ones(
            (sequence.shape[0], 1, sequence.shape[2]),
            dtype=sequence.dtype,
            device=sequence.device,
        )
        return sequence * F.dropout(mask, p=self.probability, training=True)


class PairedTemporalResidualPool(nn.Module):
    """Tiny bounded temporal correction over adjacent frame pairs.

    The mean embedding remains the main path. The output layer starts at zero,
    and the fixed residual scale prevents the temporal branch from replacing
    the spatial representation while it learns pairwise inconsistencies.
    """

    def __init__(
        self,
        embedding_dim: int,
        bottleneck_dim: int = 32,
        residual_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if bottleneck_dim < 8:
            raise ValueError("temporal bottleneck_dim must be at least 8")
        if not 0.0 < residual_scale <= 1.0:
            raise ValueError("temporal residual_scale must be in (0, 1]")
        self.residual_scale = float(residual_scale)
        self.input_projection = nn.Sequential(
            nn.Linear(embedding_dim * 2, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
        )
        self.depthwise = nn.Conv1d(
            bottleneck_dim,
            bottleneck_dim,
            kernel_size=3,
            padding=1,
            groups=bottleneck_dim,
            bias=False,
        )
        self.temporal_norm = nn.LayerNorm(bottleneck_dim)
        self.pointwise = nn.Linear(bottleneck_dim, bottleneck_dim)
        self.output_projection = nn.Linear(bottleneck_dim, embedding_dim)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, frame_embeddings: torch.Tensor) -> torch.Tensor:
        if frame_embeddings.ndim != 3:
            raise ValueError("paired temporal pooling expects (batch, frames, embedding_dim)")
        if frame_embeddings.shape[1] < 2 or frame_embeddings.shape[1] % 2:
            raise ValueError("paired temporal pooling requires an even frame count >= 2")

        base = frame_embeddings.mean(dim=1)
        pairs = frame_embeddings.reshape(
            frame_embeddings.shape[0],
            frame_embeddings.shape[1] // 2,
            2,
            frame_embeddings.shape[2],
        )
        pair_context = pairs.mean(dim=2) - base.unsqueeze(1)
        pair_change = (pairs[:, :, 1] - pairs[:, :, 0]).abs()
        sequence = self.input_projection(torch.cat((pair_context, pair_change), dim=-1))
        update = self.depthwise(sequence.transpose(1, 2)).transpose(1, 2)
        update = self.pointwise(F.gelu(self.temporal_norm(update)))
        summary = (sequence + update).mean(dim=1)
        raw_correction = self.output_projection(summary)
        correction = self.residual_scale * torch.tanh(
            raw_correction / self.residual_scale
        )
        return base + correction


class DualRateTemporalResidualPool(nn.Module):
    """Bounded correction from the centered consecutive frame burst.

    The caller supplies the canonical mean-path embedding separately. This
    lets the base retain the baseline's standard per-frame dropout while the
    temporal branch observes clean projected embeddings. Half of the input
    frames are expected to be the centered local burst produced by dual-rate
    sampling; the remaining frames preserve global texture coverage.
    """

    def __init__(
        self,
        embedding_dim: int,
        bottleneck_dim: int = 32,
        residual_scale: float = 0.05,
    ) -> None:
        super().__init__()
        if bottleneck_dim < 8:
            raise ValueError("temporal bottleneck_dim must be at least 8")
        if not 0.0 < residual_scale <= 1.0:
            raise ValueError("temporal residual_scale must be in (0, 1]")
        self.residual_scale = float(residual_scale)
        self.change_projection = nn.Sequential(
            nn.Linear(embedding_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
        )
        self.summary_projection = nn.Sequential(
            nn.Linear(2 * bottleneck_dim, bottleneck_dim),
            nn.GELU(),
        )
        self.output_projection = nn.Linear(bottleneck_dim, embedding_dim)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        frame_embeddings: torch.Tensor,
        base_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if frame_embeddings.ndim != 3:
            raise ValueError("dual-rate temporal pooling expects (batch, frames, embedding_dim)")
        frames = int(frame_embeddings.shape[1])
        if frames < 4 or frames % 4:
            raise ValueError(
                "dual-rate temporal pooling requires a frame count divisible by 4"
            )
        base = (
            frame_embeddings.mean(dim=1)
            if base_embedding is None
            else base_embedding
        )
        if base.shape != frame_embeddings.shape[:1] + frame_embeddings.shape[2:]:
            raise ValueError("dual-rate base embedding shape does not match frame embeddings")

        local_frames = frames // 2
        local_start = (frames - local_frames) // 2
        local = frame_embeddings[:, local_start : local_start + local_frames]
        local = F.layer_norm(local, (local.shape[-1],))
        changes = (local[:, 1:] - local[:, :-1]).abs()
        encoded = self.change_projection(changes)
        summary = torch.cat(
            (
                encoded.mean(dim=1),
                encoded.std(dim=1, unbiased=False),
            ),
            dim=-1,
        )
        raw_correction = self.output_projection(self.summary_projection(summary))
        correction = self.residual_scale * torch.tanh(
            raw_correction / self.residual_scale
        )
        return base + correction


def _build_backbone(name: str, pretrained: bool) -> tuple[nn.Module, nn.Module, int]:
    if name != "efficientnet_b0":
        raise ValueError(f"Unsupported texture backbone: {name}")
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)
    feature_dim = int(model.classifier[-1].in_features)
    return model.features, model.avgpool, feature_dim


class TextureEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 128,
        dropout: float = 0.2,
        pretrained: bool = True,
        backbone: str = "efficientnet_b0",
        temporal_pooling: str = "mean",
        temporal_bottleneck: int = 32,
        temporal_residual_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if backbone not in SUPPORTED_TEXTURE_BACKBONES:
            raise ValueError(f"Unsupported texture backbone: {backbone}")
        if temporal_pooling not in SUPPORTED_TEMPORAL_POOLING:
            raise ValueError(
                f"Unsupported temporal pooling: {temporal_pooling}; "
                f"choose one of {sorted(SUPPORTED_TEMPORAL_POOLING)}"
            )
        self.backbone_name = backbone
        self.temporal_pooling = temporal_pooling
        self.features, self.pool, feature_dim = _build_backbone(backbone, pretrained)
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.Hardswish(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(embedding_dim, 1)
        # Create optional modules after the shared classifier so a fixed seed
        # initializes every baseline parameter identically across ablations.
        self.temporal_locked_dropout = (
            _LockedTemporalDropout(dropout) if temporal_pooling == "paired_residual" else None
        )
        if temporal_pooling == "paired_residual":
            self.temporal_encoder: nn.Module | None = PairedTemporalResidualPool(
                embedding_dim,
                bottleneck_dim=int(temporal_bottleneck),
                residual_scale=float(temporal_residual_scale),
            )
        elif temporal_pooling == "dual_rate_residual":
            self.temporal_encoder = DualRateTemporalResidualPool(
                embedding_dim,
                bottleneck_dim=int(temporal_bottleneck),
                residual_scale=float(temporal_residual_scale),
            )
        else:
            self.temporal_encoder = None

    def forward(self, texture: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, frames, channels, height, width = texture.shape
        if channels != 3:
            raise ValueError(f"Texture encoder expects 3 channels, received {channels}")
        output = texture.reshape(batch * frames, channels, height, width)
        for layer in self.features:
            output = layer(output)
        output = self.pool(output).flatten(1)
        if self.temporal_encoder is None:
            output = self.projection(output).reshape(batch, frames, -1)
            embedding = output.mean(dim=1)
        elif self.temporal_pooling == "paired_residual":
            # The baseline projection dropout is independent per frame and
            # would create synthetic temporal differences. Project cleanly,
            # then use one locked dropout mask for the whole sequence.
            output = self.projection[0](output)
            output = self.projection[1](output)
            output = self.projection[2](output).reshape(batch, frames, -1)
            if self.temporal_locked_dropout is None:
                raise RuntimeError("temporal locked dropout is not initialized")
            output = self.temporal_locked_dropout(output)
            embedding = self.temporal_encoder(output)
        else:
            # Keep the canonical base path unchanged: standard dropout is
            # applied independently before mean pooling. The temporal branch
            # receives the clean projection so its local changes are not
            # dominated by synthetic dropout differences.
            clean = self.projection[0](output)
            clean = self.projection[1](clean)
            clean = self.projection[2](clean)
            clean_sequence = clean.reshape(batch, frames, -1)
            base = self.projection[3](clean).reshape(batch, frames, -1).mean(dim=1)
            if not isinstance(self.temporal_encoder, DualRateTemporalResidualPool):
                raise RuntimeError("dual-rate temporal encoder is not initialized")
            embedding = self.temporal_encoder(clean_sequence, base)
        return embedding, self.classifier(embedding).squeeze(-1)

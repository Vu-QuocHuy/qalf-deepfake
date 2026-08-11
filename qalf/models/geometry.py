"""Temporal and graph encoders for pose-decoupled landmark features."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

SUPPORTED_GEOMETRY_ARCHITECTURES = {
    "tcn_mean",
    "tcn_attentive",
    "graph_attentive",
    "graph_rigid_attentive",
}


class CausalDepthwiseBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        kernel_size = 3
        self.left_padding = dilation * (kernel_size - 1)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=channels,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        output = F.pad(inputs, (self.left_padding, 0))
        output = self.depthwise(output)
        output = self.pointwise(output)
        output = self.norm(output)
        output = F.silu(output)
        output = self.dropout(output)
        return residual + output


class AttentiveStatisticsPooling(nn.Module):
    """Retain persistent, variable, and transient temporal evidence."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(channels, max(channels // 2, 1)),
            nn.Tanh(),
            nn.Linear(max(channels // 2, 1), 1),
        )
        self.output = nn.Sequential(
            nn.Linear(channels * 3, channels),
            nn.LayerNorm(channels),
            nn.SiLU(),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.attention(sequence), dim=1)
        mean = torch.sum(weights * sequence, dim=1)
        variance = torch.sum(weights * (sequence - mean[:, None, :]).square(), dim=1)
        standard_deviation = torch.sqrt(variance.clamp_min(1e-6))
        maximum = sequence.amax(dim=1)
        return self.output(torch.cat([mean, standard_deviation, maximum], dim=1))


class SpatialGraphBlock(nn.Module):
    """A lightweight message-passing block over a clip-specific landmark graph."""

    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.self_projection = nn.Linear(channels, channels)
        self.neighbor_projection = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, nodes: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        neighbors = torch.einsum("bij,btjc->btic", adjacency, nodes)
        output = self.self_projection(nodes) + self.neighbor_projection(neighbors)
        output = self.dropout(F.silu(self.norm(output)))
        return nodes + output


def _knn_adjacency(positions: torch.Tensor, neighbors: int = 4) -> torch.Tensor:
    """Build a symmetric local graph from mean aligned landmark locations."""

    node_count = int(positions.shape[1])
    if node_count < 2:
        return positions.new_ones((positions.shape[0], node_count, node_count))
    k = min(max(int(neighbors), 1), node_count - 1)
    differences = positions[:, :, None, :] - positions[:, None, :, :]
    distances = differences.square().sum(dim=-1)
    indices = distances.topk(k + 1, dim=-1, largest=False).indices[..., 1:]
    adjacency = torch.zeros_like(distances)
    adjacency.scatter_(2, indices, 1.0)
    adjacency = torch.maximum(adjacency, adjacency.transpose(1, 2))
    identity = torch.eye(node_count, device=positions.device, dtype=positions.dtype)
    adjacency = adjacency + identity.unsqueeze(0)
    return adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)


class GeometryEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 96,
        embedding_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        architecture: str = "tcn_mean",
        node_count: int = 0,
        node_feature_dim: int = 0,
        rigid_feature_dim: int = 0,
        graph_neighbors: int = 4,
    ) -> None:
        super().__init__()
        if architecture not in SUPPORTED_GEOMETRY_ARCHITECTURES:
            raise ValueError(f"Unsupported geometry architecture: {architecture}")
        self.architecture = architecture
        self.node_count = int(node_count)
        self.node_feature_dim = int(node_feature_dim)
        self.rigid_feature_dim = int(rigid_feature_dim)
        self.graph_neighbors = int(graph_neighbors)
        self.uses_graph = architecture in {"graph_attentive", "graph_rigid_attentive"}
        self.uses_rigid = architecture == "graph_rigid_attentive"

        if self.uses_graph:
            if self.node_count < 2 or self.node_feature_dim != 9:
                raise ValueError(
                    "Graph geometry requires aligned position/velocity/acceleration "
                    "features with at least two landmarks"
                )
            if self.uses_rigid and self.rigid_feature_dim < 1:
                raise ValueError("graph_rigid_attentive requires rigid motion features")
            if not self.uses_rigid and self.rigid_feature_dim != 0:
                raise ValueError("graph_attentive does not consume rigid motion features")
            expected = self.node_count * self.node_feature_dim + self.rigid_feature_dim + 1
            if input_dim != expected:
                raise ValueError(
                    f"Structured geometry input mismatch: expected {expected}, got {input_dim}"
                )
            # Append the per-frame landmark validity bit to every node.
            self.node_projection = nn.Sequential(
                nn.Linear(self.node_feature_dim + 1, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            )
            self.graph = nn.ModuleList(
                SpatialGraphBlock(hidden_dim, dropout) for _ in range(num_layers)
            )
            self.node_attention = nn.Linear(hidden_dim, 1)
            self.input_projection = None
        else:
            self.input_projection = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            )
            self.node_projection = None
            self.graph = nn.ModuleList()
            self.node_attention = None

        self.temporal = nn.Sequential(
            *(CausalDepthwiseBlock(hidden_dim, 2**layer, dropout) for layer in range(num_layers))
        )
        if self.uses_rigid:
            self.rigid_projection = nn.Sequential(
                nn.Linear(self.rigid_feature_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            )
            self.rigid_temporal = nn.Sequential(
                *(
                    CausalDepthwiseBlock(hidden_dim, 2**layer, dropout)
                    for layer in range(num_layers)
                )
            )
            self.stream_fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            )
        else:
            self.rigid_projection = None
            self.rigid_temporal = None
            self.stream_fusion = None

        self.statistics_pooling = (
            AttentiveStatisticsPooling(hidden_dim) if architecture != "tcn_mean" else None
        )
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(embedding_dim, 1)

    def _structured_sequence(self, geometry: torch.Tensor) -> torch.Tensor:
        batch_size, frames, _ = geometry.shape
        component_width = self.node_count * 3
        node_width = self.node_count * self.node_feature_dim
        node_flat = geometry[..., :node_width]
        aligned, velocity, acceleration = node_flat.split(component_width, dim=-1)
        nodes = torch.cat(
            [
                aligned.reshape(batch_size, frames, self.node_count, 3),
                velocity.reshape(batch_size, frames, self.node_count, 3),
                acceleration.reshape(batch_size, frames, self.node_count, 3),
            ],
            dim=-1,
        )
        valid = geometry[..., -1:].unsqueeze(2).expand(-1, -1, self.node_count, -1)
        nodes_with_validity = torch.cat([nodes, valid], dim=-1)
        assert self.node_projection is not None
        output = self.node_projection(nodes_with_validity)
        adjacency = _knn_adjacency(
            nodes[..., :3].mean(dim=1).detach(),
            self.graph_neighbors,
        ).to(dtype=output.dtype)
        for block in self.graph:
            output = block(output, adjacency)
        assert self.node_attention is not None
        node_weights = torch.softmax(self.node_attention(output), dim=2)
        nonrigid = torch.sum(node_weights * output, dim=2)
        nonrigid = self.temporal(nonrigid.transpose(1, 2)).transpose(1, 2)

        if not self.uses_rigid:
            return nonrigid
        rigid = geometry[..., node_width : node_width + self.rigid_feature_dim]
        assert self.rigid_projection is not None
        assert self.rigid_temporal is not None
        assert self.stream_fusion is not None
        rigid = self.rigid_projection(rigid)
        rigid = self.rigid_temporal(rigid.transpose(1, 2)).transpose(1, 2)
        return self.stream_fusion(torch.cat([nonrigid, rigid], dim=-1))

    def forward(self, geometry: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.uses_graph:
            sequence = self._structured_sequence(geometry)
        else:
            assert self.input_projection is not None
            sequence = self.input_projection(geometry)
            sequence = self.temporal(sequence.transpose(1, 2)).transpose(1, 2)
        pooled = (
            sequence.mean(dim=1)
            if self.statistics_pooling is None
            else self.statistics_pooling(sequence)
        )
        embedding = self.projection(pooled)
        return embedding, self.classifier(embedding).squeeze(-1)

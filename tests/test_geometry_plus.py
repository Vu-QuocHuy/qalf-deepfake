from __future__ import annotations

import unittest

import numpy as np
import torch
from torch import nn

from qalf.data.geometry import (
    RIGID_FEATURE_DIM,
    build_geometry_features,
    geometry_feature_layout,
    geometry_input_dim,
)
from qalf.engine import qalf_loss
from qalf.models.geometry import GeometryEncoder
from qalf.models.qalf import QALFModel


class GeometryFeatureTests(unittest.TestCase):
    def test_rigid_mode_has_expected_finite_layout(self) -> None:
        frames = 8
        indices = (1, 4, 10, 33, 61, 133, 152, 197, 263, 291, 362, 454)
        rng = np.random.default_rng(7)
        base = rng.normal(0.0, 0.1, size=(468, 3)).astype(np.float32)
        base[:, :2] += 0.5
        landmarks = []
        for frame in range(frames):
            points = base.copy()
            points[:, 0] += 0.002 * frame
            points[61:292, 1] += 0.001 * np.sin(frame)
            landmarks.append(points)
        landmarks_array = np.stack(landmarks)
        features, quality = build_geometry_features(
            landmarks_array,
            np.ones(frames, dtype=bool),
            np.arange(frames, dtype=np.float32) / 25.0,
            indices=indices,
            feature_mode="aligned_motion_rigid_3d",
        )

        self.assertEqual(
            features.shape,
            (frames, geometry_input_dim(indices, "aligned_motion_rigid_3d")),
        )
        self.assertEqual(
            geometry_feature_layout(indices, "aligned_motion_rigid_3d"),
            (len(indices), 9, RIGID_FEATURE_DIM),
        )
        self.assertTrue(np.isfinite(features).all())
        self.assertTrue(np.isfinite(quality).all())


class GeometryEncoderTests(unittest.TestCase):
    def test_all_new_architectures_produce_embeddings(self) -> None:
        batch, frames, nodes = 3, 8, 6
        cases = (
            ("tcn_attentive", nodes * 9 + 1, 0, 0, 0),
            ("graph_attentive", nodes * 9 + 1, nodes, 9, 0),
            (
                "graph_rigid_attentive",
                nodes * 9 + RIGID_FEATURE_DIM + 1,
                nodes,
                9,
                RIGID_FEATURE_DIM,
            ),
        )
        for architecture, input_dim, node_count, node_dim, rigid_dim in cases:
            with self.subTest(architecture=architecture):
                encoder = GeometryEncoder(
                    input_dim=input_dim,
                    hidden_dim=16,
                    embedding_dim=12,
                    num_layers=2,
                    dropout=0.0,
                    architecture=architecture,
                    node_count=node_count,
                    node_feature_dim=node_dim,
                    rigid_feature_dim=rigid_dim,
                    graph_neighbors=2,
                )
                geometry = torch.randn(batch, frames, input_dim)
                geometry[..., -1] = 1.0
                geometry.requires_grad_(True)
                embedding, logit = encoder(geometry)
                self.assertEqual(embedding.shape, (batch, 12))
                self.assertEqual(logit.shape, (batch,))
                (embedding.mean() + logit.mean()).backward()
                self.assertIsNotNone(geometry.grad)


class GeometryLossTests(unittest.TestCase):
    def test_consistency_view_is_differentiable(self) -> None:
        model = QALFModel(
            geometry_input_dim=10,
            geometry_hidden=12,
            geometry_layers=1,
            embedding_dim=8,
            dropout=0.0,
            texture_pretrained=False,
            fusion_mode="geometry",
            geometry_consistency_noise_std=0.1,
        )
        model.train()
        geometry = torch.randn(4, 8, 10)
        geometry[..., -1] = 1.0
        outputs = model({"geometry": geometry})
        consistency = outputs["geometry_consistency_loss"]
        self.assertEqual(consistency.ndim, 0)
        self.assertTrue(torch.isfinite(consistency))
        consistency.backward()
        self.assertTrue(
            any(parameter.grad is not None for parameter in model.geometry_encoder.parameters())
        )

    def test_class_balanced_loss_weights_real_and_fake_equally(self) -> None:
        geometry_logits = torch.tensor([-2.0, 2.0, -2.0])
        labels = torch.tensor([0.0, 0.0, 1.0])
        outputs = {
            "logit": torch.zeros(3),
            "texture_logit": torch.zeros(3),
            "geometry_logit": geometry_logits,
        }
        _, parts = qalf_loss(
            outputs,
            labels,
            nn.BCEWithLogitsLoss(),
            geometry_weight=1.0,
            texture_weight=0.0,
            geometry_loss_mask=torch.ones(3),
            geometry_class_balanced=True,
        )
        criterion = nn.BCEWithLogitsLoss()
        expected = 0.5 * (
            criterion(geometry_logits[:2], labels[:2]) + criterion(geometry_logits[2:], labels[2:])
        )
        self.assertAlmostEqual(parts["geometry"], float(expected), places=6)

    def test_reliability_gate_loss_uses_only_corrupted_samples(self) -> None:
        outputs = {
            "logit": torch.zeros(3),
            "texture_logit": torch.zeros(3),
            "geometry_logit": torch.zeros(3),
            "fusion_weights": torch.tensor(
                [[0.9, 0.1], [0.2, 0.8], [0.4, 0.6]], dtype=torch.float32
            ),
            "reliability_target": torch.tensor([0, 1, -1]),
        }
        loss, parts = qalf_loss(
            outputs,
            torch.tensor([0.0, 1.0, 1.0]),
            nn.BCEWithLogitsLoss(),
            geometry_weight=0.0,
            texture_weight=0.0,
            reliability_gate_weight=1.0,
        )
        expected_reliability = -0.5 * (np.log(0.9) + np.log(0.8))
        self.assertAlmostEqual(parts["reliability"], expected_reliability, places=6)
        self.assertAlmostEqual(float(loss), float(np.log(2.0)) + expected_reliability, places=6)


if __name__ == "__main__":
    unittest.main()

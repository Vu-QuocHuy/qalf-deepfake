from __future__ import annotations

import unittest

import numpy as np
import torch
from torch import nn

from qalf.engine import qalf_loss
from qalf.models.geometry import GeometryEncoder
from qalf.models.qalf import _modality_dropout_masks


class GeometryEncoderTests(unittest.TestCase):
    def test_attentive_encoder_produces_differentiable_embeddings(self) -> None:
        encoder = GeometryEncoder(
            input_dim=55,
            hidden_dim=16,
            embedding_dim=12,
            num_layers=2,
            dropout=0.0,
            architecture="tcn_attentive",
        )
        geometry = torch.randn(3, 8, 55, requires_grad=True)
        embedding, logit = encoder(geometry)
        self.assertEqual(embedding.shape, (3, 12))
        self.assertEqual(logit.shape, (3,))
        (embedding.mean() + logit.mean()).backward()
        self.assertIsNotNone(geometry.grad)


class GeometryLossTests(unittest.TestCase):
    def test_sbi_aware_dropout_excludes_sbi_from_both_missing_modalities(self) -> None:
        draws = torch.tensor([0.05, 0.20, 0.05, 0.20, 0.80])
        geometry_supervision = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0])

        ordinary_geometry, ordinary_texture = _modality_dropout_masks(
            draws,
            probability=0.15,
            geometry_supervision=geometry_supervision,
            exclude_unsupervised_geometry=False,
        )
        aware_geometry, aware_texture = _modality_dropout_masks(
            draws,
            probability=0.15,
            geometry_supervision=geometry_supervision,
            exclude_unsupervised_geometry=True,
        )

        self.assertEqual(ordinary_geometry.tolist(), [True, False, True, False, False])
        self.assertEqual(ordinary_texture.tolist(), [False, True, False, False, False])
        self.assertEqual(aware_geometry.tolist(), [True, False, False, False, False])
        self.assertEqual(aware_texture.tolist(), [False, True, False, False, False])

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

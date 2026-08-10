import unittest

import torch

from qalf.models.fusion import QualityAwareFusion
from qalf.models.qalf import QALFModel


class FusionAblationTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)
        self.batch = 4
        self.embedding_dim = 16
        self.geometry_embedding = torch.randn(self.batch, self.embedding_dim)
        self.texture_embedding = torch.randn(self.batch, self.embedding_dim)
        self.geometry_logit = torch.randn(self.batch)
        self.texture_logit = torch.randn(self.batch)
        self.geometry_quality = torch.randn(self.batch, 5)
        self.texture_quality = torch.randn(self.batch, 5)

    def _forward(self, module, geometry_embedding=None, geometry_quality=None):
        module.eval()
        return module(
            geometry_embedding if geometry_embedding is not None else self.geometry_embedding,
            self.texture_embedding,
            self.geometry_logit,
            self.texture_logit,
            geometry_quality if geometry_quality is not None else self.geometry_quality,
            self.texture_quality,
        )

    def test_gate_ablations_have_equal_parameter_budget(self) -> None:
        modules = [
            QualityAwareFusion(self.embedding_dim, gate_mode=mode)
            for mode in ("full", "content", "quality")
        ]
        counts = [sum(parameter.numel() for parameter in module.parameters()) for module in modules]
        self.assertEqual(len(set(counts)), 1)

    def test_quality_only_weights_ignore_content(self) -> None:
        module = QualityAwareFusion(self.embedding_dim, gate_mode="quality")
        _, first = self._forward(module)
        _, second = self._forward(module, geometry_embedding=self.geometry_embedding + 100.0)
        torch.testing.assert_close(first, second)

    def test_content_only_weights_ignore_quality(self) -> None:
        module = QualityAwareFusion(self.embedding_dim, gate_mode="content")
        _, first = self._forward(module)
        _, second = self._forward(module, geometry_quality=self.geometry_quality + 100.0)
        torch.testing.assert_close(first, second)

    def test_single_branch_model_does_not_count_inactive_branch(self) -> None:
        geometry = QALFModel(
            geometry_input_dim=32,
            geometry_hidden=8,
            geometry_layers=1,
            embedding_dim=16,
            texture_pretrained=False,
            fusion_mode="geometry",
        )
        self.assertIsNone(geometry.texture_encoder)
        self.assertIsNone(geometry.fusion)
        self.assertIsNone(geometry.concat_fusion)


if __name__ == "__main__":
    unittest.main()

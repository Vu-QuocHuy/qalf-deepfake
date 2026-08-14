from __future__ import annotations

import unittest

import torch

from qalf.models.texture import TemporalResidualTCN


class TemporalResidualTCNTests(unittest.TestCase):
    def test_zero_initialized_residual_matches_mean_pool(self) -> None:
        module = TemporalResidualTCN(embedding_dim=16, bottleneck_dim=8)
        embeddings = torch.randn(2, 8, 16)
        output = module(embeddings)
        torch.testing.assert_close(output, embeddings.mean(dim=1))

    def test_temporal_module_preserves_clip_shape(self) -> None:
        module = TemporalResidualTCN(embedding_dim=16, bottleneck_dim=8)
        output = module(torch.randn(3, 8, 16))
        self.assertEqual(tuple(output.shape), (3, 16))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from qalf.models.texture import SUPPORTED_TEXTURE_BACKBONES, _build_backbone


class TextureBackboneTests(unittest.TestCase):
    def test_only_efficientnet_b0_and_b1_are_supported(self) -> None:
        self.assertEqual(
            SUPPORTED_TEXTURE_BACKBONES,
            {"efficientnet_b0", "efficientnet_b1"},
        )

    def test_efficientnet_b1_builds_without_pretrained_download(self) -> None:
        features, pool, feature_dim = _build_backbone("efficientnet_b1", pretrained=False)

        self.assertIsNotNone(features)
        self.assertIsNotNone(pool)
        self.assertEqual(feature_dim, 1280)

    def test_removed_mobilenet_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _build_backbone("mobilenet_v3_small", pretrained=False)


if __name__ == "__main__":
    unittest.main()

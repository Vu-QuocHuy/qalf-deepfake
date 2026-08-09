import unittest

from qalf.data.extraction import ExtractionConfig, temporal_sample_indices


class TemporalSamplingTest(unittest.TestCase):
    def test_fractional_source_to_target_rate(self) -> None:
        indices, protocol = temporal_sample_indices(100, 25.0, 4, 10.0)
        self.assertEqual(protocol, "center_fixed_rate")
        self.assertEqual(indices, [46, 48, 51, 53])
        self.assertEqual(len(indices), len(set(indices)))

    def test_short_span_uses_full_video_fallback(self) -> None:
        indices, protocol = temporal_sample_indices(64, 30.0, 64, 10.0)
        self.assertEqual(protocol, "full_video_uniform_fallback")
        self.assertEqual(indices, list(range(64)))

    def test_invalid_config_is_rejected(self) -> None:
        config = ExtractionConfig(target_fps=0)
        with self.assertRaises(ValueError):
            config.validate()


if __name__ == "__main__":
    unittest.main()

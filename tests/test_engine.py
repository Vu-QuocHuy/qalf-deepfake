import unittest

from qalf.engine import aggregate_predictions


class AggregationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.predictions = {
            "label": [0, 0, 1, 1],
            "score": [0.1, 0.3, 0.6, 0.8],
            "geometry_score": [0.2, 0.2, 0.7, 0.7],
            "texture_score": [0.1, 0.4, 0.5, 0.9],
            "geometry_weight": [0.5, 0.5, 0.5, 0.5],
            "texture_weight": [0.5, 0.5, 0.5, 0.5],
            "clip_index": [0, 1, 0, 1],
            "video_id": ["a", "a", "b", "b"],
            "method": ["original", "original", "fake", "fake"],
            "dataset": ["example"] * 4,
        }

    def test_mean_video_aggregation(self) -> None:
        result = aggregate_predictions(self.predictions, "mean")
        self.assertEqual(result["score"], [0.2, 0.7])
        self.assertEqual(result["clip_count"], [2, 2])

    def test_topk_uses_strongest_fused_clip(self) -> None:
        result = aggregate_predictions(self.predictions, "topk", top_k=1)
        self.assertEqual(result["score"], [0.3, 0.8])
        self.assertEqual(result["texture_score"], [0.4, 0.9])


if __name__ == "__main__":
    unittest.main()

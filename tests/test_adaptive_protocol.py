import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_adaptive import _load_threshold


class AdaptiveProtocolTest(unittest.TestCase):
    def _write(self, payload: dict) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "threshold.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_accepts_ffpp_validation_threshold(self) -> None:
        path = self._write(
            {
                "metrics": {"threshold": 0.42},
                "threshold_selection": {"datasets": ["ffpp"], "splits": ["val"]},
            }
        )
        threshold, provenance = _load_threshold(path)
        self.assertEqual(threshold, 0.42)
        self.assertEqual(provenance["splits"], ["val"])

    def test_rejects_target_test_threshold(self) -> None:
        path = self._write(
            {
                "metrics": {"threshold": 0.42},
                "threshold_selection": {
                    "datasets": ["celebdf_v2"],
                    "splits": ["test"],
                },
            }
        )
        with self.assertRaises(ValueError):
            _load_threshold(path)

    def test_rejects_missing_provenance(self) -> None:
        path = self._write({"threshold": 0.42})
        with self.assertRaises(ValueError):
            _load_threshold(path)


if __name__ == "__main__":
    unittest.main()

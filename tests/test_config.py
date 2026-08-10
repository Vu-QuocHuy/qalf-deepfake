import json
import tempfile
import unittest
from pathlib import Path

from qalf.config import load_config
from qalf.data.geometry import (
    DEFAULT_GEOMETRY_FEATURE_MODE,
    GEOMETRY_FEATURE_MODES,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _minimal_config(geometry_mode: str | None = DEFAULT_GEOMETRY_FEATURE_MODE) -> dict:
    data = {
        "num_frames": 8,
        "texture_frames": 2,
    }
    if geometry_mode is not None:
        data["geometry_mode"] = geometry_mode
    return {"data": data, "model": {}, "training": {}}


class ConfigTest(unittest.TestCase):
    def _load_temporary(self, config: dict) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            return load_config(path)

    def test_repository_training_config_loads(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "ffpp_to_celebdf.json")
        self.assertEqual(config["data"]["geometry_mode"], DEFAULT_GEOMETRY_FEATURE_MODE)

    def test_every_geometry_mode_is_accepted(self) -> None:
        for mode in GEOMETRY_FEATURE_MODES:
            with self.subTest(mode=mode):
                config = self._load_temporary(_minimal_config(mode))
                self.assertEqual(config["data"]["geometry_mode"], mode)

    def test_missing_geometry_mode_uses_shared_default(self) -> None:
        config = self._load_temporary(_minimal_config(None))
        self.assertEqual(config["data"]["geometry_mode"], DEFAULT_GEOMETRY_FEATURE_MODE)

    def test_unknown_geometry_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown_mode"):
            self._load_temporary(_minimal_config("unknown_mode"))


if __name__ == "__main__":
    unittest.main()

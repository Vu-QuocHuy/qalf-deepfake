import tempfile
import unittest
from pathlib import Path

from qalf.data.manifest import VideoRecord, load_manifest, manifest_summary, write_manifest


class ManifestTest(unittest.TestCase):
    def test_round_trip(self):
        record = VideoRecord(
            dataset="synthetic",
            split="train",
            video_id="video-1",
            label=1,
            method="fake",
            source_video="video.mp4",
            frames=["frames/000000.jpg", "frames/000001.jpg"],
            source_indices=[10, 13],
            timestamps_sec=[0.4, 0.52],
            fps=25.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            write_manifest([record], path)
            loaded = load_manifest(path)
        self.assertEqual(loaded[0].video_id, record.video_id)
        self.assertEqual(manifest_summary(loaded)["videos"], 1)

    def test_rejects_length_mismatch(self):
        record = VideoRecord(
            dataset="synthetic",
            split="train",
            video_id="bad",
            label=0,
            method="real",
            source_video="video.mp4",
            frames=["one.jpg"],
            source_indices=[1, 2],
        )
        with self.assertRaises(ValueError):
            record.validate()


if __name__ == "__main__":
    unittest.main()

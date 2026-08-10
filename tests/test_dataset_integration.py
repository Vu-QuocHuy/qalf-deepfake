import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from qalf.data.dataset import QALFVideoDataset
from qalf.data.geometry import geometry_input_dim
from qalf.data.manifest import VideoRecord, write_manifest


class DatasetIntegrationTest(unittest.TestCase):
    def test_existing_468_xyz_cache_feeds_new_3d_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame_paths = []
            for index in range(8):
                relative = Path("frames") / f"{index:03d}.jpg"
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                image = np.full((64, 64, 3), 96 + index, dtype=np.uint8)
                self.assertTrue(cv2.imwrite(str(path), image))
                frame_paths.append(str(relative))

            rng = np.random.default_rng(9)
            base = rng.uniform(0.2, 0.8, size=(468, 3)).astype(np.float32)
            landmarks = np.repeat(base[None, ...], 8, axis=0)
            landmark_relative = Path("cache") / "example.npz"
            landmark_path = root / landmark_relative
            landmark_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                landmark_path,
                landmarks=landmarks,
                detected=np.ones(8, dtype=bool),
                timestamps_sec=np.arange(8, dtype=np.float64) / 8.0,
            )
            record = VideoRecord(
                dataset="ffpp",
                split="train",
                video_id="example",
                label=0,
                method="original",
                source_video="example.mp4",
                frames=frame_paths,
                source_indices=list(range(8)),
                timestamps_sec=(np.arange(8) / 8.0).tolist(),
                fps=25.0,
                landmark_path=str(landmark_relative),
            )
            manifest = root / "manifest.jsonl"
            write_manifest([record], manifest)
            dataset = QALFVideoDataset(
                manifest,
                root,
                root,
                num_frames=8,
                texture_frames=2,
                image_size=64,
                geometry_mode="aligned_motion_3d",
                training=False,
            )
            item = dataset[0]
            self.assertEqual(tuple(item["geometry"].shape), (8, geometry_input_dim()))
            self.assertEqual(tuple(item["geometry_quality"].shape), (5,))
            self.assertEqual(tuple(item["texture"].shape), (2, 3, 64, 64))
            self.assertEqual(tuple(item["texture_quality"].shape), (5,))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from qalf.data.manifest import VideoRecord, load_manifest, write_manifest
from qalf.data.subsample import derive_strided_landmark_dataset


class StridedDatasetTests(unittest.TestCase):
    def test_derives_32_frame_manifest_and_landmarks_without_copying_jpegs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_manifest = root / "source.jsonl"
            input_landmarks = root / "source-landmarks"
            output_manifest = root / "derived" / "manifest.jsonl"
            output_landmarks = root / "derived-landmarks"
            landmark_path = Path("ffpp/train/real/Original/video.npz")
            frames = [f"frames/video/{index:06d}.jpg" for index in range(64)]
            write_manifest(
                [
                    VideoRecord(
                        dataset="ffpp",
                        split="train",
                        video_id="video",
                        label=0,
                        method="Original",
                        source_video="video.mp4",
                        frames=frames,
                        source_indices=list(range(0, 192, 3)),
                        timestamps_sec=[index / 10.0 for index in range(64)],
                        fps=30.0,
                        landmark_path=str(landmark_path),
                        quality={"landmark_detected": 64},
                    )
                ],
                input_manifest,
            )
            source_cache = input_landmarks / landmark_path
            source_cache.parent.mkdir(parents=True)
            np.savez_compressed(
                source_cache,
                landmarks=np.arange(64 * 3, dtype=np.float32).reshape(64, 1, 3),
                detected=np.asarray([index % 5 != 0 for index in range(64)]),
                image_sizes=np.repeat(np.asarray([[256, 256]]), 64, axis=0),
                source_indices=np.arange(0, 192, 3),
                timestamps_sec=np.arange(64) / 10.0,
            )

            report = derive_strided_landmark_dataset(
                input_manifest,
                input_landmarks,
                output_manifest,
                output_landmarks,
            )

            [record] = load_manifest(output_manifest)
            self.assertEqual(record.frames, frames[::2])
            self.assertEqual(record.source_indices, list(range(0, 192, 6)))
            self.assertEqual(record.timestamps_sec, [index / 10.0 for index in range(0, 64, 2)])
            self.assertEqual(record.quality["derived_frame_count"], 32)
            self.assertAlmostEqual(record.quality["derived_effective_fps"], 5.0)
            self.assertEqual(report["derived_frames"], 32)
            with np.load(output_landmarks / landmark_path) as cache:
                self.assertEqual(cache["landmarks"].shape, (32, 1, 3))
                np.testing.assert_array_equal(cache["source_indices"], np.arange(0, 192, 6))
                np.testing.assert_allclose(cache["timestamps_sec"], np.arange(0, 64, 2) / 10.0)

    def test_rejects_noncanonical_source_length(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "source.jsonl"
            write_manifest(
                [
                    VideoRecord(
                        dataset="ffpp",
                        split="train",
                        video_id="short",
                        label=0,
                        method="Original",
                        source_video="short.mp4",
                        frames=["one.jpg", "two.jpg"],
                        landmark_path="short.npz",
                    )
                ],
                manifest,
            )
            with self.assertRaisesRegex(ValueError, "expected 64 source frames"):
                derive_strided_landmark_dataset(
                    manifest,
                    root / "landmarks",
                    root / "output.jsonl",
                    root / "output-landmarks",
                )


if __name__ == "__main__":
    unittest.main()

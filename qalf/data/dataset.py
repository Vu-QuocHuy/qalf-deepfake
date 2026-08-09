"""PyTorch dataset for paired temporal geometry and canonical skin texture."""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .geometry import DEFAULT_LANDMARK_INDICES, build_geometry_features, geometry_input_dim
from .manifest import VideoRecord, load_manifest

IMAGE_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def _clip_indices(
    total: int,
    count: int,
    training: bool,
    clip_index: int = 0,
    clips_per_video: int = 1,
) -> np.ndarray:
    if total < count:
        raise ValueError(f"Sequence has {total} frames but the model requires {count}")
    if total == count:
        if not training and clips_per_video > 1:
            raise ValueError("Cannot draw multiple distinct clips when sequence length equals clip length")
        return np.arange(total)
    if training:
        start = random.randint(0, total - count)
    elif clips_per_video == 1:
        start = (total - count) // 2
    else:
        if not 0 <= clip_index < clips_per_video:
            raise IndexError(f"Invalid clip index {clip_index}/{clips_per_video}")
        starts = np.rint(np.linspace(0, total - count, clips_per_video)).astype(np.int64)
        if len(set(starts.tolist())) != clips_per_video:
            raise ValueError("clips_per_video requests duplicate temporal windows")
        start = int(starts[clip_index])
    return np.arange(start, start + count)


def _canonical_skin_map(
    image_rgb: np.ndarray,
    landmarks: np.ndarray,
    detected: bool,
    output_size: int,
    texture_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    if texture_mode not in {"canonical_skin", "full_face"}:
        raise ValueError(f"Unsupported texture mode: {texture_mode}")
    height, width = image_rgb.shape[:2]
    if detected and np.isfinite(landmarks).all() and landmarks.shape[0] > 386:
        pixels = landmarks[:, :2] * np.asarray([width, height], dtype=np.float32)
        left_eye = pixels[[33, 133, 159, 145]].mean(axis=0)
        right_eye = pixels[[362, 263, 386, 374]].mean(axis=0)
        mouth = pixels[[13, 14, 61, 291]].mean(axis=0)
        source = np.float32([left_eye, right_eye, mouth])
        target = np.float32(
            [
                [0.32 * output_size, 0.38 * output_size],
                [0.68 * output_size, 0.38 * output_size],
                [0.50 * output_size, 0.72 * output_size],
            ]
        )
        transform = cv2.getAffineTransform(source, target)
        canonical = cv2.warpAffine(
            image_rgb,
            transform,
            (output_size, output_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        landmark_quality = 1.0
    else:
        canonical = cv2.resize(image_rgb, (output_size, output_size), interpolation=cv2.INTER_AREA)
        landmark_quality = 0.0

    if texture_mode == "canonical_skin":
        mask = np.zeros((output_size, output_size), dtype=np.uint8)
        regions = (
            (0.23, 0.08, 0.77, 0.33),
            (0.08, 0.43, 0.43, 0.72),
            (0.57, 0.43, 0.92, 0.72),
            (0.42, 0.35, 0.58, 0.67),
        )
        for x1, y1, x2, y2 in regions:
            cv2.rectangle(
                mask,
                (int(x1 * output_size), int(y1 * output_size)),
                (int(x2 * output_size), int(y2 * output_size)),
                255,
                thickness=-1,
            )
        output = np.full_like(canonical, 127)
        output[mask > 0] = canonical[mask > 0]
    else:
        output = canonical
    quality = _texture_quality(output, width, height, landmark_quality)
    return output, quality


def _texture_quality(
    image: np.ndarray, source_width: int, source_height: int, landmark_quality: float
) -> np.ndarray:
    gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return np.asarray(
        [
            gray.mean() / 255.0,
            gray.std() / 128.0,
            min(np.log1p(cv2.Laplacian(gray, cv2.CV_64F).var()) / 10.0, 1.0),
            min(min(source_width, source_height) / 512.0, 1.0),
            landmark_quality,
        ],
        dtype=np.float32,
    )


def _augment(image: np.ndarray) -> np.ndarray:
    output = image.astype(np.float32)
    if random.random() < 0.5:
        alpha = random.uniform(0.85, 1.15)
        beta = random.uniform(-12.0, 12.0)
        output = np.clip(output * alpha + beta, 0, 255)
    if random.random() < 0.2:
        output = cv2.GaussianBlur(output, (3, 3), sigmaX=random.uniform(0.1, 1.2))
    if random.random() < 0.3:
        quality = random.randint(45, 90)
        ok, encoded = cv2.imencode(
            ".jpg", cv2.cvtColor(output.astype(np.uint8), cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), quality],
        )
        if ok:
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            output = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    return output.astype(np.uint8)


class QALFVideoDataset(Dataset):
    geometry_quality_dim = 5
    texture_quality_dim = 5

    def __init__(
        self,
        manifest_path: str | Path,
        frame_root: str | Path,
        landmark_root: str | Path,
        num_frames: int = 32,
        texture_frames: int = 4,
        image_size: int = 128,
        training: bool = False,
        clips_per_video: int = 1,
        geometry_mode: str = "aligned_motion",
        texture_mode: str = "canonical_skin",
        landmark_indices: tuple[int, ...] = DEFAULT_LANDMARK_INDICES,
    ) -> None:
        self.records = load_manifest(manifest_path)
        self.frame_root = Path(frame_root)
        self.landmark_root = Path(landmark_root)
        self.num_frames = int(num_frames)
        self.texture_frames = int(texture_frames)
        self.image_size = int(image_size)
        if self.num_frames < 2:
            raise ValueError("num_frames must be at least two for temporal derivatives")
        if not 1 <= self.texture_frames <= self.num_frames:
            raise ValueError("texture_frames must be between one and num_frames")
        if self.image_size < 32:
            raise ValueError("image_size must be at least 32")
        self.training = bool(training)
        self.clips_per_video = int(clips_per_video)
        if self.clips_per_video < 1:
            raise ValueError("clips_per_video must be at least one")
        if self.training and self.clips_per_video != 1:
            raise ValueError("Training uses one randomly sampled clip per video and epoch")
        self.landmark_indices = landmark_indices
        self.geometry_mode = geometry_mode
        self.geometry_input_dim = geometry_input_dim(landmark_indices, geometry_mode)
        if texture_mode not in {"canonical_skin", "full_face"}:
            raise ValueError(f"Unsupported texture mode: {texture_mode}")
        self.texture_mode = texture_mode
        for record in self.records:
            if not record.landmark_path:
                raise ValueError(f"Missing landmark_path in manifest: {record.video_id}")
            if len(record.frames) < self.num_frames:
                raise ValueError(
                    f"{record.video_id}: {len(record.frames)} frames, need {self.num_frames}"
                )

    @property
    def labels(self) -> list[int]:
        return [
            record.label
            for record in self.records
            for _ in range(self.clips_per_video)
        ]

    def __len__(self) -> int:
        return len(self.records) * self.clips_per_video

    def __getitem__(self, item: int) -> dict[str, object]:
        record_index, clip_index = divmod(item, self.clips_per_video)
        record: VideoRecord = self.records[record_index]
        with np.load(self.landmark_root / str(record.landmark_path)) as cache:
            landmarks = cache["landmarks"].copy()
            detected = cache["detected"].copy()
            timestamps = (
                cache["timestamps_sec"].copy() if "timestamps_sec" in cache.files else None
            )
        if len(landmarks) != len(record.frames) or len(detected) != len(record.frames):
            raise ValueError(
                f"{record.video_id}: landmark cache length does not match manifest frames"
            )
        if timestamps is not None and len(timestamps) not in {0, len(record.frames)}:
            raise ValueError(f"{record.video_id}: timestamp cache length mismatch")
        clip = _clip_indices(
            len(record.frames),
            self.num_frames,
            self.training,
            clip_index,
            self.clips_per_video,
        )
        geometry, geometry_quality = build_geometry_features(
            landmarks[clip],
            detected[clip],
            timestamps[clip] if timestamps is not None and len(timestamps) else None,
            self.landmark_indices,
            self.geometry_mode,
        )

        texture_positions = np.rint(
            np.linspace(0, len(clip) - 1, self.texture_frames)
        ).astype(np.int64)
        texture_tensors: list[np.ndarray] = []
        texture_qualities: list[np.ndarray] = []
        for position in texture_positions:
            source_index = int(clip[position])
            image_bgr = cv2.imread(str(self.frame_root / record.frames[source_index]))
            if image_bgr is None:
                raise FileNotFoundError(self.frame_root / record.frames[source_index])
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            canonical, quality = _canonical_skin_map(
                image_rgb,
                landmarks[source_index],
                bool(detected[source_index]),
                self.image_size,
                self.texture_mode,
            )
            if self.training:
                canonical = _augment(canonical)
                quality = _texture_quality(
                    canonical,
                    image_rgb.shape[1],
                    image_rgb.shape[0],
                    float(quality[-1]),
                )
            normalized = canonical.astype(np.float32) / 255.0
            normalized = (normalized - IMAGE_MEAN) / IMAGE_STD
            texture_tensors.append(normalized.transpose(2, 0, 1))
            texture_qualities.append(quality)

        return {
            "geometry": torch.from_numpy(geometry),
            "geometry_quality": torch.from_numpy(geometry_quality),
            "texture": torch.from_numpy(np.stack(texture_tensors).astype(np.float32)),
            "texture_quality": torch.from_numpy(np.mean(texture_qualities, axis=0)),
            "label": torch.tensor(float(record.label), dtype=torch.float32),
            "video_id": record.video_id,
            "method": record.method,
            "dataset": record.dataset,
            "clip_index": torch.tensor(clip_index, dtype=torch.int64),
        }

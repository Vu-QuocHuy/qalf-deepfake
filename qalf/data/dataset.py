"""PyTorch dataset for aligned full-face video clips."""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .manifest import VideoRecord, load_manifest
from .sbi import (
    SAMPLE_ORIGINAL_FAKE,
    SAMPLE_REAL,
    SAMPLE_SBI,
    face_mask_from_aligned_landmarks,
    generate_self_blended_clip,
    resolve_sbi_config,
)

IMAGE_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
TEXTURE_MODES = frozenset({"full_face"})


DEFAULT_TEXTURE_AUGMENTATION = {
    "flip_probability": 0.5,
    "brightness_contrast_probability": 0.6,
    "gamma_probability": 0.3,
    "gamma_min": 0.75,
    "gamma_max": 1.25,
    "downsample_probability": 0.4,
    "downsample_min_scale": 0.45,
    "blur_probability": 0.3,
    "noise_probability": 0.3,
    "noise_std": 6.0,
    "jpeg_probability": 0.5,
    "jpeg_min_quality": 35.0,
    "jpeg_max_quality": 90.0,
}


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
            raise ValueError(
                "Cannot draw multiple distinct clips when sequence length equals clip length"
            )
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


def _aligned_full_face(
    image_rgb: np.ndarray,
    landmarks: np.ndarray,
    detected: bool,
    output_size: int,
) -> tuple[np.ndarray, np.ndarray | None]:
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
        aligned_landmarks = cv2.transform(pixels[None, :, :2], transform)[0]
    else:
        canonical = cv2.resize(image_rgb, (output_size, output_size), interpolation=cv2.INTER_AREA)
        aligned_landmarks = None
    return canonical, aligned_landmarks


def _resolve_texture_augmentation(config: dict[str, float]) -> dict[str, float]:
    unknown = set(config) - set(DEFAULT_TEXTURE_AUGMENTATION)
    if unknown:
        raise ValueError(f"Unknown texture augmentation keys: {sorted(unknown)}")
    settings = {**DEFAULT_TEXTURE_AUGMENTATION, **config}
    for name in (
        "flip_probability",
        "brightness_contrast_probability",
        "gamma_probability",
        "downsample_probability",
        "blur_probability",
        "noise_probability",
        "jpeg_probability",
    ):
        if not 0.0 <= settings[name] <= 1.0:
            raise ValueError(f"Texture augmentation {name} must be in [0, 1]")
    if not 0.0 < settings["gamma_min"] <= settings["gamma_max"]:
        raise ValueError("Texture augmentation gamma range is invalid")
    if not 0.0 < settings["downsample_min_scale"] <= 1.0:
        raise ValueError("Texture augmentation downsample_min_scale must be in (0, 1]")
    if settings["noise_std"] < 0:
        raise ValueError("Texture augmentation noise_std cannot be negative")
    if not 1 <= settings["jpeg_min_quality"] <= settings["jpeg_max_quality"] <= 100:
        raise ValueError("Texture augmentation JPEG quality range is invalid")
    return settings


def _augment(image: np.ndarray, settings: dict[str, float]) -> np.ndarray:
    output = image.astype(np.float32)
    if random.random() < settings["flip_probability"]:
        output = np.ascontiguousarray(output[:, ::-1])
    if random.random() < settings["brightness_contrast_probability"]:
        alpha = random.uniform(0.85, 1.15)
        beta = random.uniform(-12.0, 12.0)
        output = np.clip(output * alpha + beta, 0, 255)
    if random.random() < settings["gamma_probability"]:
        gamma = random.uniform(settings["gamma_min"], settings["gamma_max"])
        output = 255.0 * np.power(np.clip(output / 255.0, 0.0, 1.0), gamma)
    if random.random() < settings["downsample_probability"]:
        height, width = output.shape[:2]
        scale = random.uniform(settings["downsample_min_scale"], 0.9)
        small_width = max(16, round(width * scale))
        small_height = max(16, round(height * scale))
        output = cv2.resize(output, (small_width, small_height), interpolation=cv2.INTER_AREA)
        output = cv2.resize(output, (width, height), interpolation=cv2.INTER_LINEAR)
    if random.random() < settings["blur_probability"]:
        output = cv2.GaussianBlur(output, (3, 3), sigmaX=random.uniform(0.1, 1.2))
    if random.random() < settings["noise_probability"] and settings["noise_std"] > 0:
        noise = np.random.normal(0.0, settings["noise_std"], size=output.shape)
        output = np.clip(output + noise, 0, 255)
    if random.random() < settings["jpeg_probability"]:
        quality = random.randint(
            int(settings["jpeg_min_quality"]), int(settings["jpeg_max_quality"])
        )
        ok, encoded = cv2.imencode(
            ".jpg",
            cv2.cvtColor(output.astype(np.uint8), cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), quality],
        )
        if ok:
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            output = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    return output.astype(np.uint8)


class QALFVideoDataset(Dataset):
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
        texture_mode: str = "full_face",
        fake_methods: Sequence[str] | None = None,
        texture_augmentation: dict[str, float] | None = None,
        sbi_config: dict[str, object] | None = None,
    ) -> None:
        records = load_manifest(manifest_path)
        self.fake_methods: tuple[str, ...] | None = None
        if fake_methods is not None:
            requested_methods = tuple(str(method) for method in fake_methods)
            if not requested_methods:
                raise ValueError("fake_methods cannot be empty")
            if len(set(requested_methods)) != len(requested_methods):
                raise ValueError("fake_methods contains duplicate names")
            available_methods = {record.method for record in records if record.label == 1}
            missing_methods = set(requested_methods) - available_methods
            if missing_methods:
                raise ValueError(
                    f"Manifest does not contain requested fake methods: {sorted(missing_methods)}"
                )
            self.fake_methods = requested_methods
            allowed_methods = set(requested_methods)
            records = [
                record
                for record in records
                if record.label == 0 or record.method in allowed_methods
            ]
        self.records = records
        self.frame_root = Path(frame_root)
        self.landmark_root = Path(landmark_root)
        self.num_frames = int(num_frames)
        self.texture_frames = int(texture_frames)
        self.image_size = int(image_size)
        if self.num_frames < 2:
            raise ValueError("num_frames must be at least two for a video clip")
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
        if texture_augmentation is None:
            self.texture_augmentation = dict(DEFAULT_TEXTURE_AUGMENTATION)
        elif texture_augmentation:
            self.texture_augmentation = _resolve_texture_augmentation(texture_augmentation)
        else:
            self.texture_augmentation = {}
        if texture_mode not in TEXTURE_MODES:
            raise ValueError(
                f"Unsupported texture mode: {texture_mode}; only full_face is retained"
            )
        self.texture_mode = texture_mode
        self.sbi_config = resolve_sbi_config(sbi_config)
        self.sbi_enabled = self.training and bool(self.sbi_config["enabled"])
        if self.sbi_enabled and not any(record.label == 0 for record in self.records):
            raise ValueError("SBI training requires real records")
        if self.sbi_enabled and not any(record.label == 1 for record in self.records):
            raise ValueError("SBI hybrid training requires original fake records")
        for record in self.records:
            if not record.landmark_path:
                raise ValueError(f"Missing landmark_path in manifest: {record.video_id}")
            if len(record.frames) < self.num_frames:
                raise ValueError(
                    f"{record.video_id}: {len(record.frames)} frames, need {self.num_frames}"
                )
        self.sample_specs: list[tuple[int, str]] = [
            (
                record_index,
                SAMPLE_REAL if record.label == 0 else SAMPLE_ORIGINAL_FAKE,
            )
            for record_index, record in enumerate(self.records)
        ]
        if self.sbi_enabled:
            self.sample_specs.extend(
                (record_index, SAMPLE_SBI)
                for record_index, record in enumerate(self.records)
                if record.label == 0
            )
        # Keep the number of optimizer samples per epoch comparable with the
        # baseline even though SBI adds addressable companion entries.
        self.samples_per_epoch = len(self.records) * self.clips_per_video

    @property
    def labels(self) -> list[int]:
        return [
            1 if sample_type == SAMPLE_SBI else self.records[record_index].label
            for record_index, sample_type in self.sample_specs
            for _ in range(self.clips_per_video)
        ]

    @property
    def sampling_strata(self) -> list[str]:
        return [
            sample_type for _, sample_type in self.sample_specs for _ in range(self.clips_per_video)
        ]

    def __len__(self) -> int:
        return len(self.sample_specs) * self.clips_per_video

    def __getitem__(self, item: int) -> dict[str, object]:
        sample_index, clip_index = divmod(item, self.clips_per_video)
        record_index, sample_type = self.sample_specs[sample_index]
        record: VideoRecord = self.records[record_index]
        with np.load(self.landmark_root / str(record.landmark_path)) as cache:
            landmarks = cache["landmarks"].copy()
            detected = cache["detected"].copy()
        if len(landmarks) != len(record.frames) or len(detected) != len(record.frames):
            raise ValueError(
                f"{record.video_id}: landmark cache length does not match manifest frames"
            )
        clip = _clip_indices(
            len(record.frames),
            self.num_frames,
            self.training,
            clip_index,
            self.clips_per_video,
        )
        texture_positions = np.rint(np.linspace(0, len(clip) - 1, self.texture_frames)).astype(
            np.int64
        )
        texture_tensors: list[np.ndarray] = []
        canonical_frames: list[np.ndarray] = []
        canonical_landmarks: list[np.ndarray | None] = []
        for position in texture_positions:
            source_index = int(clip[position])
            image_bgr = cv2.imread(str(self.frame_root / record.frames[source_index]))
            if image_bgr is None:
                raise FileNotFoundError(self.frame_root / record.frames[source_index])
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            canonical, aligned_landmarks = _aligned_full_face(
                image_rgb,
                landmarks[source_index],
                bool(detected[source_index]),
                self.image_size,
            )
            canonical_frames.append(canonical)
            canonical_landmarks.append(aligned_landmarks)

        if sample_type == SAMPLE_SBI:
            aligned_landmarks = next(
                (value for value in canonical_landmarks if value is not None),
                None,
            )
            face_mask = face_mask_from_aligned_landmarks(
                aligned_landmarks,
                self.image_size,
            )
            generated, _, _ = generate_self_blended_clip(
                np.stack(canonical_frames).astype(np.uint8),
                face_mask,
                self.sbi_config,
            )
            canonical_frames = list(generated)

        for canonical in canonical_frames:
            if self.training and self.texture_augmentation:
                canonical = _augment(canonical, self.texture_augmentation)
            normalized = canonical.astype(np.float32) / 255.0
            normalized = (normalized - IMAGE_MEAN) / IMAGE_STD
            texture_tensors.append(normalized.transpose(2, 0, 1))

        return {
            "texture": torch.from_numpy(np.stack(texture_tensors).astype(np.float32)),
            "label": torch.tensor(
                1.0 if sample_type == SAMPLE_SBI else float(record.label),
                dtype=torch.float32,
            ),
            "sample_type": sample_type,
            "video_id": record.video_id,
            "method": "SBI" if sample_type == SAMPLE_SBI else record.method,
            "dataset": record.dataset,
            "clip_index": torch.tensor(clip_index, dtype=torch.int64),
        }

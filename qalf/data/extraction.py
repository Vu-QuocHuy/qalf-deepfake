"""Leakage-aware video specification and temporally consistent main-face extraction."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

from .manifest import VideoRecord, load_manifest, manifest_summary, write_manifest

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


@dataclass(frozen=True)
class VideoSpec:
    dataset: str
    split: str
    video_id: str
    label: int
    method: str
    path: Path


@dataclass
class ExtractionConfig:
    frames_per_video: int = 64
    target_fps: float = 10.0
    mtcnn_batch_size: int = 8
    min_face_size: int = 64
    min_detection_probability: float = 0.90
    min_direct_detection_ratio: float = 0.75
    max_consecutive_missing: int = 6
    square_margin: float = 0.35
    output_size: int = 256
    jpeg_quality: int = 95
    min_crop_side: int = 64

    def validate(self) -> None:
        if self.frames_per_video < 2:
            raise ValueError("frames_per_video must be at least two")
        if self.target_fps <= 0:
            raise ValueError("target_fps must be positive")
        if self.mtcnn_batch_size < 1:
            raise ValueError("mtcnn_batch_size must be positive")
        if self.min_face_size < 1 or self.min_crop_side < 1 or self.output_size < 32:
            raise ValueError("minimum face/crop sizes must be positive")
        if not 0.0 <= self.min_detection_probability <= 1.0:
            raise ValueError("min_detection_probability must be in [0, 1]")
        if not 0.0 <= self.min_direct_detection_ratio <= 1.0:
            raise ValueError("min_direct_detection_ratio must be in [0, 1]")
        if not 0 <= self.max_consecutive_missing < self.frames_per_video:
            raise ValueError("max_consecutive_missing is outside the sampled sequence")
        if self.square_margin < 0:
            raise ValueError("square_margin cannot be negative")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1, 100]")


def _config_fingerprint(config: ExtractionConfig) -> str:
    """Stable identifier used to prevent unsafe resume across extraction settings."""

    fingerprint_payload = {
        "config": config.__dict__,
        "implementation": "temporal_main_face_v2_fractional_fps",
    }
    payload = json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _index_videos(directory: str | Path) -> dict[str, Path]:
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Video directory does not exist: {root}")
    index: dict[str, Path] = {}
    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            if path.stem in index:
                raise ValueError(f"Duplicate video id {path.stem!r} in {root}")
            index[path.stem] = path
    if not index:
        raise ValueError(f"No videos found in: {root}")
    return index


def _load_ffpp_pairs(path: Path) -> list[tuple[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    pairs: list[tuple[str, str]] = []
    for row in payload:
        if not isinstance(row, list) or len(row) != 2:
            raise ValueError(f"Invalid FF++ pair in {path}: {row!r}")
        left, right = (str(item).zfill(3) for item in row)
        pairs.append((left, right))
    return pairs


def build_ffpp_specs(
    dataset_root: str | Path,
    split_root: str | Path,
    methods: Sequence[str] = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"),
    compression: str = "c23",
) -> dict[str, list[VideoSpec]]:
    """Build official FF++ train/val/test records.

    Google DeepFakeDetection is intentionally unsupported here because combining its fake
    videos with FF++ real videos creates a source-domain shortcut. FaceShifter can be added
    explicitly if it follows the same directed FF++ ids and official split.
    """

    dataset_root = Path(dataset_root)
    split_root = Path(split_root)
    if "DeepFakeDetection" in methods:
        raise ValueError(
            "DeepFakeDetection uses a different real-video source. Exclude it or implement "
            "a source-matched DFD protocol with corresponding DFD real videos."
        )

    original_candidates = [
        dataset_root / "original",
        dataset_root / "original_sequences" / "youtube" / compression / "videos",
        dataset_root / "FaceForensics++" / "original_sequences" / "youtube" / compression / "videos",
    ]
    original_dir = next((path for path in original_candidates if path.is_dir()), None)
    if original_dir is None:
        raise FileNotFoundError("Could not locate FF++ original video directory")
    original_index = _index_videos(original_dir)

    method_indices: dict[str, dict[str, Path]] = {}
    for method in methods:
        candidates = [
            dataset_root / method,
            dataset_root / "manipulated_sequences" / method / compression / "videos",
            dataset_root / "FaceForensics++" / "manipulated_sequences" / method / compression / "videos",
        ]
        method_dir = next((path for path in candidates if path.is_dir()), None)
        if method_dir is None:
            raise FileNotFoundError(f"Could not locate FF++ method directory: {method}")
        method_indices[method] = _index_videos(method_dir)

    specs_by_split: dict[str, list[VideoSpec]] = {}
    original_ids_by_split: dict[str, set[str]] = {}
    expected_original_counts = {"train": 720, "val": 140, "test": 140}
    expected_pair_counts = {"train": 360, "val": 70, "test": 70}
    for split in ("train", "val", "test"):
        pairs = _load_ffpp_pairs(split_root / f"{split}.json")
        original_ids = {item for pair in pairs for item in pair}
        if (
            len(pairs) != expected_pair_counts[split]
            or len(original_ids) != expected_original_counts[split]
        ):
            raise ValueError(
                f"Unexpected FF++ {split} split size: "
                f"pairs={len(pairs)}, originals={len(original_ids)}"
            )
        original_ids_by_split[split] = original_ids
        specs: list[VideoSpec] = []
        for video_id in sorted(original_ids):
            if video_id not in original_index:
                raise FileNotFoundError(f"Missing FF++ original video: {video_id}")
            specs.append(VideoSpec("ffpp", split, video_id, 0, "original", original_index[video_id]))
        directed_ids = [item for left, right in pairs for item in (f"{left}_{right}", f"{right}_{left}")]
        for method in methods:
            for video_id in directed_ids:
                if video_id not in method_indices[method]:
                    raise FileNotFoundError(f"Missing FF++ {method} video: {video_id}")
                specs.append(
                    VideoSpec("ffpp", split, video_id, 1, method, method_indices[method][video_id])
                )
        specs_by_split[split] = specs

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = original_ids_by_split[left] & original_ids_by_split[right]
        if overlap:
            raise RuntimeError(f"FF++ identity leakage between {left}/{right}: {sorted(overlap)[:8]}")
    return specs_by_split


def build_celebdf_test_specs(
    dataset_root: str | Path,
    test_list: str | Path | None = None,
) -> list[VideoSpec]:
    """Build the official 518-video Celeb-DF-v2 test protocol.

    Labels are derived from directory names, not from the numeric list field. The official
    list uses 1 for real and 0 for fake, which is the opposite of this project's convention.
    """

    root = Path(dataset_root)
    list_path = Path(test_list) if test_list else root / "List_of_testing_videos.txt"
    if not list_path.is_file():
        raise FileNotFoundError(f"Celeb-DF test list not found: {list_path}")
    specs: list[VideoSpec] = []
    seen: set[str] = set()
    with list_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.strip().split(maxsplit=1)
            if len(fields) != 2:
                raise ValueError(f"Invalid Celeb-DF test line {line_number}: {line!r}")
            official_label, relative = fields
            relative_path = Path(relative)
            group = relative_path.parts[0]
            if group in {"Celeb-real", "YouTube-real"}:
                label, method, expected_official = 0, group, "1"
            elif group == "Celeb-synthesis":
                label, method, expected_official = 1, group, "0"
            else:
                raise ValueError(f"Unknown Celeb-DF group on line {line_number}: {group}")
            if official_label != expected_official:
                raise ValueError(
                    f"Celeb-DF label/path disagreement on line {line_number}: "
                    f"label={official_label}, path={relative}"
                )
            path = root / relative_path
            if not path.is_file():
                raise FileNotFoundError(f"Missing Celeb-DF test video: {path}")
            video_id = f"{group}__{path.stem}"
            if video_id in seen:
                raise ValueError(f"Duplicate Celeb-DF test video: {video_id}")
            seen.add(video_id)
            specs.append(VideoSpec("celebdf_v2", "test", video_id, label, method, path))
    if len(specs) != 518:
        raise ValueError(f"Expected 518 official Celeb-DF test videos, found {len(specs)}")
    return specs


def temporal_sample_indices(
    total_frames: int,
    source_fps: float,
    count: int,
    target_fps: float,
) -> tuple[list[int], str]:
    if count < 1:
        raise ValueError("Sample count must be positive")
    if target_fps <= 0:
        raise ValueError("Target FPS must be positive")
    if total_frames < count:
        raise ValueError(f"Video has only {total_frames} frames; need at least {count}")
    fps_was_assumed = not np.isfinite(source_fps) or source_fps <= 0
    if fps_was_assumed:
        source_fps = target_fps
    step = max(1.0, source_fps / target_fps)
    span = (count - 1) * step
    if span <= total_frames - 1:
        start = (total_frames - 1 - span) / 2.0
        indices = np.rint(start + np.arange(count, dtype=np.float64) * step)
        indices = indices.astype(np.int64).tolist()
        if len(set(indices)) != count:
            raise ValueError("Fixed-rate sampler generated duplicate frame indices")
        if fps_was_assumed:
            protocol = "center_assumed_target_rate"
        elif source_fps < target_fps:
            protocol = "center_native_rate_below_target"
        else:
            protocol = "center_fixed_rate"
        return indices, protocol
    indices = np.rint(np.linspace(0, total_frames - 1, count)).astype(np.int64).tolist()
    if len(set(indices)) != count:
        raise ValueError("Temporal sampler generated duplicate frame indices")
    return indices, "full_video_uniform_fallback"


def _read_target_frames(video_path: Path, indices: Sequence[int]) -> list[np.ndarray]:
    wanted = set(indices)
    decoded: dict[int, np.ndarray] = {}
    capture = cv2.VideoCapture(str(video_path))
    frame_index = 0
    while capture.isOpened() and frame_index <= indices[-1]:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index in wanted:
            decoded[frame_index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_index += 1
    capture.release()
    missing = [index for index in indices if index not in decoded]
    if missing:
        raise RuntimeError(f"Could not decode {len(missing)} requested frames: {missing[:8]}")
    return [decoded[index] for index in indices]


def _box_area(box: np.ndarray) -> float:
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def _box_iou(left: np.ndarray, right: np.ndarray) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = _box_area(left) + _box_area(right) - intersection
    return intersection / union if union > 0 else 0.0


def _track_main_face(
    boxes_per_frame: Sequence[np.ndarray | None],
    probs_per_frame: Sequence[np.ndarray | None],
    min_probability: float,
) -> tuple[list[np.ndarray | None], int]:
    selected: list[np.ndarray | None] = []
    previous: np.ndarray | None = None
    multi_face_frames = 0
    for boxes, probabilities in zip(boxes_per_frame, probs_per_frame, strict=True):
        candidates: list[tuple[np.ndarray, float]] = []
        if boxes is not None and probabilities is not None:
            for box, probability in zip(boxes, probabilities, strict=True):
                if box is not None and probability is not None and probability >= min_probability:
                    candidates.append((np.asarray(box, dtype=np.float32), float(probability)))
        multi_face_frames += int(len(candidates) > 1)
        if not candidates:
            selected.append(None)
            continue
        if previous is None:
            chosen = max(candidates, key=lambda item: (_box_area(item[0]), item[1]))[0]
        else:
            chosen = max(
                candidates,
                key=lambda item: 0.85 * _box_iou(previous, item[0]) + 0.15 * item[1],
            )[0]
        selected.append(chosen)
        previous = chosen
    return selected, multi_face_frames


def _max_missing_run(boxes: Sequence[np.ndarray | None]) -> int:
    best = current = 0
    for box in boxes:
        current = current + 1 if box is None else 0
        best = max(best, current)
    return best


def _interpolate_boxes(boxes: Sequence[np.ndarray | None]) -> tuple[list[np.ndarray], int]:
    valid = [index for index, box in enumerate(boxes) if box is not None]
    if not valid:
        raise RuntimeError("No face was detected in the sampled sequence")
    filled: list[np.ndarray] = []
    missing = 0
    for index, box in enumerate(boxes):
        if box is not None:
            filled.append(np.asarray(box, dtype=np.float32))
            continue
        missing += 1
        left = max((item for item in valid if item < index), default=None)
        right = min((item for item in valid if item > index), default=None)
        if left is None:
            value = boxes[right]
        elif right is None:
            value = boxes[left]
        else:
            alpha = (index - left) / float(right - left)
            value = (1.0 - alpha) * boxes[left] + alpha * boxes[right]
        filled.append(np.asarray(value, dtype=np.float32))
    return filled, missing


def _square_crop(frame: np.ndarray, box: np.ndarray, margin: float, min_side: int) -> np.ndarray:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (float(value) for value in box)
    side = int(math.ceil(max(x2 - x1, y2 - y1) * (1.0 + margin)))
    if side < min_side:
        raise RuntimeError(f"Face crop is too small: {side}px")
    center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    left, top = int(round(center_x - side / 2)), int(round(center_y - side / 2))
    right, bottom = left + side, top + side
    pad_left, pad_top = max(0, -left), max(0, -top)
    pad_right, pad_bottom = max(0, right - width), max(0, bottom - height)
    if pad_left or pad_top or pad_right or pad_bottom:
        frame = cv2.copyMakeBorder(
            frame, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101
        )
    crop = frame[top + pad_top : bottom + pad_top, left + pad_left : right + pad_left]
    if crop.shape[:2] != (side, side):
        raise RuntimeError(f"Invalid square crop shape: {crop.shape}")
    return crop


class MainFaceExtractor:
    def __init__(self, config: ExtractionConfig, device: str = "cpu") -> None:
        from facenet_pytorch import MTCNN

        self.config = config
        self.detector = MTCNN(
            min_face_size=config.min_face_size,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.7,
            post_process=False,
            select_largest=False,
            keep_all=True,
            device=device,
        )

    def extract(self, spec: VideoSpec, output_root: str | Path) -> VideoRecord:
        output_root = Path(output_root)
        capture = cv2.VideoCapture(str(spec.path))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if capture.isOpened() else 0
        source_fps = float(capture.get(cv2.CAP_PROP_FPS)) if capture.isOpened() else 0.0
        if not np.isfinite(source_fps) or source_fps <= 0:
            source_fps = 0.0
        capture.release()
        if total_frames <= 0:
            raise RuntimeError(f"Invalid video frame count: {spec.path}")

        indices, sampling_protocol = temporal_sample_indices(
            total_frames, source_fps, self.config.frames_per_video, self.config.target_fps
        )
        frames = _read_target_frames(spec.path, indices)
        all_boxes: list[np.ndarray | None] = []
        all_probs: list[np.ndarray | None] = []
        for start in range(0, len(frames), self.config.mtcnn_batch_size):
            boxes, probabilities = self.detector.detect(frames[start : start + self.config.mtcnn_batch_size])
            all_boxes.extend(list(boxes))
            all_probs.extend(list(probabilities))
        tracked, multi_face_frames = _track_main_face(
            all_boxes, all_probs, self.config.min_detection_probability
        )
        direct_count = sum(box is not None for box in tracked)
        missing_run = _max_missing_run(tracked)
        if direct_count / len(tracked) < self.config.min_direct_detection_ratio:
            raise RuntimeError(f"Only {direct_count}/{len(tracked)} frames have direct detections")
        if missing_run > self.config.max_consecutive_missing:
            raise RuntimeError(f"Missing face run {missing_run} exceeds configured maximum")
        boxes, interpolated_count = _interpolate_boxes(tracked)

        class_name = "real" if spec.label == 0 else "fake"
        video_dir = output_root / "frames" / spec.dataset / spec.split / class_name / spec.method / spec.video_id
        temporary = output_root / "_temporary" / spec.dataset / spec.split / spec.method / spec.video_id
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True, exist_ok=True)
        frame_paths: list[str] = []
        crop_sides: list[int] = []
        blur_scores: list[float] = []
        try:
            for order, (frame, box) in enumerate(zip(frames, boxes, strict=True)):
                crop = _square_crop(
                    frame, box, self.config.square_margin, self.config.min_crop_side
                )
                crop_sides.append(int(crop.shape[0]))
                if crop.shape[0] != self.config.output_size:
                    interpolation = (
                        cv2.INTER_AREA
                        if crop.shape[0] > self.config.output_size
                        else cv2.INTER_CUBIC
                    )
                    crop = cv2.resize(
                        crop,
                        (self.config.output_size, self.config.output_size),
                        interpolation=interpolation,
                    )
                gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
                blur_scores.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
                path = temporary / f"{order:06d}.jpg"
                ok = cv2.imwrite(
                    str(path),
                    cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
                    [int(cv2.IMWRITE_JPEG_QUALITY), self.config.jpeg_quality],
                )
                if not ok:
                    raise RuntimeError(f"Could not write extracted frame: {path}")
            if video_dir.exists():
                shutil.rmtree(video_dir)
            video_dir.parent.mkdir(parents=True, exist_ok=True)
            temporary.replace(video_dir)
            frame_paths = [
                str(path.relative_to(output_root)) for path in sorted(video_dir.glob("*.jpg"))
            ]
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

        timestamps = [
            index / source_fps if source_fps > 0 else order / self.config.target_fps
            for order, index in enumerate(indices)
        ]
        return VideoRecord(
            dataset=spec.dataset,
            split=spec.split,
            video_id=spec.video_id,
            label=spec.label,
            method=spec.method,
            source_video=str(spec.path),
            frames=frame_paths,
            source_indices=indices,
            timestamps_sec=timestamps,
            fps=source_fps,
            quality={
                "sampling_protocol": sampling_protocol,
                "extraction_config_sha256": _config_fingerprint(self.config),
                "reported_total_frames": total_frames,
                "direct_detections": direct_count,
                "interpolated_boxes": interpolated_count,
                "max_missing_run": missing_run,
                "multi_face_frames": multi_face_frames,
                "crop_side_min": min(crop_sides),
                "crop_side_max": max(crop_sides),
                "saved_frame_size": self.config.output_size,
                "blur_median": float(np.median(blur_scores)),
            },
        )


def extract_specs(
    specs: Iterable[VideoSpec],
    output_root: str | Path,
    manifest_path: str | Path,
    config: ExtractionConfig,
    device: str = "cpu",
    fail_fast: bool = True,
    resume: bool = True,
    checkpoint_every: int = 25,
) -> tuple[list[VideoRecord], list[dict[str, str]]]:
    from tqdm.auto import tqdm

    output_root, manifest_path = Path(output_root), Path(manifest_path)
    config.validate()
    specs = list(specs)
    spec_by_key: dict[tuple[str, str, str, str], VideoSpec] = {}
    for spec in specs:
        key = (spec.dataset, spec.split, spec.method, spec.video_id)
        if key in spec_by_key:
            raise ValueError(f"Duplicate extraction spec: {key}")
        spec_by_key[key] = spec
    config_fingerprint = _config_fingerprint(config)
    records: list[VideoRecord] = []
    if resume and manifest_path.is_file():
        for record in load_manifest(manifest_path):
            key = (record.dataset, record.split, record.method, record.video_id)
            spec = spec_by_key.get(key)
            if (
                spec is not None
                and record.label == spec.label
                and Path(record.source_video) == spec.path
                and record.quality.get("extraction_config_sha256") == config_fingerprint
                and len(record.frames) == config.frames_per_video
                and all((output_root / frame).is_file() for frame in record.frames)
            ):
                records.append(record)
    completed = {(record.dataset, record.split, record.method, record.video_id) for record in records}
    pending = [
        spec
        for spec in specs
        if (spec.dataset, spec.split, spec.method, spec.video_id) not in completed
    ]
    extractor = MainFaceExtractor(config, device=device) if pending else None
    errors: list[dict[str, str]] = []
    for index, spec in enumerate(tqdm(pending, desc="Extracting main-face sequences"), 1):
        try:
            assert extractor is not None
            records.append(extractor.extract(spec, output_root))
        except Exception as error:
            errors.append(
                {
                    "dataset": spec.dataset,
                    "split": spec.split,
                    "method": spec.method,
                    "class": "real" if spec.label == 0 else "fake",
                    "video_id": spec.video_id,
                    "path": str(spec.path),
                    "error": str(error),
                }
            )
            if fail_fast:
                break
        if checkpoint_every > 0 and index % checkpoint_every == 0 and records:
            records.sort(key=lambda row: (row.split, row.label, row.method, row.video_id))
            write_manifest(records, manifest_path)
    if records:
        records.sort(key=lambda row: (row.split, row.label, row.method, row.video_id))
        write_manifest(records, manifest_path)
    report_path = manifest_path.with_suffix(".report.json")
    report = {
        "requested_videos": len(specs),
        "requested_by_label": {
            "0": sum(spec.label == 0 for spec in specs),
            "1": sum(spec.label == 1 for spec in specs),
        },
        "summary": manifest_summary(records),
        "errors": errors,
        "errors_by_class": {
            class_name: sum(error["class"] == class_name for error in errors)
            for class_name in ("real", "fake")
        },
        "config": config.__dict__,
        "extraction_config_sha256": config_fingerprint,
    }
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    temporary.replace(report_path)
    if not records:
        raise RuntimeError(f"No video was extracted; inspect {report_path}")
    if errors and fail_fast:
        first = errors[0]
        raise RuntimeError(
            f"Extraction failed for {first['method']}/{first['video_id']}: {first['error']}"
        )
    return records, errors

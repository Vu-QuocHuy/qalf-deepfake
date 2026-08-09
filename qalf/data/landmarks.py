"""MediaPipe Face Mesh cache generation for already-extracted face sequences."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .manifest import VideoRecord, load_manifest, write_manifest


def _landmark_fingerprint(
    record: VideoRecord,
    static_image_mode: bool,
    min_confidence: float,
    min_detected_ratio: float,
) -> str:
    payload = {
        "frames": record.frames,
        "source_indices": record.source_indices,
        "static_image_mode": static_image_mode,
        "min_confidence": min_confidence,
        "min_detected_ratio": min_detected_ratio,
        "implementation": "mediapipe_face_mesh_468_v1",
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class FaceMeshExtractor:
    def __init__(self, static_image_mode: bool = True, min_confidence: float = 0.5) -> None:
        import mediapipe as mp

        solutions = getattr(mp, "solutions", None)
        if solutions is None or not hasattr(solutions, "face_mesh"):
            raise RuntimeError(
                "This extractor requires mediapipe.solutions.face_mesh. "
                "Install a MediaPipe 0.10.x build that exposes the solutions API."
            )
        self.mesh = solutions.face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )

    def close(self) -> None:
        self.mesh.close()

    def process(self, image_rgb: np.ndarray) -> np.ndarray | None:
        result = self.mesh.process(image_rgb)
        if not result.multi_face_landmarks:
            return None
        points = result.multi_face_landmarks[0].landmark
        return np.asarray([(point.x, point.y, point.z) for point in points], dtype=np.float32)


def extract_landmark_cache(
    manifest_path: str | Path,
    frame_root: str | Path,
    landmark_root: str | Path,
    output_manifest: str | Path,
    static_image_mode: bool = True,
    min_confidence: float = 0.5,
    min_detected_ratio: float = 0.75,
    fail_fast: bool = True,
    resume: bool = True,
    checkpoint_every: int = 25,
) -> dict[str, object]:
    from tqdm.auto import tqdm

    frame_root, landmark_root = Path(frame_root), Path(landmark_root)
    records = load_manifest(manifest_path)
    errors: list[dict[str, str]] = []
    enriched: list[VideoRecord] = []
    output_manifest = Path(output_manifest)
    input_by_key = {
        (record.dataset, record.split, record.method, record.video_id): record
        for record in records
    }
    if len(input_by_key) != len(records):
        raise ValueError("Input manifest contains duplicate dataset/split/method/video records")
    if resume and output_manifest.is_file():
        for record in load_manifest(output_manifest):
            key = (record.dataset, record.split, record.method, record.video_id)
            source_record = input_by_key.get(key)
            if (
                source_record is not None
                and record.landmark_path
                and (landmark_root / record.landmark_path).is_file()
                and record.quality.get("landmark_config_sha256")
                == _landmark_fingerprint(
                    source_record,
                    static_image_mode,
                    min_confidence,
                    min_detected_ratio,
                )
            ):
                enriched.append(record)
    completed = {
        (record.dataset, record.split, record.method, record.video_id) for record in enriched
    }
    pending = [
        record
        for record in records
        if (record.dataset, record.split, record.method, record.video_id) not in completed
    ]
    extractor = FaceMeshExtractor(static_image_mode, min_confidence) if pending else None
    try:
        for index, record in enumerate(tqdm(pending, desc="Extracting Face Mesh landmarks"), 1):
            try:
                all_points: list[np.ndarray] = []
                detected: list[bool] = []
                image_sizes: list[tuple[int, int]] = []
                point_count = 468
                for relative_frame in record.frames:
                    image = cv2.imread(str(frame_root / relative_frame))
                    if image is None:
                        raise FileNotFoundError(f"Could not read frame: {frame_root / relative_frame}")
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    points = extractor.process(image_rgb)
                    image_sizes.append((image.shape[1], image.shape[0]))
                    if points is None:
                        all_points.append(np.full((point_count, 3), np.nan, dtype=np.float32))
                        detected.append(False)
                    else:
                        point_count = int(points.shape[0])
                        all_points.append(points)
                        detected.append(True)
                detected_ratio = sum(detected) / max(len(detected), 1)
                if detected_ratio < min_detected_ratio:
                    raise RuntimeError(
                        f"Face Mesh detected only {sum(detected)}/{len(detected)} frames; "
                        f"required ratio={min_detected_ratio:.2f}"
                    )
                cache_relative = (
                    Path(record.dataset)
                    / record.split
                    / ("real" if record.label == 0 else "fake")
                    / record.method
                    / f"{record.video_id}.npz"
                )
                cache_path = landmark_root / cache_relative
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = cache_path.with_suffix(".tmp.npz")
                np.savez_compressed(
                    temporary,
                    landmarks=np.stack(all_points),
                    detected=np.asarray(detected, dtype=np.bool_),
                    image_sizes=np.asarray(image_sizes, dtype=np.int32),
                    source_indices=np.asarray(record.source_indices, dtype=np.int64),
                    timestamps_sec=np.asarray(record.timestamps_sec, dtype=np.float64),
                )
                temporary.replace(cache_path)
                record.landmark_path = str(cache_relative)
                record.quality = {
                    **record.quality,
                    "landmark_detected": int(sum(detected)),
                    "landmark_total": len(detected),
                    "landmark_detected_ratio": detected_ratio,
                    "landmark_config_sha256": _landmark_fingerprint(
                        record,
                        static_image_mode,
                        min_confidence,
                        min_detected_ratio,
                    ),
                }
                enriched.append(record)
            except Exception as error:
                errors.append(
                    {
                        "method": record.method,
                        "class": "real" if record.label == 0 else "fake",
                        "video_id": record.video_id,
                        "error": str(error),
                    }
                )
                if fail_fast:
                    break
            if checkpoint_every > 0 and index % checkpoint_every == 0 and enriched:
                enriched.sort(key=lambda row: (row.split, row.label, row.method, row.video_id))
                write_manifest(enriched, output_manifest)
    finally:
        if extractor is not None:
            extractor.close()
    if enriched:
        enriched.sort(key=lambda row: (row.split, row.label, row.method, row.video_id))
        write_manifest(enriched, output_manifest)
    report = {
        "input_videos": len(records),
        "output_videos": len(enriched),
        "errors": errors,
        "errors_by_class": {
            class_name: sum(error["class"] == class_name for error in errors)
            for class_name in ("real", "fake")
        },
    }
    report_path = output_manifest.with_suffix(".landmarks.json")
    temporary_report = report_path.with_suffix(report_path.suffix + ".tmp")
    with temporary_report.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    temporary_report.replace(report_path)
    if errors and fail_fast:
        raise RuntimeError(
            f"Landmark extraction failed for {errors[0]['video_id']}: {errors[0]['error']}"
        )
    return report

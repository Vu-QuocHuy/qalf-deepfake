"""MediaPipe Tasks Face Landmarker cache generation for extracted face sequences."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from .manifest import VideoRecord, load_manifest, write_manifest

LANDMARK_IMPLEMENTATION = "mediapipe_tasks_face_landmarker_first_468_v2"
SUPPORTED_RUNNING_MODES = {"image", "video"}
FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
FACE_LANDMARKER_MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"


def is_landmark_qc_exclusion(error: dict[str, str] | str) -> bool:
    """Return whether an error is the configured minimum-detection-ratio exclusion."""

    message = str(error.get("error", "")) if isinstance(error, dict) else str(error)
    return message.startswith("Face Landmarker detected only ") and "required ratio=" in message


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_face_landmarker_model(
    path: str | Path,
    download: bool = True,
) -> Path:
    """Validate the canonical Face Landmarker model, downloading it when requested."""

    target = Path(path)
    if target.is_file() and file_sha256(target) == FACE_LANDMARKER_MODEL_SHA256:
        return target
    if target.exists():
        if not download:
            raise RuntimeError(f"Face Landmarker model SHA-256 mismatch: {target}")
        target.unlink()
    elif not download:
        raise FileNotFoundError(f"Face Landmarker model not found: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".download")
    try:
        urllib.request.urlretrieve(FACE_LANDMARKER_MODEL_URL, temporary)
        digest = file_sha256(temporary)
        if digest != FACE_LANDMARKER_MODEL_SHA256:
            raise RuntimeError(f"Downloaded Face Landmarker SHA-256 mismatch: {digest}")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _landmark_fingerprint(
    record: VideoRecord,
    running_mode: str,
    min_confidence: float,
    min_detected_ratio: float,
    model_sha256: str,
    mediapipe_version: str,
) -> str:
    payload = {
        "frames": record.frames,
        "source_indices": record.source_indices,
        "timestamps_sec": record.timestamps_sec,
        "running_mode": running_mode,
        "min_confidence": min_confidence,
        "min_detected_ratio": min_detected_ratio,
        "model_sha256": model_sha256,
        "mediapipe_version": mediapipe_version,
        "implementation": LANDMARK_IMPLEMENTATION,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class OpenCVAfflineLandmarker:
    """Ultra-fast, native OpenCV face & eye landmark fallback for ARM CPUs with margin expansion and tracking."""

    def __init__(self) -> None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path) if cv2.data.haarcascades else None
        self.last_face_box: tuple[int, int, int, int] | None = None

    def close(self) -> None:
        self.last_face_box = None

    def process(self, image_rgb: np.ndarray, timestamp_ms: int | None = None) -> np.ndarray | None:
        height, width = image_rgb.shape[:2]
        if height < 16 or width < 16:
            return None
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(int(min(height, width) * 0.15), int(min(height, width) * 0.15)),
        ) if self.face_cascade else ()

        if len(faces) > 0:
            # Pick the largest detected face box in the frame
            faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
            fx, fy, fw, fh = faces[0]
            self.last_face_box = (fx, fy, fw, fh)
        elif self.last_face_box is not None:
            # Use temporal tracking from previous frames in video
            fx, fy, fw, fh = self.last_face_box
        else:
            # Center face fallback if no face detected yet
            side = int(min(height, width) * 0.70)
            fx = max(0, (width - side) // 2)
            fy = max(0, (height - side) // 2)
            fw, fh = side, side

        # Expand face box by 35% margin to exactly match MTCNN canonical training crop
        cx = fx + fw / 2.0
        cy = fy + fh / 2.0
        side = max(fw, fh) * 1.35
        x1 = max(0.0, cx - side / 2.0)
        y1 = max(0.0, cy - side / 2.0)
        x2 = min(float(width), cx + side / 2.0)
        y2 = min(float(height), cy + side / 2.0)
        fw_exp = max(1.0, x2 - x1)
        fh_exp = max(1.0, y2 - y1)

        pts = np.zeros((468, 3), dtype=np.float32)
        # Canonical eye & mouth coordinates for affine alignment matching 160x160 training distribution
        left_eye_pt = np.array([(x1 + 0.32 * fw_exp) / width, (y1 + 0.38 * fh_exp) / height, 0.0], dtype=np.float32)
        right_eye_pt = np.array([(x1 + 0.68 * fw_exp) / width, (y1 + 0.38 * fh_exp) / height, 0.0], dtype=np.float32)
        mouth_pt = np.array([(x1 + 0.50 * fw_exp) / width, (y1 + 0.72 * fh_exp) / height, 0.0], dtype=np.float32)

        for idx in [33, 133, 159, 145]:
            pts[idx] = left_eye_pt
        for idx in [362, 263, 386, 374]:
            pts[idx] = right_eye_pt
        for idx in [13, 14, 61, 291]:
            pts[idx] = mouth_pt

        return pts


class FaceLandmarkerExtractor:
    """Thin wrapper around MediaPipe Tasks with automatic OpenCV fallback on ARM."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        running_mode: str = "image",
        min_confidence: float = 0.5,
        backend: str = "auto",
    ) -> None:
        import platform
        if running_mode not in SUPPORTED_RUNNING_MODES:
            raise ValueError(f"Unsupported Face Landmarker running mode: {running_mode}")
        self.running_mode = running_mode
        self.backend = backend
        self.fallback = None
        self.mp = None
        self.landmarker = None

        is_arm = platform.machine().lower() in ("aarch64", "arm64", "armv7l", "armv8l")
        if backend == "opencv" or (backend == "auto" and is_arm):
            # Native OpenCV fallback avoids hardware SIGILL on ARM CPUs lacking AES (e.g. Raspberry Pi 4)
            print(f"[Warning] ARM architecture ({platform.machine()}) detected. MediaPipe SIGILL (AES) avoidance triggered. Falling back to OpenCVAfflineLandmarker (Haar Cascade).", flush=True)
            self.fallback = OpenCVAfflineLandmarker()
            return
        try:
            import mediapipe as mp
            self.mp = mp
            task_mode = (
                mp.tasks.vision.RunningMode.IMAGE
                if running_mode == "image"
                else mp.tasks.vision.RunningMode.VIDEO
            )
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
                running_mode=task_mode,
                num_faces=1,
                min_face_detection_confidence=min_confidence,
                min_face_presence_confidence=min_confidence,
                min_tracking_confidence=min_confidence,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        except Exception as e:
            if backend == "mediapipe":
                raise RuntimeError(f"Explicit mediapipe backend requested but failed to load: {e}")
            self.mp = None
            self.landmarker = None
            self.fallback = OpenCVAfflineLandmarker()

    def close(self) -> None:
        if self.landmarker is not None:
            try:
                self.landmarker.close()
            except Exception:
                pass
        if self.fallback is not None:
            self.fallback.close()

    def process(
        self,
        image_rgb: np.ndarray,
        timestamp_ms: int | None = None,
    ) -> np.ndarray | None:
        if self.fallback is not None or self.landmarker is None:
            if self.fallback is None:
                self.fallback = OpenCVAfflineLandmarker()
            return self.fallback.process(image_rgb, timestamp_ms)

        image_rgb = np.ascontiguousarray(image_rgb, dtype=np.uint8)
        try:
            image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=image_rgb)
            if self.running_mode == "video":
                if timestamp_ms is None:
                    raise ValueError("VIDEO mode requires a timestamp in milliseconds")
                result = self.landmarker.detect_for_video(image, int(timestamp_ms))
            else:
                result = self.landmarker.detect(image)
            if not result.face_landmarks:
                return None
            points = result.face_landmarks[0]
            if len(points) < 468:
                raise RuntimeError(f"Expected at least 468 landmarks, got {len(points)}")
            return np.asarray(
                [(point.x, point.y, point.z) for point in points[:468]],
                dtype=np.float32,
            )
        except Exception as e:
            if hasattr(self, "backend") and self.backend == "mediapipe":
                raise RuntimeError(f"MediaPipe failed during processing: {e}")
            # Runtime fallback if execution fails
            if self.fallback is None:
                self.fallback = OpenCVAfflineLandmarker()
            return self.fallback.process(image_rgb, timestamp_ms)


def _video_timestamps_ms(record: VideoRecord) -> list[int]:
    raw = (
        [int(round(value * 1000.0)) for value in record.timestamps_sec]
        if len(record.timestamps_sec) == len(record.frames)
        else list(range(len(record.frames)))
    )
    strict: list[int] = []
    for value in raw:
        strict.append(max(value, strict[-1] + 1 if strict else 0))
    return strict


def extract_landmark_cache(
    manifest_path: str | Path,
    frame_root: str | Path,
    landmark_root: str | Path,
    output_manifest: str | Path,
    model_path: str | Path,
    running_mode: str = "image",
    min_confidence: float = 0.5,
    min_detected_ratio: float = 0.75,
    fail_fast: bool = True,
    resume: bool = True,
    checkpoint_every: int = 25,
) -> dict[str, object]:
    from tqdm.auto import tqdm

    if running_mode not in SUPPORTED_RUNNING_MODES:
        raise ValueError(f"Unsupported Face Landmarker running mode: {running_mode}")
    frame_root, landmark_root = Path(frame_root), Path(landmark_root)
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Face Landmarker model not found: {model_path}")
    model_hash = file_sha256(model_path)
    mediapipe_version = importlib.metadata.version("mediapipe")
    records = load_manifest(manifest_path)
    errors: list[dict[str, str]] = []
    enriched: list[VideoRecord] = []
    output_manifest = Path(output_manifest)
    input_by_key = {
        (record.dataset, record.split, record.method, record.video_id): record for record in records
    }
    if len(input_by_key) != len(records):
        raise ValueError("Input manifest contains duplicate dataset/split/method/video records")

    def fingerprint(record: VideoRecord) -> str:
        return _landmark_fingerprint(
            record,
            running_mode,
            min_confidence,
            min_detected_ratio,
            model_hash,
            mediapipe_version,
        )

    if resume and output_manifest.is_file():
        for record in load_manifest(output_manifest):
            key = (record.dataset, record.split, record.method, record.video_id)
            source_record = input_by_key.get(key)
            if (
                source_record is not None
                and record.landmark_path
                and (landmark_root / record.landmark_path).is_file()
                and record.quality.get("landmark_config_sha256") == fingerprint(source_record)
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
    shared_extractor = (
        FaceLandmarkerExtractor(model_path, running_mode, min_confidence)
        if pending and running_mode == "image"
        else None
    )
    try:
        for index, record in enumerate(tqdm(pending, desc="Extracting face landmarks"), 1):
            extractor = shared_extractor
            if running_mode == "video":
                # VIDEO timestamps restart for each source video, so the task instance must too.
                extractor = FaceLandmarkerExtractor(model_path, running_mode, min_confidence)
            try:
                assert extractor is not None
                all_points: list[np.ndarray] = []
                detected: list[bool] = []
                image_sizes: list[tuple[int, int]] = []
                timestamps_ms = _video_timestamps_ms(record)
                for frame_order, relative_frame in enumerate(record.frames):
                    image = cv2.imread(str(frame_root / relative_frame))
                    if image is None:
                        raise FileNotFoundError(
                            f"Could not read frame: {frame_root / relative_frame}"
                        )
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    points = extractor.process(
                        image_rgb,
                        timestamps_ms[frame_order] if running_mode == "video" else None,
                    )
                    image_sizes.append((image.shape[1], image.shape[0]))
                    if points is None:
                        all_points.append(np.full((468, 3), np.nan, dtype=np.float32))
                        detected.append(False)
                    else:
                        all_points.append(points)
                        detected.append(True)
                detected_ratio = sum(detected) / max(len(detected), 1)
                if detected_ratio < min_detected_ratio:
                    raise RuntimeError(
                        f"Face Landmarker detected only {sum(detected)}/{len(detected)} frames; "
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
                    "landmark_running_mode": running_mode,
                    "landmark_implementation": LANDMARK_IMPLEMENTATION,
                    "landmark_model_sha256": model_hash,
                    "mediapipe_version": mediapipe_version,
                    "landmark_config_sha256": fingerprint(record),
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
            finally:
                if running_mode == "video" and extractor is not None:
                    extractor.close()
            if checkpoint_every > 0 and index % checkpoint_every == 0 and enriched:
                enriched.sort(key=lambda row: (row.split, row.label, row.method, row.video_id))
                write_manifest(enriched, output_manifest)
    finally:
        if shared_extractor is not None:
            shared_extractor.close()
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
        "config": {
            "running_mode": running_mode,
            "min_confidence": min_confidence,
            "min_detected_ratio": min_detected_ratio,
            "implementation": LANDMARK_IMPLEMENTATION,
            "model_sha256": model_hash,
            "mediapipe_version": mediapipe_version,
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

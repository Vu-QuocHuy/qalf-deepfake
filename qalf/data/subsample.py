"""Derive shorter frame/landmark sequences from an extracted dataset."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .manifest import VideoRecord, load_manifest, manifest_summary, write_manifest


def _subset_optional(values: list, positions: np.ndarray) -> list:
    if not values:
        return []
    return [values[int(position)] for position in positions]


def _derived_quality(
    record: VideoRecord,
    detected: np.ndarray,
    positions: np.ndarray,
) -> dict[str, object]:
    quality = dict(record.quality)
    # These fingerprints and sequence-level statistics describe the 64-frame
    # parent and must not masquerade as measurements of the derived sequence.
    for key in (
        "extraction_config_sha256",
        "landmark_config_sha256",
        "direct_detections",
        "interpolated_boxes",
        "max_missing_run",
        "multi_face_frames",
        "blur_median",
    ):
        quality.pop(key, None)
    selected_timestamps = _subset_optional(record.timestamps_sec, positions)
    effective_fps = None
    if len(selected_timestamps) > 1:
        differences = np.diff(np.asarray(selected_timestamps, dtype=np.float64))
        if np.all(differences > 0):
            effective_fps = float(1.0 / np.median(differences))
    detected_count = int(np.count_nonzero(detected))
    quality.update(
        {
            "sampling_protocol": "derived_fixed_stride",
            "derived_from_frame_count": len(record.frames),
            "derived_frame_count": len(positions),
            "derived_stride": int(positions[1] - positions[0]) if len(positions) > 1 else 1,
            "derived_offset": int(positions[0]),
            "derived_effective_fps": effective_fps,
            "landmark_detected": detected_count,
            "landmark_total": len(positions),
            "landmark_detected_ratio": detected_count / len(positions),
        }
    )
    return quality


def derive_strided_landmark_dataset(
    input_manifest: str | Path,
    input_landmark_root: str | Path,
    output_manifest: str | Path,
    output_landmark_root: str | Path,
    *,
    source_frames: int = 64,
    stride: int = 2,
    offset: int = 0,
) -> dict[str, object]:
    """Write a manifest/cache view that references a fixed stride of source frames.

    JPEG files are not copied. The derived manifest continues to reference the
    source frame root, while landmark arrays are subset into a separate root.
    """

    if source_frames < 2:
        raise ValueError("source_frames must be at least two")
    if stride < 1:
        raise ValueError("stride must be positive")
    if not 0 <= offset < stride:
        raise ValueError("offset must be in [0, stride)")
    positions = np.arange(offset, source_frames, stride, dtype=np.int64)
    if len(positions) < 2:
        raise ValueError("subsampling must retain at least two frames")

    input_manifest = Path(input_manifest)
    input_landmark_root = Path(input_landmark_root)
    output_manifest = Path(output_manifest)
    output_landmark_root = Path(output_landmark_root)
    if input_manifest.resolve() == output_manifest.resolve():
        raise ValueError("output_manifest must differ from input_manifest")
    if input_landmark_root.resolve() == output_landmark_root.resolve():
        raise ValueError("output_landmark_root must differ from input_landmark_root")

    derived: list[VideoRecord] = []
    for record in load_manifest(input_manifest):
        if len(record.frames) != source_frames:
            raise ValueError(
                f"{record.video_id}: expected {source_frames} source frames, "
                f"found {len(record.frames)}"
            )
        if not record.landmark_path:
            raise ValueError(f"{record.video_id}: source manifest has no landmark_path")
        source_cache = input_landmark_root / record.landmark_path
        if not source_cache.is_file():
            raise FileNotFoundError(source_cache)
        with np.load(source_cache) as cache:
            arrays = {key: cache[key].copy() for key in cache.files}
        for required in ("landmarks", "detected"):
            if required not in arrays:
                raise ValueError(f"{record.video_id}: landmark cache is missing {required}")
            if arrays[required].shape[0] != source_frames:
                raise ValueError(
                    f"{record.video_id}: {required} has {arrays[required].shape[0]} rows, "
                    f"expected {source_frames}"
                )
        subset_arrays = {
            key: value[positions] if value.ndim > 0 and value.shape[0] == source_frames else value
            for key, value in arrays.items()
        }
        cache_path = output_landmark_root / record.landmark_path
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(cache_path.suffix + ".tmp.npz")
        np.savez_compressed(temporary, **subset_arrays)
        temporary.replace(cache_path)

        derived.append(
            replace(
                record,
                frames=_subset_optional(record.frames, positions),
                source_indices=_subset_optional(record.source_indices, positions),
                timestamps_sec=_subset_optional(record.timestamps_sec, positions),
                quality=_derived_quality(record, subset_arrays["detected"], positions),
            )
        )

    write_manifest(derived, output_manifest)
    report = {
        "input_manifest": str(input_manifest),
        "output_manifest": str(output_manifest),
        "input_landmark_root": str(input_landmark_root),
        "output_landmark_root": str(output_landmark_root),
        "source_frames": source_frames,
        "derived_frames": len(positions),
        "stride": stride,
        "offset": offset,
        "summary": manifest_summary(derived),
    }
    report_path = output_manifest.with_suffix(".subsample.json")
    temporary_report = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary_report.parent.mkdir(parents=True, exist_ok=True)
    temporary_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary_report.replace(report_path)
    return report

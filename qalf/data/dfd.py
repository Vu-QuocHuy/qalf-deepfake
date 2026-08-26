"""Prepare a portable DFD evaluation manifest from class-specific exports."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

from qalf.data.manifest import VideoRecord, load_manifest, manifest_summary, write_manifest


def _relative_to_dataset_root(path: Path, dataset_root: Path) -> str:
    try:
        relative = path.resolve().relative_to(dataset_root.resolve())
    except ValueError as error:
        raise ValueError(f"DFD input must be inside dataset root {dataset_root}: {path}") from error
    return relative.as_posix()


def _resolve_frame_path(
    record: VideoRecord,
    original_path: str,
    frame_directory: Path,
) -> Path:
    """Resolve both the flattened Windows export and the original Kaggle layout."""

    relative = Path(original_path)
    candidates = (
        frame_directory / record.video_id / relative.name,
        frame_directory.parent / relative,
        frame_directory / relative,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"{record.video_id}: frame {relative.name} was not found; checked "
        + ", ".join(str(path) for path in candidates)
    )


def _resolve_landmark_path(record: VideoRecord, landmark_directory: Path) -> Path:
    if not record.landmark_path:
        raise ValueError(f"{record.video_id}: landmark_path is missing")
    relative = Path(record.landmark_path)
    candidates = (
        landmark_directory / relative.name,
        landmark_directory / relative,
        landmark_directory.parent / relative,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"{record.video_id}: landmark cache was not found; checked "
        + ", ".join(str(path) for path in candidates)
    )


def _prepare_class_records(
    manifest_path: str | Path,
    frame_directory: str | Path,
    landmark_directory: str | Path,
    dataset_root: Path,
    expected_label: int,
) -> list[VideoRecord]:
    records = load_manifest(manifest_path)
    frame_directory = Path(frame_directory)
    landmark_directory = Path(landmark_directory)
    class_name = "real" if expected_label == 0 else "fake"

    prepared: list[VideoRecord] = []
    for record in records:
        if record.dataset != "dfd":
            raise ValueError(
                f"{record.video_id}: expected dataset='dfd', got {record.dataset!r}"
            )
        if record.label != expected_label:
            raise ValueError(
                f"{record.video_id}: expected {class_name} label {expected_label}, "
                f"got {record.label}"
            )
        resolved_frames = [
            _resolve_frame_path(record, frame, frame_directory) for frame in record.frames
        ]
        resolved_landmark = _resolve_landmark_path(record, landmark_directory)
        prepared.append(
            replace(
                record,
                frames=[
                    _relative_to_dataset_root(frame, dataset_root) for frame in resolved_frames
                ],
                landmark_path=_relative_to_dataset_root(resolved_landmark, dataset_root),
            )
        )
    return prepared


def prepare_dfd_manifest(
    *,
    real_manifest: str | Path,
    fake_manifest: str | Path,
    real_frame_directory: str | Path,
    fake_frame_directory: str | Path,
    real_landmark_directory: str | Path,
    fake_landmark_directory: str | Path,
    dataset_root: str | Path,
    output_manifest: str | Path,
) -> dict[str, Any]:
    """Merge DFD real/fake manifests and rewrite paths under one dataset root."""

    dataset_root = Path(dataset_root)
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"DFD dataset root does not exist: {dataset_root}")

    real_records = _prepare_class_records(
        real_manifest,
        real_frame_directory,
        real_landmark_directory,
        dataset_root,
        expected_label=0,
    )
    fake_records = _prepare_class_records(
        fake_manifest,
        fake_frame_directory,
        fake_landmark_directory,
        dataset_root,
        expected_label=1,
    )
    records = real_records + fake_records
    video_id_counts = Counter(record.video_id for record in records)
    duplicates = sorted(video_id for video_id, count in video_id_counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate DFD video_id values: {duplicates[:10]}")

    output_manifest = Path(output_manifest)
    write_manifest(records, output_manifest)
    summary = manifest_summary(records)
    report = {
        **summary,
        "dataset": "dfd",
        "splits": sorted({record.split for record in records}),
        "real_videos": len(real_records),
        "fake_videos": len(fake_records),
        "dataset_root": str(dataset_root.resolve()),
        "output_manifest": str(output_manifest.resolve()),
    }
    report_path = output_manifest.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report

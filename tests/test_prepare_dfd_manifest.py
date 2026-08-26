from __future__ import annotations

import json
from pathlib import Path

import pytest

from qalf.data.dfd import prepare_dfd_manifest
from qalf.data.manifest import VideoRecord, load_manifest, write_manifest


def _record(video_id: str, label: int) -> VideoRecord:
    class_name = "real" if label == 0 else "fake"
    method = "original" if label == 0 else "DeepFakeDetection"
    return VideoRecord(
        dataset="dfd",
        split="all",
        video_id=video_id,
        label=label,
        method=method,
        source_video=f"/kaggle/input/{video_id}.mp4",
        frames=[
            f"frames/dfd/all/{class_name}/{method}/{video_id}/000000.jpg",
            f"frames/dfd/all/{class_name}/{method}/{video_id}/000001.jpg",
        ],
        source_indices=[0, 1],
        timestamps_sec=[0.0, 0.1],
        fps=10.0,
        landmark_path=f"dfd/all/{class_name}/{method}/{video_id}.npz",
    )


def _write_flat_export(root: Path, class_name: str, record: VideoRecord) -> tuple[Path, Path]:
    frame_directory = root / class_name / "extracted" / "frames"
    video_directory = frame_directory / record.video_id
    video_directory.mkdir(parents=True)
    for frame in record.frames:
        (video_directory / Path(frame).name).write_bytes(b"jpeg")
    landmark_directory = root / class_name / "landmarks" / "landmarks"
    landmark_directory.mkdir(parents=True)
    (landmark_directory / f"{record.video_id}.npz").write_bytes(b"npz")
    return frame_directory, landmark_directory


def test_prepare_dfd_manifest_rebases_flat_exports(tmp_path: Path) -> None:
    real = _record("01__exit_phone_room", 0)
    fake = _record("01_02__exit_phone_room__YVGY8LOK", 1)
    real_frames, real_landmarks = _write_flat_export(tmp_path, "real", real)
    fake_frames, fake_landmarks = _write_flat_export(tmp_path, "fake", fake)
    real_manifest = tmp_path / "real.jsonl"
    fake_manifest = tmp_path / "fake.jsonl"
    output_manifest = tmp_path / "manifests" / "dfd_all_landmarks.jsonl"
    write_manifest([real], real_manifest)
    write_manifest([fake], fake_manifest)

    report = prepare_dfd_manifest(
        real_manifest=real_manifest,
        fake_manifest=fake_manifest,
        real_frame_directory=real_frames,
        fake_frame_directory=fake_frames,
        real_landmark_directory=real_landmarks,
        fake_landmark_directory=fake_landmarks,
        dataset_root=tmp_path,
        output_manifest=output_manifest,
    )

    records = load_manifest(output_manifest)
    assert [record.label for record in records] == [0, 1]
    assert records[0].frames[0] == (
        "real/extracted/frames/01__exit_phone_room/000000.jpg"
    )
    assert records[1].landmark_path == (
        "fake/landmarks/landmarks/01_02__exit_phone_room__YVGY8LOK.npz"
    )
    assert report["videos"] == 2
    assert report["real_videos"] == 1
    assert report["fake_videos"] == 1
    assert json.loads(output_manifest.with_suffix(".report.json").read_text())["frames"] == 4


def test_prepare_dfd_manifest_rejects_wrong_class_label(tmp_path: Path) -> None:
    wrong_real = _record("fake_in_real_manifest", 1)
    real_manifest = tmp_path / "real.jsonl"
    fake_manifest = tmp_path / "fake.jsonl"
    write_manifest([wrong_real], real_manifest)
    write_manifest([_record("fake", 1)], fake_manifest)

    with pytest.raises(ValueError, match="expected real label 0"):
        prepare_dfd_manifest(
            real_manifest=real_manifest,
            fake_manifest=fake_manifest,
            real_frame_directory=tmp_path,
            fake_frame_directory=tmp_path,
            real_landmark_directory=tmp_path,
            fake_landmark_directory=tmp_path,
            dataset_root=tmp_path,
            output_manifest=tmp_path / "output.jsonl",
        )

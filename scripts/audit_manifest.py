#!/usr/bin/env python3
"""Audit extracted-frame and landmark manifests before training."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qalf.data.manifest import load_manifest, manifest_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", nargs="+", required=True)
    parser.add_argument("--frame-root", required=True)
    parser.add_argument("--landmark-root")
    parser.add_argument("--expected-frames", type=int, default=64)
    args = parser.parse_args()

    frame_root = Path(args.frame_root)
    landmark_root = Path(args.landmark_root) if args.landmark_root else None
    split_ids: dict[str, set[str]] = {}
    report: dict[str, object] = {"manifests": {}, "overlap": {}}
    failures: list[str] = []
    for manifest in args.manifest:
        records = load_manifest(manifest)
        split = records[0].split
        split_ids[split] = {record.video_id for record in records}
        frame_histogram = Counter(len(record.frames) for record in records)
        record_keys: set[tuple[str, str, str, str]] = set()
        for record in records:
            key = (record.dataset, record.split, record.method, record.video_id)
            if key in record_keys:
                failures.append(f"duplicate manifest record: {key}")
            record_keys.add(key)
            if record.split != split:
                failures.append(
                    f"{record.video_id}: mixed split {record.split!r} in {split!r} manifest"
                )
            if len(record.frames) != args.expected_frames:
                failures.append(f"{record.video_id}: {len(record.frames)} frames")
            if len(set(record.frames)) != len(record.frames):
                failures.append(f"{record.video_id}: duplicate frame paths")
            if len(set(record.source_indices)) != len(record.source_indices):
                failures.append(f"{record.video_id}: duplicate source indices")
            if any(
                right <= left
                for left, right in zip(
                    record.source_indices, record.source_indices[1:], strict=False
                )
            ):
                failures.append(f"{record.video_id}: source indices are not strictly increasing")
            if any(
                right <= left
                for left, right in zip(
                    record.timestamps_sec, record.timestamps_sec[1:], strict=False
                )
            ):
                failures.append(f"{record.video_id}: timestamps are not strictly increasing")
            if "extraction_config_sha256" not in record.quality:
                failures.append(f"{record.video_id}: missing extraction configuration fingerprint")
            for frame in record.frames:
                if not (frame_root / frame).is_file():
                    failures.append(f"{record.video_id}: missing frame {frame}")
            if landmark_root:
                if not record.landmark_path:
                    failures.append(f"{record.video_id}: missing landmark_path")
                elif not (landmark_root / record.landmark_path).is_file():
                    failures.append(f"{record.video_id}: missing landmark cache")
                if "landmark_config_sha256" not in record.quality:
                    failures.append(f"{record.video_id}: missing landmark configuration fingerprint")
        report["manifests"][manifest] = {
            **manifest_summary(records),
            "frame_count_histogram": dict(frame_histogram),
        }
    names = sorted(split_ids)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = sorted(split_ids[left] & split_ids[right])
            report["overlap"][f"{left}/{right}"] = overlap[:20]
            if overlap:
                failures.append(f"video-id overlap {left}/{right}: {overlap[:8]}")
    report["failures"] = failures
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

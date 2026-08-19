#!/usr/bin/env python3
"""Package server-extracted Celeb-DF frames + landmarks for transfer to Raspberry Pi 4.

Usage with separate folders (fast uncompressed tar):
    python scripts/pack_server_data.py \
        --frame-root "E:/DeepFakeData/data/extracted/celebdf" \
        --landmark-root "E:/DeepFakeData/data/landmarks/celebdf-landmark" \
        --no-compress \
        --output celebdf_server_extracted.tar

Usage with gzip compression:
    python scripts/pack_server_data.py \
        --frame-root "E:/DeepFakeData/data/extracted/celebdf" \
        --landmark-root "E:/DeepFakeData/data/landmarks/celebdf-landmark" \
        --output celebdf_server_extracted.tar.gz
"""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
import time
from pathlib import Path


def find_files_recursively(root: Path, pattern: str) -> list[Path]:
    return sorted(root.rglob(pattern))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pack server-extracted frames + landmarks + manifests into a tar/tar.gz archive"
    )
    parser.add_argument(
        "--extracted-root",
        default=None,
        help="Unified root directory of extracted data (contains frames/, landmarks/, manifests/)",
    )
    parser.add_argument(
        "--frame-root",
        default=None,
        help="Root directory of extracted frames (e.g. E:/DeepFakeData/data/extracted/celebdf)",
    )
    parser.add_argument(
        "--landmark-root",
        default=None,
        help="Root directory of extracted landmarks (e.g. E:/DeepFakeData/data/landmarks/celebdf-landmark)",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional path to celebdf_test_landmarks.jsonl (auto-detected if omitted)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output archive path (default: celebdf_server_extracted.tar or .tar.gz)",
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Create uncompressed .tar instead of .tar.gz (SUPER FAST: ~10s vs ~3 mins)",
    )
    args = parser.parse_args()

    # 1. Resolve source paths
    if args.extracted_root:
        base_extracted = Path(args.extracted_root).resolve()
        frame_dir = base_extracted / "frames" if (base_extracted / "frames").is_dir() else base_extracted
        landmark_dir = base_extracted / "landmarks" if (base_extracted / "landmarks").is_dir() else base_extracted
        manifest_search_roots = [base_extracted / "manifests", base_extracted]
    elif args.frame_root and args.landmark_root:
        frame_dir = Path(args.frame_root).resolve()
        landmark_dir = Path(args.landmark_root).resolve()
        manifest_search_roots = [
            landmark_dir / "manifests",
            landmark_dir,
            frame_dir / "manifests",
            frame_dir,
        ]
    else:
        parser.error("Must provide either --extracted-root OR both --frame-root and --landmark-root")

    if not frame_dir.is_dir():
        raise FileNotFoundError(f"Frame root not found: {frame_dir}")
    if not landmark_dir.is_dir():
        raise FileNotFoundError(f"Landmark root not found: {landmark_dir}")

    # 2. Locate Manifests
    manifest_files: list[Path] = []
    if args.manifest:
        m_path = Path(args.manifest).resolve()
        if not m_path.is_file():
            raise FileNotFoundError(f"Specified manifest not found: {m_path}")
        manifest_files.append(m_path)
    else:
        for m_root in manifest_search_roots:
            if m_root.is_dir():
                for f in m_root.glob("*.jsonl"):
                    if f not in manifest_files:
                        manifest_files.append(f)
                for f in m_root.glob("*.json"):
                    if f not in manifest_files:
                        manifest_files.append(f)

    # 3. Locate Frames and Landmarks
    actual_frame_root = frame_dir / "frames" if (frame_dir / "frames").is_dir() else frame_dir
    actual_landmark_root = landmark_dir / "landmarks" if (landmark_dir / "landmarks").is_dir() else landmark_dir

    print("Scanning directories for files...", flush=True)
    frame_files = find_files_recursively(actual_frame_root, "*.jpg")
    if not frame_files:
        frame_files = find_files_recursively(frame_dir, "*.jpg")
        actual_frame_root = frame_dir

    landmark_files = find_files_recursively(actual_landmark_root, "*.npz")
    if not landmark_files:
        landmark_files = find_files_recursively(landmark_dir, "*.npz")
        actual_landmark_root = landmark_dir

    # 4. Resolve output path
    default_ext = ".tar" if args.no_compress else ".tar.gz"
    output = Path(args.output) if args.output else Path(f"celebdf_server_extracted{default_ext}")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.no_compress else "w:gz"

    print("=" * 72)
    print("  PACKING SERVER-EXTRACTED DATA FOR PI4 DEPLOYMENT")
    print("=" * 72)
    print(f"Frame Root       : {actual_frame_root} ({len(frame_files)} JPEGs)")
    print(f"Landmark Root    : {actual_landmark_root} ({len(landmark_files)} NPZs)")
    print(f"Manifest Files   : {len(manifest_files)} found")
    for mf in manifest_files:
        print(f"  - {mf.name} ({mf.stat().st_size / 1024:.1f} KB)")
    print(f"Output Archive   : {output}")
    print(f"Mode             : {'Fast Uncompressed (.tar)' if args.no_compress else 'Gzip Compressed (.tar.gz)'}")
    print("=" * 72)

    if len(frame_files) == 0:
        raise RuntimeError(f"No .jpg frames found under {frame_dir}")
    if len(landmark_files) == 0:
        raise RuntimeError(f"No .npz landmark files found under {landmark_dir}")

    start_time = time.perf_counter()
    packed_count = 0

    with tarfile.open(str(output), mode) as tar:
        # 1. Add Manifests
        print("\n[1/3] Packing manifests...", flush=True)
        for mf in manifest_files:
            tar.add(str(mf), arcname=f"manifests/{mf.name}")
            packed_count += 1

        # 2. Add Frames
        print(f"\n[2/3] Packing {len(frame_files)} frames (please wait)...", flush=True)
        t0 = time.perf_counter()
        for idx, fp in enumerate(frame_files, 1):
            rel_p = fp.relative_to(actual_frame_root)
            tar.add(str(fp), arcname=f"frames/{rel_p.as_posix()}")
            packed_count += 1
            if idx % 1000 == 0 or idx == len(frame_files):
                pct = (idx / len(frame_files)) * 100
                speed = idx / max(time.perf_counter() - t0, 0.001)
                print(f"\r  Progress: {idx}/{len(frame_files)} frames ({pct:.1f}%) - {speed:.0f} frames/sec", end="", flush=True)
        print()

        # 3. Add Landmarks
        print(f"\n[3/3] Packing {len(landmark_files)} landmark caches...", flush=True)
        for idx, lp in enumerate(landmark_files, 1):
            rel_p = lp.relative_to(actual_landmark_root)
            tar.add(str(lp), arcname=f"landmarks/{rel_p.as_posix()}")
            packed_count += 1
            if idx % 100 == 0 or idx == len(landmark_files):
                pct = (idx / len(landmark_files)) * 100
                print(f"\r  Progress: {idx}/{len(landmark_files)} landmarks ({pct:.1f}%)", end="", flush=True)
        print()

    total_time = time.perf_counter() - start_time
    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"\n{'=' * 72}")
    print(f"[SUCCESS] Packed {packed_count} files into {output}")
    print(f"          Archive Size: {size_mb:.1f} MB")
    print(f"          Total Time  : {total_time:.1f} seconds")
    print(f"{'=' * 72}")
    print("\nNext step (Transfer to Pi4):")
    print(f"  scp {output.name} pi@100.101.16.32:/mnt/usb_data/")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Helper script to package the official 518 Celeb-DF v2 test videos into a .tar file."""

import argparse
import shutil
import tarfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Package 518 Celeb-DF v2 test videos into .tar")
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path to Celeb-DF-v2 folder (e.g. F:/DeepFakedata/Celeb_DFv2)",
    )
    parser.add_argument(
        "--test-list",
        default=None,
        help="Path to List_of_testing_videos.txt (default: dataset-root/List_of_testing_videos.txt)",
    )
    parser.add_argument(
        "--output-tar",
        default=None,
        help="Output .tar file path (default: dataset-root/celebdf_test_518.tar)",
    )
    args = parser.parse_args()

    root = Path(args.dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root directory not found: {root}")

    test_list_file = Path(args.test_list) if args.test_list else root / "List_of_testing_videos.txt"
    if not test_list_file.is_file():
        # Fallback to current directory
        if Path("List_of_testing_videos.txt").is_file():
            test_list_file = Path("List_of_testing_videos.txt")
        else:
            raise FileNotFoundError(f"List_of_testing_videos.txt not found at: {test_list_file}")

    output_tar = Path(args.output_tar) if args.output_tar else root / "celebdf_test_518.tar"
    export_dir = root / "celebdf_test_518_temp"
    export_dir.mkdir(parents=True, exist_ok=True)

    with open(test_list_file, "r", encoding="utf-8") as f:
        lines = [line.strip().split(maxsplit=1) for line in f if line.strip()]

    print(f"[*] Found {len(lines)} videos in test list: {test_list_file}")
    copied = 0

    for official_label, rel_path in lines:
        src = root / rel_path
        dst = export_dir / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_file():
            shutil.copy2(src, dst)
            copied += 1
        else:
            print(f"[!] Warning: Video not found at {src}")

    shutil.copy2(test_list_file, export_dir / "List_of_testing_videos.txt")
    print(f"[*] Successfully copied {copied}/{len(lines)} videos to temporary folder.")

    print(f"[*] Packaging into {output_tar} ...")
    with tarfile.open(output_tar, "w") as tar:
        tar.add(export_dir, arcname="celebdf_test_518")

    # Clean up temporary folder
    shutil.rmtree(export_dir, ignore_errors=True)

    size_mb = output_tar.stat().st_size / (1024 * 1024)
    print(f"[+] Done! Archive created successfully: {output_tar}")
    print(f"[+] Archive Size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()

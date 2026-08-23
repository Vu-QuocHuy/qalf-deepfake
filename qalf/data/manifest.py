"""Portable JSONL video manifest used by extraction, landmarks, train, and evaluation."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class VideoRecord:
    dataset: str
    split: str
    video_id: str
    label: int
    method: str
    source_video: str
    frames: list[str] = field(default_factory=list)
    source_indices: list[int] = field(default_factory=list)
    timestamps_sec: list[float] = field(default_factory=list)
    fps: float = 0.0
    landmark_path: str | None = None
    quality: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.label not in (0, 1):
            raise ValueError(f"{self.video_id}: label must be 0 or 1")
        if not self.frames:
            raise ValueError(f"{self.video_id}: frame list is empty")
        if self.source_indices and len(self.source_indices) != len(self.frames):
            raise ValueError(f"{self.video_id}: source_indices/frame length mismatch")
        if self.timestamps_sec and len(self.timestamps_sec) != len(self.frames):
            raise ValueError(f"{self.video_id}: timestamps/frame length mismatch")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> VideoRecord:
        allowed = {item.name for item in cls.__dataclass_fields__.values()}
        record = cls(**{key: value for key, value in payload.items() if key in allowed})
        record.validate()
        return record


def write_manifest(records: Iterable[VideoRecord], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    if count == 0:
        temporary.unlink(missing_ok=True)
        raise ValueError("Refusing to write an empty manifest")
    temporary.replace(output)


def load_manifest(path: str | Path) -> list[VideoRecord]:
    records: list[VideoRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(VideoRecord.from_dict(json.loads(line)))
            except Exception as error:
                raise ValueError(f"Invalid manifest line {line_number}: {error}") from error
    if not records:
        raise ValueError(f"Manifest is empty: {path}")
    return records


def manifest_summary(records: Iterable[VideoRecord]) -> dict[str, Any]:
    rows = list(records)
    by_label = {str(label): sum(row.label == label for row in rows) for label in (0, 1)}
    methods = sorted({row.method for row in rows})
    return {
        "videos": len(rows),
        "frames": sum(len(row.frames) for row in rows),
        "by_label": by_label,
        "methods": {method: sum(row.method == method for row in rows) for method in methods},
    }

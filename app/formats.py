from __future__ import annotations

"""Output formatting utilities."""

import json
from pathlib import Path
from typing import Iterable, List

from .align import Segment


def _format_time(seconds: float) -> str:
    ms = int(seconds * 1000)
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_txt(segments: Iterable[Segment], path: Path) -> None:
    lines = []
    for seg in segments:
        lines.append(
            f"[{seg.start:0>10.3f} -> {seg.end:0>10.3f}] {seg.speaker}: {seg.text}"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_srt(segments: List[Segment], path: Path) -> None:
    entries = []
    for i, seg in enumerate(segments, 1):
        entries.append(
            f"{i}\n{_format_time(seg.start)} --> {_format_time(seg.end)}\n{seg.speaker}: {seg.text}\n"
        )
    path.write_text("\n".join(entries), encoding="utf-8")


def write_json(segments: List[Segment], path: Path) -> None:
    data = {
        "segments": [
            {
                "start": seg.start,
                "end": seg.end,
                "speaker": seg.speaker,
                "text": seg.text,
                "words": [w.__dict__ for w in seg.words],
            }
            for seg in segments
        ]
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

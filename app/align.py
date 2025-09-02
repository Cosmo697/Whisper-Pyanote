from __future__ import annotations

"""Alignment utilities to merge ASR and diarization."""

from dataclasses import dataclass
from typing import Iterable, List

from .asr import ASRSegment
from .diarization import DiarizationSegment


@dataclass
class Segment:
    start: float
    end: float
    speaker: str
    text: str
    words: List


def assign_speakers(asr_segments: List[ASRSegment], diar_segments: List[DiarizationSegment]) -> List[Segment]:
    """Assign speaker labels to ASR segments.

    Uses two-pointer sweep. Complexity: O(n + m)."""

    results: List[Segment] = []
    j = 0
    for seg in asr_segments:
        best_label = "unknown"
        best_overlap = 0.0
        while j < len(diar_segments) and diar_segments[j].end <= seg.start:
            j += 1
        k = j
        while k < len(diar_segments) and diar_segments[k].start < seg.end:
            overlap = min(seg.end, diar_segments[k].end) - max(seg.start, diar_segments[k].start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = diar_segments[k].speaker
            k += 1
        # normalize speaker label to Speaker N
        speaker_idx = best_label.split("_")[-1] if "_" in best_label else best_label
        speaker = f"Speaker {int(speaker_idx) + 1}" if speaker_idx.isdigit() else best_label
        results.append(
            Segment(
                start=seg.start,
                end=seg.end,
                speaker=speaker,
                text=seg.text,
                words=seg.words,
            )
        )
    return results

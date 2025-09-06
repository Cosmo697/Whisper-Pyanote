# app/align.py  (REPLACE FILE)

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .asr import ASRSegment, Word
from .diarization import DiarizationSegment


@dataclass
class Segment:
    start: float
    end: float
    speaker: str
    text: str
    words: List[Word]


def _normalize_speaker(label: str) -> str:
    base = label
    if "_" in base:
        base = base.split("_")[-1]
    try:
        idx = int(base)
        return f"Speaker {idx + 1}"
    except ValueError:
        if base.lower().startswith("speaker"):
            for token in base.replace("-", "_").split("_")[1:]:
                if token.isdigit():
                    return f"Speaker {int(token) + 1}"
        return label


def _nearest_diar_boundary(t: float, diar: List[DiarizationSegment]) -> Optional[float]:
    nearest = None
    best = 1e9
    for d in diar:
        for edge in (d.start, d.end):
            dist = abs(edge - t)
            if dist < best:
                best = dist
                nearest = edge
    return nearest


def _speaker_at(
    t: float,
    diar: List[DiarizationSegment],
    *,
    boundary_eps: float = 0.12,     # seconds: treat near-boundary as inside
    nearest_eps: float = 0.35       # seconds: snap to nearest diar segment if close
) -> Optional[str]:
    # 1) inclusive with epsilon
    for d in diar:
        if (d.start - boundary_eps) <= t <= (d.end + boundary_eps):
            return _normalize_speaker(d.speaker)

    # 2) nearest segment if close enough
    best_seg: Optional[DiarizationSegment] = None
    best_dist = 1e9
    for d in diar:
        dist = min(abs(t - d.start), abs(t - d.end))
        if dist < best_dist:
            best_dist = dist
            best_seg = d
    if best_seg is not None and best_dist <= nearest_eps:
        return _normalize_speaker(best_seg.speaker)

    return None


def _join_words(words: List[Word]) -> str:
    text = "".join(w.word for w in words).strip()
    if text:
        return text
    return " ".join(w.word.strip() for w in words).strip()


def assign_speakers(
    asr_segments: List[ASRSegment],
    diar_segments: List[DiarizationSegment],
    *,
    snap_window: float = 0.60,
    min_snap_shift: float = 0.08,
    boundary_eps: float = 0.12,      # new: boundary tolerance
    nearest_eps: float = 0.35,       # new: nearest diar fallback
    fill_unknown_window: float = 0.40  # new: neighbor-fill window
) -> List[Segment]:
    """Word-level speaker assignment with boundary tolerance, nearest fallback,
    neighbor fill for unknowns, boundary snapping, and micro-run smoothing."""
    out: List[Segment] = []

    if not diar_segments:
        for seg in asr_segments:
            out.append(Segment(seg.start, seg.end, "unknown", seg.text, seg.words))
        return out

    for seg in asr_segments:
        words = seg.words or []
        if not words:
            # fallback to segment-level if no word stamps
            best = "unknown"
            best_overlap = 0.0
            for d in diar_segments:
                if d.end <= seg.start or d.start >= seg.end:
                    continue
                ov = min(seg.end, d.end) - max(seg.start, d.start)
                if ov > best_overlap:
                    best_overlap = ov
                    best = _normalize_speaker(d.speaker)
            out.append(Segment(seg.start, seg.end, best, seg.text, seg.words))
            continue

        # --- Per-word initial assignment
        assigned: List[Tuple[Word, Optional[str]]] = []
        for w in words:
            mid = (w.start + w.end) / 2.0
            spk = _speaker_at(mid, diar_segments, boundary_eps=boundary_eps, nearest_eps=nearest_eps)
            assigned.append((w, spk))

        # --- Neighbor fill for unknowns (single words or short runs)
        n = len(assigned)
        # pass 1: fill with agreeing neighbors if both sides same speaker
        for i in range(n):
            w, spk = assigned[i]
            if spk is not None:
                continue
            # find nearest known left/right
            L = i - 1
            while L >= 0 and assigned[L][1] is None:
                L -= 1
            R = i + 1
            while R < n and assigned[R][1] is None:
                R += 1
            left = assigned[L][1] if L >= 0 else None
            right = assigned[R][1] if R < n else None
            if left is not None and right is not None and left == right:
                assigned[i] = (w, left)

        # pass 2: fill from nearest neighbor within a time window
        for i in range(n):
            w, spk = assigned[i]
            if spk is not None:
                continue
            # compute distance to nearest known left/right
            left_j = None
            for j in range(i - 1, -1, -1):
                if assigned[j][1] is not None:
                    left_j = j
                    break
            right_j = None
            for j in range(i + 1, n):
                if assigned[j][1] is not None:
                    right_j = j
                    break

            chosen = None
            if left_j is not None or right_j is not None:
                left_dist = float("inf")
                right_dist = float("inf")
                if left_j is not None:
                    left_dist = max(0.0, w.start - assigned[left_j][0].end)
                if right_j is not None:
                    right_dist = max(0.0, assigned[right_j][0].start - w.end)

                if left_dist <= fill_unknown_window and right_dist <= fill_unknown_window:
                    # both close: choose the nearer one
                    chosen = assigned[left_j][1] if left_dist <= right_dist else assigned[right_j][1]
                elif left_dist <= fill_unknown_window:
                    chosen = assigned[left_j][1]
                elif right_dist <= fill_unknown_window:
                    chosen = assigned[right_j][1]

            if chosen is None:
                # last resort: keep unknown (rare)
                chosen = "unknown"
            assigned[i] = (w, chosen)

        # --- Build runs from filled assignment
        runs: List[Segment] = []
        cur_words: List[Word] = []
        cur_speaker: Optional[str] = None

        for w, spk in assigned:
            spk = spk or "unknown"
            if cur_speaker is None:
                cur_speaker, cur_words = spk, [w]
            elif spk == cur_speaker:
                cur_words.append(w)
            else:
                runs.append(Segment(
                    start=cur_words[0].start,
                    end=cur_words[-1].end,
                    speaker=cur_speaker,
                    text=_join_words(cur_words),
                    words=list(cur_words),
                ))
                cur_speaker, cur_words = spk, [w]
        if cur_words:
            runs.append(Segment(
                start=cur_words[0].start,
                end=cur_words[-1].end,
                speaker=cur_speaker or "unknown",
                text=_join_words(cur_words),
                words=list(cur_words),
            ))

        # --- Snap boundaries between adjacent runs to nearby diar edges
        for i in range(1, len(runs)):
            left = runs[i - 1]
            right = runs[i]
            candidates = [left.end, right.start, (left.end + right.start) / 2.0]
            best_edge = None
            best_dist = snap_window
            for t in candidates:
                edge = _nearest_diar_boundary(t, diar_segments)
                if edge is None:
                    continue
                dist = abs(edge - t)
                if dist <= best_dist:
                    best_dist = dist
                    best_edge = edge
            if best_edge is not None:
                left.end = min(max(left.start, best_edge), right.end)
                right.start = left.end

        # --- Merge adjacent same-speaker runs with tiny gaps/overlaps
        def duration(s: Segment) -> float:
            return max(0.0, s.end - s.start)

        merged: List[Segment] = []
        for r in runs:
            if merged and r.speaker == merged[-1].speaker and r.start <= merged[-1].end + 0.02:
                merged[-1].end = max(merged[-1].end, r.end)
                if r.text:
                    merged[-1].text = (merged[-1].text + (" " if merged[-1].text and not merged[-1].text.endswith(" ") else "") + r.text).strip()
                merged[-1].words.extend(r.words)
            else:
                merged.append(r)

        out.extend(merged)

    # Global sort & final coalesce
    out.sort(key=lambda s: (s.start, s.end))
    final: List[Segment] = []
    for s in out:
        if final and s.speaker == final[-1].speaker and s.start <= final[-1].end + 0.02:
            final[-1].end = max(final[-1].end, s.end)
            if s.text:
                final[-1].text = (final[-1].text + (" " if final[-1].text and not final[-1].text.endswith(" ") else "") + s.text).strip()
            final[-1].words.extend(s.words)
        else:
            final.append(s)
    return final

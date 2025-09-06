from __future__ import annotations
"""
Whisper ASR wrapper using faster-whisper with word timestamps and optional VAD controls.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from faster_whisper import WhisperModel

from .errors import TranscriptionError

logger = logging.getLogger(__name__)


@dataclass
class Word:
    start: float
    end: float
    word: str
    prob: float


@dataclass
class ASRSegment:
    start: float
    end: float
    text: str
    words: List[Word]


def load_model(model_name: str, device: str, model_dir: Path) -> WhisperModel:
    """
    Load a faster-whisper model with an appropriate compute_type for the device.
    """
    compute_type = "float16" if device == "cuda" else "int8"
    logger.info("Loading Whisper model %s on %s", model_name, device)
    return WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        download_root=str(model_dir),
    )


def transcribe(
    model: WhisperModel,
    audio_path: Path,
    language: Optional[str],
    *,
    enable_vad: bool = True,
    vad_min_silence_ms: int = 250,
) -> List[ASRSegment]:
    """
    Run faster-whisper with word timestamps enabled.
    - enable_vad: toggles the built-in VAD filter (can reduce 'smeared' boundaries if disabled).
    - vad_min_silence_ms: minimum silence for VAD segmentation if enabled.
    """
    try:
        segments, _ = model.transcribe(
            str(audio_path),
            language=language if language and language != "auto" else None,
            vad_filter=enable_vad,
            vad_parameters={"min_silence_duration_ms": vad_min_silence_ms} if enable_vad else None,
            word_timestamps=True,
            beam_size=5,
            temperature=0.0,
        )
    except Exception as exc:  # pragma: no cover
        raise TranscriptionError(str(exc)) from exc

    out: List[ASRSegment] = []
    for seg in segments:
        words = [
            Word(start=w.start, end=w.end, word=w.word, prob=w.probability)
            for w in (seg.words or [])
            if w.start is not None and w.end is not None
        ]
        out.append(ASRSegment(start=seg.start, end=seg.end, text=seg.text.strip(), words=words))
    return out

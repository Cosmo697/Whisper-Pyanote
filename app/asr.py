from __future__ import annotations

"""Whisper ASR wrapper using faster-whisper."""

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


def load_model(model_name: str, model_dir: Path, device: str) -> WhisperModel:
    """Load a Whisper model.

    Parameters
    ----------
    model_name:
        Name or path of the Whisper model.
    model_dir:
        Directory for caching models.
    device:
        ``"cuda"`` or ``"cpu"``.
    """

    compute_type = "float16" if device == "cuda" else "int8"
    logger.info("Loading Whisper model %s on %s", model_name, device)
    return WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        download_root=str(model_dir),
    )


def transcribe(model: WhisperModel, audio_path: Path, language: Optional[str]) -> List[ASRSegment]:
    """Transcribe audio file.

    Streaming is handled internally by faster-whisper through ffmpeg.
    Complexity: O(n) for n audio duration.
    """

    try:
        segments, _info = model.transcribe(
            str(audio_path),
            language=language,
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,
        )
    except Exception as exc:  # pragma: no cover
        raise TranscriptionError(str(exc)) from exc

    results: List[ASRSegment] = []
    for seg in segments:
        words = [
            Word(start=w.start, end=w.end, word=w.word, prob=w.probability)
            for w in seg.words or []
        ]
        results.append(
            ASRSegment(start=seg.start, end=seg.end, text=seg.text.strip(), words=words)
        )
    return results

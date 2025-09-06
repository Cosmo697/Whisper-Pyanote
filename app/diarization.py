# app/diarization.py  (UPDATED)

from __future__ import annotations

"""pyannote.audio diarization wrapper."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
from pyannote.audio import Pipeline

from .errors import DiarizationError

logger = logging.getLogger(__name__)


@dataclass
class DiarizationSegment:
    start: float
    end: float
    speaker: str


def load_pipeline(token: str, model_dir: Path) -> Pipeline:
    """Load diarization pipeline with token."""
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=token,
            cache_dir=str(model_dir),
        )
    except Exception as exc:  # pragma: no cover
        raise DiarizationError(str(exc)) from exc
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)
    return pipeline


def diarize(
    pipeline: Pipeline,
    audio_path: Path,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> List[DiarizationSegment]:
    """Run diarization pipeline with optional speaker-count constraints."""
    try:
        kwargs = {}
        if num_speakers is not None and num_speakers > 0:
            kwargs["num_speakers"] = num_speakers
        else:
            if min_speakers is not None and min_speakers > 0:
                kwargs["min_speakers"] = min_speakers
            if max_speakers is not None and max_speakers > 0:
                kwargs["max_speakers"] = max_speakers

        diarization = pipeline(str(audio_path), **kwargs)
    except Exception as exc:  # pragma: no cover
        raise DiarizationError(str(exc)) from exc

    results: List[DiarizationSegment] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        results.append(DiarizationSegment(start=turn.start, end=turn.end, speaker=speaker))
    return results

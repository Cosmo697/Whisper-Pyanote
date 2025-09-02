from __future__ import annotations

"""pyannote.audio diarization wrapper."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

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


def diarize(pipeline: Pipeline, audio_path: Path) -> List[DiarizationSegment]:
    """Run diarization pipeline."""

    try:
        diarization = pipeline(str(audio_path))
    except Exception as exc:  # pragma: no cover
        raise DiarizationError(str(exc)) from exc
    results: List[DiarizationSegment] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        results.append(DiarizationSegment(start=turn.start, end=turn.end, speaker=speaker))
    return results

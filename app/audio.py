from __future__ import annotations

"""Audio input and preprocessing."""

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf

from .errors import AudioProcessingError

# Supported audio/video extensions
SUPPORTED_SUFFIXES = {
    ".wav",
    ".mp3",
    ".m4a",
    ".mp4",
    ".flac",
    ".ogg",
}


logger = logging.getLogger(__name__)


def extract_audio(input_path: Path, tmp_dir: Path) -> Path:
    """Extract 16 kHz mono WAV using ffmpeg.

    Uses subprocess with list arguments to avoid shell injection (OWASP)."""

    if not input_path.exists():
        raise AudioProcessingError(f"Input file {input_path} does not exist")
    if input_path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise AudioProcessingError("Unsupported file type")

    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_path = tmp_dir / "audio.wav"

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(out_path),
    ]

    logger.debug("Running ffmpeg: %s", " ".join(cmd))
    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise AudioProcessingError("ffmpeg failed") from exc

    return out_path


def read_audio_chunks(audio_path: Path, chunk_seconds: float = 30.0) -> Iterable[np.ndarray]:
    """Yield audio in chunks to limit memory usage.

    Complexity: O(n) time and O(1) additional memory where n is number of samples."""

    with sf.SoundFile(audio_path) as f:
        samples_per_chunk = int(chunk_seconds * f.samplerate)
        while True:
            data = f.read(samples_per_chunk, dtype="float32")
            if not len(data):
                break
            yield data

from __future__ import annotations

"""CLI entrypoint for transcription and diarization."""

import argparse
import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Optional

import torch

from .align import assign_speakers
from .asr import ASRSegment, load_model, transcribe
from .config import load_config
from .diarization import diarize, load_pipeline
from .errors import AudioProcessingError, ConfigError, DiarizationError, TranscriptionError
from .formats import write_json, write_srt, write_txt
from .logging_utils import setup_logging
from .audio import extract_audio

logger = logging.getLogger(__name__)


def retry(fn, attempts: int = 3, delay: float = 1.0):
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # pragma: no cover - generic retry
            logger.warning("Attempt %d/%d failed: %s", i + 1, attempts, exc)
            time.sleep(delay * (2**i))
    raise


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Transcribe with speaker diarization")
    parser.add_argument("input", help="input audio or video file")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--lang", default="auto", help="language code or auto")
    parser.add_argument("--model", default="medium.en", help="whisper model name")
    parser.add_argument(
        "--hf-token-file", default=".env", help="path to .env containing HF_TOKEN"
    )
    parser.add_argument("--verbose", action="store_true", help="verbose logging")
    args = parser.parse_args(argv)

    setup_logging(args.verbose)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        cfg = load_config(Path(args.hf_token_file))
    except ConfigError as exc:
        logger.error("%s", exc)
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models_dir = Path(".models")
    models_dir.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        try:
            wav_path = extract_audio(Path(args.input), tmp_dir)
        except AudioProcessingError as exc:
            logger.error("Audio processing failed: %s", exc)
            return 1

        def load_asr():
            return load_model(args.model, models_dir, device)

        try:
            asr_model = retry(load_asr)
        except Exception as exc:  # pragma: no cover
            logger.error("ASR model load failed: %s", exc)
            return 1

        lang = None if args.lang == "auto" else args.lang
        try:
            asr_segments = transcribe(asr_model, wav_path, lang)
        except TranscriptionError as exc:
            logger.error("Transcription failed: %s", exc)
            return 1

        if not cfg.hf_token:
            logger.error("HF_TOKEN missing; diarization requires Hugging Face token")
            return 1

        def load_diar():
            return load_pipeline(cfg.hf_token, models_dir)

        try:
            pipeline = retry(load_diar)
        except Exception as exc:  # pragma: no cover
            logger.error("Diarization model load failed: %s", exc)
            return 1

        try:
            diar_segments = diarize(pipeline, wav_path)
        except DiarizationError as exc:
            logger.error("Diarization failed: %s", exc)
            return 1

    final_segments = assign_speakers(asr_segments, diar_segments)

    write_txt(final_segments, out_dir / "transcript.txt")
    write_srt(final_segments, out_dir / "subtitles.srt")
    write_json(final_segments, out_dir / "segments.json")
    logger.info("Wrote outputs to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

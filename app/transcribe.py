# app/transcribe.py

from __future__ import annotations
"""
CLI entrypoint for transcription and diarization, with word-aware boundary snapping.

Model choices are limited to:
  - large-v2
  - large-v3
  - turbo  (mapped to: h2oai/faster-whisper-large-v3-turbo)

New:
  --stream-console  -> print transcript lines to the console as they are decoded.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, List

import torch

from .align import assign_speakers, Segment
from .asr import ASRSegment, load_model, transcribe, Word
from .config import AppConfig, ConfigError, load_config
from .diarization import DiarizationSegment, DiarizationError, load_pipeline, diarize
from .logging_utils import setup_logging

logger = logging.getLogger(__name__)

# Map friendly CLI names to real model identifiers supported by faster-whisper.
MODEL_MAP = {
    "large-v2": "large-v2",
    "large-v3": "large-v3",
    "turbo": "h2oai/faster-whisper-large-v3-turbo",
}


def write_txt(segments: list[Segment], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for s in segments:
            f.write(f"[{s.speaker}] {s.text}\n")


def write_srt(segments: list[Segment], path: Path) -> None:
    def srt_time(t: float) -> str:
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = t % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")

    with path.open("w", encoding="utf-8") as f:
        for i, s in enumerate(segments, 1):
            f.write(f"{i}\n{srt_time(s.start)} --> {srt_time(s.end)}\n[{s.speaker}] {s.text}\n\n")


def write_json(segments: list[Segment], path: Path) -> None:
    payload = [
        {
            "start": s.start,
            "end": s.end,
            "speaker": s.speaker,
            "text": s.text,
            "words": [
                {
                    "start": w.start,
                    "end": w.end,
                    "word": w.word,
                    "prob": getattr(w, "prob", None),
                }
                for w in s.words
            ],
        }
        for s in segments
    ]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Transcribe + diarize with word-aware boundary snapping."
    )
    p.add_argument("audio", help="Path to input audio/video")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--lang", default="auto", help="Language code or 'auto'")

    # Restrict model choices to your requested trio.
    p.add_argument(
        "--model",
        default="large-v2",
        choices=sorted(MODEL_MAP.keys()),
        help="Whisper model: large-v2 | large-v3 | turbo",
    )

    p.add_argument("--hf-token-file", default=".env", help="Path to .env with HF_TOKEN for diarization")
    p.add_argument("-v", "--verbose", action="count", default=0, help="Increase logging verbosity")

    # Boundary snapping controls (for late speaker-change fixes in assign_speakers)
    p.add_argument("--snap-window", type=float, default=0.60,
                   help="Seconds to search for a nearby diarization boundary when splitting speaker runs")
    p.add_argument("--min-snap-shift", type=float, default=0.08,
                   help="Merge micro-runs shorter than this many seconds")

    # VAD controls
    p.add_argument("--no-vad", action="store_true", help="Disable faster-whisper VAD filter")
    p.add_argument("--vad-min-silence-ms", type=int, default=250, help="VAD min silence (ms) if VAD enabled")

    # Speaker-count constraints for pyannote
    p.add_argument("--num-speakers", type=int, default=None, help="Exact number of speakers")
    p.add_argument("--min-speakers", type=int, default=None, help="Lower bound on number of speakers")
    p.add_argument("--max-speakers", type=int, default=None, help="Upper bound on number of speakers")

    # Live console streaming of ASR segments
    p.add_argument("--stream-console", action="store_true",
                   help="Print ASR segments to console as they are decoded (use Ctrl+C to cancel).")

    # Legacy flag kept as a no-op to avoid breaking old callers; it no longer forces a model.
    p.add_argument("--max-accuracy", action="store_true",
                   help="(Deprecated) Kept for compatibility; select the model you want with --model.")

    return p


def _transcribe_with_stream(
    model,
    audio_path: Path,
    language: Optional[str],
    enable_vad: bool,
    vad_min_sil_ms: int,
) -> List[ASRSegment]:
    """
    Stream ASR segments to the console while accumulating them in memory.
    Returns the full list of ASRSegment (unchanged downstream behavior).
    """
    # Use faster-whisper's generator to stream segments
    kwargs = {
        "language": language,
        "vad_filter": enable_vad,
        "vad_parameters": {"min_silence_duration_ms": vad_min_sil_ms},
        "word_timestamps": True,
    }
    segments_gen, info = model.transcribe(str(audio_path), **kwargs)

    # Pretty duration if available
    total = getattr(info, "duration", None)
    if total is not None:
        mm = int(total // 60)
        ss = int(total % 60)
        print(f"[stream] Duration: {mm:02d}:{ss:02d}", flush=True)

    out: List[ASRSegment] = []
    for seg in segments_gen:
        # Console line
        print(f"[{seg.start:8.2f}–{seg.end:8.2f}] {seg.text}", flush=True)

        # Build words list for downstream alignment
        words: List[Word] = []
        for w in getattr(seg, "words", []) or []:
            # faster-whisper uses 'probability' field; keep .prob for our dataclass
            prob = getattr(w, "prob", None)
            if prob is None:
                prob = getattr(w, "probability", None)
            words.append(Word(start=w.start, end=w.end, word=w.word, prob=prob))

        out.append(ASRSegment(start=seg.start, end=seg.end, text=seg.text, words=words))

    # Language detection summary (if present)
    lang = getattr(info, "language", None)
    lp = getattr(info, "language_probability", None)
    if lang:
        if lp is not None:
            print(f"[stream] Detected language '{lang}' (p={lp:.2f})", flush=True)
        else:
            print(f"[stream] Detected language '{lang}'", flush=True)

    return out


def main() -> int:
    args = build_arg_parser().parse_args()
    setup_logging(args.verbose)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Diarization token
    try:
        cfg: AppConfig = load_config(Path(args.hf_token_file))
    except ConfigError as exc:
        logger.error("%s", exc)
        return 1

    # Device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models_dir = Path(".models")

    # Resolve model identifier
    try:
        model_id = MODEL_MAP[args.model]
    except KeyError:
        logger.error("Unsupported model: %s", args.model)
        return 1

    # ASR
    model = load_model(model_id, device, models_dir)
    lang = None if (args.lang or "").lower() == "auto" else args.lang
    logger.info("Using Whisper model %s on %s", args.model, device)

    try:
        if args.stream_console:
            print("[stream] Live transcript on. Press Ctrl+C to cancel.\n", flush=True)
            asr_segments: list[ASRSegment] = _transcribe_with_stream(
                model,
                Path(args.audio),
                lang,
                enable_vad=not args.no_vad,
                vad_min_sil_ms=args.vad_min_silence_ms,
            )
        else:
            # Original non-streaming path
            asr_segments = transcribe(
                model,
                Path(args.audio),
                lang,
                enable_vad=not args.no_vad,
                vad_min_silence_ms=args.vad_min_silence_ms,
            )
    except KeyboardInterrupt:
        print("\n[stream] Cancelled by user.", flush=True)
        return 130  # typical Ctrl+C code
    except Exception as exc:
        logger.exception("Transcription failed: %s", exc)
        return 1

    # Diarization (optional if token present)
    diar_segments: list[DiarizationSegment] = []
    if getattr(cfg, "hf_token", None):
        try:
            pipeline = load_pipeline(cfg.hf_token, models_dir)
            kwargs: dict = {}
            if args.num_speakers and args.num_speakers > 0:
                kwargs["num_speakers"] = args.num_speakers
            else:
                if args.min_speakers:
                    kwargs["min_speakers"] = args.min_speakers
                if args.max_speakers:
                    kwargs["max_speakers"] = args.max_speakers
            logger.info("Diarization kwargs: %s", kwargs or "(none)")
            diar_segments = diarize(pipeline, Path(args.audio), **kwargs)
        except DiarizationError as exc:
            logger.error("Diarization failed: %s", exc)
            return 1
    else:
        logger.warning("HF_TOKEN not found; skipping diarization. All segments will be 'Unknown'.")

    # Word-level alignment & speaker assignment
    final_segments: list[Segment] = assign_speakers(
        asr_segments,
        diar_segments,
        snap_window=float(args.snap_window),
        min_snap_shift=float(args.min_snap_shift),
    )

    # Name outputs after the input file
    base = Path(args.audio).stem
    write_txt(final_segments, out_dir / f"{base}.txt")
    write_srt(final_segments, out_dir / f"{base}.srt")
    write_json(final_segments, out_dir / f"{base}.json")

    logger.info("Wrote outputs to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

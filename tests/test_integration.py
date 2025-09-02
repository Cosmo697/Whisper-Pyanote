import os
import wave
import contextlib
import math
from pathlib import Path

import pytest

from app import transcribe


def _make_wav(path: Path, seconds: int = 2):
    fr = 16000
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fr)
        for i in range(fr * seconds):
            val = int(32767.0 * 0.1 * math.sin(2 * math.pi * 440 * i / fr))
            wf.writeframes(val.to_bytes(2, byteorder="little", signed=True))


@pytest.mark.skipif(not os.getenv("RUN_FULL_TESTS"), reason="requires models")
def test_cli(tmp_path: Path, monkeypatch):
    audio = tmp_path / "tone.wav"
    _make_wav(audio, 2)
    out = tmp_path / "out"
    args = [str(audio), "--out", str(out), "--hf-token-file", str(tmp_path / "fake.env")]
    (tmp_path / "fake.env").write_text("HF_TOKEN=fake")
    with pytest.raises(SystemExit):
        transcribe.main(args)

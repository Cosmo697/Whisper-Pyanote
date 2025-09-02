# Whisper-Pyanote

Production-ready transcription with speaker diarization for Windows.

## Hardware/OS
- Windows 11
- Python 3.x
- NVIDIA RTX 4070 Laptop GPU (CUDA)
- FFmpeg installed and on `PATH`

## Setup
```powershell
# in project root
.\setup.ps1
```
This creates `.venv/` using system `torch` packages, installs pinned requirements, and warms default models into `.models/`.

Fill `.env` with your Hugging Face token (`HF_TOKEN=`) to enable diarization.

## Usage
```powershell
.\.venv\Scripts\Activate.ps1
python -m app.transcribe input.mp4 --out outdir --lang auto --model medium.en --hf-token-file .env
```
Outputs:
- `transcript.txt`
- `subtitles.srt`
- `segments.json`

## Hugging Face Token
Create a token at https://hf.co/settings/tokens and store it in `.env`:
```
HF_TOKEN=xxxxxxxx
```
No environment variables are modified.

## Model Choices
`faster-whisper` models scale from `tiny` (~1GB VRAM) to `large-v3` (~10GB). RTX 4070 Laptop can run `large-v3` in FP16.
Pyannote diarization uses `pyannote/speaker-diarization-3.1`; models download to `.models/`.

## Performance & Scalability
- Whisper inference is linear in audio length (O(n)).
- Diarization pipeline also processes audio in linear time.
- Streaming via ffmpeg avoids loading whole files.
- Expected throughput: ~150x realtime for `medium.en` on RTX 4070.
- Scale vertically with larger GPUs or horizontally by processing files in parallel.

## Troubleshooting
- Ensure FFmpeg is installed and its path is available in PowerShell.
- If CUDA libraries are missing on Windows, download from [whisper-standalone-win](https://github.com/Purfview/whisper-standalone-win).
- If diarization fails, confirm `HF_TOKEN` and model access have been accepted.

## Benchmarks
Run a simple timing:
```powershell
.\bench.ps1 .\samples\demo.wav
```

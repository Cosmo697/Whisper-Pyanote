# --- setup.ps1 ---------------------------------------------------------------
# Idempotent environment bootstrap + optional model prefetch.
# Run from an open PowerShell window in the project folder.

$ErrorActionPreference = "Stop"

# Detect if we're already inside a venv
$insideVenv = $env:VIRTUAL_ENV -ne $null

# 1) Create venv only if missing (you said you want system torch kept)
if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
  python -m venv .venv --system-site-packages
}

# 2) Activate only if not already in a venv
if (-not $insideVenv) {
  & ".\.venv\Scripts\Activate.ps1"
}

# 3) Upgrade pip (quiet)
python -m pip install --upgrade pip

# 4) Install deps
pip install -r requirements.txt

# 5) Ensure .models dir and .env exist
if (-not (Test-Path .models)) { New-Item -ItemType Directory -Path .models | Out-Null }

if (-not (Test-Path .env)) {
@'
HF_TOKEN=
'@ | Out-File -Encoding utf8 .env
  Write-Host "Created .env. Paste your Hugging Face token after HF_TOKEN= and re-run if needed."
}

# 6) Load HF token from .env (robust trim)
$env:HF_TOKEN = (
  Get-Content .env |
    Where-Object { $_ -match '^\s*HF_TOKEN\s*=' } |
    ForEach-Object { $_.Split('=')[1].Trim('"', ' ', "`t") }
)

# 7) Optional: pre-download models if libs are present.
@'
import os

models_dir = ".models"

def has_pkg(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False

# Auto-pick device if torch is present
device = "cpu"
compute_type = None
try:
    import torch
    if torch.cuda.is_available():
        device = "cuda"
        compute_type = "float16"
except Exception:
    pass

# faster-whisper (optional)
if has_pkg("faster_whisper"):
    try:
        from faster_whisper import WhisperModel
        kwargs = dict(download_root=models_dir)
        if device:
            kwargs["device"] = device
        if compute_type:
            kwargs["compute_type"] = compute_type
        WhisperModel("medium.en", **kwargs)
        print(f"Whisper model cached to {models_dir} (device={device}, compute_type={compute_type or 'default'}).")
    except Exception as e:
        print(f"[warn] faster_whisper prefetch failed: {e}")
else:
    print("[info] faster_whisper not installed; skipping Whisper download.")

# pyannote diarization (optional)
hf_token = os.getenv("HF_TOKEN") or ""
if has_pkg("pyannote.audio"):
    if hf_token.strip():
        try:
            from pyannote.audio import Pipeline
            Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token.strip(),
                cache_dir=models_dir,
            )
            print("pyannote diarization model cached.")
        except Exception as e:
            print(f"[warn] pyannote prefetch failed: {e}")
    else:
        print("[info] HF_TOKEN empty; skipping pyannote download.")
else:
    print("[info] pyannote.audio not installed; skipping diarization download.")

print("Bootstrap complete.")
'@ | python -

Write-Host "`nDone."
# Uncomment if you still run by double-click and want the window to pause:
# Read-Host "Press Enter to exit"
# ---------------------------------------------------------------------------

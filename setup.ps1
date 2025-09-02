param()
python -m venv .venv --system-site-packages
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

if (-Not (Test-Path .models)) { New-Item -ItemType Directory -Path .models | Out-Null }
if (-Not (Test-Path .env)) {
@"HF_TOKEN="@ | Out-File -Encoding utf8 .env
}

$env:HF_TOKEN = (Get-Content .env | Select-String 'HF_TOKEN=' | ForEach-Object { $_.Line.Split('=')[1] })
python - <<'PY'
import os
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
models_dir = '.models'
WhisperModel('medium.en', device='cpu', download_root=models_dir)
if os.getenv('HF_TOKEN'):
    Pipeline.from_pretrained('pyannote/speaker-diarization-3.1', use_auth_token=os.getenv('HF_TOKEN'), cache_dir=models_dir)
PY

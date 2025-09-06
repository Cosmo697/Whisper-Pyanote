@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ======================================================================
REM  TranscribeHere.bat  —  Unified launcher (drag & drop enabled)
REM  - Activates .venv
REM  - Lets you choose speaker constraints: Auto / Exact / Bounded
REM  - Lets you pick a Whisper model: large-v2 | large-v3 | turbo
REM  - Optional VAD toggle
REM  - Optional live transcript to console (--stream-console)
REM  - Suppresses noisy deprecation warnings
REM  - Processes all dropped files with proper quoting
REM ======================================================================

REM Go to repo root (this .bat’s folder)
cd /d "%~dp0"

REM Activate venv
if not exist ".\.venv\Scripts\activate.bat" (
  echo ❌ .venv not found. Create it first (e.g.:  py -3 -m venv .venv) && pause && exit /b 1
)
call .\.venv\Scripts\activate.bat

REM ---- Suppress deprecation-style warnings
set "PYTHONWARNINGS=ignore::DeprecationWarning,ignore:.*deprecated.*:UserWarning"

REM ------------------ Prompt: VAD on/off ---------------------------------
set "LANG=auto"
set "VAD_MIN_SIL_MS=250"
set "USE_VAD=Y"

echo.
echo VAD (voice activity detection) helps trim silences. Requires onnxruntime.
echo Enable VAD?  [Y]es / [N]o   (default: Yes)
set /p USE_VAD="> "
if /I "%USE_VAD%"=="N" set "USE_VAD=N"
if /I "%USE_VAD%"=="NO" set "USE_VAD=N"
if "%USE_VAD%"=="" set "USE_VAD=Y"

REM ------------------ Prompt: Speaker constraints ------------------------
:choose_spk_mode
echo.
echo Speaker constraints:
echo   [1] Auto (no constraints)
echo   [2] Exact N speakers
echo   [3] Bounded min..max speakers
set "SPK_MODE="
set /p SPK_MODE="Choose 1, 2, or 3 (default 1): "
if "%SPK_MODE%"=="" set "SPK_MODE=1"
if not "%SPK_MODE%"=="1" if not "%SPK_MODE%"=="2" if not "%SPK_MODE%"=="3" goto choose_spk_mode

set "NUM_SPEAKERS="
set "MIN_SPEAKERS="
set "MAX_SPEAKERS="
if "%SPK_MODE%"=="2" (
  set /p NUM_SPEAKERS="Exact speaker count (e.g., 2, 3, 4): "
)
if "%SPK_MODE%"=="3" (
  set /p MIN_SPEAKERS="Min speakers (e.g., 2): "
  set /p MAX_SPEAKERS="Max speakers (e.g., 3): "
)

REM ------------------ Prompt: Model selection (3 choices) ----------------
:choose_model
echo.
echo Whisper model:
echo   [1] large-v2
echo   [2] large-v3
echo   [3] turbo  (large-v3-turbo via HF CT2)
set "MODEL_CHOICE="
set /p MODEL_CHOICE="Choose 1-3 (default 1 = large-v2): "
if "%MODEL_CHOICE%"=="" set "MODEL_CHOICE=1"
if "%MODEL_CHOICE%"=="1" set "MODEL=large-v2"
if "%MODEL_CHOICE%"=="2" set "MODEL=large-v3"
if "%MODEL_CHOICE%"=="3" set "MODEL=turbo"
if not defined MODEL goto choose_model

REM ------------------ Prompt: Live stream to console ---------------------
set "STREAM=Y"
echo.
echo Show live transcript in console while processing?  [Y]es / [N]o   (default: Yes)
set /p STREAM="> "
if /I "%STREAM%"=="N" set "STREAM=N"
if /I "%STREAM%"=="NO" set "STREAM=N"
if "%STREAM%"=="" set "STREAM=Y"

echo.
echo ===== Settings =====
echo   VAD:                 %USE_VAD%   (silence=%VAD_MIN_SIL_MS% ms)
if "%SPK_MODE%"=="1" echo   Speakers:           Auto (no constraints)
if "%SPK_MODE%"=="2" echo   Speakers:           Exact %NUM_SPEAKERS%
if "%SPK_MODE%"=="3" echo   Speakers:           Min %MIN_SPEAKERS%  Max %MAX_SPEAKERS%
echo   Whisper model:       %MODEL%
echo   Live transcript:     %STREAM%
echo   Language:            %LANG%
echo =====================

REM ------------------ Ensure files were dropped --------------------------
if "%~1"=="" (
  echo.
  echo Drag ^^& drop audio/video files onto this .bat, or pass file paths as args.
  echo Example: TranscribeHere.bat  "D:\Audio\clip.wav"
  echo.
  pause
  exit /b 1
)

REM ------------------ Process each file (robust SHIFT loop) --------------
:each
if "%~1"=="" goto done

set "ABS=%~f1"
set "DIR=%~dp1"
set "BASE=%~n1"
set "OUT=%DIR%%BASE%"
if not exist "%OUT%" mkdir "%OUT%"

echo.
echo Processing: "%ABS%"
echo Output dir: "%OUT%"

set "PYARGS=-m app.transcribe "%ABS%" --out "%OUT%" --hf-token-file ".env" --lang %LANG% --vad-min-silence-ms %VAD_MIN_SIL_MS% --model %MODEL%"

set "VAD_SW="
if /I "%USE_VAD%"=="N" set "VAD_SW=--no-vad"

set "SPK_ARGS="
if "%SPK_MODE%"=="2" (
  if not "%NUM_SPEAKERS%"=="" set "SPK_ARGS=--num-speakers %NUM_SPEAKERS%"
)
if "%SPK_MODE%"=="3" (
  if not "%MIN_SPEAKERS%"=="" set "SPK_ARGS=%SPK_ARGS% --min-speakers %MIN_SPEAKERS%"
  if not "%MAX_SPEAKERS%"=="" set "SPK_ARGS=%SPK_ARGS% --max-speakers %MAX_SPEAKERS%"
)

set "STREAM_SW="
if /I "%STREAM%"=="Y" set "STREAM_SW=--stream-console"

echo Options: model=%MODEL%  vad=%USE_VAD%  live=%STREAM%  lang=%LANG%  %SPK_ARGS%

python %PYARGS% %VAD_SW% %SPK_ARGS% %STREAM_SW%
if errorlevel 1 echo ⚠️  Failed on "%ABS%"

shift
goto each

:done
echo.
echo ✅ Finished.
pause
endlocal

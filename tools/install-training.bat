@echo off
rem ============================================================================
rem  Style training setup (optional). install.bat offers it; can also be run alone.
rem  Safe to run again: finished steps are skipped.
rem    1. Stable Audio 3 trainer code + its own Python with CUDA PyTorch (~5 GB)
rem    2. The "base" models the app generates with when a trained style is used (2 x 2.3 GB)
rem    3. Hugging Face license + login (in your browser / this window; nobody else sees your token)
rem    4. Download of the trainer's models (checks your access works)
rem  Argument "nopause": don't wait for a key at the end (used by install.bat).
rem ============================================================================
setlocal EnableExtensions
title samplegen style training setup
for %%I in ("%~dp0..") do set "ROOT=%%~fI\"
cd /d "%ROOT%"
set "TRAINER=%ROOT%trainer\stable-audio-3"
set "PY=%TRAINER%\.venv\Scripts\python.exe"
set "CKPTS=%ROOT%engine\ComfyUI_windows_portable\ComfyUI\models\checkpoints"
set "LOG=%ROOT%logs\install-training.log"
if not exist "%ROOT%logs" mkdir "%ROOT%logs"
echo samplegen training setup %date% %time% > "%LOG%"

echo.
echo   Style training setup
echo   --------------------
echo [1/6] Trainer code (Stability AI stable-audio-3)...
if exist "%TRAINER%\pyproject.toml" (
  echo       already there
) else (
  curl -fL --retry 5 -o "%ROOT%trainer\sa3.zip" https://codeload.github.com/Stability-AI/stable-audio-3/zip/refs/heads/main || goto :fail
  tar -xf "%ROOT%trainer\sa3.zip" -C "%ROOT%trainer" || goto :fail
  move "%ROOT%trainer\stable-audio-3-main" "%TRAINER%" >nul || goto :fail
  del "%ROOT%trainer\sa3.zip"
)

echo [2/6] Python + CUDA PyTorch for training (several GB, takes a while)...
cd /d "%TRAINER%"
rem Order matters: `uv sync` creates the environment with the Python version the project pins
rem (.python-version) and would DELETE an environment made with another version, torch included.
rem So: sync first (without torch), then add CUDA torch, and never sync again afterwards.
set "SYNCED=%TRAINER%\.venv\samplegen-synced.txt"
if not exist "%SYNCED%" (
  uv sync --extra lora --no-install-package torch --no-install-package torchaudio >> "%LOG%" 2>&1 || goto :fail
  echo synced > "%SYNCED%"
)
rem The project only pins CUDA torch on Linux; on Windows it must be installed explicitly.
uv pip install --python "%PY%" torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126 >> "%LOG%" 2>&1 || goto :fail
"%PY%" -c "import torch; ok = torch.cuda.is_available(); print('torch', torch.__version__, 'cuda', ok); raise SystemExit(0 if ok else 3)" >> "%LOG%" 2>&1 || goto :nocuda
rem Imported by the training code but only declared in the project's "ui" extra (matplotlib)
rem or not declared at all (wandb). Installing the whole "ui" extra would pull in gradio for nothing.
uv pip install --python "%PY%" "matplotlib>=3.10.8" wandb >> "%LOG%" 2>&1 || goto :fail
"%PY%" -c "import matplotlib, wandb, pytorch_lightning; from stable_audio_3.training.diffusion import DiffusionCondTrainingWrapper; print('training imports ok')" >> "%LOG%" 2>&1 || goto :fail
cd /d "%ROOT%"

echo [3/6] Base models for the app (2 x 2.3 GB)...
for %%M in (small_sfx_base small_music_base) do (
  if exist "%CKPTS%\stable_audio_3_%%M.safetensors" (
    echo       %%M already downloaded
  ) else (
    curl -fL --retry 5 -C - -o "%CKPTS%\stable_audio_3_%%M.safetensors.part" https://huggingface.co/Comfy-Org/stable-audio-3/resolve/main/checkpoints/stable_audio_3_%%M.safetensors || goto :fail
    move /y "%CKPTS%\stable_audio_3_%%M.safetensors.part" "%CKPTS%\stable_audio_3_%%M.safetensors" >nul || goto :fail
  )
)

rem Already logged in to Hugging Face (token saved by a previous run)? Skip license + login.
if exist "%USERPROFILE%\.cache\huggingface\token" (
  echo [4/6] + [5/6] Already logged in to Hugging Face - skipped.
  goto :download
)

echo.
echo [4/6] Accept the Stable Audio 3 license (four browser tabs are opening).
echo       On each page, log in to Hugging Face (free account) and agree at the top of the page.
echo       The trainer needs the -base models AND the regular ones (for their text encoder).
start "" "https://huggingface.co/stabilityai/stable-audio-3-small-sfx-base"
start "" "https://huggingface.co/stabilityai/stable-audio-3-small-music-base"
start "" "https://huggingface.co/stabilityai/stable-audio-3-small-sfx"
start "" "https://huggingface.co/stabilityai/stable-audio-3-small-music"
echo       When you have accepted ALL FOUR, come back here.
pause

echo.
echo [5/6] Log in to Hugging Face from this window.
echo       A tab with your tokens opens: create a token with "Read" access and copy it.
echo       Then paste it below (right-click pastes; it stays hidden) and press Enter.
echo       Answer "n" if it asks about git credentials.
start "" "https://huggingface.co/settings/tokens"
if exist "%TRAINER%\.venv\Scripts\hf.exe" (
  "%TRAINER%\.venv\Scripts\hf.exe" auth login || goto :fail
) else (
  "%TRAINER%\.venv\Scripts\huggingface-cli.exe" login || goto :fail
)

:download
echo.
echo [6/6] Downloading the trainer's models (checks your access)...
cd /d "%TRAINER%"
"%PY%" -c "from stable_audio_3.model_configs import base_models; [print(n, base_models[n].resolve()) for n in ('small-sfx-base', 'small-music-base')]" >> "%LOG%" 2>&1 || goto :noaccess
rem Training also loads the text encoder from the regular (non-base) repos: check those too.
"%PY%" -c "from huggingface_hub import hf_hub_download as d; [print(r, d(r, 't5gemma-b-b-ul2/config.json')) for r in ('stabilityai/stable-audio-3-small-sfx', 'stabilityai/stable-audio-3-small-music')]" >> "%LOG%" 2>&1 || goto :noaccess
cd /d "%ROOT%"

echo TRAINING SETUP OK >> "%LOG%"
echo.
echo   Style training is ready: open the Train tab in samplegen.
if /i not "%~1"=="nopause" pause
exit /b 0

:nocuda
echo NO CUDA >> "%LOG%"
echo.
echo   PyTorch was installed without GPU support. Details in logs\install-training.log
pause
exit /b 1

:noaccess
echo NO ACCESS >> "%LOG%"
echo.
echo   Could not download from Stability's model pages. Usually the license wasn't accepted on
echo   ALL FOUR pages, or the token was wrong:
echo     https://huggingface.co/stabilityai/stable-audio-3-small-sfx-base
echo     https://huggingface.co/stabilityai/stable-audio-3-small-music-base
echo     https://huggingface.co/stabilityai/stable-audio-3-small-sfx
echo     https://huggingface.co/stabilityai/stable-audio-3-small-music
echo   Accept/check, then run tools\install-training.bat again (finished steps are skipped).
pause
exit /b 1

:fail
echo FAILED >> "%LOG%"
echo.
echo   Something failed. Details are in logs\install-training.log
pause
exit /b 1

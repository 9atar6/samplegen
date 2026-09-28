@echo off
rem ============================================================================
rem  samplegen installer. Safe to run again: whatever is already installed is skipped.
rem  Needs: Windows 10/11, an NVIDIA GPU with 8 GB+ VRAM, ~40 GB free disk, internet.
rem  Everything goes inside this folder, except your sample library (you choose where).
rem ============================================================================
setlocal EnableExtensions
title samplegen install
cd /d "%~dp0"
set "ROOT=%~dp0"
set "ENGINE_DIR=%ROOT%engine"
set "ENGINE=%ENGINE_DIR%\ComfyUI_windows_portable"
set "COMFY=%ENGINE%\ComfyUI"
set "EPY=%ENGINE%\python_embeded\python.exe"
set "CKPTS=%COMFY%\models\checkpoints"
set "TENC=%COMFY%\models\text_encoders"
set "NODES=%COMFY%\custom_nodes"
set "COMFY_VERSION=v0.37.0"
set "HF=https://huggingface.co"
set "LOG=%ROOT%logs\install.log"
if not exist "%ROOT%logs" mkdir "%ROOT%logs"
echo samplegen install %date% %time% > "%LOG%"

echo.
echo   samplegen installer
echo   -------------------
echo   Downloads about 20 GB the first time. Safe to run again at any point.
echo.

echo [1/8] Checking the GPU...
where nvidia-smi >nul 2>&1
if errorlevel 1 (
  echo       WARNING: no NVIDIA GPU driver found. samplegen needs an NVIDIA GPU with 8 GB or more.
  echo       Close this window to stop, or press a key to continue anyway.
  pause >nul
) else (
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
)

echo [2/8] uv (installs and runs Python for the app)...
where uv >nul 2>&1
if errorlevel 1 (
  powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex" >> "%LOG%" 2>&1 || goto :fail
  set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)
where uv >nul 2>&1 || goto :fail

echo [3/8] The app's Python environment...
uv sync >> "%LOG%" 2>&1 || goto :fail

echo [4/8] The engine (ComfyUI %COMFY_VERSION%, about 2 GB)...
if exist "%COMFY%\main.py" (
  echo       already installed
) else (
  if not exist "%ENGINE_DIR%" mkdir "%ENGINE_DIR%"
  curl -fL --retry 5 -o "%ENGINE_DIR%\7zr.exe" https://www.7-zip.org/a/7zr.exe || goto :fail
  curl -fL --retry 5 -C - -o "%ENGINE_DIR%\comfyui.7z" https://github.com/Comfy-Org/ComfyUI/releases/download/%COMFY_VERSION%/ComfyUI_windows_portable_nvidia.7z || goto :fail
  echo       unpacking...
  "%ENGINE_DIR%\7zr.exe" x -y "-o%ENGINE_DIR%" "%ENGINE_DIR%\comfyui.7z" >> "%LOG%" 2>&1 || goto :fail
  del "%ENGINE_DIR%\comfyui.7z" "%ENGINE_DIR%\7zr.exe"
)
if not exist "%NODES%\samplegen_nodes\__init__.py" (
  rem The engine loads samplegen's own nodes (float WAV in/out, inpainting) through a folder link.
  mklink /J "%NODES%\samplegen_nodes" "%ROOT%engine_nodes\samplegen_nodes" >> "%LOG%" 2>&1 || goto :fail
)

echo [5/8] Sound models (about 16 GB)...
call :get "%CKPTS%\stable_audio_3_small_sfx.safetensors" "%HF%/Comfy-Org/stable-audio-3/resolve/main/checkpoints/stable_audio_3_small_sfx.safetensors" || goto :fail
call :get "%CKPTS%\stable_audio_3_medium.safetensors" "%HF%/Comfy-Org/stable-audio-3/resolve/main/checkpoints/stable_audio_3_medium.safetensors" || goto :fail
call :get "%TENC%\t5gemma_b_b_ul2.safetensors" "%HF%/Comfy-Org/stable-audio-3/resolve/main/text_encoders/t5gemma_b_b_ul2.safetensors" || goto :fail
call :get "%CKPTS%\Foundation-1.2-Samples.safetensors" "%HF%/RoyalCities/Foundation-1/resolve/main/Foundation-1.2-Samples.safetensors" || goto :fail
call :get "%CKPTS%\Foundation-1.2-Keybeds.safetensors" "%HF%/RoyalCities/Foundation-1/resolve/main/Foundation-1.2-Keybeds.safetensors" || goto :fail
call :get "%TENC%\t5_base.safetensors" "%HF%/google-t5/t5-base/resolve/main/model.safetensors" || goto :fail

echo [6/8] Stem splitting extension (AudioSeparation)...
if exist "%NODES%\AudioSeparation\__init__.py" (
  echo       already installed
) else (
  curl -fL --retry 5 -o "%ENGINE_DIR%\audiosep.zip" https://codeload.github.com/set-soft/AudioSeparation/zip/refs/heads/main || goto :fail
  tar -xf "%ENGINE_DIR%\audiosep.zip" -C "%NODES%" || goto :fail
  move "%NODES%\AudioSeparation-main" "%NODES%\AudioSeparation" >nul || goto :fail
  del "%ENGINE_DIR%\audiosep.zip"
)
"%EPY%" -s -m pip install -r "%NODES%\AudioSeparation\requirements.txt" >> "%LOG%" 2>&1 || goto :fail

echo [7/8] Where your samples go...
if exist "%ROOT%samplegen.local.json" (
  echo       already set in samplegen.local.json
  goto :library_done
)
set "LIB=%USERPROFILE%\Music\samplegen-library"
echo       Press Enter to use %LIB%
echo       or type another folder (for example E:\samplegen-library^):
set /p "LIB=      Folder: "
set "LIB=%LIB:"=%"
set "LIBJSON=%LIB:\=\\%"
> "%ROOT%samplegen.local.json" echo {"library": "%LIBJSON%"}
echo       Samples will be saved in %LIB%
:library_done

echo [8/8] Style training (optional)...
echo       Lets you teach samplegen your own sounds. Needs a free Hugging Face account and about 6 GB more.
choice /c YN /m "      Set it up now"
if errorlevel 2 (
  echo       Skipped. Run tools\install-training.bat any time later.
) else (
  call "%ROOT%tools\install-training.bat" nopause || goto :fail
)

echo INSTALL OK >> "%LOG%"
echo.
echo   All done. Start samplegen with samplegen.bat
echo.
pause
exit /b 0

rem ---------------------------------------------------------------------------
:get  <destination> <url>   download once, via a .part file so an interrupted download resumes
if exist "%~1" (
  echo       %~nx1 already downloaded
  exit /b 0
)
echo       %~nx1 ...
curl -fL --retry 5 -C - -o "%~1.part" "%~2" || exit /b 1
move /y "%~1.part" "%~1" >nul || exit /b 1
exit /b 0

:fail
echo INSTALL FAILED >> "%LOG%"
echo.
echo   Something failed. Details are in logs\install.log
echo   Fix the cause (internet, disk space...) and run install.bat again: finished steps are skipped.
pause
exit /b 1

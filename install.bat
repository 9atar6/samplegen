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
set "NEED_GB=40"
if not exist "%ROOT%logs" mkdir "%ROOT%logs"
echo samplegen install %date% %time% > "%LOG%"

echo.
echo   =====================================================================
echo     samplegen installer
echo   =====================================================================
echo     This sets up everything samplegen needs, in this folder:
echo       - the app and its Python             (a few minutes)
echo       - the sound engine (ComfyUI)          (about 2 GB)
echo       - the sound models                    (about 16 GB)
echo     The first install takes roughly 30 to 90 minutes, mostly downloading.
echo     You can keep using your PC meanwhile. Keep this window open.
echo.
echo     If anything goes wrong, just run install.bat again:
echo     finished steps are skipped and downloads resume where they stopped.
echo   =====================================================================
echo.
echo     Press any key to start (or close this window to cancel).
pause >nul
echo.

echo [1/9] Checking your PC...
call :check_disk
call :check_gpu

echo.
echo [2/9] uv (the tool that installs Python for the app)...
where uv >nul 2>&1
if errorlevel 1 (
  echo       downloading uv...
  powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex" >> "%LOG%" 2>&1 || goto :fail
  set "PATH=%USERPROFILE%\.local\bin;%PATH%"
) else (
  echo       already installed
)
where uv >nul 2>&1 || goto :fail

echo.
echo [3/9] The app's Python environment...
echo       This can take a few minutes with nothing moving on screen. That's normal.
uv sync >> "%LOG%" 2>&1 || goto :fail
echo       done

echo.
echo [4/9] The sound engine (ComfyUI %COMFY_VERSION%, about 2 GB)...
if exist "%COMFY%\main.py" (
  echo       already installed
) else (
  if not exist "%ENGINE_DIR%" mkdir "%ENGINE_DIR%"
  curl -fL --retry 5 -o "%ENGINE_DIR%\7zr.exe" https://www.7-zip.org/a/7zr.exe || goto :fail
  curl -fL --retry 5 -C - -o "%ENGINE_DIR%\comfyui.7z" https://github.com/Comfy-Org/ComfyUI/releases/download/%COMFY_VERSION%/ComfyUI_windows_portable_nvidia.7z || goto :fail
  echo       unpacking, 2-5 minutes...
  "%ENGINE_DIR%\7zr.exe" x -y "-o%ENGINE_DIR%" "%ENGINE_DIR%\comfyui.7z" >> "%LOG%" 2>&1 || goto :fail
  del "%ENGINE_DIR%\comfyui.7z" "%ENGINE_DIR%\7zr.exe"
)
if not exist "%NODES%\samplegen_nodes\__init__.py" (
  rem The engine loads samplegen's own nodes (float WAV in/out, inpainting) through a folder link.
  mklink /J "%NODES%\samplegen_nodes" "%ROOT%engine_nodes\samplegen_nodes" >> "%LOG%" 2>&1 || goto :fail
)

echo.
echo [5/9] Sound models (about 16 GB, the long part)...
echo       Each file shows its own progress bar.
call :get "%CKPTS%\stable_audio_3_small_sfx.safetensors" "%HF%/Comfy-Org/stable-audio-3/resolve/main/checkpoints/stable_audio_3_small_sfx.safetensors" || goto :fail
call :get "%CKPTS%\stable_audio_3_medium.safetensors" "%HF%/Comfy-Org/stable-audio-3/resolve/main/checkpoints/stable_audio_3_medium.safetensors" || goto :fail
call :get "%TENC%\t5gemma_b_b_ul2.safetensors" "%HF%/Comfy-Org/stable-audio-3/resolve/main/text_encoders/t5gemma_b_b_ul2.safetensors" || goto :fail
call :get "%CKPTS%\Foundation-1.2-Samples.safetensors" "%HF%/RoyalCities/Foundation-1/resolve/main/Foundation-1.2-Samples.safetensors" || goto :fail
call :get "%CKPTS%\Foundation-1.2-Keybeds.safetensors" "%HF%/RoyalCities/Foundation-1/resolve/main/Foundation-1.2-Keybeds.safetensors" || goto :fail
call :get "%TENC%\t5_base.safetensors" "%HF%/google-t5/t5-base/resolve/main/model.safetensors" || goto :fail

echo.
echo [6/9] Stem splitting (AudioSeparation extension)...
if exist "%NODES%\AudioSeparation\__init__.py" (
  echo       already installed
) else (
  curl -fL --retry 5 -o "%ENGINE_DIR%\audiosep.zip" https://codeload.github.com/set-soft/AudioSeparation/zip/refs/heads/main || goto :fail
  tar -xf "%ENGINE_DIR%\audiosep.zip" -C "%NODES%" || goto :fail
  rem A half-finished earlier attempt would make "move" nest the folder instead of replacing it.
  if exist "%NODES%\AudioSeparation" rmdir /s /q "%NODES%\AudioSeparation"
  move "%NODES%\AudioSeparation-main" "%NODES%\AudioSeparation" >nul || goto :fail
  del "%ENGINE_DIR%\audiosep.zip"
)
echo       installing its Python packages (a minute or two)...
"%EPY%" -s -m pip install -r "%NODES%\AudioSeparation\requirements.txt" >> "%LOG%" 2>&1 || goto :fail

echo.
echo [7/9] MIDI extraction (turns loops and melodies into MIDI notes)...
call "%ROOT%tools\install-midi.bat" nopause || goto :fail

echo.
echo [8/9] Where should your samples be saved?
if exist "%ROOT%samplegen.local.json" (
  echo       already chosen earlier ^(to change it, delete samplegen.local.json and run install.bat again^)
  goto :library_done
)
set "LIB=%USERPROFILE%\Music\samplegen-library"
echo       Everything you generate, keep and export goes into one folder.
echo       - Press Enter to use:  %LIB%
echo       - Or type another folder, e.g.  E:\samplegen-library   then press Enter.
echo       (It's created if it doesn't exist. An external drive is fine.)
set /p "LIB=      Folder: "
set "LIB=%LIB:"=%"
rem Written by Python, not "echo": cmd would save accented letters (Jose with an accent, Musica...)
rem in the console codepage and the app, reading UTF-8, would silently ignore the setting.
rem The folder goes through the environment, not the command line, so "E:\" can't break the quoting.
".venv\Scripts\python.exe" -c "import json,os; open('samplegen.local.json','w',encoding='utf-8').write(json.dumps({'library': os.path.abspath(os.environ['LIB'].strip())}))" >> "%LOG%" 2>&1 || goto :fail
echo       Samples will be saved in %LIB%
:library_done

echo.
echo [9/9] Style training (optional, you can add it any time later)
echo       Teach samplegen your own sounds: give it 20-50 of your samples and it learns their
echo       character. It needs about 6 GB more and a free Hugging Face account; the setup opens
echo       the web pages you need and tells you exactly what to click.
choice /c YN /m "      Set it up now"
if errorlevel 2 (
  echo       Skipped. To add it later, double-click tools\install-training.bat
  call :offer_search
) else (
  call "%ROOT%tools\install-training.bat" nopause || goto :fail
)

echo.
choice /c YN /m "Put a samplegen shortcut on your desktop"
if not errorlevel 2 call :shortcut

echo INSTALL OK >> "%LOG%"
echo.
echo   =====================================================================
echo     All done!
echo   =====================================================================
echo     To start samplegen: double-click samplegen.bat (or the desktop shortcut).
echo     It opens in your web browser. The first start takes about a minute
echo     while the engine wakes up; the first sound of each model takes
echo     a little longer while the model loads.
echo     To quit samplegen, close its black window.
echo   =====================================================================
echo.
choice /c YN /m "Start samplegen now"
if errorlevel 2 exit /b 0
start "" "%ROOT%samplegen.bat"
exit /b 0

rem ---------------------------------------------------------------------------
:get  <destination> <url>   download once, via a .part file so an interrupted download resumes
if exist "%~1" (
  echo       %~nx1 already downloaded
  exit /b 0
)
echo       %~nx1
curl -fL --retry 5 -C - -o "%~1.part" "%~2" || exit /b 1
move /y "%~1.part" "%~1" >nul || exit /b 1
exit /b 0

:check_disk
set "FREE_GB="
for /f "usebackq" %%F in (`powershell -NoProfile -Command "[math]::Floor((Get-Item -LiteralPath '%ROOT%').PSDrive.Free / 1GB)"`) do set "FREE_GB=%%F"
if not defined FREE_GB exit /b 0
echo       Free disk space here: %FREE_GB% GB
if %FREE_GB% GEQ %NEED_GB% exit /b 0
echo.
echo       WARNING: samplegen needs about %NEED_GB% GB on this drive for a first install.
echo       Free some space (or move this folder to a bigger drive), then run install.bat again.
echo       Press any key to try anyway, or close this window to stop.
pause >nul
exit /b 0

:check_gpu
where nvidia-smi >nul 2>&1
if errorlevel 1 (
  echo.
  echo       WARNING: no NVIDIA graphics driver found. samplegen needs an NVIDIA GPU with 8 GB+
  echo       of video memory, with its driver installed ^(nvidia.com/drivers^).
  echo       Press any key to continue anyway, or close this window to stop.
  pause >nul
  exit /b 0
)
set "VRAM_MB="
for /f "tokens=1" %%V in ('nvidia-smi --query-gpu^=memory.total --format^=csv^,noheader^,nounits') do if not defined VRAM_MB set "VRAM_MB=%%V"
nvidia-smi --query-gpu=name --format=csv,noheader
if not defined VRAM_MB exit /b 0
echo       Video memory: %VRAM_MB% MB
if %VRAM_MB% GEQ 7500 exit /b 0
echo.
echo       WARNING: less than 8 GB of video memory. Short sounds may work; long ones and
echo       instruments will probably run out of memory.
echo       Press any key to continue anyway, or close this window to stop.
pause >nul
exit /b 0

:offer_search
echo.
echo       Search by sound (optional, about 1.5 GB): find sounds in your library by what they
echo       sound like - "metallic scrape", or "more like this one". Style training includes it.
choice /c YN /m "      Set up search by sound now"
if errorlevel 2 (
  echo       Skipped. To add it later, double-click tools\install-search.bat
  exit /b 0
)
call "%ROOT%tools\install-search.bat" nopause
exit /b 0

:shortcut
powershell -NoProfile -Command "$s = (New-Object -ComObject WScript.Shell).CreateShortcut([IO.Path]::Combine([Environment]::GetFolderPath('Desktop'), 'samplegen.lnk')); $s.TargetPath = '%ROOT%samplegen.bat'; $s.WorkingDirectory = '%ROOT%'; $s.Save()" >> "%LOG%" 2>&1 && echo       Shortcut created. || echo       Could not create the shortcut ^(not a problem: use samplegen.bat^).
exit /b 0

:fail
echo INSTALL FAILED >> "%LOG%"
echo.
echo   =====================================================================
echo     Something went wrong. Nothing is lost:
echo     1. Check your internet connection and free disk space.
echo     2. Run install.bat again: finished steps are skipped.
echo     If it keeps failing, send the file logs\install.log to whoever
echo     shared samplegen with you.
echo   =====================================================================
pause
exit /b 1

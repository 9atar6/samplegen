@echo off
rem ============================================================================
rem  Search by sound (optional): lets samplegen find sounds in your library by what they
rem  sound like ("metallic scrape", "more like this one"). About 1.5 GB.
rem  Not needed if style training is installed: search then uses the trainer's Python.
rem  Safe to run again. Argument "nopause": don't wait for a key at the end.
rem ============================================================================
setlocal EnableExtensions
title samplegen search setup
for %%I in ("%~dp0..") do set "ROOT=%%~fI\"
cd /d "%ROOT%"
set "VENV=%ROOT%search\.venv"
set "PY=%VENV%\Scripts\python.exe"
set "LOG=%ROOT%logs\install-search.log"
if not exist "%ROOT%logs" mkdir "%ROOT%logs"
echo samplegen search setup %date% %time% > "%LOG%"

echo.
echo   Search by sound (about 1.5 GB, 5-15 minutes)
echo   --------------------------------------------
if exist "%ROOT%trainer\stable-audio-3\.venv\Scripts\python.exe" (
  echo       Style training is installed: search already works with it. Nothing to do.
  goto :done
)
where uv >nul 2>&1
if errorlevel 1 set "PATH=%USERPROFILE%\.local\bin;%PATH%"
where uv >nul 2>&1 || goto :nouv

echo       Setting up its Python (nothing moves on screen for a while, that's normal)...
if not exist "%PY%" (
  uv venv "%VENV%" --python 3.11 >> "%LOG%" 2>&1 || goto :fail
)
rem CPU PyTorch: search runs on the processor and leaves the graphics card to generation.
uv pip install --python "%PY%" torch torchaudio --index-url https://download.pytorch.org/whl/cpu >> "%LOG%" 2>&1 || goto :fail
uv pip install --python "%PY%" "transformers>=4.40" soundfile numpy >> "%LOG%" 2>&1 || goto :fail
echo       Downloading the CLAP model (about 800 MB)...
cd /d "%ROOT%trainer"
"%PY%" -c "from transformers import ClapModel, ClapProcessor; ClapModel.from_pretrained('laion/larger_clap_general'); ClapProcessor.from_pretrained('laion/larger_clap_general'); import clap_describe; print('clap ok')" >> "%LOG%" 2>&1 || goto :fail
cd /d "%ROOT%"
echo SEARCH SETUP OK >> "%LOG%"

:done
echo       Search by sound is ready. In the Library, switch the search box to "by sound".
echo       (samplegen listens to your library in the background the first time.)
if /i not "%~1"=="nopause" pause
exit /b 0

:nouv
echo   uv isn't installed: run install.bat first.
pause
exit /b 1

:fail
echo FAILED >> "%LOG%"
echo.
echo   Search setup failed. Details are in logs\install-search.log
echo   Check your internet connection and run this again.
if /i not "%~1"=="nopause" pause
exit /b 1

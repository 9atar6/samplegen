@echo off
rem ============================================================================
rem  MIDI extraction (Spotify basic-pitch): turns loops and melodies into MIDI notes.
rem  install.bat runs this; you can also double-click it on its own. Safe to run again.
rem  It gets its own small Python (3.10): basic-pitch needs older libraries than the app.
rem  Argument "nopause": don't wait for a key at the end (used by install.bat).
rem ============================================================================
setlocal EnableExtensions
title samplegen MIDI setup
for %%I in ("%~dp0..") do set "ROOT=%%~fI\"
cd /d "%ROOT%"
set "MIDI=%ROOT%midi"
set "PY=%MIDI%\.venv\Scripts\python.exe"
set "LOG=%ROOT%logs\install-midi.log"
if not exist "%ROOT%logs" mkdir "%ROOT%logs"
echo samplegen MIDI setup %date% %time% > "%LOG%"

echo.
echo   MIDI extraction (about 300 MB, a few minutes)
echo   ---------------------------------------------
where uv >nul 2>&1
if errorlevel 1 set "PATH=%USERPROFILE%\.local\bin;%PATH%"
where uv >nul 2>&1 || goto :nouv

echo       Setting up its Python (nothing moves on screen for a while, that's normal)...
if not exist "%PY%" (
  uv venv "%MIDI%\.venv" --python 3.10 >> "%LOG%" 2>&1 || goto :fail
)
rem Always run: quick when everything is there, and repairs a half-finished install.
rem setuptools<81: basic-pitch's resampling library still imports pkg_resources, which newer setuptools dropped.
uv pip install --python "%PY%" "basic-pitch==0.4.0" "setuptools<81" >> "%LOG%" 2>&1 || goto :fail

echo       Checking it works...
"%PY%" -c "from basic_pitch.inference import predict; from basic_pitch import ICASSP_2022_MODEL_PATH; print('basic-pitch ok', ICASSP_2022_MODEL_PATH)" >> "%LOG%" 2>&1 || goto :fail
echo MIDI SETUP OK >> "%LOG%"
echo       MIDI extraction is ready: use the MIDI button on any loop or melody.
if /i not "%~1"=="nopause" pause
exit /b 0

:nouv
echo   uv isn't installed: run install.bat first.
pause
exit /b 1

:fail
echo FAILED >> "%LOG%"
echo.
echo   MIDI setup failed. Details are in logs\install-midi.log
echo   Check your internet connection and run this again.
if /i not "%~1"=="nopause" pause
exit /b 1

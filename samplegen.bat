@echo off
rem samplegen: starts the app (and the engine if needed), then opens your browser.
rem Close this window to quit.
title samplegen
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto :not_installed
if not exist "engine\ComfyUI_windows_portable\ComfyUI\main.py" goto :not_installed
echo.
echo   Starting samplegen... your browser opens in a moment.
echo   The engine needs about a minute to wake up (the top-right light turns green).
echo   Keep this window open while you use samplegen; close it to quit.
".venv\Scripts\python.exe" -m samplegen
if errorlevel 1 (
  echo.
  echo   samplegen stopped with an error ^(see above^). Details are in the logs folder.
  pause
)
exit /b

:not_installed
echo.
echo   samplegen isn't installed yet: double-click install.bat first.
echo.
pause

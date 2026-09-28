@echo off
rem samplegen: starts the app (and the engine if needed), then opens your browser.
rem Close this window to quit.
title samplegen
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto :not_installed
if not exist "engine\ComfyUI_windows_portable\ComfyUI\main.py" goto :not_installed
".venv\Scripts\python.exe" -m samplegen
if errorlevel 1 pause
exit /b

:not_installed
echo samplegen isn't installed yet: double-click install.bat first.
pause

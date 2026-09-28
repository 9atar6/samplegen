@echo off
rem Runs the samplegen test suite directly in Windows (no sandbox).
rem Results are also saved to logs\test-results.txt.
title samplegen tests
cd /d "%~dp0.."
if not exist logs mkdir logs
".venv\Scripts\python.exe" -m pytest --cov --cov-report=term-missing -q > logs\test-results.txt 2>&1
type logs\test-results.txt
echo.
echo Done. You can close this window.
pause

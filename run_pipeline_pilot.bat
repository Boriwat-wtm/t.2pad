@echo off
rem Pilot: first 20 pages, full pipeline, VLM on Intel iGPU. Downloads the dataset (~1.4 GB) the first time.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
.venv-paddle\Scripts\python.exe run_pipeline.py --limit 20
echo.
echo Pilot finished. Send back the last lines (avg s/page).
pause

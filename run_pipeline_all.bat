@echo off
rem All 1651 pages. Resumable: close any time, run again to continue.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
.venv-paddle\Scripts\python.exe run_pipeline.py
echo.
pause

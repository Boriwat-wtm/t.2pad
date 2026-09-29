@echo off
rem Score the pipeline outputs with OmniDocBench v1.6 @ 7279eea (Edit_dist + TEDS, no CDM).
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set "PATH=%USERPROFILE%\.local\bin;%PATH%"

if not exist OmniDocBench\.git (
    git clone -q https://github.com/opendatalab/OmniDocBench.git || goto :err
)
git -C OmniDocBench checkout -q 7279eea || goto :err
if not exist .venv-eval\Scripts\python.exe (
    uv venv --python 3.10 .venv-eval || goto :err
    uv pip install --python .venv-eval\Scripts\python.exe -e OmniDocBench || goto :err
)
.venv-eval\Scripts\python.exe eval_pipeline.py %1 || goto :err
pause
exit /b 0

:err
echo.
echo [eval] FAILED - copy the error above and send it back.
pause
exit /b 1

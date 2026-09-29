@echo off
rem Extra setup for the full pipeline (run after setup.bat). Installs Paddle CPU + PaddleOCR in .venv-paddle.
cd /d "%~dp0"
set "PATH=%USERPROFILE%\.local\bin;%PATH%"

if not exist .venv\Scripts\python.exe (
    echo [setup_pipeline] run setup.bat first
    goto :err
)
if not exist .venv-paddle\Scripts\python.exe (
    uv venv --python 3.12 .venv-paddle || goto :err
)
echo [setup_pipeline] installing paddlepaddle CPU + paddleocr 3.7.0 ...
uv pip install --python .venv-paddle\Scripts\python.exe paddlepaddle==3.3.1 "paddleocr[doc-parser]==3.7.0" openai huggingface_hub || goto :err
.venv-paddle\Scripts\python.exe -c "import paddle, paddleocr; print('paddle', paddle.__version__, 'paddleocr', paddleocr.__version__)" || goto :err
echo.
echo [setup_pipeline] DONE. Next: run_pipeline_pilot.bat
pause
exit /b 0

:err
echo.
echo [setup_pipeline] FAILED - copy the error above and send it back.
pause
exit /b 1

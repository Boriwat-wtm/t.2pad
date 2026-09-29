@echo off
rem Step 0: create Python env + install deps + prepare model. Run once.
cd /d "%~dp0"

where uv >nul 2>nul
if errorlevel 1 (
    echo [setup] installing uv ...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)

if not exist .venv\Scripts\python.exe (
    uv venv --python 3.12 .venv || goto :err
)
echo [setup] installing PyTorch CPU + OpenVINO deps (a few GB, takes a while) ...
uv pip install --python .venv\Scripts\python.exe torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match || goto :err
uv pip install --python .venv\Scripts\python.exe "openvino>=2025.4.1" "nncf>=2.18.0" "transformers==4.54.0" "datasets>=3.6.0" pillow "numpy>=1.21.6" "opencv-python>=4.11.0.86" "huggingface-hub>=0.32.4" tqdm requests "protobuf>=6.32.0" "sentencepiece>=0.2.1" "einops>=0.8.1" || goto :err

.venv\Scripts\python.exe -c "import openvino as ov; print('OpenVINO', ov.get_version(), 'devices:', ov.Core().available_devices)" || goto :err
.venv\Scripts\python.exe prepare_model.py || goto :err
echo.
echo [setup] DONE. Next: run_test.bat
pause
exit /b 0

:err
echo.
echo [setup] FAILED - copy the error above and send it back.
pause
exit /b 1

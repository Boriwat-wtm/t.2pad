@echo off
rem Speed test: 20 OmniDocBench pages on Intel iGPU, then on CPU.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
echo ===== GPU (Intel Arc iGPU) =====
.venv\Scripts\python.exe run_test.py --device GPU --n 20
echo.
echo ===== CPU =====
.venv\Scripts\python.exe run_test.py --device CPU --n 20
echo.
echo Finished. Send back the two SUMMARY blocks above.
pause

@echo off
rem Diagnose low table score: same 50 table pages, VLM at f32 vs the existing f16 run.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
.venv-paddle\Scripts\python.exe make_table_pages.py 50 || goto :err
.venv-paddle\Scripts\python.exe run_pipeline.py --precision f32 --pages pages_tables50.txt --tag tables50 || goto :err
echo.
echo ===== scoring f16 (existing run) on the 50 table pages =====
call eval.bat paddleocr_vl15_ov_gpu_f16_fp --pages pages_tables50.txt
echo ===== scoring f32 on the same 50 table pages =====
call eval.bat paddleocr_vl15_ov_gpu_f32_fp_tables50 --pages pages_tables50.txt
echo Send back both SCORE tables.
pause
exit /b 0
:err
echo FAILED - send the error above.
pause
exit /b 1

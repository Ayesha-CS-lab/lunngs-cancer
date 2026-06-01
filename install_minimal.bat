@echo off
echo Installing minimal dependencies for dataset preparation...
echo.

REM Install only what's needed for dataset preparation
pip install --default-timeout=100 kagglehub
pip install --default-timeout=100 tqdm
pip install --default-timeout=100 Pillow

echo.
echo Minimal dependencies installed!
echo You can now run: python prepare_kaggle_dataset.py
pause

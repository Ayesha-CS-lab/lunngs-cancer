@echo off
echo ========================================
echo Installing Dependencies in Stages
echo ========================================
echo.
echo This installs packages in small groups to avoid timeouts.
echo.

echo [1/5] Installing basic scientific libraries...
pip install --default-timeout=100 pandas numpy scikit-learn
if %errorlevel% neq 0 (
    echo ERROR: Failed to install basic libraries
    pause
    exit /b 1
)

echo.
echo [2/5] Installing PyTorch (this is large, ~240MB, may take time)...
pip install --default-timeout=200 torch torchvision --index-url https://download.pytorch.org/whl/cpu
if %errorlevel% neq 0 (
    echo ERROR: Failed to install PyTorch
    pause
    exit /b 1
)

echo.
echo [3/5] Installing image processing libraries...
pip install --default-timeout=100 opencv-python Pillow
if %errorlevel% neq 0 (
    echo ERROR: Failed to install image libraries
    pause
    exit /b 1
)

echo.
echo [4/5] Installing augmentation and utilities...
pip install --default-timeout=100 albumentations matplotlib seaborn
if %errorlevel% neq 0 (
    echo ERROR: Failed to install augmentation libraries
    pause
    exit /b 1
)

echo.
echo [5/5] Installing additional ML tools...
pip install --default-timeout=100 xgboost joblib pydicom
if %errorlevel% neq 0 (
    echo ERROR: Failed to install ML tools
    pause
    exit /b 1
)

echo.
echo ========================================
echo SUCCESS! All dependencies installed
echo ========================================
echo.
echo You can now train models:
echo   python train.py --model efficientnet --data_dir data/kaggle_lung_cancer --epochs 50
echo.
pause

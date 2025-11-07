@echo off
REM Quick Start Script for GPT Training (Windows)
REM This script helps you set up and start training quickly

echo ==================================================
echo GPT Training Quick Start
echo ==================================================
echo.

REM Step 1: Check Python
echo Step 1: Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo X Python not found. Please install Python 3.8 or later.
    exit /b 1
)
for /f "tokens=*" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo [OK] Found: %PYTHON_VERSION%
echo.

REM Step 2: Create virtual environment
echo Step 2: Setting up virtual environment...
if exist venv (
    echo Virtual environment already exists. Skipping...
) else (
    echo Creating virtual environment...
    python -m venv venv
    echo [OK] Virtual environment created
)
echo.

REM Step 3: Activate virtual environment
echo Step 3: Activating virtual environment...
call venv\Scripts\activate.bat
echo [OK] Virtual environment activated
echo.

REM Step 4: Install dependencies
echo Step 4: Installing dependencies...
echo This may take a few minutes...
python -m pip install --upgrade pip
pip install -r requirements.txt
echo [OK] Dependencies installed
echo.

REM Step 5: Check hardware
echo Step 5: Detecting hardware...
python check_hardware.py
echo.

REM Step 6: Prepare data
echo Step 6: Preparing dataset...
if exist data\train.bin if exist data\val.bin (
    echo Dataset already prepared. Skipping...
) else (
    echo Preparing Shakespeare dataset...
    cd data
    python prepare.py
    cd ..
    echo [OK] Dataset prepared
)
echo.

REM Step 7: Ready to train
echo ==================================================
echo Setup Complete! Ready to train.
echo ==================================================
echo.
echo To start training, run:
echo   python train.py
echo.
echo Or for interactive hardware selection:
echo   python train.py --interactive
echo.
echo For help, see:
echo   - GETTING_STARTED.md (comprehensive guide)
echo   - HARDWARE_FEATURE_SUMMARY.md (hardware features)
echo.
echo Happy training!
pause

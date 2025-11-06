@echo off
REM =============================================================================
REM GPT Training - Complete Automated Setup (Windows)
REM =============================================================================

echo ======================================================================
echo            GPT Training - Automated Setup Script
echo ======================================================================
echo.

REM =============================================================================
REM Step 1: Check/Install Python
REM =============================================================================

echo Step 1: Checking Python...

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Python not found!
    echo.
    echo Please install Python 3.8 or later from:
    echo   https://www.python.org/downloads/
    echo.
    echo IMPORTANT: Check "Add Python to PATH" during installation!
    echo.
    echo After installing Python, run this script again.
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [OK] Python %PYTHON_VERSION% found
echo.

REM =============================================================================
REM Step 2: Create Virtual Environment
REM =============================================================================

echo Step 2: Setting up virtual environment...

if exist venv (
    echo Virtual environment already exists
) else (
    echo Creating virtual environment...
    python -m venv venv
    echo [OK] Virtual environment created
)

REM Activate virtual environment
call venv\Scripts\activate.bat
echo [OK] Virtual environment activated
echo.

REM =============================================================================
REM Step 3: Install Dependencies
REM =============================================================================

echo Step 3: Installing dependencies...
echo This may take a few minutes...

python -m pip install --upgrade pip --quiet
pip install -r requirements.txt

echo [OK] Dependencies installed
echo.

REM =============================================================================
REM Step 4: Detect Hardware
REM =============================================================================

echo Step 4: Detecting hardware...
python check_hardware.py
echo.

REM =============================================================================
REM Step 5: Prepare Dataset
REM =============================================================================

echo Step 5: Preparing dataset...

if exist data\train.bin if exist data\val.bin (
    echo Dataset already prepared
) else (
    echo Preparing Shakespeare dataset...
    cd data
    python prepare.py
    cd ..
    echo [OK] Dataset prepared
)

echo.

REM =============================================================================
REM All Done!
REM =============================================================================

echo ======================================================================
echo                     [OK] SETUP COMPLETE!
echo ======================================================================
echo.
echo.
echo Quick Start Commands:
echo.
echo   python gpt.py info          # Check your setup
echo   python gpt.py train         # Start training (interactive)
echo   python gpt.py generate      # Generate text
echo   python gpt.py hardware      # Check hardware options
echo.
echo Want to train right now?
set /p START_TRAIN="Start interactive training? (y/n) [y]: "

if "%START_TRAIN%"=="" set START_TRAIN=y
if /i "%START_TRAIN%"=="y" (
    python gpt.py train
) else (
    echo.
    echo No problem! When you're ready, run:
    echo   python gpt.py train
    echo.
)

echo.
echo Documentation:
echo   - QUICK_REFERENCE.md    - Command cheat sheet
echo   - GETTING_STARTED.md    - Beginner's guide
echo   - README.md             - Full documentation
echo.
echo Happy training!
pause

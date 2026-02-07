@echo off
chcp 65001 >nul
title ILIN Setup
echo ==========================================
echo   ILIN - Integrated Localized Intelligence Node
echo   Setup Script
echo ==========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://python.org
    pause
    exit /b 1
)

echo [✓] Python found
python --version
echo.

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo [1/4] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [✓] Virtual environment created
) else (
    echo [1/4] Virtual environment already exists
)
echo.

REM Activate virtual environment
echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)
echo [✓] Virtual environment activated
echo.

REM Upgrade pip
echo [3/4] Upgrading pip...
python -m pip install --upgrade pip
echo [✓] Pip upgraded
echo.

REM Install requirements
echo [4/4] Installing dependencies...
echo This may take a few minutes...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)
echo [✓] Dependencies installed
echo.

REM Download embedding model
echo [5/4] Downloading embedding model (first time only)...
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
echo [✓] Embedding model ready
echo.

echo ==========================================
echo   Setup Complete!
echo ==========================================
echo.
echo To run ILIN:
echo   - Run API:   run_api.bat
echo   - Run UI:    run_ui.bat
echo   - Run Both:  run_both.bat
echo.
echo Make sure Ollama is running: ollama serve
echo.
pause

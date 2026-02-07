@echo off
chcp 65001 >nul
title ILIN API Server
echo ==========================================
echo   ILIN API Server
echo   http://localhost:8000
echo ==========================================
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo [ERROR] Virtual environment not found!
    echo Please run setup.bat first
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if Ollama is running
echo Checking Ollama connection...
python -c "import requests; requests.get('http://localhost:11434/api/tags', timeout=5)" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Ollama doesn't seem to be running!
    echo Please start Ollama first: ollama serve
echo.
    choice /C YN /M "Do you want to continue anyway"
    if errorlevel 2 exit /b 1
) else (
    echo [✓] Ollama is running
echo.
)

echo Starting API Server...
echo Press Ctrl+C to stop
echo.

python run_api.py

if errorlevel 1 (
    echo.
    echo [ERROR] API Server crashed or stopped
    pause
)

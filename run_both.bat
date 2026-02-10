@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
title ILIN - Intelligence Node Launcher
cls

echo ==========================================
echo   ILIN - Integrated Localized Intelligence
echo ==========================================
echo(

REM 1. Check for Virtual Environment
echo [1/3] Checking virtual environment...
if not exist "venv" (
    echo [MISSING] Virtual environment not found.
    echo Creating virtual environment now...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create venv. Is Python installed and in PATH?
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
) else (
    echo [OK] Virtual environment found.
)

REM 2. Check and Install Requirements
echo [2/3] Checking requirements (this may take a moment)...
call venv\Scripts\activate.bat
python -m pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [ERROR] Failed to verify or install requirements.
    pause
    exit /b 1
)
echo [OK] Requirements are satisfied.

REM 3. Check for Ollama
echo [3/3] Checking Ollama connection...
python -c "import requests; requests.get('http://localhost:11434/api/tags', timeout=2)" >nul 2>&1
if not errorlevel 1 (
    echo [OK] Ollama is running.
    goto :LaunchServices
)

echo [WARNING] Ollama is not detected at http://localhost:11434
echo Please ensure Ollama is running (ollama serve).
echo(
choice /C YN /M "Do you want to continue anyway"
if errorlevel 2 exit /b 1

:LaunchServices


echo(
echo ==========================================
echo   Launching Services...
echo ==========================================
echo(
echo API Server: http://localhost:8000
echo Web UI:     http://localhost:8501
echo(
echo - Close the windows to stop the services.
echo(

REM Start API in a new window
start "ILIN API Server" cmd /k "call venv\Scripts\activate.bat && python run_api.py"

REM Small delay to let API initialize
timeout /t 3 /nobreak >nul

REM Start UI in a new window
start "ILIN Web UI" cmd /k "call venv\Scripts\activate.bat && set PYTHONPATH=%CD% && python run_ui.py"

echo(
echo All services launched successfully!
echo(
pause

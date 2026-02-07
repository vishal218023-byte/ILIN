@echo off
chcp 65001 >nul
title ILIN - Running API and UI
echo ==========================================
echo   ILIN - Starting API and UI
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
    echo Please start Ollama first in another terminal: ollama serve
echo.
    choice /C YN /M "Do you want to continue anyway"
    if errorlevel 2 exit /b 1
) else (
    echo [✓] Ollama is running
)
echo.

echo Starting API Server and Web UI...
echo.
echo API will be available at: http://localhost:8000
echo UI will be available at:  http://localhost:8501
echo.
echo Press Ctrl+C twice to stop both services
echo.

REM Start API in background
start "ILIN API Server" cmd /k "call venv\Scripts\activate.bat && python run_api.py"

REM Wait a bit for API to start
timeout /t 3 /nobreak >nul

REM Start UI
start "ILIN Web UI" cmd /k "call venv\\Scripts\\activate.bat && set PYTHONPATH=%CD% && streamlit run app\\ui\\streamlit_app.py --server.port 8501 --server.address localhost"

echo.
echo Both services started!
echo.
echo - API Server window:   ILIN API Server
echo - Web UI window:       ILIN Web UI
echo.
pause

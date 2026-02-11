@echo off
chcp 65001 >nul
title ILIN Web UI
echo ==========================================
echo   ILIN Web Interface
echo   http://localhost:8501
echo   Network: http://0.0.0.0:8501
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

REM Check if API is running
echo Checking API server...
python -c "import requests; requests.get('http://localhost:8000/health', timeout=3)" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] API server doesn't seem to be running!
    echo For full functionality, start the API first: run_api.bat
echo.
    choice /C YN /M "Do you want to continue in direct mode"
    if errorlevel 2 exit /b 1
    echo.
    echo Starting UI in DIRECT mode ^(bypassing API^)...
    set PYTHONPATH=%CD%
    streamlit run app\\ui\\streamlit_app.py --server.port 8501 --server.address 0.0.0.0
) else (
    echo [✓] API server is running
echo.
    echo Starting Web UI...
    echo Local: http://localhost:8501
    echo Network: http://0.0.0.0:8501
    echo Press Ctrl+C to stop
echo.

    set PYTHONPATH=%CD%
    streamlit run app\\ui\\streamlit_app.py --server.port 8501 --server.address 0.0.0.0
)

if errorlevel 1 (
    echo.
    echo [ERROR] UI stopped or crashed
    pause
)

@echo off
chcp 65001 >nul
title ILIN - Integrated Localized Intelligence Node
color 0B

:menu
cls
echo.
echo  ╔══════════════════════════════════════════════════════════════╗
echo  ║                                                              ║
echo  ║     🤖 ILIN - Integrated Localized Intelligence Node         ║
echo  ║                                                              ║
echo  ║          AI-Powered Offline RAG System                       ║
echo  ║                                                              ║
echo  ╚══════════════════════════════════════════════════════════════╝
echo.
echo  [1] 🚀 Setup Environment (First Time Only)
echo  [2] ▶️  Run API Server Only
echo  [3] 🌐 Run Web UI Only
echo  [4] ⚡ Run Both API and UI
echo  [5] 📋 Check System Status
echo  [6] 🗑️  Clean Up (Remove venv)
echo  [7] ❌ Exit
echo.
echo ════════════════════════════════════════════════════════════════
echo.

set /p choice="Enter your choice (1-7): "

if "%choice%"=="1" goto setup
if "%choice%"=="2" goto api
if "%choice%"=="3" goto ui
if "%choice%"=="4" goto both
if "%choice%"=="5" goto status
if "%choice%"=="6" goto cleanup
if "%choice%"=="7" exit

echo [ERROR] Invalid choice! Please try again.
timeout /t 2 >nul
goto menu

:setup
call setup.bat
goto menu

:api
call run_api.bat
goto menu

:ui
call run_ui.bat
goto menu

:both
call run_both.bat
goto menu

:status
cls
echo.
echo  ╔══════════════════════════════════════════════════════════════╗
echo  ║                    System Status Check                       ║
echo  ╚══════════════════════════════════════════════════════════════╝
echo.

echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo   ❌ Python: NOT FOUND
) else (
    echo   ✅ Python: 
    python --version
)
echo.

echo Checking Virtual Environment...
if exist "venv" (
    echo   ✅ Virtual Environment: EXISTS
) else (
    echo   ❌ Virtual Environment: NOT FOUND
    echo      Run option 1 (Setup) first
)
echo.

echo Checking Ollama...
python -c "import requests; requests.get('http://localhost:11434/api/tags', timeout=3)" >nul 2>&1
if errorlevel 1 (
    echo   ❌ Ollama: NOT RUNNING
    echo      Start with: ollama serve
) else (
    echo   ✅ Ollama: RUNNING
)
echo.

echo Checking API Server...
python -c "import requests; requests.get('http://localhost:8000/health', timeout=2)" >nul 2>&1
if errorlevel 1 (
    echo   ❌ API Server: NOT RUNNING
) else (
    echo   ✅ API Server: RUNNING (http://localhost:8000)
)
echo.

echo Checking Documents...
if exist "data\documents" (
    for /f %%A in ('dir /b /a-d "data\documents" 2^>nul ^| find /c /v ""') do echo   📁 Documents: %%A files
) else (
    echo   📁 Documents: 0 files
)
echo.

if exist "data\vector_indices\faiss_index.bin" (
    echo   ✅ Vector Index: EXISTS
) else (
    echo   ⚠️  Vector Index: NOT CREATED YET
)
echo.

echo ════════════════════════════════════════════════════════════════
echo.
pause
goto menu

:cleanup
cls
echo.
echo  ╔══════════════════════════════════════════════════════════════╗
echo  ║                      Clean Up Environment                    ║
echo  ╚══════════════════════════════════════════════════════════════╝
echo.
echo This will remove the virtual environment and reset the project.
echo You will need to run Setup again to use ILIN.
echo.
set /p confirm="Are you sure? (yes/no): "
if /i "%confirm%"=="yes" (
    echo.
    echo Removing virtual environment...
    if exist "venv" rmdir /s /q venv
    echo ✅ Virtual environment removed!
    echo.
    echo You can now run Setup (Option 1) to reinstall.
) else (
    echo Cancelled.
)
timeout /t 3 >nul
goto menu

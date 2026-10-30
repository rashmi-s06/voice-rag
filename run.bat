@echo off
cd /d "%~dp0"
echo ===================================================
echo Starting Voice-Enabled RAG System...
echo ===================================================

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .venv\Scripts\python.exe!
    pause
    exit /b 1
)

echo Launching app at http://127.0.0.1:7860...
".venv\Scripts\python.exe" app.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Application exited with error code %ERRORLEVEL%.
    pause
)

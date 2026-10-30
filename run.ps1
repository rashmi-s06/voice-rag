$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "Starting Voice-Enabled RAG System..." -ForegroundColor Green
Write-Host "===================================================" -ForegroundColor Cyan

$pythonExe = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    Write-Host "[ERROR] Virtual environment not found at $pythonExe!" -ForegroundColor Red
    pause
    exit 1
}

Write-Host "Launching app at http://127.0.0.1:7860..." -ForegroundColor Yellow
& $pythonExe app.py

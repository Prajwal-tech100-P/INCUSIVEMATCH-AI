@echo off
title InclusiveMatch AI Server
cd /d "%~dp0"

if not exist "%~dp0venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at %~dp0venv
    pause
    exit /b 1
)

echo ========================================================
echo   Launching InclusiveMatch AI (Virtual Environment)
echo ========================================================
"%~dp0venv\Scripts\python.exe" "%~dp0app.py"
if %errorlevel% neq 0 (
    echo.
    echo [Server exited with error code %errorlevel%]
    pause
)


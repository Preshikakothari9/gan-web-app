@echo off
title GAN Studio — Setup & Launch
color 0A

echo.
echo  ╔══════════════════════════════════════════╗
echo  ║         GAN Studio — Auto Setup          ║
echo  ║   Image Generator + Text Generator       ║
echo  ╚══════════════════════════════════════════╝
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo  [ERROR] Python not found. Install from https://python.org
    echo          Make sure to check "Add Python to PATH"
    pause & exit /b 1
)

echo  [1/4] Python found ✓
echo.

:: Create virtual environment
if not exist "venv\" (
    echo  [2/4] Creating virtual environment...
    python -m venv venv
) else (
    echo  [2/4] Virtual environment already exists ✓
)

:: Activate and install
echo.
echo  [3/4] Installing dependencies (first time may take 5-10 min)...
call venv\Scripts\activate.bat
pip install -r backend\requirements.txt --quiet

echo.
echo  [4/4] Launching GAN Studio (backend + frontend)...
echo.
echo  ════════════════════════════════════════════
echo   Open  →  http://localhost:5000
echo  ════════════════════════════════════════════
echo.

cd backend
python app.py
pause
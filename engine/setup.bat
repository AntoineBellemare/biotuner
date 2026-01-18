@echo off
REM Biotuner v2 Setup Script for Windows
REM Run this script to set up both backend and frontend

echo.
echo ========================================
echo   Biotuner v2 - FastAPI + React Setup
echo ========================================
echo.

REM Backend Setup
echo [Backend] Setting up...
cd backend

REM Check if virtual environment exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing Python dependencies...
pip install -r requirements.txt

echo [Backend] Setup complete!
echo.

REM Frontend Setup
echo [Frontend] Setting up...
cd ..\frontend

REM Install dependencies
echo Installing Node dependencies...
call npm install

echo [Frontend] Setup complete!
echo.

REM Success message
echo ========================================
echo   Setup Complete!
echo ========================================
echo.
echo To start the application:
echo.
echo 1. Start Backend (in one terminal):
echo    cd backend
echo    venv\Scripts\activate
echo    python main.py
echo.
echo 2. Start Frontend (in another terminal):
echo    cd frontend
echo    npm run dev
echo.
echo 3. Open browser:
echo    http://localhost:5173
echo.

pause

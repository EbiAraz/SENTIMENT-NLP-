@echo off
REM Setup script for Windows users

echo.
echo ========================================
echo Sentiment Analysis Project - Setup
echo ========================================
echo.

echo Checking Python version...
python --version > nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found or not in PATH
    echo Please install Python 3.10+ from python.org
    pause
    exit /b 1
)

echo.
echo Installing dependencies...
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo ========================================
echo ✓ Setup complete!
echo ========================================
echo.
echo To launch the app, run:
echo   python -m streamlit run app.py
echo.
echo The app will open at http://localhost:8501/
echo.
pause

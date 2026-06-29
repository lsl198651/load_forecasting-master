@echo off
chcp 65001 >nul 2>&1
echo ========================================
echo Power Load Forecasting System - Startup
echo ========================================
echo.

echo [1/4] Checking Python environment...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python environment not detected. Please install Python 3.8+ first.
    pause
    exit /b 1
)

echo.
echo [2/4] Installing dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo WARNING: Some dependencies failed to install. Please install manually.
)

echo.
echo [3/4] Configuring Baidu Qianfan API...
echo Please set QIANFAN_API_KEY in system environment variables,
echo or configure it directly in generate_report.py file.

echo.
echo [4/4] Starting Django server...
cd /d "%~dp0"
cd server
python manage.py runserver 127.0.0.1:8000

echo.
echo ========================================
echo System started successfully!
echo Please visit: http://127.0.0.1:8000/dashboard/
echo ========================================
pause
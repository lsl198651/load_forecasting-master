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
echo [3/4] Configuring AI models...
echo Please configure API keys in .env file for:
echo   - QIANFAN_API_KEY (Baidu Qianfan)
echo   - DASHSCOPE_API_KEY (Alibaba DashScope)
echo   - DOUBAN_API_KEY (ByteDance Doubao)
echo   - XUNFEI_API_KEY (iFlytek Spark)

echo.
echo [4/4] Starting server and opening browser...
cd /d "%~dp0"
python start_server.py

pause
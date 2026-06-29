@echo off
echo ========================================
echo 电力负荷预测与智能分析系统启动脚本
echo ========================================
echo.

echo [1/4] 检查Python环境...
python --version
if %errorlevel% neq 0 (
    echo 错误：未检测到Python环境，请先安装Python 3.8+
    pause
    exit /b 1
)

echo.
echo [2/4] 安装依赖包...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo 警告：部分依赖安装失败，请手动安装
)

echo.
echo [3/4] 设置百度千帆API密钥...
echo 请在系统环境变量中设置 QIANFAN_API_KEY
echo 或在 generate_report.py 文件中直接配置

echo.
echo [4/4] 启动Django服务器...
cd server
python manage.py runserver 127.0.0.1:8000

echo.
echo ========================================
echo 系统启动完成！
echo 请访问: http://127.0.0.1:8000/dashboard/
echo ========================================
pause
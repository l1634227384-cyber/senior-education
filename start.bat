@echo off
chcp 65001 >nul
echo ========================================
echo 高等教育个性化学习资源智能体系统
echo ========================================
echo.

:: 检查Python
py --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未找到Python，请先安装Python 3.9+
    pause
    exit /b 1
)

:: 安装依赖
echo [1/3] 安装依赖包...
py -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [警告] 部分依赖安装可能失败，请手动检查
)

:: 创建目录
echo [2/3] 初始化目录结构...
if not exist "data" mkdir data
if not exist "uploads" mkdir uploads
if not exist "resources" mkdir resources
if not exist "static" mkdir static

:: 启动服务
echo [3/3] 启动智能学习系统...
echo.
echo 访问地址: http://localhost:8000
echo 按 Ctrl+C 停止服务
echo.
py main.py

pause

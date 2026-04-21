@echo off
cd /d "%~dp0"
python airone_gui.py
if errorlevel 1 (
    echo.
    echo [ERROR] AirOne GUI failed to start. Make sure AirOne is installed:
    echo   pip install -e .
    pause
)

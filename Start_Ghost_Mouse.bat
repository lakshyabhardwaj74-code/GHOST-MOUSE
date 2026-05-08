@echo off
title Ghost Mouse Launcher
echo ========================================
echo        Starting Ghost Mouse Widget      
echo ========================================
echo.

cd /d "%~dp0"

if exist ghost_env\Scripts\activate.bat (
    echo [1/3] Activating virtual environment...
    call ghost_env\Scripts\activate.bat
) else (
    echo Error: virtual environment 'ghost_env' not found.
    pause
    exit /b
)

echo [2/3] Installing and verifying requirements...
pip install -r requirements.txt -q

echo [3/3] Launching Widget...
echo.
echo ----------------------------------------------------
echo SUCCESS! Ghost Mouse is now running in the background.
echo ----------------------------------------------------
echo Look for the Ghost Mouse icon in your system tray (near the clock).
echo Right-click the icon to start/stop tracking.
echo.
echo You can close this window now.
start /b pythonw ghost_mouse.py

@echo off
setlocal

REM Resolve project root from this script location.
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "PROJECT_ROOT=%%~fI"
cd /d "%PROJECT_ROOT%"

where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python not found in PATH. Please install Python 3.10+ and retry.
  exit /b 1
)

echo [INFO] Starting crypto-analysis server (auto-restart on crash)...
echo [INFO] Close this window to stop the server.
echo.

:RESTART
echo [%date% %time%] Starting server...
python run.py
echo.
echo [%date% %time%] Server stopped. Restarting in 3 seconds... (Close window to quit)
timeout /t 3 /nobreak >nul
goto RESTART

endlocal

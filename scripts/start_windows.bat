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

echo [INFO] Starting crypto-analysis server...
python run.py

endlocal

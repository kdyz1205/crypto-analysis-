@echo off
REM Template: Trading OS desktop launcher
REM
REM Copy this to "C:\Users\<you>\Desktop\Trading OS.bat" (rename the file)
REM then double-click it to start the server. Shortcut can pin to taskbar.
REM
REM 2026-04-24: hardcoded Python path. Reason: `python` on PATH can
REM resolve to wrong env (hermes-agent venv / Microsoft Store stub),
REM and the server fails to boot with "module not found" errors. Pin
REM the full path to the interpreter that actually has the deps.
REM Override PY env var if you move deps to a venv later.

title Trading OS
cd /d "C:\Users\alexl\Desktop\crypto-analysis-"

set PORT=8000
set LAUNCH_PATH=/v2
set "PY=C:\Users\alexl\AppData\Local\Programs\Python\Python312\python.exe"

if not exist "%PY%" (
    echo ============================================
    echo  ERROR: Python not found at
    echo   %PY%
    echo  Edit this .bat or set PY env var to override.
    echo ============================================
    pause
    exit /b 1
)

echo ============================================
echo   Trading OS - starting server on port 8000
echo   Python: %PY%
echo   After it says "Uvicorn running", browser
echo   will open automatically to /v2.
echo ============================================
echo.
echo   Close this window to stop the server.
echo.

"%PY%" run.py

echo.
echo Server stopped. Press any key to close.
pause >nul

@echo off
REM MA Ribbon EMA21 Backtest Panel launcher.
REM Starts the FastAPI server on http://127.0.0.1:8765 and opens the browser.

setlocal

REM Project root = directory containing this scripts/ folder.
set PROJECT_ROOT=%~dp0..
cd /d "%PROJECT_ROOT%"

REM Use the system Python that has fastapi/uvicorn/pandas (NOT the empty hermes-agent venv).
set PY="C:\Users\alexl\AppData\Local\Programs\Python\Python312\python.exe"

REM Open the browser AFTER 2s so the server has time to bind.
start "" /b cmd /c "timeout /t 2 /nobreak > nul && start http://127.0.0.1:8765/"

REM Run the panel server in this window. Ctrl+C to stop.
%PY% -m backtests.ma_ribbon_ema21.web_app --port 8765

endlocal

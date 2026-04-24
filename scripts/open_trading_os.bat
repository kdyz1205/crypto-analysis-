@echo off
rem Thin wrapper so Windows can double-click the PowerShell launcher.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0open_trading_os.ps1"

@echo off
rem Thin wrapper so Windows can double-click the PowerShell launcher even
rem when PowerShell script execution policy is restricted (-Bypass only
rem affects this single invocation; system policy stays unchanged).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0open_trading_os.ps1"

# Trading OS launcher
#
# Double-clicking the desktop shortcut runs this via a thin .bat wrapper.
# It: (1) checks if the uvicorn server on port 8000 is responding;
#     (2) starts it in a hidden background process if not;
#     (3) waits up to 30s for it to be ready;
#     (4) opens http://127.0.0.1:8000/v2 in the default browser.
#
# Safe to click repeatedly — if the server is already running we skip
# straight to opening the browser.

param()

$Url = 'http://127.0.0.1:8000/v2'
$Root = 'C:\Users\alexl\Desktop\crypto-analysis-'
$ErrorActionPreference = 'SilentlyContinue'

function Test-Server {
    try {
        $response = Invoke-WebRequest -Uri $Url -TimeoutSec 2 -UseBasicParsing -ErrorAction Stop
        return $response.StatusCode -eq 200
    } catch {
        return $false
    }
}

Write-Host ""
Write-Host "  Trading OS Launcher" -ForegroundColor Cyan
Write-Host "  -------------------" -ForegroundColor Cyan
Write-Host ""

if (Test-Server) {
    Write-Host "  Server already running on port 8000." -ForegroundColor Green
} else {
    Write-Host "  Server not running. Starting in background..." -ForegroundColor Yellow
    $logOut = Join-Path $Root 'data\uvicorn_codex_out.log'
    $logErr = Join-Path $Root 'data\uvicorn_codex_err.log'

    # Start uvicorn inside a hidden cmd window so redirection works. Kept
    # alive for the duration of the user's session; closes when they log out.
    $startCmd = "cd /d ""$Root"" && python -m uvicorn server.app:app --host 127.0.0.1 --port 8000 --reload >> ""$logOut"" 2>> ""$logErr"""
    Start-Process -WindowStyle Hidden -FilePath 'cmd.exe' -ArgumentList '/c', $startCmd

    Write-Host "  Waiting for server to come up (up to 30s)..." -ForegroundColor Yellow
    $ready = $false
    for ($i = 1; $i -le 30; $i++) {
        Start-Sleep -Seconds 1
        if (Test-Server) {
            Write-Host "  Server ready after ${i}s." -ForegroundColor Green
            $ready = $true
            break
        }
    }
    if (-not $ready) {
        Write-Host ""
        Write-Host "  Server did not start within 30s." -ForegroundColor Red
        Write-Host "  Check the error log:" -ForegroundColor Red
        Write-Host "    $logErr" -ForegroundColor Red
        Write-Host ""
        Read-Host "  Press Enter to close"
        exit 1
    }
}

Write-Host ""
Write-Host "  Opening $Url in your browser..." -ForegroundColor Cyan
Start-Process $Url
Start-Sleep -Seconds 2

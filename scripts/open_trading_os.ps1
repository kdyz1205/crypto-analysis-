# Trading OS launcher for Windows desktop shortcuts.
#
# Starts the FastAPI server on port 8000 when needed, then opens /v2.
# Uses a known-good Python 3.12 interpreter instead of whatever `python`
# happens to resolve to in PATH.

param()

$ErrorActionPreference = 'SilentlyContinue'
$Root = Split-Path -Parent $PSScriptRoot
$Url = 'http://127.0.0.1:8000/v2'
$LogOut = Join-Path $Root 'data\uvicorn_launcher_out.log'
$LogErr = Join-Path $Root 'data\uvicorn_launcher_err.log'

function Test-Server {
    try {
        $response = Invoke-WebRequest -Uri $Url -TimeoutSec 2 -UseBasicParsing -ErrorAction Stop
        return $response.StatusCode -eq 200
    } catch {
        return $false
    }
}

function Resolve-Python {
    $candidates = @(
        $env:TRADING_OS_PYTHON,
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
        "$env:USERPROFILE\AppData\Local\Programs\Python\Python312\python.exe"
    )

    foreach ($candidate in $candidates) {
        if (-not $candidate) { continue }
        if (-not (Test-Path -LiteralPath $candidate)) { continue }
        & $candidate -c "import uvicorn, fastapi" *> $null
        if ($LASTEXITCODE -eq 0) {
            return $candidate
        }
    }
    return $null
}

Write-Host ''
Write-Host '  Trading OS Launcher' -ForegroundColor Cyan
Write-Host '  -------------------' -ForegroundColor Cyan
Write-Host ''

if (Test-Server) {
    Write-Host '  Server already running on port 8000.' -ForegroundColor Green
} else {
    $Python = Resolve-Python
    if (-not $Python) {
        Write-Host '  Could not find a Python with uvicorn + fastapi installed.' -ForegroundColor Red
        Write-Host '  Install requirements or set TRADING_OS_PYTHON to the correct python.exe.' -ForegroundColor Red
        Read-Host '  Press Enter to close'
        exit 1
    }

    Write-Host "  Starting server with $Python" -ForegroundColor Yellow
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogOut) | Out-Null

    $env:PORT = '8000'
    $env:LAUNCH_PATH = '/v2'
    $env:PYTHONIOENCODING = 'utf-8'
    if (-not $env:MAR_BB_AUTOSTART) {
        $env:MAR_BB_AUTOSTART = '0'
    }

    $command = "cd /d ""$Root"" && set PORT=8000 && set LAUNCH_PATH=/v2 && set PYTHONIOENCODING=utf-8 && set MAR_BB_AUTOSTART=$env:MAR_BB_AUTOSTART && ""$Python"" run.py >> ""$LogOut"" 2>> ""$LogErr"""
    Start-Process -WindowStyle Hidden -FilePath 'cmd.exe' -ArgumentList '/c', $command

    Write-Host '  Waiting for server to come up (up to 30s)...' -ForegroundColor Yellow
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
        Write-Host ''
        Write-Host '  Server did not start within 30s.' -ForegroundColor Red
        Write-Host "  Error log: $LogErr" -ForegroundColor Red
        Read-Host '  Press Enter to close'
        exit 1
    }
}

Write-Host ''
Write-Host "  Opening $Url in your browser..." -ForegroundColor Cyan
Start-Process $Url
Start-Sleep -Seconds 2

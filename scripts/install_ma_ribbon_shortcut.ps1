# Install a desktop shortcut for the MA Ribbon EMA21 Backtest Panel.
# Run once: PowerShell -ExecutionPolicy Bypass -File scripts\install_ma_ribbon_shortcut.ps1

$projectRoot = (Resolve-Path "$PSScriptRoot\..").Path
$bat         = Join-Path $projectRoot "scripts\launch_ma_ribbon_panel.bat"
$shortcut    = Join-Path ([Environment]::GetFolderPath("Desktop")) "MA Ribbon Panel.lnk"

if (-not (Test-Path $bat)) {
    Write-Error "Launcher not found: $bat"
    exit 1
}

$wsh = New-Object -ComObject WScript.Shell
$lnk = $wsh.CreateShortcut($shortcut)
$lnk.TargetPath       = $bat
$lnk.WorkingDirectory = $projectRoot
$lnk.Description      = "MA Ribbon EMA21 Backtest Panel (localhost:8765)"
# imageres.dll #93 = chart/graph icon
$lnk.IconLocation     = "$env:SystemRoot\System32\imageres.dll,93"
$lnk.Save()

Write-Host "Created shortcut: $shortcut"
Write-Host "  Target: $bat"
Write-Host "  Double-click it to open the MA Ribbon backtest panel."

# One-time installer: places a "Trading OS" shortcut on the user's
# desktop pointing at open_trading_os.bat. Run once; re-run to update.

$Root = 'C:\Users\alexl\Desktop\crypto-analysis-'
$TargetBat = Join-Path $Root 'scripts\open_trading_os.bat'

if (-not (Test-Path $TargetBat)) {
    Write-Host "ERROR: target script not found: $TargetBat" -ForegroundColor Red
    exit 1
}

$Desktop = [Environment]::GetFolderPath('Desktop')
$ShortcutPath = Join-Path $Desktop 'Trading OS.lnk'

try {
    $WshShell = New-Object -ComObject WScript.Shell
    $Shortcut = $WshShell.CreateShortcut($ShortcutPath)
    $Shortcut.TargetPath = $TargetBat
    $Shortcut.WorkingDirectory = $Root
    # Icon: shell32.dll index 13 is a generic browser-globe icon. Plenty
    # of alternatives: 14 (desktop), 137 (browser), 238 (chart). 13 is
    # the closest to "open web app" and present on every Windows.
    $Shortcut.IconLocation = "$env:SystemRoot\System32\SHELL32.dll,13"
    $Shortcut.Description = 'Launch Trading OS (auto-starts backend if needed)'
    $Shortcut.Save()
    Write-Host ""
    Write-Host "  Shortcut created on your Desktop:" -ForegroundColor Green
    Write-Host "    $ShortcutPath" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Double-click it to launch Trading OS." -ForegroundColor Green
    Write-Host "  First click after reboot: ~5s to start server + open browser." -ForegroundColor Green
    Write-Host "  Subsequent clicks: instant (server already running)." -ForegroundColor Green
    Write-Host ""
} catch {
    Write-Host "ERROR: could not create shortcut: $_" -ForegroundColor Red
    exit 1
}

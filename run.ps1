# InclusiveMatch AI PowerShell Launcher
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

$venvPython = Join-Path $scriptDir "venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "[ERROR] Virtual environment not found at: $venvPython" -ForegroundColor Red
    exit 1
}

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  Launching InclusiveMatch AI (Virtual Environment)" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

& $venvPython (Join-Path $scriptDir "app.py")

